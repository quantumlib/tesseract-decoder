#include "bp/tesseract_post_processor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <vector>

#include "bp/check_update.h"

namespace bp {

TesseractPostProcessor::TesseractPostProcessor(
    const stim::DetectorErrorModel& merged_dem,
    const TannerGraph<LLR_INT>& graph,
    const std::vector<std::vector<int>>& /*hyperedge_observables*/,
    const BPParams& params)
    : graph_(graph),
      params_(params),
      tesseract_decoder_([&]() {
        tesseract_decoder::TesseractConfig config;
        config.dem = merged_dem;
        config.merge_errors = false;
        config.det_beam = params.tesseract_det_beam;
        config.beam_climbing = false;
        config.no_revisit_dets = true;
        config.pqlimit = params.tesseract_pqlimit;
        if (params.osd_weight == 0) {
          config.detector_orders = {tesseract_decoder::DetectorOrder()};
        } else {
          config.detector_orders = tesseract_decoder::make_detector_orders(
              4, tesseract_decoder::DetectorOrder::Method::BFS,
              params.random_seed ? params.random_seed : 1);
        }
        return config;
      }()),
      osd_fallback_(std::make_unique<OsdPostProcessor>(
          graph,
          static_cast<size_t>((params.osd_weight == 0)
                                  ? std::max(0, params.osd_order)
                                  : std::max(30, params.osd_order)),
          static_cast<size_t>((params.osd_weight == 0)
                                  ? std::max(0, params.osd_weight)
                                  : std::max(2, params.osd_weight)),
          params.osd_truncation_factor, params.osd_early_exit)) {}

std::vector<uint8_t> TesseractPostProcessor::process(
    const BPResult& bp_result, const std::vector<LLR_INT>& posteriors,
    const std::vector<uint64_t>& detection_events,
    const std::vector<std::vector<int>>& hyperedge_observables,
    std::vector<size_t>* predicted_errors,
    const std::vector<LLR_INT>* custom_priors) {
  profile_.num_calls++;
  if (predicted_errors) {
    predicted_errors->clear();
  }

  const size_t num_variables = graph_.variable_nodes.size();
  const size_t num_checks = graph_.check_nodes.size();

  size_t num_obs = 0;
  for (const auto& obs_list : hyperedge_observables) {
    for (int o : obs_list) {
      if (o >= 0) num_obs = std::max(num_obs, static_cast<size_t>(o + 1));
    }
  }

  auto var_cost = [&](size_t v) -> int64_t {
    LLR_INT p = custom_priors ? (*custom_priors)[v] : graph_.variable_nodes[v].prior;
    return std::max<int64_t>(1, p);
  };

  auto compute_solution_cost = [&](const std::vector<bool>& err) -> int64_t {
    int64_t c = 0;
    for (size_t v = 0; v < num_variables; ++v) {
      if (err[v]) c += var_cost(v);
    }
    return c;
  };

  auto extract_observables = [&](const std::vector<bool>& err) -> std::vector<uint8_t> {
    std::vector<uint8_t> obs_status(num_obs, 0);
    if (predicted_errors) {
      predicted_errors->clear();
    }
    for (size_t v = 0; v < num_variables; ++v) {
      if (err[v]) {
        if (predicted_errors) {
          predicted_errors->push_back(v);
        }
        if (v < hyperedge_observables.size()) {
          for (int o : hyperedge_observables[v]) {
            if (o >= 0 && static_cast<size_t>(o) < num_obs) {
              obs_status[o] ^= 1;
            }
          }
        }
      }
    }
    return obs_status;
  };

  std::vector<bool> best_error(num_variables, false);
  int64_t best_cost = std::numeric_limits<int64_t>::max();

  if (bp_result.converged) {
    auto t0 = std::chrono::steady_clock::now();
    for (size_t v = 0; v < num_variables; ++v) {
      if (posteriors[v] < 0) {
        best_error[v] = true;
      }
    }
    profile_.num_converged_early_exits++;
    auto obs_status = extract_observables(best_error);
    auto t1 = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();
    profile_.t_converged_passthrough += dt;
    profile_.t_hard_decision += dt;
    return obs_status;
  }

  // Case B: BP did not converge (or had suspect low-confidence / high-degree variables) ->
  // Multi-Threshold Residual-Syndrome Peeling + Defect-Cluster Un-Peeling + Tesseract A*
  profile_.num_osd_invocations++;
  auto t_start_total = std::chrono::steady_clock::now();

  // Prepare Pass-1 Reweighted Priors (custom_priors) once per shot
  auto t_step2_start = std::chrono::steady_clock::now();
  std::vector<tesseract_decoder::ErrorProbabilityUpdate> updates;
  if (custom_priors != nullptr) {
    for (size_t v = 0; v < num_variables; ++v) {
      LLR_INT cp = (*custom_priors)[v];
      if (cp != graph_.variable_nodes[v].prior &&
          v < tesseract_decoder_.dem_error_to_error.size() &&
          tesseract_decoder_.dem_error_to_error[v] != std::numeric_limits<size_t>::max()) {
        double llr_val = llr_int_to_double(cp);
        double p_hat = 1.0 / (1.0 + std::exp(llr_val));
        p_hat = std::clamp(p_hat, 1e-7, 0.499);
        updates.push_back(tesseract_decoder::ErrorProbabilityUpdate{v, p_hat});
      }
    }
  }
  auto t_step2_end = std::chrono::steady_clock::now();
  profile_.t_sort += std::chrono::duration<double>(t_step2_end - t_step2_start).count();
  profile_.sum_num_errors += updates.size();

  const bool is_fast_mode = (params_.osd_weight == 0);
  struct PeelTrial {
    int32_t thresh;
    size_t max_deg;
    size_t max_res_dets;
    int32_t unpeel_limit = 100000000;
  };
  std::vector<PeelTrial> trials;
  if (is_fast_mode) {
    // Fast Search mode: single hard-decision peel + 2-hop clustered A*, with component un-peel
    // backup only if Trial 1 returns low_confidence
    trials.push_back({0, 100, 36, 100000000});
    trials.push_back({0, 3, 28, 2500});
  } else {
    // Full Accurate Search mode: multi-threshold ladder + component-local un-peeling + OSD-2
    constexpr int32_t kPeelThreshold = 2500;
    constexpr int32_t kMidThreshold = 1250;
    trials.push_back({0, 3, 32, 100000000});
    trials.push_back({0, 3, 28, kPeelThreshold});
    trials.push_back({0, 3, 28, kMidThreshold});
    trials.push_back({kMidThreshold, 3, 20, 100000000});
    trials.push_back({kPeelThreshold, 3, 18, 100000000});
  }

  std::vector<uint64_t> prev_valid_residual;
  bool first_trial = true;

  for (const auto& trial : trials) {
    auto t_step1_start = std::chrono::steady_clock::now();
    std::vector<bool> is_peeled(num_variables, false);
    for (size_t v = 0; v < num_variables; ++v) {
      if (posteriors[v] < -trial.thresh) {
        size_t deg = graph_.var_edge_offsets[v + 1] - graph_.var_edge_offsets[v];
        if (trial.thresh == 0 || deg <= trial.max_deg) {
          is_peeled[v] = true;
        }
      }
    }

    std::vector<uint8_t> check_active(num_checks, 0);
    for (uint64_t d : detection_events) {
      if (d < num_checks) {
        check_active[d] ^= 1;
      }
    }
    for (size_t v = 0; v < num_variables; ++v) {
      if (is_peeled[v]) {
        size_t c_start = graph_.var_edge_offsets[v];
        size_t c_end = graph_.var_edge_offsets[v + 1];
        for (size_t e = c_start; e < c_end; ++e) {
          size_t c = graph_.var_edges[e];
          if (c < num_checks) {
            check_active[c] ^= 1;
          }
        }
      }
    }

    // Connected-component un-peeling of unsatisfied checks
    std::vector<uint8_t> check_active_backup;
    std::vector<bool> is_peeled_backup;
    if (trial.unpeel_limit < 100000000 || trial.max_deg < 100) {
      check_active_backup = check_active;
      is_peeled_backup = is_peeled;
      std::vector<size_t> unsat_queue;
      std::vector<uint8_t> check_visited(num_checks, 0);
      for (size_t c = 0; c < num_checks; ++c) {
        if (check_active[c] == 1) {
          unsat_queue.push_back(c);
          check_visited[c] = 1;
        }
      }
      size_t q_head = 0;
      while (q_head < unsat_queue.size()) {
        size_t c = unsat_queue[q_head++];
        size_t v_start = graph_.check_edge_offsets[c];
        size_t v_end = graph_.check_edge_offsets[c + 1];
        for (size_t e = v_start; e < v_end; ++e) {
          size_t v = graph_.check_edges[e];
          size_t deg = graph_.var_edge_offsets[v + 1] - graph_.var_edge_offsets[v];
          if (is_peeled[v] && (posteriors[v] > -trial.unpeel_limit || deg > trial.max_deg)) {
            is_peeled[v] = false;
            size_t c2_start = graph_.var_edge_offsets[v];
            size_t c2_end = graph_.var_edge_offsets[v + 1];
            for (size_t e2 = c2_start; e2 < c2_end; ++e2) {
              size_t c2 = graph_.var_edges[e2];
              if (c2 < num_checks) {
                check_active[c2] ^= 1;
                if (!check_visited[c2]) {
                  check_visited[c2] = 1;
                  unsat_queue.push_back(c2);
                }
              }
            }
          }
        }
      }
    }

    std::vector<uint64_t> residual_detections;
    for (size_t c = 0; c < num_checks; ++c) {
      if (check_active[c] == 1) {
        residual_detections.push_back(c);
      }
    }

    if (!check_active_backup.empty() && residual_detections.size() > trial.max_res_dets &&
        trial.thresh == 0) {
      check_active = std::move(check_active_backup);
      is_peeled = std::move(is_peeled_backup);
      residual_detections.clear();
      for (size_t c = 0; c < num_checks; ++c) {
        if (check_active[c] == 1) {
          residual_detections.push_back(c);
        }
      }
    }

    if (residual_detections.size() > detection_events.size()) {
      is_peeled.assign(num_variables, false);
      residual_detections = detection_events;
    }

    std::vector<uint64_t> valid_residual_detections;
    valid_residual_detections.reserve(residual_detections.size());
    for (uint64_t d : residual_detections) {
      if (d < tesseract_decoder_.num_detectors && !tesseract_decoder_.d2e[d].empty()) {
        valid_residual_detections.push_back(d);
      }
    }

    auto t_step1_end = std::chrono::steady_clock::now();
    double dt_step1 = std::chrono::duration<double>(t_step1_end - t_step1_start).count();
    profile_.t_hard_decision += dt_step1 * 0.5;
    profile_.t_residual += dt_step1 * 0.5;

    if (!first_trial && valid_residual_detections == prev_valid_residual) {
      continue;
    }
    first_trial = false;
    prev_valid_residual = valid_residual_detections;

    if (valid_residual_detections.empty()) {
      int64_t cost = compute_solution_cost(is_peeled);
      if (cost < best_cost) {
        best_cost = cost;
        best_error = is_peeled;
      }
      if (is_fast_mode) {
        break;
      }
      continue;
    }

    auto t_step3_start = std::chrono::steady_clock::now();
    bool solved_via_clusters = false;

    // Decompose valid_residual_detections into connected components (up to 2 detector hops)
    // so separated defect clusters across large codes are solved independently by A* in linear
    // time instead of multiplying state spaces across disconnected clusters.
    if (valid_residual_detections.size() >= 6) {
      std::vector<int32_t> det_to_res_idx(num_checks, -1);
      const size_t n_res = valid_residual_detections.size();
      for (size_t i = 0; i < n_res; ++i) {
        if (valid_residual_detections[i] < num_checks) {
          det_to_res_idx[valid_residual_detections[i]] = static_cast<int32_t>(i);
        }
      }
      std::vector<uint8_t> res_visited(n_res, 0);
      std::vector<std::vector<uint64_t>> clusters;
      std::vector<uint32_t> hop1_checks;
      for (size_t i = 0; i < n_res; ++i) {
        if (res_visited[i]) continue;
        std::vector<uint64_t> cluster;
        res_visited[i] = 1;
        cluster.push_back(valid_residual_detections[i]);
        size_t c_head = 0;
        while (c_head < cluster.size()) {
          uint64_t d0 = cluster[c_head++];
          if (d0 >= num_checks) continue;
          hop1_checks.clear();
          // Hop 1: d0 -> v1 -> d1
          for (size_t e1 = graph_.check_edge_offsets[d0]; e1 < graph_.check_edge_offsets[d0 + 1];
               ++e1) {
            uint32_t v1 = graph_.check_edges[e1];
            for (size_t e2 = graph_.var_edge_offsets[v1]; e2 < graph_.var_edge_offsets[v1 + 1];
                 ++e2) {
              uint32_t d1 = graph_.var_edges[e2];
              if (d1 >= num_checks || d1 == d0) continue;
              int32_t r_idx = det_to_res_idx[d1];
              if (r_idx >= 0 && !res_visited[r_idx]) {
                res_visited[r_idx] = 1;
                cluster.push_back(d1);
              }
              hop1_checks.push_back(d1);
            }
          }
          // Hop 2: d1 -> v2 -> d2
          for (uint32_t d1 : hop1_checks) {
            for (size_t e1 = graph_.check_edge_offsets[d1]; e1 < graph_.check_edge_offsets[d1 + 1];
                 ++e1) {
              uint32_t v2 = graph_.check_edges[e1];
              for (size_t e2 = graph_.var_edge_offsets[v2]; e2 < graph_.var_edge_offsets[v2 + 1];
                   ++e2) {
                uint32_t d2 = graph_.var_edges[e2];
                if (d2 >= num_checks) continue;
                int32_t r_idx2 = det_to_res_idx[d2];
                if (r_idx2 >= 0 && !res_visited[r_idx2]) {
                  res_visited[r_idx2] = 1;
                  cluster.push_back(d2);
                }
              }
            }
          }
        }
        std::sort(cluster.begin(), cluster.end());
        clusters.push_back(std::move(cluster));
      }

      if (clusters.size() > 1) {
        std::vector<uint64_t> unsolved_dets;
        std::vector<bool> cand_error = is_peeled;
        for (const auto& cl : clusters) {
          if (cl.size() > trial.max_res_dets) {
            unsolved_dets.insert(unsolved_dets.end(), cl.begin(), cl.end());
            continue;
          }
          tesseract_decoder::DecodeResult cl_res =
              tesseract_decoder_.decode_result(cl, updates);
          profile_.sum_num_detectors += cl.size();
          if (cl_res.low_confidence) {
            unsolved_dets.insert(unsolved_dets.end(), cl.begin(), cl.end());
            continue;
          }
          for (size_t v : cl_res.predicted_errors) {
            if (v < num_variables) cand_error[v] = !cand_error[v];
          }
        }
        if (!unsolved_dets.empty() &&
            unsolved_dets.size() < valid_residual_detections.size() &&
            unsolved_dets.size() <= trial.max_res_dets) {
          std::sort(unsolved_dets.begin(), unsolved_dets.end());
          tesseract_decoder::DecodeResult rem_res =
              tesseract_decoder_.decode_result(unsolved_dets, updates);
          profile_.sum_num_detectors += unsolved_dets.size();
          if (!rem_res.low_confidence) {
            for (size_t v : rem_res.predicted_errors) {
              if (v < num_variables) cand_error[v] = !cand_error[v];
            }
            unsolved_dets.clear();
          }
        }
        if (unsolved_dets.empty()) {
          std::vector<uint8_t> syn_verify(num_checks, 0);
          for (uint64_t d : detection_events) {
            if (d < num_checks) syn_verify[d] ^= 1;
          }
          for (size_t v = 0; v < num_variables; ++v) {
            if (cand_error[v]) {
              for (size_t e = graph_.var_edge_offsets[v]; e < graph_.var_edge_offsets[v + 1]; ++e) {
                uint32_t c = graph_.var_edges[e];
                if (c < num_checks) syn_verify[c] ^= 1;
              }
            }
          }
          bool syn_ok = true;
          for (size_t c = 0; c < num_checks; ++c) {
            if (syn_verify[c] != 0) {
              syn_ok = false;
              break;
            }
          }
          if (syn_ok) {
            solved_via_clusters = true;
            int64_t cost = compute_solution_cost(cand_error);
            if (cost < best_cost) {
              best_cost = cost;
              best_error = std::move(cand_error);
            }
          }
        }
      }
    }

    if (!solved_via_clusters && valid_residual_detections.size() <= trial.max_res_dets) {
      tesseract_decoder::DecodeResult decode_res =
          tesseract_decoder_.decode_result(valid_residual_detections, updates);
      profile_.sum_num_detectors += valid_residual_detections.size();

      if (!decode_res.low_confidence) {
        std::vector<bool> cand_error = is_peeled;
        for (size_t v : decode_res.predicted_errors) {
          if (v < num_variables) cand_error[v] = !cand_error[v];
        }
        int64_t cost = compute_solution_cost(cand_error);
        if (cost < best_cost) {
          best_cost = cost;
          best_error = std::move(cand_error);
        }
      }
    }
    auto t_step3_end = std::chrono::steady_clock::now();
    profile_.t_elimination += std::chrono::duration<double>(t_step3_end - t_step3_start).count();
    if (is_fast_mode && best_cost < std::numeric_limits<int64_t>::max()) {
      break;
    }
  }

  // Run OSD candidate on unconverged shots (always in Accurate mode, or as fallback in Fast mode)
  if (osd_fallback_ && !bp_result.converged &&
      (!is_fast_mode || best_cost == std::numeric_limits<int64_t>::max())) {
    auto t_fb_start = std::chrono::steady_clock::now();
    std::vector<size_t> osd_preds;
    osd_fallback_->process(bp_result, posteriors, detection_events, hyperedge_observables,
                           &osd_preds, custom_priors);
    std::vector<bool> osd_error(num_variables, false);
    std::vector<uint8_t> syn_check(num_checks, 0);
    for (uint64_t d : detection_events) {
      if (d < num_checks) syn_check[d] ^= 1;
    }
    for (size_t v : osd_preds) {
      if (v < num_variables) {
        osd_error[v] = true;
        for (size_t e = graph_.var_edge_offsets[v]; e < graph_.var_edge_offsets[v + 1]; ++e) {
          uint32_t c = graph_.var_edges[e];
          if (c < num_checks) syn_check[c] ^= 1;
        }
      }
    }
    bool valid_osd = true;
    for (size_t c = 0; c < num_checks; ++c) {
      if (syn_check[c] != 0) {
        valid_osd = false;
        break;
      }
    }
    if (valid_osd) {
      int64_t osd_cost = compute_solution_cost(osd_error);
      if (osd_cost < best_cost) {
        best_cost = osd_cost;
        best_error = std::move(osd_error);
      }
    }
    auto t_fb_end = std::chrono::steady_clock::now();
    profile_.t_elimination += std::chrono::duration<double>(t_fb_end - t_fb_start).count();
  }

  auto t_step4_start = std::chrono::steady_clock::now();
  profile_.sum_pivots_found += (best_cost < std::numeric_limits<int64_t>::max() ? 1 : 0);
  auto obs_status = extract_observables(best_error);
  auto t_step4_end = std::chrono::steady_clock::now();
  profile_.t_solution += std::chrono::duration<double>(t_step4_end - t_step4_start).count();
  profile_.t_total += std::chrono::duration<double>(t_step4_end - t_start_total).count();
  return obs_status;
}

}  // namespace bp
