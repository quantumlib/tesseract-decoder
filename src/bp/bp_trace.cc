#include "bp/bp_trace.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "bp/check_update.h"
#include "bp/hard_decision_post_processor.h"
#include "bp/multi_pass_bp_decoder.h"
#include "bp/osd_post_processor.h"
#include "bp/tanner_graph_util.h"
#include "bp/tesseract_bp_decoder.h"
#include "bp/tesseract_post_processor.h"

namespace bp {

TannerGraphTopology TesseractBpDecoder::get_topology() const {
  TannerGraphTopology topo;
  topo.num_vars = graph_.variable_nodes.size();
  topo.num_checks = graph_.check_nodes.size();
  topo.num_edges = graph_.var_edges.size();
  topo.num_observables = num_observables();

  topo.priors.resize(topo.num_vars);
  topo.obs_masks.assign(topo.num_vars, 0ULL);
  topo.hyperedge_observables = hyperedge_observables_;

  for (size_t v = 0; v < topo.num_vars; ++v) {
    topo.priors[v] = llr_int_to_double(graph_.variable_nodes[v].prior);
    if (v < hyperedge_observables_.size()) {
      for (int obs : hyperedge_observables_[v]) {
        if (obs >= 0 && obs < 64) {
          topo.obs_masks[v] |= (1ULL << obs);
        }
      }
    }
  }

  topo.edges.resize(topo.num_edges);
  for (size_t v = 0; v < topo.num_vars; ++v) {
    size_t start = graph_.var_edge_offsets[v];
    size_t end = graph_.var_edge_offsets[v + 1];
    for (size_t e = start; e < end; ++e) {
      topo.edges[e] = {v, graph_.var_edges[e]};
    }
  }

  return topo;
}

std::vector<uint8_t> OsdPostProcessor::process_with_trace(
    const BPResult& bp_result, const std::vector<LLR_INT>& posteriors,
    const std::vector<uint64_t>& detection_events,
    const std::vector<std::vector<int>>& hyperedge_observables,
    OsdTrace& trace,
    std::vector<size_t>* predicted_errors,
    const std::vector<LLR_INT>* custom_priors) {
  trace = OsdTrace();
  size_t num_errors = graph_.variable_nodes.size();
  size_t num_detectors = graph_.check_nodes.size();

  if (bp_result.converged) {
    trace.triggered = false;
    std::vector<uint8_t> res = process(bp_result, posteriors, detection_events,
                                       hyperedge_observables, predicted_errors, custom_priors);
    if (predicted_errors != nullptr) {
      trace.best_error = *predicted_errors;
    }
    return res;
  }

  trace.triggered = true;

  // 1. Initial hard decision and residual syndrome
  std::vector<bool> e_hard(num_errors, false);
  for (size_t i = 0; i < num_errors; ++i) {
    if (posteriors[i] < 0) {
      e_hard[i] = true;
    }
  }

  stim::simd_bits<64> S_res(num_detectors);
  for (auto d : detection_events) {
    if (d < num_detectors) S_res[d] ^= true;
  }
  for (size_t i = 0; i < num_errors; ++i) {
    if (e_hard[i]) {
      size_t start = graph_.var_edge_offsets[i];
      size_t end = graph_.var_edge_offsets[i + 1];
      for (size_t e = start; e < end; ++e) {
        uint32_t d = graph_.var_edges[e];
        if (d < num_detectors) S_res[d] ^= true;
      }
    }
  }

  trace.residual_syndrome.resize(num_detectors, false);
  for (size_t d = 0; d < num_detectors; ++d) {
    trace.residual_syndrome[d] = S_res[d];
  }

  // 2. Reliability sort
  size_t max_osd_cols = num_errors;
  if (osd_truncation_factor_ > 0.0f) {
    max_osd_cols = std::min(num_errors, static_cast<size_t>(num_detectors * osd_truncation_factor_));
  }
  std::vector<size_t> sorted_cols(num_errors);
  std::iota(sorted_cols.begin(), sorted_cols.end(), 0);
  auto cmp = [&](size_t a, size_t b) {
    return std::abs(posteriors[a]) < std::abs(posteriors[b]);
  };
  if (max_osd_cols < num_errors) {
    std::nth_element(sorted_cols.begin(), sorted_cols.begin() + max_osd_cols, sorted_cols.end(),
                     cmp);
    std::sort(sorted_cols.begin(), sorted_cols.begin() + max_osd_cols, cmp);
  } else {
    std::sort(sorted_cols.begin(), sorted_cols.end(), cmp);
  }
  trace.col_to_var.assign(sorted_cols.begin(), sorted_cols.begin() + max_osd_cols);

  // 3. Pivot vs Free variable identification via Gaussian elimination
  std::vector<stim::simd_bits<64>> H_rows(num_detectors, stim::simd_bits<64>(max_osd_cols));
  for (size_t i = 0; i < max_osd_cols; ++i) {
    size_t c = sorted_cols[i];
    size_t start = graph_.var_edge_offsets[c];
    size_t end = graph_.var_edge_offsets[c + 1];
    for (size_t e = start; e < end; ++e) {
      uint32_t d = graph_.var_edges[e];
      if (d < num_detectors) H_rows[d][i] = true;
    }
  }

  size_t num_solved = 0;
  for (size_t i = 0; i < max_osd_cols && num_solved < num_detectors; ++i) {
    size_t c = sorted_cols[i];
    size_t pivot_r = SIZE_MAX;
    for (size_t r = num_solved; r < num_detectors; ++r) {
      if (H_rows[r][i]) {
        pivot_r = r;
        break;
      }
    }
    if (pivot_r == SIZE_MAX) {
      trace.free_vars.push_back(c);
      continue;
    }
    if (pivot_r != num_solved) {
      std::swap(H_rows[pivot_r], H_rows[num_solved]);
    }
    pivot_r = num_solved;
    for (size_t r = 0; r < num_detectors; ++r) {
      if (r != pivot_r && H_rows[r][i]) {
        H_rows[r] ^= H_rows[pivot_r];
      }
    }
    trace.pivot_vars.push_back(c);
    num_solved++;
  }

  std::vector<size_t> local_preds;
  std::vector<size_t>* err_ptr = predicted_errors ? predicted_errors : &local_preds;
  std::vector<uint8_t> obs_res = process(bp_result, posteriors, detection_events,
                                         hyperedge_observables, err_ptr, custom_priors);
  trace.best_error = *err_ptr;
  return obs_res;
}

std::vector<uint8_t> TesseractPostProcessor::process_with_trace(
    const BPResult& bp_result, const std::vector<LLR_INT>& posteriors,
    const std::vector<uint64_t>& detection_events,
    const std::vector<std::vector<int>>& hyperedge_observables,
    TesseractTrace& trace,
    OsdTrace* osd_fallback_trace,
    std::vector<size_t>* predicted_errors,
    const std::vector<LLR_INT>* custom_priors) {
  trace = TesseractTrace();
  if (osd_fallback_trace != nullptr) {
    *osd_fallback_trace = OsdTrace();
  }

  const size_t num_variables = graph_.variable_nodes.size();
  const size_t num_checks = graph_.check_nodes.size();

  trace.residual_llr_costs.resize(num_variables);
  for (size_t v = 0; v < num_variables; ++v) {
    LLR_INT p = custom_priors ? (*custom_priors)[v] : graph_.variable_nodes[v].prior;
    trace.residual_llr_costs[v] = llr_int_to_double(p);
  }

  if (bp_result.converged) {
    trace.triggered = false;
    std::vector<size_t> local_preds;
    std::vector<size_t>* err_ptr = predicted_errors ? predicted_errors : &local_preds;
    std::vector<uint8_t> obs_res = process(bp_result, posteriors, detection_events,
                                           hyperedge_observables, err_ptr, custom_priors);
    trace.final_flipped_vars = *err_ptr;
    return obs_res;
  }

  trace.triggered = true;

  // Compute primary peel state (hard-decision peel)
  std::vector<bool> is_peeled(num_variables, false);
  for (size_t v = 0; v < num_variables; ++v) {
    if (posteriors[v] < 0) {
      is_peeled[v] = true;
    }
  }

  std::vector<uint8_t> check_active(num_checks, 0);
  for (uint64_t d : detection_events) {
    if (d < num_checks) check_active[d] ^= 1;
  }
  for (size_t v = 0; v < num_variables; ++v) {
    if (is_peeled[v]) {
      for (size_t e = graph_.var_edge_offsets[v]; e < graph_.var_edge_offsets[v + 1]; ++e) {
        size_t c = graph_.var_edges[e];
        if (c < num_checks) check_active[c] ^= 1;
      }
    }
  }

  // Record peeled_vars, active_vars, and unsolved_dets
  for (size_t v = 0; v < num_variables; ++v) {
    if (is_peeled[v]) {
      trace.peeled_vars.push_back(v);
    } else {
      trace.active_vars.push_back(v);
    }
  }
  for (size_t c = 0; c < num_checks; ++c) {
    if (check_active[c] == 1) {
      trace.unsolved_dets.push_back(c);
    }
  }

  // 2-Hop BFS connected components on unsolved_dets
  if (!trace.unsolved_dets.empty()) {
    std::vector<int32_t> det_to_res_idx(num_checks, -1);
    const size_t n_res = trace.unsolved_dets.size();
    for (size_t i = 0; i < n_res; ++i) {
      if (trace.unsolved_dets[i] < num_checks) {
        det_to_res_idx[trace.unsolved_dets[i]] = static_cast<int32_t>(i);
      }
    }
    std::vector<uint8_t> res_visited(n_res, 0);
    std::vector<uint32_t> hop1_checks;
    for (size_t i = 0; i < n_res; ++i) {
      if (res_visited[i]) continue;
      std::vector<size_t> cluster;
      res_visited[i] = 1;
      cluster.push_back(trace.unsolved_dets[i]);
      size_t c_head = 0;
      while (c_head < cluster.size()) {
        size_t d0 = cluster[c_head++];
        if (d0 >= num_checks) continue;
        hop1_checks.clear();
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
      std::vector<size_t> c_vars;
      std::vector<uint8_t> var_seen(num_variables, 0);
      for (size_t d : cluster) {
        for (size_t e = graph_.check_edge_offsets[d]; e < graph_.check_edge_offsets[d + 1]; ++e) {
          uint32_t v = graph_.check_edges[e];
          if (v < num_variables && !var_seen[v]) {
            var_seen[v] = 1;
            c_vars.push_back(v);
          }
        }
      }
      std::sort(c_vars.begin(), c_vars.end());
      trace.comp_dets.push_back(std::move(cluster));
      trace.comp_vars.push_back(std::move(c_vars));
    }
  }

  std::vector<size_t> local_preds;
  std::vector<size_t>* err_ptr = predicted_errors ? predicted_errors : &local_preds;
  std::vector<uint8_t> obs_res = process(bp_result, posteriors, detection_events,
                                         hyperedge_observables, err_ptr, custom_priors);
  trace.final_flipped_vars = *err_ptr;

  // Compute symmetric difference between final_flipped_vars and peeled_vars as astar_flipped_vars
  std::vector<bool> final_mask(num_variables, false);
  for (size_t v : trace.final_flipped_vars) {
    if (v < num_variables) final_mask[v] = true;
  }
  for (size_t v = 0; v < num_variables; ++v) {
    if (final_mask[v] != is_peeled[v]) {
      trace.astar_flipped_vars.push_back(v);
    }
  }

  if (osd_fallback_ && osd_fallback_trace != nullptr) {
    std::vector<size_t> osd_preds;
    osd_fallback_->process_with_trace(bp_result, posteriors, detection_events,
                                      hyperedge_observables, *osd_fallback_trace, &osd_preds,
                                      custom_priors);
    if (osd_preds == trace.final_flipped_vars) {
      trace.used_osd_fallback = true;
    }
  }

  return obs_res;
}

BpDecodeTrace TesseractBpDecoder::decode_trace(
    const std::vector<uint64_t>& detection_events,
    const std::shared_ptr<PostProcessor>& post_processor,
    const std::vector<LLR_INT>* custom_priors) {
  BpDecodeTrace trace;
  trace.topology = get_topology();
  if (custom_priors != nullptr) {
    for (size_t v = 0; v < trace.topology.num_vars && v < custom_priors->size(); ++v) {
      trace.topology.priors[v] = llr_int_to_double((*custom_priors)[v]);
    }
  }
  trace.detection_events = detection_events;

  const size_t num_vars = graph_.variable_nodes.size();
  const size_t num_checks = graph_.check_nodes.size();
  const size_t num_edges = graph_.var_edges.size();

  std::vector<size_t> dets(detection_events.begin(), detection_events.end());
  graph_.add_detection_events(dets);

  for (size_t v = 0; v < num_vars; ++v) {
    LLR_INT p = (custom_priors != nullptr) ? (*custom_priors)[v] : graph_.variable_nodes[v].prior;
    graph_.variable_nodes[v].posterior = p;
  }
  std::fill(graph_.check_to_var_messages.begin(), graph_.check_to_var_messages.end(), 0);
  std::fill(graph_.var_to_check_messages.begin(), graph_.var_to_check_messages.end(), 0);

  bool is_random_sched = params_.random_schedule || params_.schedule == "random-serial" ||
                         params_.schedule == "stochastic-serial" || params_.schedule == "random";

  using T_MAG = typename llr_traits<LLR_INT>::magnitude_type;
  const T_MAG max_mag = std::numeric_limits<T_MAG>::max();

  std::vector<size_t> check_order(num_checks);
  std::iota(check_order.begin(), check_order.end(), 0);
  uint64_t rng_state = params_.random_seed ? params_.random_seed : 123456789ULL;
  auto fast_rand = [&rng_state]() -> uint32_t {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return static_cast<uint32_t>((rng_state * 0x2545F4914F6CDD1DULL) >> 32);
  };

  bool has_converged = false;
  size_t iter = 0;

  for (iter = 0; iter < params_.max_iter; ++iter) {
    BpIterationTrace iter_trace;
    iter_trace.iter = iter + 1;
    iter_trace.var_to_check_messages.assign(num_edges, 0.0);
    iter_trace.check_to_var_messages.assign(num_edges, 0.0);
    iter_trace.posteriors.resize(num_vars);
    iter_trace.hard_decisions.resize(num_vars);
    iter_trace.check_satisfied.resize(num_checks);
    iter_trace.residual_syndrome.resize(num_checks);

    if (params_.schedule == "parallel") {
      // Phase 1: V -> C
      for (size_t v = 0; v < num_vars; ++v) {
        LLR_INT p = (custom_priors != nullptr) ? (*custom_priors)[v] : graph_.variable_nodes[v].prior;
        LLR_INT total_c2v = p;
        size_t start = graph_.var_edge_offsets[v];
        size_t end = graph_.var_edge_offsets[v + 1];
        for (size_t v_e = start; v_e < end; ++v_e) {
          total_c2v = sat_add(total_c2v, graph_.check_to_var_messages[v_e]);
        }
        for (size_t v_e = start; v_e < end; ++v_e) {
          LLR_INT v2c_msg = total_c2v - graph_.check_to_var_messages[v_e];
          size_t c_e = graph_.var_edge_rev_indices[v_e];
          graph_.var_to_check_messages[c_e] = v2c_msg;
          iter_trace.var_to_check_messages[v_e] = llr_int_to_double(v2c_msg);
        }
      }
      // Phase 2: C -> V
      for (size_t c = 0; c < num_checks; ++c) {
        const auto& check = graph_.check_nodes[c];
        size_t start = graph_.check_edge_offsets[c];
        size_t end = graph_.check_edge_offsets[c + 1];
        if (start == end) continue;

        T_MAG min1 = max_mag;
        T_MAG min2 = max_mag;
        size_t min1_idx = SIZE_MAX;
        bool total_sign_prod = false;
        for (size_t c_e = start; c_e < end; ++c_e) {
          LLR_INT v2c_msg = graph_.var_to_check_messages[c_e];
          T_MAG mag = static_cast<T_MAG>(std::abs(v2c_msg));
          total_sign_prod ^= (v2c_msg < 0);
          if (mag < min1) {
            min2 = min1;
            min1 = mag;
            min1_idx = c_e;
          } else if (mag < min2) {
            min2 = mag;
          }
        }
        for (size_t c_e = start; c_e < end; ++c_e) {
          bool extrinsic_sign = total_sign_prod ^ (graph_.var_to_check_messages[c_e] < 0);
          bool final_sign = check.syndrome ^ extrinsic_sign;
          T_MAG mag_to_send = (c_e == min1_idx) ? min2 : min1;
          T_MAG normalized_mag = static_cast<T_MAG>(mag_to_send * params_.normalization_factor);
          LLR_INT final_mag = static_cast<LLR_INT>(
              std::min(normalized_mag, static_cast<T_MAG>(std::numeric_limits<LLR_INT>::max())));
          LLR_INT c2v_msg = final_sign ? -final_mag : final_mag;
          size_t v_e = graph_.check_edge_rev_indices[c_e];
          graph_.check_to_var_messages[v_e] = c2v_msg;
          iter_trace.check_to_var_messages[v_e] = llr_int_to_double(c2v_msg);
        }
      }
      // Update posteriors
      for (size_t v = 0; v < num_vars; ++v) {
        LLR_INT p = (custom_priors != nullptr) ? (*custom_priors)[v] : graph_.variable_nodes[v].prior;
        LLR_INT post = p;
        for (size_t v_e = graph_.var_edge_offsets[v]; v_e < graph_.var_edge_offsets[v + 1]; ++v_e) {
          post = sat_add(post, graph_.check_to_var_messages[v_e]);
        }
        graph_.variable_nodes[v].posterior = post;
      }
    } else {
      // Layered / Serial Min-Sum
      if (is_random_sched && num_checks > 1) {
        for (size_t i = num_checks - 1; i > 0; --i) {
          size_t j = fast_rand() % (i + 1);
          std::swap(check_order[i], check_order[j]);
        }
      }

      for (size_t c_idx = 0; c_idx < num_checks; ++c_idx) {
        size_t c = check_order[c_idx];
        const auto& check = graph_.check_nodes[c];
        size_t start = graph_.check_edge_offsets[c];
        size_t end = graph_.check_edge_offsets[c + 1];
        if (start == end) continue;

        T_MAG min1 = max_mag;
        T_MAG min2 = max_mag;
        size_t min1_idx = SIZE_MAX;
        uint8_t total_sign_prod = 0;

        for (size_t c_e = start; c_e < end; ++c_e) {
          size_t v = graph_.check_edges[c_e];
          LLR_INT old_r = graph_.check_to_var_messages[c_e];
          LLR_INT q_msg = graph_.variable_nodes[v].posterior - old_r;
          size_t v_e = graph_.check_edge_rev_indices[c_e];
          iter_trace.var_to_check_messages[v_e] = llr_int_to_double(q_msg);

          T_MAG mag = static_cast<T_MAG>(std::abs(q_msg));
          uint8_t sign = (q_msg < 0) ? 1 : 0;
          total_sign_prod ^= sign;
          if (mag < min1) {
            min2 = min1;
            min1 = mag;
            min1_idx = c_e;
          } else if (mag < min2) {
            min2 = mag;
          }
        }

        for (size_t c_e = start; c_e < end; ++c_e) {
          size_t v = graph_.check_edges[c_e];
          LLR_INT old_r = graph_.check_to_var_messages[c_e];
          LLR_INT q_msg = graph_.variable_nodes[v].posterior - old_r;
          uint8_t q_sign = (q_msg < 0) ? 1 : 0;
          uint8_t extrinsic_sign = total_sign_prod ^ q_sign;
          uint8_t final_sign = check.syndrome ^ extrinsic_sign;

          T_MAG min_mag = (c_e == min1_idx) ? min2 : min1;
          T_MAG normalized_mag = static_cast<T_MAG>(min_mag * params_.normalization_factor);
          LLR_INT final_mag = static_cast<LLR_INT>(
              std::min(normalized_mag, static_cast<T_MAG>(std::numeric_limits<LLR_INT>::max())));
          LLR_INT new_r = final_sign ? -final_mag : final_mag;
          graph_.check_to_var_messages[c_e] = new_r;
          graph_.variable_nodes[v].posterior = sat_add(q_msg, new_r);

          size_t v_e = graph_.check_edge_rev_indices[c_e];
          iter_trace.check_to_var_messages[v_e] = llr_int_to_double(new_r);
        }
      }
    }

    for (size_t v = 0; v < num_vars; ++v) {
      LLR_INT post = graph_.variable_nodes[v].posterior;
      iter_trace.posteriors[v] = llr_int_to_double(post);
      iter_trace.hard_decisions[v] = (post < 0);
    }

    bool all_sat = true;
    for (size_t c = 0; c < num_checks; ++c) {
      bool res = graph_.check_nodes[c].syndrome;
      for (size_t c_e = graph_.check_edge_offsets[c]; c_e < graph_.check_edge_offsets[c + 1];
           ++c_e) {
        size_t v = graph_.check_edges[c_e];
        if (iter_trace.hard_decisions[v]) {
          res = !res;
        }
      }
      iter_trace.residual_syndrome[c] = res;
      iter_trace.check_satisfied[c] = !res;
      if (res) all_sat = false;
    }

    iter_trace.converged = all_sat;
    trace.iterations.push_back(std::move(iter_trace));

    if (params_.stop_at_convergence && all_sat) {
      has_converged = true;
      iter++;
      break;
    }
    if (iter == params_.max_iter - 1) {
      has_converged = all_sat;
    }
  }

  std::vector<LLR_INT> posteriors_int(num_vars);
  for (size_t v = 0; v < num_vars; ++v) {
    posteriors_int[v] = graph_.variable_nodes[v].posterior;
  }
  graph_.remove_detection_events(dets);

  trace.bp_converged = has_converged;
  trace.num_iters = iter;
  BPResult bp_res{has_converged, iter};

  std::shared_ptr<PostProcessor> pp = post_processor ? post_processor : create_post_processor();
  if (auto tess_pp = std::dynamic_pointer_cast<TesseractPostProcessor>(pp)) {
    trace.post_processor_type = "tesseract";
    trace.predicted_observables = tess_pp->process_with_trace(
        bp_res, posteriors_int, detection_events, hyperedge_observables_, trace.tesseract_trace,
        &trace.osd_trace, &trace.predicted_errors, custom_priors);
  } else if (auto osd_pp = std::dynamic_pointer_cast<OsdPostProcessor>(pp)) {
    trace.post_processor_type = "osd";
    trace.predicted_observables = osd_pp->process_with_trace(
        bp_res, posteriors_int, detection_events, hyperedge_observables_, trace.osd_trace,
        &trace.predicted_errors, custom_priors);
  } else {
    trace.post_processor_type = "hard";
    trace.predicted_observables = pp->process(
        bp_res, posteriors_int, detection_events, hyperedge_observables_, &trace.predicted_errors,
        custom_priors);
  }

  return trace;
}

TannerGraphTopology MultiPassBpDecoder::get_component_topology(size_t comp_idx) const {
  if (comp_idx >= 2) {
    throw std::out_of_range("comp_idx must be 0 or 1");
  }
  return component_decoders_[comp_idx].decoder->get_topology();
}

MultiPassBpDecodeTrace MultiPassBpDecoder::decode_trace(
    const std::vector<uint64_t>& detection_events) {
  MultiPassBpDecodeTrace mp_trace;
  mp_trace.component_topologies = {
      component_decoders_[0].decoder->get_topology(),
      component_decoders_[1].decoder->get_topology(),
  };
  mp_trace.detector_components = global_det_to_comp_id_;
  mp_trace.pass_schedule = pass_schedule_;

  std::array<std::vector<uint64_t>, 2> comp_syndromes;
  for (uint64_t d : detection_events) {
    if (d < global_det_to_comp_id_.size()) {
      int c = global_det_to_comp_id_[d];
      if (c == 0 || c == 1) {
        comp_syndromes[c].push_back(d);
      }
    }
  }

  std::array<std::vector<LLR_INT>, 2> custom_priors;
  for (size_t c = 0; c < 2; ++c) {
    size_t n_vars = component_decoders_[c].decoder->num_variables();
    const auto& var_nodes = component_decoders_[c].decoder->graph().variable_nodes;
    custom_priors[c].resize(n_vars);
    for (size_t v = 0; v < n_vars; ++v) {
      custom_priors[c][v] = var_nodes[v].prior;
    }
  }

  std::array<std::vector<uint8_t>, 2> last_comp_obs;
  for (size_t pass = 0; pass < pass_schedule_.size(); ++pass) {
    const bool is_final_pass = (pass + 1 == pass_schedule_.size());
    std::array<std::vector<size_t>, 2> pass_predicted_errors;

    for (size_t comp : pass_schedule_[pass]) {
      ComponentPassTrace cp_trace;
      cp_trace.pass = pass;
      cp_trace.comp_idx = comp;
      cp_trace.trace = component_decoders_[comp].decoder->decode_trace(
          comp_syndromes[comp], component_decoders_[comp].post_processor,
          (pass == 0) ? nullptr : &custom_priors[comp]);
      pass_predicted_errors[comp] = cp_trace.trace.predicted_errors;
      last_comp_obs[comp] = cp_trace.trace.predicted_observables;
      mp_trace.component_passes.push_back(std::move(cp_trace));
    }

    if (!is_final_pass) {
      for (size_t source_comp : pass_schedule_[pass]) {
        const auto& reweight_map = component_decoders_[source_comp].reweight_rules;
        if (reweight_map.empty()) continue;

        for (size_t v_src : pass_predicted_errors[source_comp]) {
          auto it = reweight_map.find(v_src);
          if (it == reweight_map.end()) continue;
          for (const auto& rule : it->second) {
            LLR_INT& cur_p = custom_priors[rule.target_comp_idx][rule.target_var_idx];
            LLR_INT old_p = cur_p;
            LLR_INT new_p = old_p;
            if (soft_reweight_) {
              new_p = std::min(old_p, static_cast<LLR_INT>((old_p + 3 * rule.reweight_llr) / 4));
            } else {
              new_p = std::min(old_p, rule.reweight_llr);
            }
            if (new_p < old_p) {
              cur_p = new_p;
              ReweightEventTrace ev;
              ev.pass = pass;
              ev.source_comp_idx = source_comp;
              ev.source_var_idx = v_src;
              ev.target_comp_idx = rule.target_comp_idx;
              ev.target_var_idx = rule.target_var_idx;
              ev.old_prior = llr_int_to_double(old_p);
              ev.new_prior = llr_int_to_double(new_p);
              mp_trace.reweight_events.push_back(ev);
            }
          }
        }
      }
    }
  }

  mp_trace.predicted_observables = decode(detection_events);
  return mp_trace;
}

}  // namespace bp
