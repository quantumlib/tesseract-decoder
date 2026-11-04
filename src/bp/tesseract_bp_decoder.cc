#include "bp/tesseract_bp_decoder.h"

#include <algorithm>
#include <chrono>
#include <map>
#include <stdexcept>

#include "bp/bp_params.h"
#include "bp/bp_serial_min_sum.h"
#include "bp/osd_post_processor.h"
#include "common.h"

namespace bp {
namespace {

using ProfileClock = std::chrono::steady_clock;

double seconds_since(const ProfileClock::time_point& start) {
  return std::chrono::duration<double>(ProfileClock::now() - start).count();
}

double compute_error_set_weight(const TannerGraph& graph,
                                const std::vector<LLR_INT>* custom_priors,
                                const std::vector<size_t>& errs) {
  const std::vector<LLR_INT>& p_vec = custom_priors ? *custom_priors : graph.priors;
  double w = 0.0;
  for (size_t v : errs) {
    if (v < p_vec.size()) {
      w += std::max(1e-3, std::abs(llr_int_to_double(p_vec[v])));
    }
  }
  return w;
}

uint64_t isd_order_seed(uint32_t first_det, size_t K, size_t order) {
  uint64_t z = 0x9E3779B97F4A7C15ULL ^ (static_cast<uint64_t>(first_det) << 20) ^
               (static_cast<uint64_t>(K) << 1) ^
               (static_cast<uint64_t>(order) * 0xD1B54A32D192ED03ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z ^= z >> 31;
  return z == 0 ? 1 : z;
}

}  // namespace

TesseractBpDecoder::TesseractBpDecoder(const stim::DetectorErrorModel& dem, const BPParams& params)
    : params_(params) {
  const auto t_construct_start = ProfileClock::now();

  stim::DetectorErrorModel flattened = dem.flattened();
  flat_dem_ = stim::DetectorErrorModel();

  if (params_.bp_merge_duplicates) {
    struct PureDetMergedError {
      double total_prob = 0.0;
      std::map<std::vector<int>, double> obs_coset_probs;
    };
    std::map<std::vector<uint32_t>, PureDetMergedError> merged_by_dets;
    std::vector<std::vector<uint32_t>> dets_order;

    for (const auto& instruction : flattened.instructions) {
      if (instruction.type != stim::DemInstructionType::DEM_ERROR) continue;
      double p = instruction.arg_data.empty() ? 0.0 : instruction.arg_data[0];
      if (p <= 0.0) continue;

      std::vector<uint32_t> dets;
      std::vector<int> obs;
      for (const auto& target : instruction.target_data) {
        if (target.is_relative_detector_id()) {
          dets.push_back(static_cast<uint32_t>(target.val()));
        } else if (target.is_observable_id()) {
          obs.push_back(static_cast<int>(target.val()));
        }
      }
      std::sort(dets.begin(), dets.end());
      size_t w = 0;
      for (size_t r = 0; r < dets.size();) {
        size_t r2 = r + 1;
        while (r2 < dets.size() && dets[r2] == dets[r]) ++r2;
        if ((r2 - r) & 1) dets[w++] = dets[r];
        r = r2;
      }
      dets.resize(w);
      if (dets.empty()) continue;

      std::sort(obs.begin(), obs.end());
      size_t ow = 0;
      for (size_t r = 0; r < obs.size();) {
        size_t r2 = r + 1;
        while (r2 < obs.size() && obs[r2] == obs[r]) ++r2;
        if ((r2 - r) & 1) obs[ow++] = obs[r];
        r = r2;
      }
      obs.resize(ow);

      auto it = merged_by_dets.find(dets);
      if (it == merged_by_dets.end()) {
        PureDetMergedError entry;
        entry.total_prob = p;
        entry.obs_coset_probs[obs] = p;
        merged_by_dets.emplace(dets, std::move(entry));
        dets_order.push_back(std::move(dets));
      } else {
        double p_old = it->second.total_prob;
        it->second.total_prob = p_old * (1.0 - p) + (1.0 - p_old) * p;
        double& p_coset = it->second.obs_coset_probs[obs];
        p_coset = p_coset * (1.0 - p) + (1.0 - p_coset) * p;
      }
    }

    for (const auto& dets : dets_order) {
      const auto& entry = merged_by_dets.at(dets);
      const std::vector<int>* best_obs = nullptr;
      double best_coset_prob = -1.0;
      for (const auto& [obs_vec, c_prob] : entry.obs_coset_probs) {
        if (c_prob > best_coset_prob) {
          best_coset_prob = c_prob;
          best_obs = &obs_vec;
        }
      }
      std::vector<stim::DemTarget> targets;
      targets.reserve(dets.size() + (best_obs ? best_obs->size() : 0));
      for (uint32_t d : dets) {
        targets.push_back(stim::DemTarget::relative_detector_id(d));
      }
      if (best_obs) {
        for (int o : *best_obs) {
          targets.push_back(stim::DemTarget::observable_id(o));
        }
      }
      double eff_prob = std::clamp(entry.total_prob, MIN_PROB, 0.499999);
      flat_dem_.append_error_instruction(eff_prob, targets, "");
    }
  } else {
    for (const auto& instruction : flattened.instructions) {
      if (instruction.type != stim::DemInstructionType::DEM_ERROR) continue;
      double p = instruction.arg_data.empty() ? 0.0 : instruction.arg_data[0];
      if (p <= 0.0) continue;

      std::vector<uint32_t> dets;
      std::vector<int> obs;
      for (const auto& target : instruction.target_data) {
        if (target.is_relative_detector_id()) {
          dets.push_back(static_cast<uint32_t>(target.val()));
        } else if (target.is_observable_id()) {
          obs.push_back(static_cast<int>(target.val()));
        }
      }
      std::sort(dets.begin(), dets.end());
      size_t w = 0;
      for (size_t r = 0; r < dets.size();) {
        size_t r2 = r + 1;
        while (r2 < dets.size() && dets[r2] == dets[r]) ++r2;
        if ((r2 - r) & 1) dets[w++] = dets[r];
        r = r2;
      }
      dets.resize(w);
      if (dets.empty()) continue;

      std::sort(obs.begin(), obs.end());
      size_t ow = 0;
      for (size_t r = 0; r < obs.size();) {
        size_t r2 = r + 1;
        while (r2 < obs.size() && obs[r2] == obs[r]) ++r2;
        if ((r2 - r) & 1) obs[ow++] = obs[r];
        r = r2;
      }
      obs.resize(ow);

      std::vector<stim::DemTarget> targets;
      targets.reserve(dets.size() + obs.size());
      for (uint32_t d : dets) {
        targets.push_back(stim::DemTarget::relative_detector_id(d));
      }
      for (int o : obs) {
        targets.push_back(stim::DemTarget::observable_id(o));
      }
      double eff_prob = std::clamp(p, MIN_PROB, 0.499999);
      flat_dem_.append_error_instruction(eff_prob, targets, "");
    }
  }
  for (const auto& instruction : flattened.instructions) {
    if (instruction.type != stim::DemInstructionType::DEM_ERROR) {
      flat_dem_.append_dem_instruction(instruction);
    }
  }

  size_t num_vars = 0;
  size_t num_checks = flat_dem_.count_detectors();
  for (const auto& instruction : flat_dem_.instructions) {
    if (instruction.type == stim::DemInstructionType::DEM_ERROR) {
      num_vars++;
      for (const auto& target : instruction.target_data) {
        if (target.is_relative_detector_id()) {
          num_checks = std::max(num_checks, static_cast<size_t>(target.val()) + 1);
        }
      }
    }
  }

  hyperedge_observables_.resize(num_vars);
  std::vector<LLR_INT> priors(num_vars);

  size_t var_idx = 0;
  for (const auto& instruction : flat_dem_.instructions) {
    if (instruction.type == stim::DemInstructionType::DEM_ERROR) {
      double p = instruction.arg_data[0];
      priors[var_idx] = prob_double_to_llr_int(p);

      for (const auto& target : instruction.target_data) {
        if (target.is_observable_id()) {
          hyperedge_observables_[var_idx].push_back(target.val());
        }
      }
      var_idx++;
    }
  }

  graph_ = TannerGraph(num_vars, num_checks, priors);
  var_idx = 0;
  for (const auto& instruction : flat_dem_.instructions) {
    if (instruction.type == stim::DemInstructionType::DEM_ERROR) {
      for (const auto& target : instruction.target_data) {
        if (target.is_relative_detector_id()) {
          graph_.add_edge(var_idx, target.val());
        }
      }
      var_idx++;
    }
  }
  graph_.build();

  profile_.t_construct += seconds_since(t_construct_start);
}

tesseract_decoder::common::Symptom TesseractBpDecoder::variable_symptom(size_t var_idx) const {
  tesseract_decoder::common::Symptom symptom;
  size_t start = graph_.var_edge_offsets[var_idx];
  size_t end = graph_.var_edge_offsets[var_idx + 1];
  for (size_t e = start; e < end; ++e) {
    symptom.detectors.push_back(static_cast<int>(graph_.var_edges[e]));
  }
  std::sort(symptom.detectors.begin(), symptom.detectors.end());
  if (var_idx < hyperedge_observables_.size()) {
    for (int obs : hyperedge_observables_[var_idx]) {
      if (obs >= 0) {
        symptom.observables.push_back(obs);
      }
    }
    std::sort(symptom.observables.begin(), symptom.observables.end());
  }
  return symptom;
}

std::shared_ptr<OsdPostProcessor> TesseractBpDecoder::create_osd_post_processor(
    size_t osd_order, size_t osd_weight) const {
  const size_t eff_order = params_.osd_alpha_isd ? osd_order : 0;
  const size_t eff_weight = params_.osd_alpha_isd ? osd_weight : 0;
  return std::make_shared<OsdPostProcessor>(graph_, eff_order, eff_weight,
                                            params_.osd_truncation_factor, params_.osd_early_exit,
                                            params_.osd_col_truncation, params_.osd_partial_sort);
}

std::shared_ptr<OsdPostProcessor> TesseractBpDecoder::create_post_processor() const {
  if (params_.osd_order < 0) {
    return nullptr;
  }
  size_t order = static_cast<size_t>(params_.osd_order);
  size_t weight = static_cast<size_t>(std::max(0, params_.osd_weight));
  if (!params_.osd_alpha_isd) {
    order = 0;
    weight = 0;
  }
  return create_osd_post_processor(order, weight);
}

size_t TesseractBpDecoder::num_observables() const {
  size_t max_obs = 0;
  for (const auto& obs_list : hyperedge_observables_) {
    for (int obs : obs_list) {
      if (obs >= 0) {
        max_obs = std::max(max_obs, static_cast<size_t>(obs) + 1);
      }
    }
  }
  return max_obs;
}

std::vector<uint8_t> TesseractBpDecoder::decode(
    const std::vector<uint64_t>& detection_events,
    const std::shared_ptr<OsdPostProcessor>& post_processor,
    const std::vector<LLR_INT>* custom_priors,
    std::vector<size_t>* predicted_errors) {
  std::vector<std::vector<LLR_INT>> priors_batch;
  if (custom_priors != nullptr) {
    priors_batch.push_back(*custom_priors);
  }
  std::vector<std::vector<size_t>> errors_batch;
  auto batch_res = decode_batch({detection_events}, post_processor,
                                custom_priors ? &priors_batch : nullptr,
                                predicted_errors ? &errors_batch : nullptr);
  if (predicted_errors != nullptr && !errors_batch.empty()) {
    *predicted_errors = std::move(errors_batch[0]);
  }
  return batch_res.empty() ? std::vector<uint8_t>{} : std::move(batch_res[0]);
}

double TesseractBpDecoder::cost_from_errors(const std::vector<size_t>& predicted_errors) const {
  double total_cost = 0.0;
  for (size_t v : predicted_errors) {
    if (v < graph_.priors.size()) {
      total_cost += std::max(1e-3, std::abs(llr_int_to_double(graph_.priors[v])));
    }
  }
  return total_cost;
}

std::vector<std::vector<uint8_t>> TesseractBpDecoder::decode_batch(
    const std::vector<std::vector<uint64_t>>& detection_events_batch,
    const std::shared_ptr<OsdPostProcessor>& post_processor,
    const std::vector<std::vector<LLR_INT>>* custom_priors_per_shot,
    std::vector<std::vector<size_t>>* predicted_errors_batch,
    const std::vector<std::vector<std::pair<size_t, LLR_INT>>>* sparse_custom_priors_per_shot,
    std::vector<uint8_t>* low_confidence_shots) {
  size_t num_shots = detection_events_batch.size();
  std::vector<std::vector<uint8_t>> results(num_shots);
  if (predicted_errors_batch != nullptr) {
    predicted_errors_batch->resize(num_shots);
  }
  if (low_confidence_shots != nullptr) {
    low_confidence_shots->assign(num_shots, 0);
  }
  last_shot_iters_.assign(num_shots, 0);
  last_shot_min_abs_llr_.assign(num_shots, 32767);
  if (num_shots == 0) return results;

  size_t num_vars = graph_.num_vars;
  size_t num_obs = num_observables();

  std::vector<LLR_INT> temp_custom_priors;
  auto get_shot_dense_priors = [&](size_t s) -> const std::vector<LLR_INT>* {
    if (custom_priors_per_shot != nullptr) {
      return &(*custom_priors_per_shot)[s];
    }
    if (sparse_custom_priors_per_shot != nullptr && !(*sparse_custom_priors_per_shot)[s].empty()) {
      temp_custom_priors = graph_.priors;
      for (const auto& [v, p] : (*sparse_custom_priors_per_shot)[s]) {
        if (v < num_vars) temp_custom_priors[v] = p;
      }
      return &temp_custom_priors;
    }
    return nullptr;
  };

  std::vector<std::pair<size_t, LLR_INT>> temp_sparse_priors;
  auto get_shot_sparse_priors =
      [&](size_t s) -> const std::vector<std::pair<size_t, LLR_INT>>* {
    if (sparse_custom_priors_per_shot != nullptr) {
      return &(*sparse_custom_priors_per_shot)[s];
    }
    if (custom_priors_per_shot != nullptr) {
      temp_sparse_priors.clear();
      const auto& p_shot = (*custom_priors_per_shot)[s];
      for (size_t v = 0; v < num_vars; ++v) {
        if (p_shot[v] != graph_.priors[v]) {
          temp_sparse_priors.push_back({v, p_shot[v]});
        }
      }
      return &temp_sparse_priors;
    }
    return nullptr;
  };

  if (params_.bp_mode == BpMode::kWavefront) {
    std::vector<size_t> dets_size_t;
    std::vector<LLR_INT> unconv_post_final;
    std::vector<LLR_INT> unconv_post_tstar;
    const bool need_unconv_post = (post_processor != nullptr);

    for (size_t shot = 0; shot < num_shots; ++shot) {
      const auto& evs = detection_events_batch[shot];
      dets_size_t.assign(evs.begin(), evs.end());
      const auto* sp_priors = get_shot_sparse_priors(shot);

      const auto t_bp_start = ProfileClock::now();
      SparseShotBpResult s_res = bp_wavefront_min_sum_shot(
          graph_, dets_size_t, params_.max_iter, params_.normalization_factor,
          params_.stop_at_convergence, sp_priors,
          need_unconv_post ? &unconv_post_final : nullptr,
          need_unconv_post ? &unconv_post_tstar : nullptr);
      profile_.t_bp += seconds_since(t_bp_start);
      profile_.num_decode_calls++;

      if (s_res.bp_res.converged || post_processor == nullptr) {
        last_shot_iters_[shot] =
            (s_res.bp_res.converged ? 0 : 200) + static_cast<int>(s_res.bp_res.num_iters);
        profile_.num_shots++;
        if (s_res.bp_res.converged) {
          profile_.num_converged++;
          profile_.num_primary_bp_converged++;
        }
        profile_.record_iters(s_res.bp_res.num_iters, s_res.bp_res.converged);
        profile_.sum_active_checks += s_res.active_checks_visited;

        const auto t_post_start = ProfileClock::now();
        std::vector<uint8_t> obs_result(num_obs, 0);
        for (size_t v : s_res.flipped_vars) {
          for (int obs : hyperedge_observables_[v]) {
            if (obs >= 0) obs_result[obs] ^= 1;
          }
        }
        results[shot] = std::move(obs_result);
        if (predicted_errors_batch != nullptr) {
          (*predicted_errors_batch)[shot] = std::move(s_res.flipped_vars);
        }
        profile_.t_post += seconds_since(t_post_start);
      } else {
        last_shot_iters_[shot] = 200 + static_cast<int>(s_res.bp_res.num_iters);
        int min_m = 32767;
        for (LLR_INT x : unconv_post_final) {
          min_m = std::min(min_m, std::abs(static_cast<int>(x)));
        }
        last_shot_min_abs_llr_[shot] = min_m;
        profile_.num_shots++;
        profile_.record_iters(s_res.bp_res.num_iters, false);
        profile_.sum_active_checks += s_res.active_checks_visited;

        const auto t_post_start = ProfileClock::now();
        struct CandidateSolution {
          std::vector<uint8_t> obs;
          double weight = 0.0;
        };
        std::vector<CandidateSolution> candidates;
        candidates.reserve(4);

        const auto* shot_priors = get_shot_dense_priors(shot);
        std::vector<size_t> best_errs;
        std::vector<size_t> cand_errs;
        std::vector<uint8_t> best_obs = post_processor->process(
            s_res.bp_res, unconv_post_final, evs, hyperedge_observables_, &best_errs, shot_priors);
        double best_w = compute_error_set_weight(graph_, shot_priors, best_errs);
        candidates.push_back({best_obs, best_w});

        auto try_sparse_seed = [&](const std::vector<LLR_INT>& post_seed) {
          if (post_seed.empty()) return;
          cand_errs.clear();
          std::vector<uint8_t> cand_obs = post_processor->process(
              s_res.bp_res, post_seed, evs, hyperedge_observables_, &cand_errs, shot_priors);
          double w = compute_error_set_weight(graph_, shot_priors, cand_errs);
          candidates.push_back({cand_obs, w});
          if (w + 1e-5 < best_w) {
            best_w = w;
            best_obs = std::move(cand_obs);
            best_errs = std::move(cand_errs);
          }
        };

        if (num_obs > 0) {
          if (params_.osd_min_syndrome_snapshot && !unconv_post_tstar.empty() &&
              unconv_post_tstar != unconv_post_final) {
            try_sparse_seed(unconv_post_tstar);
          }
          if (params_.osd_alpha_isd && best_errs.size() >= 3 && graph_.num_checks > 24) {
            const std::vector<LLR_INT>& base_post =
                (!params_.osd_min_syndrome_snapshot || unconv_post_tstar.empty())
                    ? unconv_post_final
                    : unconv_post_tstar;
            const std::vector<LLR_INT>& p_vec = shot_priors ? *shot_priors : graph_.priors;
            std::vector<LLR_INT> isd_post(num_vars);
            for (double alpha : {0.35, 0.65}) {
              if (best_errs.size() < 3) break;
              const double one_minus_alpha = 1.0 - alpha;
              for (size_t v = 0; v < num_vars; ++v) {
                isd_post[v] = static_cast<LLR_INT>(
                    std::lround(one_minus_alpha * base_post[v] + alpha * p_vec[v]));
              }
              try_sparse_seed(isd_post);
            }
          }
        }

        const size_t total_valid_candidates = candidates.size();
        bool low_confidence = false;
        if (total_valid_candidates < 2) {
          low_confidence = true;
        } else {
          size_t agree_count = 0;
          double min_diff_w = std::numeric_limits<double>::infinity();
          for (const auto& cand : candidates) {
            if (cand.obs == best_obs) {
              agree_count++;
            } else {
              if (cand.weight < min_diff_w) {
                min_diff_w = cand.weight;
              }
            }
          }
          double agreement_fraction = static_cast<double>(agree_count) / total_valid_candidates;
          double delta_w = std::isinf(min_diff_w)
                               ? std::numeric_limits<double>::infinity()
                               : (min_diff_w - best_w);
          const double fallback_margin = params_.fallback_margin;
          low_confidence = (agreement_fraction < 0.999 || delta_w < fallback_margin);
        }
        if (low_confidence_shots != nullptr) {
          (*low_confidence_shots)[shot] = low_confidence ? 1 : 2;
        }

        results[shot] = std::move(best_obs);
        if (predicted_errors_batch != nullptr) {
          (*predicted_errors_batch)[shot] = std::move(best_errs);
        }
        profile_.t_post += seconds_since(t_post_start);
      }
    }
    return results;
  }

  const size_t batch_stride = BP_BATCH_SIZE;
  const size_t max_batch_shots = params_.bp_simd_batch ? BP_BATCH_SIZE : 1;
  std::vector<std::vector<size_t>> current_batch;
  std::vector<size_t> current_shot_indices;
  alignas(64) std::vector<LLR_INT> posteriors_flat(num_vars * batch_stride);
  alignas(64) std::vector<LLR_INT> tstar_posteriors_flat(num_vars * batch_stride);
  alignas(64) std::vector<LLR_INT> custom_priors_flat(num_vars * batch_stride);
  std::vector<LLR_INT> shot_posteriors(num_vars);

  // Two-stage survivor compaction (Markidis et al., arXiv:2610.28867v1):
  // If max_iter > 4 and stop_at_convergence is enabled, run a fast Stage-1 pass of 4 iterations
  // on all batches, immediately retiring converged shots (>82% on bb_d6) and compacting only
  // the unconverged survivors across batches into dense batch_stride survivor batches for Stage-2.
  const bool use_survivor_compaction = params_.bp_compaction && params_.stop_at_convergence &&
                                       (params_.max_iter > 4) && (num_shots > max_batch_shots);
  const size_t stage1_iters = use_survivor_compaction ? 4 : params_.max_iter;
  const bool enable_unconv_osd =
      (post_processor != nullptr) && (num_obs > 0 || predicted_errors_batch != nullptr);
  const bool enable_unconv_relay =
      enable_unconv_osd && params_.osd_relay_legs && (params_.max_iter >= 10);
  struct SurvivorBpState {
    size_t real_shot_idx;
    std::vector<LLR_INT> posteriors;
    std::vector<LLR_INT> tstar_posteriors;
    std::vector<LLR_INT> check_to_var_messages;
  };
  std::vector<SurvivorBpState> survivor_states;

  struct UnconvergedShotState {
    size_t real_shot_idx;
    BPResult bp_res;
    std::vector<LLR_INT> post_final;
    BPResult tstar_res{false, 0, 0};
    std::vector<LLR_INT> post_tstar;
    BPResult r1_res{false, 0, 0};
    std::vector<LLR_INT> post_r1;
    BPResult r2_res{false, 0, 0};
    std::vector<LLR_INT> post_r2;
    bool retired = false;
  };
  std::vector<UnconvergedShotState> still_unconv;

  auto finalize_shot = [&](size_t real_shot_idx, const BPResult& bp_res,
                           const std::vector<LLR_INT>& flat_post, size_t lane_b,
                           int iter_offset = 0) {
    const auto t_deint_start = ProfileClock::now();
    int min_m = 32767;
    for (size_t v = 0; v < num_vars; ++v) {
      LLR_INT val = flat_post[v * batch_stride + lane_b];
      shot_posteriors[v] = val;
      min_m = std::min(min_m, std::abs(static_cast<int>(val)));
    }
    const int eff_offset = bp_res.converged ? iter_offset : 200;
    last_shot_iters_[real_shot_idx] = eff_offset + static_cast<int>(bp_res.num_iters);
    last_shot_min_abs_llr_[real_shot_idx] = min_m;
    profile_.t_deinterleave += seconds_since(t_deint_start);

    profile_.num_shots++;
    profile_.num_converged += bp_res.converged ? 1 : 0;
    if (bp_res.converged && iter_offset == 0) {
      profile_.num_primary_bp_converged++;
    }
    profile_.record_iters(bp_res.num_iters, bp_res.converged);

    const auto t_post_start = ProfileClock::now();
    const auto* shot_priors = get_shot_dense_priors(real_shot_idx);
    if (post_processor != nullptr) {
      results[real_shot_idx] = post_processor->process(
          bp_res, shot_posteriors, detection_events_batch[real_shot_idx],
          hyperedge_observables_,
          predicted_errors_batch ? &(*predicted_errors_batch)[real_shot_idx] : nullptr,
          shot_priors);
    } else {
      if (predicted_errors_batch != nullptr) {
        (*predicted_errors_batch)[real_shot_idx].clear();
      }
      std::vector<uint8_t> obs_result(num_obs, 0);
      for (size_t v = 0; v < num_vars; ++v) {
        if (shot_posteriors[v] < 0) {
          if (predicted_errors_batch != nullptr) {
            (*predicted_errors_batch)[real_shot_idx].push_back(v);
          }
          for (int obs : hyperedge_observables_[v]) {
            if (obs >= 0) obs_result[obs] ^= 1;
          }
        }
      }
      results[real_shot_idx] = std::move(obs_result);
    }
    profile_.t_post += seconds_since(t_post_start);
  };

  auto record_unconverged_shot = [&](size_t real_shot_idx, const BPResult& bp_res, size_t lane_b,
                                     bool has_tstar) {
    const auto t_deint_start = ProfileClock::now();
    UnconvergedShotState uc;
    uc.real_shot_idx = real_shot_idx;
    uc.bp_res = bp_res;
    uc.post_final.resize(num_vars);
    for (size_t v = 0; v < num_vars; ++v) {
      uc.post_final[v] = posteriors_flat[v * batch_stride + lane_b];
    }
    if (has_tstar) {
      uc.tstar_res = bp_res;
      uc.post_tstar.resize(num_vars);
      for (size_t v = 0; v < num_vars; ++v) {
        uc.post_tstar[v] = tstar_posteriors_flat[v * batch_stride + lane_b];
      }
    }
    profile_.t_deinterleave += seconds_since(t_deint_start);
    still_unconv.push_back(std::move(uc));
  };

  auto populate_batch_custom_priors =
      [&](size_t actual_size, auto get_real_shot_fn) -> const std::vector<LLR_INT>* {
    if (custom_priors_per_shot != nullptr) {
      for (size_t v = 0; v < num_vars; ++v) {
        for (size_t b = 0; b < actual_size; ++b) {
          size_t real_shot_idx = get_real_shot_fn(b);
          custom_priors_flat[v * batch_stride + b] = (*custom_priors_per_shot)[real_shot_idx][v];
        }
        for (size_t b = actual_size; b < batch_stride; ++b) {
          custom_priors_flat[v * batch_stride + b] = graph_.priors[v];
        }
      }
      return &custom_priors_flat;
    }
    if (sparse_custom_priors_per_shot != nullptr) {
      for (size_t v = 0; v < num_vars; ++v) {
        const LLR_INT p = graph_.priors[v];
        for (size_t b = 0; b < batch_stride; ++b) {
          custom_priors_flat[v * batch_stride + b] = p;
        }
      }
      for (size_t b = 0; b < actual_size; ++b) {
        size_t real_shot_idx = get_real_shot_fn(b);
        for (const auto& [v, p] : (*sparse_custom_priors_per_shot)[real_shot_idx]) {
          if (v < num_vars) {
            custom_priors_flat[v * batch_stride + b] = p;
          }
        }
      }
      return &custom_priors_flat;
    }
    return nullptr;
  };

  for (size_t shot = 0; shot < num_shots; ++shot) {
    current_batch.push_back(std::vector<size_t>(detection_events_batch[shot].begin(),
                                                detection_events_batch[shot].end()));
    current_shot_indices.push_back(shot);

    if (current_batch.size() == max_batch_shots || shot == num_shots - 1) {
      size_t actual_size = current_batch.size();
      while (current_batch.size() < batch_stride) {
        current_batch.push_back({});
      }

      const std::vector<LLR_INT>* priors_flat_ptr = populate_batch_custom_priors(
          actual_size, [&](size_t b) { return current_shot_indices[b]; });

      const bool track_stage1_tstar =
          (enable_unconv_osd && params_.osd_min_syndrome_snapshot);
      const auto t_bp_start = ProfileClock::now();
      std::vector<BPResult> bp_results = bp_serial_min_sum(
          graph_, current_batch, posteriors_flat, stage1_iters,
          params_.normalization_factor, params_.stop_at_convergence, params_.random_schedule,
          params_.random_seed, priors_flat_ptr, 0,
          track_stage1_tstar ? &tstar_posteriors_flat : nullptr,
          params_.bp_serial_schedule, params_.bp_simd_batch,
          0, params_.bp_compaction);
      profile_.t_bp += seconds_since(t_bp_start);
      profile_.num_decode_calls++;

      for (size_t b = 0; b < actual_size; ++b) {
        size_t real_shot_idx = current_shot_indices[b];
        if (bp_results[b].converged) {
          finalize_shot(real_shot_idx, bp_results[b], posteriors_flat, b);
        } else if (use_survivor_compaction) {
          SurvivorBpState st;
          st.real_shot_idx = real_shot_idx;
          st.posteriors.resize(num_vars);
          for (size_t v = 0; v < num_vars; ++v) {
            st.posteriors[v] = posteriors_flat[v * batch_stride + b];
          }
          if (track_stage1_tstar) {
            st.tstar_posteriors.resize(num_vars);
            for (size_t v = 0; v < num_vars; ++v) {
              st.tstar_posteriors[v] = tstar_posteriors_flat[v * batch_stride + b];
            }
          }
          st.check_to_var_messages.resize(graph_.num_edges);
          for (size_t e = 0; e < graph_.num_edges; ++e) {
            st.check_to_var_messages[e] = graph_.check_to_var_messages[e * batch_stride + b];
          }
          survivor_states.push_back(std::move(st));
        } else if (enable_unconv_osd) {
          record_unconverged_shot(real_shot_idx, bp_results[b], b, track_stage1_tstar);
        } else {
          finalize_shot(real_shot_idx, bp_results[b], posteriors_flat, b);
        }
      }
      current_batch.clear();
      current_shot_indices.clear();
    }
  }

  // Stage 2: Process compacted unconverged survivor shots in dense batch_stride batches,
  // resuming from iteration 4 (stage1_iters) with saved (L, R) state.
  std::vector<size_t> current_survivor_state_indices;
  for (size_t s_idx = 0; s_idx < survivor_states.size(); ++s_idx) {
    size_t real_shot = survivor_states[s_idx].real_shot_idx;
    current_batch.push_back(std::vector<size_t>(detection_events_batch[real_shot].begin(),
                                                detection_events_batch[real_shot].end()));
    current_shot_indices.push_back(real_shot);
    current_survivor_state_indices.push_back(s_idx);

    if (current_batch.size() == max_batch_shots || s_idx == survivor_states.size() - 1) {
      size_t actual_size = current_batch.size();
      while (current_batch.size() < batch_stride) {
        current_batch.push_back({});
      }

      const std::vector<LLR_INT>* priors_flat_ptr = populate_batch_custom_priors(
          actual_size, [&](size_t b) { return current_shot_indices[b]; });

      const bool track_stage2_tstar = enable_unconv_osd && params_.osd_min_syndrome_snapshot;
      for (size_t v = 0; v < num_vars; ++v) {
        for (size_t b = 0; b < actual_size; ++b) {
          posteriors_flat[v * batch_stride + b] =
              survivor_states[current_survivor_state_indices[b]].posteriors[v];
          if (track_stage2_tstar &&
              !survivor_states[current_survivor_state_indices[b]].tstar_posteriors.empty()) {
            tstar_posteriors_flat[v * batch_stride + b] =
                survivor_states[current_survivor_state_indices[b]].tstar_posteriors[v];
          }
        }
        for (size_t b = actual_size; b < batch_stride; ++b) {
          posteriors_flat[v * batch_stride + b] = graph_.priors[v];
          if (track_stage2_tstar) {
            tstar_posteriors_flat[v * batch_stride + b] = graph_.priors[v];
          }
        }
      }
      for (size_t e = 0; e < graph_.num_edges; ++e) {
        for (size_t b = 0; b < actual_size; ++b) {
          graph_.check_to_var_messages[e * batch_stride + b] =
              survivor_states[current_survivor_state_indices[b]].check_to_var_messages[e];
        }
        for (size_t b = actual_size; b < batch_stride; ++b) {
          graph_.check_to_var_messages[e * batch_stride + b] = 0;
        }
      }
      const auto t_bp_start = ProfileClock::now();
      std::vector<BPResult> bp_results = bp_serial_min_sum(
          graph_, current_batch, posteriors_flat, params_.max_iter,
          params_.normalization_factor, params_.stop_at_convergence, params_.random_schedule,
          params_.random_seed, priors_flat_ptr, 0,
          track_stage2_tstar ? &tstar_posteriors_flat : nullptr,
          params_.bp_serial_schedule, params_.bp_simd_batch,
          stage1_iters, true);
      profile_.t_bp += seconds_since(t_bp_start);
      profile_.num_decode_calls++;

      for (size_t b = 0; b < actual_size; ++b) {
        size_t real_shot_idx = current_shot_indices[b];
        if (bp_results[b].converged || !enable_unconv_osd) {
          finalize_shot(real_shot_idx, bp_results[b], posteriors_flat, b);
        } else {
          record_unconverged_shot(real_shot_idx, bp_results[b], b, track_stage2_tstar);
        }
      }
      current_batch.clear();
      current_shot_indices.clear();
      current_survivor_state_indices.clear();
    }
  }

  if (still_unconv.empty()) {
    return results;
  }

  // Stage 3: Damped Relay Legs on compacted still-unconverged survivors.
  // Any shot that converges in Relay Leg 1 or Relay Leg 2 is immediately retired via
  // finalize_shot (incurring 0 OSD Gaussian-elimination calls).
  if (enable_unconv_relay) {
    const size_t relay_iters = std::min<size_t>(params_.max_iter, 30);
    struct RelayLegSpec {
      float norm_factor;
      int damping_num;
      uint64_t seed_xor;
      int leg_id;
    };
    const RelayLegSpec relay_specs[2] = {
        {0.875f, 1, 0x9e3779b97f4a7c15ULL, 1},
        {0.625f, 2, 0xbf58476d1ce4e5b9ULL, 2},
    };

    for (const auto& spec : relay_specs) {
      std::vector<size_t> leg_indices;
      leg_indices.reserve(still_unconv.size());
      for (size_t idx = 0; idx < still_unconv.size(); ++idx) {
        if (still_unconv[idx].retired) continue;
        leg_indices.push_back(idx);
      }

      for (size_t k = 0; k < leg_indices.size(); ++k) {
        size_t idx = leg_indices[k];
        size_t real_shot = still_unconv[idx].real_shot_idx;
        current_batch.push_back(std::vector<size_t>(detection_events_batch[real_shot].begin(),
                                                    detection_events_batch[real_shot].end()));
        current_shot_indices.push_back(idx);

        if (current_batch.size() == max_batch_shots || k == leg_indices.size() - 1) {
          size_t actual_size = current_batch.size();
          while (current_batch.size() < batch_stride) {
            current_batch.push_back({});
          }

          const std::vector<LLR_INT>* priors_flat_ptr = populate_batch_custom_priors(
              actual_size, [&](size_t b) { return still_unconv[current_shot_indices[b]].real_shot_idx; });

          const auto t_bp_start = ProfileClock::now();
          std::vector<BPResult> r_results = bp_serial_min_sum(
              graph_, current_batch, posteriors_flat, relay_iters,
              spec.norm_factor, params_.stop_at_convergence, params_.random_schedule,
              params_.random_seed ^ spec.seed_xor, priors_flat_ptr, spec.damping_num,
              params_.osd_min_syndrome_snapshot ? &tstar_posteriors_flat : nullptr,
              params_.bp_serial_schedule, params_.bp_simd_batch);
          profile_.t_bp += seconds_since(t_bp_start);
          profile_.num_decode_calls++;

          const auto& src_flat =
              params_.osd_min_syndrome_snapshot ? tstar_posteriors_flat : posteriors_flat;
          for (size_t b = 0; b < actual_size; ++b) {
            size_t u_i = current_shot_indices[b];
            auto& uc = still_unconv[u_i];
            if (r_results[b].converged) {
              finalize_shot(uc.real_shot_idx, r_results[b], posteriors_flat, b, 100);
              uc.retired = true;
              continue;
            }
            const auto t_deint_start = ProfileClock::now();
            BPResult& dst_res = (spec.leg_id == 1) ? uc.r1_res : uc.r2_res;
            std::vector<LLR_INT>& dst_post = (spec.leg_id == 1) ? uc.post_r1 : uc.post_r2;
            dst_res = r_results[b];
            dst_post.resize(num_vars);
            for (size_t v = 0; v < num_vars; ++v) {
              dst_post[v] = src_flat[v * batch_stride + b];
            }
            profile_.t_deinterleave += seconds_since(t_deint_start);
          }

          current_batch.clear();
          current_shot_indices.clear();
        }
      }
    }
  }

  // Evaluate Multi-Seed OSD across distinct BP trajectories (Stage-2 final/t*, Relay-1 t*,
  // Relay-2 t*) on truly unconverged shots and select the minimum-prior-weight valid codeword.
  std::vector<size_t> best_errs;
  std::vector<size_t> cand_errs;
  for (size_t u = 0; u < still_unconv.size(); ++u) {
    auto& uc = still_unconv[u];
    if (uc.retired) continue;
    const size_t real_shot_idx = uc.real_shot_idx;
    const auto& dets = detection_events_batch[real_shot_idx];
    const std::vector<LLR_INT>* shot_priors = get_shot_dense_priors(real_shot_idx);

    profile_.num_shots++;
    profile_.record_iters(uc.bp_res.num_iters, false);
    last_shot_iters_[real_shot_idx] = 200 + static_cast<int>(uc.bp_res.num_iters);
    {
      int min_m = 32767;
      for (LLR_INT x : uc.post_final) {
        min_m = std::min(min_m, std::abs(static_cast<int>(x)));
      }
      last_shot_min_abs_llr_[real_shot_idx] = min_m;
    }

    const auto t_post_start = ProfileClock::now();
    struct CandidateSolution {
      std::vector<uint8_t> obs;
      double weight = 0.0;
    };
    std::vector<CandidateSolution> candidates;
    candidates.reserve(8);

    best_errs.clear();
    std::vector<uint8_t> best_obs = post_processor->process(
        uc.bp_res, uc.post_final, dets, hyperedge_observables_, &best_errs, shot_priors);
    double best_w = compute_error_set_weight(graph_, shot_priors, best_errs);
    candidates.push_back({best_obs, best_w});

    auto try_seed = [&](const BPResult& res, const std::vector<LLR_INT>& post_seed) {
      if (post_seed.empty()) return;
      cand_errs.clear();
      std::vector<uint8_t> cand_obs = post_processor->process(
          res, post_seed, dets, hyperedge_observables_, &cand_errs, shot_priors);
      double w = compute_error_set_weight(graph_, shot_priors, cand_errs);
      candidates.push_back({cand_obs, w});
      if (w + 1e-5 < best_w) {
        best_w = w;
        best_obs = std::move(cand_obs);
        best_errs = std::move(cand_errs);
      }
    };

    if (params_.osd_min_syndrome_snapshot && !uc.post_tstar.empty() &&
        uc.post_tstar != uc.post_final) {
      try_seed(uc.tstar_res, uc.post_tstar);
    }
    if (params_.osd_relay_legs) {
      try_seed(uc.r1_res, uc.post_r1);
      try_seed(uc.r2_res, uc.post_r2);
    }

    if (params_.osd_alpha_isd && best_errs.size() >= 3 && graph_.num_checks > 24 &&
        (num_obs > 0 || best_errs.size() >= 4)) {
      const std::vector<LLR_INT>& base_post =
          (!params_.osd_min_syndrome_snapshot || uc.post_tstar.empty())
              ? uc.post_final
              : uc.post_tstar;
      const BPResult& base_res =
          (!params_.osd_min_syndrome_snapshot || uc.post_tstar.empty()) ? uc.bp_res : uc.tstar_res;
      const std::vector<LLR_INT>& p_vec = shot_priors ? *shot_priors : graph_.priors;
      std::vector<LLR_INT> isd_post(num_vars);

      // Order A (alpha = 0.35) & Order B (alpha = 0.65): Prior-blended orderings
      for (double alpha : {0.35, 0.65}) {
        if (best_errs.size() < 3) break;
        const double one_minus_alpha = 1.0 - alpha;
        for (size_t v = 0; v < num_vars; ++v) {
          isd_post[v] = static_cast<LLR_INT>(
              std::lround(one_minus_alpha * base_post[v] + alpha * p_vec[v]));
        }
        try_seed(base_res, isd_post);
      }

      // Orders C, D, E: Deterministic SplitMix64 hash perturbations (+- 2.5 LLR units)
      const uint32_t first_det = dets.empty() ? 0u : static_cast<uint32_t>(dets[0]);
      const size_t K = dets.size();
      for (size_t order = 1; order <= 3; ++order) {
        if (best_errs.size() < 3) break;
        const uint64_t seed = isd_order_seed(first_det, K, order) ^
                              ((params_.random_seed > (1ULL << 32)) ? params_.random_seed : 0ULL);
        for (size_t v = 0; v < num_vars; ++v) {
          uint64_t h = seed ^ (static_cast<uint64_t>(v) * 0x9E3779B97F4A7C15ULL);
          h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ULL;
          h = (h ^ (h >> 27)) * 0x94D049BB133111EBULL;
          h ^= h >> 31;
          const int32_t noise = static_cast<int32_t>(h % 4096) - 2048;
          isd_post[v] = base_post[v] + noise;
        }
        try_seed(base_res, isd_post);
      }
    }

    const size_t total_valid_candidates = candidates.size();
    bool low_confidence = false;
    if (total_valid_candidates < 2) {
      low_confidence = true;
    } else {
      size_t agree_count = 0;
      double min_diff_w = std::numeric_limits<double>::infinity();
      for (const auto& cand : candidates) {
        if (cand.obs == best_obs) {
          agree_count++;
        } else {
          if (cand.weight < min_diff_w) {
            min_diff_w = cand.weight;
          }
        }
      }
      double agreement_fraction = static_cast<double>(agree_count) / total_valid_candidates;
      double delta_w = std::isinf(min_diff_w)
                           ? std::numeric_limits<double>::infinity()
                           : (min_diff_w - best_w);
      const double fallback_margin = params_.fallback_margin;
      low_confidence = (agreement_fraction < 0.999 || delta_w < fallback_margin);
    }
    if (low_confidence_shots != nullptr) {
      (*low_confidence_shots)[real_shot_idx] = low_confidence ? 1 : 2;
    }

    results[real_shot_idx] = std::move(best_obs);
    if (predicted_errors_batch != nullptr) {
      (*predicted_errors_batch)[real_shot_idx] = std::move(best_errs);
    }
    profile_.t_post += seconds_since(t_post_start);
  }

  return results;
}

}  // namespace bp
