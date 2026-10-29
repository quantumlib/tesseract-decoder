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

  flat_dem_ = stim::DetectorErrorModel();
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
  return std::make_shared<OsdPostProcessor>(graph_, osd_order, osd_weight,
                                            params_.osd_truncation_factor, params_.osd_early_exit);
}

std::shared_ptr<OsdPostProcessor> TesseractBpDecoder::create_post_processor() const {
  size_t order = (params_.osd_order >= 0) ? static_cast<size_t>(params_.osd_order) : 24;
  size_t weight = (params_.osd_order >= 0 || params_.osd_weight > 0)
                      ? static_cast<size_t>(std::max(0, params_.osd_weight))
                      : 2;
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

std::vector<std::vector<uint8_t>> TesseractBpDecoder::decode_batch(
    const std::vector<std::vector<uint64_t>>& detection_events_batch,
    const std::shared_ptr<OsdPostProcessor>& post_processor,
    const std::vector<std::vector<LLR_INT>>* custom_priors_per_shot,
    std::vector<std::vector<size_t>>* predicted_errors_batch) {
  size_t num_shots = detection_events_batch.size();
  std::vector<std::vector<uint8_t>> results(num_shots);
  if (predicted_errors_batch != nullptr) {
    predicted_errors_batch->resize(num_shots);
  }

  size_t num_vars = graph_.num_vars;
  size_t num_obs = num_observables();
  std::vector<std::vector<size_t>> current_batch;
  std::vector<size_t> current_shot_indices;
  alignas(64) std::vector<LLR_INT> posteriors_flat(num_vars * BP_BATCH_SIZE);
  alignas(64) std::vector<LLR_INT> tstar_posteriors_flat(num_vars * BP_BATCH_SIZE);
  alignas(64) std::vector<LLR_INT> custom_priors_flat(num_vars * BP_BATCH_SIZE);
  std::vector<LLR_INT> shot_posteriors(num_vars);

  // Two-stage survivor compaction (Markidis et al., arXiv:2610.28867v1):
  // If max_iter > 4 and stop_at_convergence is enabled, run a fast Stage-1 pass of 4 iterations
  // on all batches, immediately retiring converged shots (>82% on bb_d6) and compacting only
  // the unconverged survivors across batches into dense BP_BATCH_SIZE survivor batches for Stage-2.
  const bool use_survivor_compaction = params_.stop_at_convergence && (params_.max_iter > 4) &&
                                       (num_shots > BP_BATCH_SIZE);
  const size_t stage1_iters = use_survivor_compaction ? 4 : params_.max_iter;
  const bool enable_unconv_relay =
      (post_processor != nullptr) && (num_obs > 0 || predicted_errors_batch != nullptr) &&
      (params_.max_iter >= 10);
  std::vector<size_t> survivor_shot_indices;

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
                           const std::vector<LLR_INT>& flat_post, size_t lane_b) {
    const auto t_deint_start = ProfileClock::now();
    for (size_t v = 0; v < num_vars; ++v) {
      shot_posteriors[v] = flat_post[v * BP_BATCH_SIZE + lane_b];
    }
    profile_.t_deinterleave += seconds_since(t_deint_start);

    profile_.num_shots++;
    profile_.num_converged += bp_res.converged ? 1 : 0;
    profile_.record_iters(bp_res.num_iters, bp_res.converged);

    const auto t_post_start = ProfileClock::now();
    if (post_processor != nullptr) {
      results[real_shot_idx] = post_processor->process(
          bp_res, shot_posteriors, detection_events_batch[real_shot_idx],
          hyperedge_observables_,
          predicted_errors_batch ? &(*predicted_errors_batch)[real_shot_idx] : nullptr,
          custom_priors_per_shot ? &(*custom_priors_per_shot)[real_shot_idx] : nullptr);
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
      uc.post_final[v] = posteriors_flat[v * BP_BATCH_SIZE + lane_b];
    }
    if (has_tstar) {
      uc.tstar_res = bp_res;
      uc.post_tstar.resize(num_vars);
      for (size_t v = 0; v < num_vars; ++v) {
        uc.post_tstar[v] = tstar_posteriors_flat[v * BP_BATCH_SIZE + lane_b];
      }
    }
    profile_.t_deinterleave += seconds_since(t_deint_start);
    still_unconv.push_back(std::move(uc));
  };

  for (size_t shot = 0; shot < num_shots; ++shot) {
    current_batch.push_back(std::vector<size_t>(detection_events_batch[shot].begin(),
                                                detection_events_batch[shot].end()));
    current_shot_indices.push_back(shot);

    if (current_batch.size() == BP_BATCH_SIZE || shot == num_shots - 1) {
      size_t actual_size = current_batch.size();
      while (current_batch.size() < BP_BATCH_SIZE) {
        current_batch.push_back({});
      }

      const std::vector<LLR_INT>* priors_flat_ptr = nullptr;
      if (custom_priors_per_shot != nullptr) {
        for (size_t v = 0; v < num_vars; ++v) {
          for (size_t b = 0; b < actual_size; ++b) {
            size_t real_shot_idx = current_shot_indices[b];
            custom_priors_flat[v * BP_BATCH_SIZE + b] =
                (*custom_priors_per_shot)[real_shot_idx][v];
          }
          for (size_t b = actual_size; b < BP_BATCH_SIZE; ++b) {
            custom_priors_flat[v * BP_BATCH_SIZE + b] = graph_.priors[v];
          }
        }
        priors_flat_ptr = &custom_priors_flat;
      }

      const bool track_stage1_tstar = (!use_survivor_compaction && enable_unconv_relay);
      const auto t_bp_start = ProfileClock::now();
      std::vector<BPResult> bp_results = bp_serial_min_sum(
          graph_, current_batch, posteriors_flat, stage1_iters,
          params_.normalization_factor, params_.stop_at_convergence, params_.random_schedule,
          params_.random_seed, priors_flat_ptr, 0,
          track_stage1_tstar ? &tstar_posteriors_flat : nullptr);
      profile_.t_bp += seconds_since(t_bp_start);
      profile_.num_decode_calls++;

      for (size_t b = 0; b < actual_size; ++b) {
        size_t real_shot_idx = current_shot_indices[b];
        if (bp_results[b].converged) {
          finalize_shot(real_shot_idx, bp_results[b], posteriors_flat, b);
        } else if (use_survivor_compaction) {
          survivor_shot_indices.push_back(real_shot_idx);
        } else if (enable_unconv_relay) {
          record_unconverged_shot(real_shot_idx, bp_results[b], b, track_stage1_tstar);
        } else {
          finalize_shot(real_shot_idx, bp_results[b], posteriors_flat, b);
        }
      }
      current_batch.clear();
      current_shot_indices.clear();
    }
  }

  // Stage 2: Process compacted unconverged survivor shots in dense BP_BATCH_SIZE batches
  for (size_t s_idx = 0; s_idx < survivor_shot_indices.size(); ++s_idx) {
    size_t real_shot = survivor_shot_indices[s_idx];
    current_batch.push_back(std::vector<size_t>(detection_events_batch[real_shot].begin(),
                                                detection_events_batch[real_shot].end()));
    current_shot_indices.push_back(real_shot);

    if (current_batch.size() == BP_BATCH_SIZE || s_idx == survivor_shot_indices.size() - 1) {
      size_t actual_size = current_batch.size();
      while (current_batch.size() < BP_BATCH_SIZE) {
        current_batch.push_back({});
      }

      const std::vector<LLR_INT>* priors_flat_ptr = nullptr;
      if (custom_priors_per_shot != nullptr) {
        for (size_t v = 0; v < num_vars; ++v) {
          for (size_t b = 0; b < actual_size; ++b) {
            size_t real_shot_idx = current_shot_indices[b];
            custom_priors_flat[v * BP_BATCH_SIZE + b] =
                (*custom_priors_per_shot)[real_shot_idx][v];
          }
          for (size_t b = actual_size; b < BP_BATCH_SIZE; ++b) {
            custom_priors_flat[v * BP_BATCH_SIZE + b] = graph_.priors[v];
          }
        }
        priors_flat_ptr = &custom_priors_flat;
      }

      const auto t_bp_start = ProfileClock::now();
      std::vector<BPResult> bp_results = bp_serial_min_sum(
          graph_, current_batch, posteriors_flat, params_.max_iter,
          params_.normalization_factor, params_.stop_at_convergence, params_.random_schedule,
          params_.random_seed, priors_flat_ptr, 0,
          enable_unconv_relay ? &tstar_posteriors_flat : nullptr);
      profile_.t_bp += seconds_since(t_bp_start);
      profile_.num_decode_calls++;

      for (size_t b = 0; b < actual_size; ++b) {
        size_t real_shot_idx = current_shot_indices[b];
        if (bp_results[b].converged || !enable_unconv_relay) {
          finalize_shot(real_shot_idx, bp_results[b], posteriors_flat, b);
        } else {
          record_unconverged_shot(real_shot_idx, bp_results[b], b, true);
        }
      }
      current_batch.clear();
      current_shot_indices.clear();
    }
  }

  if (still_unconv.empty()) {
    return results;
  }

  // Stage 3: Damped Relay Legs on compacted still-unconverged survivors.
  // Any shot that converges in Relay Leg 1 or Relay Leg 2 is immediately retired via
  // finalize_shot (incurring 0 OSD Gaussian-elimination calls).
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

      if (current_batch.size() == BP_BATCH_SIZE || k == leg_indices.size() - 1) {
        size_t actual_size = current_batch.size();
        while (current_batch.size() < BP_BATCH_SIZE) {
          current_batch.push_back({});
        }

        const std::vector<LLR_INT>* priors_flat_ptr = nullptr;
        if (custom_priors_per_shot != nullptr) {
          for (size_t v = 0; v < num_vars; ++v) {
            for (size_t b = 0; b < actual_size; ++b) {
              size_t r_shot = still_unconv[current_shot_indices[b]].real_shot_idx;
              custom_priors_flat[v * BP_BATCH_SIZE + b] = (*custom_priors_per_shot)[r_shot][v];
            }
            for (size_t b = actual_size; b < BP_BATCH_SIZE; ++b) {
              custom_priors_flat[v * BP_BATCH_SIZE + b] = graph_.priors[v];
            }
          }
          priors_flat_ptr = &custom_priors_flat;
        }

        const auto t_bp_start = ProfileClock::now();
        std::vector<BPResult> r_results = bp_serial_min_sum(
            graph_, current_batch, posteriors_flat, relay_iters,
            spec.norm_factor, params_.stop_at_convergence, true,
            params_.random_seed ^ spec.seed_xor, priors_flat_ptr, spec.damping_num,
            &tstar_posteriors_flat);
        profile_.t_bp += seconds_since(t_bp_start);
        profile_.num_decode_calls++;

        for (size_t b = 0; b < actual_size; ++b) {
          size_t u_i = current_shot_indices[b];
          auto& uc = still_unconv[u_i];
          if (r_results[b].converged) {
            finalize_shot(uc.real_shot_idx, r_results[b], posteriors_flat, b);
            uc.retired = true;
            continue;
          }
          const auto t_deint_start = ProfileClock::now();
          BPResult& dst_res = (spec.leg_id == 1) ? uc.r1_res : uc.r2_res;
          std::vector<LLR_INT>& dst_post = (spec.leg_id == 1) ? uc.post_r1 : uc.post_r2;
          dst_res = r_results[b];
          dst_post.resize(num_vars);
          for (size_t v = 0; v < num_vars; ++v) {
            dst_post[v] = tstar_posteriors_flat[v * BP_BATCH_SIZE + b];
          }
          profile_.t_deinterleave += seconds_since(t_deint_start);
        }

        current_batch.clear();
        current_shot_indices.clear();
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
    const std::vector<LLR_INT>* shot_priors =
        custom_priors_per_shot ? &(*custom_priors_per_shot)[real_shot_idx] : nullptr;

    profile_.num_shots++;
    profile_.record_iters(uc.bp_res.num_iters, false);

    const auto t_post_start = ProfileClock::now();
    best_errs.clear();
    std::vector<uint8_t> best_obs = post_processor->process(
        uc.bp_res, uc.post_final, dets, hyperedge_observables_, &best_errs, shot_priors);
    double best_w = compute_error_set_weight(graph_, shot_priors, best_errs);

    auto try_seed = [&](const BPResult& res, const std::vector<LLR_INT>& post_seed) {
      if (post_seed.empty()) return;
      cand_errs.clear();
      std::vector<uint8_t> cand_obs = post_processor->process(
          res, post_seed, dets, hyperedge_observables_, &cand_errs, shot_priors);
      double w = compute_error_set_weight(graph_, shot_priors, cand_errs);
      if (w + 1e-5 < best_w) {
        best_w = w;
        best_obs = std::move(cand_obs);
        best_errs = std::move(cand_errs);
      }
    };

    if (uc.post_tstar != uc.post_final) {
      try_seed(uc.tstar_res, uc.post_tstar);
    }
    try_seed(uc.r1_res, uc.post_r1);
    try_seed(uc.r2_res, uc.post_r2);

    if (best_errs.size() >= 3 && graph_.num_checks > 24 &&
        (num_obs > 0 || best_errs.size() >= 4)) {
      const std::vector<LLR_INT>& base_post =
          uc.post_tstar.empty() ? uc.post_final : uc.post_tstar;
      const BPResult& base_res = uc.post_tstar.empty() ? uc.bp_res : uc.tstar_res;
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
        const uint64_t seed = isd_order_seed(first_det, K, order);
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

    results[real_shot_idx] = std::move(best_obs);
    if (predicted_errors_batch != nullptr) {
      (*predicted_errors_batch)[real_shot_idx] = std::move(best_errs);
    }
    profile_.t_post += seconds_since(t_post_start);
  }

  return results;
}

}  // namespace bp
