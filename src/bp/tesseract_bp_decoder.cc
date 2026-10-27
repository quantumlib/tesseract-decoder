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
  alignas(64) std::vector<LLR_INT> custom_priors_flat(num_vars * BP_BATCH_SIZE);
  std::vector<LLR_INT> shot_posteriors(num_vars);

  // Two-stage survivor compaction (Markidis et al., arXiv:2610.28867v1):
  // If max_iter > 4 and stop_at_convergence is enabled, run a fast Stage-1 pass of 4 iterations
  // on all batches, immediately retiring converged shots (>82% on bb_d6) and compacting only
  // the unconverged survivors across batches into dense BP_BATCH_SIZE survivor batches for Stage-2.
  const bool use_survivor_compaction = params_.stop_at_convergence && (params_.max_iter > 4) &&
                                       (num_shots > BP_BATCH_SIZE);
  const size_t stage1_iters = use_survivor_compaction ? 4 : params_.max_iter;
  std::vector<size_t> survivor_shot_indices;

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

      const auto t_bp_start = ProfileClock::now();
      std::vector<BPResult> bp_results = bp_serial_min_sum(
          graph_, current_batch, posteriors_flat, stage1_iters,
          params_.normalization_factor, params_.stop_at_convergence, params_.random_schedule,
          params_.random_seed, priors_flat_ptr);
      profile_.t_bp += seconds_since(t_bp_start);
      profile_.num_decode_calls++;

      for (size_t b = 0; b < actual_size; ++b) {
        size_t real_shot_idx = current_shot_indices[b];
        if (!use_survivor_compaction || bp_results[b].converged) {
          finalize_shot(real_shot_idx, bp_results[b], posteriors_flat, b);
        } else {
          survivor_shot_indices.push_back(real_shot_idx);
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
          params_.random_seed, priors_flat_ptr);
      profile_.t_bp += seconds_since(t_bp_start);
      profile_.num_decode_calls++;

      for (size_t b = 0; b < actual_size; ++b) {
        finalize_shot(current_shot_indices[b], bp_results[b], posteriors_flat, b);
      }
      current_batch.clear();
      current_shot_indices.clear();
    }
  }

  return results;
}

}  // namespace bp
