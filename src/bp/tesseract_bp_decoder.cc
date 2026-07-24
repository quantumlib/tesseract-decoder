#include "bp/tesseract_bp_decoder.h"

#include <algorithm>
#include <stdexcept>

#include "bp/batched_bp_parallel_min_sum.h"
#include "bp/batched_bp_serial_min_sum.h"
#include "bp/bp_parallel_min_sum.h"
#include "bp/bp_serial_min_sum.h"
#include "bp/check_update.h"  // For prob_double_to_llr_int
#include "bp/osd_post_processor.h"
#include "common.h"  // For merge_indistinguishable_errors

namespace bp {

TesseractBpDecoder::TesseractBpDecoder(const stim::DetectorErrorModel& dem, const BPParams& params)
    : params_(params) {
  std::vector<size_t> error_index_map;
  stim::DetectorErrorModel flat_dem = common::merge_indistinguishable_errors(dem, error_index_map);

  // First pass: count detectors and observables to size variables.
  size_t num_vars = 0;
  size_t num_checks = 0;
  for (const auto& instruction : flat_dem.flattened().instructions) {
    if (instruction.type == stim::DemInstructionType::DEM_ERROR) {
      num_vars++;
      for (const auto& target : instruction.target_data) {
        if (target.is_relative_detector_id()) {
          num_checks = std::max(num_checks, (size_t)target.val() + 1);
        }
      }
    }
  }

  hyperedge_observables_.resize(num_vars);
  std::vector<LLR_INT> priors(num_vars);

  // Second pass: fill priors and collect observables
  size_t var_idx = 0;
  for (const auto& instruction : flat_dem.flattened().instructions) {
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

  // Create temporary graph using constructor to size nodes correctly.
  TannerGraph<LLR_INT> tg(num_vars, num_checks, priors);

  // Third pass: add edges to the graph. We could combine this with the second pass if we build the
  // graph then, but doing it here ensures nodes are sized.
  var_idx = 0;
  for (const auto& instruction : flat_dem.flattened().instructions) {
    if (instruction.type == stim::DemInstructionType::DEM_ERROR) {
      for (const auto& target : instruction.target_data) {
        if (target.is_relative_detector_id()) {
          tg.add_edge(var_idx, target.val());
        }
      }
      var_idx++;
    }
  }

  tg.build();
  graph_ = tg;  // Move or copy the built graph
  batched_graph_.build_from_unbatched(graph_);
}

std::vector<uint8_t> TesseractBpDecoder::decode(
    const std::vector<uint64_t>& detection_events,
    const std::shared_ptr<PostProcessor>& post_processor) {
  std::vector<LLR_INT> posteriors(graph_.variable_nodes.size());

  BPResult result;
  if (params_.schedule == "parallel" && params_.update_rule == "min-sum") {
    std::vector<size_t> dets(detection_events.begin(), detection_events.end());
    result =
        bp_parallel_min_sum<LLR_INT>(graph_, dets, posteriors, params_.max_iter,
                                     params_.normalization_factor, params_.stop_at_convergence);
  } else if (params_.schedule == "serial" && params_.update_rule == "min-sum") {
    std::vector<size_t> dets(detection_events.begin(), detection_events.end());
    result = bp_serial_min_sum<LLR_INT>(graph_, dets, posteriors, params_.max_iter,
                                        params_.normalization_factor, params_.stop_at_convergence);
  } else {
    throw std::invalid_argument(
        "Unsupported schedule/update_rule combination. Only min-sum is supported in this phase.");
  }

  return post_processor->process(result, posteriors, detection_events, hyperedge_observables_);
}

std::shared_ptr<PostProcessor> TesseractBpDecoder::create_osd_post_processor(
    size_t osd_order, size_t osd_weight) const {
  return std::make_shared<OsdPostProcessor>(graph_, osd_order, osd_weight);
}

size_t TesseractBpDecoder::num_observables() const {
  size_t max_obs = 0;
  for (const auto& obs_list : hyperedge_observables_) {
    for (int obs : obs_list) {
      if (obs >= 0) {
        max_obs = std::max(max_obs, (size_t)obs + 1);
      }
    }
  }
  return max_obs;
}

std::vector<std::vector<uint8_t>> TesseractBpDecoder::decode_batch(
    const std::vector<std::vector<uint64_t>>& detection_events_batch,
    const std::shared_ptr<PostProcessor>& post_processor) {
  size_t num_shots = detection_events_batch.size();
  std::vector<std::vector<uint8_t>> results(num_shots);

  std::vector<std::vector<size_t>> current_batch;
  std::vector<std::vector<LLR_INT>> posteriors_batch(
      BP_BATCH_SIZE, std::vector<LLR_INT>(graph_.variable_nodes.size()));

  for (size_t shot = 0; shot < num_shots; ++shot) {
    current_batch.push_back(std::vector<size_t>(detection_events_batch[shot].begin(),
                                                detection_events_batch[shot].end()));

    if (current_batch.size() == BP_BATCH_SIZE || shot == num_shots - 1) {
      size_t actual_size = current_batch.size();
      while (current_batch.size() < BP_BATCH_SIZE) {
        current_batch.push_back({});  // empty syndrome
      }

      std::vector<BPResult> bp_results;
      if (params_.schedule == "serial") {
        bp_results = batched_bp_serial_min_sum(batched_graph_, current_batch, posteriors_batch,
                                               params_.max_iter, params_.normalization_factor,
                                               params_.stop_at_convergence);
      } else {
        bp_results = batched_bp_parallel_min_sum(batched_graph_, current_batch, posteriors_batch,
                                                 params_.max_iter, params_.normalization_factor,
                                                 params_.stop_at_convergence);
      }

      for (size_t b = 0; b < actual_size; ++b) {
        size_t real_shot_idx = shot - actual_size + 1 + b;
        results[real_shot_idx] =
            post_processor->process(bp_results[b], posteriors_batch[b],
                                    detection_events_batch[real_shot_idx], hyperedge_observables_);
      }
      current_batch.clear();
    }
  }

  return results;
}

}  // namespace bp
