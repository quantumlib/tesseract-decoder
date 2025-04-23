// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "confidences_decoder.h"

using confidences_decoder::ConfidencesDecoder;

// Helper function that returns the given DEM with one of the observables
// transformed into a detector.
stim::DetectorErrorModel fix_obs(const stim::DetectorErrorModel& dem,
                                    uint64_t obs_idx) {

  if (obs_idx >= dem.count_observables()) {
    throw std::runtime_error("Observable ID must correspond to a valid "
                                 "observable index in the DEM.");
  }

  stim::DetectorErrorModel result;
  uint64_t num_detectors = dem.count_detectors();
    for (stim::DemInstruction inst : dem.instructions) {
      if (inst.type != stim::DemInstructionType::DEM_ERROR) {
        result.append_dem_instruction(inst);
      } else {
        // Save data related to instruction and copy over while replacing
        // specific observable with a detector.
        std::vector<stim::DemTarget> target_data;
        for (auto tar: inst.target_data) {
          target_data.emplace_back(tar);
        }
        for (auto& tar: target_data) {
          if (tar.is_observable_id() && tar.val() == obs_idx) {
            tar.data = num_detectors;
          }
        }
        result.append_error_instruction(inst.arg_data[0], target_data, inst.tag);
      }
    }
    return result;
}

template <typename InnerDecoder>
ConfidencesDecoder<InnerDecoder>::ConfidencesDecoder(const stim::DetectorErrorModel& dem) {
  for (size_t obs_idx = 0; obs_idx < dem.count_observables(); obs_idx++) {
    obs_fixed_decoders.emplace_back(fix_obs(dem, obs_idx));
  }
  num_detectors = dem.count_detectors();
}

template <typename InnerDecoder>
std::vector<std::vector<double>> ConfidencesDecoder<InnerDecoder>::decode_to_confidences(
      const std::vector<stim::SparseShot>& shots) {

  std::vector<std::vector<double>> results;
  results.reserve(shots.size());
  size_t num_obs = obs_fixed_decoders.size();
  for (const auto& shot: shots) {
    std::vector<double> weight_diffs;
    weight_diffs.reserve(num_obs);

    stim::SparseShot shot_with_obs_flip = shot;
    shot_with_obs_flip.hits.emplace_back(num_detectors);

    for (size_t obs_idx = 0; obs_idx < num_obs; obs_idx++) {

      // Decode without and with the observable detector activated.
      double weight_no_flip = obs_fixed_decoders[obs_idx].decode_to_weight(shot);
      double weight_flip = obs_fixed_decoders[obs_idx].decode_to_weight(shot_with_obs_flip);

      // Weights are assumed to be negative LLRs. Return the weight difference
      // with flipping (obs mask 1) as positive.
      double diff = weight_flip - weight_no_flip;
      weight_diffs.emplace_back(diff);
    }
    results.emplace_back(weight_diffs);
  }

  return results;
}
