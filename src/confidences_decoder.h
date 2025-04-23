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

#ifndef CONFIDENCES_DECODER_H
#define CONFIDENCES_DECODER_H

#include "stim.h"

namespace confidences_decoder {

// Helper function declaration
stim::DetectorErrorModel fix_obs(const stim::DetectorErrorModel& dem,
                                    uint64_t obs_idx);

// InnerDecoder is required to
//  (1) be constructable using a DEM, and
//  (2) have a `double decode_to_weight(stim::SparseShot shot)` function
//      that returns the weight (negative log likelihood ratio) of the
//      decoder solution.
template <typename InnerDecoder>
struct ConfidencesDecoder {

  ConfidencesDecoder(const stim::DetectorErrorModel& dem);

  // Inner decoders where the ith decoder treats the ith observable to
  // as the final detector.
  std::vector<InnerDecoder> obs_fixed_decoders;

  uint64_t num_detectors;

  // Uses the templated inner decoders to return weight difference (difference
  // of LLRS) of each observable flipping in each shot. So a probability of 0
  // (weight difference of \infty) corresponds to a decoder that is sure that
  // the observable shouldn't be flipped while a probability of 1 (weight
  // difference of -\infty) corresponds to a decoder that is sure that the
  // observable should be flipped. The outer vector runs over shots while the
  // inner vector runs over logical observables. To convert to a probability
  // of flip p, we would compute p = 1 / (1 + exp(weight_difference)).
  std::vector<std::vector<double>> decode_to_confidences(
      const std::vector<stim::SparseShot>& shots);
};


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

} // namespace confidences_decoder

#endif  // CONFIDENCES_DECODER_H
