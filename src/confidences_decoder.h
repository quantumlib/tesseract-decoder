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

#include "multithread.h"
#include "stim.h"

namespace confidences_decoder {

// InnerDecoder is required to
//  (1) be constructable using a DEM, and
//  (2) have a `double decode_to_weight(stim::SparseShot shot)` function
//      that returns the weight (negative log likelihood ratio) of the
//      decoder solution.
template <typename InnerDecoder>
struct ConfidencesDecoder {

  ConfidencesDecoder(stim::DetectorErrorModel& dem);

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
      std::vector<stim::SparseShot>& shots);
};

} // namespace confidences_decoder

#endif  // CONFIDENCES_DECODER_H
