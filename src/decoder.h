// Copyright 2026 Google LLC
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

#ifndef TESSERACT_DECODER_DECODER_H
#define TESSERACT_DECODER_DECODER_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace tesseract_decoder {

struct DecodeResult {
  // Sorted observable IDs flipped by the decoder.
  std::vector<int> predictions;

  // Indices into the original flattened DEM. A populated result may be empty.
  std::vector<size_t> predicted_errors;
  bool predicted_errors_populated = false;

  // Low-confidence results should be discarded instead of counted as successes.
  bool low_confidence = false;

  // Cost of the final decoding decision.
  double total_cost = 0;
};

inline void validate_observable_predictions(const std::vector<int>& predictions,
                                            uint64_t num_observables) {
  for (int observable : predictions) {
    if (observable < 0 || static_cast<uint64_t>(observable) >= num_observables) {
      throw std::invalid_argument("Decoder predicted observable " + std::to_string(observable) +
                                  " outside the DEM's observable range [0, " +
                                  std::to_string(num_observables) + ").");
    }
  }
}

class Decoder {
 public:
  virtual ~Decoder() = default;
  virtual DecodeResult decode_result(const std::vector<uint64_t>& detections) = 0;
};

}  // namespace tesseract_decoder

#endif  // TESSERACT_DECODER_DECODER_H
