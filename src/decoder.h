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
#include <vector>

namespace tesseract_decoder {

struct DecoderResult {
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

class Decoder {
 public:
  virtual ~Decoder() = default;
  virtual DecoderResult decode_result(const std::vector<uint64_t>& detections) = 0;
};

}  // namespace tesseract_decoder

#endif  // TESSERACT_DECODER_DECODER_H
