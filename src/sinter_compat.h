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

#ifndef TESSERACT_DECODER_SINTER_COMPAT_H
#define TESSERACT_DECODER_SINTER_COMPAT_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

#include "decoder.h"

namespace tesseract_decoder {

enum class SinterOutputFormat {
  Predictions,
  PredictionsAndDiscard,
};

inline size_t sinter_output_bytes(uint64_t num_observables, SinterOutputFormat format) {
  return (num_observables + 7) / 8 +
         static_cast<size_t>(format == SinterOutputFormat::PredictionsAndDiscard);
}

inline void pack_sinter_decode_result(const DecodeResult& decoded, uint64_t num_observables,
                                      SinterOutputFormat format, std::span<uint8_t> output) {
  const size_t num_observable_bytes = (num_observables + 7) / 8;
  if (output.size() != sinter_output_bytes(num_observables, format)) {
    throw std::invalid_argument("Sinter output buffer has the wrong size.");
  }

  std::fill(output.begin(), output.end(), 0);
  validate_observable_predictions(decoded.predictions, num_observables);
  for (int observable : decoded.predictions) {
    output[observable / 8] ^= static_cast<uint8_t>(1U << (observable % 8));
  }
  if (format == SinterOutputFormat::PredictionsAndDiscard) {
    output[num_observable_bytes] = decoded.low_confidence;
  }
}

}  // namespace tesseract_decoder

#endif  // TESSERACT_DECODER_SINTER_COMPAT_H
