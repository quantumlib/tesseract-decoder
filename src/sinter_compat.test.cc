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

#include "sinter_compat.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <stdexcept>

namespace tesseract_decoder {
namespace {

TEST(SinterCompat, PacksPredictionsAndDiscard) {
  DecodeResult decoded{
      .predictions = {0, 9},
      .low_confidence = true,
  };
  std::array<uint8_t, 3> output{};

  pack_sinter_decode_result(decoded, 10, SinterOutputFormat::PredictionsAndDiscard, output);

  EXPECT_EQ(output, (std::array<uint8_t, 3>{0b00000001, 0b00000010, 1}));
}

TEST(SinterCompat, RejectsNegativePredictedObservable) {
  DecodeResult decoded{.predictions = {-1}};
  std::array<uint8_t, 1> output{};

  EXPECT_THROW(pack_sinter_decode_result(decoded, 1, SinterOutputFormat::Predictions, output),
               std::invalid_argument);
}

TEST(SinterCompat, RejectsOutOfRangePredictedObservable) {
  DecodeResult decoded{.predictions = {1}};
  std::array<uint8_t, 1> output{};

  EXPECT_THROW(pack_sinter_decode_result(decoded, 1, SinterOutputFormat::Predictions, output),
               std::invalid_argument);
}

}  // namespace
}  // namespace tesseract_decoder
