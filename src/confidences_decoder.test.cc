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
#include "tesseract_inner_decoder.h"

#include "gtest/gtest.h"
#include "stim.h"

TEST(confidences_decoder, ConfidenceExamples) {
 // Test a few confidences
  stim::DetectorErrorModel dem("error(0.1) D0 L0\n"
                              "error(0.2) D0 D1\n"
                              "error(0.3) D1 L1");

  confidences_decoder::ConfidencesDecoder<TesseractInnerDecoder> decoder(dem);

  // Magic numbers for shot weights (truncated). Computed by hand.
  std::vector<double> confidences = decoder.decode_to_confidences({0});
  double weight_difference = -0.0364;
  EXPECT_TRUE(abs(confidences[0] - weight_difference) < 0.001);
  EXPECT_TRUE(abs(confidences[1] + weight_difference) < 0.001);

  confidences = decoder.decode_to_confidences({1});
  weight_difference = 2.7362;
  EXPECT_TRUE(abs(confidences[0] - weight_difference) < 0.001);
  EXPECT_TRUE(abs(confidences[1] + weight_difference) < 0.001);
}
