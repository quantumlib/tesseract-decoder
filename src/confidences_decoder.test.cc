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

static stim::simd_bits<64> obs_mask(uint64_t v) {
    stim::simd_bits<64> result(64);
    result.ptr_simd[0] = v;
    return result;
}

//TEST(confidences_decoder, ConfidenceExamples) {
//  // Test a few confidences
//  stim::DetectorErrorModel dem("error(0.1) D0 L0"
//                               "error(0.2) D0 D1"
//                               "error(0.3) D1 L1");
//
//  std::vector<stim::SparseShot> shots;
//  shots.emplace_back(stim::SparseShot{{0}, obs_mask(0)});
//  shots.emplace_back(stim::SparseShot{{1}, obs_mask(1)});
//
//  confidences_decoder::ConfidencesDecoder<TesseractInnerDecoder> decoder(dem);
//  std::vector<std::vector<double>> confidences = decoder.decode_to_confidences(shots);
//
//  // Magic numbers for shot weights (truncated). Computed by hand.
//  std::vector<double> weight_differences = {-0.0159, 1.1883};
//
//  EXPECT_TRUE(abs(confidences[0][0] - weight_differences[0]) < 0.001);
//  EXPECT_TRUE(abs(confidences[0][1] + weight_differences[0]) < 0.001);
//  EXPECT_TRUE(abs(confidences[1][0] - weight_differences[1]) < 0.001);
//  EXPECT_TRUE(abs(confidences[1][1] + weight_differences[1]) < 0.001);
//}
