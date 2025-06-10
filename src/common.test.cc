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

#include "common.h"

#include "gtest/gtest.h"
#include "stim.h"

TEST(common, ErrorsStructFromDemInstruction) {
  // Test a pathological DEM error instruction
  stim::DetectorErrorModel dem("error(0.1) D0 ^  D0 D1  L0 L1 L1");
  stim::DemInstruction instruction = dem.instructions.at(0);
  common::Error ES(instruction);
  EXPECT_EQ(ES.symptom.detectors, std::vector<int>{1});
  EXPECT_EQ(ES.symptom.observables, 0b01);
}

TEST(common, DemFromCountsRejectsZeroProbabilityErrors) {
  stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0
        error(0) D1
        error(0.2) D2
        detector(0, 0, 0) D0
        detector(0, 0, 0) D1
        detector(0, 0, 0) D2
      )DEM");

  std::vector<size_t> counts{1, 7, 4};
  size_t num_shots = 10;
  EXPECT_THROW({
    common::dem_from_counts(dem, counts, num_shots);
  }, std::invalid_argument);

  stim::DetectorErrorModel cleaned = common::remove_zero_probability_errors(dem);
  stim::DetectorErrorModel out_dem =
      common::dem_from_counts(cleaned, std::vector<size_t>{1, 4}, num_shots);

  auto flat = out_dem.flattened();
  ASSERT_EQ(out_dem.count_errors(), 2);
  ASSERT_GE(flat.instructions.size(), 2);

  EXPECT_EQ(flat.instructions[0].type,
            stim::DemInstructionType::DEM_ERROR);
  EXPECT_NEAR(flat.instructions[0].arg_data[0], 0.1, 1e-9);
  ASSERT_EQ(flat.instructions[1].type,
            stim::DemInstructionType::DEM_ERROR);
  EXPECT_NEAR(flat.instructions[1].arg_data[0], 0.4, 1e-9);
}

TEST(common, DemFromCountsSimpleTwoErrors) {
  stim::DetectorErrorModel dem(R"DEM(
        error(0.25) D0
        error(0.35) D1
        detector(0, 0, 0) D0
        detector(0, 0, 0) D1
      )DEM");

  std::vector<size_t> counts{5, 7};
  size_t num_shots = 20;
  stim::DetectorErrorModel out_dem =
      common::dem_from_counts(dem, counts, num_shots);

  auto flat = out_dem.flattened();
  ASSERT_EQ(out_dem.count_errors(), 2);

  ASSERT_GE(flat.instructions.size(), 2);
  EXPECT_EQ(flat.instructions[0].type,
            stim::DemInstructionType::DEM_ERROR);
  EXPECT_NEAR(flat.instructions[0].arg_data[0], 0.25, 1e-9);
  EXPECT_EQ(flat.instructions[1].type,
            stim::DemInstructionType::DEM_ERROR);
  EXPECT_NEAR(flat.instructions[1].arg_data[0], 0.35, 1e-9);
}

TEST(common, RemoveZeroProbabilityErrors) {
  stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0
        error(0) D1
        error(0.2) D2
        detector(0, 0, 0) D0
        detector(0, 0, 0) D1
        detector(0, 0, 0) D2
      )DEM");

  stim::DetectorErrorModel cleaned =
      common::remove_zero_probability_errors(dem);

  EXPECT_EQ(cleaned.count_errors(), 2);
  auto flat = cleaned.flattened();
  ASSERT_EQ(flat.instructions[0].type,
            stim::DemInstructionType::DEM_ERROR);
  EXPECT_NEAR(flat.instructions[0].arg_data[0], 0.1, 1e-9);
  ASSERT_EQ(flat.instructions[1].type,
            stim::DemInstructionType::DEM_ERROR);
  EXPECT_NEAR(flat.instructions[1].arg_data[0], 0.2, 1e-9);
}
