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
  EXPECT_EQ(ES.symptom.observables, std::vector<int>{0});
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
  EXPECT_THROW({ common::dem_from_counts(dem, counts, num_shots); }, std::invalid_argument);

  stim::DetectorErrorModel cleaned = common::remove_zero_probability_errors(dem);
  stim::DetectorErrorModel out_dem =
      common::dem_from_counts(cleaned, std::vector<size_t>{1, 4}, num_shots);

  auto flat = out_dem.flattened();
  ASSERT_EQ(out_dem.count_errors(), 2);
  ASSERT_GE(flat.instructions.size(), 2);

  EXPECT_EQ(flat.instructions[0].type, stim::DemInstructionType::DEM_ERROR);
  EXPECT_NEAR(flat.instructions[0].arg_data[0], 0.1, 1e-9);
  ASSERT_EQ(flat.instructions[1].type, stim::DemInstructionType::DEM_ERROR);
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
  stim::DetectorErrorModel out_dem = common::dem_from_counts(dem, counts, num_shots);

  auto flat = out_dem.flattened();
  ASSERT_EQ(out_dem.count_errors(), 2);

  ASSERT_GE(flat.instructions.size(), 2);
  EXPECT_EQ(flat.instructions[0].type, stim::DemInstructionType::DEM_ERROR);
  EXPECT_NEAR(flat.instructions[0].arg_data[0], 0.25, 1e-9);
  EXPECT_EQ(flat.instructions[1].type, stim::DemInstructionType::DEM_ERROR);
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

  stim::DetectorErrorModel cleaned = common::remove_zero_probability_errors(dem);

  EXPECT_EQ(cleaned.count_errors(), 2);
  auto flat = cleaned.flattened();
  ASSERT_EQ(flat.instructions[0].type, stim::DemInstructionType::DEM_ERROR);
  EXPECT_NEAR(flat.instructions[0].arg_data[0], 0.1, 1e-9);
  ASSERT_EQ(flat.instructions[1].type, stim::DemInstructionType::DEM_ERROR);
  EXPECT_NEAR(flat.instructions[1].arg_data[0], 0.2, 1e-9);
}

// Helper function to compare the two methods.
void assert_merged_probabilities_are_equal(double p1, double p2) {
  // Merge probabilities using the exclusive OR formula.
  double merged_p_direct = p1 + p2 - 2 * p1 * p2;

  // Convert to likelihood costs, merge, and then convert back.
  double cost1 = std::log(p1 / (1 - p1));
  double cost2 = std::log(p2 / (1 - p2));
  double merged_cost = common::merge_weights(cost1, cost2);
  double merged_p_via_costs = 1 / (1 + std::exp(merged_cost));

  // The two methods should produce nearly same results.
  ASSERT_NEAR(merged_p_direct, merged_p_via_costs, 1e-12);
}

TEST(CommonTest, merge_weights_is_equivalent_to_probability_xor) {
  // Test with small probabilities
  assert_merged_probabilities_are_equal(0.001, 0.002);

  // Test with larger probabilities
  assert_merged_probabilities_are_equal(0.1, 0.25);

  // Test with a mix of small and large probabilities
  assert_merged_probabilities_are_equal(0.05, 0.8);

  // Test with a probability close to 0.5, where the formula is sensitive
  assert_merged_probabilities_are_equal(0.49, 0.51);

  // Test with identical probabilities
  assert_merged_probabilities_are_equal(0.01, 0.01);
}

// Helper function to create a simple DEM with two identical errors.
stim::DetectorErrorModel create_dem_with_two_errors(double p1, double p2) {
  stim::DetectorErrorModel dem;
  std::vector<stim::DemTarget> targets = {stim::DemTarget::relative_detector_id(0)};

  dem.append_error_instruction(p1, targets, "");
  dem.append_error_instruction(p2, targets, "");
  return dem;
}

// Function to get the probability from a merged DEM.
double get_merged_probability(const stim::DetectorErrorModel& merged_dem) {
  if (merged_dem.instructions.size() != 1 ||
      merged_dem.instructions[0].type != stim::DemInstructionType::DEM_ERROR) {
    throw std::runtime_error("Expected a single DEM_ERROR instruction in the merged model.");
  }
  return merged_dem.instructions[0].arg_data[0];
}

TEST(CommonTest, merge_indistinguishable_errors_two_errors) {
  // Case 1: Both probabilities are low.
  double p1 = 0.1;
  double p2 = 0.2;
  double expected_merged_p = p1 * (1 - p2) + p2 * (1 - p1);
  auto dem1 = create_dem_with_two_errors(p1, p2);
  auto merged_dem1 = common::merge_indistinguishable_errors(dem1);
  ASSERT_NEAR(get_merged_probability(merged_dem1), expected_merged_p, 1e-9);

  // Case 2: One low, one high probability.
  p1 = 0.1;
  p2 = 0.8;
  expected_merged_p = p1 * (1 - p2) + p2 * (1 - p1);
  auto dem2 = create_dem_with_two_errors(p1, p2);
  auto merged_dem2 = common::merge_indistinguishable_errors(dem2);
  ASSERT_NEAR(get_merged_probability(merged_dem2), expected_merged_p, 1e-9);

  // Case 3: One high, one low probability.
  p1 = 0.8;
  p2 = 0.1;
  expected_merged_p = p1 * (1 - p2) + p2 * (1 - p1);
  auto dem3 = create_dem_with_two_errors(p1, p2);
  auto merged_dem3 = common::merge_indistinguishable_errors(dem3);
  ASSERT_NEAR(get_merged_probability(merged_dem3), expected_merged_p, 1e-9);

  // Case 4: Both probabilities are high.
  p1 = 0.8;
  p2 = 0.9;
  expected_merged_p = p1 * (1 - p2) + p2 * (1 - p1);
  auto dem4 = create_dem_with_two_errors(p1, p2);
  auto merged_dem4 = common::merge_indistinguishable_errors(dem4);
  ASSERT_NEAR(get_merged_probability(merged_dem4), expected_merged_p, 1e-9);
}
