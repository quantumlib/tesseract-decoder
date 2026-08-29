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

#include "dem_decomposition.h"

#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace tesseract_decoder {
namespace {

TEST(DemDecompositionTest, PreservesExistingGroupsAndTags) {
  stim::DetectorErrorModel dem(R"DEM(
        error[physical mechanism](0.1) D0 L0 ^ D1 L1
        detector[x detector](1, 2) D0
        detector[z detector](3, 4) D1
        logical_observable L0
        logical_observable L1
    )DEM");

  EXPECT_EQ(decompose_errors_using_detector_assignment(dem, {0, 1}).str(), dem.str());
}

TEST(DemDecompositionTest, DecomposesMixedErrorAndPreservesTag) {
  stim::DetectorErrorModel dem(R"DEM(
        error[x evidence](0.01) D0 L0
        error[z evidence](0.02) D1 L1
        error[correlated](0.1) D0 D1 L0 L1
        detector D0
        detector D1
        logical_observable L0
        logical_observable L1
    )DEM");
  stim::DetectorErrorModel expected(R"DEM(
        error[x evidence](0.01) D0 L0
        error[z evidence](0.02) D1 L1
        error[correlated](0.1) D0 L0 ^ D1 L1
        detector D0
        detector D1
        logical_observable L0
        logical_observable L1
    )DEM");

  EXPECT_EQ(decompose_errors_using_detector_assignment(dem, {0, 1}).str(), expected.str());
}

TEST(DemDecompositionTest, RejectsAmbiguousMissingComponents) {
  stim::DetectorErrorModel dem(R"DEM(
        error[ambiguous](0.1) D0 D1 L0
        detector D0
        detector D1
        logical_observable L0
    )DEM");
  EXPECT_THROW(decompose_errors_using_detector_assignment(dem, {0, 1}), std::invalid_argument);
}

TEST(DemDecompositionTest, RejectsMixedAndDetectorlessExistingGroups) {
  stim::DetectorErrorModel mixed(R"DEM(
        error[mixed group](0.1) D0 D1 ^ D0
        detector D0
        detector D1
    )DEM");
  EXPECT_THROW(decompose_errors_using_detector_assignment(mixed, {0, 1}), std::invalid_argument);

  stim::DetectorErrorModel logical_only_group(R"DEM(
        error[logical only group](0.1) L0 ^ D0
        detector D0
        detector D1
        logical_observable L0
    )DEM");
  EXPECT_THROW(decompose_errors_using_detector_assignment(logical_only_group, {0, 1}),
               std::invalid_argument);

  stim::DetectorErrorModel logical_only_error(R"DEM(
        error[logical only](0.1) L0
        detector D0
        detector D1
        logical_observable L0
    )DEM");
  EXPECT_THROW(decompose_errors_using_detector_assignment(logical_only_error, {0, 1}),
               std::invalid_argument);
}

TEST(DemDecompositionTest, SplitKeepsSameComponentGroupsInOneTaggedMechanism) {
  stim::DetectorErrorModel dem(R"DEM(
        error[physical mechanism](0.1) D0 L0 ^ D1 L1 ^ D2
        detector[x0](1) D0
        detector[x1](2) D1
        detector[z](3) D2
        logical_observable L0
        logical_observable L1
    )DEM");

  auto components = split_dem_by_component(dem, {7, 7, 9});
  ASSERT_EQ(components.size(), 2);
  EXPECT_EQ(components.at(7).count_errors(), 1);
  EXPECT_EQ(components.at(9).count_errors(), 1);
  EXPECT_NE(components.at(7).str().find("error[physical mechanism](0.1) D0 L0 ^ D1 L1"),
            std::string::npos);
  EXPECT_NE(components.at(9).str().find("error[physical mechanism](0.1) D2"), std::string::npos);
  EXPECT_NE(components.at(7).str().find("detector[x0](1) D0"), std::string::npos);
  EXPECT_NE(components.at(9).str().find("logical_observable L1"), std::string::npos);
}

TEST(DemDecompositionTest, SplitRejectsCancellationToDetectorlessComponent) {
  stim::DetectorErrorModel dem(R"DEM(
        error[cancels](0.1) D0 ^ D0 ^ D1
        detector D0
        detector D1
    )DEM");
  EXPECT_THROW(split_dem_by_component(dem, {0, 1}), std::invalid_argument);
}

TEST(DemDecompositionTest, ValidatesExplicitAssignments) {
  stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0
        detector D0
        detector D1
    )DEM");
  EXPECT_THROW(decompose_errors_using_detector_assignment(dem, {0}), std::invalid_argument);
  EXPECT_THROW(decompose_errors_using_detector_assignment(dem, {0, -1}), std::invalid_argument);
}

}  // namespace
}  // namespace tesseract_decoder
