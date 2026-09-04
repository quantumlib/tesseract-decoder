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

#include "error_correlations.h"

#include <vector>

#include "gtest/gtest.h"

namespace tesseract_decoder {
namespace {

TEST(ErrorCorrelationsTest, DocumentsPairedMechanismRatioFormula) {
  stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0 ^ D1 L0
        error(0.2) D0
        error(0.05) D1 L1
        detector D0
        detector D1
    )DEM");
  ComponentSymptom d0{{0}, {}};
  ComponentSymptom d1_l0{{1}, {0}};

  CorrelationEvidence evidence = collect_correlation_evidence(dem, {0, 1});
  // XOR(0.1, 0.2) = 0.1 * 0.8 + 0.2 * 0.9 = 0.26.
  EXPECT_NEAR(evidence.symptom_probabilities.at(d0), 0.26, 1e-12);
  // Only the shared 0.1 mechanism is paired evidence. Independent one-sided
  // combinations are deliberately not added as if this were an exact joint.
  EXPECT_NEAR(evidence.paired_mechanism_probabilities.at(d0).at(d1_l0), 0.1, 1e-12);

  ReweightProbsMap probabilities = derive_reweight_probabilities(evidence);
  ASSERT_EQ(probabilities.at(d0).size(), 1);
  EXPECT_EQ(probabilities.at(d0)[0].affected_symptom.detectors, d1_l0.detectors);
  EXPECT_EQ(probabilities.at(d0)[0].affected_symptom.observables, d1_l0.observables);
  EXPECT_NEAR(probabilities.at(d0)[0].probability, 0.1 / 0.26, 1e-12);
}

TEST(ErrorCorrelationsTest, OneSidedMechanismsDoNotCreatePairedEvidence) {
  stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0
        error(0.2) D1
        detector D0
        detector D1
    )DEM");
  CorrelationEvidence evidence = collect_correlation_evidence(dem, {0, 1});
  EXPECT_TRUE(evidence.paired_mechanism_probabilities.empty());
  EXPECT_TRUE(derive_reweight_probabilities(evidence).empty());
}

TEST(ErrorCorrelationsTest, CombinesSameComponentGroupsBeforeRecordingEvidence) {
  stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0 L0 ^ D1 L1 ^ D2
        detector D0
        detector D1
        detector D2
    )DEM");
  ComponentSymptom combined{{0, 1}, {0, 1}};
  ComponentSymptom affected{{2}, {}};

  CorrelationEvidence evidence = collect_correlation_evidence(dem, {0, 0, 1});
  EXPECT_NEAR(evidence.symptom_probabilities.at(combined), 0.1, 1e-12);
  EXPECT_NEAR(evidence.paired_mechanism_probabilities.at(combined).at(affected), 0.1, 1e-12);
}

TEST(ErrorCorrelationsTest, RejectsAmbiguousGroupsAndAssignments) {
  stim::DetectorErrorModel mixed(R"DEM(
        error(0.1) D0 D1
        detector D0
        detector D1
    )DEM");
  EXPECT_THROW(collect_correlation_evidence(mixed, {0, 1}), std::invalid_argument);

  stim::DetectorErrorModel logical_only(R"DEM(
        error(0.1) L0 ^ D0
        detector D0
        detector D1
        logical_observable L0
    )DEM");
  EXPECT_THROW(collect_correlation_evidence(logical_only, {0, 1}), std::invalid_argument);
  EXPECT_THROW(collect_correlation_evidence(mixed, {0}), std::invalid_argument);
  EXPECT_THROW(collect_correlation_evidence(mixed, {0, -1}), std::invalid_argument);
}

}  // namespace
}  // namespace tesseract_decoder
