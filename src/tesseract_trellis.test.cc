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

#include "tesseract_trellis.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>

#include "stim.h"

TEST(TesseractTrellisDecoderTest, ComputesObservableProbabilityForAmbiguousSyndrome) {
  stim::DetectorErrorModel dem(R"DEM(
    error(0.1) D0
    error(0.2) D0 L0
    detector(0, 0, 0) D0
  )DEM");

  TesseractTrellisConfig config;
  config.dem = dem;
  config.beam_width = 16;
  TesseractTrellisDecoder decoder(config);

  decoder.decode_shot({0});
  EXPECT_FALSE(decoder.low_confidence_flag);
  EXPECT_EQ(decoder.predicted_obs_mask, 1);
  EXPECT_NEAR(decoder.observable_probability(), 0.18 / 0.26, 1e-12);

  decoder.decode_shot({});
  EXPECT_FALSE(decoder.low_confidence_flag);
  EXPECT_EQ(decoder.predicted_obs_mask, 0);
  EXPECT_NEAR(decoder.observable_probability(), 0.02 / 0.74, 1e-12);
}

TEST(TesseractTrellisDecoderTest, SumsProbabilityMassAcrossMultipleExplanations) {
  // A min-cost decoder would prefer the single right branch over any one left
  // chain. The trellis should sum the three distinct left chains' probability
  // mass and predict L0. These chains are not mergeable because they touch
  // different hidden detectors.
  //
  //   L0 chains                         no-L0 chain
  //   D0 --a1(L0)-- D1 --b1-- *         D0 --r-- *
  //   D0 --a2(L0)-- D2 --b2-- *
  //   D0 --a3(L0)-- D3 --b3-- *
  stim::DetectorErrorModel dem(R"DEM(
    error(0.2) D0 D1 L0
    error(0.2) D1
    error(0.2) D0 D2 L0
    error(0.2) D2
    error(0.2) D0 D3 L0
    error(0.2) D3
    error(0.1) D0
    detector(0, 0, 0) D0
    detector(1, 0, 0) D1
    detector(2, 0, 0) D2
    detector(3, 0, 0) D3
  )DEM");

  TesseractTrellisConfig config;
  config.dem = dem;
  config.beam_width = 64;
  TesseractTrellisDecoder decoder(config);

  decoder.decode_shot({0});

  const double p_chain_edge = 0.2;
  const double p_right = 0.1;
  const double q_chain_edge = 1 - p_chain_edge;
  const double chain_present = p_chain_edge * p_chain_edge;
  const double chain_absent = q_chain_edge * q_chain_edge;
  const double odd_chain_mass = 3 * chain_present * chain_absent * chain_absent +
                                chain_present * chain_present * chain_present;
  const double even_chain_mass =
      chain_absent * chain_absent * chain_absent + 3 * chain_present * chain_present * chain_absent;
  const double mass_l0 = odd_chain_mass * (1 - p_right);
  const double mass_no_l0 = even_chain_mass * p_right;
  const double expected_probability = mass_l0 / (mass_l0 + mass_no_l0);
  const double right_cost = -std::log(p_right / (1 - p_right));
  const double left_chain_cost = 2 * -std::log(p_chain_edge / (1 - p_chain_edge));

  EXPECT_FALSE(decoder.low_confidence_flag);
  EXPECT_EQ(decoder.predicted_obs_mask, 1);
  EXPECT_TRUE(decoder.config.merge_errors);
  EXPECT_LT(right_cost, left_chain_cost);
  EXPECT_GT(expected_probability, 0.5);
  EXPECT_NEAR(decoder.observable_probability(), expected_probability, 1e-12);
}

TEST(TesseractTrellisDecoderTest, ReportsNanProbabilityForInvalidDetector) {
  stim::DetectorErrorModel dem(R"DEM(
    error(0.1) D0 L0
    detector(0, 0, 0) D0
  )DEM");

  TesseractTrellisConfig config;
  config.dem = dem;
  TesseractTrellisDecoder decoder(config);

  decoder.decode_shot({1});
  EXPECT_TRUE(decoder.low_confidence_flag);
  EXPECT_TRUE(std::isnan(decoder.observable_probability()));
}

TEST(TesseractTrellisDecoderTest, RankingModesDecodeAmbiguousSyndrome) {
  stim::DetectorErrorModel dem(R"DEM(
    error(0.1) D0
    error(0.2) D0 L0
    detector(0, 0, 0) D0
  )DEM");

  for (auto ranking_mode :
       {TesseractTrellisRankingMode::MassOnly, TesseractTrellisRankingMode::FutureDetcostRanked,
        TesseractTrellisRankingMode::FutureActiveDetcostRanked}) {
    TesseractTrellisConfig config;
    config.dem = dem;
    config.beam_width = 16;
    config.ranking_mode = ranking_mode;
    TesseractTrellisDecoder decoder(config);

    decoder.decode_shot({0});
    EXPECT_FALSE(decoder.low_confidence_flag);
    EXPECT_EQ(decoder.predicted_obs_mask, 1);
    EXPECT_NEAR(decoder.observable_probability(), 0.18 / 0.26, 1e-12);
  }
}

TEST(TesseractTrellisDecoderTest, BeamEpsSmokeTest) {
  stim::DetectorErrorModel dem(R"DEM(
    error(0.1) D0
    error(0.2) D0 L0
    detector(0, 0, 0) D0
  )DEM");

  TesseractTrellisConfig config;
  config.dem = dem;
  config.beam_width = 16;
  config.beam_eps = 0.1;
  TesseractTrellisDecoder decoder(config);

  decoder.decode_shot({0});
  EXPECT_FALSE(decoder.low_confidence_flag);
  EXPECT_TRUE(std::isfinite(decoder.observable_probability()));
}

TEST(TesseractTrellisDecoderTest, MergeErrorsMatchesOtherDecoders) {
  stim::DetectorErrorModel dem(R"DEM(
    error(0.1) D0 L0
    error(0.2) D0 L0
    error(0) D0
    detector(0, 0, 0) D0
  )DEM");

  TesseractTrellisConfig merged_config;
  merged_config.dem = dem;
  merged_config.beam_width = 16;
  TesseractTrellisDecoder merged_decoder(merged_config);

  EXPECT_TRUE(merged_decoder.config.merge_errors);
  EXPECT_EQ(merged_decoder.errors.size(), 1);
  EXPECT_EQ(merged_decoder.config.dem.count_errors(), 1);
  EXPECT_EQ(merged_decoder.dem_error_to_error.size(), 3);
  EXPECT_EQ(merged_decoder.dem_error_to_error[0], 0);
  EXPECT_EQ(merged_decoder.dem_error_to_error[1], 0);
  EXPECT_EQ(merged_decoder.dem_error_to_error[2], std::numeric_limits<size_t>::max());
  EXPECT_EQ(merged_decoder.error_to_dem_error.size(), 1);
  EXPECT_EQ(merged_decoder.error_to_dem_error[0], 0);

  merged_decoder.decode_shot({0});
  EXPECT_FALSE(merged_decoder.low_confidence_flag);
  EXPECT_EQ(merged_decoder.predicted_obs_mask, 1);
  EXPECT_NEAR(merged_decoder.observable_probability(), 1.0, 1e-12);

  TesseractTrellisConfig unmerged_config;
  unmerged_config.dem = dem;
  unmerged_config.beam_width = 16;
  unmerged_config.merge_errors = false;
  TesseractTrellisDecoder unmerged_decoder(unmerged_config);

  EXPECT_FALSE(unmerged_decoder.config.merge_errors);
  EXPECT_EQ(unmerged_decoder.errors.size(), 2);
  EXPECT_EQ(unmerged_decoder.config.dem.count_errors(), 2);
  EXPECT_EQ(unmerged_decoder.dem_error_to_error.size(), 3);
  EXPECT_EQ(unmerged_decoder.dem_error_to_error[0], 0);
  EXPECT_EQ(unmerged_decoder.dem_error_to_error[1], 1);
  EXPECT_EQ(unmerged_decoder.dem_error_to_error[2], std::numeric_limits<size_t>::max());

  unmerged_decoder.decode_shot({0});
  EXPECT_FALSE(unmerged_decoder.low_confidence_flag);
  EXPECT_EQ(unmerged_decoder.predicted_obs_mask, 1);
  EXPECT_NEAR(unmerged_decoder.observable_probability(), merged_decoder.observable_probability(),
              1e-12);
}

TEST(TesseractTrellisDecoderTest, RejectsMoreThanOneObservable) {
  stim::DetectorErrorModel dem(R"DEM(
    error(0.1) D0 L0
    error(0.1) D0 L1
    detector(0, 0, 0) D0
  )DEM");

  TesseractTrellisConfig config;
  config.dem = dem;

  try {
    TesseractTrellisDecoder decoder(config);
    FAIL() << "Expected TesseractTrellisDecoder construction to fail.";
  } catch (const std::invalid_argument& err) {
    EXPECT_NE(std::string(err.what()).find("supports at most one observable"), std::string::npos);
  }
}
