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

#include "multi_pass_tesseract_decoder.h"

#include <cmath>
#include <vector>

#include "gtest/gtest.h"

namespace tesseract_decoder {

class MultiPassTestPeer {
 public:
  static const std::vector<std::vector<size_t>>& get_pass_schedule(
      const MultiPassTesseractDecoder& decoder) {
    return decoder.pass_schedule;
  }
  static const TesseractDecoder& get_component_decoder(const MultiPassTesseractDecoder& decoder,
                                                       size_t component) {
    return *decoder.component_decoders.at(component).decoder;
  }
  static TesseractDecoder& get_component_decoder(MultiPassTesseractDecoder& decoder,
                                                 size_t component) {
    return *decoder.component_decoders.at(component).decoder;
  }
  static const std::vector<double>& get_original_costs(const MultiPassTesseractDecoder& decoder,
                                                       size_t component) {
    return decoder.component_decoders.at(component).original_costs;
  }
};

namespace {

stim::DetectorErrorModel correlated_dem() {
  return stim::DetectorErrorModel(R"DEM(
        error[bridge](0.1) D0 ^ D1 L0
        error[source only](0.01) D0
        error[target only](0.2) D1 L0
        detector D0
        detector D1
        logical_observable L0
    )DEM");
}

void expect_costs_restored(const MultiPassTesseractDecoder& decoder) {
  for (size_t component = 0; component < decoder.num_components(); ++component) {
    const auto& component_decoder = MultiPassTestPeer::get_component_decoder(decoder, component);
    const auto& original_costs = MultiPassTestPeer::get_original_costs(decoder, component);
    ASSERT_EQ(component_decoder.errors.size(), original_costs.size());
    ASSERT_EQ(component_decoder.error_costs.size(), original_costs.size());
    for (size_t error = 0; error < original_costs.size(); ++error) {
      EXPECT_DOUBLE_EQ(component_decoder.errors[error].likelihood_cost, original_costs[error]);
      EXPECT_DOUBLE_EQ(component_decoder.error_costs[error].likelihood_cost, original_costs[error]);
      EXPECT_DOUBLE_EQ(
          component_decoder.error_costs[error].min_cost,
          original_costs[error] / component_decoder.errors[error].symptom.detectors.size());
    }
  }
}

TEST(MultiPassTesseractDecoderTest, ValidatesSupportedShapeAndStrategy) {
  stim::DetectorErrorModel dem = correlated_dem();
  for (size_t passes : {1, 2}) {
    EXPECT_NO_THROW(MultiPassTesseractDecoder(dem, passes, {4, 9}));
  }
  for (size_t passes : {0, 3}) {
    EXPECT_THROW(MultiPassTesseractDecoder(dem, passes, {4, 9}), std::invalid_argument);
  }
  EXPECT_THROW(MultiPassTesseractDecoder(dem, 1, {4}), std::invalid_argument);
  EXPECT_THROW(MultiPassTesseractDecoder(dem, 1, {4, -1}), std::invalid_argument);
  EXPECT_THROW(MultiPassTesseractDecoder(dem, 1, {4, 4}), std::invalid_argument);
  EXPECT_THROW(MultiPassTesseractDecoder(dem, 1, {4, 9}, TesseractConfig(),
                                         static_cast<SchedulingStrategy>(999)),
               std::invalid_argument);

  MultiPassTesseractDecoder decoder(dem, 1, {4, 9});
  EXPECT_THROW(decoder.decode({2}), std::invalid_argument);
}

TEST(MultiPassTesseractDecoderTest, SeparateConfigsConstructMatchingDecoders) {
  TesseractConfig component_config;
  component_config.dem = stim::DetectorErrorModel(R"DEM(
        error(0.1) D0
        error(0.2) D1 L0
        detector[{"basis":"X"}] D0
        detector[{"basis":"Z"}] D1
        logical_observable L0
    )DEM");

  std::unique_ptr<Decoder> monolithic_decoder =
      std::make_unique<TesseractDecoder>(component_config);
  EXPECT_NE(dynamic_cast<TesseractDecoder*>(monolithic_decoder.get()), nullptr);
  EXPECT_EQ(dynamic_cast<MultiPassTesseractDecoder*>(monolithic_decoder.get()), nullptr);

  MultiPassTesseractConfig config;
  config.component_config = component_config;
  config.num_passes = 1;
  std::unique_ptr<Decoder> multi_pass_decoder = std::make_unique<MultiPassTesseractDecoder>(config);
  EXPECT_NE(dynamic_cast<MultiPassTesseractDecoder*>(multi_pass_decoder.get()), nullptr);
  EXPECT_EQ(dynamic_cast<TesseractDecoder*>(multi_pass_decoder.get()), nullptr);
}

TEST(MultiPassTesseractDecoderTest, MultiPassConfigClassifiesCanonicalBasisTags) {
  MultiPassTesseractConfig config;
  config.component_config.dem = stim::DetectorErrorModel(R"DEM(
        error(0.1) D0
        error(0.2) D1 L0
        detector[{"measure_basis":"invalid","basis":"X","unrelated":5}] D0
        detector[{"basis":"Z"}] D1
        logical_observable L0
    )DEM");
  config.num_passes = 1;

  std::unique_ptr<Decoder> decoder = std::make_unique<MultiPassTesseractDecoder>(config);
  EXPECT_NE(dynamic_cast<MultiPassTesseractDecoder*>(decoder.get()), nullptr);
  DecodeResult result = decoder->decode_result({1});
  EXPECT_EQ(result.predictions, std::vector<int>({0}));
  EXPECT_FALSE(result.predicted_errors_populated);
}

TEST(MultiPassTesseractDecoderTest, MultiPassConfigRejectsNoncanonicalBasisMetadata) {
  const std::vector<std::string> detector_instructions = {
      R"DEM(detector[{"measure_basis":"X"}] D0
            detector[{"basis":"Z"}] D1)DEM",
      R"DEM(detector[{"md":{"basis":"X"}}] D0
            detector[{"basis":"Z"}] D1)DEM",
      R"DEM(detector(0, 0, 0, 0) D0
            detector[{"basis":"Z"}] D1)DEM",
      R"DEM(detector[{"basis":"Y"}] D0
            detector[{"basis":"Z"}] D1)DEM",
      R"DEM(detector[not-json] D0
            detector[{"basis":"Z"}] D1)DEM",
  };

  for (const std::string& detector_instruction : detector_instructions) {
    MultiPassTesseractConfig config;
    config.component_config.dem = stim::DetectorErrorModel(
        ("error(0.1) D0\nerror(0.2) D1 L0\n" + detector_instruction).c_str());
    config.num_passes = 1;
    EXPECT_THROW((void)MultiPassTesseractDecoder(config), std::invalid_argument);
  }
}

TEST(MultiPassTesseractDecoderTest, BuildsDetectorOrdersFromEachComponentDem) {
  stim::DetectorErrorModel dem(R"DEM(
        error(0.02) D0 D1 L0
        error(0.1) D3 D4 D5 D6 L1
        error(0.02) D3 D4 D6
        error(0.3) D4 D5
        detector D0
        detector D1
        detector D2
        detector D3
        detector D4
        detector D5
        detector D6
        logical_observable L0
        logical_observable L1
  )DEM");
  constexpr size_t num_det_orders = 1;
  constexpr uint64_t seed = 1;
  TesseractConfig config;
  config.detector_orders = make_detector_orders(num_det_orders, DetectorOrder::Method::BFS, seed);
  MultiPassTesseractDecoder decoder(dem, 1, {0, 0, 0, 1, 1, 1, 1}, config);
  auto monolithic_orders = build_det_orders(dem, num_det_orders, DetectorOrder::Method::BFS, seed);
  bool differs_from_monolithic = false;
  for (size_t component = 0; component < decoder.num_components(); ++component) {
    const TesseractDecoder& component_decoder =
        MultiPassTestPeer::get_component_decoder(decoder, component);
    auto component_orders = build_det_orders(component_decoder.config.dem, num_det_orders,
                                             DetectorOrder::Method::BFS, seed);
    ASSERT_EQ(component_decoder.config.detector_orders.size(), component_orders.size());
    for (size_t order = 0; order < component_orders.size(); ++order) {
      EXPECT_EQ(component_decoder.config.detector_orders[order].get_order(),
                component_orders[order]);
    }
    differs_from_monolithic |= component_orders != monolithic_orders;
  }
  EXPECT_TRUE(differs_from_monolithic);
}

TEST(MultiPassTesseractDecoderTest, HonorsMergeErrorsBothWaysInOnePass) {
  stim::DetectorErrorModel dem(R"DEM(
        error[a](0.1) D0 L0
        error[b](0.2) D0 L0
        error[c](0.1) D1 L1
        error[d](0.2) D1 L1
        detector D0
        detector D1
        logical_observable L0
        logical_observable L1
    )DEM");

  for (bool merge_errors : {false, true}) {
    TesseractConfig config;
    config.merge_errors = merge_errors;
    MultiPassTesseractDecoder decoder(dem, 1, {0, 1}, config);
    MultiPassExecutionPlan plan = decoder.get_execution_plan();
    EXPECT_EQ(plan.monolithic_statistics.error_mechanism_count, merge_errors ? 2 : 4);
    ASSERT_EQ(plan.components.size(), 2);
    for (size_t component = 0; component < 2; ++component) {
      EXPECT_EQ(plan.components[component].error_mechanism_count, merge_errors ? 1 : 2);
      EXPECT_EQ(MultiPassTestPeer::get_component_decoder(decoder, component).config.merge_errors,
                merge_errors);
    }
    EXPECT_EQ(decoder.decode({0}), std::vector<int>({0}));
    EXPECT_EQ(decoder.decode({1}), std::vector<int>({1}));
  }
}

TEST(MultiPassTesseractDecoderTest, RejectsUnmergedTwoPassReweighting) {
  TesseractConfig config;
  config.merge_errors = false;

  try {
    MultiPassTesseractDecoder decoder(correlated_dem(), 2, {0, 1}, config);
    FAIL() << "Expected two-pass decoding with unmerged errors to be rejected.";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("Two-pass decoding requires merge_errors=true"),
              std::string::npos);
  }
}

TEST(MultiPassTesseractDecoderTest, PreservesExplicitDetectorOrders) {
  TesseractConfig config;
  config.detector_orders = make_literal_detector_orders({{1, 0}, {0, 1}});
  MultiPassTesseractDecoder decoder(correlated_dem(), 2, {0, 1}, config);
  for (size_t component = 0; component < decoder.num_components(); ++component) {
    const auto& component_orders =
        MultiPassTestPeer::get_component_decoder(decoder, component).config.detector_orders;
    ASSERT_EQ(component_orders.size(), config.detector_orders.size());
    for (size_t order = 0; order < component_orders.size(); ++order) {
      EXPECT_EQ(component_orders[order].get_order(), config.detector_orders[order].get_order());
    }
  }

  config.detector_orders = make_literal_detector_orders({{0}});
  EXPECT_THROW(MultiPassTesseractDecoder(correlated_dem(), 2, {0, 1}, config),
               std::invalid_argument);
}

TEST(MultiPassTesseractDecoderTest, ResolvesMixedDetectorOrdersAgainstEachComponentDem) {
  constexpr uint64_t seed = 1234;
  TesseractConfig config;
  config.detector_orders.emplace_back(DetectorOrder::Method::BFS, seed);
  config.detector_orders.emplace_back(std::vector<size_t>{1, 0});
  config.detector_orders.emplace_back(DetectorOrder::Method::Coordinate, seed);

  MultiPassTesseractDecoder decoder(correlated_dem(), 2, {0, 1}, config);
  for (size_t component = 0; component < decoder.num_components(); ++component) {
    const TesseractDecoder& component_decoder =
        MultiPassTestPeer::get_component_decoder(decoder, component);
    const auto& orders = component_decoder.config.detector_orders;
    ASSERT_EQ(orders.size(), 3);
    EXPECT_EQ(orders[0].get_order(), build_det_orders(component_decoder.config.dem, 1,
                                                      DetectorOrder::Method::BFS, seed)[0]);
    EXPECT_EQ(orders[1].get_order(), (std::vector<size_t>{1, 0}));
    EXPECT_EQ(orders[2].get_order(), build_det_orders(component_decoder.config.dem, 1,
                                                      DetectorOrder::Method::Coordinate, seed)[0]);
  }
}

TEST(MultiPassTesseractDecoderTest, CausalReweightUsesSafeCapAndReportsFinalPassCost) {
  MultiPassTesseractDecoder decoder(correlated_dem(), 2, {0, 1});
  EXPECT_EQ(MultiPassTestPeer::get_pass_schedule(decoder),
            std::vector<std::vector<size_t>>({{0}, {1}}));

  Decoder& decoder_interface = decoder;
  DecodeResult result = decoder_interface.decode_result({0, 1});
  EXPECT_EQ(result.predictions, std::vector<int>({0}));
  EXPECT_TRUE(result.predicted_errors.empty());
  EXPECT_FALSE(result.predicted_errors_populated);
  EXPECT_FALSE(result.low_confidence);
  EXPECT_EQ(decoder.get_last_shot_num_reweights(), 1);
  EXPECT_NEAR(result.total_cost, -std::log(0.499 / 0.501), 1e-12);
  EXPECT_GT(result.total_cost, 0);
  expect_costs_restored(decoder);
}

TEST(MultiPassTesseractDecoderTest, RestoresCostsWhenLaterDecodeThrows) {
  MultiPassTesseractDecoder decoder(correlated_dem(), 2, {0, 1});
  const auto& schedule = MultiPassTestPeer::get_pass_schedule(decoder);
  ASSERT_EQ(schedule, std::vector<std::vector<size_t>>({{0}, {1}}));
  TesseractDecoder& target = MultiPassTestPeer::get_component_decoder(decoder, schedule[1][0]);
  auto saved_orders = target.config.detector_orders;
  target.config.detector_orders.clear();

  EXPECT_THROW(decoder.decode({0, 1}), std::runtime_error);
  EXPECT_GT(decoder.get_last_shot_num_reweights(), 0);
  expect_costs_restored(decoder);

  target.config.detector_orders = saved_orders;
  EXPECT_EQ(decoder.decode({0, 1}), std::vector<int>({0}));
  expect_costs_restored(decoder);
}

TEST(MultiPassTesseractDecoderTest, RepeatedSparseShotsDoNotInheritState) {
  TesseractConfig config;
  config.sparsify_errors = true;
  config.sparsify_base_degree = 1;
  MultiPassTesseractDecoder decoder(correlated_dem(), 2, {0, 1}, config);

  DecodeResult first = decoder.decode_result({0, 1});
  size_t first_reweights = decoder.get_last_shot_num_reweights();
  expect_costs_restored(decoder);
  EXPECT_TRUE(decoder.decode({}).empty());
  EXPECT_EQ(decoder.get_last_shot_num_reweights(), 0);
  expect_costs_restored(decoder);
  DecodeResult repeated = decoder.decode_result({0, 1});

  MultiPassTesseractDecoder fresh(correlated_dem(), 2, {0, 1}, config);
  DecodeResult fresh_result = fresh.decode_result({0, 1});
  EXPECT_EQ(repeated.predictions, first.predictions);
  EXPECT_EQ(repeated.predictions, fresh_result.predictions);
  EXPECT_EQ(repeated.low_confidence, first.low_confidence);
  EXPECT_DOUBLE_EQ(repeated.total_cost, first.total_cost);
  EXPECT_DOUBLE_EQ(repeated.total_cost, fresh_result.total_cost);
  EXPECT_EQ(decoder.get_last_shot_num_reweights(), first_reweights);
  expect_costs_restored(decoder);
}

TEST(MultiPassTesseractDecoderTest, AggregatesLowConfidenceAcrossPasses) {
  MultiPassTesseractDecoder decoder(correlated_dem(), 2, {0, 1});
  const auto& schedule = MultiPassTestPeer::get_pass_schedule(decoder);
  ASSERT_EQ(schedule, std::vector<std::vector<size_t>>({{0}, {1}}));
  MultiPassTestPeer::get_component_decoder(decoder, schedule[0][0]).config.pqlimit = 1;

  DecodeResult result = decoder.decode_result({0, 1});
  EXPECT_TRUE(result.low_confidence);
  EXPECT_EQ(result.predictions, std::vector<int>({0}));
}

TEST(MultiPassTesseractDecoderTest, ExecutionPlanIsComputedOnDemand) {
  MultiPassTesseractDecoder decoder(correlated_dem(), 2, {4, 9});
  MultiPassExecutionPlan plan = decoder.get_execution_plan();
  EXPECT_EQ(plan.num_passes, 2);
  EXPECT_EQ(plan.components.size(), 2);
  EXPECT_EQ(plan.dependencies.size(), 2);
  EXPECT_EQ(plan.pass_schedule, std::vector<std::vector<size_t>>({{0}, {1}}));
  EXPECT_NE(plan.str().find("strategy: causal"), std::string::npos);
}

TEST(MultiPassTesseractDecoderTest, StaticSchedulerRunsBothComponentsInEveryPass) {
  MultiPassTesseractDecoder decoder(correlated_dem(), 2, {0, 1}, TesseractConfig(),
                                    SchedulingStrategy::Static);
  MultiPassExecutionPlan plan = decoder.get_execution_plan();
  EXPECT_EQ(plan.strategy, SchedulingStrategy::Static);
  EXPECT_EQ(plan.pass_schedule, std::vector<std::vector<size_t>>({{0, 1}, {0, 1}}));
  EXPECT_EQ(decoder.decode({0, 1}), std::vector<int>({0}));
}

}  // namespace
}  // namespace tesseract_decoder
