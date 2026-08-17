#include "multi_pass_tesseract_decoder.h"

#include <algorithm>
#include <fstream>
#include <random>
#include <vector>

#include "gtest/gtest.h"

using namespace tesseract;

stim::DetectorErrorModel load_test_dem(const std::string& filename) {
  std::string path = "testdata/surfacecodes/" + filename;
  std::ifstream is(path);
  if (!is.is_open()) {
    is.open(filename);
  }
  if (!is.is_open()) {
    throw std::runtime_error("Could not open file: " + filename);
  }
  std::stringstream ss;
  ss << is.rdbuf();
  stim::Circuit circuit(ss.str().c_str());
  return stim::ErrorAnalyzer::circuit_to_detector_error_model(circuit, true, true, false, false,
                                                              false, 0.0);
}

auto chromobius_classifier = [](int index, const std::vector<double>& coords,
                                const std::string& tag) -> int {
  if (coords.size() < 4) return -1;
  int c3 = (int)coords[3];
  if (c3 >= 0 && c3 <= 2) return 0;  // Basis X
  if (c3 >= 3 && c3 <= 5) return 1;  // Basis Z
  return -1;
};

TEST(MultiPassTesseractDecoderTest, AcceptsOnlyOneOrTwoPasses) {
  stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0
        error(0.1) D1 L0
        detector D0
        detector D1
        logical_observable L0
    )DEM");
  auto classifier = [](int index, const std::vector<double>&, const std::string&) -> int {
    return index;
  };

  for (auto strategy : {SchedulingStrategy::Static, SchedulingStrategy::Causal}) {
    for (size_t num_passes : {1, 2}) {
      EXPECT_NO_THROW(MultiPassTesseractDecoder(dem, num_passes, classifier, TesseractConfig(), 1,
                                                DetOrder::DetBFS, 0, strategy));
    }
    for (size_t num_passes : {0, 3, 4}) {
      EXPECT_THROW(MultiPassTesseractDecoder(dem, num_passes, classifier, TesseractConfig(), 1,
                                             DetOrder::DetBFS, 0, strategy),
                   std::invalid_argument);
    }
  }
}

TEST(MultiPassTesseractDecoderTest, RequiresTwoFullyClassifiedComponents) {
  stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0
        error(0.1) D1 L0
        error(0.1) D2
        detector D0
        detector D1
        detector D2
        logical_observable L0
    )DEM");

  std::vector<int> calls(3);
  std::vector<int> labels = {4, 4, 9};
  auto classifier = [&](int index, const std::vector<double>&, const std::string&) -> int {
    calls[index]++;
    return labels[index];
  };
  MultiPassTesseractDecoder decoder(dem, 1, classifier);
  EXPECT_EQ(decoder.num_components(), 2);
  EXPECT_EQ(calls, std::vector<int>({1, 1, 1}));

  EXPECT_THROW(MultiPassTesseractDecoder(dem, 1, std::vector<int>({4, 4, 4})),
               std::invalid_argument);
  EXPECT_THROW(MultiPassTesseractDecoder(dem, 1, std::vector<int>({4, 9, 12})),
               std::invalid_argument);

  try {
    MultiPassTesseractDecoder(dem, 1, std::vector<int>({4, -1, 9}));
    FAIL() << "Expected an unclassified detector to be rejected.";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("D1"), std::string::npos);
  }
}

TEST(MultiPassTesseractDecoderTest, TwoPassCorrelationBenefit) {
  // Component 0: D0 (Causal)
  // Component 1: D1 (Affected) -> Observable L0
  // Rule: D0 ^ D1 exists with probability 0.1
  // Independent: D0 with prob 0.01, D1 with prob 0.2
  // If D0 is detected and explained by the bridging error, D1's probability
  // should increase.

  stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0 ^ D1 L0
        error(0.01) D0
        error(0.2) D1 L0
        detector D0
        detector D1
        logical_observable L0
    )DEM");

  // Classifier: D0 -> Comp 0, D1 -> Comp 1
  auto classifier = [](int index, const std::vector<double>& coords,
                       const std::string& tag) -> int { return index; };

  TesseractConfig config;
  config.verbose = true;
  MultiPassTesseractDecoder decoder(dem, 2, classifier, config);

  // Shot 1: D0 and D1 both fire.
  // Pass 1: Decode Comp 0. D0 is explained by the bridging error (implicit).
  // Reweight: D1 L0 in Comp 1 becomes more likely.
  // Pass 2: Decode Comp 1.
  std::vector<uint64_t> detections = {0, 1};
  MultiPassDecodeResult result = decoder.decode_result(detections);

  // In this specific model, if D0 and D1 both fire,
  // the most likely explanation is the bridging error (0.1)
  // vs independent (0.01 * 0.2 = 0.002).
  // The bridging error flips L0.
  // So we expect L0 to be flipped.
  ASSERT_TRUE(std::find(result.predictions.begin(), result.predictions.end(), 0) !=
              result.predictions.end());
  EXPECT_DOUBLE_EQ(result.total_cost, 0.0);
}

TEST(MultiPassTesseractDecoderTest, DisjointDecoding) {
  stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0 L0
        error(0.1) D1 L1
        detector D0
        detector D1
        logical_observable L0
        logical_observable L1
    )DEM");

  auto classifier = [](int index, const std::vector<double>& coords,
                       const std::string& tag) -> int { return index; };

  MultiPassTesseractDecoder decoder(dem, 1, classifier);

  std::vector<uint64_t> detections = {0};
  std::vector<int> result = decoder.decode(detections);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0], 0);

  detections = {1};
  result = decoder.decode(detections);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0], 1);
}

TEST(MultiPassTesseractDecoderTest, CausalScheduleSurfaceCode) {
  // A simplified d=2 surface code style DEM
  // D0, D1: Basis X (Class 0), Affected by correlations from Basis Z
  // D2, D3: Basis Z (Class 1), Causal (Reweight Basis X)
  // Error: D2 ^ D0 (Bridge)
  stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0 D2 L0
        error(0.01) D0 L0
        error(0.01) D2
        error(0.1) D1 D3 L0
        error(0.01) D1 L0
        error(0.01) D3
        detector D0
        detector D1
        detector D2
        detector D3
        logical_observable L0
    )DEM");

  // Class 0: Detectors 0, 1
  // Class 1: Detectors 2, 3
  auto classifier = [](int index, const std::vector<double>& coords,
                       const std::string& tag) -> int { return (index < 2) ? 0 : 1; };

  MultiPassTesseractDecoder decoder(dem, 2, classifier, TesseractConfig(), 1, DetOrder::DetBFS, 0,
                                    SchedulingStrategy::Causal);

  const auto& schedule = MultiPassDebugger::get_pass_schedule(decoder);
  ASSERT_EQ(schedule.size(), 2);

  ASSERT_EQ(schedule[0].size(), 1);
  ASSERT_EQ(schedule[0][0], 1);  // Component 1 (Class 1) runs first
  ASSERT_EQ(schedule[1].size(), 1);
  ASSERT_EQ(schedule[1][0], 0);  // Component 0 (Class 0) runs last
}

TEST(MultiPassTesseractDecoderTest, ExecutionPlanReflectsDecoderState) {
  stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0 D1 L0
        error(0.01) D0
        error(0.2) D1 L0
        detector D0
        detector D1
        logical_observable L0
    )DEM");

  MultiPassTesseractDecoder decoder(dem, 2, std::vector<int>({4, 9}), TesseractConfig(), 1,
                                    DetOrder::DetIndex);
  MultiPassExecutionPlan plan = decoder.get_execution_plan();

  EXPECT_EQ(plan.strategy, SchedulingStrategy::Causal);
  EXPECT_EQ(plan.monolithic_statistics.detector_count, 2);
  EXPECT_EQ(plan.monolithic_statistics.error_mechanism_count, 3);
  EXPECT_DOUBLE_EQ(plan.monolithic_statistics.average_detector_row_weight, 2.0);
  ASSERT_EQ(plan.components.size(), 2);
  EXPECT_EQ(plan.components[0].classifier_label, 4);
  EXPECT_EQ(plan.components[1].classifier_label, 9);
  for (const auto& component : plan.components) {
    EXPECT_EQ(component.detector_count, 1);
    EXPECT_EQ(component.error_mechanism_count, 1);
    EXPECT_DOUBLE_EQ(component.average_detector_row_weight, 1.0);
  }
  ASSERT_EQ(plan.dependencies.size(), 2);
  EXPECT_EQ(plan.dependencies[0].source_component, 0);
  EXPECT_EQ(plan.dependencies[0].target_component, 1);
  EXPECT_GT(plan.dependencies[0].rule_count, 0);
  EXPECT_EQ(plan.pass_schedule, std::vector<std::vector<size_t>>({{0}, {1}}));
  EXPECT_NE(plan.str().find("monolithic DEM: detectors=2 error_mechanisms=3"), std::string::npos);
  EXPECT_NE(plan.str().find("component 0: label=4 detectors=1 observable=no error_mechanisms=1"),
            std::string::npos);
  EXPECT_NE(plan.str().find("pass 2: [1]"), std::string::npos);
}

TEST(MultiPassTesseractDecoderTest, SurfaceCodePartitioning) {
  std::vector<int> distances = {3, 5, 7};
  for (int d : distances) {
    int q = 2 * d * d - 1;
    std::string filename = "r=" + std::to_string(d) + ",d=" + std::to_string(d) +
                           ",p=0.001,noise=si1000,c=surface_code_X,q=" + std::to_string(q) +
                           ",gates=cz.stim";
    stim::DetectorErrorModel dem = load_test_dem(filename);
    MultiPassTesseractDecoder decoder(dem, 1, chromobius_classifier);
    ASSERT_EQ(decoder.num_components(), 2) << "Failed partitioning for d=" << d;
  }
}

TEST(MultiPassTesseractDecoderTest, SurfaceCodeCausalScheduling) {
  std::vector<int> distances = {3, 5, 7};
  for (int d : distances) {
    int q = 2 * d * d - 1;
    std::string filename = "r=" + std::to_string(d) + ",d=" + std::to_string(d) +
                           ",p=0.001,noise=si1000,c=surface_code_X,q=" + std::to_string(q) +
                           ",gates=cz.stim";
    stim::DetectorErrorModel dem = load_test_dem(filename);

    // 1-Pass: Should only schedule X component (0)
    {
      MultiPassTesseractDecoder decoder(dem, 1, chromobius_classifier, TesseractConfig(), 1,
                                        DetOrder::DetBFS, 0, SchedulingStrategy::Causal);
      const auto& schedule = MultiPassDebugger::get_pass_schedule(decoder);
      ASSERT_EQ(schedule.size(), 1);
      ASSERT_EQ(schedule[0].size(), 1);
      ASSERT_EQ(schedule[0][0], 0) << "1-pass failed for d=" << d;
    }

    // 2-Pass: Should schedule Z (1) then X (0)
    {
      MultiPassTesseractDecoder decoder(dem, 2, chromobius_classifier, TesseractConfig(), 1,
                                        DetOrder::DetBFS, 0, SchedulingStrategy::Causal);
      const auto& schedule = MultiPassDebugger::get_pass_schedule(decoder);
      ASSERT_EQ(schedule.size(), 2);
      ASSERT_EQ(schedule[0].size(), 1);
      ASSERT_EQ(schedule[0][0], 1) << "2-pass P0 failed for d=" << d;
      ASSERT_EQ(schedule[1].size(), 1);
      ASSERT_EQ(schedule[1][0], 0) << "2-pass P1 failed for d=" << d;
    }
  }
}

TEST(MultiPassTesseractDecoderTest, PerfectResetSurfaceCode) {
  std::vector<int> distances = {3, 5, 7};
  for (int d : distances) {
    int q = 2 * d * d - 1;
    std::string filename = "r=" + std::to_string(d) + ",d=" + std::to_string(d) +
                           ",p=0.001,noise=si1000,c=surface_code_X,q=" + std::to_string(q) +
                           ",gates=cz.stim";
    stim::DetectorErrorModel dem = load_test_dem(filename);
    MultiPassTesseractDecoder decoder(dem, 2, chromobius_classifier, TesseractConfig(), 1,
                                      DetOrder::DetBFS, 0, SchedulingStrategy::Causal);

    size_t n_comp = MultiPassDebugger::num_components(decoder);

    // Capture initial state
    std::vector<std::vector<double>> initial_likelihoods(n_comp);
    std::vector<std::vector<ErrorCost>> initial_error_costs(n_comp);
    for (size_t i = 0; i < n_comp; ++i) {
      const auto& comp_dec = MultiPassDebugger::get_component_decoder(decoder, i);
      for (const auto& err : comp_dec.errors) {
        initial_likelihoods[i].push_back(err.likelihood_cost);
      }
      initial_error_costs[i] = TesseractDebugger::get_error_costs(comp_dec);
    }

    // Run shots
    std::mt19937_64 rng(12345);
    size_t total_reweights_in_test = 0;
    for (int shot = 0; shot < 100; ++shot) {
      std::vector<uint64_t> detections;
      for (uint64_t det_idx = 0; det_idx < dem.count_detectors(); ++det_idx) {
        if (std::uniform_real_distribution<double>(0, 1)(rng) < 0.05) {
          detections.push_back(det_idx);
        }
      }

      decoder.decode(detections);
      total_reweights_in_test += decoder.get_last_shot_num_reweights();

      // Verify state is restored
      for (size_t i = 0; i < n_comp; ++i) {
        const auto& comp_dec = MultiPassDebugger::get_component_decoder(decoder, i);

        for (size_t ei = 0; ei < comp_dec.errors.size(); ++ei) {
          ASSERT_DOUBLE_EQ(comp_dec.errors[ei].likelihood_cost, initial_likelihoods[i][ei])
              << "Likelihood mismatch at d=" << d << " shot=" << shot << " comp=" << i
              << " err=" << ei;
        }

        const auto& current_error_costs = TesseractDebugger::get_error_costs(comp_dec);
        ASSERT_EQ(current_error_costs.size(), initial_error_costs[i].size());
        for (size_t ei = 0; ei < current_error_costs.size(); ++ei) {
          ASSERT_DOUBLE_EQ(current_error_costs[ei].likelihood_cost,
                           initial_error_costs[i][ei].likelihood_cost)
              << "Internal likelihood mismatch at d=" << d << " shot=" << shot << " comp=" << i
              << " err=" << ei;
          ASSERT_DOUBLE_EQ(current_error_costs[ei].min_cost, initial_error_costs[i][ei].min_cost)
              << "Internal min_cost mismatch at d=" << d << " shot=" << shot << " comp=" << i
              << " err=" << ei;
        }
      }
    }
    ASSERT_GT(total_reweights_in_test, 0)
        << "Test was trivial for d=" << d << ". No reweighting occurred.";
  }
}

TEST(MultiPassTesseractDecoderTest, BoundaryConditionAndCappingTest) {
  stim::DetectorErrorModel dem(R"DEM(
        error(0.49) D0 D1 L0
        error(0.5) D0
        detector D0
        detector D1
        logical_observable L0
    )DEM");

  auto classifier = [](int index, const std::vector<double>& coords,
                       const std::string& tag) -> int { return index; };

  TesseractConfig config;
  config.dem = dem;

  MultiPassTesseractDecoder decoder(dem, 2, classifier, config, 1, DetOrder::DetIndex, 12345,
                                    SchedulingStrategy::Causal);

  std::vector<uint64_t> hits = {0};
  ASSERT_NO_THROW(decoder.decode(hits));
}

TEST(MultiPassTesseractDecoderTest, PriorReweightingOccurs) {
  stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0 D1 L0
        error(0.01) D0
        error(0.2) D1 L0
        detector D0
        detector D1
        logical_observable L0
    )DEM");

  auto classifier = [](int index, const std::vector<double>& coords,
                       const std::string& tag) -> int { return index; };

  TesseractConfig config;
  config.dem = dem;

  MultiPassTesseractDecoder decoder(dem, 2, classifier, config, 1, DetOrder::DetIndex, 12345,
                                    SchedulingStrategy::Causal);

  std::vector<uint64_t> hits = {0};
  decoder.decode(hits);

  // Rigorously assert that prior LLR reweights occurred successfully on the raw
  // un-decomposed DEM!
  ASSERT_GT(decoder.get_last_shot_num_reweights(), 0);
}

TEST(MultiPassTesseractDecoderTest, MultipleCausalTriggersMaxProbValidation) {
  stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D1 D0 L0
        error(0.15) D2 D0 L0
        error(0.01) D1
        error(0.01) D2
        error(0.2) D0 L0
        detector D0
        detector D1
        detector D2
        logical_observable L0
    )DEM");

  auto classifier = [](int index, const std::vector<double>& coords,
                       const std::string& tag) -> int {
    if (index == 0) return 0;
    return 1;
  };

  TesseractConfig config;
  config.dem = dem;

  MultiPassTesseractDecoder decoder(dem, 2, classifier, config, 1, DetOrder::DetIndex, 12345,
                                    SchedulingStrategy::Causal);

  // 1. Run a shot that triggers BOTH causal detectors D1 and D2
  std::vector<uint64_t> detections = {1, 2};
  decoder.decode(detections);

  const auto& comp0 = MultiPassDebugger::get_component_decoder_full(decoder, 0);

  // 2. Find the target error index for symptom D0 L0.
  ComponentSymptom target_symptom{{0}, {0}};
  auto it = comp0.symptom_to_error_index.find(target_symptom);
  ASSERT_NE(it, comp0.symptom_to_error_index.end());

  // Now fully active using vector<size_t> degeneracy mapping!
  size_t target_err_idx = it->second[0];

  double final_p = comp0.decoder->errors[target_err_idx].get_probability();

  // 3. Assert that the Surgical Reset successfully restored the cost back to
  // the baseline after exactly 2 LLR reweighting rules were triggered and
  // applied!
  ASSERT_DOUBLE_EQ(final_p, 0.33199999999999996);
  ASSERT_EQ(decoder.get_last_shot_num_reweights(), 2);
}

TEST(MultiPassTesseractDecoderTest, CorrelationRulesDistinguishObservables) {
  stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0 D1 L0
        error(0.01) D0
        error(0.2) D1 L0
        error(0.05) D1 L1
        detector D0
        detector D1
        logical_observable L0
        logical_observable L1
    )DEM");

  auto classifier = [](int index, const std::vector<double>& coords,
                       const std::string& tag) -> int { return index; };

  TesseractConfig config;
  config.dem = dem;

  MultiPassTesseractDecoder decoder(dem, 2, classifier, config, 1, DetOrder::DetIndex, 12345,
                                    SchedulingStrategy::Causal);

  decoder.decode({0});
  EXPECT_EQ(decoder.get_last_shot_num_reweights(), 1);
}
