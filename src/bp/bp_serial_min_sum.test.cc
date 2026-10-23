#include "bp/bp_serial_min_sum.h"

#include "bp/multi_pass_bp_decoder.h"
#include "bp/tesseract_bp_decoder.h"
#include "gtest/gtest.h"
#include "stim.h"

namespace bp {

const double kLowErrorProb = 0.1;
const double kHighErrorProb = 0.9;
const LLR_INT kPositivePriorLlr = prob_double_to_llr_int(kLowErrorProb);
const LLR_INT kNegativePriorLlr = prob_double_to_llr_int(kHighErrorProb);
const float kNormalizationFactor = 0.875f;

TEST(BpSerialMinSumTest, ConvergesOnSimpleErrors) {
  TannerGraph graph(5, 4,
                    {kPositivePriorLlr, kPositivePriorLlr, kPositivePriorLlr,
                     kPositivePriorLlr, kPositivePriorLlr});
  graph.add_edge(0, 0);
  graph.add_edge(1, 0);
  graph.add_edge(1, 1);
  graph.add_edge(2, 1);
  graph.add_edge(2, 2);
  graph.add_edge(3, 2);
  graph.add_edge(3, 3);
  graph.add_edge(4, 3);
  graph.build();

  std::vector<std::vector<size_t>> batched_syndromes;
  batched_syndromes.push_back({1, 2});  // Shot 0: Error at V2
  batched_syndromes.push_back({0, 3});  // Shot 1: Error at V0, V4
  for (size_t i = 2; i < BP_BATCH_SIZE; i++) batched_syndromes.push_back({});

  std::vector<LLR_INT> batched_posteriors(5 * BP_BATCH_SIZE, 0);

  auto batched_results = bp_serial_min_sum(
      graph, batched_syndromes, batched_posteriors, 20, kNormalizationFactor, true);

  EXPECT_TRUE(batched_results[0].converged);
  EXPECT_GT(batched_posteriors[0 * BP_BATCH_SIZE + 0], 0);
  EXPECT_GT(batched_posteriors[1 * BP_BATCH_SIZE + 0], 0);
  EXPECT_LT(batched_posteriors[2 * BP_BATCH_SIZE + 0], 0);
  EXPECT_GT(batched_posteriors[3 * BP_BATCH_SIZE + 0], 0);
  EXPECT_GT(batched_posteriors[4 * BP_BATCH_SIZE + 0], 0);

  EXPECT_TRUE(batched_results[1].converged);
  EXPECT_LT(batched_posteriors[0 * BP_BATCH_SIZE + 1], 0);
  EXPECT_GT(batched_posteriors[1 * BP_BATCH_SIZE + 1], 0);
  EXPECT_GT(batched_posteriors[2 * BP_BATCH_SIZE + 1], 0);
  EXPECT_GT(batched_posteriors[3 * BP_BATCH_SIZE + 1], 0);
  EXPECT_LT(batched_posteriors[4 * BP_BATCH_SIZE + 1], 0);
}

TEST(BpSerialMinSumTest, ConvergesToZeroSyndromeDespiteIncorrectPrior) {
  TannerGraph graph(5, 4,
                    {kPositivePriorLlr, kPositivePriorLlr, kNegativePriorLlr,
                     kPositivePriorLlr, kPositivePriorLlr});
  graph.add_edge(0, 0);
  graph.add_edge(1, 0);
  graph.add_edge(1, 1);
  graph.add_edge(2, 1);
  graph.add_edge(2, 2);
  graph.add_edge(3, 2);
  graph.add_edge(3, 3);
  graph.add_edge(4, 3);
  graph.build();

  std::vector<std::vector<size_t>> batched_syndromes = {{}};
  std::vector<LLR_INT> batched_posteriors(5 * BP_BATCH_SIZE, 0);

  auto results = bp_serial_min_sum(graph, batched_syndromes, batched_posteriors, 20,
                                   kNormalizationFactor, true);

  EXPECT_TRUE(results[0].converged);
  for (size_t v = 0; v < 5; ++v) {
    EXPECT_GT(batched_posteriors[v * BP_BATCH_SIZE + 0], 0);
  }
}

TEST(BpSerialMinSumTest, ConvergesWithRandomSchedulePermutations) {
  TannerGraph graph(5, 4,
                    {kPositivePriorLlr, kPositivePriorLlr, kPositivePriorLlr,
                     kPositivePriorLlr, kPositivePriorLlr});
  graph.add_edge(0, 0);
  graph.add_edge(1, 0);
  graph.add_edge(1, 1);
  graph.add_edge(2, 1);
  graph.add_edge(2, 2);
  graph.add_edge(3, 2);
  graph.add_edge(3, 3);
  graph.add_edge(4, 3);
  graph.build();

  std::vector<std::vector<size_t>> batched_syndromes;
  batched_syndromes.push_back({1, 2});
  batched_syndromes.push_back({0, 3});
  for (size_t i = 2; i < BP_BATCH_SIZE; i++) batched_syndromes.push_back({});

  std::vector<LLR_INT> batched_posteriors(5 * BP_BATCH_SIZE, 0);

  auto batched_results = bp_serial_min_sum(
      graph, batched_syndromes, batched_posteriors, 20, kNormalizationFactor, true, true, 12345);

  EXPECT_TRUE(batched_results[0].converged);
  EXPECT_GT(batched_posteriors[0 * BP_BATCH_SIZE + 0], 0);
  EXPECT_GT(batched_posteriors[1 * BP_BATCH_SIZE + 0], 0);
  EXPECT_LT(batched_posteriors[2 * BP_BATCH_SIZE + 0], 0);
  EXPECT_GT(batched_posteriors[3 * BP_BATCH_SIZE + 0], 0);
  EXPECT_GT(batched_posteriors[4 * BP_BATCH_SIZE + 0], 0);

  EXPECT_TRUE(batched_results[1].converged);
  EXPECT_LT(batched_posteriors[0 * BP_BATCH_SIZE + 1], 0);
  EXPECT_GT(batched_posteriors[1 * BP_BATCH_SIZE + 1], 0);
  EXPECT_GT(batched_posteriors[2 * BP_BATCH_SIZE + 1], 0);
  EXPECT_GT(batched_posteriors[3 * BP_BATCH_SIZE + 1], 0);
  EXPECT_LT(batched_posteriors[4 * BP_BATCH_SIZE + 1], 0);
}

TEST(BpSerialMinSumTest, MultiPassBpCorrelatedDemReweighting) {
  stim::DetectorErrorModel dem(R"DEM(
    error(0.1) D0 D1 L0
    error(0.01) D0
    error(0.02) D1 L0
    detector D0
    detector D1
    logical_observable L0
  )DEM");

  BPParams params;
  params.max_iter = 20;
  params.normalization_factor = 0.875f;

  MultiPassBpDecoder decoder(dem, params, /*num_passes=*/2, /*detector_components=*/{0, 1});

  const auto& sched = decoder.pass_schedule();
  ASSERT_EQ(sched.size(), 2);
  EXPECT_EQ(sched[0], std::vector<size_t>{0});
  EXPECT_EQ(sched[1], std::vector<size_t>{1});

  EXPECT_EQ(decoder.num_observables(), 1);
  EXPECT_EQ(decoder.total_detectors(), 2);

  auto res_empty = decoder.decode({});
  ASSERT_EQ(res_empty.size(), 1);
  EXPECT_EQ(res_empty[0], 0);

  auto res_d0 = decoder.decode({0});
  ASSERT_EQ(res_d0.size(), 1);
  EXPECT_EQ(res_d0[0], 0);

  auto res_both = decoder.decode({0, 1});
  ASSERT_EQ(res_both.size(), 1);
  EXPECT_EQ(res_both[0], 1);

  std::vector<std::vector<uint64_t>> batch = {{0, 1}, {}, {0, 1}, {0}, {1}};
  auto batch_res = decoder.decode_batch(batch);
  ASSERT_EQ(batch_res.size(), 5);
  EXPECT_EQ(batch_res[0], std::vector<uint8_t>{1});
  EXPECT_EQ(batch_res[1], std::vector<uint8_t>{0});
  EXPECT_EQ(batch_res[2], std::vector<uint8_t>{1});
  EXPECT_EQ(batch_res[3], std::vector<uint8_t>{0});
  EXPECT_EQ(batch_res[4], std::vector<uint8_t>{1});

  auto prof = decoder.profile();
  EXPECT_GT(prof.num_shots, 0);
  EXPECT_GT(prof.num_decode_calls, 0);

  decoder.reset_profile();
  auto prof_reset = decoder.profile();
  EXPECT_EQ(prof_reset.num_shots, 0);
  EXPECT_EQ(prof_reset.num_decode_calls, 0);
}

TEST(BpSerialMinSumTest, MultiPassBpClassification) {
  stim::DetectorErrorModel dem(R"DEM(
    detector(0, 0, 0, 1.0) D0
    detector(1, 0, 0, 2.0) D1
    detector(0, 0, 0, 4.0) D2
    detector(1, 0, 0, 5.0) D3
    error(0.1) D0 D1 L0
    error(0.1) D2 D3 L1
  )DEM");

  auto classified = classify_detector_components(dem);
  ASSERT_EQ(classified.size(), 4);
  EXPECT_EQ(classified[0], 0);
  EXPECT_EQ(classified[1], 0);
  EXPECT_EQ(classified[2], 1);
  EXPECT_EQ(classified[3], 1);

  BPParams params;
  params.max_iter = 20;
  params.normalization_factor = 0.875f;
  params.osd_order = 0;

  MultiPassBpDecoder decoder(dem, params, /*num_passes=*/1, /*detector_components=*/{});

  const auto& sched = decoder.pass_schedule();
  ASSERT_EQ(sched.size(), 1);
  EXPECT_EQ(sched[0], (std::vector<size_t>{0, 1}));

  EXPECT_EQ(decoder.total_detectors(), 4);
  EXPECT_EQ(decoder.num_observables(), 2);

  auto res0 = decoder.decode({0, 1});
  ASSERT_EQ(res0.size(), 2);
  EXPECT_EQ(res0[0], 1);
  EXPECT_EQ(res0[1], 0);

  auto res1 = decoder.decode({2, 3});
  ASSERT_EQ(res1.size(), 2);
  EXPECT_EQ(res1[0], 0);
  EXPECT_EQ(res1[1], 1);

  auto res2 = decoder.decode({0, 1, 2, 3});
  ASSERT_EQ(res2.size(), 2);
  EXPECT_EQ(res2[0], 1);
  EXPECT_EQ(res2[1], 1);

  auto osd_prof = decoder.osd_profile();
  EXPECT_GT(osd_prof.num_calls, 0);
}

TEST(BpSerialMinSumTest, MultiPassBpBasisTagClassification) {
  stim::DetectorErrorModel dem(R"DEM(
    detector[{"measure_basis": "X"}] D0
    detector[{"measure_basis": "X"}] D1
    detector[{"measure_basis": "Z"}] D2
    detector[{"measure_basis": "Z"}] D3
    error(0.1) D0 D1
    error(0.1) D2 D3
  )DEM");

  auto classified = classify_detector_components(dem);
  ASSERT_EQ(classified.size(), 4);
  EXPECT_EQ(classified[0], 0);
  EXPECT_EQ(classified[1], 0);
  EXPECT_EQ(classified[2], 1);
  EXPECT_EQ(classified[3], 1);
}

TEST(BpSerialMinSumTest, BpParamsPresetsAndClusterOsd) {
  stim::DetectorErrorModel dem(R"DEM(
    error(0.1) D0 D1 L0
    error(0.05) D1 D2 L0
    error(0.05) D0 D2
    detector D0
    detector D1
    detector D2
    logical_observable L0
  )DEM");

  BPParams osd_params = BPParams::Osd(10);
  EXPECT_EQ(osd_params.max_iter, 10);
  EXPECT_EQ(osd_params.osd_order, 24);
  EXPECT_EQ(osd_params.osd_weight, 2);

  BPParams fast_params = BPParams::Fast(5);
  EXPECT_EQ(fast_params.max_iter, 5);
  EXPECT_EQ(fast_params.osd_order, 10);
  EXPECT_EQ(fast_params.osd_weight, 0);
  EXPECT_TRUE(fast_params.osd_early_exit);

  TesseractBpDecoder dec_osd(dem, osd_params);
  auto post_osd = dec_osd.create_post_processor();
  ASSERT_NE(post_osd, nullptr);
  auto res = dec_osd.decode({0, 1}, post_osd);
  ASSERT_EQ(res.size(), 1);
  EXPECT_EQ(res[0], 1);
}

}  // namespace bp
