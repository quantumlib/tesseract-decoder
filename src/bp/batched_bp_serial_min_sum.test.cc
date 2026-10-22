#include "bp/batched_bp_serial_min_sum.h"

#include "bp/bp_test_util.h"
#include "bp/hard_decision_post_processor.h"
#include "bp/multi_pass_bp_decoder.h"
#include "bp/tesseract_bp_decoder.h"
#include "bp/tesseract_post_processor.h"
#include "gtest/gtest.h"
#include "stim.h"

namespace bp {

const float kLowErrorProb = 0.1f;
const float kHighErrorProb = 0.9f;
const float kPositivePriorLlr = logf((1 - kLowErrorProb) / kLowErrorProb);
const float kNegativePriorLlr = logf((1 - kHighErrorProb) / kHighErrorProb);
const float kNormalizationFactor = 0.875f;

TEST(BatchedBpSerialMinSumTest, ConvergesOnSimpleErrors) {
  TannerGraph<float> graph(5, 4,
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

  BatchedTannerGraph<float> batched_graph;
  batched_graph.build_from_unbatched(graph);

  std::vector<std::vector<size_t>> batched_syndromes;
  batched_syndromes.push_back({1, 2});  // Shot 0: Error at V2
  batched_syndromes.push_back({0, 3});  // Shot 1: Error at V0, V4
  for (size_t i = 2; i < BP_BATCH_SIZE; i++) batched_syndromes.push_back({});  // Other shots empty

  std::vector<float> batched_posteriors(5 * BP_BATCH_SIZE, 0.0f);

  auto batched_results = batched_bp_serial_min_sum(
      batched_graph, batched_syndromes, batched_posteriors, 20, kNormalizationFactor, true);

  // Shot 0 should converge to fixing V2
  EXPECT_TRUE(batched_results[0].converged);
  EXPECT_GT(batched_posteriors[0 * BP_BATCH_SIZE + 0], 0);
  EXPECT_GT(batched_posteriors[1 * BP_BATCH_SIZE + 0], 0);
  EXPECT_LT(batched_posteriors[2 * BP_BATCH_SIZE + 0], 0);  // V2 is negative (error)
  EXPECT_GT(batched_posteriors[3 * BP_BATCH_SIZE + 0], 0);
  EXPECT_GT(batched_posteriors[4 * BP_BATCH_SIZE + 0], 0);

  // Shot 1 should converge to fixing V0, V4
  EXPECT_TRUE(batched_results[1].converged);
  EXPECT_LT(batched_posteriors[0 * BP_BATCH_SIZE + 1], 0);  // V0 is negative
  EXPECT_GT(batched_posteriors[1 * BP_BATCH_SIZE + 1], 0);
  EXPECT_GT(batched_posteriors[2 * BP_BATCH_SIZE + 1], 0);
  EXPECT_GT(batched_posteriors[3 * BP_BATCH_SIZE + 1], 0);
  EXPECT_LT(batched_posteriors[4 * BP_BATCH_SIZE + 1], 0);  // V4 is negative
}

TEST(BatchedBpSerialMinSumTest, ConvergesWithRandomSchedulePermutations) {
  TannerGraph<float> graph(5, 4,
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

  BatchedTannerGraph<float> batched_graph;
  batched_graph.build_from_unbatched(graph);

  std::vector<std::vector<size_t>> batched_syndromes;
  batched_syndromes.push_back({1, 2});  // Shot 0: Error at V2
  batched_syndromes.push_back({0, 3});  // Shot 1: Error at V0, V4
  for (size_t i = 2; i < BP_BATCH_SIZE; i++) batched_syndromes.push_back({});

  std::vector<float> batched_posteriors(5 * BP_BATCH_SIZE, 0.0f);

  auto batched_results =
      batched_bp_serial_min_sum(batched_graph, batched_syndromes, batched_posteriors, 20,
                                kNormalizationFactor, true, true, 12345);

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

TEST(BatchedBpSerialMinSumTest, MultiPassBpCorrelatedDemReweighting) {
  // Correlated error model where error causes D0 in comp 0 and D1 L0 in comp 1.
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
  params.schedule = "serial";
  params.update_rule = "min-sum";
  params.normalization_factor = 0.875f;

  MultiPassBpDecoder decoder(dem, params, /*num_passes=*/2, /*detector_components=*/{0, 1});

  // Verify schedule:
  // Component 1 affects observable L0.
  // Component 0 reweights Component 1.
  // Under causal schedule: pass 0 runs comp 0, pass 1 runs comp 1.
  const auto& sched = decoder.pass_schedule();
  ASSERT_EQ(sched.size(), 2);
  EXPECT_EQ(sched[0], std::vector<size_t>{0});
  EXPECT_EQ(sched[1], std::vector<size_t>{1});

  EXPECT_EQ(decoder.num_observables(), 1);
  EXPECT_EQ(decoder.total_detectors(), 2);

  // Single-shot tests:
  // No error -> no correction
  auto res_empty = decoder.decode({});
  ASSERT_EQ(res_empty.size(), 1);
  EXPECT_EQ(res_empty[0], 0);

  // Error only on D0 -> no observable flip
  auto res_d0 = decoder.decode({0});
  ASSERT_EQ(res_d0.size(), 1);
  EXPECT_EQ(res_d0[0], 0);

  // Correlated error on D0 and D1 -> comp 0 predicts D0, reweights D1 in comp 1,
  // comp 1 predicts D1 and corrects L0
  auto res_both = decoder.decode({0, 1});
  ASSERT_EQ(res_both.size(), 1);
  EXPECT_EQ(res_both[0], 1);

  // Batched decode test
  std::vector<std::vector<uint64_t>> batch = {{0, 1}, {}, {0, 1}, {0}, {1}};
  auto batch_res = decoder.decode_batch(batch);
  ASSERT_EQ(batch_res.size(), 5);
  EXPECT_EQ(batch_res[0], std::vector<uint8_t>{1});
  EXPECT_EQ(batch_res[1], std::vector<uint8_t>{0});
  EXPECT_EQ(batch_res[2], std::vector<uint8_t>{1});
  EXPECT_EQ(batch_res[3], std::vector<uint8_t>{0});
  EXPECT_EQ(batch_res[4], std::vector<uint8_t>{1});

  // Verify profile aggregation
  auto prof = decoder.profile();
  EXPECT_GT(prof.num_shots, 0);
  EXPECT_GT(prof.num_decode_calls, 0);

  decoder.reset_profile();
  auto prof_reset = decoder.profile();
  EXPECT_EQ(prof_reset.num_shots, 0);
  EXPECT_EQ(prof_reset.num_decode_calls, 0);
}

TEST(BatchedBpSerialMinSumTest, MultiPassBpClassificationAndStaticSchedule) {
  // Test coordinate-based auto-classification with superdense color code style coordinates:
  // D0, D1 have last coordinate < 3.0 -> Component 0 (X checks)
  // D2, D3 have last coordinate >= 3.0 -> Component 1 (Z checks)
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
  params.schedule = "serial";
  params.update_rule = "min-sum";
  params.normalization_factor = 0.875f;
  params.osd_order = 0;

  // Use Static schedule with 1 pass
  MultiPassBpDecoder decoder(dem, params, /*num_passes=*/1, /*detector_components=*/{},
                             tesseract_decoder::SchedulingStrategy::Static);

  const auto& sched = decoder.pass_schedule();
  ASSERT_EQ(sched.size(), 1);
  EXPECT_EQ(sched[0], (std::vector<size_t>{0, 1}));

  EXPECT_EQ(decoder.total_detectors(), 4);
  EXPECT_EQ(decoder.num_observables(), 2);

  // Comp 0 error on D0, D1 -> flips L0
  auto res0 = decoder.decode({0, 1});
  ASSERT_EQ(res0.size(), 2);
  EXPECT_EQ(res0[0], 1);
  EXPECT_EQ(res0[1], 0);

  // Comp 1 error on D2, D3 -> flips L1
  auto res1 = decoder.decode({2, 3});
  ASSERT_EQ(res1.size(), 2);
  EXPECT_EQ(res1[0], 0);
  EXPECT_EQ(res1[1], 1);

  // Both errors
  auto res2 = decoder.decode({0, 1, 2, 3});
  ASSERT_EQ(res2.size(), 2);
  EXPECT_EQ(res2[0], 1);
  EXPECT_EQ(res2[1], 1);

  // Verify OSD profile
  auto osd_prof = decoder.osd_profile();
  EXPECT_GT(osd_prof.num_calls, 0);
}

TEST(BatchedBpSerialMinSumTest, MultiPassBpBasisTagClassification) {
  // Test JSON metadata tags with measure_basis
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

TEST(BatchedBpSerialMinSumTest, TesseractPostProcessorSinglePass) {
  stim::DetectorErrorModel dem(R"DEM(
    error(0.1) D0 L0
    error(0.1) D0 D1
    error(0.1) D1 D2
    error(0.1) D2 D3
    error(0.1) D3 L0
    detector D0
    detector D1
    detector D2
    detector D3
    logical_observable L0
  )DEM");

  BPParams params;
  params.max_iter = 20;
  params.schedule = "serial";
  params.update_rule = "min-sum";
  params.normalization_factor = 0.875f;
  params.post_processor = "tesseract";
  params.tesseract_det_beam = 5;
  params.tesseract_pqlimit = 2000;

  TesseractBpDecoder decoder(dem, params);
  auto pp = decoder.create_post_processor();
  ASSERT_NE(pp, nullptr);
  auto tess_pp = std::dynamic_pointer_cast<TesseractPostProcessor>(pp);
  ASSERT_NE(tess_pp, nullptr);

  // Test single shots:
  // Shot 0: empty
  auto res0 = decoder.decode({}, pp);
  ASSERT_EQ(res0.size(), 1);
  EXPECT_EQ(res0[0], 0);

  // Shot 1: D0 -> error on var 0, flips L0
  auto res1 = decoder.decode({0}, pp);
  ASSERT_EQ(res1.size(), 1);
  EXPECT_EQ(res1[0], 1);

  // Shot 2: D0, D1 -> error on var 1, no L0 flip
  auto res2 = decoder.decode({0, 1}, pp);
  ASSERT_EQ(res2.size(), 1);
  EXPECT_EQ(res2[0], 0);

  // Shot 3: D3 -> error on var 4, flips L0
  auto res3 = decoder.decode({3}, pp);
  ASSERT_EQ(res3.size(), 1);
  EXPECT_EQ(res3[0], 1);

  // Batched decode:
  std::vector<std::vector<uint64_t>> batch = {{}, {0}, {0, 1}, {3}};
  auto batch_res = decoder.decode_batch(batch, pp);
  ASSERT_EQ(batch_res.size(), 4);
  EXPECT_EQ(batch_res[0], std::vector<uint8_t>{0});
  EXPECT_EQ(batch_res[1], std::vector<uint8_t>{1});
  EXPECT_EQ(batch_res[2], std::vector<uint8_t>{0});
  EXPECT_EQ(batch_res[3], std::vector<uint8_t>{1});

  // Verify profile
  const auto& prof = tess_pp->profile();
  EXPECT_EQ(prof.num_calls, 8);

  // Now test with max_iter = 0 to force BP unconverged -> triggers residual peeling / Tesseract A*
  BPParams params_unconv = params;
  params_unconv.max_iter = 0;
  TesseractBpDecoder decoder_unconv(dem, params_unconv);
  auto pp_unconv = decoder_unconv.create_post_processor();
  auto tess_pp_unconv = std::dynamic_pointer_cast<TesseractPostProcessor>(pp_unconv);
  ASSERT_NE(tess_pp_unconv, nullptr);

  auto res_unconv = decoder_unconv.decode({0}, pp_unconv);
  ASSERT_EQ(res_unconv.size(), 1);
  EXPECT_EQ(res_unconv[0], 1);

  const auto& prof_unconv = tess_pp_unconv->profile();
  EXPECT_EQ(prof_unconv.num_calls, 1);
  EXPECT_EQ(prof_unconv.num_osd_invocations, 1);
  EXPECT_GT(prof_unconv.t_total, 0.0);
}

TEST(BatchedBpSerialMinSumTest, TesseractPostProcessorMultiPass) {
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
  params.schedule = "serial";
  params.update_rule = "min-sum";
  params.normalization_factor = 0.875f;
  params.post_processor = "tesseract";
  params.tesseract_det_beam = 5;
  params.tesseract_pqlimit = 2000;

  MultiPassBpDecoder decoder(dem, params, /*num_passes=*/2, /*detector_components=*/{0, 1});

  // Single shot with D0 and D1
  auto res_both = decoder.decode({0, 1});
  ASSERT_EQ(res_both.size(), 1);
  EXPECT_EQ(res_both[0], 1);

  // Single shot empty
  auto res_empty = decoder.decode({});
  ASSERT_EQ(res_empty.size(), 1);
  EXPECT_EQ(res_empty[0], 0);

  // Batched decode
  std::vector<std::vector<uint64_t>> batch = {{0, 1}, {}, {1}, {0}};
  auto batch_res = decoder.decode_batch(batch);
  ASSERT_EQ(batch_res.size(), 4);
  EXPECT_EQ(batch_res[0], std::vector<uint8_t>{1});
  EXPECT_EQ(batch_res[1], std::vector<uint8_t>{0});
  EXPECT_EQ(batch_res[2], std::vector<uint8_t>{1});
  EXPECT_EQ(batch_res[3], std::vector<uint8_t>{0});

  // Verify OsdProfile merging from TesseractPostProcessor
  auto prof = decoder.osd_profile();
  EXPECT_GT(prof.num_calls, 0);
}

}  // namespace bp