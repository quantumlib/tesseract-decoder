#include "bp/osd_post_processor.h"

#include <gtest/gtest.h>

#include "bp/bp_params.h"
#include "bp/multi_pass_bp_decoder.h"
#include "bp/tesseract_bp_decoder.h"
#include "stim.h"

namespace bp {

TEST(OsdPostProcessorTest, BasicOSD0) {
  size_t num_vars = 3;
  size_t num_checks = 2;

  std::vector<LLR_INT> priors = {
      llr_double_to_int(5.0),
      llr_double_to_int(0.5),
      llr_double_to_int(-10.0)
  };

  TannerGraph graph(num_vars, num_checks, priors);
  graph.add_edge(0, 0);
  graph.add_edge(1, 0);
  graph.add_edge(1, 1);
  graph.add_edge(2, 1);
  graph.build();

  std::vector<uint64_t> detection_events = {0};
  std::vector<std::vector<int>> hyperedge_observables = {{0}, {0}, {0}};

  OsdPostProcessor osd(graph, 0, 0);
  BPResult result{false, 10};

  std::vector<LLR_INT> posteriors = priors;
  auto obs_correction = osd.process(result, posteriors, detection_events, hyperedge_observables);

  ASSERT_EQ(obs_correction.size(), 1);
  EXPECT_EQ(obs_correction[0], 0);
}

TEST(OsdPostProcessorTest, OSD1Improvement) {
  size_t num_vars = 3;
  size_t num_checks = 2;

  std::vector<LLR_INT> priors = {llr_double_to_int(1.0), llr_double_to_int(1.0),
                                 llr_double_to_int(1.5)};

  TannerGraph graph(num_vars, num_checks, priors);
  graph.add_edge(0, 0);
  graph.add_edge(1, 1);
  graph.add_edge(2, 0);
  graph.add_edge(2, 1);
  graph.build();

  std::vector<uint64_t> detection_events = {0, 1};
  std::vector<std::vector<int>> hyperedge_observables = {{0}, {0}, {0}};

  BPResult result{false, 10};
  std::vector<LLR_INT> posteriors = priors;

  OsdPostProcessor osd0(graph, 10, 0);
  auto obs0 = osd0.process(result, posteriors, detection_events, hyperedge_observables);
  ASSERT_EQ(obs0.size(), 1);
  EXPECT_EQ(obs0[0], 0);

  OsdPostProcessor osd1(graph, 10, 1);
  auto obs1 = osd1.process(result, posteriors, detection_events, hyperedge_observables);
  ASSERT_EQ(obs1.size(), 1);
  EXPECT_EQ(obs1[0], 1);
}

TEST(OsdPostProcessorTest, SaturatedWrongEHardUnflip) {
  // Tests Sub-Cause 1A: when BP saturates a false-positive variable to a large
  // negative LLR (-20.0, e_hard=1) and many dummy variables have smaller |LLR|
  // than 20.0, Interior Local Subgraph Pull-In (V_interior) pulls the false
  // positive into the working RREF and unflips it.
  size_t num_vars = 25;
  size_t num_checks = 22;

  std::vector<LLR_INT> priors(num_vars, llr_double_to_int(3.0));
  priors[0] = llr_double_to_int(2.0);
  priors[1] = llr_double_to_int(4.0);

  TannerGraph graph(num_vars, num_checks, priors);
  // True error is variable 0 (touches checks 0 and 1, observable 0).
  graph.add_edge(0, 0);
  graph.add_edge(0, 1);
  // False positive variable 1 (touches check 0 only, observable 1).
  graph.add_edge(1, 0);
  // Dummy variables 2..21 touch disjoint checks 2..21 so they are outside D_local.
  for (size_t v = 2; v < 22; ++v) {
    graph.add_edge(v, v);
  }
  // Extra variables touching check 1.
  graph.add_edge(22, 1);
  graph.add_edge(23, 0);
  graph.add_edge(24, 0);
  graph.build();

  std::vector<uint64_t> detection_events = {0, 1};
  std::vector<std::vector<int>> hyperedge_observables(num_vars);
  hyperedge_observables[0] = {0};
  hyperedge_observables[1] = {1};

  std::vector<LLR_INT> posteriors(num_vars, llr_double_to_int(1.5));
  posteriors[0] = llr_double_to_int(0.5);    // True error (uncertain)
  posteriors[1] = llr_double_to_int(-25.0);  // Saturated false positive (e_hard=1)
  posteriors[22] = llr_double_to_int(0.2);

  BPResult result{false, 50};
  // Even with a tiny osd_order = 2, V_interior pulls in variable 1 and unflips it.
  OsdPostProcessor osd(graph, 2, 1);
  auto obs = osd.process(result, posteriors, detection_events, hyperedge_observables);
  ASSERT_EQ(obs.size(), 2);
  EXPECT_EQ(obs[0], 1);
  EXPECT_EQ(obs[1], 0);
}

TEST(OsdPostProcessorTest, MultiClusterDisjointImprovingMoves) {
  // Tests Sub-Cause 1C: two independent residual clusters where each cluster
  // requires 1 free-column flip to replace 2 expensive pivot columns.
  // Disjoint improving-move composition solves both clusters simultaneously.
  size_t num_vars = 6;
  size_t num_checks = 4;

  std::vector<LLR_INT> priors = {
      llr_double_to_int(2.0), llr_double_to_int(2.0), llr_double_to_int(2.5),
      llr_double_to_int(2.0), llr_double_to_int(2.0), llr_double_to_int(2.5)};

  TannerGraph graph(num_vars, num_checks, priors);
  // Cluster A: vars 0, 1 (pivots on checks 0, 1), var 2 (free col on checks 0, 1)
  graph.add_edge(0, 0);
  graph.add_edge(1, 1);
  graph.add_edge(2, 0);
  graph.add_edge(2, 1);
  // Cluster B: vars 3, 4 (pivots on checks 2, 3), var 5 (free col on checks 2, 3)
  graph.add_edge(3, 2);
  graph.add_edge(4, 3);
  graph.add_edge(5, 2);
  graph.add_edge(5, 3);
  graph.build();

  std::vector<uint64_t> detection_events = {0, 1, 2, 3};
  std::vector<std::vector<int>> hyperedge_observables = {
      {}, {}, {0}, {}, {}, {1}};

  std::vector<LLR_INT> posteriors = {
      llr_double_to_int(0.5), llr_double_to_int(0.6), llr_double_to_int(1.2),
      llr_double_to_int(0.7), llr_double_to_int(0.8), llr_double_to_int(1.3)};

  BPResult result{false, 50};
  // Even with osd_weight = 1, multi-cluster composition composes the weight-1
  // improving move in Cluster A (var 2) with the weight-1 improving move in Cluster B (var 5).
  OsdPostProcessor osd1(graph, 10, 1);
  auto obs = osd1.process(result, posteriors, detection_events, hyperedge_observables);
  ASSERT_EQ(obs.size(), 2);
  EXPECT_EQ(obs[0], 1);
  EXPECT_EQ(obs[1], 1);

  OsdPostProcessor osd2(graph, 10, 2);
  auto obs2 = osd2.process(result, posteriors, detection_events, hyperedge_observables);
  ASSERT_EQ(obs2.size(), 2);
  EXPECT_EQ(obs2[0], 1);
  EXPECT_EQ(obs2[1], 1);
}

TEST(OsdPostProcessorTest, GreedyCoordinateDescentRefinement) {
  // Tests 2-pass greedy coordinate-descent refinement on top of OSD-2/3:
  // When a 4-free-column toggle across coupled pivots strictly reduces total
  // error weight beyond OSD-2/3, coordinate descent toggles the remaining free
  // column to reach the minimum-weight codeword.
  size_t num_vars = 8;
  size_t num_checks = 4;

  std::vector<LLR_INT> priors = {
      llr_double_to_int(5.0), llr_double_to_int(5.0), llr_double_to_int(5.0),
      llr_double_to_int(5.0), llr_double_to_int(1.2), llr_double_to_int(1.2),
      llr_double_to_int(1.2), llr_double_to_int(1.2)};

  TannerGraph graph(num_vars, num_checks, priors);
  // Pivot columns 0..3 on checks 0..3 (cost 5.0 each -> OSD-0 weight = 20.0)
  for (size_t i = 0; i < 4; ++i) {
    graph.add_edge(i, i);
  }
  // Free columns 4..7 (cost 1.2 each -> flipping all 4 clears checks 0..3 with weight 4.8)
  graph.add_edge(4, 0);
  graph.add_edge(5, 1);
  graph.add_edge(6, 2);
  graph.add_edge(7, 3);
  graph.build();

  std::vector<uint64_t> detection_events = {0, 1, 2, 3};
  std::vector<std::vector<int>> hyperedge_observables = {
      {}, {}, {}, {}, {0}, {0}, {0}, {1}};

  std::vector<LLR_INT> posteriors = {
      llr_double_to_int(0.1), llr_double_to_int(0.2), llr_double_to_int(0.3),
      llr_double_to_int(0.4), llr_double_to_int(1.0), llr_double_to_int(1.1),
      llr_double_to_int(1.2), llr_double_to_int(1.3)};

  BPResult result{false, 20};
  OsdPostProcessor osd(graph, 10, 2);
  std::vector<size_t> errs;
  auto obs = osd.process(result, posteriors, detection_events, hyperedge_observables, &errs);
  ASSERT_EQ(obs.size(), 2);
  EXPECT_EQ(obs[0], 1);
  EXPECT_EQ(obs[1], 1);
  EXPECT_EQ(errs.size(), 4u);
}

TEST(OsdPostProcessorTest, CostFromErrors) {
  stim::DetectorErrorModel dem(R"DEM(
    error(0.1) D0
    error(0.01) D1
  )DEM");
  BPParams params;
  TesseractBpDecoder decoder(dem, params);
  EXPECT_EQ(decoder.num_variables(), 2u);
  double c0 = decoder.cost_from_errors({0});
  double c1 = decoder.cost_from_errors({1});
  EXPECT_NEAR(c0, std::log((1.0 - 0.1) / 0.1), 1e-2);
  EXPECT_NEAR(c1, std::log((1.0 - 0.01) / 0.01), 1e-2);
  EXPECT_NEAR(decoder.cost_from_errors({0, 1}), c0 + c1, 1e-2);
}

TEST(OsdPostProcessorTest, LowConfidenceShotsPropagation) {
  stim::DetectorErrorModel dem(R"DEM(
    error(0.001) D0 D1 L0
    error(0.001) D1 D2
    error(0.001) D2 D3 L0
    error(0.001) D3 D0
    detector D0
    detector D1
    detector D2
    detector D3
    logical_observable L0
  )DEM");

  BPParams params = BPParams::Osd(0, 10, 1);
  params.fallback_margin = 10.0;
  TesseractBpDecoder decoder(dem, params);
  auto pp = decoder.create_post_processor();

  std::vector<std::vector<uint64_t>> batch = {{0, 2}};
  std::vector<uint8_t> low_conf;
  std::vector<std::vector<size_t>> pred_errors;
  auto res = decoder.decode_batch(batch, pp, nullptr, &pred_errors, nullptr, &low_conf);
  ASSERT_EQ(res.size(), 1u);
  ASSERT_EQ(low_conf.size(), 1u);

  MultiPassBpDecoder mp_decoder(dem, params);
  std::vector<uint8_t> mp_low_conf;
  std::vector<double> mp_costs;
  auto mp_res = mp_decoder.decode_batch(batch, &mp_low_conf, &mp_costs);
  ASSERT_EQ(mp_res.size(), 1u);
  ASSERT_EQ(mp_low_conf.size(), 1u);
  ASSERT_EQ(mp_costs.size(), 1u);
}

}  // namespace bp
