#include "bp/osd_post_processor.h"

#include <gtest/gtest.h>

#include "bp/bp_params.h"

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

}  // namespace bp
