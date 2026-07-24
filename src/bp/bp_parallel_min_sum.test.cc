#include "bp/bp_parallel_min_sum.h"

#include "bp/bp_test_util.h"
#include "gtest/gtest.h"

namespace bp {

// Define constants for test parameters to improve readability and consistency.
const float kLowErrorProb = 0.1f;
const float kHighErrorProb = 0.9f;
const float kPositivePriorLlr = logf((1 - kLowErrorProb) / kLowErrorProb);
const float kNegativePriorLlr = logf((1 - kHighErrorProb) / kHighErrorProb);
const float kNormalizationFactor = 0.875f;

TEST(BpParallelMinSumTest, ConvergesToZeroSyndromeDespiteIncorrectPrior) {
  // Create a simple 5x4 Tanner graph for a distance-5 repetition code.
  TannerGraph<float> graph(5, 4, {0, 0, 0, 0, 0});
  graph.add_edge(0, 0);
  graph.add_edge(1, 0);
  graph.add_edge(1, 1);
  graph.add_edge(2, 1);
  graph.add_edge(2, 2);
  graph.add_edge(3, 2);
  graph.add_edge(3, 3);
  graph.add_edge(4, 3);
  graph.build();

  // Assign priors. Assume V2 is flipped with high probability.
  graph.variable_nodes[0].prior = kPositivePriorLlr;
  graph.variable_nodes[1].prior = kPositivePriorLlr;
  graph.variable_nodes[2].prior = kNegativePriorLlr;  // Strong prior belief of a flip.
  graph.variable_nodes[3].prior = kPositivePriorLlr;
  graph.variable_nodes[4].prior = kPositivePriorLlr;

  // This test checks if the decoder trusts the syndrome over the prior.
  // The prior suggests an error at V2, but the syndrome is empty (no detection events).
  // The decoder should find that the most likely state is the all-zeros codeword,
  // as this perfectly matches the zero syndrome.
  std::vector<size_t> detection_events = {};
  std::vector<float> posteriors(5);

  BPResult result = bp_parallel_min_sum(graph, detection_events, posteriors, 20, 0.5f, true);

  EXPECT_TRUE(result.converged);
  // Expect the decoder to conclude no error occurred, resulting in the all-zeros codeword.
  // A decoded bit of 0 corresponds to a positive LLR.
  EXPECT_GT(posteriors[0], 0);
  EXPECT_GT(posteriors[1], 0);
  EXPECT_GT(posteriors[2], 0);
  EXPECT_GT(posteriors[3], 0);
  EXPECT_GT(posteriors[4], 0);
}

TEST(BpParallelMinSumTest, CorrectsSingleErrorWithSyndrome) {
  // Create a simple 5x4 Tanner graph for a distance-5 repetition code.
  TannerGraph<float> graph(5, 4, {0, 0, 0, 0, 0});
  graph.add_edge(0, 0);
  graph.add_edge(1, 0);
  graph.add_edge(1, 1);
  graph.add_edge(2, 1);
  graph.add_edge(2, 2);
  graph.add_edge(3, 2);
  graph.add_edge(3, 3);
  graph.add_edge(4, 3);
  graph.build();

  // Assign a uniform low-error prior to all variable nodes.
  for (size_t i = 0; i < 5; ++i) {
    graph.variable_nodes[i].prior = kPositivePriorLlr;
  }

  // Introduce a syndrome that corresponds to a single flip at V2.
  // A flip at V2 violates checks C1 and C2.
  std::vector<size_t> detection_events = {1, 2};
  std::vector<float> posteriors(5);

  BPResult result =
      bp_parallel_min_sum(graph, detection_events, posteriors, 20, kNormalizationFactor, true);

  EXPECT_TRUE(result.converged);
  // Expect the decoder to identify that V2 is the most likely error.
  // The posterior LLR for V2 should be negative, and all others positive.
  EXPECT_GT(posteriors[0], 0);
  EXPECT_GT(posteriors[1], 0);
  EXPECT_LT(posteriors[2], 0);
  EXPECT_GT(posteriors[3], 0);
  EXPECT_GT(posteriors[4], 0);
}

TEST(BpParallelMinSumTest, CorrectsMultipleSpacedErrors) {
  // Use the same d=5 repetition code graph.
  TannerGraph<float> graph(5, 4, {0, 0, 0, 0, 0});
  graph.add_edge(0, 0);
  graph.add_edge(1, 0);
  graph.add_edge(1, 1);
  graph.add_edge(2, 1);
  graph.add_edge(2, 2);
  graph.add_edge(3, 2);
  graph.add_edge(3, 3);
  graph.add_edge(4, 3);
  graph.build();

  // Assign a uniform low-error prior to all variable nodes.
  for (size_t i = 0; i < 5; ++i) {
    graph.variable_nodes[i].prior = kPositivePriorLlr;
  }

  // Introduce a syndrome for two spatially separated errors: V0 and V4.
  // A flip at V0 violates C0. A flip at V4 violates C3.
  std::vector<size_t> detection_events = {0, 3};
  std::vector<float> posteriors(5);

  BPResult result =
      bp_parallel_min_sum(graph, detection_events, posteriors, 10, kNormalizationFactor, true);

  EXPECT_TRUE(result.converged);
  // Expect the decoder to identify both V0 and V4 as errors.
  EXPECT_LT(posteriors[0], 0);
  EXPECT_GT(posteriors[1], 0);
  EXPECT_GT(posteriors[2], 0);
  EXPECT_GT(posteriors[3], 0);
  EXPECT_LT(posteriors[4], 0);
}

TEST(BpParallelMinSumTest, ReportsNonConvergenceOnCyclicGraph) {
  // Create a Tanner graph with a short 4-cycle (a square).
  // V0 -- C0 -- V1
  // |           |
  // C3          C1
  // |           |
  // V3 -- C2 -- V2
  TannerGraph<float> graph(4, 4, {0, 0, 0, 0});
  graph.add_edge(0, 0);
  graph.add_edge(1, 0);
  graph.add_edge(1, 1);
  graph.add_edge(2, 1);
  graph.add_edge(2, 2);
  graph.add_edge(3, 2);
  graph.add_edge(3, 3);
  graph.add_edge(0, 3);
  graph.build();

  // Assign unbiased priors (LLR = 0, p=0.5) for maximum ambiguity.
  for (size_t i = 0; i < 4; ++i) {
    graph.variable_nodes[i].prior = 0.0f;
  }

  // Introduce an ambiguous syndrome. C0 and C2 are triggered.
  // This could be caused by errors on V0 and V2, OR on V1 and V3.
  // The decoder has no basis to prefer one over the other and should oscillate.
  std::vector<size_t> detection_events = {0, 2};
  std::vector<float> posteriors(4);

  // Run for enough iterations to see oscillation, but expect it to fail.
  BPResult result =
      bp_parallel_min_sum(graph, detection_events, posteriors, 1000, kNormalizationFactor, true);

  // On this ambiguous graph, BP is not expected to converge to a valid solution.
  EXPECT_FALSE(result.converged);
}

}  // namespace bp
