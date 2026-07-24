#include "bp/bp_serial_min_sum.h"

#include "bp/bp_test_util.h"
#include "gtest/gtest.h"

namespace bp {

const float kLowErrorProb = 0.1f;
const float kHighErrorProb = 0.9f;
const float kPositivePriorLlr = logf((1 - kLowErrorProb) / kLowErrorProb);
const float kNegativePriorLlr = logf((1 - kHighErrorProb) / kHighErrorProb);
const float kNormalizationFactor = 0.875f;

TEST(BpSerialMinSumTest, ConvergesToZeroSyndromeDespiteIncorrectPrior) {
  TannerGraph<float> graph(5, 4,
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

  std::vector<size_t> detection_events = {};
  std::vector<float> posteriors(5);

  BPResult result =
      bp_serial_min_sum(graph, detection_events, posteriors, 20, kNormalizationFactor, true);

  EXPECT_TRUE(result.converged);
  EXPECT_GT(posteriors[0], 0);
  EXPECT_GT(posteriors[1], 0);
  EXPECT_GT(posteriors[2], 0);
  EXPECT_GT(posteriors[3], 0);
  EXPECT_GT(posteriors[4], 0);
}

TEST(BpSerialMinSumTest, CorrectsSingleErrorWithSyndrome) {
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

  std::vector<size_t> detection_events = {1, 2};
  std::vector<float> posteriors(5);

  BPResult result =
      bp_serial_min_sum(graph, detection_events, posteriors, 20, kNormalizationFactor, true);

  EXPECT_TRUE(result.converged);
  EXPECT_GT(posteriors[0], 0);
  EXPECT_GT(posteriors[1], 0);
  EXPECT_LT(posteriors[2], 0);
  EXPECT_GT(posteriors[3], 0);
  EXPECT_GT(posteriors[4], 0);
}

}  // namespace bp