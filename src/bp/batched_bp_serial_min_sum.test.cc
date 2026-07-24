#include "bp/batched_bp_serial_min_sum.h"

#include "bp/bp_test_util.h"
#include "gtest/gtest.h"

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

  std::vector<std::vector<float>> batched_posteriors(BP_BATCH_SIZE, std::vector<float>(5, 0));

  auto batched_results = batched_bp_serial_min_sum(
      batched_graph, batched_syndromes, batched_posteriors, 20, kNormalizationFactor, true);

  // Shot 0 should converge to fixing V2
  EXPECT_TRUE(batched_results[0].converged);
  EXPECT_GT(batched_posteriors[0][0], 0);
  EXPECT_GT(batched_posteriors[0][1], 0);
  EXPECT_LT(batched_posteriors[0][2], 0);  // V2 is negative (error)
  EXPECT_GT(batched_posteriors[0][3], 0);
  EXPECT_GT(batched_posteriors[0][4], 0);

  // Shot 1 should converge to fixing V0, V4
  EXPECT_TRUE(batched_results[1].converged);
  EXPECT_LT(batched_posteriors[1][0], 0);  // V0 is negative
  EXPECT_GT(batched_posteriors[1][1], 0);
  EXPECT_GT(batched_posteriors[1][2], 0);
  EXPECT_GT(batched_posteriors[1][3], 0);
  EXPECT_LT(batched_posteriors[1][4], 0);  // V4 is negative
}

}  // namespace bp