#include "bp/batched_bp_parallel_min_sum.h"

#include "bp/bp_parallel_min_sum.h"
#include "bp/bp_test_util.h"
#include "gtest/gtest.h"

namespace bp {

const float kLowErrorProb = 0.1f;
const float kHighErrorProb = 0.9f;
const float kPositivePriorLlr = logf((1 - kLowErrorProb) / kLowErrorProb);
const float kNegativePriorLlr = logf((1 - kHighErrorProb) / kHighErrorProb);
const float kNormalizationFactor = 0.875f;

TEST(BatchedBpParallelMinSumTest, MatchesUnbatchedImplementation) {
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

  auto batched_results = batched_bp_parallel_min_sum(
      batched_graph, batched_syndromes, batched_posteriors, 20, kNormalizationFactor, true);

  for (size_t b = 0; b < 2; ++b) {
    std::vector<float> unbatched_posteriors(5, 0);
    TannerGraph<float> unbatched_copy = graph;
    auto unbatched_result = bp_parallel_min_sum(
        unbatched_copy, batched_syndromes[b], unbatched_posteriors, 20, kNormalizationFactor, true);

    EXPECT_EQ(batched_results[b].converged, unbatched_result.converged)
        << "Shot " << b << " convergence mismatch.";
    EXPECT_EQ(batched_results[b].num_iters, unbatched_result.num_iters)
        << "Shot " << b << " iter mismatch.";

    for (size_t i = 0; i < 5; ++i) {
      EXPECT_FLOAT_EQ(batched_posteriors[b][i], unbatched_posteriors[i])
          << "Shot " << b << " variable " << i << " mismatch.";
    }
  }
}

}  // namespace bp
