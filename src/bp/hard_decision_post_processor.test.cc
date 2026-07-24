#include "bp/hard_decision_post_processor.h"

#include <gtest/gtest.h>

namespace bp {

TEST(HardDecisionPostProcessorTest, EmptyInputs) {
  HardDecisionPostProcessor pp;
  BPResult result{false, 0};
  std::vector<LLR_INT> posteriors;
  std::vector<uint64_t> detection_events;
  std::vector<std::vector<int>> hyperedge_observables;

  auto correction = pp.process(result, posteriors, detection_events, hyperedge_observables);
  EXPECT_TRUE(correction.empty());
}

TEST(HardDecisionPostProcessorTest, NoErrors) {
  HardDecisionPostProcessor pp;
  BPResult result{false, 0};
  std::vector<LLR_INT> posteriors = {10, 20, 30};
  std::vector<uint64_t> detection_events;
  std::vector<std::vector<int>> hyperedge_observables = {{0}, {1}, {0, 1}};

  auto correction = pp.process(result, posteriors, detection_events, hyperedge_observables);

  EXPECT_EQ(correction.size(), 2);
  EXPECT_EQ(correction[0], 0);
  EXPECT_EQ(correction[1], 0);
}

TEST(HardDecisionPostProcessorTest, SingleError) {
  HardDecisionPostProcessor pp;
  BPResult result{false, 0};
  std::vector<LLR_INT> posteriors = {-10, 20, 30};
  std::vector<uint64_t> detection_events;
  std::vector<std::vector<int>> hyperedge_observables = {{0}, {1}, {0, 1}};

  auto correction = pp.process(result, posteriors, detection_events, hyperedge_observables);

  EXPECT_EQ(correction.size(), 2);
  EXPECT_EQ(correction[0], 1);
  EXPECT_EQ(correction[1], 0);
}

TEST(HardDecisionPostProcessorTest, MultipleErrorsAndOverlaps) {
  HardDecisionPostProcessor pp;
  BPResult result{false, 0};
  std::vector<LLR_INT> posteriors = {-10, -20, 30};
  std::vector<uint64_t> detection_events;
  std::vector<std::vector<int>> hyperedge_observables = {{0}, {0, 1}, {1}};

  auto correction = pp.process(result, posteriors, detection_events, hyperedge_observables);

  EXPECT_EQ(correction.size(), 2);
  EXPECT_EQ(correction[0], 0);  // 1 ^ 1 = 0
  EXPECT_EQ(correction[1], 1);  // 0 ^ 1 = 1
}

TEST(HardDecisionPostProcessorTest, IgnoresPositivePosteriors) {
  HardDecisionPostProcessor pp;
  BPResult result{false, 0};
  std::vector<LLR_INT> posteriors = {10, -20, 30};
  std::vector<uint64_t> detection_events;
  std::vector<std::vector<int>> hyperedge_observables = {{0}, {1}, {0, 1}};

  auto correction = pp.process(result, posteriors, detection_events, hyperedge_observables);

  EXPECT_EQ(correction.size(), 2);
  EXPECT_EQ(correction[0], 0);
  EXPECT_EQ(correction[1], 1);
}

}  // namespace bp
