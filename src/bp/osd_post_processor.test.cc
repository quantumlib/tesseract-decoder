#include "bp/osd_post_processor.h"

#include <gtest/gtest.h>

#include "bp/check_update.h"  // For llr_double_to_int

namespace bp {

TEST(OsdPostProcessorTest, BasicOSD0) {
  size_t num_vars = 3;
  size_t num_checks = 2;

  // E0 (touches D0)
  // E1 (touches D0, D1)
  // E2 (touches D1)
  std::vector<LLR_INT> priors = {
      llr_double_to_int(5.0),   // Reliable 0 (high positive LLR)
      llr_double_to_int(0.5),   // Unreliable 0
      llr_double_to_int(-10.0)  // Reliable 1 (high negative LLR)
  };

  TannerGraph<LLR_INT> graph(num_vars, num_checks, priors);
  graph.add_edge(0, 0);
  graph.add_edge(1, 0);
  graph.add_edge(1, 1);
  graph.add_edge(2, 1);
  graph.build();

  std::vector<uint64_t> detection_events = {0};                           // Syndrome D0=1, D1=0
  std::vector<std::vector<int>> hyperedge_observables = {{0}, {0}, {0}};  // All flip L0

  OsdPostProcessor osd(graph, 0, 0);  // osd_order=0, osd_weight=0
  BPResult result{false, 10};         // BP did not converge

  std::vector<LLR_INT> posteriors = priors;  // Let's use priors as posteriors for OSD-0 test
  auto obs_correction = osd.process(result, posteriors, detection_events, hyperedge_observables);

  ASSERT_EQ(obs_correction.size(), 1);
  EXPECT_EQ(obs_correction[0], 0);
}

TEST(OsdPostProcessorTest, OSD1Improvement) {
  size_t num_vars = 3;
  size_t num_checks = 2;

  // E0 (touches D0)
  // E1 (touches D1)
  // E2 (touches D0, D1)
  std::vector<LLR_INT> priors = {llr_double_to_int(1.0), llr_double_to_int(1.0),
                                 llr_double_to_int(1.5)};

  TannerGraph<LLR_INT> graph(num_vars, num_checks, priors);
  graph.add_edge(0, 0);
  graph.add_edge(1, 1);
  graph.add_edge(2, 0);
  graph.add_edge(2, 1);
  graph.build();

  std::vector<uint64_t> detection_events = {0, 1};  // Syndrome D0=1, D1=1
  std::vector<std::vector<int>> hyperedge_observables = {{0}, {0}, {0}};

  BPResult result{false, 10};
  std::vector<LLR_INT> posteriors = priors;

  // OSD-0 should pick E0 and E1 to satisfy D0, D1. Cost = 1.0 + 1.0 = 2.0.
  OsdPostProcessor osd0(graph, 10, 0);
  auto obs0 = osd0.process(result, posteriors, detection_events, hyperedge_observables);
  ASSERT_EQ(obs0.size(), 1);
  EXPECT_EQ(obs0[0], 0);

  // OSD-1 should consider flipping E2 (the free variable).
  // E2 satisfies both D0, D1. Cost = 1.5.
  // 1.5 < 2.0, so it should prefer E2.
  OsdPostProcessor osd1(graph, 10, 1);
  auto obs1 = osd1.process(result, posteriors, detection_events, hyperedge_observables);
  ASSERT_EQ(obs1.size(), 1);
  EXPECT_EQ(obs1[0], 1);
}

}  // namespace bp
