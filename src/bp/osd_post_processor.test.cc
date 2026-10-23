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

}  // namespace bp
