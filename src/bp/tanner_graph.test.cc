#include "bp/tanner_graph.h"

#include "gtest/gtest.h"

TEST(TannerGraph, ConstructTannerGraph) {
  std::vector<bp::LLR_INT> priors = {5, 6, 7};
  bp::TannerGraph<bp::LLR_INT> tg(3, 2, priors);
  tg.add_edge(0, 0);
  tg.add_edge(1, 0);
  tg.add_edge(1, 1);
  tg.add_edge(2, 1);

  tg.build();

  ASSERT_EQ(tg.variable_nodes.size(), 3);
  ASSERT_EQ(tg.check_nodes.size(), 2);
  ASSERT_EQ(tg.count_edges(), 4);

  ASSERT_EQ(tg.var_edges[tg.var_edge_offsets[0] + 0], 0);
  ASSERT_EQ(tg.var_edges[tg.var_edge_offsets[1] + 0], 0);
  ASSERT_EQ(tg.var_edges[tg.var_edge_offsets[1] + 1], 1);
  ASSERT_EQ(tg.var_edges[tg.var_edge_offsets[2] + 0], 1);

  ASSERT_EQ(tg.check_edges[tg.check_edge_offsets[0] + 0], 0);
  ASSERT_EQ(tg.check_edges[tg.check_edge_offsets[0] + 1], 1);
  ASSERT_EQ(tg.check_edges[tg.check_edge_offsets[1] + 0], 1);
  ASSERT_EQ(tg.check_edges[tg.check_edge_offsets[1] + 1], 2);

  for (size_t i = 0; i < tg.variable_nodes.size(); i++) {
    ASSERT_EQ(tg.variable_nodes[i].prior, priors[i]);
  }
}
