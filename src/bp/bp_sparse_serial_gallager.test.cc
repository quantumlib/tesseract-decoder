#include "bp/bp_sparse_serial_gallager.h"

#include <cmath>
#include <vector>

#include "bp/bp_test_util.h"
#include "bp/check_update.h"
#include "bp/tanner_graph.h"
#include "gtest/gtest.h"

TEST(SparseSerialGallager, RepCode) {
  // Test on rep code, which BP is exact for (since the Tanner graph is a tree)
  double p0 = 0.18;
  double p1 = 0.31;
  double p2 = 0.23;
  bp::TannerGraph<bp::LLR_INT> tg(3, 2,
                                  {bp::prob_double_to_llr_int(p0), bp::prob_double_to_llr_int(p1),
                                   bp::prob_double_to_llr_int(p2)});
  tg.add_edge(0, 0);
  tg.add_edge(1, 0);
  tg.add_edge(1, 1);
  tg.add_edge(2, 1);
  tg.build();

  double eps = 0.005;
  size_t num_iters = 5;

  std::vector<bp::LLR_INT> posteriors(tg.variable_nodes.size());
  bp::GallagerLookupTable lut;

  bp::prepare_tanner_graph_for_bp_sparse_serial_gallager(tg, lut);
  {
    // First test syndrome 11
    std::vector<size_t> dets = {0, 1};
    double p010 = (1 - p0) * p1 * (1 - p2);
    double p101 = p0 * (1 - p1) * p2;
    double phigh = p010 / (p101 + p010);
    bp::LLR_INT llr_high = bp::prob_double_to_llr_int(phigh);
    auto res = bp::bp_sparse_serial_gallager(tg, dets, posteriors, num_iters, lut, true);
    ASSERT_TRUE(res.converged);
    std::cout << std::endl;
    ASSERT_TRUE((posteriors[0] < 0) ^ (posteriors[1] < 0));
    ASSERT_TRUE((posteriors[1] < 0) ^ (posteriors[2] < 0));
    bp::bp_sparse_serial_gallager(tg, dets, posteriors, num_iters, lut, false);
    std::vector<bp::LLR_INT> expected_posteriors = {-llr_high, llr_high, -llr_high};
    bp::assert_llrs_close(posteriors, expected_posteriors, eps);
  }
}

TEST(SparseSerialGallager, SingleCheck) {
  // A tree Tanner graph for which BP is exact. This example
  // is just a single parity check of higher weight, to test
  // a more complex check update in isolation.
  std::vector<double> probs = {0.02, 0.001, 0.99, 0.04, 0.9, 0.0001};
  std::vector<bp::LLR_INT> llrs;
  for (auto& p : probs) {
    llrs.push_back(bp::prob_double_to_llr_int(p));
  }
  bp::TannerGraph<bp::LLR_INT> tg(llrs.size(), 1, llrs);
  for (size_t i = 0; i < llrs.size(); i++) {
    tg.add_edge(i, 0);
  }
  tg.build();

  std::vector<bp::LLR_INT> posteriors(tg.variable_nodes.size());
  bp::GallagerLookupTable lut;

  // Compute the expected posteriors naively using brute-force
  std::vector<double> prob_1_even(probs.size(), 0);
  std::vector<double> prob_0_even(probs.size(), 0);
  std::vector<double> prob_1_odd(probs.size(), 0);
  std::vector<double> prob_0_odd(probs.size(), 0);

  for (size_t i = 0; i < (size_t)(1 << llrs.size()); i++) {
    double p_i = 1;
    bool parity = 0;
    for (size_t j = 0; j < llrs.size(); j++) {
      if ((1 << j) & i) {
        p_i *= probs[j];
        parity ^= 1;
      } else {
        p_i *= (1 - probs[j]);
      }
    }
    for (size_t j = 0; j < llrs.size(); j++) {
      if ((1 << j) & i) {
        if (parity) {
          prob_1_odd[j] += p_i;
        } else {
          prob_1_even[j] += p_i;
        }
      } else {
        if (parity) {
          prob_0_odd[j] += p_i;
        } else {
          prob_0_even[j] += p_i;
        }
      }
    }
  }

  // Sanity check
  for (size_t i = 0; i < probs.size(); i++) {
    EXPECT_NEAR(prob_0_even[i] + prob_0_odd[i] + prob_1_even[i] + prob_1_odd[i], 1.0, 0.00001);
  }

  bp::prepare_tanner_graph_for_bp_sparse_serial_gallager(tg, lut);

  {
    // First test syndrome 1
    std::vector<size_t> dets = {0};
    bp::bp_sparse_serial_gallager(tg, dets, posteriors, 10, lut);
    std::vector<bp::LLR_INT> expected_posteriors;
    for (size_t i = 0; i < probs.size(); i++) {
      expected_posteriors.push_back(
          bp::prob_double_to_llr_int(prob_1_odd[i] / (prob_0_odd[i] + prob_1_odd[i])));
    }
    bp::assert_llrs_close(posteriors, expected_posteriors, 0.1);
  }
}

TEST(SparseSerialGallager, SingleCheckNodePair) {
  bp::TannerGraph<bp::LLR_INT> tg(1, 1, {bp::llr_double_to_int(5.0)});
  tg.add_edge(0, 0);
  tg.build();
  bp::GallagerLookupTable lut;
  bp::prepare_tanner_graph_for_bp_sparse_serial_gallager(tg, lut);
  std::vector<bp::LLR_INT> posteriors(1);
  bp::bp_sparse_serial_gallager(tg, {0}, posteriors, 3, lut);
  EXPECT_NEAR(bp::llr_int_to_double(posteriors[0]),
              5 - bp::llr_int_to_double(bp::LLR_FOR_PROB_ZERO), 0.01);
}
