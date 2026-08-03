#include "error_correlations.h"

#include <vector>

#include "gtest/gtest.h"

using namespace tesseract;

TEST(TwoPassCorrelationsTest, JointProbabilities) {
  stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0 ^ D1
        error(0.2) D0
    )DEM");

  std::vector<int> global_det_to_comp_id = {0, 1};
  auto joint = get_hyperedge_joint_probabilities(dem, global_det_to_comp_id);

  Hyperedge h0 = {0};
  Hyperedge h1 = {1};

  // P(D0) = 0.1 XOR 0.2 = 0.1*(1-0.2) + 0.2*(1-0.1) = 0.08 + 0.18 = 0.26
  EXPECT_NEAR(joint[h0][h0], 0.26, 1e-6);
  // P(D1) = 0.1
  EXPECT_NEAR(joint[h1][h1], 0.1, 1e-6);
  // P(D0 and D1) = 0.1
  EXPECT_NEAR(joint[h0][h1], 0.1, 1e-6);
  EXPECT_NEAR(joint[h1][h0], 0.1, 1e-6);
}

TEST(TwoPassCorrelationsTest, ImpliedProbabilities) {
  JointProbsMap joint;
  Hyperedge h0 = {0};
  Hyperedge h1 = {1};

  joint[h0][h0] = 0.2;
  joint[h1][h1] = 0.1;
  joint[h0][h1] = 0.05;
  joint[h1][h0] = 0.05;

  auto implied = get_implied_hyperedge_probabilities(joint);

  // P(D1 | D0) = 0.05 / 0.2 = 0.25
  bool found = false;
  for (const auto& imp : implied[h0]) {
    if (imp.affected_hyperedge == h1) {
      EXPECT_NEAR(imp.probability, 0.25, 1e-6);
      found = true;
    }
  }
  EXPECT_TRUE(found);

  // P(D0 | D1) = 0.05 / 0.1 = 0.5
  found = false;
  for (const auto& imp : implied[h1]) {
    if (imp.affected_hyperedge == h0) {
      EXPECT_NEAR(imp.probability, 0.5, 1e-6);
      found = true;
    }
  }
  EXPECT_TRUE(found);
}
