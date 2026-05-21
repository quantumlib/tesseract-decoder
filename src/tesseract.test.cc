#include "tesseract.h"

#include <gtest/gtest.h>

#include "stim.h"

namespace {

using namespace common;

TEST(tesseract, DecodeToErrorsCorrectness_SimpleGrid) {
  stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0 D1
        error(0.1) D1 D2
        error(0.1) D3 D4
        error(0.1) D0 D3
        detector(0, 0, 0) D0
        detector(1, 0, 0) D1
        detector(2, 0, 0) D2
        detector(3, 0, 0) D3
        detector(4, 0, 0) D4
    )DEM");

  TesseractConfig config{dem};
  config.merge_errors = false;
  TesseractDecoder decoder(config);

  // Case 1: Detectors D0, D1 fire. Should pick error 0.
  std::vector<uint64_t> detections = {0, 1};
  decoder.decode_to_errors(detections);
  std::vector<size_t> expected_errors = {0};
  EXPECT_EQ(decoder.predicted_errors_buffer, expected_errors);

  // Case 2: Detectors D0, D3 fire. Should pick error 3.
  detections = {0, 3};
  decoder.decode_to_errors(detections);
  expected_errors = {3};
  EXPECT_EQ(decoder.predicted_errors_buffer, expected_errors);

  // Case 3: Detectors D1, D2 fire. Should pick error 1.
  detections = {1, 2};
  decoder.decode_to_errors(detections);
  expected_errors = {1};
  EXPECT_EQ(decoder.predicted_errors_buffer, expected_errors);

  // Case 4: Detectors D3, D4 fire. Should pick error 2.
  detections = {3, 4};
  decoder.decode_to_errors(detections);
  expected_errors = {2};
  EXPECT_EQ(decoder.predicted_errors_buffer, expected_errors);

  // Case 5: All detectors fire.
  detections = {0, 1, 2, 3, 4};
  decoder.decode_to_errors(detections);
  // Optimal errors for this syndrome could be {0, 1, 2, 3} or similar.
  // We just check that the sum of costs is minimized.
  double total_cost = 0;
  for (size_t ei : decoder.predicted_errors_buffer) {
    total_cost += decoder.errors[ei].likelihood_cost;
  }
  EXPECT_LT(total_cost, 0.5); // 4 * -log(0.1) is roughly 9.2, so cost should be low.
}

TEST(tesseract, EneighborsCorrectness_SimpleGrid) {
  stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0 D1
        error(0.1) D1 D2
        error(0.1) D3 D4
        error(0.1) D0 D3
        detector(0, 0, 0) D0
        detector(1, 0, 0) D1
        detector(2, 0, 0) D2
        detector(3, 0, 0) D3
        detector(4, 0, 0) D4
    )DEM");

  TesseractConfig t_config{dem};
  t_config.merge_errors = false;
  TesseractDecoder t_dec(t_config);

  // Expected neighbors
  // e0 (D0,D1) neighbors are D2,D3
  std::vector<int> expected_e0_neighbors = {2, 3};
  // e1 (D1,D2) neighbors are D0
  std::vector<int> expected_e1_neighbors = {0};
  // e2 (D3,D4) neighbors are D0
  std::vector<int> expected_e2_neighbors = {0};
  // e3 (D0,D3) neighbors are D1,D4
  std::vector<int> expected_e3_neighbors = {1, 4};
  // e4 (D1,D4) neighbors are D0,D3
  // Wait, there is no e4. e3 is (D0,D3).

  // Sort the actual vectors for reliable comparison
  for (size_t i = 0; i < t_dec.get_eneighbors().size(); ++i) {
    std::sort(t_dec.get_eneighbors()[i].begin(), t_dec.get_eneighbors()[i].end());
  }

  EXPECT_EQ(t_dec.get_eneighbors()[0], expected_e0_neighbors);
  EXPECT_EQ(t_dec.get_eneighbors()[1], expected_e1_neighbors);
  EXPECT_EQ(t_dec.get_eneighbors()[2], expected_e2_neighbors);
  EXPECT_EQ(t_dec.get_eneighbors()[3], expected_e3_neighbors);
}

TEST(tesseract, EneighborsCorrectness_Line) {
  stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0 D1
        error(0.1) D1 D2
        error(0.1) D2 D3
        error(0.1) D3 D4
        error(0.1) D4 D5
        detector(0, 0, 0) D0
        detector(1, 0, 0) D1
        detector(2, 0, 0) D2
        detector(3, 0, 0) D3
        detector(4, 0, 0) D4
        detector(5, 0, 0) D5
    )DEM");

  TesseractConfig t_config{dem};
  t_config.merge_errors = false;
  TesseractDecoder t_dec(t_config);

  // Expected neighbors
  // e0 (D0,D1) neighbors are D2
  std::vector<int> expected_e0_neighbors = {2};
  // e1 (D1,D2) neighbors are D0,D3
  std::vector<int> expected_e1_neighbors = {0, 3};
  // e2 (D2,D3) neighbors are D1,D4
  std::vector<int> expected_e2_neighbors = {1, 4};
  // e3 (D3,D4) neighbors are D2,D5
  std::vector<int> expected_e3_neighbors = {2, 5};
  // e4 (D4,D5) neighbors are D3
  std::vector<int> expected_e4_neighbors = {3};

  // Sort the actual vectors for reliable comparison
  for (size_t i = 0; i < t_dec.get_eneighbors().size(); ++i) {
    std::sort(t_dec.get_eneighbors()[i].begin(), t_dec.get_eneighbors()[i].end());
  }

  EXPECT_EQ(t_dec.get_eneighbors()[0], expected_e0_neighbors);
  EXPECT_EQ(t_dec.get_eneighbors()[1], expected_e1_neighbors);
  EXPECT_EQ(t_dec.get_eneighbors()[2], expected_e2_neighbors);
  EXPECT_EQ(t_dec.get_eneighbors()[3], expected_e3_neighbors);
  EXPECT_EQ(t_dec.get_eneighbors()[4], expected_e4_neighbors);
}

TEST(tesseract, EneighborsCorrectness_ComplexGrid) {
  stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0 D1
        error(0.1) D1 D2
        error(0.1) D3 D4
        error(0.1) D4 D5
        error(0.1) D6 D7
        error(0.1) D7 D8
        error(0.1) D1 D4 D7
        error(0.1) D0 D3 D6
        detector(0, 0, 0) D0
        detector(1, 0, 0) D1
        detector(2, 0, 0) D2
        detector(3, 0, 0) D3
        detector(4, 0, 0) D4
        detector(5, 0, 0) D5
        detector(6, 0, 0) D6
        detector(7, 0, 0) D7
        detector(8, 0, 0) D8
    )DEM");

  TesseractConfig t_config{dem};
  t_config.merge_errors = false;
  TesseractDecoder t_dec(t_config);

  // Expected neighbors
  // e0 (D0,D1) neighbors are D2,D3,D4,D6,D7
  std::vector<int> expected_e0_neighbors = {2, 3, 4, 6, 7};
  // e1 (D1,D2) neighbors are D0,D4,D7
  std::vector<int> expected_e1_neighbors = {0, 4, 7};
  // e2 (D3,D4) neighbors are D0,D1,D5,D6,D7
  std::vector<int> expected_e2_neighbors = {0, 1, 5, 6, 7};
  // e3 (D4,D5) neighbors are D1,D3,D7
  std::vector<int> expected_e3_neighbors = {1, 3, 7};
  // e4 (D6,D7) neighbors are D0,D1,D3,D4,D8
  std::vector<int> expected_e4_neighbors = {0, 1, 3, 4, 8};
  // e5 (D7,D8) neighbors are D1,D4,D6
  std::vector<int> expected_e5_neighbors = {1, 4, 6};
  // e6 (D1,D4,D7) neighbors are D0,2,3,5,6,8
  std::vector<int> expected_e6_neighbors = {0, 2, 3, 5, 6, 8};
  // e7 (D0,D3,D6) neighbors are D1,4,7
  std::vector<int> expected_e7_neighbors = {1, 4, 7};

  // Sort the actual vectors for reliable comparison
  for (size_t i = 0; i < t_dec.get_eneighbors().size(); ++i) {
    std::sort(t_dec.get_eneighbors()[i].begin(), t_dec.get_eneighbors()[i].end());
  }

  EXPECT_EQ(t_dec.get_eneighbors()[0], expected_e0_neighbors);
  EXPECT_EQ(t_dec.get_eneighbors()[1], expected_e1_neighbors);
  EXPECT_EQ(t_dec.get_eneighbors()[2], expected_e2_neighbors);
  EXPECT_EQ(t_dec.get_eneighbors()[3], expected_e3_neighbors);
  EXPECT_EQ(t_dec.get_eneighbors()[4], expected_e4_neighbors);
  EXPECT_EQ(t_dec.get_eneighbors()[5], expected_e5_neighbors);
  EXPECT_EQ(t_dec.get_eneighbors()[6], expected_e6_neighbors);
  EXPECT_EQ(t_dec.get_eneighbors()[7], expected_e7_neighbors);
}

TEST(tesseract, DecodeToErrorsThrowsOnInvalidSymptom) {
  stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0 D1
        error(0.1) D1 D2
        error(0.1) D2 D3
        detector(0, 0, 0) D0
        detector(1, 0, 0) D1
        detector(2, 0, 0) D2
    )DEM");

  TesseractConfig config{dem};
  TesseractDecoder decoder(config);

  uint64_t invalid_symptom = decoder.num_detectors;

  try {
    decoder.decode_to_errors({invalid_symptom});
  } catch (const std::runtime_error& err) {
    EXPECT_EQ("Symptom " + std::to_string(invalid_symptom) +
                  " references a detector >= num_detectors (= " +
                  std::to_string(decoder.num_detectors) + ").",
              err.what());
  }
}

TEST(TesseractDetcostTest, ComparesRatiosNotRawCosts) {
  stim::DetectorErrorModel dem = stim::DetectorErrorModel(R"DEM(
    error(0.005322067133022559) D0 D1 D3
    error(0.0051237598826648) D0 D1 D2
  )DEM");

  TesseractConfig cfg;
  cfg.dem = dem;
  cfg.merge_errors = false;
  TesseractDecoder dec(cfg);

  std::vector<DetectorCostTuple> tuples(dec.errors.size());
  // residual x = {D0, D1}
  std::cout << "dec.d2e.size() = " << dec.d2e.size() << std::endl;
  for (int ei : dec.d2e[0]) tuples[ei].detectors_count++;
  for (int ei : dec.d2e[1]) tuples[ei].detectors_count++;

  double got = dec.get_detcost(0, tuples);
  double expected = 5.230557212477344 / 2.0;  // from D0 D1 D3

  EXPECT_NEAR(got, expected, 1e-12);
}

// Test to ensure update_internal_costs correctly reflects changes to error likelihoods
TEST(tesseract, UpdateInternalCostsBehavior) {
  // Define a simple DEM with two errors that can explain detector D0
  // Error 0: D0 (prob 0.2) -> likelihood_cost: ~1.386
  // Error 1: D0 (prob 0.1) -> likelihood_cost: ~2.197
  // Initially, Error 0 is more likely (lower likelihood_cost)
  stim::DetectorErrorModel dem(R"DEM(
        error(0.2) D0
        error(0.1) D0
        detector(0,0,0) D0
    )DEM");

  TesseractConfig config{dem};
  config.merge_errors = false; // Important: do not merge errors for this test
  TesseractDecoder decoder(config);

  // Initial decode: D0 fires. Should pick Error 0 (index 0) as it's more likely.
  std::vector<uint64_t> detections = {0};
  decoder.decode_to_errors(detections);
  ASSERT_EQ(decoder.predicted_errors_buffer.size(), 1);
  ASSERT_EQ(decoder.predicted_errors_buffer[0], 0); // Should pick Error 0 (index 0)

  // Manually change the likelihood_cost of Error 1 to be lower (more likely) than Error 0
  // Original: Error 0 (prob 0.2, cost ~1.386), Error 1 (prob 0.1, cost ~2.197)
  // Modify: Error 1 to prob 0.3 (cost ~0.847). Now Error 1 is more likely.
  decoder.errors[1].set_with_probability(0.3);

  // Call update_internal_costs to re-synchronize the decoder's state
  decoder.update_internal_costs({1});

  // Decode again with the same detections.
  // Now, D0 fires. It should pick Error 1 (index 1) as it's now more likely.
  decoder.decode_to_errors(detections);
  ASSERT_EQ(decoder.predicted_errors_buffer.size(), 1);
  ASSERT_EQ(decoder.predicted_errors_buffer[0], 1); // Should now pick Error 1 (index 1)
}

} // namespace
