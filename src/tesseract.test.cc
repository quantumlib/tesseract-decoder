// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tesseract.h"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <queue>
#include <vector>

#include "gtest/gtest.h"
#include "simplex.h"
#include "stim.h"
#include "utils.h"

namespace tesseract_decoder {
namespace {

constexpr uint64_t test_data_seed = 752024;

bool simplex_test_compare(stim::DetectorErrorModel& dem, std::vector<stim::SparseShot>& shots) {
  TesseractConfig tesseract_config{dem};
  TesseractDecoder tesseract_decoder(tesseract_config);

  SimplexConfig simplex_config{dem};
  SimplexDecoder simplex_decoder(simplex_config);

  for (size_t shot = 0; shot < shots.size(); shot++) {
    tesseract_decoder.decode_to_errors(shots[shot].hits);
    double tesseract_cost =
        tesseract_decoder.cost_from_errors(tesseract_decoder.predicted_errors_buffer);

    if (tesseract_decoder.low_confidence_flag) {
      // Simplex c++ does not yet support undecodable shots -- i.e. detection
      // event configurations with no error solution.
      std::cout << "not decoding shot " << shot
                << " with simplex because Tesseract found no solution" << std::endl;
      continue;
    }

    simplex_decoder.decode_to_errors(shots[shot].hits);
    double simplex_cost = simplex_decoder.cost_from_errors(simplex_decoder.predicted_errors_buffer);

    // If there is a mismatch in weights, print diagnostic information
    if (std::abs(tesseract_cost - simplex_cost) > EPSILON) {
      std::cout << "shot " << shot << " ";
      for (size_t d : shots[shot].hits) {
        std::cout << "D" << d << " ";
      }
      std::cout << std::endl;
      std::cout << "Error: For shot " << shot
                << " tesseract got solution with cost:" << tesseract_cost
                << " simplex got solution with cost: " << simplex_cost << std::endl;
      std::cout << "tesseract used errors ";
      for (size_t dem_ei : tesseract_decoder.predicted_errors_buffer) {
        std::cout << dem_ei << ", ";
      }
      std::cout << std::endl;
      std::cout << " and had cost " << tesseract_cost << std::endl;
      std::cout << "simplex used errors ";
      for (size_t dem_ei : simplex_decoder.predicted_errors_buffer) {
        std::cout << dem_ei << ", ";
      }
      std::cout << std::endl;
      std::cout << " and had cost " << simplex_cost << std::endl;
      return false;
    }
  }
  return true;
}

TEST(tesseract, Tesseract_simplex_test) {
  bool long_tests = std::getenv("TESSERACT_LONG_TESTS") != nullptr;
  auto p_errs =
      long_tests ? std::vector<float>{0.001f, 0.003f, 0.005f} : std::vector<float>{0.003f};
  auto distances = long_tests ? std::vector<size_t>{3, 5, 7} : std::vector<size_t>{3};
  auto rounds = long_tests ? std::vector<size_t>{2, 5, 10} : std::vector<size_t>{2};
  size_t base_shots = long_tests ? 1000 : 100;

  for (float p_err : p_errs) {
    for (size_t distance : distances) {
      for (const size_t num_rounds : rounds) {
        const size_t num_shots = base_shots / num_rounds / distance;
        std::cout << "p_err = " << p_err << " distance = " << distance
                  << " num_rounds = " << num_rounds << " num_shots = " << num_shots << std::endl;
        stim::CircuitGenParameters params(num_rounds, /*distance=*/distance,
                                          /*task=*/"rotated_memory_x");
        params.after_clifford_depolarization = p_err;
        params.before_round_data_depolarization = p_err;
        params.before_measure_flip_probability = p_err;
        params.after_reset_flip_probability = p_err;
        stim::Circuit circuit = stim::generate_surface_code_circuit(params).circuit;
        stim::DetectorErrorModel dem = stim::ErrorAnalyzer::circuit_to_detector_error_model(
            circuit, /*decompose_errors=*/false, /*fold_loops=*/true,
            /*allow_gauge_detectors=*/true,
            /*approximate_disjoint_errors_threshold=*/1,
            /*ignore_decomposition_failures=*/false,
            /*block_decomposition_from_introducing_remnant_edges=*/false);
        for (bool merge_errors : {true, false}) {
          stim::DetectorErrorModel new_dem = dem;
          if (merge_errors) {
            std::vector<size_t> error_index_map;
            new_dem = common::merge_indistinguishable_errors(dem, error_index_map);
          }
          std::vector<stim::SparseShot> shots;
          sample_shots(test_data_seed, circuit, num_shots, shots);
          ASSERT_TRUE(simplex_test_compare(new_dem, shots));
        }
      }
    }
  }
}

// Same test as above but with automation using the simplex decoder
TEST(tesseract, Tesseract_simplex_DEM_exhaustive_test) {
  for (stim::DetectorErrorModel dem : {stim::DetectorErrorModel(R"DEM(
          error(0.1) D0 D1 L0
          error(0.1) D1 D2
          error(0.1) D2 D3
          error(0.1) D3 D0
          detector(0, 0, 0) D0
          detector(1, 0, 0) D1
          detector(2, 0, 0) D2
          detector(3, 0, 0) D3
        )DEM"),
                                       stim::DetectorErrorModel(R"DEM(
          error(0.011) D0
          error(0.02) D1 D2
          error(0.033) D1 D2 D3
          error(0.09) D1
          error(0.042) D3 D5
          error(0.043) D3 D4
          error(0.05) D2 D4 D5
          detector(0, 0, 0) D0
          detector(1, 0, 0) D1
          detector(2, 0, 0) D2
          detector(3, 0, 0) D3
          detector(4, 0, 0) D4
          detector(5, 0, 0) D5
        )DEM"),
                                       stim::DetectorErrorModel(R"DEM(
          error(0.02) D0
          error(0.02) D1
          error(0.02) D1 D0
          error(0.03) D1 D3
          error(0.02) D0 D2
          error(0.02) D0 D3
          error(0.02) D2 D3
          error(0.02) D2
          error(0.02) D3
          detector(0, 0, 0) D0
          detector(0, 0, 0) D1
          detector(0, 0, 1) D2
          detector(0, 0, 1) D3
        )DEM"),
                                       stim::DetectorErrorModel(R"DEM(
          error(0.02) D0
          error(0.02) D1
          error(0.02) D1 D0
          error(0.03) D1 D3
          error(0.02) D0 D2
          error(0.02) D0 D3
          error(0.02) D2 D3
          error(0.03) D3 D5
          error(0.02) D2
          error(0.03) D3
          detector(1, 0, 0) D0
          detector(0, 1, 0) D1
          detector(1, 0, 1) D2
          detector(0, 0, 1) D3
          detector(1, 1, 2) D4
          detector(0, 0, 2) D5
        )DEM")}) {
    size_t num_detectors = dem.count_detectors();
    std::vector<std::vector<bool>> detection_event(1 << num_detectors);
    ASSERT_LE(num_detectors, 64);
    // Try all possible dets sets on num_detectors detectors
    std::vector<stim::SparseShot> shots;
    for (uint64_t bitstring = 0; bitstring < (1ULL << num_detectors); ++bitstring) {
      stim::SparseShot shot;
      for (size_t d = 0; d < num_detectors; ++d) {
        if (bitstring & (1 << (num_detectors - d - 1))) {
          shot.hits.push_back(d);
        }
      }
      shots.push_back(shot);
    }

    bool return_val = simplex_test_compare(dem, shots);
    ASSERT_TRUE(return_val);
  }
}

TEST(tesseract, DecodersStripZeroProbabilityErrors) {
  stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0
        error(0) D1
        error(0.2) D2
        detector(0,0,0) D0
        detector(0,0,0) D1
        detector(0,0,0) D2
      )DEM");

  TesseractConfig t_config{dem};
  TesseractDecoder t_dec(t_config);
  EXPECT_EQ(t_dec.config.dem.count_errors(), 2);
  EXPECT_EQ(t_dec.errors.size(), 2);

  SimplexConfig s_config{dem};
  SimplexDecoder s_dec(s_config);
  EXPECT_EQ(s_dec.config.dem.count_errors(), 2);
  EXPECT_EQ(s_dec.errors.size(), 2);
}

TEST(tesseract, GetDetectorCoordsAllowsLogicalObservableInstructionsInDem) {
  stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0 L0
        detector(1,2,3) D0
        logical_observable L0
      )DEM");

  std::vector<std::vector<double>> detector_coords = get_detector_coords(dem);
  ASSERT_EQ(detector_coords.size(), 1);
  ASSERT_EQ(detector_coords[0].size(), 3);
  EXPECT_EQ(detector_coords[0][0], 1);
  EXPECT_EQ(detector_coords[0][1], 2);
  EXPECT_EQ(detector_coords[0][2], 3);
}
TEST(tesseract, SimplexAllowsLogicalObservableInstructionsInDem) {
  stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0 L0
        detector(0,0,0) D0
        logical_observable L0
      )DEM");

  EXPECT_NO_THROW({ SimplexDecoder s_dec(SimplexConfig{dem}); });
}

TEST(tesseract, DecoderErrorIndexMapsAreInOriginalDemCoordinates) {
  stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0
        error(0) D1
        error(0.2) D2
        error(0.3) D2
        detector(0,0,0) D0
        detector(0,0,0) D1
        detector(0,0,0) D2
      )DEM");

  TesseractDecoder t_dec(TesseractConfig{dem});
  SimplexDecoder s_dec(SimplexConfig{dem});

  EXPECT_EQ(t_dec.dem_error_to_error.size(), 4);
  EXPECT_EQ(t_dec.dem_error_to_error[1], std::numeric_limits<size_t>::max());
  EXPECT_EQ(t_dec.dem_error_to_error[2], t_dec.dem_error_to_error[3]);
  EXPECT_EQ(t_dec.error_to_dem_error[t_dec.dem_error_to_error[2]], 2);

  EXPECT_EQ(s_dec.dem_error_to_error.size(), 4);
  EXPECT_EQ(s_dec.dem_error_to_error[1], std::numeric_limits<size_t>::max());
  EXPECT_EQ(s_dec.dem_error_to_error[2], s_dec.dem_error_to_error[3]);
  EXPECT_EQ(s_dec.error_to_dem_error[s_dec.dem_error_to_error[2]], 2);

  std::vector<size_t> removed_error = {1};
  EXPECT_THROW(t_dec.cost_from_errors(removed_error), std::invalid_argument);
  EXPECT_THROW(s_dec.cost_from_errors(removed_error), std::invalid_argument);
  EXPECT_THROW(t_dec.get_flipped_observables(removed_error), std::invalid_argument);
  EXPECT_THROW(s_dec.get_flipped_observables(removed_error), std::invalid_argument);
}

TEST(tesseract, EneighborsCorrectness) {
  stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0 D1
        error(0.1) D1 D2
        error(0.1) D2 D3
        error(0.1) D4 D5
        error(0.1) D0 D2 D4
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
  std::vector<int> expected_e0_neighbors = {2, 4};
  std::vector<int> expected_e1_neighbors = {0, 3, 4};
  std::vector<int> expected_e2_neighbors = {0, 1, 4};
  std::vector<int> expected_e3_neighbors = {0, 2};
  std::vector<int> expected_e4_neighbors = {1, 3, 5};

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
  // e6 (D1,D4,D7) neighbors are D0,D2,D3,D5,D6,D8
  std::vector<int> expected_e6_neighbors = {0, 2, 3, 5, 6, 8};
  // e7 (D0,D3,D6) neighbors are D1,D4,D7
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

TEST(TesseractSparsifyTest, SuggestReactivateLimit) {
  EXPECT_EQ(suggest_sparsify_reactivate_limit(2, 2), 1);
  EXPECT_EQ(suggest_sparsify_reactivate_limit(2, 3), 3);
  EXPECT_EQ(suggest_sparsify_reactivate_limit(0, 2), 0);
  EXPECT_EQ(suggest_sparsify_reactivate_limit(1, std::numeric_limits<int>::max()),
            std::numeric_limits<int>::max());
  EXPECT_THROW(suggest_sparsify_reactivate_limit(2, -1), std::invalid_argument);
}

TEST(TesseractSparsifyTest, AutoReactivateLimitClampedToErrorCountBeforeOverflow) {
  stim::DetectorErrorModel dem = stim::DetectorErrorModel(R"DEM(
    error(0.1) D0
    detector(0, 0, 0) D0
    detector(1, 0, 0) D1
    detector(2, 0, 0) D2
    detector(3, 0, 0) D3
    detector(4, 0, 0) D4
    detector(5, 0, 0) D5
    detector(6, 0, 0) D6
    detector(7, 0, 0) D7
    detector(8, 0, 0) D8
    detector(9, 0, 0) D9
  )DEM");

  TesseractConfig cfg;
  cfg.dem = dem;
  cfg.merge_errors = false;
  cfg.sparsify_errors = true;
  cfg.sparsify_base_degree = std::numeric_limits<int>::max();
  cfg.sparsify_reactivate_limit = -1;
  TesseractDecoder dec(cfg);

  EXPECT_EQ(dec.config.sparsify_reactivate_limit, 1);
}

TEST(tesseract, InfinitePqlimitDoesNotReserveMaxVector) {
  stim::DetectorErrorModel dem = stim::DetectorErrorModel(R"DEM(
    error(0.1) D0
    error(0.1) D1
    error(0.2) D0 D1 L0
  )DEM");

  TesseractConfig cfg;
  cfg.dem = dem;
  cfg.merge_errors = false;
  cfg.pqlimit = std::numeric_limits<size_t>::max();
  TesseractDecoder dec(cfg);

  EXPECT_NO_THROW(dec.decode_to_errors({0, 1}));
  std::vector<size_t> expected = {2};
  EXPECT_EQ(dec.predicted_errors_buffer, expected);
}

TEST(TesseractSparsifyTest, HighDegreeErrorRemoved) {
  stim::DetectorErrorModel dem = stim::DetectorErrorModel(R"DEM(
    error(0.1) D0
    error(0.1) D1
    error(0.1) D2
    error(0.1) D3
    error(0.01) D0 D1 D2 D3
  )DEM");

  // Case 1: Without sparsification (default)
  // Should prefer the degree 4 error (Error 4) because it has lower cost than 4 degree-1 errors.
  {
    TesseractConfig cfg;
    cfg.dem = dem;
    cfg.merge_errors = false;
    cfg.sparsify_errors = false;
    TesseractDecoder dec(cfg);

    dec.decode_to_errors({0, 1, 2, 3});

    std::vector<size_t> expected = {4};
    EXPECT_EQ(dec.predicted_errors_buffer, expected);
  }

  // Case 2: With sparsification and limit = 0
  // The degree 4 error (optional) is NOT reactivated, so it must use the 4 degree-1 errors.
  {
    TesseractConfig cfg;
    cfg.dem = dem;
    cfg.merge_errors = false;
    cfg.sparsify_errors = true;
    cfg.sparsify_base_degree = 2;
    cfg.sparsify_max_degree = 4;
    cfg.sparsify_reactivate_limit = 0;
    TesseractDecoder dec(cfg);

    dec.decode_to_errors({0, 1, 2, 3});

    std::vector<size_t> got = dec.predicted_errors_buffer;
    std::sort(got.begin(), got.end());
    std::vector<size_t> expected = {0, 1, 2, 3};
    EXPECT_EQ(got, expected);
  }

  // Case 3: With sparsification and limit = 1
  // The degree 4 error (optional) IS reactivated because limit is 1, so it should be preferred
  // again.
  {
    TesseractConfig cfg;
    cfg.dem = dem;
    cfg.merge_errors = false;
    cfg.sparsify_errors = true;
    cfg.sparsify_base_degree = 2;
    cfg.sparsify_max_degree = 4;
    cfg.sparsify_reactivate_limit = 1;
    TesseractDecoder dec(cfg);

    dec.decode_to_errors({0, 1, 2, 3});

    std::vector<size_t> expected = {4};
    EXPECT_EQ(dec.predicted_errors_buffer, expected);
  }
}

TEST(tesseract, MoreThan64Observables) {
  std::string dem_str = "error(0.1)";
  for (int i = 0; i < 70; i++) {
    dem_str += " D" + std::to_string(i) + " L" + std::to_string(i);
  }
  dem_str += "\n";
  stim::DetectorErrorModel dem(dem_str.c_str());

  TesseractConfig tesseract_config{dem};
  TesseractDecoder tesseract_decoder(tesseract_config);

  std::vector<uint64_t> hits;
  for (int i = 0; i < 70; i++) hits.push_back(i);
  tesseract_decoder.decode_to_errors(hits);

  std::vector<int> flipped =
      tesseract_decoder.get_flipped_observables(tesseract_decoder.predicted_errors_buffer);

  ASSERT_EQ(flipped.size(), 70);
  for (int i = 0; i < 70; i++) {
    ASSERT_EQ(flipped[i], i);
  }
}

TEST(utils, DuplicateDetectorCoords) {
  std::string dem_str = "detector(0, 0, 1) D0\ndetector(0, 0, 2) D0\nerror(0.1) D0\n";
  stim::DetectorErrorModel dem(dem_str.c_str());
  auto coords = get_detector_coords(dem);
  ASSERT_EQ(coords.size(), 1);
  ASSERT_EQ(coords[0].size(), 3);
  // Match Stim's DetectorErrorModel::get_detector_coordinates behavior: the
  // first declaration for a detector wins.
  ASSERT_EQ(coords[0][2], 1.0);
}

TEST(utils, SparseDetectorCoords) {
  std::string dem_str = "detector(1, 2, 3) D1\nerror(0.1) D0 D1\n";
  stim::DetectorErrorModel dem(dem_str.c_str());
  auto coords = get_detector_coords(dem);
  ASSERT_EQ(coords.size(), 2);
  EXPECT_TRUE(coords[0].empty());
  ASSERT_EQ(coords[1].size(), 3);
  EXPECT_EQ(coords[1][0], 1.0);
  EXPECT_EQ(coords[1][1], 2.0);
  EXPECT_EQ(coords[1][2], 3.0);
}

TEST(utils, BuildDetOrdersCoordinateSparse) {
  std::string dem_str = "detector(10, 0, 0) D1\nerror(0.1) D0 D1\n";
  stim::DetectorErrorModel dem(dem_str.c_str());
  auto orders = build_det_orders(dem, 1, DetOrder::DetCoordinate, 0);
  ASSERT_EQ(orders.size(), 1);
  ASSERT_EQ(orders[0].size(), 2);
}

TEST(simplex, DuplicateDetectorCoords) {
  std::string dem_str = "detector(0, 0, 1) D0\ndetector(0, 0, 2) D0\nerror(0.1) D0\n";
  stim::DetectorErrorModel dem(dem_str.c_str());
  SimplexConfig config{dem};
  EXPECT_NO_THROW({ SimplexDecoder decoder(config); });
}

TEST(simplex, SparseDetectorCoords) {
  std::string dem_str = "detector(1, 2, 3) D1\nerror(0.1) D0 D1\n";
  stim::DetectorErrorModel dem(dem_str.c_str());
  SimplexConfig config{dem};
  EXPECT_NO_THROW({ SimplexDecoder decoder(config); });
}

TEST(utils, DetectorGraphUsesPositiveParityReducedSymptoms) {
  stim::DetectorErrorModel dem(R"DEM(
    error(0) D0 D1
    error(0.1) D0 D0 D1
    error(0.2) D1 D2 D3
  )DEM");

  EXPECT_EQ(build_detector_graph(dem),
            (std::vector<std::vector<size_t>>{{}, {2, 3}, {1, 3}, {1, 2}}));
}

TEST(utils, DetectorGraphRejectsNegativeErrorProbabilities) {
  const std::vector<double> args{-0.1};
  const std::vector<stim::DemTarget> targets{stim::DemTarget::relative_detector_id(0),
                                             stim::DemTarget::relative_detector_id(1)};
  stim::DetectorErrorModel dem;
  dem.instructions.push_back(
      stim::DemInstruction{args, targets, "", stim::DemInstructionType::DEM_ERROR});

  EXPECT_THROW(build_detector_graph(dem), std::invalid_argument);
}

TEST(utils, EmptyDemHasEmptyBfsOrders) {
  stim::DetectorErrorModel dem;
  EXPECT_EQ(build_det_orders(dem, 3, DetOrder::DetBFS, 0), std::vector<std::vector<size_t>>(3));
}

TEST(utils, PreprocessingPreservesAllDetectorOrderEntries) {
  stim::DetectorErrorModel dem("error(0) D2\nerror(0.1) D3 D3");
  TesseractConfig config{dem};
  config.det_orders = build_det_orders(dem, 1, DetOrder::DetBFS, 0);

  TesseractDecoder decoder(config);
  EXPECT_EQ(decoder.num_detectors, 4);
  EXPECT_EQ(decoder.config.det_orders[0].size(), 4);
}

TEST(utils, BfsOrdersContainDetectorsInTraversalOrder) {
  // A path with scrambled detector IDs: D0--D4--D1--D3--D2. The inverse of a
  // traversal of this path is not itself a BFS traversal.
  stim::DetectorErrorModel dem(R"DEM(
    error(0.1) D0 D4
    error(0.1) D4 D1
    error(0.1) D1 D3
    error(0.1) D3 D2
  )DEM");
  const auto graph = build_detector_graph(dem);
  const auto orders = build_det_orders(dem, 16, DetOrder::DetBFS, 0);

  for (const auto& detector_at_position : orders) {
    ASSERT_EQ(detector_at_position.size(), graph.size());
    std::vector<size_t> sorted_order = detector_at_position;
    std::sort(sorted_order.begin(), sorted_order.end());
    EXPECT_EQ(sorted_order, (std::vector<size_t>{0, 1, 2, 3, 4}));

    std::vector<size_t> distance(graph.size(), std::numeric_limits<size_t>::max());
    std::queue<size_t> queue;
    distance[detector_at_position[0]] = 0;
    queue.push(detector_at_position[0]);
    while (!queue.empty()) {
      const size_t detector = queue.front();
      queue.pop();
      for (size_t neighbor : graph[detector]) {
        if (distance[neighbor] == std::numeric_limits<size_t>::max()) {
          distance[neighbor] = distance[detector] + 1;
          queue.push(neighbor);
        }
      }
    }
    for (size_t position = 1; position < detector_at_position.size(); ++position) {
      EXPECT_LE(distance[detector_at_position[position - 1]],
                distance[detector_at_position[position]]);
    }
  }
}

TEST(utils, CoordinateOrdersContainDetectorsInProjectionOrder) {
  // Coordinates are intentionally declared out of detector-ID order. A 1D
  // projection can only produce the coordinate-sorted order or its reverse.
  stim::DetectorErrorModel dem(R"DEM(
    detector(2) D3
    detector(0) D0
    detector(3) D1
    detector(1) D2
  )DEM");

  const auto order = build_det_orders(dem, 1, DetOrder::DetCoordinate, 0)[0];
  EXPECT_TRUE(order == (std::vector<size_t>{0, 2, 3, 1}) ||
              order == (std::vector<size_t>{1, 3, 2, 0}));
}

TEST(tesseract, CoordinateOrderBuilderAndDecoderUseSameTraversalConvention) {
  // Sorting by coordinate produces either [D0, D2, D3, D1] or its reverse.
  // At beam 0, both traversals find the lower-cost correction {E1, E3}, which
  // generated this symptom and flips L0. Before the representation fix,
  // build_det_orders returned the inverse rank map instead; interpreted as a
  // traversal, either inverse confidently chooses {E0, E2} and misses L0.
  stim::DetectorErrorModel dem(R"DEM(
    detector(0) D0
    detector(3) D1
    detector(1) D2
    detector(2) D3
    error(0.10) D1
    error(0.30) D0 D1 D2 L0
    error(0.12) D2 D3
    error(0.08) D0 D3
  )DEM");

  const auto detector_at_position = build_det_orders(dem, 1, DetOrder::DetCoordinate, 0)[0];
  std::vector<size_t> legacy_position_of_detector(detector_at_position.size());
  for (size_t position = 0; position < detector_at_position.size(); ++position) {
    legacy_position_of_detector[detector_at_position[position]] = position;
  }

  TesseractConfig corrected_config{dem};
  corrected_config.det_beam = 0;
  corrected_config.merge_errors = false;
  corrected_config.det_orders = {detector_at_position};
  TesseractDecoder corrected_decoder(corrected_config);
  corrected_decoder.decode_to_errors({1, 2, 3});
  EXPECT_FALSE(corrected_decoder.low_confidence_flag);
  auto corrected_errors = corrected_decoder.predicted_errors_buffer;
  std::sort(corrected_errors.begin(), corrected_errors.end());
  EXPECT_EQ(corrected_errors, (std::vector<size_t>{1, 3}));
  EXPECT_EQ(corrected_decoder.get_flipped_observables(corrected_decoder.predicted_errors_buffer),
            (std::vector<int>{0}));

  TesseractConfig legacy_config = corrected_config;
  legacy_config.det_orders = {legacy_position_of_detector};
  TesseractDecoder legacy_decoder(legacy_config);
  legacy_decoder.decode_to_errors({1, 2, 3});
  EXPECT_FALSE(legacy_decoder.low_confidence_flag);
  auto legacy_errors = legacy_decoder.predicted_errors_buffer;
  std::sort(legacy_errors.begin(), legacy_errors.end());
  EXPECT_EQ(legacy_errors, (std::vector<size_t>{0, 2}));
  EXPECT_TRUE(
      legacy_decoder.get_flipped_observables(legacy_decoder.predicted_errors_buffer).empty());
  EXPECT_LT(corrected_decoder.cost_from_errors(corrected_decoder.predicted_errors_buffer),
            legacy_decoder.cost_from_errors(legacy_decoder.predicted_errors_buffer));
}

TEST(utils, DetectorCoordinatesAreKeyedAndAllowMissingOrShortCoordinates) {
  stim::DetectorErrorModel dem(R"DEM(
    detector(2, 20) D2
    detector(0) D0
    detector(99) D2
    error(0.1) D3
  )DEM");

  const auto coords = get_detector_coords(dem);
  ASSERT_EQ(coords.size(), 4);
  EXPECT_EQ(coords[0], (std::vector<double>{0}));
  EXPECT_TRUE(coords[1].empty());
  EXPECT_EQ(coords[2], (std::vector<double>{2, 20}));
  EXPECT_TRUE(coords[3].empty());

  const auto order = build_det_orders(dem, 1, DetOrder::DetCoordinate, 0)[0];
  ASSERT_EQ(order.size(), 4);
  EXPECT_EQ(order[2], 1);
  EXPECT_EQ(order[3], 3);
}

TEST(tesseract, DetectorOrdersMustBePermutations) {
  stim::DetectorErrorModel dem("error(0.1) D0 D1 D2");

  TesseractConfig valid_config{dem};
  valid_config.det_orders = {{2, 0, 1}};
  EXPECT_NO_THROW({ TesseractDecoder decoder(valid_config); });

  TesseractConfig wrong_size_config{dem};
  wrong_size_config.det_orders = {{0, 1}};
  EXPECT_THROW({ TesseractDecoder decoder(wrong_size_config); }, std::invalid_argument);

  TesseractConfig duplicate_config{dem};
  duplicate_config.det_orders = {{0, 0, 2}};
  EXPECT_THROW({ TesseractDecoder decoder(duplicate_config); }, std::invalid_argument);

  TesseractConfig out_of_range_config{dem};
  out_of_range_config.det_orders = {{0, 1, 3}};
  EXPECT_THROW({ TesseractDecoder decoder(out_of_range_config); }, std::invalid_argument);
}

TEST(tesseract, SelectedDetectorOrderIndexMustBeInRange) {
  stim::DetectorErrorModel dem("error(0.1) D0");
  TesseractDecoder decoder(TesseractConfig{dem});
  EXPECT_THROW(decoder.decode_to_errors({}, 1, 0), std::out_of_range);
}

TEST(utils, GariSourcePrefixB8Decodes) {
  stim::DetectorErrorModel gari_dem(R"DEM(
    error(0.1) D0 D3 L0
    error(0.1) D1 D6 L1
    detector D9
  )DEM");

  FILE* file = std::tmpfile();
  ASSERT_NE(file, nullptr);
  std::fputc('\x09', file);  // D0 D3.
  std::fputc('\x42', file);  // D1 D6.
  std::rewind(file);
  auto format = stim::format_name_to_enum_map().at("b8");
  auto reader = stim::MeasureRecordReader<stim::MAX_BITWORD_WIDTH>::make(file, format.id, 0, 7, 0);
  std::vector<stim::SparseShot> shots;
  stim::SparseShot shot;
  while (reader->start_and_read_entire_record(shot)) {
    shots.push_back(shot);
    shot.clear();
  }
  fclose(file);

  ASSERT_EQ(shots.size(), 2);
  EXPECT_EQ(shots[0].hits, (std::vector<uint64_t>{0, 3}));
  EXPECT_EQ(shots[1].hits, (std::vector<uint64_t>{1, 6}));

  TesseractDecoder tesseract(TesseractConfig{gari_dem});
  SimplexDecoder simplex(SimplexConfig{gari_dem});
  EXPECT_EQ(tesseract.decode(shots[0].hits), (std::vector<int>{0}));
  EXPECT_EQ(tesseract.decode(shots[1].hits), (std::vector<int>{1}));
  EXPECT_EQ(simplex.decode(shots[0].hits), (std::vector<int>{0}));
  EXPECT_EQ(simplex.decode(shots[1].hits), (std::vector<int>{1}));
}

}  // namespace
}  // namespace tesseract_decoder
