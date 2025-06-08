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

#include <vector>

#include "gtest/gtest.h"
#include "simplex.h"
#include "stim.h"
#include "utils.h"

constexpr uint64_t test_data_seed = 752024;

bool simplex_test_compare(stim::DetectorErrorModel& dem, std::vector<stim::SparseShot>& shots, int openmp_threads) {
  TesseractConfig tesseract_config{dem};
  if (openmp_threads > 0) {
    tesseract_config.with_openmp = true;
    if (tesseract_config.beam_climbing) {
      tesseract_config.beam_climbing_openmp_threads = openmp_threads;
    } else {
      tesseract_config.det_orders_openmp_threads = openmp_threads;
    }
  }
  TesseractDecoder tesseract_decoder(tesseract_config);

  SimplexConfig simplex_config{dem};
  SimplexDecoder simplex_decoder(simplex_config);

  for (size_t shot = 0; shot < shots.size(); shot++) {
    auto decoding_result = tesseract_decoder.decode_to_errors(shots[shot].hits);
    std::vector<size_t> predicted_errors = decoding_result.first;
    bool low_confidence_flag = decoding_result.second;
    double tesseract_cost = tesseract_decoder.cost_from_errors(predicted_errors);

    if (low_confidence_flag) {
      // Simplex c++ does not yet support undecodable shots -- i.e. detection
      // event configurations with no error solution.
      std::cout << "not decoding shot " << shot
                << " with simplex because Tesseract found no solution"
                << std::endl;
      continue;
    }

    simplex_decoder.decode_to_errors(shots[shot].hits);
    double simplex_cost = simplex_decoder.cost_from_errors(
        simplex_decoder.predicted_errors_buffer);

    // If there is a mismatch in weights, print diagnostic information
    if (std::abs(tesseract_cost - simplex_cost) > EPSILON) {
      std::cout << "shot " << shot << " ";
      for (size_t d : shots[shot].hits) {
        std::cout << "D" << d << " ";
      }
      std::cout << std::endl;
      std::cout << "Error: For shot " << shot
                << " tesseract got solution with cost:" << tesseract_cost
                << " simplex got solution with cost: " << simplex_cost
                << std::endl;
      std::cout << "tesseract used errors ";
      for (size_t ei : predicted_errors) {
        std::cout << ei << ", ";
        std::cout << tesseract_decoder.errors[ei].str() << std::endl;
      }
      std::cout << " and had cost " << tesseract_cost << std::endl;
      std::cout << "simplex used errors ";
      for (size_t ei : simplex_decoder.predicted_errors_buffer) {
        std::cout << ei << ", ";
        std::cout << simplex_decoder.errors[ei].str() << std::endl;
      }
      std::cout << " and had cost " << simplex_cost << std::endl;
      return false;
    }
  }
  return true;
}

TEST(tesseract, Tesseract_simplex_test) {
  for (float p_err : {0.001, 0.003, 0.005}) {
    for (size_t distance : {3, 5}) {
      for (const size_t num_rounds : {2, 5, 10}) {
        const size_t num_shots = 1000 / num_rounds / distance;
        std::cout << "p_err = " << p_err << " distance = " << distance
                  << " num_rounds = " << num_rounds
                  << " num_shots = " << num_shots << std::endl;
        stim::CircuitGenParameters params(num_rounds, /*distance=*/distance,
                                          /*task=*/"rotated_memory_x");
        params.after_clifford_depolarization = p_err;
        params.before_round_data_depolarization = p_err;
        params.before_measure_flip_probability = p_err;
        params.after_reset_flip_probability = p_err;
        stim::Circuit circuit =
            stim::generate_surface_code_circuit(params).circuit;
        stim::DetectorErrorModel dem =
            stim::ErrorAnalyzer::circuit_to_detector_error_model(
                circuit, /*decompose_errors=*/false, /*fold_loops=*/true,
                /*allow_gauge_detectors=*/true,
                /*approximate_disjoint_errors_threshold=*/1,
                /*ignore_decomposition_failures=*/false,
                /*block_decomposition_from_introducing_remnant_edges=*/false);
        for (bool merge_errors : {true, false}) {
          stim::DetectorErrorModel new_dem = dem;
          if (merge_errors) {
            new_dem = common::merge_identical_errors(dem);
          }
          std::vector<stim::SparseShot> shots;
          sample_shots(test_data_seed, circuit, num_shots, shots);
          ASSERT_TRUE(simplex_test_compare(new_dem, shots, 0));
          ASSERT_TRUE(simplex_test_compare(new_dem, shots, 2));
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
    for (uint64_t bitstring = 0; bitstring < (1ULL << num_detectors);
         ++bitstring) {
      stim::SparseShot shot;
      for (size_t d = 0; d < num_detectors; ++d) {
        if (bitstring & (1 << (num_detectors - d - 1))) {
          shot.hits.push_back(d);
        }
      }
      shots.push_back(shot);
    }

    ASSERT_TRUE(simplex_test_compare(dem, shots, 0));
    ASSERT_TRUE(simplex_test_compare(dem, shots, 2));
  }
}
