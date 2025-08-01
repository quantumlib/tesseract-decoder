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

#include <chrono>

#include "simplex.h"
#include "stim.h"
#include "utils.h"

constexpr uint64_t test_data_seed = 752024;

template <typename Decoder>
void benchmark_decoder(Decoder& decoder, stim::Circuit& circuit, size_t num_shots) {
  // Sample data
  std::vector<stim::SparseShot> shots;
  sample_shots(test_data_seed, circuit, num_shots, shots);

  // Use volatile to try to ensure compiler does not optimize out the decoding
  volatile size_t total_num_errors_used = 0;
  size_t num_low_confidence = 0;
  size_t num_errors = 0;
  size_t num_decoded = 0;
  auto vector_to_u64_mask = [](const std::vector<int>& v) {
    uint64_t mask = 0;
    for (int i : v) {
      mask ^= (1ULL << i);
    }
    return mask;
  };

  auto benchmark_func = [&]() {
    for (size_t shot = 0; shot < num_shots; ++shot) {
      decoder.decode_to_errors(shots[shot].hits);
      uint64_t obs = vector_to_u64_mask(decoder.mask_from_errors(decoder.predicted_errors_buffer));
      num_errors += (!decoder.low_confidence_flag and (obs != shots[shot].obs_mask_as_u64()));
      num_low_confidence += decoder.low_confidence_flag;
      total_num_errors_used += decoder.predicted_errors_buffer.size();
      ++num_decoded;
    }
  };

  double num_milliseconds = 0.0;
  auto start_time = std::chrono::steady_clock::now();
  do {
    benchmark_func();
    auto end_time = std::chrono::steady_clock::now();
    num_milliseconds = std::chrono::duration<double, std::milli>(end_time - start_time).count();
  } while (num_milliseconds < 1000.0);
  std::cout << (num_milliseconds / num_decoded) << " milliseconds per shot " << num_decoded
            << " shots " << num_low_confidence << " low confidence " << num_errors << " errors "
            << " total_num_errors_used = " << total_num_errors_used << std::endl;
}

void benchmark_tesseract(std::string circuit_path, size_t num_shots) {
  FILE* file = fopen(circuit_path.c_str(), "r");
  if (!file) {
    throw std::invalid_argument("Could not open the file: " + circuit_path);
  }
  stim::Circuit circuit = stim::Circuit::from_file(file);
  fclose(file);
  stim::DetectorErrorModel dem = stim::ErrorAnalyzer::circuit_to_detector_error_model(
      circuit, /*decompose_errors=*/false, /*fold_loops=*/true,
      /*allow_gauge_detectors=*/true,
      /*approximate_disjoint_errors_threshold=*/1,
      /*ignore_decomposition_failures=*/false,
      /*block_decomposition_from_introducing_remnant_edges=*/false);
  dem = common::remove_zero_probability_errors(dem);
  TesseractConfig config{dem};
  config.det_beam = 20;
  config.pqlimit = 10'000'000;
  TesseractDecoder decoder(config);
  std::cout << "\tTesseract:";
  benchmark_decoder(decoder, circuit, num_shots);
}

void benchmark_simplex(std::string circuit_path, size_t num_shots) {
  FILE* file = fopen(circuit_path.c_str(), "r");
  if (!file) {
    throw std::invalid_argument("Could not open the file: " + circuit_path);
  }
  stim::Circuit circuit = stim::Circuit::from_file(file);
  fclose(file);
  stim::DetectorErrorModel dem = stim::ErrorAnalyzer::circuit_to_detector_error_model(
      circuit, /*decompose_errors=*/false, /*fold_loops=*/true,
      /*allow_gauge_detectors=*/true,
      /*approximate_disjoint_errors_threshold=*/1,
      /*ignore_decomposition_failures=*/false,
      /*block_decomposition_from_introducing_remnant_edges=*/false);
  dem = common::remove_zero_probability_errors(dem);
  SimplexConfig config{dem};
  config.parallelize = true;
  SimplexDecoder decoder(config);
  std::cout << "\tSimplex:";
  benchmark_decoder(decoder, circuit, num_shots);
}

int main() {
  for (std::string circuit_fname : get_files_recursive("testdata")) {
    if (circuit_fname.find("d=11") != std::string::npos or
        circuit_fname.find("d=13") != std::string::npos or
        circuit_fname.find("d=15") != std::string::npos or
        circuit_fname.find("d=17") != std::string::npos or
        circuit_fname.find("d=19") != std::string::npos or
        circuit_fname.find("d=21") != std::string::npos) {
      continue;
    }
    if (circuit_fname.find("colorcodes") == std::string::npos and
        circuit_fname.find("surfacecodes") == std::string::npos) {
      continue;
    }
    if (circuit_fname.find("uniform") != std::string::npos) {
      continue;
    }
    if (circuit_fname.find("p=0.0005") == std::string::npos and
        circuit_fname.find("p=0.001") == std::string::npos and
        circuit_fname.find("p=0.002") == std::string::npos) {
      continue;
    }
    std::cout << "Benchmark on " << circuit_fname << std::endl;
    benchmark_tesseract(circuit_fname, 20);
    benchmark_simplex(circuit_fname, 20);
  }
  return 0;
}
