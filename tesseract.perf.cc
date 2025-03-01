#include "tesseract.h"

#include "common/benchmark/benchmark.h"
#include "simplex.h"
#include "stim.h"
#include "utils.h"

constexpr uint64_t test_data_seed = 752024;

template <typename Decoder>
void benchmark_decoder(Decoder& decoder, stim::Circuit& circuit, size_t num_shots) {
  // Sample data
  std::vector<stim::SparseShot> shots;
  sample_shots(test_data_seed, circuit, num_shots, shots);

  // Try to ensure compiler does not optimize out the decoding
  volatile size_t total_num_errors_used = 0;
  size_t num_low_confidence = 0;
  size_t num_errors = 0;
  size_t num_decoded = 0;
  benchmark_go([&]() {
    for (size_t shot = 0; shot < num_shots; ++shot) {
      decoder.decode_to_errors(shots[shot].hits);
      common::ObservablesMask obs = decoder.mask_from_errors(decoder.predicted_errors_buffer);
      num_errors += (!decoder.low_confidence_flag and (obs != shots[shot].obs_mask_as_u64()));
      num_low_confidence += decoder.low_confidence_flag;
      total_num_errors_used += decoder.predicted_errors_buffer.size();
      ++num_decoded;
    }
  })
      .goal_micros(num_shots)
      .show_rate("shots", (double)(num_decoded));
  std::cout << num_decoded << " shots " << num_low_confidence << " low confidence " << num_errors
            << " errors "
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
      circuit, /*decompose_errors=*/false, /*fold_loops=*/true, /*allow_gauge_detectors=*/true,
      /*approximate_disjoint_errors_threshold=*/1, /*ignore_decomposition_failures=*/false,
      /*block_decomposition_from_introducing_remnant_edges=*/false);
  TesseractConfig config{dem};
  config.det_beam = 20;
  config.pqlimit = 10'000'000;
  TesseractDecoder decoder(config);
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
      circuit, /*decompose_errors=*/false, /*fold_loops=*/true, /*allow_gauge_detectors=*/true,
      /*approximate_disjoint_errors_threshold=*/1, /*ignore_decomposition_failures=*/false,
      /*block_decomposition_from_introducing_remnant_edges=*/false);
  SimplexConfig config{dem};
  config.parallelize = true;
  SimplexDecoder decoder(config);
  benchmark_decoder(decoder, circuit, num_shots);
}

BENCHMARK(TesseractDecoder) {
  for (std::string circuit_fname : get_files_recursive("tesseract/testdata")) {
    if (circuit_fname.find("d=11") != std::string::npos) {
      continue;
    }
    if (circuit_fname.find("uniform") != std::string::npos) {
      continue;
    }
    if (circuit_fname.find("p=0.003") != std::string::npos) {
      continue;
    }
    std::cout << "Benchmark on " << circuit_fname << std::endl;
    benchmark_tesseract(circuit_fname, 20);
  }
}

BENCHMARK(SimplexDecoder) {
  for (std::string circuit_fname : get_files_recursive("tesseract/testdata")) {
    if (circuit_fname.find("d=11") != std::string::npos) {
      continue;
    }
    if (circuit_fname.find("uniform") != std::string::npos) {
      continue;
    }
    if (circuit_fname.find("p=0.003") != std::string::npos) {
      continue;
    }
    std::cout << "Benchmark on " << circuit_fname << std::endl;
    benchmark_simplex(circuit_fname, 20);
  }
}
