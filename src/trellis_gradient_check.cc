#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tesseract_trellis.h"

namespace {

struct SyndromeCounts {
  uint64_t syndrome;
  uint64_t label_zero_count;
  uint64_t label_one_count;
};

struct CoordinateCheck {
  size_t error_index;
  double analytic;
  double finite_difference;
};

stim::DetectorErrorModel read_dem(const std::string& path) {
  FILE* file = fopen(path.c_str(), "r");
  if (file == nullptr) {
    throw std::invalid_argument("could not open DEM: " + path);
  }
  stim::DetectorErrorModel dem = stim::DetectorErrorModel::from_file(file);
  fclose(file);
  return dem;
}

std::vector<SyndromeCounts> read_counts(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::invalid_argument("could not open syndrome counts: " + path);
  }
  std::vector<SyndromeCounts> rows;
  SyndromeCounts row{};
  while (input >> row.syndrome >> row.label_zero_count >> row.label_one_count) {
    rows.push_back(row);
  }
  if (!input.eof()) {
    throw std::invalid_argument("malformed syndrome-count row in: " + path);
  }
  if (rows.empty()) {
    throw std::invalid_argument("no syndrome-count rows in: " + path);
  }
  return rows;
}

std::vector<uint64_t> syndrome_hits(uint64_t syndrome) {
  std::vector<uint64_t> hits;
  while (syndrome != 0) {
    const uint64_t bit = static_cast<uint64_t>(__builtin_ctzll(syndrome));
    hits.push_back(bit);
    syndrome &= syndrome - 1;
  }
  return hits;
}

double shifted_probability(double probability, double logit_shift) {
  const double logit = std::log(probability / (1.0 - probability));
  return 1.0 / (1.0 + std::exp(-(logit + logit_shift)));
}

stim::DetectorErrorModel perturb_error_logit(const stim::DetectorErrorModel& dem,
                                             size_t target_error_index, double logit_shift) {
  stim::DetectorErrorModel result;
  size_t error_index = 0;
  for (const auto& instruction : dem.flattened().instructions) {
    if (instruction.type == stim::DemInstructionType::DEM_ERROR) {
      double probability = instruction.arg_data[0];
      if (error_index == target_error_index) {
        probability = shifted_probability(probability, logit_shift);
      }
      result.append_error_instruction(probability, instruction.target_data, instruction.tag);
      ++error_index;
    } else {
      result.append_dem_instruction(instruction);
    }
  }
  if (target_error_index >= error_index) {
    throw std::out_of_range("error index is outside the preprocessed DEM");
  }
  return result;
}

double conditional_nll(const stim::DetectorErrorModel& dem, const std::vector<SyndromeCounts>& rows,
                       size_t beam_width) {
  TesseractTrellisConfig config;
  config.dem = dem;
  config.beam_width = beam_width;
  config.merge_errors = false;
  TesseractTrellisDecoder decoder(config);
  long double loss = 0.0;
  uint64_t shots = 0;
  for (const auto& row : rows) {
    decoder.decode_shot(syndrome_hits(row.syndrome));
    const double probability = decoder.observable_probability();
    if (decoder.low_confidence_flag || !std::isfinite(probability) || probability <= 0.0 ||
        probability >= 1.0) {
      throw std::runtime_error("non-finite observable probability in finite-difference pass");
    }
    loss -= row.label_one_count * std::log(probability);
    loss -= row.label_zero_count * std::log1p(-probability);
    shots += row.label_zero_count + row.label_one_count;
  }
  return static_cast<double>(loss / shots);
}

void print_json(size_t beam_width, size_t error_count, uint64_t sample_shots, double baseline_nll,
                double epsilon, const std::vector<CoordinateCheck>& checks) {
  double max_absolute_error = 0.0;
  double max_relative_error = 0.0;
  for (const auto& check : checks) {
    const double absolute_error = std::abs(check.analytic - check.finite_difference);
    const double scale = std::max({std::abs(check.analytic), std::abs(check.finite_difference),
                                   std::numeric_limits<double>::epsilon()});
    max_absolute_error = std::max(max_absolute_error, absolute_error);
    max_relative_error = std::max(max_relative_error, absolute_error / scale);
  }

  std::cout << std::setprecision(17);
  std::cout << "{\n"
            << "  \"schema_version\": 1,\n"
            << "  \"beam_width\": " << beam_width << ",\n"
            << "  \"preprocessed_errors\": " << error_count << ",\n"
            << "  \"sample_shots\": " << sample_shots << ",\n"
            << "  \"baseline_conditional_nll\": " << baseline_nll << ",\n"
            << "  \"finite_difference_epsilon\": " << epsilon << ",\n"
            << "  \"max_absolute_error\": " << max_absolute_error << ",\n"
            << "  \"max_relative_error\": " << max_relative_error << ",\n"
            << "  \"checks\": [\n";
  for (size_t i = 0; i < checks.size(); ++i) {
    const auto& check = checks[i];
    const double absolute_error = std::abs(check.analytic - check.finite_difference);
    const double scale = std::max({std::abs(check.analytic), std::abs(check.finite_difference),
                                   std::numeric_limits<double>::epsilon()});
    std::cout << "    {\"error_index\": " << check.error_index
              << ", \"analytic\": " << check.analytic
              << ", \"finite_difference\": " << check.finite_difference
              << ", \"absolute_error\": " << absolute_error
              << ", \"relative_error\": " << absolute_error / scale << "}";
    std::cout << (i + 1 == checks.size() ? "\n" : ",\n");
  }
  std::cout << "  ]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage: trellis_gradient_check DEM COUNTS BEAM CHECKS EPSILON\n";
    return EXIT_FAILURE;
  }
  try {
    const std::string dem_path = argv[1];
    const std::string counts_path = argv[2];
    const size_t beam_width = std::stoull(argv[3]);
    const size_t check_count = std::stoull(argv[4]);
    const double epsilon = std::stod(argv[5]);
    if (beam_width == 0 || check_count == 0 || !std::isfinite(epsilon) || epsilon <= 0.0) {
      throw std::invalid_argument("BEAM, CHECKS, and EPSILON must be positive");
    }

    const auto rows = read_counts(counts_path);
    TesseractTrellisConfig config;
    config.dem = read_dem(dem_path);
    config.beam_width = beam_width;
    TesseractTrellisDecoder decoder(config);
    const stim::DetectorErrorModel preprocessed_dem = decoder.config.dem;
    std::vector<double> gradient(decoder.errors.size(), 0.0);
    long double loss = 0.0;
    uint64_t sample_shots = 0;

    for (const auto& row : rows) {
      const auto logit_gradient =
          decoder.decode_shot_with_observable_logit_gradient(syndrome_hits(row.syndrome));
      const double probability = decoder.observable_probability();
      if (decoder.low_confidence_flag || !std::isfinite(probability) || probability <= 0.0 ||
          probability >= 1.0 || logit_gradient.size() != gradient.size()) {
        throw std::runtime_error("invalid analytic gradient result");
      }
      const uint64_t row_shots = row.label_zero_count + row.label_one_count;
      loss -= row.label_one_count * std::log(probability);
      loss -= row.label_zero_count * std::log1p(-probability);
      const double logit_loss_gradient =
          row_shots * probability - static_cast<double>(row.label_one_count);
      for (size_t error_index = 0; error_index < gradient.size(); ++error_index) {
        gradient[error_index] += logit_loss_gradient * logit_gradient[error_index];
      }
      sample_shots += row_shots;
    }
    for (double& value : gradient) {
      value /= sample_shots;
    }
    const double baseline_nll = static_cast<double>(loss / sample_shots);

    std::vector<size_t> ranked_indices(gradient.size());
    for (size_t i = 0; i < ranked_indices.size(); ++i) {
      ranked_indices[i] = i;
    }
    std::sort(ranked_indices.begin(), ranked_indices.end(),
              [&](size_t a, size_t b) { return std::abs(gradient[a]) > std::abs(gradient[b]); });
    ranked_indices.resize(std::min(check_count, ranked_indices.size()));

    std::vector<CoordinateCheck> checks;
    checks.reserve(ranked_indices.size());
    for (size_t error_index : ranked_indices) {
      const auto plus_dem = perturb_error_logit(preprocessed_dem, error_index, epsilon);
      const auto minus_dem = perturb_error_logit(preprocessed_dem, error_index, -epsilon);
      const double finite_difference = (conditional_nll(plus_dem, rows, beam_width) -
                                        conditional_nll(minus_dem, rows, beam_width)) /
                                       (2.0 * epsilon);
      checks.push_back({error_index, gradient[error_index], finite_difference});
    }
    print_json(beam_width, gradient.size(), sample_shots, baseline_nll, epsilon, checks);
  } catch (const std::exception& ex) {
    std::cerr << "trellis gradient validation failed: " << ex.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
