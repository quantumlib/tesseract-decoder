#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "tesseract_trellis.h"

namespace {

struct SyndromeCounts {
  std::string syndrome;
  std::vector<uint64_t> detections;
  uint64_t train_zero_count;
  uint64_t train_one_count;
  uint64_t test_zero_count;
  uint64_t test_one_count;
};

enum class FitObjective {
  Observable,
  Syndrome,
  SyndromeCensored,
};

struct ObjectiveResult {
  double data_nll;
  std::vector<double> gradient;
  uint64_t shots;
  double elapsed_seconds;
};

struct HistoryEntry {
  size_t step;
  double data_nll;
  double regularization;
  double objective;
  double gradient_norm;
  double accepted_step_size;
  double elapsed_seconds;
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
  std::string syndrome;
  uint64_t train_zero_count;
  uint64_t train_one_count;
  uint64_t test_zero_count;
  uint64_t test_one_count;
  while (input >> syndrome >> train_zero_count >> train_one_count >> test_zero_count >>
         test_one_count) {
    std::vector<uint64_t> detections;
    if (syndrome.rfind("dets:", 0) == 0) {
      const std::string encoded = syndrome.substr(5);
      if (encoded != "-") {
        size_t begin = 0;
        while (begin < encoded.size()) {
          const size_t end = encoded.find(',', begin);
          detections.push_back(std::stoull(encoded.substr(begin, end - begin)));
          if (end == std::string::npos) {
            break;
          }
          begin = end + 1;
        }
      }
    } else {
      uint64_t mask = std::stoull(syndrome);
      while (mask != 0) {
        const uint64_t bit = static_cast<uint64_t>(__builtin_ctzll(mask));
        detections.push_back(bit);
        mask &= mask - 1;
      }
    }
    rows.push_back({syndrome, std::move(detections), train_zero_count, train_one_count,
                    test_zero_count, test_one_count});
  }
  if (!input.eof()) {
    throw std::invalid_argument("malformed syndrome-count row in: " + path);
  }
  if (rows.empty()) {
    throw std::invalid_argument("no syndrome-count rows in: " + path);
  }
  return rows;
}

double probability_from_logit(double logit) {
  if (logit >= 0.0) {
    return 1.0 / (1.0 + std::exp(-logit));
  }
  const double exp_logit = std::exp(logit);
  return exp_logit / (1.0 + exp_logit);
}

std::vector<double> error_logits(const stim::DetectorErrorModel& dem) {
  std::vector<double> logits;
  for (const auto& instruction : dem.flattened().instructions) {
    if (instruction.type != stim::DemInstructionType::DEM_ERROR) {
      continue;
    }
    const double probability = instruction.arg_data[0];
    if (!std::isfinite(probability) || probability <= 0.0 || probability >= 1.0) {
      throw std::invalid_argument("all fitted DEM probabilities must lie strictly between 0 and 1");
    }
    logits.push_back(std::log(probability / (1.0 - probability)));
  }
  return logits;
}

stim::DetectorErrorModel dem_with_logits(const stim::DetectorErrorModel& dem,
                                         const std::vector<double>& logits) {
  stim::DetectorErrorModel result;
  size_t error_index = 0;
  for (const auto& instruction : dem.flattened().instructions) {
    if (instruction.type == stim::DemInstructionType::DEM_ERROR) {
      if (error_index >= logits.size()) {
        throw std::invalid_argument("too few fitted logits for DEM");
      }
      result.append_error_instruction(probability_from_logit(logits[error_index]),
                                      instruction.target_data, instruction.tag);
      ++error_index;
    } else {
      result.append_dem_instruction(instruction);
    }
  }
  if (error_index != logits.size()) {
    throw std::invalid_argument("too many fitted logits for DEM");
  }
  return result;
}

ObjectiveResult objective_and_gradient(const stim::DetectorErrorModel& dem,
                                       const std::vector<SyndromeCounts>& rows, size_t beam_width,
                                       size_t num_threads, FitObjective fit_objective,
                                       uint64_t censored_total_shots) {
  const auto started = std::chrono::steady_clock::now();
  const size_t num_errors = dem.count_errors();
  std::vector<std::vector<double>> partial_gradients(num_threads,
                                                     std::vector<double>(num_errors, 0.0));
  std::vector<long double> partial_losses(num_threads, 0.0);
  std::vector<long double> partial_selected_masses(num_threads, 0.0);
  std::vector<uint64_t> partial_shots(num_threads, 0);
  std::vector<std::vector<double>> partial_mass_gradients(num_threads,
                                                          std::vector<double>(num_errors, 0.0));
  std::vector<std::exception_ptr> errors(num_threads);
  std::vector<std::thread> workers;
  workers.reserve(num_threads);

  for (size_t thread_index = 0; thread_index < num_threads; ++thread_index) {
    workers.emplace_back([&, thread_index]() {
      try {
        TesseractTrellisConfig config;
        config.dem = dem;
        config.beam_width = beam_width;
        config.merge_errors = false;
        TesseractTrellisDecoder decoder(config);
        auto& gradient = partial_gradients[thread_index];
        for (size_t row_index = thread_index; row_index < rows.size(); row_index += num_threads) {
          const auto& row = rows[row_index];
          const uint64_t row_shots = row.train_zero_count + row.train_one_count;
          if (row_shots == 0) {
            continue;
          }
          if (fit_objective == FitObjective::Observable) {
            const auto logit_gradient =
                decoder.decode_shot_with_observable_logit_gradient(row.detections);
            const double probability = decoder.observable_probability();
            if (decoder.low_confidence_flag || !std::isfinite(probability) || probability <= 0.0 ||
                probability >= 1.0 || logit_gradient.size() != gradient.size()) {
              throw std::runtime_error("invalid observable gradient while fitting");
            }
            partial_losses[thread_index] -= row.train_one_count * std::log(probability);
            partial_losses[thread_index] -= row.train_zero_count * std::log1p(-probability);
            const double logit_loss_gradient =
                row_shots * probability - static_cast<double>(row.train_one_count);
            for (size_t error_index = 0; error_index < gradient.size(); ++error_index) {
              gradient[error_index] += logit_loss_gradient * logit_gradient[error_index];
            }
          } else {
            const auto log_probability_gradient =
                decoder.decode_shot_with_syndrome_log_probability_gradient(row.detections);
            if (decoder.low_confidence_flag || !std::isfinite(decoder.syndrome_log_probability) ||
                log_probability_gradient.size() != gradient.size()) {
              throw std::runtime_error("invalid syndrome gradient while fitting");
            }
            partial_losses[thread_index] -= row_shots * decoder.syndrome_log_probability;
            for (size_t error_index = 0; error_index < gradient.size(); ++error_index) {
              gradient[error_index] -= row_shots * log_probability_gradient[error_index];
            }
            if (fit_objective == FitObjective::SyndromeCensored) {
              const double probability = std::exp(decoder.syndrome_log_probability);
              partial_selected_masses[thread_index] += probability;
              for (size_t error_index = 0; error_index < gradient.size(); ++error_index) {
                partial_mass_gradients[thread_index][error_index] +=
                    probability * log_probability_gradient[error_index];
              }
            }
          }
          partial_shots[thread_index] += row_shots;
        }
      } catch (...) {
        errors[thread_index] = std::current_exception();
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  for (const auto& error : errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }

  ObjectiveResult result{0.0, std::vector<double>(num_errors, 0.0), 0, 0.0};
  long double total_loss = 0.0;
  long double selected_mass = 0.0;
  std::vector<double> mass_gradient(num_errors, 0.0);
  for (size_t thread_index = 0; thread_index < num_threads; ++thread_index) {
    total_loss += partial_losses[thread_index];
    selected_mass += partial_selected_masses[thread_index];
    result.shots += partial_shots[thread_index];
    for (size_t error_index = 0; error_index < num_errors; ++error_index) {
      result.gradient[error_index] += partial_gradients[thread_index][error_index];
      mass_gradient[error_index] += partial_mass_gradients[thread_index][error_index];
    }
  }
  if (result.shots == 0) {
    throw std::runtime_error("selected syndromes contain no training shots");
  }
  if (fit_objective == FitObjective::SyndromeCensored) {
    if (censored_total_shots < result.shots || selected_mass <= 0.0 || selected_mass >= 1.0) {
      throw std::runtime_error("invalid censored syndrome mass or total shot count");
    }
    const uint64_t other_shots = censored_total_shots - result.shots;
    total_loss -= other_shots * std::log1pl(-selected_mass);
    const double other_gradient_scale = other_shots / (1.0 - selected_mass);
    for (size_t error_index = 0; error_index < num_errors; ++error_index) {
      result.gradient[error_index] += other_gradient_scale * mass_gradient[error_index];
    }
    result.shots = censored_total_shots;
  }
  result.data_nll = static_cast<double>(total_loss / result.shots);
  for (double& value : result.gradient) {
    value /= result.shots;
  }
  result.elapsed_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  return result;
}

std::vector<double> decode_values(const stim::DetectorErrorModel& dem,
                                  const std::vector<SyndromeCounts>& rows, size_t beam_width,
                                  size_t num_threads, FitObjective fit_objective) {
  std::vector<double> values(rows.size(), std::numeric_limits<double>::quiet_NaN());
  std::vector<std::exception_ptr> errors(num_threads);
  std::vector<std::thread> workers;
  workers.reserve(num_threads);
  for (size_t thread_index = 0; thread_index < num_threads; ++thread_index) {
    workers.emplace_back([&, thread_index]() {
      try {
        TesseractTrellisConfig config;
        config.dem = dem;
        config.beam_width = beam_width;
        config.merge_errors = false;
        TesseractTrellisDecoder decoder(config);
        for (size_t row_index = thread_index; row_index < rows.size(); row_index += num_threads) {
          decoder.decode_shot(rows[row_index].detections);
          values[row_index] = fit_objective == FitObjective::Observable
                                  ? decoder.observable_probability()
                                  : decoder.syndrome_log_probability;
          if (decoder.low_confidence_flag || !std::isfinite(values[row_index])) {
            throw std::runtime_error("invalid trellis value while scoring");
          }
        }
      } catch (...) {
        errors[thread_index] = std::current_exception();
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  for (const auto& error : errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }
  return values;
}

double nll_from_values(const std::vector<double>& values, const std::vector<SyndromeCounts>& rows,
                       bool test, FitObjective fit_objective, uint64_t censored_total_shots) {
  long double loss = 0.0;
  long double selected_mass = 0.0;
  uint64_t shots = 0;
  for (size_t i = 0; i < rows.size(); ++i) {
    const uint64_t zeros = test ? rows[i].test_zero_count : rows[i].train_zero_count;
    const uint64_t ones = test ? rows[i].test_one_count : rows[i].train_one_count;
    if (fit_objective == FitObjective::Observable) {
      const double probability = values[i];
      if (probability <= 0.0 || probability >= 1.0) {
        throw std::runtime_error("conditional NLL requires probabilities strictly between 0 and 1");
      }
      loss -= ones * std::log(probability);
      loss -= zeros * std::log1p(-probability);
    } else {
      loss -= (zeros + ones) * values[i];
      if (fit_objective == FitObjective::SyndromeCensored) {
        selected_mass += std::exp(values[i]);
      }
    }
    shots += zeros + ones;
  }
  if (fit_objective == FitObjective::SyndromeCensored) {
    if (censored_total_shots < shots || selected_mass <= 0.0 || selected_mass >= 1.0) {
      throw std::runtime_error("invalid censored syndrome mass or total shot count while scoring");
    }
    loss -= (censored_total_shots - shots) * std::log1pl(-selected_mass);
    shots = censored_total_shots;
  }
  return static_cast<double>(loss / shots);
}

void write_scores(const std::string& path, const std::vector<SyndromeCounts>& rows,
                  const std::vector<double>& baseline_probabilities,
                  const std::vector<double>& fitted_probabilities) {
  std::ofstream output(path);
  if (!output) {
    throw std::invalid_argument("could not write scores: " + path);
  }
  output << "syndrome train_zero train_one test_zero test_one baseline_q fitted_q\n";
  output << std::setprecision(17);
  for (size_t i = 0; i < rows.size(); ++i) {
    output << rows[i].syndrome << ' ' << rows[i].train_zero_count << ' ' << rows[i].train_one_count
           << ' ' << rows[i].test_zero_count << ' ' << rows[i].test_one_count << ' '
           << baseline_probabilities[i] << ' ' << fitted_probabilities[i] << '\n';
  }
}

void write_summary(const std::string& path, size_t beam_width, size_t num_threads,
                   size_t num_errors, size_t num_syndromes, uint64_t train_shots,
                   uint64_t test_shots, size_t steps, double learning_rate, double l2,
                   double max_shift, double baseline_test_nll, double fitted_test_nll,
                   double final_shift_l2, double final_shift_max, size_t shifts_at_bound,
                   double total_seconds, FitObjective fit_objective,
                   const std::vector<HistoryEntry>& history) {
  std::ofstream output(path);
  if (!output) {
    throw std::invalid_argument("could not write summary: " + path);
  }
  output << std::setprecision(17);
  output << "{\n"
         << "  \"schema_version\": 2,\n"
         << "  \"fit_objective\": \""
         << (fit_objective == FitObjective::Observable         ? "observable"
             : fit_objective == FitObjective::SyndromeCensored ? "syndrome_censored"
                                                               : "syndrome")
         << "\",\n"
         << "  \"beam_width\": " << beam_width << ",\n"
         << "  \"threads\": " << num_threads << ",\n"
         << "  \"preprocessed_errors\": " << num_errors << ",\n"
         << "  \"syndromes\": " << num_syndromes << ",\n"
         << "  \"train_shots\": " << train_shots << ",\n"
         << "  \"test_shots\": " << test_shots << ",\n"
         << "  \"steps\": " << steps << ",\n"
         << "  \"learning_rate\": " << learning_rate << ",\n"
         << "  \"l2\": " << l2 << ",\n"
         << "  \"max_logit_shift\": " << max_shift << ",\n"
         << "  \"baseline_test_nll\": " << baseline_test_nll << ",\n"
         << "  \"fitted_test_nll\": " << fitted_test_nll << ",\n"
         << "  \"final_shift_l2\": " << final_shift_l2 << ",\n"
         << "  \"final_shift_max\": " << final_shift_max << ",\n"
         << "  \"shifts_at_bound\": " << shifts_at_bound << ",\n"
         << "  \"total_seconds\": " << total_seconds << ",\n"
         << "  \"history\": [\n";
  for (size_t i = 0; i < history.size(); ++i) {
    const auto& entry = history[i];
    output << "    {\"step\": " << entry.step << ", \"data_nll\": " << entry.data_nll
           << ", \"regularization\": " << entry.regularization
           << ", \"objective\": " << entry.objective
           << ", \"gradient_norm\": " << entry.gradient_norm
           << ", \"accepted_step_size\": " << entry.accepted_step_size
           << ", \"elapsed_seconds\": " << entry.elapsed_seconds << "}";
    output << (i + 1 == history.size() ? "\n" : ",\n");
  }
  output << "  ]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 12 && argc != 13 && argc != 15) {
    std::cerr << "usage: trellis_fit_pilot DEM COUNTS OUT_DEM OUT_SCORES OUT_SUMMARY BEAM "
                 "THREADS STEPS LEARNING_RATE L2 MAX_SHIFT "
                 "[observable|syndrome|syndrome_censored [TRAIN_TOTAL TEST_TOTAL]]\n";
    return EXIT_FAILURE;
  }
  try {
    const std::string dem_path = argv[1];
    const std::string counts_path = argv[2];
    const std::string out_dem_path = argv[3];
    const std::string out_scores_path = argv[4];
    const std::string out_summary_path = argv[5];
    const size_t beam_width = std::stoull(argv[6]);
    const size_t num_threads = std::stoull(argv[7]);
    const size_t steps = std::stoull(argv[8]);
    const double learning_rate = std::stod(argv[9]);
    const double l2 = std::stod(argv[10]);
    const double max_shift = std::stod(argv[11]);
    const std::string fit_objective_name = argc >= 13 ? argv[12] : "observable";
    const FitObjective fit_objective =
        fit_objective_name == "observable" ? FitObjective::Observable
        : fit_objective_name == "syndrome" ? FitObjective::Syndrome
        : fit_objective_name == "syndrome_censored"
            ? FitObjective::SyndromeCensored
            : throw std::invalid_argument("fit objective must be observable or syndrome");
    const uint64_t train_total_shots = argc == 15 ? std::stoull(argv[13]) : 0;
    const uint64_t test_total_shots = argc == 15 ? std::stoull(argv[14]) : 0;
    if (beam_width == 0 || num_threads == 0 || num_threads > 2 || steps == 0 ||
        !std::isfinite(learning_rate) || learning_rate <= 0.0 || !std::isfinite(l2) || l2 < 0.0 ||
        !std::isfinite(max_shift) || max_shift <= 0.0 ||
        ((fit_objective == FitObjective::SyndromeCensored) != (argc == 15))) {
      throw std::invalid_argument("invalid fit parameter");
    }

    const auto total_started = std::chrono::steady_clock::now();
    const auto rows = read_counts(counts_path);
    TesseractTrellisConfig initial_config;
    initial_config.dem = read_dem(dem_path);
    initial_config.beam_width = beam_width;
    TesseractTrellisDecoder initial_decoder(initial_config);
    const stim::DetectorErrorModel baseline_dem = initial_decoder.config.dem;
    const std::vector<double> baseline_logits = error_logits(baseline_dem);
    std::vector<double> logits = baseline_logits;
    std::vector<HistoryEntry> history;
    history.reserve(steps + 1);

    for (size_t step = 0; step < steps; ++step) {
      const auto current_dem = dem_with_logits(baseline_dem, logits);
      auto objective = objective_and_gradient(current_dem, rows, beam_width, num_threads,
                                              fit_objective, train_total_shots);
      double regularization = 0.0;
      for (size_t i = 0; i < logits.size(); ++i) {
        const double shift = logits[i] - baseline_logits[i];
        regularization += 0.5 * l2 * shift * shift;
        objective.gradient[i] += l2 * shift;
      }
      double gradient_norm_squared = 0.0;
      for (double value : objective.gradient) {
        gradient_norm_squared += value * value;
      }
      const double gradient_norm = std::sqrt(gradient_norm_squared);
      const double current_objective = objective.data_nll + regularization;
      double accepted_step_size = 0.0;
      double candidate_step_size = learning_rate;
      std::vector<double> candidate_logits(logits.size());
      for (size_t attempt = 0; attempt < 12 && gradient_norm > 0.0; ++attempt) {
        for (size_t i = 0; i < logits.size(); ++i) {
          const double updated =
              logits[i] - candidate_step_size * objective.gradient[i] / gradient_norm;
          candidate_logits[i] =
              std::clamp(updated, baseline_logits[i] - max_shift, baseline_logits[i] + max_shift);
        }
        const auto candidate_dem = dem_with_logits(baseline_dem, candidate_logits);
        const auto candidate_values =
            decode_values(candidate_dem, rows, beam_width, num_threads, fit_objective);
        const double candidate_data_nll =
            nll_from_values(candidate_values, rows, false, fit_objective, train_total_shots);
        double candidate_regularization = 0.0;
        for (size_t i = 0; i < logits.size(); ++i) {
          const double shift = candidate_logits[i] - baseline_logits[i];
          candidate_regularization += 0.5 * l2 * shift * shift;
        }
        if (candidate_data_nll + candidate_regularization < current_objective) {
          logits = candidate_logits;
          accepted_step_size = candidate_step_size;
          break;
        }
        candidate_step_size *= 0.5;
      }
      history.push_back({step, objective.data_nll, regularization, current_objective, gradient_norm,
                         accepted_step_size, objective.elapsed_seconds});
      std::cerr << "step " << step << '/' << steps << " data_nll=" << std::setprecision(10)
                << objective.data_nll << " objective=" << current_objective
                << " grad_norm=" << gradient_norm << " accepted_step=" << accepted_step_size
                << " seconds=" << objective.elapsed_seconds << std::endl;
      if (accepted_step_size == 0.0) {
        std::cerr << "stopping because backtracking found no improving step" << std::endl;
        break;
      }
    }

    const auto fitted_dem = dem_with_logits(baseline_dem, logits);
    auto final_objective = objective_and_gradient(fitted_dem, rows, beam_width, num_threads,
                                                  fit_objective, train_total_shots);
    double final_regularization = 0.0;
    double final_gradient_norm_squared = 0.0;
    double final_shift_l2_squared = 0.0;
    double final_shift_max = 0.0;
    size_t shifts_at_bound = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
      const double shift = logits[i] - baseline_logits[i];
      final_regularization += 0.5 * l2 * shift * shift;
      final_objective.gradient[i] += l2 * shift;
      final_shift_l2_squared += shift * shift;
      final_shift_max = std::max(final_shift_max, std::abs(shift));
      if (std::abs(shift) >= max_shift - 1e-12) {
        ++shifts_at_bound;
      }
    }
    for (double value : final_objective.gradient) {
      final_gradient_norm_squared += value * value;
    }
    history.push_back({steps, final_objective.data_nll, final_regularization,
                       final_objective.data_nll + final_regularization,
                       std::sqrt(final_gradient_norm_squared), 0.0,
                       final_objective.elapsed_seconds});

    const auto baseline_probabilities =
        decode_values(baseline_dem, rows, beam_width, num_threads, FitObjective::Observable);
    const auto fitted_probabilities =
        decode_values(fitted_dem, rows, beam_width, num_threads, FitObjective::Observable);
    const auto baseline_test_values =
        fit_objective == FitObjective::Observable
            ? baseline_probabilities
            : decode_values(baseline_dem, rows, beam_width, num_threads, fit_objective);
    const auto fitted_test_values =
        fit_objective == FitObjective::Observable
            ? fitted_probabilities
            : decode_values(fitted_dem, rows, beam_width, num_threads, fit_objective);
    const double baseline_test_nll =
        nll_from_values(baseline_test_values, rows, true, fit_objective, test_total_shots);
    const double fitted_test_nll =
        nll_from_values(fitted_test_values, rows, true, fit_objective, test_total_shots);
    uint64_t train_shots = std::accumulate(
        rows.begin(), rows.end(), uint64_t{0}, [](uint64_t total, const SyndromeCounts& row) {
          return total + row.train_zero_count + row.train_one_count;
        });
    uint64_t test_shots = std::accumulate(rows.begin(), rows.end(), uint64_t{0},
                                          [](uint64_t total, const SyndromeCounts& row) {
                                            return total + row.test_zero_count + row.test_one_count;
                                          });
    if (fit_objective == FitObjective::SyndromeCensored) {
      train_shots = train_total_shots;
      test_shots = test_total_shots;
    }

    std::ofstream dem_output(out_dem_path);
    if (!dem_output) {
      throw std::invalid_argument("could not write fitted DEM: " + out_dem_path);
    }
    dem_output << fitted_dem.str();
    write_scores(out_scores_path, rows, baseline_probabilities, fitted_probabilities);
    const double total_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - total_started).count();
    write_summary(out_summary_path, beam_width, num_threads, logits.size(), rows.size(),
                  train_shots, test_shots, steps, learning_rate, l2, max_shift, baseline_test_nll,
                  fitted_test_nll, std::sqrt(final_shift_l2_squared), final_shift_max,
                  shifts_at_bound, total_seconds, fit_objective, history);
    std::cerr << "finished fit in " << total_seconds << " seconds; held-out NLL "
              << baseline_test_nll << " -> " << fitted_test_nll << std::endl;
  } catch (const std::exception& ex) {
    std::cerr << "trellis pilot fit failed: " << ex.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
