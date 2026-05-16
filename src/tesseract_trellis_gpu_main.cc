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

#include <argparse/argparse.hpp>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <thread>

#include "common.h"
#include "stim.h"
#include "tesseract_trellis.h"
#include "tesseract_trellis_gpu.h"
#include "utils.h"

namespace {

TesseractTrellisRankingMode parse_ranking_mode(const std::string& value) {
  if (value == "mass") return TesseractTrellisRankingMode::MassOnly;
  if (value == "future-detcost") return TesseractTrellisRankingMode::FutureDetcostRanked;
  if (value == "future-active-detcost") {
    return TesseractTrellisRankingMode::FutureActiveDetcostRanked;
  }
  throw std::invalid_argument("Unknown trellis ranking mode: " + value);
}

double confidence_margin(double mass0, double mass1) {
  const double total = mass0 + mass1;
  if (!(total > 0.0) || !std::isfinite(total)) {
    return 0.0;
  }
  return std::abs(mass1 - mass0) / total;
}

double median_of(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  const size_t mid = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + mid, values.end());
  if ((values.size() & 1u) != 0) {
    return values[mid];
  }
  const double hi = values[mid];
  std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
  return 0.5 * (values[mid - 1] + hi);
}

}  // namespace

struct Args {
  std::string circuit_path;
  std::string dem_path;

  size_t sample_num_shots = 0;
  size_t max_errors = SIZE_MAX;
  uint64_t sample_seed;

  size_t shot_range_begin = 0;
  size_t shot_range_end = 0;

  std::string in_fname = "";
  std::string in_format = "";
  std::string obs_in_fname = "";
  std::string obs_in_format = "";
  bool append_observables = false;
  std::string out_fname = "";
  std::string out_format = "";

  std::string dem_out_fname = "";
  std::string stats_out_fname = "";

  size_t num_threads = 1;
  size_t gpu_shot_batch_size = 1024;
  size_t gpu_merge_period = 10000;
  size_t beam_width = 1024;
  size_t adaptive_beam_width = 0;
  double adaptive_confidence_threshold = 0.0;
  bool adaptive_dynamic_beam = false;
  double future_detcost_scale = 2.0;
  std::string ranking_mode = "mass";

  bool verbose = false;
  bool print_stats = false;

  bool has_observables() {
    return append_observables || !obs_in_fname.empty() || (sample_num_shots > 0);
  }

  void validate() {
    if (circuit_path.empty() && dem_path.empty()) {
      throw std::invalid_argument("Must provide at least one of --circuit or --dem");
    }
    int num_data_sources = int(sample_num_shots > 0) + int(!in_fname.empty());
    if (num_data_sources != 1) {
      throw std::invalid_argument("Requires exactly 1 source of shots.");
    }
    if (!in_fname.empty() && in_format.empty()) {
      throw std::invalid_argument("If --in is provided, must also specify --in-format.");
    }
    if (!out_fname.empty() && out_format.empty()) {
      throw std::invalid_argument("If --out is provided, must also specify --out-format.");
    }
    if (!in_format.empty() && !stim::format_name_to_enum_map().contains(in_format)) {
      throw std::invalid_argument("Invalid format: " + in_format);
    }
    if (!obs_in_format.empty() && !stim::format_name_to_enum_map().contains(obs_in_format)) {
      throw std::invalid_argument("Invalid format: " + obs_in_format);
    }
    if (!out_format.empty() && !stim::format_name_to_enum_map().contains(out_format)) {
      throw std::invalid_argument("Invalid format: " + out_format);
    }
    if (!obs_in_fname.empty() && in_fname.empty()) {
      throw std::invalid_argument(
          "Cannot load observable flips without a corresponding detection event data file.");
    }
    if (num_threads == 0) {
      throw std::invalid_argument("--threads must be at least 1.");
    }
    if (num_threads > 1000) {
      throw std::invalid_argument("There is a maximum limit of 1000 threads.");
    }
    if (gpu_shot_batch_size == 0) {
      throw std::invalid_argument("--gpu-shot-batch must be at least 1.");
    }
    if (gpu_merge_period == 0) {
      throw std::invalid_argument("--gpu-merge-period must be at least 1.");
    }
    if ((shot_range_begin || shot_range_end) && shot_range_end < shot_range_begin) {
      throw std::invalid_argument("Provided shot range must have end >= begin.");
    }
    if (sample_num_shots > 0 && circuit_path.empty()) {
      throw std::invalid_argument("Cannot sample shots without a circuit.");
    }
    if (beam_width == 0) {
      throw std::invalid_argument("--beam must be at least 1.");
    }
    if (adaptive_beam_width != 0 && adaptive_beam_width <= beam_width) {
      throw std::invalid_argument("--adaptive-beam must be larger than --beam.");
    }
    if (adaptive_dynamic_beam && adaptive_beam_width == 0) {
      throw std::invalid_argument("--adaptive-dynamic-beam requires --adaptive-beam.");
    }
    if (!std::isfinite(adaptive_confidence_threshold) || adaptive_confidence_threshold < 0.0 ||
        adaptive_confidence_threshold > 1.0) {
      throw std::invalid_argument("--adaptive-confidence-threshold must satisfy 0 <= threshold <= 1.");
    }
    if (!std::isfinite(future_detcost_scale) || future_detcost_scale < 0.0) {
      throw std::invalid_argument("--future-detcost-scale must be finite and nonnegative.");
    }
    parse_ranking_mode(ranking_mode);
  }

  void extract(TesseractTrellisConfig& config, std::vector<stim::SparseShot>& shots,
               std::unique_ptr<stim::MeasureRecordWriter>& writer) {
    stim::Circuit circuit;
    if (!circuit_path.empty()) {
      FILE* file = fopen(circuit_path.c_str(), "r");
      if (!file) {
        throw std::invalid_argument("Could not open the file: " + circuit_path);
      }
      circuit = stim::Circuit::from_file(file);
      fclose(file);
    }

    if (!dem_path.empty()) {
      FILE* file = fopen(dem_path.c_str(), "r");
      if (!file) {
        throw std::invalid_argument("Could not open the file: " + dem_path);
      }
      config.dem = stim::DetectorErrorModel::from_file(file);
      fclose(file);
    } else {
      assert(!circuit_path.empty());
      config.dem = stim::ErrorAnalyzer::circuit_to_detector_error_model(
          circuit, /*decompose_errors=*/false, /*fold_loops=*/true,
          /*allow_gauge_detectors=*/true,
          /*approximate_disjoint_errors_threshold=*/1,
          /*ignore_decomposition_failures=*/false,
          /*block_decomposition_from_introducing_remnant_edges=*/false);
    }

    config.beam_width = adaptive_dynamic_beam ? adaptive_beam_width : beam_width;
    config.future_detcost_scale = future_detcost_scale;
    config.gpu_merge_period = gpu_merge_period;
    if (adaptive_dynamic_beam) {
      config.gpu_dynamic_initial_beam_width = beam_width;
      config.gpu_dynamic_confidence_threshold = adaptive_confidence_threshold;
    }
    config.verbose = verbose;
    config.track_kept_state_stats = print_stats;
    config.ranking_mode = parse_ranking_mode(ranking_mode);

    if (sample_num_shots > 0) {
      assert(!circuit_path.empty());
      std::mt19937_64 rng(sample_seed);
      size_t num_detectors = circuit.count_detectors();
      const auto [dets, obs] =
          stim::sample_batch_detection_events<64>(circuit, sample_num_shots, rng);
      stim::simd_bit_table<64> obs_T = obs.transposed();
      shots.resize(sample_num_shots);
      for (size_t k = 0; k < sample_num_shots; k++) {
        shots[k].obs_mask = obs_T[k];
        for (size_t d = 0; d < num_detectors; d++) {
          if (dets[d][k]) {
            shots[k].hits.push_back(d);
          }
        }
      }
    }

    if (!in_fname.empty()) {
      FILE* shots_file = fopen(in_fname.c_str(), "r");
      if (!shots_file) {
        throw std::invalid_argument("Could not open the file: " + in_fname);
      }
      stim::FileFormatData shots_in_format = stim::format_name_to_enum_map().at(in_format);
      auto reader = stim::MeasureRecordReader<stim::MAX_BITWORD_WIDTH>::make(
          shots_file, shots_in_format.id, 0, config.dem.count_detectors(),
          append_observables * config.dem.count_observables());
      stim::SparseShot sparse_shot;
      sparse_shot.clear();
      while (reader->start_and_read_entire_record(sparse_shot)) {
        shots.push_back(sparse_shot);
        sparse_shot.clear();
      }
      fclose(shots_file);
    }

    if (!obs_in_fname.empty()) {
      FILE* obs_file = fopen(obs_in_fname.c_str(), "r");
      if (!obs_file) {
        throw std::invalid_argument("Could not open the file: " + obs_in_fname);
      }
      stim::FileFormatData obs_format = stim::format_name_to_enum_map().at(obs_in_format);
      auto obs_reader = stim::MeasureRecordReader<stim::MAX_BITWORD_WIDTH>::make(
          obs_file, obs_format.id, 0, 0, config.dem.count_observables());
      stim::SparseShot sparse_shot;
      sparse_shot.clear();
      size_t num_obs_shots = 0;
      while (obs_reader->start_and_read_entire_record(sparse_shot)) {
        if (num_obs_shots >= shots.size()) {
          throw std::invalid_argument("Shot data ended before obs data.");
        }
        shots[num_obs_shots].obs_mask = sparse_shot.obs_mask;
        sparse_shot.clear();
        ++num_obs_shots;
      }
      if (num_obs_shots != shots.size()) {
        throw std::invalid_argument("Obs data ended before shot data ended.");
      }
      fclose(obs_file);
    }

    if (shot_range_begin || shot_range_end) {
      if (shot_range_end > shots.size()) {
        throw std::invalid_argument("Shot range end is past end of shots array.");
      }
      std::vector<stim::SparseShot> shots_in_range(shots.begin() + shot_range_begin,
                                                   shots.begin() + shot_range_end);
      std::swap(shots_in_range, shots);
    }

    if (!out_fname.empty()) {
      stim::FileFormatData predictions_out_format = stim::format_name_to_enum_map().at(out_format);
      FILE* predictions_file = stdout;
      if (out_fname != "-") {
        predictions_file = fopen(out_fname.c_str(), "w");
      }
      writer = stim::MeasureRecordWriter::make(predictions_file, predictions_out_format.id);
      writer->begin_result_type('L');
    }
  }
};

int main(int argc, char* argv[]) {
  std::cout.precision(16);
  argparse::ArgumentParser program("tesseract_trellis_gpu");
  Args args;
  program.add_argument("--circuit").help("Stim circuit file path").store_into(args.circuit_path);
  program.add_argument("--dem").help("Stim dem file path").store_into(args.dem_path);
  program.add_argument("--sample-num-shots").store_into(args.sample_num_shots);
  program.add_argument("--max-errors").store_into(args.max_errors);
  program.add_argument("--sample-seed")
      .default_value(static_cast<uint64_t>(std::random_device()()))
      .store_into(args.sample_seed);
  program.add_argument("--shot-range-begin")
      .default_value(size_t(0))
      .store_into(args.shot_range_begin);
  program.add_argument("--shot-range-end").default_value(size_t(0)).store_into(args.shot_range_end);
  program.add_argument("--in").default_value(std::string("")).store_into(args.in_fname);
  program.add_argument("--in-format", "--in_format")
      .default_value(std::string(""))
      .store_into(args.in_format);
  program.add_argument("--in-includes-appended-observables", "--in_includes_appended_observables")
      .default_value(false)
      .store_into(args.append_observables)
      .flag();
  program.add_argument("--obs_in", "--obs-in")
      .default_value(std::string(""))
      .store_into(args.obs_in_fname);
  program.add_argument("--obs-in-format", "--obs_in_format")
      .default_value(std::string(""))
      .store_into(args.obs_in_format);
  program.add_argument("--out").default_value(std::string("")).store_into(args.out_fname);
  program.add_argument("--out-format").default_value(std::string("")).store_into(args.out_format);
  program.add_argument("--dem-out").default_value(std::string("")).store_into(args.dem_out_fname);
  program.add_argument("--stats-out")
      .default_value(std::string(""))
      .store_into(args.stats_out_fname);
  program.add_argument("--threads")
      .default_value(size_t(
          std::thread::hardware_concurrency() == 0 ? 1 : std::thread::hardware_concurrency()))
      .store_into(args.num_threads);
  program.add_argument("--gpu-shot-batch")
      .help("Batch this many shots together in the GPU-oriented runtime path.")
      .default_value(size_t(1024))
      .store_into(args.gpu_shot_batch_size);
  program.add_argument("--gpu-merge-period")
      .help("In the GPU-oriented runtime, only do exact duplicate-state merge every N layers.")
      .default_value(size_t(10000))
      .store_into(args.gpu_merge_period);
  program.add_argument("--beam").default_value(size_t(1024)).store_into(args.beam_width);
  program.add_argument("--adaptive-beam")
      .help("If nonzero, rerun low-confidence shots using this larger beam.")
      .default_value(size_t(0))
      .store_into(args.adaptive_beam_width);
  program.add_argument("--adaptive-confidence-threshold")
      .help(
          "Rerun shots whose final |obs1_mass - obs0_mass| / (obs1_mass + obs0_mass) is at or "
          "below this threshold.")
      .default_value(0.0)
      .store_into(args.adaptive_confidence_threshold);
  program.add_argument("--adaptive-dynamic-beam")
      .help(
          "Use --beam as the starting active beam and --adaptive-beam as the maximum GPU beam. "
          "The active beam grows during saturated top-k steps when candidate observable mass "
          "confidence is at or below --adaptive-confidence-threshold.")
      .flag()
      .store_into(args.adaptive_dynamic_beam);
  program.add_argument("--ranking-mode")
      .help("Trellis ranking mode: mass, future-detcost, or future-active-detcost")
      .default_value(std::string("mass"))
      .store_into(args.ranking_mode);
  program.add_argument("--future-detcost-scale")
      .help("Multiplier applied to future detector-cost ranking penalties.")
      .default_value(2.0)
      .store_into(args.future_detcost_scale);
  program.add_argument("--verbose").flag().store_into(args.verbose);
  program.add_argument("--print-stats").flag().store_into(args.print_stats);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return EXIT_FAILURE;
  }

  args.validate();
  TesseractTrellisConfig config;
  std::vector<stim::SparseShot> shots;
  std::unique_ptr<stim::MeasureRecordWriter> writer;
  args.extract(config, shots, writer);

  TesseractTrellisGpuDecoder probe(config);
  if (args.verbose || !probe.using_gpu()) {
    std::cerr << "backend = " << probe.backend_description() << std::endl;
  }
  const size_t execution_threads = probe.using_gpu() ? 1 : args.num_threads;
  if (probe.using_gpu() && args.num_threads != 1) {
    std::cerr << "backend = metal-gpu; forcing --threads=1 because this prototype uses one GPU queue"
              << std::endl;
  }
  if (probe.using_gpu() && args.verbose) {
    std::cerr << "gpu_shot_batch_size = " << args.gpu_shot_batch_size << std::endl;
    std::cerr << "gpu_merge_period = " << args.gpu_merge_period << std::endl;
  }

  std::vector<uint64_t> obs_predicted(shots.size());
  std::vector<double> mass0_predicted(shots.size());
  std::vector<double> mass1_predicted(shots.size());
  std::vector<double> decoding_time_seconds(shots.size());
  std::vector<size_t> num_states_expanded_per_shot(shots.size());
  std::vector<size_t> num_states_merged_per_shot(shots.size());
  std::vector<size_t> max_beam_size_per_shot(shots.size());
  std::vector<size_t> max_frontier_width_per_shot(shots.size());
  std::vector<size_t> kept_state_min_per_shot(shots.size());
  std::vector<double> kept_state_median_per_shot(shots.size());
  std::vector<double> kept_state_mean_per_shot(shots.size());
  std::vector<size_t> kept_state_max_per_shot(shots.size());
  std::vector<double> time_expand_per_shot(shots.size());
  std::vector<double> time_collapse_per_shot(shots.size());
  std::vector<double> time_truncate_per_shot(shots.size());
  std::vector<double> time_reconstruct_per_shot(shots.size());
  size_t merge_calls_total = 0;
  size_t merge_input_candidates_total = 0;
  size_t merge_output_candidates_total = 0;
  size_t merge_duplicate_layers_total = 0;
  size_t merge_skipped_layers_total = 0;
  const bool adaptive_enabled = args.adaptive_beam_width != 0 && !args.adaptive_dynamic_beam;
  size_t adaptive_rerun_shots = 0;
  size_t adaptive_first_pass_errors = 0;
  double adaptive_first_pass_time_seconds = 0.0;
  double adaptive_rerun_time_seconds = 0.0;
  double confidence_margin_min = 0.0;
  double confidence_margin_median = 0.0;
  double confidence_margin_mean = 0.0;
  double confidence_margin_max = 0.0;
  nlohmann::json confidence_threshold_sweep = nlohmann::json::array();
  size_t gpu_dynamic_beam_limit_sample_count = 0;
  size_t gpu_dynamic_beam_limit_min = 0;
  double gpu_dynamic_beam_limit_median = 0.0;
  double gpu_dynamic_beam_limit_mean = 0.0;
  size_t gpu_dynamic_beam_limit_max = 0;
  size_t gpu_dynamic_beam_grow_events = 0;
  std::vector<std::atomic<bool>> low_confidence(shots.size());
  std::vector<std::unique_ptr<TesseractTrellisGpuDecoder>> decoders(execution_threads);

  bool has_obs = args.has_observables();
  size_t num_errors = 0;
  size_t num_low_confidence = 0;
  double total_time_seconds = 0;
  size_t num_observables = config.dem.count_observables();

  size_t shot = 0;
  if (probe.using_gpu()) {
    decoders[0] = std::make_unique<TesseractTrellisGpuDecoder>(config);
    auto& decoder = *decoders[0];
    if (args.max_errors == SIZE_MAX) {
      auto start_time = std::chrono::high_resolution_clock::now();
      decoder.decode_shots_batched(shots, obs_predicted, args.gpu_shot_batch_size,
                                   &mass0_predicted, &mass1_predicted);
      auto stop_time = std::chrono::high_resolution_clock::now();
      adaptive_first_pass_time_seconds =
          std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count() /
          1e6;
      total_time_seconds = adaptive_first_pass_time_seconds;
      shot = shots.size();
      const double per_shot_seconds = shots.empty() ? 0.0 : adaptive_first_pass_time_seconds / shots.size();
      merge_calls_total = decoder.merge_calls;
      merge_input_candidates_total = decoder.merge_input_candidates;
      merge_output_candidates_total = decoder.merge_output_candidates;
      merge_duplicate_layers_total = decoder.merge_duplicate_layers;
      merge_skipped_layers_total = decoder.merge_skipped_layers;

      std::vector<double> confidence_margins(shots.size());
      std::vector<size_t> rerun_indices;
      rerun_indices.reserve(shots.size());
      for (size_t shot_index = 0; shot_index < shots.size(); ++shot_index) {
        decoding_time_seconds[shot_index] = per_shot_seconds;
        const double margin = confidence_margin(mass0_predicted[shot_index],
                                                mass1_predicted[shot_index]);
        confidence_margins[shot_index] = margin;
        const bool should_rerun =
            adaptive_enabled && margin <= args.adaptive_confidence_threshold;
        low_confidence[shot_index] = should_rerun;
        if (should_rerun) {
          rerun_indices.push_back(shot_index);
        }
        if (obs_predicted[shot_index] != shots[shot_index].obs_mask_as_u64()) {
          ++adaptive_first_pass_errors;
        }
      }
      if (!confidence_margins.empty()) {
        auto [min_it, max_it] =
            std::minmax_element(confidence_margins.begin(), confidence_margins.end());
        confidence_margin_min = *min_it;
        confidence_margin_max = *max_it;
        confidence_margin_median = median_of(confidence_margins);
        double sum = 0.0;
        for (double v : confidence_margins) {
          sum += v;
        }
        confidence_margin_mean = sum / confidence_margins.size();
      }
      if (has_obs) {
        const std::vector<double> thresholds = {0.0,    1e-12, 1e-9, 1e-6, 1e-4,
                                                1e-3,  1e-2,  0.05, 0.1,  0.2,
                                                0.5,   0.9,   1.0};
        for (double threshold : thresholds) {
          size_t count_leq = 0;
          size_t first_pass_errors_leq = 0;
          size_t first_pass_errors_gt = 0;
          for (size_t shot_index = 0; shot_index < shots.size(); ++shot_index) {
            const bool is_error =
                obs_predicted[shot_index] != shots[shot_index].obs_mask_as_u64();
            if (confidence_margins[shot_index] <= threshold) {
              ++count_leq;
              if (is_error) {
                ++first_pass_errors_leq;
              }
            } else if (is_error) {
              ++first_pass_errors_gt;
            }
          }
          confidence_threshold_sweep.push_back(
              {{"threshold", threshold},
               {"count_leq", count_leq},
               {"first_pass_errors_leq", first_pass_errors_leq},
               {"first_pass_errors_gt", first_pass_errors_gt}});
        }
      }
      gpu_dynamic_beam_limit_sample_count = decoder.gpu_dynamic_beam_limit_sample_count;
      gpu_dynamic_beam_limit_min = decoder.gpu_dynamic_beam_limit_min;
      gpu_dynamic_beam_limit_median = decoder.gpu_dynamic_beam_limit_median;
      gpu_dynamic_beam_limit_mean = decoder.gpu_dynamic_beam_limit_mean;
      gpu_dynamic_beam_limit_max = decoder.gpu_dynamic_beam_limit_max;
      gpu_dynamic_beam_grow_events = decoder.gpu_dynamic_beam_grow_events;

      if (adaptive_enabled && !rerun_indices.empty()) {
        TesseractTrellisConfig rerun_config = config;
        rerun_config.beam_width = args.adaptive_beam_width;
        TesseractTrellisGpuDecoder rerun_decoder(rerun_config);
        std::vector<stim::SparseShot> rerun_shots;
        rerun_shots.reserve(rerun_indices.size());
        for (size_t index : rerun_indices) {
          rerun_shots.push_back(shots[index]);
        }
        std::vector<uint64_t> rerun_obs_predicted;
        std::vector<double> rerun_mass0;
        std::vector<double> rerun_mass1;
        auto rerun_start_time = std::chrono::high_resolution_clock::now();
        rerun_decoder.decode_shots_batched(rerun_shots, rerun_obs_predicted,
                                           args.gpu_shot_batch_size, &rerun_mass0, &rerun_mass1);
        auto rerun_stop_time = std::chrono::high_resolution_clock::now();
        adaptive_rerun_time_seconds =
            std::chrono::duration_cast<std::chrono::microseconds>(rerun_stop_time - rerun_start_time)
                .count() /
            1e6;
        adaptive_rerun_shots = rerun_indices.size();
        total_time_seconds += adaptive_rerun_time_seconds;
        merge_calls_total += rerun_decoder.merge_calls;
        merge_input_candidates_total += rerun_decoder.merge_input_candidates;
        merge_output_candidates_total += rerun_decoder.merge_output_candidates;
        merge_duplicate_layers_total += rerun_decoder.merge_duplicate_layers;
        merge_skipped_layers_total += rerun_decoder.merge_skipped_layers;
        for (size_t k = 0; k < rerun_indices.size(); ++k) {
          const size_t shot_index = rerun_indices[k];
          obs_predicted[shot_index] = rerun_obs_predicted[k];
          mass0_predicted[shot_index] = rerun_mass0[k];
          mass1_predicted[shot_index] = rerun_mass1[k];
          decoding_time_seconds[shot_index] +=
              adaptive_rerun_time_seconds / std::max<size_t>(1, adaptive_rerun_shots);
        }
      }

      num_low_confidence = adaptive_rerun_shots;
      for (size_t shot_index = 0; shot_index < shots.size(); ++shot_index) {
        if (writer) {
          writer->write_bits((uint8_t*)&obs_predicted[shot_index], num_observables);
          writer->write_end();
        }
        if (obs_predicted[shot_index] != shots[shot_index].obs_mask_as_u64()) {
          ++num_errors;
        }
      }
      if (args.print_stats) {
        std::cout << "gpu_batch_stats"
                  << " gpu_shot_batch = " << args.gpu_shot_batch_size
                  << " gpu_merge_period = " << args.gpu_merge_period
                  << " merge_calls = " << merge_calls_total
                  << " merge_duplicate_layers = " << merge_duplicate_layers_total
                  << " merge_skipped_layers = " << merge_skipped_layers_total
                  << " adaptive_rerun_shots = " << adaptive_rerun_shots
                  << " dynamic_beam_limit_mean = " << gpu_dynamic_beam_limit_mean
                  << " dynamic_beam_grow_events = " << gpu_dynamic_beam_grow_events
                  << " total_wall_time_seconds = " << total_time_seconds << '\n';
      }
    } else {
      for (size_t batch_start = 0; batch_start < shots.size();
           batch_start += args.gpu_shot_batch_size) {
        const size_t batch_end = std::min(batch_start + args.gpu_shot_batch_size, shots.size());
        for (size_t shot_index = batch_start; shot_index < batch_end; ++shot_index) {
          auto start_time = std::chrono::high_resolution_clock::now();
          decoder.decode_shot(shots[shot_index].hits);
          auto stop_time = std::chrono::high_resolution_clock::now();
          decoding_time_seconds[shot_index] =
              std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count() /
              1e6;
          obs_predicted[shot_index] = decoder.predicted_obs_mask;
          low_confidence[shot_index] = decoder.low_confidence_flag;
          mass0_predicted[shot_index] = decoder.total_mass_obs0;
          mass1_predicted[shot_index] = decoder.total_mass_obs1;
          num_states_expanded_per_shot[shot_index] = decoder.num_states_expanded;
          num_states_merged_per_shot[shot_index] = decoder.num_states_merged;
          max_beam_size_per_shot[shot_index] = decoder.max_beam_size_seen;
          max_frontier_width_per_shot[shot_index] = decoder.max_frontier_width_seen;
          kept_state_min_per_shot[shot_index] = decoder.kept_state_min;
          kept_state_median_per_shot[shot_index] = decoder.kept_state_median;
          kept_state_mean_per_shot[shot_index] = decoder.kept_state_mean;
          kept_state_max_per_shot[shot_index] = decoder.kept_state_max;
          time_expand_per_shot[shot_index] = decoder.time_expand_seconds;
          time_collapse_per_shot[shot_index] = decoder.time_collapse_seconds;
          time_truncate_per_shot[shot_index] = decoder.time_truncate_seconds;
          time_reconstruct_per_shot[shot_index] = decoder.time_reconstruct_seconds;
          merge_calls_total += decoder.merge_calls;
          merge_input_candidates_total += decoder.merge_input_candidates;
          merge_output_candidates_total += decoder.merge_output_candidates;
          merge_duplicate_layers_total += decoder.merge_duplicate_layers;
          merge_skipped_layers_total += decoder.merge_skipped_layers;

          if (writer) {
            writer->write_bits((uint8_t*)&obs_predicted[shot_index], num_observables);
            writer->write_end();
          }
          if (low_confidence[shot_index]) {
            ++num_low_confidence;
          } else if (obs_predicted[shot_index] != shots[shot_index].obs_mask_as_u64()) {
            ++num_errors;
          }
          total_time_seconds += decoding_time_seconds[shot_index];
          if (args.print_stats) {
            std::cout << "num_shots = " << (shot_index + 1)
                      << " num_low_confidence = " << num_low_confidence
                      << " num_errors = " << num_errors
                      << " states_expanded = " << num_states_expanded_per_shot[shot_index]
                      << " states_merged = " << num_states_merged_per_shot[shot_index]
                      << " max_beam = " << max_beam_size_per_shot[shot_index]
                      << " frontier_width = " << max_frontier_width_per_shot[shot_index]
                      << " total_time_seconds = " << total_time_seconds << '\n';
            std::cout << "kept_states" << " min=" << kept_state_min_per_shot[shot_index]
                      << " median=" << kept_state_median_per_shot[shot_index]
                      << " mean=" << kept_state_mean_per_shot[shot_index]
                      << " max=" << kept_state_max_per_shot[shot_index] << '\n';
            std::cout << "branch_masses" << " obs0=" << mass0_predicted[shot_index]
                      << " obs1=" << mass1_predicted[shot_index] << '\n';
            std::cout << "phase_times_seconds" << " expand=" << time_expand_per_shot[shot_index]
                      << " collapse=" << time_collapse_per_shot[shot_index]
                      << " truncate=" << time_truncate_per_shot[shot_index]
                      << " reconstruct=" << time_reconstruct_per_shot[shot_index] << '\n';
          }
          ++shot;
          if (num_errors >= args.max_errors) {
            batch_start = shots.size();
            break;
          }
        }
      }
    }
  } else {
    shot = parallel_for_shots_in_order(
        shots.size(), execution_threads,
        [&](size_t thread_index, size_t shot_index) {
          if (!decoders[thread_index]) {
            decoders[thread_index] = std::make_unique<TesseractTrellisGpuDecoder>(config);
          }
          auto& decoder = *decoders[thread_index];
          auto start_time = std::chrono::high_resolution_clock::now();
          decoder.decode_shot(shots[shot_index].hits);
          auto stop_time = std::chrono::high_resolution_clock::now();
          decoding_time_seconds[shot_index] = std::chrono::duration_cast<std::chrono::microseconds>(
                                                  stop_time - start_time)
                                                  .count() /
                                              1e6;
          obs_predicted[shot_index] = decoder.predicted_obs_mask;
          low_confidence[shot_index] = decoder.low_confidence_flag;
          mass0_predicted[shot_index] = decoder.total_mass_obs0;
          mass1_predicted[shot_index] = decoder.total_mass_obs1;
          num_states_expanded_per_shot[shot_index] = decoder.num_states_expanded;
          num_states_merged_per_shot[shot_index] = decoder.num_states_merged;
          max_beam_size_per_shot[shot_index] = decoder.max_beam_size_seen;
          max_frontier_width_per_shot[shot_index] = decoder.max_frontier_width_seen;
          kept_state_min_per_shot[shot_index] = decoder.kept_state_min;
          kept_state_median_per_shot[shot_index] = decoder.kept_state_median;
          kept_state_mean_per_shot[shot_index] = decoder.kept_state_mean;
          kept_state_max_per_shot[shot_index] = decoder.kept_state_max;
          time_expand_per_shot[shot_index] = decoder.time_expand_seconds;
          time_collapse_per_shot[shot_index] = decoder.time_collapse_seconds;
          time_truncate_per_shot[shot_index] = decoder.time_truncate_seconds;
          time_reconstruct_per_shot[shot_index] = decoder.time_reconstruct_seconds;
          merge_calls_total += decoder.merge_calls;
          merge_input_candidates_total += decoder.merge_input_candidates;
          merge_output_candidates_total += decoder.merge_output_candidates;
          merge_duplicate_layers_total += decoder.merge_duplicate_layers;
          merge_skipped_layers_total += decoder.merge_skipped_layers;
        },
        [&](size_t shot_index) {
          if (writer) {
            writer->write_bits((uint8_t*)&obs_predicted[shot_index], num_observables);
            writer->write_end();
          }
          if (low_confidence[shot_index]) {
            ++num_low_confidence;
          } else if (obs_predicted[shot_index] != shots[shot_index].obs_mask_as_u64()) {
            ++num_errors;
          }
          total_time_seconds += decoding_time_seconds[shot_index];
          if (args.print_stats) {
            std::cout << "num_shots = " << (shot_index + 1)
                      << " num_low_confidence = " << num_low_confidence
                      << " num_errors = " << num_errors
                      << " states_expanded = " << num_states_expanded_per_shot[shot_index]
                      << " states_merged = " << num_states_merged_per_shot[shot_index]
                      << " max_beam = " << max_beam_size_per_shot[shot_index]
                      << " frontier_width = " << max_frontier_width_per_shot[shot_index]
                      << " total_time_seconds = " << total_time_seconds << '\n';
            std::cout << "kept_states" << " min=" << kept_state_min_per_shot[shot_index]
                      << " median=" << kept_state_median_per_shot[shot_index]
                      << " mean=" << kept_state_mean_per_shot[shot_index]
                      << " max=" << kept_state_max_per_shot[shot_index] << '\n';
            std::cout << "branch_masses" << " obs0=" << mass0_predicted[shot_index]
                      << " obs1=" << mass1_predicted[shot_index] << '\n';
            std::cout << "phase_times_seconds" << " expand=" << time_expand_per_shot[shot_index]
                      << " collapse=" << time_collapse_per_shot[shot_index]
                      << " truncate=" << time_truncate_per_shot[shot_index]
                      << " reconstruct=" << time_reconstruct_per_shot[shot_index] << '\n';
          }
          return num_errors < args.max_errors;
        });
  }

  if (!args.dem_out_fname.empty()) {
    throw std::invalid_argument(
        "--dem-out is not supported by tesseract_trellis_gpu without path reconstruction.");
  }

  bool print_final_stats = true;
  if (!args.stats_out_fname.empty()) {
    nlohmann::json stats_json = {{"circuit_path", args.circuit_path},
                                 {"dem_path", args.dem_path},
                                 {"beam_width", args.beam_width},
                                 {"gpu_max_beam_width", config.beam_width},
                                 {"future_detcost_scale", args.future_detcost_scale},
                                 {"ranking_mode", args.ranking_mode},
                                 {"sample_seed", args.sample_seed},
                                 {"sample_num_shots", args.sample_num_shots},
                                 {"gpu_shot_batch_size", args.gpu_shot_batch_size},
                                 {"gpu_merge_period", args.gpu_merge_period},
                                 {"adaptive_dynamic_beam", args.adaptive_dynamic_beam},
                                 {"adaptive_beam_width", args.adaptive_beam_width},
                                 {"adaptive_confidence_threshold",
                                  args.adaptive_confidence_threshold},
                                 {"adaptive_rerun_shots", adaptive_rerun_shots},
                                 {"adaptive_first_pass_errors", adaptive_first_pass_errors},
                                 {"adaptive_first_pass_time_seconds",
                                  adaptive_first_pass_time_seconds},
                                 {"adaptive_rerun_time_seconds", adaptive_rerun_time_seconds},
                                 {"confidence_margin_min", confidence_margin_min},
                                 {"confidence_margin_median", confidence_margin_median},
                                 {"confidence_margin_mean", confidence_margin_mean},
                                 {"confidence_margin_max", confidence_margin_max},
                                 {"confidence_threshold_sweep", confidence_threshold_sweep},
                                 {"gpu_dynamic_beam_limit_sample_count",
                                  gpu_dynamic_beam_limit_sample_count},
                                 {"gpu_dynamic_beam_limit_min", gpu_dynamic_beam_limit_min},
                                 {"gpu_dynamic_beam_limit_median",
                                  gpu_dynamic_beam_limit_median},
                                 {"gpu_dynamic_beam_limit_mean", gpu_dynamic_beam_limit_mean},
                                 {"gpu_dynamic_beam_limit_max", gpu_dynamic_beam_limit_max},
                                 {"gpu_dynamic_beam_grow_events",
                                  gpu_dynamic_beam_grow_events},
                                 {"num_threads", execution_threads},
                                 {"num_errors", num_errors},
                                 {"num_low_confidence", num_low_confidence},
                                 {"num_shots", shot},
                                 {"total_time_seconds", total_time_seconds},
                                 {"backend", probe.backend_description()},
                                 {"merge_calls", merge_calls_total},
                                 {"merge_input_candidates", merge_input_candidates_total},
                                 {"merge_output_candidates", merge_output_candidates_total},
                                 {"merge_duplicate_layers", merge_duplicate_layers_total},
                                 {"merge_skipped_layers", merge_skipped_layers_total}};
    if (args.stats_out_fname == "-") {
      std::cout << stats_json << std::endl;
      print_final_stats = false;
    } else {
      std::ofstream out(args.stats_out_fname, std::ofstream::out);
      out << stats_json << std::endl;
    }
  }

  if (print_final_stats) {
    std::cout << "num_shots = " << shot << " num_low_confidence = " << num_low_confidence;
    if (has_obs) {
      std::cout << " num_errors = " << num_errors;
    }
    std::cout << " total_time_seconds = " << total_time_seconds << std::endl;
  }
}
