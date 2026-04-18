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
#include <atomic>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <thread>

#include "common.h"
#include "stim.h"
#include "tesseract_trellis.h"
#include "utils.h"

namespace {

TesseractTrellisPruneMode parse_prune_mode(const std::string& value) {
  if (value == "merged") return TesseractTrellisPruneMode::MergedStates;
  if (value == "keep-best") return TesseractTrellisPruneMode::KeepBest;
  if (value == "branch") return TesseractTrellisPruneMode::BranchEntries;
  if (value == "none") return TesseractTrellisPruneMode::NoMerge;
  throw std::invalid_argument("Unknown trellis prune mode: " + value);
}

TesseractTrellisRankingMode parse_ranking_mode(const std::string& value) {
  if (value == "mass") return TesseractTrellisRankingMode::MassOnly;
  if (value == "future-detcost") return TesseractTrellisRankingMode::FutureDetcostRanked;
  throw std::invalid_argument("Unknown trellis ranking mode: " + value);
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
  size_t beam_width = 1024;
  size_t merge_interval = 1;
  std::string prune_mode = "merged";
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
    if ((shot_range_begin || shot_range_end) && shot_range_end < shot_range_begin) {
      throw std::invalid_argument("Provided shot range must have end >= begin.");
    }
    if (sample_num_shots > 0 && circuit_path.empty()) {
      throw std::invalid_argument("Cannot sample shots without a circuit.");
    }
    if (beam_width == 0) {
      throw std::invalid_argument("--beam must be at least 1.");
    }
    if (merge_interval == 0) {
      throw std::invalid_argument("--merge-interval must be at least 1.");
    }
    parse_prune_mode(prune_mode);
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

    config.beam_width = beam_width;
    config.merge_interval = merge_interval;
    config.verbose = verbose;
    config.prune_mode = parse_prune_mode(prune_mode);
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
  argparse::ArgumentParser program("tesseract_trellis");
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
  program.add_argument("--beam").default_value(size_t(1024)).store_into(args.beam_width);
  program.add_argument("--merge-interval").default_value(size_t(1)).store_into(args.merge_interval);
  program.add_argument("--prune-mode")
      .help(
          "Trellis pruning mode: merged, keep-best, branch, or none. "
          "merged sums probabilities of all branches with the same residual detection events. "
          "keep-best keeps only the single highest-probability branch for each residual detection "
          "state. "
          "branch ranks branches individually, but still merges exact duplicate (state, "
          "observable) entries first. "
          "none skips even that exact-duplicate merge, so identical branches may occupy multiple "
          "beam slots.")
      .default_value(std::string("merged"))
      .store_into(args.prune_mode);
  program.add_argument("--ranking-mode")
      .help("Trellis ranking mode: mass or future-detcost")
      .default_value(std::string("mass"))
      .store_into(args.ranking_mode);
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

  std::vector<uint64_t> obs_predicted(shots.size());
  std::vector<double> mass0_predicted(shots.size());
  std::vector<double> mass1_predicted(shots.size());
  std::vector<double> decoding_time_seconds(shots.size());
  std::vector<size_t> num_states_expanded_per_shot(shots.size());
  std::vector<size_t> num_states_merged_per_shot(shots.size());
  std::vector<size_t> max_beam_size_per_shot(shots.size());
  std::vector<size_t> max_frontier_width_per_shot(shots.size());
  std::vector<double> time_expand_per_shot(shots.size());
  std::vector<double> time_collapse_per_shot(shots.size());
  std::vector<double> time_truncate_per_shot(shots.size());
  std::vector<double> time_reconstruct_per_shot(shots.size());
  std::vector<std::atomic<bool>> low_confidence(shots.size());
  const stim::DetectorErrorModel original_dem = config.dem.flattened();
  std::vector<std::unique_ptr<TesseractTrellisDecoder>> decoders(args.num_threads);

  bool has_obs = args.has_observables();
  size_t num_errors = 0;
  size_t num_low_confidence = 0;
  double total_time_seconds = 0;
  size_t num_observables = config.dem.count_observables();

  size_t shot = parallel_for_shots_in_order(
      shots.size(), args.num_threads,
      [&](size_t thread_index, size_t shot_index) {
        if (!decoders[thread_index]) {
          decoders[thread_index] = std::make_unique<TesseractTrellisDecoder>(config);
        }
        auto& decoder = *decoders[thread_index];
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
        time_expand_per_shot[shot_index] = decoder.time_expand_seconds;
        time_collapse_per_shot[shot_index] = decoder.time_collapse_seconds;
        time_truncate_per_shot[shot_index] = decoder.time_truncate_seconds;
        time_reconstruct_per_shot[shot_index] = decoder.time_reconstruct_seconds;
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
          std::cout << "branch_masses" << " obs0=" << mass0_predicted[shot_index]
                    << " obs1=" << mass1_predicted[shot_index] << '\n';
          std::cout << "phase_times_seconds" << " expand=" << time_expand_per_shot[shot_index]
                    << " collapse=" << time_collapse_per_shot[shot_index]
                    << " truncate=" << time_truncate_per_shot[shot_index]
                    << " reconstruct=" << time_reconstruct_per_shot[shot_index] << '\n';
        }
        return num_errors < args.max_errors;
      });

  if (!args.dem_out_fname.empty()) {
    throw std::invalid_argument(
        "--dem-out is not supported by tesseract_trellis without path reconstruction.");
  }

  bool print_final_stats = true;
  if (!args.stats_out_fname.empty()) {
    nlohmann::json stats_json = {{"circuit_path", args.circuit_path},
                                 {"dem_path", args.dem_path},
                                 {"beam_width", args.beam_width},
                                 {"sample_seed", args.sample_seed},
                                 {"sample_num_shots", args.sample_num_shots},
                                 {"num_threads", args.num_threads},
                                 {"num_errors", num_errors},
                                 {"num_low_confidence", num_low_confidence},
                                 {"num_shots", shot},
                                 {"total_time_seconds", total_time_seconds}};
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
