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

#include <algorithm>
#include <argparse/argparse.hpp>
#include <atomic>
#include <chrono>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <thread>
#include <vector>

#include "bp/hard_decision_post_processor.h"
#include "bp/osd_post_processor.h"
#include "bp/tesseract_bp_decoder.h"
#include "common.h"
#include "stim.h"
#include "utils.h"

using namespace bp;

struct Args {
  std::string circuit_path;
  std::string dem_path;
  bool no_merge_errors = false;

  // Sampling options
  size_t sample_num_shots = 0;
  size_t max_errors = SIZE_MAX;
  uint64_t sample_seed;

  // Shot data file options
  std::string in_fname = "";
  std::string in_format = "";
  std::string obs_in_fname = "";
  std::string obs_in_format = "";
  bool append_observables = false;
  std::string out_fname = "";
  std::string out_format = "";

  // BP parameters
  size_t max_iter = 20;
  std::string update_rule = "min-sum";
  std::string schedule = "serial";
  size_t num_threads = 1;
  double normalization_factor = 0.625;
  int osd_order = -1;  // -1 means HardDecision, >= 0 means OSD
  int osd_weight = 0;
  bool use_batched_bp = false;

  std::string stats_out_fname = "";
  std::string sinter_csv_out = "";
  bool verbose = false;
  bool print_stats = false;

  bool has_observables() {
    return append_observables || !obs_in_fname.empty() || (sample_num_shots > 0);
  }

  void validate() {
    if (circuit_path.empty() and dem_path.empty()) {
      throw std::invalid_argument("Must provide at least one of --circuit or --dem");
    }
    int num_data_sources = int(sample_num_shots > 0) + int(!in_fname.empty());
    if (num_data_sources != 1) {
      throw std::invalid_argument("Requires exactly 1 source of shots.");
    }
    if (!in_fname.empty() and in_format.empty()) {
      throw std::invalid_argument("If --in is provided, must also specify --in-format.");
    }
    if (!out_fname.empty() and out_format.empty()) {
      throw std::invalid_argument("If --out is provided, must also specify --out-format.");
    }
    if (num_threads == 0) {
      throw std::invalid_argument("--threads must be at least 1.");
    }
  }

  void extract(BPParams& params, stim::DetectorErrorModel& dem,
               std::vector<stim::SparseShot>& shots,
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
      dem = stim::DetectorErrorModel::from_file(file);
      fclose(file);
    } else {
      dem = stim::ErrorAnalyzer::circuit_to_detector_error_model(circuit, false, true, true, 1,
                                                                 false, false);
    }

    params.max_iter = max_iter;
    params.update_rule = update_rule;
    params.schedule = schedule;
    params.normalization_factor = (float)normalization_factor;

    if (sample_num_shots > 0) {
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
          shots_file, shots_in_format.id, 0, dem.count_detectors(),
          append_observables * dem.count_observables());

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
      stim::FileFormatData shots_obs_in_format = stim::format_name_to_enum_map().at(obs_in_format);
      auto obs_reader = stim::MeasureRecordReader<stim::MAX_BITWORD_WIDTH>::make(
          obs_file, shots_obs_in_format.id, 0, 0, dem.count_observables());
      stim::SparseShot sparse_shot;
      sparse_shot.clear();
      size_t num_obs_shots = 0;
      while (obs_reader->start_and_read_entire_record(sparse_shot)) {
        shots[num_obs_shots].obs_mask = sparse_shot.obs_mask;
        sparse_shot.clear();
        ++num_obs_shots;
      }
      fclose(obs_file);
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
  argparse::ArgumentParser program("bp");
  Args args;
  program.add_argument("--circuit").help("Stim circuit file path").store_into(args.circuit_path);
  program.add_argument("--dem").help("Stim dem file path").store_into(args.dem_path);
  program.add_argument("--sample-num-shots")
      .help("Number of shots to sample")
      .store_into(args.sample_num_shots);
  program.add_argument("--max-errors").help("Maximum errors to sample").store_into(args.max_errors);
  program.add_argument("--sample-seed")
      .default_value(static_cast<uint64_t>(12345))
      .store_into(args.sample_seed);

  program.add_argument("--in").help("File to read detection events from").store_into(args.in_fname);
  program.add_argument("--in-format").help("Format of input file").store_into(args.in_format);
  program.add_argument("--obs_in")
      .help("File to read observable flips from")
      .store_into(args.obs_in_fname);
  program.add_argument("--obs-in-format")
      .help("Format of obs input file")
      .store_into(args.obs_in_format);
  program.add_argument("--out").help("File to write predictions to").store_into(args.out_fname);
  program.add_argument("--out-format").help("Format of output file").store_into(args.out_format);

  program.add_argument("--max-iter").default_value(size_t(20)).store_into(args.max_iter);
  program.add_argument("--update-rule")
      .default_value(std::string("min-sum"))
      .store_into(args.update_rule);
  program.add_argument("--schedule").default_value(std::string("serial")).store_into(args.schedule);
  program.add_argument("--normalization-factor")
      .default_value(0.625)
      .store_into(args.normalization_factor);
  program.add_argument("--osd-order").default_value(-1).store_into(args.osd_order);
  program.add_argument("--osd-weight").default_value(0).store_into(args.osd_weight);
  program.add_argument("--batched")
      .help("Use AVX-512 batching across shots")
      .flag()
      .store_into(args.use_batched_bp);
  program.add_argument("--threads")
      .default_value(size_t(
          std::thread::hardware_concurrency() == 0 ? 1 : std::thread::hardware_concurrency()))
      .store_into(args.num_threads);

  program.add_argument("--stats-out")
      .help("JSON stats output file")
      .store_into(args.stats_out_fname);
  program.add_argument("--sinter-csv-out")
      .help("Sinter CSV stats output line")
      .store_into(args.sinter_csv_out);
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

  BPParams params;
  stim::DetectorErrorModel dem;
  std::vector<stim::SparseShot> shots;
  std::unique_ptr<stim::MeasureRecordWriter> writer;
  args.extract(params, dem, shots, writer);

  std::vector<std::unique_ptr<TesseractBpDecoder>> decoders(args.num_threads);
  std::vector<std::shared_ptr<PostProcessor>> pps(args.num_threads);

  size_t total_shots = shots.size();
  size_t num_observables = dem.count_observables();
  std::atomic<size_t> num_errors(0);
  std::atomic<size_t> num_discards(0);
  std::atomic<double> total_time_seconds(0);
  std::atomic<size_t> processed_shots(0);

  auto start_global_time = std::chrono::high_resolution_clock::now();

  if (args.use_batched_bp) {
    size_t num_batches = (total_shots + 63) / 64;
    std::vector<std::vector<std::vector<uint8_t>>> all_predictions(num_batches);
    std::vector<double> batch_time_seconds(num_batches);

    parallel_for_shots_in_order(
        num_batches, args.num_threads,
        [&](size_t thread_idx, size_t batch_idx) {
          if (!decoders[thread_idx]) {
            decoders[thread_idx] = std::make_unique<TesseractBpDecoder>(dem, params);
            pps[thread_idx] = (args.osd_order >= 0)
                                  ? decoders[thread_idx]->create_osd_post_processor(args.osd_order,
                                                                                    args.osd_weight)
                                  : std::make_shared<HardDecisionPostProcessor>();
          }

          size_t shot_start = batch_idx * 64;
          size_t shot_end = std::min(shot_start + 64, total_shots);
          std::vector<std::vector<uint64_t>> batch_syndromes;
          for (size_t s = shot_start; s < shot_end; ++s) {
            std::vector<uint64_t> dets(shots[s].hits.begin(), shots[s].hits.end());
            batch_syndromes.push_back(dets);
          }

          auto start_batch = std::chrono::high_resolution_clock::now();
          all_predictions[batch_idx] =
              decoders[thread_idx]->decode_batch(batch_syndromes, pps[thread_idx]);
          auto stop_batch = std::chrono::high_resolution_clock::now();
          batch_time_seconds[batch_idx] =
              std::chrono::duration_cast<std::chrono::microseconds>(stop_batch - start_batch)
                  .count() /
              1e6;
        },
        [&](size_t batch_idx) {
          size_t shot_start = batch_idx * 64;
          size_t shot_end = std::min(shot_start + 64, total_shots);
          size_t actual_size = shot_end - shot_start;

          for (size_t b = 0; b < actual_size; ++b) {
            size_t s = shot_start + b;
            std::vector<uint8_t> predicted_flips = all_predictions[batch_idx][b];
            uint64_t predicted_mask = 0;
            for (size_t o = 0; o < num_observables; ++o) {
              if (predicted_flips[o]) predicted_mask ^= (1ULL << o);
            }
            if (writer) {
              writer->write_bits((uint8_t*)&predicted_flips[0], num_observables);
              writer->write_end();
            }
            if (predicted_mask != shots[s].obs_mask_as_u64()) {
              num_errors++;
            }
            processed_shots++;
          }
          total_time_seconds = total_time_seconds + batch_time_seconds[batch_idx];
          if (args.print_stats && processed_shots % 1024 == 0) {
            std::cout << "Processed " << processed_shots << " shots, errors: " << num_errors
                      << std::endl;
          }
          return num_errors < args.max_errors;
        });
  } else {
    std::vector<std::vector<uint8_t>> all_predictions(total_shots);
    std::vector<double> shot_time_seconds(total_shots);

    parallel_for_shots_in_order(
        total_shots, args.num_threads,
        [&](size_t thread_idx, size_t shot_idx) {
          if (!decoders[thread_idx]) {
            decoders[thread_idx] = std::make_unique<TesseractBpDecoder>(dem, params);
            pps[thread_idx] = (args.osd_order >= 0)
                                  ? decoders[thread_idx]->create_osd_post_processor(args.osd_order,
                                                                                    args.osd_weight)
                                  : std::make_shared<HardDecisionPostProcessor>();
          }
          std::vector<uint64_t> dets(shots[shot_idx].hits.begin(), shots[shot_idx].hits.end());
          auto start_shot = std::chrono::high_resolution_clock::now();
          all_predictions[shot_idx] = decoders[thread_idx]->decode(dets, pps[thread_idx]);
          auto stop_shot = std::chrono::high_resolution_clock::now();
          shot_time_seconds[shot_idx] =
              std::chrono::duration_cast<std::chrono::microseconds>(stop_shot - start_shot)
                  .count() /
              1e6;
        },
        [&](size_t shot_idx) {
          std::vector<uint8_t> predicted_flips = all_predictions[shot_idx];
          uint64_t predicted_mask = 0;
          for (size_t o = 0; o < num_observables; ++o) {
            if (predicted_flips[o]) predicted_mask ^= (1ULL << o);
          }
          if (writer) {
            writer->write_bits((uint8_t*)&predicted_flips[0], num_observables);
            writer->write_end();
          }
          if (predicted_mask != shots[shot_idx].obs_mask_as_u64()) {
            num_errors++;
          }
          processed_shots++;
          total_time_seconds = total_time_seconds + shot_time_seconds[shot_idx];
          if (args.print_stats && processed_shots % 100 == 0) {
            std::cout << "Processed " << processed_shots << " shots, errors: " << num_errors
                      << std::endl;
          }
          return num_errors < args.max_errors;
        });
  }

  auto stop_global_time = std::chrono::high_resolution_clock::now();
  double global_elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(stop_global_time - start_global_time)
          .count() /
      1e6;

  std::string decoder_name =
      std::string(args.use_batched_bp ? "batched-" : "scalar-") + args.schedule + "-bp";
  if (args.osd_order >= 0) decoder_name += "+osd";

  if (!args.stats_out_fname.empty()) {
    nlohmann::json stats_json = {{"circuit_path", args.circuit_path},
                                 {"dem_path", args.dem_path},
                                 {"max_errors", args.max_errors},
                                 {"sample_seed", args.sample_seed},
                                 {"total_time_seconds", global_elapsed},
                                 {"num_errors", num_errors.load()},
                                 {"num_shots", processed_shots.load()},
                                 {"num_discards", num_discards.load()},
                                 {"decoder", decoder_name}};
    if (args.stats_out_fname == "-") {
      std::cout << stats_json << std::endl;
    } else {
      std::ofstream out(args.stats_out_fname);
      out << stats_json << std::endl;
    }
  }

  if (!args.sinter_csv_out.empty()) {
    std::stringstream csv_line;
    csv_line << processed_shots.load() << "," << num_errors.load() << "," << num_discards.load()
             << "," << std::fixed << std::setprecision(4) << global_elapsed << "," << decoder_name
             << ",none,\"{\"\"path\"\":\"\"" << args.circuit_path << "\"\"}\",";
    if (args.sinter_csv_out == "-") {
      std::cout << csv_line.str() << std::endl;
    } else {
      std::ofstream out(args.sinter_csv_out, std::ios::app);
      out << csv_line.str() << std::endl;
    }
  }

  if (args.stats_out_fname.empty() && args.sinter_csv_out.empty()) {
    std::cout << "num_shots = " << processed_shots.load() << " num_errors = " << num_errors.load()
              << " total_time_seconds = " << global_elapsed << std::endl;
  }

  return 0;
}
