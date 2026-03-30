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
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <queue>
#include <thread>

#include "common.h"
#include "stim.h"
#include "tesseract_ftl.h"
#include "utils.h"

struct Args {
  std::string circuit_path;
  std::string dem_path;
  bool no_merge_errors = false;

  uint64_t det_order_seed;
  size_t num_det_orders = 10;
  bool det_order_bfs = false;
  bool det_order_index = false;
  bool det_order_coordinate = false;

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

  size_t det_beam;
  double det_penalty = 0;
  bool beam_climbing = false;
  bool no_revisit_dets = false;
  size_t pqlimit;

  size_t subset_detcost_size = 0;

  bool verbose = false;
  bool print_stats = false;

  bool has_observables() {
    return append_observables || !obs_in_fname.empty() || (sample_num_shots > 0);
  }

  void validate() {
    if (circuit_path.empty() && dem_path.empty()) {
      throw std::invalid_argument("Must provide at least one of --circuit or --dem");
    }
    int det_order_flags = int(det_order_bfs) + int(det_order_index) + int(det_order_coordinate);
    if (det_order_flags > 1) {
      throw std::invalid_argument(
          "Only one of --det-order-bfs, --det-order-index, or --det-order-coordinate may be set.");
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
    if (shot_range_begin || shot_range_end) {
      if (shot_range_end < shot_range_begin) {
        throw std::invalid_argument("Provided shot range must have end >= begin.");
      }
    }
    if (sample_num_shots > 0 && circuit_path.empty()) {
      throw std::invalid_argument("Cannot sample shots without a circuit.");
    }
    if (beam_climbing && det_beam == INF_DET_BEAM) {
      throw std::invalid_argument("Beam climbing requires a finite beam");
    }
    if (subset_detcost_size > 1) {
      throw std::invalid_argument("This prototype currently supports --subset-detcost-size <= 1");
    }
  }

  void extract(TesseractFTLConfig& config, std::vector<stim::SparseShot>& shots,
               std::unique_ptr<stim::MeasureRecordWriter>& writer) {
    stim::Circuit circuit;
    if (!circuit_path.empty()) {
      FILE* file = fopen(circuit_path.c_str(), "r");
      if (!file) throw std::invalid_argument("Could not open the file: " + circuit_path);
      circuit = stim::Circuit::from_file(file);
      fclose(file);
    }

    if (!dem_path.empty()) {
      FILE* file = fopen(dem_path.c_str(), "r");
      if (!file) throw std::invalid_argument("Could not open the file: " + dem_path);
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

    config.merge_errors = !no_merge_errors;
    config.subset_detcost_size = subset_detcost_size;

    {
      DetOrder order = DetOrder::DetBFS;
      if (det_order_index) {
        order = DetOrder::DetIndex;
      } else if (det_order_coordinate) {
        order = DetOrder::DetCoordinate;
      }
      config.det_orders = build_det_orders(config.dem, num_det_orders, order, det_order_seed);
    }

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
          if (dets[d][k]) shots[k].hits.push_back(d);
        }
      }
    }

    if (!in_fname.empty()) {
      FILE* shots_file = fopen(in_fname.c_str(), "r");
      if (!shots_file) throw std::invalid_argument("Could not open the file: " + in_fname);
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
      if (!obs_file) throw std::invalid_argument("Could not open the file: " + obs_in_fname);
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
      if (out_fname != "-") predictions_file = fopen(out_fname.c_str(), "w");
      writer = stim::MeasureRecordWriter::make(predictions_file, predictions_out_format.id);
      writer->begin_result_type('L');
    }

    config.det_beam = det_beam;
    config.det_penalty = det_penalty;
    config.beam_climbing = beam_climbing;
    config.no_revisit_dets = no_revisit_dets;
    config.pqlimit = pqlimit;
    config.verbose = verbose;
  }
};

int main(int argc, char* argv[]) {
  std::cout.precision(16);
  argparse::ArgumentParser program("tesseract_ftl");
  Args args;

  program.add_argument("--circuit").help("Stim circuit file path").store_into(args.circuit_path);
  program.add_argument("--dem").help("Stim dem file path").store_into(args.dem_path);
  program.add_argument("--no-merge-errors")
      .help("If provided, will not merge identical error mechanisms.")
      .store_into(args.no_merge_errors);
  program.add_argument("--subset-detcost-size")
      .help("0 = plain detcost delegate, 1 = singleton fractional lower bound")
      .default_value(size_t(0))
      .store_into(args.subset_detcost_size);

  program.add_argument("--num-det-orders")
      .help("Number of ways to orient the manifold when reordering the detectors")
      .metavar("N")
      .default_value(size_t(1))
      .store_into(args.num_det_orders);
  program.add_argument("--det-order-bfs")
      .help("Use BFS-based detector ordering")
      .flag()
      .store_into(args.det_order_bfs);
  program.add_argument("--det-order-index")
      .help("Randomly choose increasing or decreasing detector index order")
      .flag()
      .store_into(args.det_order_index);
  program.add_argument("--det-order-coordinate")
      .help("Random geometric detector orientation ordering")
      .flag()
      .store_into(args.det_order_coordinate);
  program.add_argument("--det-order-seed")
      .help("Seed used when initializing the random detector traversal orderings.")
      .default_value(static_cast<uint64_t>(518278944))
      .store_into(args.det_order_seed);

  program.add_argument("--sample-num-shots")
      .help("Sample the requested number of shots from the Stim circuit.")
      .store_into(args.sample_num_shots);
  program.add_argument("--max-errors")
      .help("Stop after at least this many errors have been observed.")
      .store_into(args.max_errors);
  program.add_argument("--sample-seed")
      .help("Seed used when initializing the random number generator for sampling shots")
      .default_value(static_cast<uint64_t>(std::random_device()()))
      .store_into(args.sample_seed);

  program.add_argument("--shot-range-begin")
      .default_value(size_t(0))
      .store_into(args.shot_range_begin);
  program.add_argument("--shot-range-end").default_value(size_t(0)).store_into(args.shot_range_end);

  program.add_argument("--in").default_value(std::string("")).store_into(args.in_fname);
  std::string in_formats;
  bool first = true;
  for (const auto& [key, value] : stim::format_name_to_enum_map()) {
    if (!first) in_formats += "/";
    first = false;
    in_formats += key;
  }
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
  program.add_argument("--beam").default_value(INF_DET_BEAM).store_into(args.det_beam);
  program.add_argument("--det-penalty").default_value(0.0).store_into(args.det_penalty);
  program.add_argument("--beam-climbing").flag().store_into(args.beam_climbing);
  program.add_argument("--no-revisit-dets").flag().store_into(args.no_revisit_dets);
  program.add_argument("--pqlimit")
      .default_value(std::numeric_limits<size_t>::max())
      .store_into(args.pqlimit);
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

  TesseractFTLConfig config;
  std::vector<stim::SparseShot> shots;
  std::unique_ptr<stim::MeasureRecordWriter> writer;
  args.extract(config, shots, writer);

  std::vector<uint64_t> obs_predicted(shots.size());
  std::vector<double> cost_predicted(shots.size());
  std::vector<double> decoding_time_seconds(shots.size());
  std::vector<std::atomic<bool>> low_confidence(shots.size());
  const stim::DetectorErrorModel original_dem = config.dem.flattened();
  std::vector<std::unique_ptr<TesseractFTLDecoder>> decoders(args.num_threads);
  std::vector<std::vector<size_t>> error_use_per_thread(
      args.num_threads, std::vector<size_t>(original_dem.count_errors()));
  std::vector<TesseractFTLStats> decoder_stats_per_thread(args.num_threads);

  bool has_obs = args.has_observables();
  size_t num_errors = 0;
  size_t num_low_confidence = 0;
  double total_time_seconds = 0;
  size_t num_observables = config.dem.count_observables();

  size_t shot = parallel_for_shots_in_order(
      shots.size(), args.num_threads,
      [&](size_t thread_index, size_t shot_index) {
        if (!decoders[thread_index]) {
          decoders[thread_index] = std::make_unique<TesseractFTLDecoder>(config);
        }
        auto& decoder = *decoders[thread_index];
        auto& error_use = error_use_per_thread[thread_index];
        auto start_time = std::chrono::high_resolution_clock::now();
        decoder.decode_to_errors(shots[shot_index].hits);
        auto stop_time = std::chrono::high_resolution_clock::now();
        decoding_time_seconds[shot_index] =
            std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count() /
            1e6;
        obs_predicted[shot_index] =
            vector_to_u64_mask(decoder.get_flipped_observables(decoder.predicted_errors_buffer));
        low_confidence[shot_index] = decoder.low_confidence_flag;
        cost_predicted[shot_index] = decoder.cost_from_errors(decoder.predicted_errors_buffer);
        decoder_stats_per_thread[thread_index].accumulate(decoder.stats);
        if (!has_obs || shots[shot_index].obs_mask_as_u64() == obs_predicted[shot_index]) {
          for (size_t ei : decoder.predicted_errors_buffer) ++error_use[ei];
        }
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
                    << " total_time_seconds = " << total_time_seconds << std::endl;
          std::cout << "cost = " << cost_predicted[shot_index] << std::endl;
          std::cout.flush();
        }
        return num_errors < args.max_errors;
      });

  std::vector<size_t> error_use_totals(original_dem.count_errors());
  for (const auto& error_use : error_use_per_thread) {
    for (size_t ei = 0; ei < error_use_totals.size(); ++ei) error_use_totals[ei] += error_use[ei];
  }
  TesseractFTLStats decoder_stats_total;
  for (const auto& s : decoder_stats_per_thread) decoder_stats_total.accumulate(s);

  if (!args.dem_out_fname.empty()) {
    size_t num_usage_dem_shots = shot;
    if (has_obs) num_usage_dem_shots -= num_errors;
    stim::DetectorErrorModel est_dem =
        common::dem_from_counts(original_dem, error_use_totals, num_usage_dem_shots);
    std::ofstream out(args.dem_out_fname, std::ofstream::out);
    if (!out.is_open()) throw std::invalid_argument("Failed to open " + args.dem_out_fname);
    out << est_dem << '\n';
  }

  bool print_final_stats = true;
  if (!args.stats_out_fname.empty()) {
    nlohmann::json stats_json = {
        {"circuit_path", args.circuit_path},
        {"dem_path", args.dem_path},
        {"max_errors", args.max_errors},
        {"sample_seed", args.sample_seed},
        {"det_beam", args.det_beam},
        {"det_penalty", args.det_penalty},
        {"beam_climbing", args.beam_climbing},
        {"no_revisit_dets", args.no_revisit_dets},
        {"pqlimit", args.pqlimit},
        {"num_det_orders", args.num_det_orders},
        {"det_order_seed", args.det_order_seed},
        {"subset_detcost_size", args.subset_detcost_size},
        {"total_time_seconds", total_time_seconds},
        {"num_errors", num_errors},
        {"num_low_confidence", num_low_confidence},
        {"num_shots", shot},
        {"num_threads", args.num_threads},
        {"sample_num_shots", args.sample_num_shots},
        {"ftl_num_pq_pushed", decoder_stats_total.num_pq_pushed},
        {"ftl_num_nodes_popped", decoder_stats_total.num_nodes_popped},
        {"ftl_max_queue_size", decoder_stats_total.max_queue_size},
        {"ftl_heuristic_calls", decoder_stats_total.heuristic_calls},
        {"ftl_plain_heuristic_calls", decoder_stats_total.plain_heuristic_calls},
        {"ftl_projection_heuristic_calls", decoder_stats_total.projection_heuristic_calls},
        {"ftl_exact_refinement_calls", decoder_stats_total.exact_refinement_calls},
        {"ftl_lp_calls", decoder_stats_total.lp_calls},
        {"ftl_lp_reinserts", decoder_stats_total.lp_reinserts},
        {"ftl_projected_nodes_generated", decoder_stats_total.projected_nodes_generated},
        {"ftl_projected_nodes_refined", decoder_stats_total.projected_nodes_refined},
        {"ftl_total_lp_refinement_gain", decoder_stats_total.total_lp_refinement_gain},
        {"ftl_max_lp_refinement_gain", decoder_stats_total.max_lp_refinement_gain},
        {"ftl_lp_total_seconds", decoder_stats_total.lp_total_seconds},
    };

    if (args.stats_out_fname == "-") {
      std::cout << stats_json << std::endl;
      print_final_stats = false;
    } else {
      std::ofstream out(args.stats_out_fname, std::ofstream::out);
      out << stats_json << std::endl;
    }
  }

  if (print_final_stats) {
    std::cout << "num_shots = " << shot;
    std::cout << " num_low_confidence = " << num_low_confidence;
    if (has_obs) std::cout << " num_errors = " << num_errors;
    std::cout << " total_time_seconds = " << total_time_seconds;
    if (args.subset_detcost_size > 0) {
      std::cout << " lp_calls = " << decoder_stats_total.lp_calls;
      std::cout << " lp_reinserts = " << decoder_stats_total.lp_reinserts;
      std::cout << " projected_nodes_generated = " << decoder_stats_total.projected_nodes_generated;
      std::cout << " projected_nodes_refined = " << decoder_stats_total.projected_nodes_refined;
    }
    std::cout << std::endl;
  }
  return 0;
}
