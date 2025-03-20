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
#include <nlohmann/json.hpp>
#include <thread>

#include "common.h"
#include "stim.h"
#include "tesseract.h"

struct Args {
  std::string circuit_path;
  std::string dem_path;
  bool no_merge_errors = false;

  // Manifold orientation options
  uint64_t det_order_seed;
  size_t num_det_orders = 10;

  // Sampling options
  size_t sample_num_shots = 0;
  size_t max_errors = SIZE_MAX;
  uint64_t sample_seed;

  // If either of these are nonzero, only the shots in the range
  // [shot_range_begin, shot_range_end) will be decoded.
  size_t shot_range_begin = 0;
  size_t shot_range_end = 0;

  // Shot data file options
  std::string in_fname = "";
  std::string in_format = "";
  std::string obs_in_fname = "";
  std::string obs_in_format = "";
  bool append_observables = false;
  std::string out_fname = "";
  std::string out_format = "";

  // If dem_out is present, a usage-frequency dem will be computed and output to
  // this file.
  std::string dem_out_fname = "";

  // If stats_out_fname is present, basic statistics and metadata will be
  // written to this file.
  std::string stats_out_fname = "";

  // The most effective way of parallelizing is over shots, confining each ILP
  // solver to a single thread.
  size_t num_threads = 1;

  // Parameters that limit the algorithm's runtime at a potential accuracy or
  // completion cost.
  size_t det_beam;
  double det_penalty = 0;
  bool beam_climbing = false;
  bool no_revisit_dets = false;
  bool at_most_two_errors_per_detector = false;
  size_t pqlimit;

  bool verbose = false;
  bool print_stats = false;

  bool has_observables() {
    return append_observables || !obs_in_fname.empty() ||
           (sample_num_shots > 0);
  }

  void validate() {
    if (circuit_path.empty() and dem_path.empty()) {
      throw std::invalid_argument(
          "Must provide at least one of --circuit or --dem");
    }

    int num_data_sources = int(sample_num_shots > 0) + int(!in_fname.empty());
    if (num_data_sources != 1) {
      throw std::invalid_argument("Requires exactly 1 source of shots.");
    }
    if (!in_fname.empty() and in_format.empty()) {
      throw std::invalid_argument(
          "If --in is provided, must also specify --in-format.");
    }
    if (!out_fname.empty() and out_format.empty()) {
      throw std::invalid_argument(
          "If --out is provided, must also specify --out-format.");
    }
    if (!in_format.empty() &&
        !stim::format_name_to_enum_map().contains(in_format)) {
      throw std::invalid_argument("Invalid format: " + in_format);
    }
    if (!obs_in_format.empty() &&
        !stim::format_name_to_enum_map().contains(obs_in_format)) {
      throw std::invalid_argument("Invalid format: " + obs_in_format);
    }
    if (!out_format.empty() &&
        !stim::format_name_to_enum_map().contains(out_format)) {
      throw std::invalid_argument("Invalid format: " + out_format);
    }
    if (!obs_in_fname.empty() and in_fname.empty()) {
      throw std::invalid_argument(
          "Cannot load observable flips without a corresponding detection "
          "event data file.");
    }
    if (num_threads > 1000) {
      throw std::invalid_argument(
          "There is a maximum limit of 1000 threads imposed to avoid "
          "accidentally overloading a "
          "host. You specified " +
          std::to_string(num_threads) + "threads.");
    }
    if (shot_range_begin or shot_range_end) {
      if (shot_range_end < shot_range_begin) {
        throw std::invalid_argument(
            "Provided shot range must have end >= begin.");
      }
    }
    if (sample_num_shots > 0 and circuit_path.empty()) {
      throw std::invalid_argument("Cannot sample shots without a circuit.");
    }
    if (beam_climbing and det_beam == INF_DET_BEAM) {
      throw std::invalid_argument("Beam climbing requires a finite beam");
    }
  }

  void extract(TesseractConfig& config, std::vector<stim::SparseShot>& shots,
               std::unique_ptr<stim::MeasureRecordWriter>& writer) {
    // Get a circuit, if available
    stim::Circuit circuit;
    if (!circuit_path.empty()) {
      FILE* file = fopen(circuit_path.c_str(), "r");
      if (!file) {
        throw std::invalid_argument("Could not open the file: " + circuit_path);
      }
      circuit = stim::Circuit::from_file(file);
      fclose(file);
    }

    // Get a DEM, preferring to use the specified one and falling back to
    // generating one from the circuit
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

    if (!no_merge_errors) {
      config.dem = common::merge_identical_errors(config.dem);
    }

    // Sample orientations of the error model to use for the det priority
    {
      config.det_orders.resize(num_det_orders);
      std::mt19937_64 rng(det_order_seed);
      std::normal_distribution<double> dist(/*mean=*/0, /*stddev=*/1);

      std::vector<std::vector<double>> detector_coords =
          get_detector_coords(config.dem);

      std::vector<double> inner_products(config.dem.count_detectors());

      for (size_t det_order = 0; det_order < num_det_orders; ++det_order) {
        // Sample a direction
        std::vector<double> orientation_vector;
        for (size_t i = 0; i < detector_coords.at(0).size(); ++i) {
          orientation_vector.push_back(dist(rng));
        }

        for (size_t i = 0; i < detector_coords.size(); ++i) {
          inner_products[i] = 0;
          for (size_t j = 0; j < orientation_vector.size(); ++j) {
            inner_products[i] += detector_coords[i][j] * orientation_vector[j];
          }
        }
        std::vector<size_t> perm(config.dem.count_detectors());
        std::iota(perm.begin(), perm.end(), 0);
        std::sort(perm.begin(), perm.end(),
                  [&](const size_t& i, const size_t& j) {
                    return inner_products[i] > inner_products[j];
                  });
        // Invert the permutation
        std::vector<size_t> inv_perm(config.dem.count_detectors());
        for (size_t i = 0; i < perm.size(); ++i) {
          inv_perm[perm[i]] = i;
        }
        config.det_orders[det_order] = inv_perm;
      }
    }

    if (sample_num_shots > 0) {
      assert(!circuit_path.empty());
      std::mt19937_64 rng(sample_seed);
      size_t num_detectors = circuit.count_detectors();
      const auto [dets, obs] = stim::sample_batch_detection_events<64>(
          circuit, sample_num_shots, rng);
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
      // Load the shots from a file
      FILE* shots_file = fopen(in_fname.c_str(), "r");
      if (!shots_file) {
        throw std::invalid_argument("Could not open the file: " + in_fname);
      }
      stim::FileFormatData shots_in_format =
          stim::format_name_to_enum_map().at(in_format);
      auto reader = stim::MeasureRecordReader<stim::MAX_BITWORD_WIDTH>::make(
          shots_file, shots_in_format.id, 0, config.dem.count_detectors(),
          append_observables * config.dem.count_observables());

      // Load the shots from a file
      stim::SparseShot sparse_shot;
      sparse_shot.clear();
      while (reader->start_and_read_entire_record(sparse_shot)) {
        shots.push_back(sparse_shot);
        sparse_shot.clear();
      }
      fclose(shots_file);
    }

    // Load observable flips, if applicable
    if (!obs_in_fname.empty()) {
      FILE* obs_file = fopen(obs_in_fname.c_str(), "r");
      if (!obs_file) {
        throw std::invalid_argument("Could not open the file: " + obs_in_fname);
      }
      stim::FileFormatData shots_obs_in_format =
          stim::format_name_to_enum_map().at(obs_in_format);
      auto obs_reader =
          stim::MeasureRecordReader<stim::MAX_BITWORD_WIDTH>::make(
              obs_file, shots_obs_in_format.id, 0, 0,
              config.dem.count_observables());
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

    // Subselect shots, if applicable
    if (shot_range_begin or shot_range_end) {
      assert(shot_range_end >= shot_range_begin);
      if (shot_range_end > shots.size()) {
        throw std::invalid_argument(
            "Shot range end is past end of shots array.");
      }
      std::vector<stim::SparseShot> shots_in_range(
          shots.begin() + shot_range_begin, shots.begin() + shot_range_end);
      std::swap(shots_in_range, shots);
    }

    if (!out_fname.empty()) {
      // Create a writer instance to write the predicted obs to a file
      stim::FileFormatData predictions_out_format =
          stim::format_name_to_enum_map().at(out_format);
      FILE* predictions_file = stdout;
      if (out_fname != "-") {
        predictions_file = fopen(out_fname.c_str(), "w");
      }
      writer = stim::MeasureRecordWriter::make(predictions_file,
                                               predictions_out_format.id);
      writer->begin_result_type('L');
      // TODO: ensure the fclose happens after all predictions are written to
      // the writer.
    }
    config.det_beam = det_beam;
    config.det_penalty = det_penalty;
    config.beam_climbing = beam_climbing;
    config.no_revisit_dets = no_revisit_dets;
    config.at_most_two_errors_per_detector = at_most_two_errors_per_detector;
    config.pqlimit = pqlimit;
    config.verbose = verbose;
  }
};

int main(int argc, char* argv[]) {
  std::cout.precision(16);
  argparse::ArgumentParser program("simplex");
  Args args;
  program.add_argument("--circuit")
      .help("Stim circuit file path")
      .store_into(args.circuit_path);
  program.add_argument("--dem")
      .help("Stim dem file path")
      .store_into(args.dem_path);
  program.add_argument("--no-merge-errors")
      .help("If provided, will not merge identical error mechanisms.")
      .store_into(args.no_merge_errors);
  program.add_argument("--num-det-orders")
      .help(
          "Number of ways to orient the manifold when reordering the detectors")
      .metavar("N")
      .default_value(size_t(1))
      .store_into(args.num_det_orders);
  program.add_argument("--det-order-seed")
      .help(
          "Seed used when initializing the random detector traversal "
          "orderings.")
      .metavar("N")
      .default_value(static_cast<uint64_t>(std::random_device()()))
      .store_into(args.det_order_seed);
  program.add_argument("--sample-num-shots")
      .help(
          "If provided, will sample the requested number of shots from the "
          "Stim circuit and decode "
          "them. May end early if --max-errors errors are reached before "
          "decoding all shots.")
      .store_into(args.sample_num_shots);
  program.add_argument("--max-errors")
      .help(
          "If provided, will sample at least this many errors from the Stim "
          "circuit and decode "
          "them.")
      .store_into(args.max_errors);
  program.add_argument("--sample-seed")
      .help(
          "Seed used when initializing the random number generator for "
          "sampling shots")
      .metavar("N")
      .default_value(static_cast<uint64_t>(std::random_device()()))
      .store_into(args.sample_seed);
  program.add_argument("--shot-range-begin")
      .help(
          "Useful for processing a fragment of a file. If shot_range_begin == "
          "0 and shot_range_end "
          "== 0 (the default), then all available shots will be decoded. "
          "Otherwise, only those in "
          "the range [shot_range_begin, shot_range_end) will be decoded.")
      .default_value(size_t(0))
      .store_into(args.shot_range_begin);
  program.add_argument("--shot-range-end")
      .help(
          "Useful for processing a fragment of a file. If shot_range_begin == "
          "0 and shot_range_end "
          "== 0 (the default), then all available shots will be decoded. "
          "Otherwise, only those in "
          "the range [shot_range_begin, shot_range_end) will be decoded.")
      .default_value(size_t(0))
      .store_into(args.shot_range_end);
  program.add_argument("--in")
      .help(
          "File to read detection events (and possibly observable flips) from")
      .metavar("filename")
      .default_value(std::string(""))
      .store_into(args.in_fname);
  std::string in_formats = "";
  bool first = true;
  for (const auto& [key, value] : stim::format_name_to_enum_map()) {
    if (!first) in_formats += "/";
    first = false;
    in_formats += key;
  }
  program.add_argument("--in-format", "--in_format")
      .help("Format of the file to read detection events from (" + in_formats +
            ")")
      .metavar(in_formats)
      .default_value(std::string(""))
      .store_into(args.in_format);
  program
      .add_argument("--in-includes-appended-observables",
                    "--in_includes_appended_observables")
      .help(
          "If present, assumes that the observable flips are appended to the "
          "end of each shot.")
      .default_value(false)
      .store_into(args.append_observables)
      .flag();
  program.add_argument("--obs_in", "--obs-in")
      .help("File to read observable flips from")
      .metavar("filename")
      .default_value(std::string(""))
      .store_into(args.obs_in_fname);
  program.add_argument("--obs-in-format", "--obs_in_format")
      .help("Format of the file to observable flips from (" + in_formats + ")")
      .metavar(in_formats)
      .default_value(std::string(""))
      .store_into(args.obs_in_format);
  program.add_argument("--out")
      .help("File to write observable flip predictions to (or - for stdout)")
      .metavar("filename")
      .default_value(std::string(""))
      .store_into(args.out_fname);
  program.add_argument("--out-format")
      .help("Format of the file to write observable flip predictions to (" +
            in_formats + ")")
      .metavar(in_formats)
      .default_value(std::string(""))
      .store_into(args.out_format);
  program.add_argument("--dem-out")
      .help("File to write matching frequency dem to")
      .metavar("filename")
      .default_value(std::string(""))
      .store_into(args.dem_out_fname);
  program.add_argument("--stats-out")
      .help("File to write high-level statistics and metadata to")
      .metavar("filename")
      .default_value(std::string(""))
      .store_into(args.stats_out_fname);
  program.add_argument("--threads")
      .help("Number of decoder threads to use")
      .metavar("N")
      .default_value(size_t(std::thread::hardware_concurrency()))
      .store_into(args.num_threads);
  program.add_argument("--beam")
      .help("Beam to use for truncation (default = infinity)")
      .metavar("N")
      .default_value(INF_DET_BEAM)
      .store_into(args.det_beam);
  program.add_argument("--det-penalty")
      .help(
          "Penalty cost to add per activated detector in the residual "
          "syndrome.")
      .metavar("D")
      .default_value(0.0)
      .store_into(args.det_penalty);
  program.add_argument("--beam-climbing")
      .help("Use beam-climbing heuristic")
      .flag()
      .store_into(args.beam_climbing);
  program.add_argument("--no-revisit-dets")
      .help("Use no-revisit-dets heuristic")
      .flag()
      .store_into(args.no_revisit_dets);
  program.add_argument("--at-most-two-errors-per-detector")
      .help("Use heuristic limitation of at most 2 errors per detector")
      .flag()
      .store_into(args.at_most_two_errors_per_detector);
  program.add_argument("--pqlimit")
      .help("Maximum size of the priority queue (default = infinity)")
      .metavar("N")
      .default_value(std::numeric_limits<size_t>::max())
      .store_into(args.pqlimit);
  program.add_argument("--verbose")
      .help("Increases output verbosity")
      .flag()
      .store_into(args.verbose);
  program.add_argument("--print-stats")
      .help(
          "Prints out the number of shots (and number of errors, if known) "
          "during decoding.")
      .flag()
      .store_into(args.print_stats);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return EXIT_FAILURE;
  }
  args.validate();
  TesseractConfig config;
  std::vector<stim::SparseShot> shots;
  std::unique_ptr<stim::MeasureRecordWriter> writer;
  args.extract(config, shots, writer);
  std::atomic<size_t> next_unclaimed_shot;
  std::vector<std::atomic<bool>> finished(shots.size());
  std::vector<common::ObservablesMask> obs_predicted(shots.size());
  std::vector<double> cost_predicted(shots.size());
  std::vector<double> decoding_time_seconds(shots.size());
  std::vector<std::atomic<bool>> low_confidence(shots.size());
  std::vector<std::thread> decoder_threads;
  std::vector<std::atomic<size_t>> error_use_totals(config.dem.count_errors());
  bool has_obs = args.has_observables();
  std::atomic<bool> worker_threads_please_terminate = false;
  std::atomic<size_t> num_worker_threads_active;
  for (size_t t = 0; t < args.num_threads; ++t) {
    // After this value returns to 0, we know that no further shots will
    // transition to finished.
    ++num_worker_threads_active;
    decoder_threads.push_back(std::thread(
        [&config, &next_unclaimed_shot, &shots, &obs_predicted, &cost_predicted,
         &decoding_time_seconds, &low_confidence, &finished, &error_use_totals,
         &has_obs, &worker_threads_please_terminate,
         &num_worker_threads_active]() {
          TesseractDecoder decoder(config);
          std::vector<size_t> error_use(config.dem.count_errors());
          for (size_t shot; !worker_threads_please_terminate and
                            ((shot = next_unclaimed_shot++) < shots.size());) {
            auto start_time = std::chrono::high_resolution_clock::now();
            decoder.decode_to_errors(shots[shot].hits);
            auto stop_time = std::chrono::high_resolution_clock::now();
            decoding_time_seconds[shot] =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    stop_time - start_time)
                    .count() /
                1e6;
            obs_predicted[shot] =
                decoder.mask_from_errors(decoder.predicted_errors_buffer);
            low_confidence[shot] = decoder.low_confidence_flag;
            cost_predicted[shot] =
                decoder.cost_from_errors(decoder.predicted_errors_buffer);
            if (!has_obs or
                shots[shot].obs_mask_as_u64() == obs_predicted[shot]) {
              // Only count the error uses for shots that did not have a logical
              // error, if we know the obs flips.
              for (size_t ei : decoder.predicted_errors_buffer) {
                ++error_use[ei];
              }
            }
            finished[shot] = true;
          }
          // Add the error counts to the total
          for (size_t ei = 0; ei < error_use_totals.size(); ++ei) {
            error_use_totals[ei] += error_use[ei];
          }
          --num_worker_threads_active;
        }));
  }
  size_t num_errors = 0;
  size_t num_low_confidence = 0;
  double total_time_seconds = 0;
  size_t num_observables = config.dem.count_observables();
  size_t shot = 0;
  for (; shot < shots.size(); ++shot) {
    while (num_worker_threads_active and !finished[shot]) {
      // We break once the number of active worker threads is 0, at which point
      // there will be no further changes to finished[shot].
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    // There can be no further changes to finished[shot]. If it is true, we
    // process it and go to the next shot. If it is false, we break now as it
    // will never be decoded and no subsequent shots will be decoded.
    if (!finished[shot]) {
      assert(num_worker_threads_active == 0);
      // This and subsequent shots will never become decoded.
      break;
    }

    if (writer) {
      writer->write_bits((uint8_t*)&obs_predicted[shot], num_observables);
      writer->write_end();
    }

    if (low_confidence[shot]) {
      ++num_low_confidence;
    } else if (obs_predicted[shot] != shots[shot].obs_mask_as_u64()) {
      ++num_errors;
    }

    total_time_seconds += decoding_time_seconds[shot];

    if (args.print_stats) {
      std::cout << "num_shots = " << (shot + 1)
                << " num_low_confidence = " << num_low_confidence
                << " num_errors = " << num_errors
                << " total_time_seconds = " << total_time_seconds << std::endl;
      std::cout << "cost = " << cost_predicted[shot] << std::endl;
      std::cout.flush();
    }

    if (num_errors >= args.max_errors) {
      worker_threads_please_terminate = true;
    }
  }
  for (size_t t = 0; t < args.num_threads; ++t) {
    decoder_threads[t].join();
  }

  if (!args.dem_out_fname.empty()) {
    std::vector<size_t> counts(error_use_totals.begin(),
                               error_use_totals.end());
    size_t num_usage_dem_shots = shot;
    if (has_obs) {
      // When we know the obs, we only count non-error shots.
      num_usage_dem_shots -= num_errors;
    }
    stim::DetectorErrorModel est_dem =
        common::dem_from_counts(config.dem, counts, num_usage_dem_shots);
    std::ofstream out(args.dem_out_fname, std::ofstream::out);
    if (!out.is_open()) {
      throw std::invalid_argument("Failed to open " + args.dem_out_fname);
    }
    out << est_dem << '\n';
  }

  bool print_final_stats = true;
  if (!args.stats_out_fname.empty()) {
    nlohmann::json stats_json = {{"circuit_path", args.circuit_path},
                                 {"dem_path", args.dem_path},
                                 {"max_errors", args.max_errors},
                                 {"sample_seed", args.sample_seed},
                                 {"at_most_two_errors_per_detector",
                                  args.at_most_two_errors_per_detector},
                                 {"det_beam", args.det_beam},
                                 {"det_penalty", args.det_penalty},
                                 {"beam_climbing", args.beam_climbing},
                                 {"no_revisit_dets", args.no_revisit_dets},
                                 {"pqlimit", args.pqlimit},
                                 {"num_det_orders", args.num_det_orders},
                                 {"det_order_seed", args.det_order_seed},
                                 {"total_time_seconds", total_time_seconds},
                                 {"num_errors", num_errors},
                                 {"num_low_confidence", num_low_confidence},
                                 {"num_shots", shot},
                                 {"num_threads", args.num_threads},
                                 {"sample_num_shots", args.sample_num_shots}};

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
    if (has_obs) {
      std::cout << " num_errors = " << num_errors;
    }
    std::cout << " total_time_seconds = " << total_time_seconds;
    std::cout << std::endl;
  }
}
