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
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <thread>
#include <unordered_map>

#include "common.h"
#include "simplex.h"
#include "stim.h"
#include "utils.h"

namespace {

constexpr char kGariLayoutSchema[] = "tesseract.gari_layout.v1";

struct GariLayout {
  std::string schema;
  size_t source_detector_count;
  size_t gari_detector_count;
  std::vector<size_t> source_to_gari;
};

std::invalid_argument gari_layout_error(const std::string& path, const std::string& detail) {
  return std::invalid_argument("Invalid GARI layout '" + path + "': " + detail);
}

const nlohmann::json& required_layout_field(const nlohmann::json& document, const std::string& path,
                                            const char* field) {
  auto item = document.find(field);
  if (item == document.end()) {
    throw gari_layout_error(path, "missing required field '" + std::string(field) + "'.");
  }
  return *item;
}

size_t read_layout_size(const nlohmann::json& value, const std::string& path,
                        const std::string& field) {
  if (!value.is_number_integer()) {
    throw gari_layout_error(
        path, "field '" + field + "' must be an integer, but is " + value.type_name() + ".");
  }

  int64_t signed_value;
  if (value.is_number_unsigned()) {
    uint64_t unsigned_value = value.get<uint64_t>();
    if (unsigned_value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      throw gari_layout_error(path, "field '" + field + "' is too large.");
    }
    signed_value = static_cast<int64_t>(unsigned_value);
  } else {
    signed_value = value.get<int64_t>();
  }
  if (signed_value < 0) {
    throw gari_layout_error(path, "field '" + field + "' must be nonnegative, but is " +
                                      std::to_string(signed_value) + ".");
  }
  if (static_cast<uint64_t>(signed_value) > std::numeric_limits<size_t>::max()) {
    throw gari_layout_error(path, "field '" + field + "' is too large.");
  }
  return static_cast<size_t>(signed_value);
}

GariLayout load_gari_layout(const std::string& path) {
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::invalid_argument("Could not open GARI layout: " + path);
  }

  nlohmann::json document;
  try {
    input >> document;
  } catch (const nlohmann::json::exception& err) {
    throw gari_layout_error(path, "could not parse JSON: " + std::string(err.what()));
  }
  if (!document.is_object()) {
    throw gari_layout_error(path, "top-level JSON value must be an object.");
  }

  const nlohmann::json& schema_json = required_layout_field(document, path, "schema");
  if (!schema_json.is_string()) {
    throw gari_layout_error(path, "field 'schema' must be a string.");
  }
  std::string schema = schema_json.get<std::string>();
  if (schema != kGariLayoutSchema) {
    throw gari_layout_error(path, "field 'schema' must be '" + std::string(kGariLayoutSchema) +
                                      "', but is '" + schema + "'.");
  }

  size_t source_detector_count =
      read_layout_size(required_layout_field(document, path, "source_detector_count"), path,
                       "source_detector_count");
  size_t gari_detector_count = read_layout_size(
      required_layout_field(document, path, "gari_detector_count"), path, "gari_detector_count");
  if (gari_detector_count < source_detector_count) {
    throw gari_layout_error(path, "field 'gari_detector_count' must be at least " +
                                      std::to_string(source_detector_count) + ", but is " +
                                      std::to_string(gari_detector_count) + ".");
  }

  const nlohmann::json& mapping_json = required_layout_field(document, path, "source_to_gari");
  if (!mapping_json.is_array()) {
    throw gari_layout_error(path, "field 'source_to_gari' must be an array.");
  }
  if (mapping_json.size() != source_detector_count) {
    throw gari_layout_error(path, "field 'source_to_gari' must contain " +
                                      std::to_string(source_detector_count) + " entries, but has " +
                                      std::to_string(mapping_json.size()) + ".");
  }

  std::vector<size_t> source_to_gari;
  source_to_gari.reserve(source_detector_count);
  std::unordered_map<size_t, size_t> target_to_source;
  for (size_t source = 0; source < mapping_json.size(); ++source) {
    std::string field = "source_to_gari[" + std::to_string(source) + "]";
    size_t target = read_layout_size(mapping_json[source], path, field);
    if (target >= gari_detector_count) {
      throw gari_layout_error(path, "field '" + field + "' is " + std::to_string(target) +
                                        ", outside the target range [0, " +
                                        std::to_string(gari_detector_count) + ").");
    }
    auto [previous, inserted] = target_to_source.emplace(target, source);
    if (!inserted) {
      throw gari_layout_error(path, "source detectors " + std::to_string(previous->second) +
                                        " and " + std::to_string(source) +
                                        " both map to GARI detector " + std::to_string(target) +
                                        "; v1 layouts require an injective mapping.");
    }
    source_to_gari.push_back(target);
  }
  return {schema, source_detector_count, gari_detector_count, std::move(source_to_gari)};
}

std::vector<uint64_t> map_gari_hits(std::vector<uint64_t> source_hits, const GariLayout& layout,
                                    const std::string& path) {
  for (uint64_t& source : source_hits) {
    if (source >= layout.source_to_gari.size()) {
      throw gari_layout_error(path, "source detector index " + std::to_string(source) +
                                        " is outside the source range [0, " +
                                        std::to_string(layout.source_detector_count) + ").");
    }
    source = layout.source_to_gari[source];
  }
  std::sort(source_hits.begin(), source_hits.end());
  return source_hits;
}

}  // namespace

struct Args {
  std::string circuit_path;
  std::string dem_path;
  std::string gari_layout_path;
  std::string gari_layout_schema;
  size_t gari_source_detector_count = 0;
  size_t gari_detector_count = 0;
  bool no_merge_errors = false;

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

  // The most effective way of parallelizing simplex decoder is over shots,
  // confining each ILP solver to a single thread.
  size_t num_threads = 1;
  // The ILP solver we use (HiGHS) can exploit some parallelism while decoding a
  // single shot, but this is much less effective than just bulk parallelism
  // over shots. It is bad to combine ILP parallelism with bulk parallelization
  // over shots because it causes many threads to be spawned which overloads the
  // machine.
  bool enable_ilp_solver_parallelism = false;

  // A window length of 0 means to not use any windowing. A nonzero window
  // length activates sliding ILP window decoding. If a nonzero window length is
  // provided, then a nonzero window slide length must be provided as well.
  size_t window_length = 0;
  size_t window_slide_length = 0;

  bool verbose = false;
  bool print_stats = false;

  bool has_observables() {
    return append_observables || !obs_in_fname.empty() || (sample_num_shots > 0);
  }

  void validate() {
    if (circuit_path.empty() and dem_path.empty()) {
      throw std::invalid_argument("Must provide at least one of --circuit or --dem");
    }
    if (!gari_layout_path.empty() and dem_path.empty()) {
      throw std::invalid_argument("--gari-layout requires --dem.");
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
    if (!in_format.empty() && !stim::format_name_to_enum_map().contains(in_format)) {
      throw std::invalid_argument("Invalid format: " + in_format);
    }
    if (!obs_in_format.empty() && !stim::format_name_to_enum_map().contains(obs_in_format)) {
      throw std::invalid_argument("Invalid format: " + obs_in_format);
    }
    if (!out_format.empty() && !stim::format_name_to_enum_map().contains(out_format)) {
      throw std::invalid_argument("Invalid format: " + out_format);
    }
    if (!obs_in_fname.empty() and in_fname.empty()) {
      throw std::invalid_argument(
          "Cannot load observable flips without a corresponding detection "
          "event data file.");
    }
    if (num_threads == 0) {
      throw std::invalid_argument("--threads must be at least 1.");
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
        throw std::invalid_argument("Provided shot range must have end >= begin.");
      }
    }
    if ((window_length != 0) != (window_slide_length != 0)) {
      throw std::invalid_argument(
          "a window length > 0 is provided if and only if a window slide "
          "length > 0 is provided.");
    }
    if (window_slide_length > window_length) {
      throw std::invalid_argument("Must have window_slide_length <= window_length");
    }
    if (sample_num_shots > 0 and circuit_path.empty()) {
      throw std::invalid_argument("Cannot sample shots without a circuit.");
    }
  }

  void extract(SimplexConfig& config, std::vector<stim::SparseShot>& shots,
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

    config.merge_errors = !no_merge_errors;

    std::optional<GariLayout> gari_layout;
    if (!gari_layout_path.empty()) {
      gari_layout = load_gari_layout(gari_layout_path);
      size_t target_count = config.dem.count_detectors();
      if (target_count != gari_layout->gari_detector_count) {
        throw gari_layout_error(
            gari_layout_path, "field 'gari_detector_count' is " +
                                  std::to_string(gari_layout->gari_detector_count) + ", but DEM '" +
                                  dem_path + "' contains " + std::to_string(target_count) +
                                  " detectors.");
      }
      if (!circuit_path.empty()) {
        size_t source_count = circuit.count_detectors();
        if (source_count != gari_layout->source_detector_count) {
          throw gari_layout_error(gari_layout_path,
                                  "field 'source_detector_count' is " +
                                      std::to_string(gari_layout->source_detector_count) +
                                      ", but circuit '" + circuit_path + "' contains " +
                                      std::to_string(source_count) + " detectors.");
        }
      }
      gari_layout_schema = gari_layout->schema;
      gari_source_detector_count = gari_layout->source_detector_count;
      gari_detector_count = gari_layout->gari_detector_count;
    } else if (!circuit_path.empty() and !dem_path.empty() and
               circuit.count_detectors() != config.dem.count_detectors()) {
      throw std::invalid_argument(
          "Circuit '" + circuit_path + "' contains " + std::to_string(circuit.count_detectors()) +
          " detectors, but DEM '" + dem_path + "' contains " +
          std::to_string(config.dem.count_detectors()) +
          ". Supply --gari-layout when the source and target detector layouts differ.");
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
        std::vector<uint64_t> source_hits;
        for (size_t d = 0; d < num_detectors; d++) {
          if (dets[d][k]) {
            source_hits.push_back(d);
          }
        }
        if (gari_layout) {
          // The GARI matrix augments the physical syndrome with zero-valued virtual constraints.
          // Sparse shots contain mapped physical hits only, so virtual rows remain zero.
          shots[k].hits = map_gari_hits(std::move(source_hits), *gari_layout, gari_layout_path);
        } else {
          shots[k].hits = std::move(source_hits);
        }
      }
    }

    if (!in_fname.empty()) {
      // Load the shots from a file
      FILE* shots_file = fopen(in_fname.c_str(), "r");
      if (!shots_file) {
        throw std::invalid_argument("Could not open the file: " + in_fname);
      }
      stim::FileFormatData shots_in_format = stim::format_name_to_enum_map().at(in_format);
      size_t source_detector_count =
          gari_layout ? gari_layout->source_detector_count : config.dem.count_detectors();
      auto reader = stim::MeasureRecordReader<stim::MAX_BITWORD_WIDTH>::make(
          shots_file, shots_in_format.id, 0, source_detector_count,
          append_observables * config.dem.count_observables());

      // Load the shots from a file
      stim::SparseShot sparse_shot;
      sparse_shot.clear();
      while (reader->start_and_read_entire_record(sparse_shot)) {
        if (gari_layout) {
          sparse_shot.hits =
              map_gari_hits(std::move(sparse_shot.hits), *gari_layout, gari_layout_path);
        }
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
      stim::FileFormatData shots_obs_in_format = stim::format_name_to_enum_map().at(obs_in_format);
      auto obs_reader = stim::MeasureRecordReader<stim::MAX_BITWORD_WIDTH>::make(
          obs_file, shots_obs_in_format.id, 0, 0, config.dem.count_observables());
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
        throw std::invalid_argument("Shot range end is past end of shots array (size " +
                                    std::to_string(shots.size()) + ").");
      }
      std::vector<stim::SparseShot> shots_in_range(shots.begin() + shot_range_begin,
                                                   shots.begin() + shot_range_end);
      std::swap(shots_in_range, shots);
    }

    if (!out_fname.empty()) {
      // Create a writer instance to write the predicted obs to a file
      stim::FileFormatData predictions_out_format = stim::format_name_to_enum_map().at(out_format);
      FILE* predictions_file = stdout;
      // An output path of "-" means stdout.
      if (out_fname != "-") {
        predictions_file = fopen(out_fname.c_str(), "w");
      }
      writer = stim::MeasureRecordWriter::make(predictions_file, predictions_out_format.id);
      writer->begin_result_type('L');
      // TODO: ensure the fclose happens after all predictions are written to
      // the writer.
    }

    config.parallelize = enable_ilp_solver_parallelism;
    config.window_length = window_length;
    config.window_slide_length = window_slide_length;
    config.verbose = verbose;
  }
};

int main(int argc, char* argv[]) {
  std::cout.precision(16);
  argparse::ArgumentParser program("simplex");
  Args args;
  program.add_argument("--circuit").help("Stim circuit file path").store_into(args.circuit_path);
  program.add_argument("--dem").help("Stim dem file path").store_into(args.dem_path);
  program.add_argument("--gari-layout")
      .help(
          "JSON layout emitted by gari_convert. Maps detector data from the original circuit or "
          "source shot file into the supplied GARI matrix file. Unmapped virtual detector rows "
          "are treated as zero.")
      .metavar("FILE")
      .default_value(std::string(""))
      .store_into(args.gari_layout_path);
  program.add_argument("--no-merge-errors")
      .help("If provided, will not merge identical error mechanisms.")
      .store_into(args.no_merge_errors);
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
      .help("File to read detection events (and possibly observable flips) from")
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
      .help("Format of the file to read detection events from (" + in_formats + ")")
      .metavar(in_formats)
      .default_value(std::string(""))
      .store_into(args.in_format);
  program.add_argument("--in-includes-appended-observables", "--in_includes_appended_observables")
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
      .help("Format of the file to write observable flip predictions to (" + in_formats + ")")
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
      .default_value(size_t(
          std::thread::hardware_concurrency() == 0 ? 1 : std::thread::hardware_concurrency()))
      .store_into(args.num_threads);
  program.add_argument("--parallelize-ilp")
      .help(
          "Enable sub-shot parallelism with the ILP solver. Not recommended "
          "unless --threads=1")
      .default_value(bool(false))
      .store_into(args.enable_ilp_solver_parallelism)
      .flag();
  program.add_argument("--window-length")
      .help(
          "Length of sliding time window to use for sliding ILP window "
          "decoding (default = 0 = do "
          "not use windowing).")
      .metavar("N")
      .default_value(size_t(0))
      .store_into(args.window_length);
  program.add_argument("--window-slide-length")
      .help(
          "Length of the slide for each slide of the sliding time window for "
          "sliding ILP window "
          "decoding (default = 0 = do "
          "not use windowing).")
      .metavar("N")
      .default_value(size_t(0))
      .store_into(args.window_slide_length);
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
  SimplexConfig config;
  std::vector<stim::SparseShot> shots;
  std::unique_ptr<stim::MeasureRecordWriter> writer;
  try {
    args.validate();
    args.extract(config, shots, writer);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    return EXIT_FAILURE;
  }
  size_t num_observables = config.dem.count_observables();
  std::vector<stim::simd_bits<64>> obs_predicted(shots.size(),
                                                 stim::simd_bits<64>(num_observables));
  std::vector<double> cost_predicted(shots.size());
  std::vector<double> decoding_time_seconds(shots.size());
  const stim::DetectorErrorModel original_dem = config.dem.flattened();
  std::vector<std::unique_ptr<SimplexDecoder>> decoders(args.num_threads);
  std::vector<std::vector<size_t>> error_use_per_thread(
      args.num_threads, std::vector<size_t>(original_dem.count_errors()));
  bool has_obs = args.has_observables();
  size_t num_errors = 0;
  double total_time_seconds = 0;
  size_t shot = parallel_for_shots_in_order(
      shots.size(), args.num_threads,
      [&](size_t thread_index, size_t shot_index) {
        if (!decoders[thread_index]) {
          decoders[thread_index] = std::make_unique<SimplexDecoder>(config);
        }
        auto& decoder = *decoders[thread_index];
        auto& error_use = error_use_per_thread[thread_index];
        auto start_time = std::chrono::high_resolution_clock::now();
        decoder.decode_to_errors(shots[shot_index].hits);
        auto stop_time = std::chrono::high_resolution_clock::now();
        decoding_time_seconds[shot_index] =
            std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count() /
            1e6;
        obs_predicted[shot_index].clear();
        for (int obs_idx : decoder.get_flipped_observables(decoder.predicted_errors_buffer)) {
          obs_predicted[shot_index][obs_idx] ^= 1;
        }
        cost_predicted[shot_index] = decoder.cost_from_errors(decoder.predicted_errors_buffer);
        if (!has_obs or shots[shot_index].obs_mask == obs_predicted[shot_index]) {
          for (size_t ei : decoder.predicted_errors_buffer) {
            ++error_use[ei];
          }
        }
      },
      [&](size_t shot_index) {
        if (writer) {
          writer->write_bits(obs_predicted[shot_index].u8, num_observables);
          writer->write_end();
        }
        if (has_obs && obs_predicted[shot_index] != shots[shot_index].obs_mask) {
          ++num_errors;
        }
        total_time_seconds += decoding_time_seconds[shot_index];
        if (args.print_stats) {
          std::cout << "num_shots = " << (shot_index + 1);
          if (has_obs) {
            std::cout << " num_errors = " << num_errors;
          } else {
            std::cout << " num_errors = N/A";
          }
          std::cout << " total_time_seconds = " << total_time_seconds << std::endl;
          std::cout << "cost = " << cost_predicted[shot_index] << std::endl;
          std::cout.flush();
        }
        // Disable early termination due to \`--max-errors\` when we don't have the ground-truth
        // observables
        return !has_obs || num_errors < args.max_errors;
      });

  std::vector<size_t> error_use_totals(original_dem.count_errors());
  for (const auto& error_use : error_use_per_thread) {
    for (size_t ei = 0; ei < error_use_totals.size(); ++ei) {
      error_use_totals[ei] += error_use[ei];
    }
  }

  if (!args.dem_out_fname.empty()) {
    std::vector<size_t> counts(error_use_totals.begin(), error_use_totals.end());
    size_t num_usage_dem_shots = shot;
    if (has_obs) {
      // When we know the obs, we only count non-error shots.
      num_usage_dem_shots -= num_errors;
    }
    stim::DetectorErrorModel est_dem =
        common::dem_from_counts(original_dem, counts, num_usage_dem_shots);
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
                                 {"total_time_seconds", total_time_seconds},
                                 {"num_errors", has_obs ? nlohmann::json(num_errors) : nullptr},
                                 {"num_shots", shot},
                                 {"sample_num_shots", args.sample_num_shots}};

    if (!args.gari_layout_path.empty()) {
      stats_json["gari_layout_path"] = args.gari_layout_path;
      stats_json["gari_layout_schema"] = args.gari_layout_schema;
      stats_json["source_detector_count"] = args.gari_source_detector_count;
      stats_json["gari_detector_count"] = args.gari_detector_count;
    }

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
    if (has_obs) {
      std::cout << " num_errors = " << num_errors;
    }
    std::cout << " total_time_seconds = " << total_time_seconds;
    std::cout << std::endl;
  }
}
