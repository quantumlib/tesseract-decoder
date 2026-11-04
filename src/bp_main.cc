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
#include <array>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <memory>
#include <nlohmann/json.hpp>
#include <thread>
#include <vector>

#include "bp/multi_pass_bp_decoder.h"
#include "bp/osd_post_processor.h"
#include "bp/tesseract_bp_decoder.h"
#include "common.h"
#include "multi_pass/multi_pass_tesseract_decoder.h"
#include "stim.h"
#include "tesseract.h"
#include "utils.h"

using namespace bp;
using namespace tesseract_decoder;

struct Args {
  std::string circuit_path;
  std::string dem_path;

  size_t sample_num_shots = 0;
  size_t max_errors = SIZE_MAX;
  uint64_t sample_seed = 12345;

  std::string in_fname;
  std::string in_format;
  std::string obs_in_fname;
  std::string obs_in_format;
  bool append_observables = false;
  std::string out_fname;
  std::string out_format;

  size_t max_iter = 20;
  size_t num_threads = 1;
  double normalization_factor = 0.75;
  int osd_order = 10;
  int osd_weight = 2;
  int osd_w2_limit = 60;
  bool fast = false;
  bool use_batched_bp = true;
  bool random_schedule = false;
  double osd_truncation_factor = 1.2;
  bool profile = false;
  bool multipass = true;
  size_t num_passes = 2;
  bool multipass_soft_reweight = true;
  std::string bp_mode_str = "full";
  bool tesseract_fallback = false;
  int fallback_det_beam = 20;
  double fallback_margin = 1.0;

  // Explicit BP-OSD ablation flags
  bool bp_simd_batch = true;
  bool bp_compaction = false;
  bool bp_merge_duplicates = true;
  bool bp_serial_schedule = true;
  BpReweightMode bp_reweight_mode = BpReweightMode::kSoft;
  bool osd_early_exit = true;
  bool osd_col_truncation = true;
  bool osd_partial_sort = true;
  bool osd_min_syndrome_snapshot = true;
  bool osd_relay_legs = true;
  bool osd_alpha_isd = true;

  std::string stats_out_fname;
  std::string sinter_csv_out;
  bool print_stats = false;

  bool has_observables() const {
    return append_observables || !obs_in_fname.empty() || (sample_num_shots > 0);
  }

  void validate() const {
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
    if (num_threads == 0) {
      throw std::invalid_argument("--threads must be at least 1.");
    }
    if (bp_mode_str != "full" && bp_mode_str != "wavefront") {
      throw std::invalid_argument("--bp-mode must be one of: full, wavefront");
    }
  }

  void extract(BPParams& params, stim::DetectorErrorModel& dem,
               std::vector<stim::SparseShot>& shots,
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
      dem = stim::DetectorErrorModel::from_file(file);
      fclose(file);
    } else {
      dem = stim::ErrorAnalyzer::circuit_to_detector_error_model(circuit, false, true, true, 1,
                                                                 false, false);
    }

    params.max_iter = max_iter;
    params.normalization_factor = static_cast<float>(normalization_factor);
    params.random_schedule = random_schedule;
    params.random_seed = sample_seed;
    params.osd_truncation_factor = static_cast<float>(osd_truncation_factor);
    params.osd_order = osd_order;
    params.osd_weight = osd_weight;
    params.use_soft_reweighting = (bp_reweight_mode == BpReweightMode::kSoft);
    params.bp_mode = (bp_mode_str == "wavefront" && !random_schedule) ? BpMode::kWavefront
                                                                      : BpMode::kFull;
    params.fallback_margin = fallback_margin;

    params.bp_simd_batch = bp_simd_batch;
    params.bp_compaction = bp_compaction;
    params.bp_merge_duplicates = bp_merge_duplicates;
    params.bp_serial_schedule = bp_serial_schedule;
    params.bp_reweight_mode = bp_reweight_mode;
    params.osd_early_exit = osd_early_exit;
    params.osd_col_truncation = osd_col_truncation;
    params.osd_partial_sort = osd_partial_sort;
    params.osd_min_syndrome_snapshot = osd_min_syndrome_snapshot;
    params.osd_relay_legs = osd_relay_legs;
    params.osd_alpha_isd = osd_alpha_isd;

    if (sample_num_shots > 0) {
      std::mt19937_64 rng(sample_seed);
      if (!circuit_path.empty()) {
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
      } else {
        stim::DetectorErrorModel flat = dem.flattened();
        size_t num_detectors = flat.count_detectors();
        size_t num_obs = flat.count_observables();
        stim::simd_bit_table<64> dets(num_detectors, sample_num_shots);
        stim::simd_bit_table<64> obs_T(sample_num_shots, std::max<size_t>(1, num_obs));
        std::vector<uint32_t> det_targets;
        std::vector<uint32_t> obs_targets;
        for (const auto& inst : flat.instructions) {
          if (inst.type != stim::DemInstructionType::DEM_ERROR) continue;
          double p = inst.arg_data.empty() ? 0.0 : inst.arg_data[0];
          if (p <= 0.0) continue;
          det_targets.clear();
          obs_targets.clear();
          for (const auto& t : inst.target_data) {
            if (t.is_relative_detector_id()) det_targets.push_back(t.val());
            else if (t.is_observable_id()) obs_targets.push_back(t.val());
          }
          std::geometric_distribution<size_t> geom(p);
          size_t k = geom(rng);
          while (k < sample_num_shots) {
            for (uint32_t d : det_targets) dets[d][k] ^= 1;
            for (uint32_t o : obs_targets) obs_T[k][o] ^= 1;
            k += 1 + geom(rng);
          }
        }
        shots.resize(sample_num_shots);
        for (size_t k = 0; k < sample_num_shots; k++) {
          shots[k].obs_mask = obs_T[k];
          for (size_t d = 0; d < num_detectors; d++) {
            if (dets[d][k]) shots[k].hits.push_back(d);
          }
        }
      }
    }

    if (!in_fname.empty()) {
      FILE* shots_file = fopen(in_fname.c_str(), "r");
      if (!shots_file) throw std::invalid_argument("Could not open the file: " + in_fname);
      stim::FileFormatData shots_in_format = stim::format_name_to_enum_map().at(in_format);
      auto reader = stim::MeasureRecordReader<stim::MAX_BITWORD_WIDTH>::make(
          shots_file, shots_in_format.id, 0, dem.count_detectors(),
          append_observables * dem.count_observables());

      stim::SparseShot sparse_shot;
      while (reader->start_and_read_entire_record(sparse_shot)) {
        shots.push_back(sparse_shot);
        sparse_shot.clear();
      }
      fclose(shots_file);
    }

    if (!obs_in_fname.empty()) {
      FILE* obs_file = fopen(obs_in_fname.c_str(), "r");
      if (!obs_file) throw std::invalid_argument("Could not open the file: " + obs_in_fname);
      stim::FileFormatData shots_obs_in_format = stim::format_name_to_enum_map().at(obs_in_format);
      auto obs_reader = stim::MeasureRecordReader<stim::MAX_BITWORD_WIDTH>::make(
          obs_file, shots_obs_in_format.id, 0, 0, dem.count_observables());
      stim::SparseShot sparse_shot;
      size_t num_obs_shots = 0;
      while (obs_reader->start_and_read_entire_record(sparse_shot)) {
        shots[num_obs_shots++].obs_mask = sparse_shot.obs_mask;
        sparse_shot.clear();
      }
      fclose(obs_file);
    }

    if (!out_fname.empty()) {
      stim::FileFormatData predictions_out_format = stim::format_name_to_enum_map().at(out_format);
      FILE* predictions_file = (out_fname == "-") ? stdout : fopen(out_fname.c_str(), "w");
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
  program.add_argument("--normalization-factor")
      .default_value(0.75)
      .store_into(args.normalization_factor);
  program.add_argument("--osd-order").default_value(10).store_into(args.osd_order);
  program.add_argument("--osd-weight").default_value(2).store_into(args.osd_weight);
  program.add_argument("--osd-w2-limit").default_value(60).store_into(args.osd_w2_limit);
  program.add_argument("--osd-truncation-factor")
      .help(
          "Multiplier defining the maximum number of free columns for the dense OSD Gaussian "
          "elimination matrix (e.g. 1.2 bounds to 1.2x the number of defect detectors). Use 0.0 to "
          "disable truncation.")
      .default_value(1.2)
      .store_into(args.osd_truncation_factor);
  program.add_argument("--disable-osd-early-exit")
      .help("Disable residual syndrome early stopping in OSD elimination")
      .flag()
      .action([&args](const std::string&) { args.osd_early_exit = false; });
  program.add_argument("--fast")
      .help("Use fast BP-OSD preset (max_iter=10, osd_order=10, osd_weight=0, batched)")
      .flag()
      .store_into(args.fast);
  program.add_argument("--profile")
      .help("Print decode and per-phase OSD timing breakdowns")
      .flag()
      .store_into(args.profile);
  program.add_argument("--batched")
      .help("Use SIMD batching across shots (enabled by default)")
      .flag()
      .action([&args](const std::string&) { args.bp_simd_batch = true; });
  program.add_argument("--bp-simd-batch")
      .help("Enable 16-lane AVX2 SIMD batching across shots in BP (default: enabled)")
      .flag()
      .action([&args](const std::string&) { args.bp_simd_batch = true; });
  program.add_argument("--no-bp-simd-batch")
      .help("Disable 16-lane SIMD batching in BP (force scalar single-shot batch size 1)")
      .flag()
      .action([&args](const std::string&) { args.bp_simd_batch = false; });
  program.add_argument("--bp-compaction")
      .help("Enable 2-stage screening and unconverged survivor compaction in BP (default: disabled)")
      .flag()
      .action([&args](const std::string&) { args.bp_compaction = true; });
  program.add_argument("--no-bp-compaction")
      .help("Disable 2-stage survivor compaction in BP (run all max_iter on uncompacted batch)")
      .flag()
      .action([&args](const std::string&) { args.bp_compaction = false; });
  program.add_argument("--multipass")
      .help("Enable Multi-Pass Shattered BP decoding (default: enabled)")
      .flag()
      .action([&args](const std::string&) { args.multipass = true; });
  program.add_argument("--no-multipass")
      .help("Disable Multi-Pass Shattered BP decoding (use monolithic Tanner graph)")
      .flag()
      .action([&args](const std::string&) { args.multipass = false; });
  program.add_argument("--bp-merge-duplicates")
      .help("Merge DEM error mechanisms with identical detector symptoms (default: enabled)")
      .flag()
      .action([&args](const std::string&) { args.bp_merge_duplicates = true; });
  program.add_argument("--no-bp-merge-duplicates")
      .help("Disable merging of duplicate-symptom DEM columns")
      .flag()
      .action([&args](const std::string&) { args.bp_merge_duplicates = false; });
  program.add_argument("--bp-serial")
      .help("Use layered serial check updates in BP (default: enabled)")
      .flag()
      .action([&args](const std::string&) { args.bp_serial_schedule = true; });
  program.add_argument("--no-bp-serial")
      .help("Use parallel flooding check updates in BP instead of layered serial updates")
      .flag()
      .action([&args](const std::string&) { args.bp_serial_schedule = false; });
  program.add_argument("--bp-schedule")
      .help("BP message-passing schedule: serial or flooding (default: serial)")
      .action([&args](const std::string& val) {
        if (val == "serial") {
          args.bp_serial_schedule = true;
        } else if (val == "flooding" || val == "parallel") {
          args.bp_serial_schedule = false;
        } else {
          throw std::invalid_argument("--bp-schedule must be 'serial' or 'flooding'");
        }
      });
  program.add_argument("--bp-random-order")
      .help("Enable per-iteration randomized check permutation in BP (default: disabled)")
      .flag()
      .action([&args](const std::string&) { args.random_schedule = true; });
  program.add_argument("--no-bp-random-order")
      .help("Disable per-iteration randomized check permutation (use fixed order 0..M-1)")
      .flag()
      .action([&args](const std::string&) { args.random_schedule = false; });
  program.add_argument("--random-schedule")
      .help("Use randomized check node permutation in serial BP to prevent trapping sets")
      .flag()
      .action([&args](const std::string&) { args.random_schedule = true; });
  program.add_argument("--num-passes")
      .help("Number of passes in Multi-Pass BP (default: 2)")
      .default_value(size_t(2))
      .store_into(args.num_passes);
  program.add_argument("--bp-reweight")
      .help("Cross-pass LLR reweighting mode: soft (75/25 blend), hard (100%), or none (default: soft)")
      .action([&args](const std::string& val) {
        if (val == "soft") {
          args.bp_reweight_mode = BpReweightMode::kSoft;
          args.multipass_soft_reweight = true;
        } else if (val == "hard") {
          args.bp_reweight_mode = BpReweightMode::kHard;
          args.multipass_soft_reweight = false;
        } else if (val == "none") {
          args.bp_reweight_mode = BpReweightMode::kNone;
          args.multipass_soft_reweight = false;
        } else {
          throw std::invalid_argument("--bp-reweight must be 'soft', 'hard', or 'none'");
        }
      });
  program.add_argument("--bp-soft-reweight")
      .help("Enable soft (75% conditional / 25% prior) cross-pass LLR reweighting (default: enabled)")
      .flag()
      .action([&args](const std::string&) {
        args.bp_reweight_mode = BpReweightMode::kSoft;
        args.multipass_soft_reweight = true;
      });
  program.add_argument("--no-bp-soft-reweight")
      .help("Disable soft cross-pass LLR reweighting (use hard replacement)")
      .flag()
      .action([&args](const std::string&) {
        args.bp_reweight_mode = BpReweightMode::kHard;
        args.multipass_soft_reweight = false;
      });
  program.add_argument("--multipass-soft-reweight")
      .help("Use soft LLR reweighting across passes (enabled by default)")
      .flag()
      .action([&args](const std::string&) {
        args.bp_reweight_mode = BpReweightMode::kSoft;
        args.multipass_soft_reweight = true;
      });
  program.add_argument("--disable-multipass-soft-reweight")
      .help("Disable soft LLR reweighting across passes")
      .flag()
      .action([&args](const std::string&) {
        args.bp_reweight_mode = BpReweightMode::kHard;
        args.multipass_soft_reweight = false;
      });
  program.add_argument("--osd-early-exit")
      .help("Enable residual-syndrome early exit during OSD Gauss-Jordan RREF (default: enabled)")
      .flag()
      .action([&args](const std::string&) { args.osd_early_exit = true; });
  program.add_argument("--no-osd-early-exit")
      .help("Disable residual-syndrome early exit in OSD RREF (reduce all rank(H) pivots)")
      .flag()
      .action([&args](const std::string&) { args.osd_early_exit = false; });
  program.add_argument("--osd-col-truncation")
      .help("Enable K=1.2*M column truncation in OSD matrix construction/RREF (default: enabled)")
      .flag()
      .action([&args](const std::string&) { args.osd_col_truncation = true; });
  program.add_argument("--no-osd-col-truncation")
      .help("Disable K=1.2*M column truncation in OSD (reduce across all N columns)")
      .flag()
      .action([&args](const std::string&) { args.osd_col_truncation = false; });
  program.add_argument("--osd-partial-sort")
      .help("Use O(N + K log K) std::nth_element partial sort in OSD (default: enabled)")
      .flag()
      .action([&args](const std::string&) { args.osd_partial_sort = true; });
  program.add_argument("--no-osd-partial-sort")
      .help("Disable partial sort in OSD (use full std::sort over all N variables)")
      .flag()
      .action([&args](const std::string&) { args.osd_partial_sort = false; });
  program.add_argument("--osd-min-syndrome")
      .help("Seed OSD from the minimum-syndrome BP snapshot L^(t*) (default: enabled)")
      .flag()
      .action([&args](const std::string&) { args.osd_min_syndrome_snapshot = true; });
  program.add_argument("--no-osd-min-syndrome")
      .help("Seed OSD only from the final BP iteration L^(T) instead of L^(t*)")
      .flag()
      .action([&args](const std::string&) { args.osd_min_syndrome_snapshot = false; });
  program.add_argument("--osd-relay")
      .help("Enable 2 damped BP relay legs (R1, R2) on unconverged shots (default: enabled)")
      .flag()
      .action([&args](const std::string&) { args.osd_relay_legs = true; });
  program.add_argument("--no-osd-relay")
      .help("Disable damped BP relay legs (R1, R2)")
      .flag()
      .action([&args](const std::string&) { args.osd_relay_legs = false; });
  program.add_argument("--osd-alpha-isd")
      .help("Enable prior-blended alpha-ISD seeds and higher-order OSD search (default: enabled)")
      .flag()
      .action([&args](const std::string&) { args.osd_alpha_isd = true; });
  program.add_argument("--no-osd-alpha-isd")
      .help("Disable prior-blended alpha-ISD seeds and force OSD-0")
      .flag()
      .action([&args](const std::string&) { args.osd_alpha_isd = false; });
  program.add_argument("--bp-mode")
      .help("BP execution mode: full or wavefront (default: full)")
      .default_value(std::string("full"))
      .store_into(args.bp_mode_str);
  program.add_argument("--tesseract-fallback")
      .help("Enable fallback to Monolithic Tesseract on low-confidence shots")
      .flag()
      .store_into(args.tesseract_fallback);
  program.add_argument("--fallback-det-beam")
      .help("Beam size for Tesseract (default: 20)")
      .default_value(20)
      .store_into(args.fallback_det_beam);
  program.add_argument("--fallback-margin")
      .help("Margin threshold delta_w (default: 1.0)")
      .default_value(1.0)
      .store_into(args.fallback_margin);
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
  program.add_argument("--print-stats").flag().store_into(args.print_stats);

  std::vector<std::string> normalized_args;
  normalized_args.reserve(argc);
  for (int i = 0; i < argc; ++i) {
    std::string a(argv[i]);
    if (a.rfind("--", 0) == 0) {
      auto eq_pos = a.find('=');
      if (eq_pos != std::string::npos) {
        normalized_args.push_back(a.substr(0, eq_pos));
        normalized_args.push_back(a.substr(eq_pos + 1));
        continue;
      }
    }
    normalized_args.push_back(std::move(a));
  }

  try {
    program.parse_args(normalized_args);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return EXIT_FAILURE;
  }
  if (args.fast) {
    if (!program.is_used("--max-iter")) args.max_iter = 10;
    if (!program.is_used("--osd-order")) args.osd_order = 10;
    if (!program.is_used("--osd-weight")) args.osd_weight = 0;
    args.use_batched_bp = true;
  }
  args.validate();

  BPParams params;
  stim::DetectorErrorModel dem;
  std::vector<stim::SparseShot> shots;
  std::unique_ptr<stim::MeasureRecordWriter> writer;
  args.extract(params, dem, shots, writer);

  std::vector<std::unique_ptr<TesseractBpDecoder>> decoders(args.num_threads);
  std::vector<std::unique_ptr<MultiPassBpDecoder>> mp_decoders(args.num_threads);
  std::vector<std::shared_ptr<OsdPostProcessor>> pps(args.num_threads);
  std::vector<std::unique_ptr<TesseractDecoder>> fallback_decoders(args.num_threads);

  MultiPassBpConfig mp_config{params, args.num_passes, {}, args.multipass_soft_reweight};

  const size_t total_shots = shots.size();
  const size_t num_observables = dem.count_observables();
  const bool has_obs = args.has_observables();
  std::atomic<size_t> num_errors(0);
  std::atomic<size_t> num_discards(0);
  std::atomic<double> total_time_seconds(0);
  std::atomic<size_t> processed_shots(0);
  std::atomic<size_t> num_stage2_triggered(0);
  std::atomic<size_t> num_fallback_triggered(0);
  std::atomic<size_t> num_fallback_substituted(0);
  // Stage-2 trigger counts indexed by the low-confidence reason code (1..5)
  // emitted by MultiPassBpDecoder. Reason 5 is the only one raised on shots
  // where both BP passes converged without OSD (lc0 == 0 && lc1 == 0).
  std::array<size_t, 6> num_stage2_by_reason{};
  double stage1_time_seconds = 0.0;
  double stage2_time_seconds = 0.0;
  double stage3_time_seconds = 0.0;

  const auto start_global_time = std::chrono::high_resolution_clock::now();
  const size_t target_batches = std::max<size_t>(1, args.num_threads * 4);
  const size_t dynamic_chunk =
      ((total_shots + target_batches - 1) / target_batches + 15) & ~size_t(15);
  const size_t batch_size =
      args.use_batched_bp ? std::clamp<size_t>(dynamic_chunk, 16, 256) : 1;
  const size_t num_batches = (total_shots + batch_size - 1) / batch_size;
  const size_t print_stride = args.use_batched_bp ? 1024 : 100;
  std::vector<std::vector<std::vector<uint8_t>>> all_predictions(num_batches);
  std::vector<std::vector<uint8_t>> all_low_conf(num_batches);
  std::vector<std::vector<double>> all_pred_costs(num_batches);
  std::vector<double> batch_time_seconds(num_batches);
  const bool deferred_cascade = (args.multipass && args.tesseract_fallback);

  parallel_for_shots_in_order(
      num_batches, args.num_threads,
      [&](size_t thread_idx, size_t batch_idx) {
        if (args.multipass) {
          if (!mp_decoders[thread_idx]) {
            mp_decoders[thread_idx] = std::make_unique<MultiPassBpDecoder>(dem, mp_config);
          }
        }
        if (!args.multipass) {
          if (!decoders[thread_idx]) {
            decoders[thread_idx] = std::make_unique<TesseractBpDecoder>(dem, params);
            if (args.osd_order >= 0) {
              pps[thread_idx] = decoders[thread_idx]->create_post_processor();
            }
          }
          if (args.tesseract_fallback && !fallback_decoders[thread_idx]) {
            TesseractConfig fb_cfg;
            fb_cfg.dem = dem;
            fb_cfg.det_beam = args.fallback_det_beam;
            fb_cfg.beam_climbing = true;
            fb_cfg.no_revisit_dets = true;
            fb_cfg.pqlimit = (args.fallback_det_beam >= 20) ? 1000000 : 200000;
            fb_cfg.detector_orders = make_detector_orders(
                (args.fallback_det_beam >= 20) ? 21 : 16,
                DetectorOrder::Method::Index, 2384753);
            fallback_decoders[thread_idx] = std::make_unique<TesseractDecoder>(fb_cfg);
          }
        }

        size_t shot_start = batch_idx * batch_size;
        size_t shot_end = std::min(shot_start + batch_size, total_shots);
        std::vector<std::vector<uint64_t>> batch_syndromes;
        batch_syndromes.reserve(shot_end - shot_start);
        for (size_t s = shot_start; s < shot_end; ++s) {
          batch_syndromes.emplace_back(shots[s].hits.begin(), shots[s].hits.end());
        }

        auto start_batch = std::chrono::high_resolution_clock::now();
        std::vector<uint8_t> low_confidence_shots;
        std::vector<std::vector<size_t>> predicted_errors_batch;

        if (args.multipass) {
          all_predictions[batch_idx] = mp_decoders[thread_idx]->decode_batch(
              batch_syndromes,
              args.tesseract_fallback ? &all_low_conf[batch_idx] : nullptr,
              args.tesseract_fallback ? &all_pred_costs[batch_idx] : nullptr);
        } else {
          all_predictions[batch_idx] = decoders[thread_idx]->decode_batch(
              batch_syndromes, pps[thread_idx], /*custom_priors_per_shot=*/nullptr,
              args.tesseract_fallback ? &predicted_errors_batch : nullptr,
              /*sparse_custom_priors_per_shot=*/nullptr,
              args.tesseract_fallback ? &low_confidence_shots : nullptr);
          if (args.tesseract_fallback) {
            size_t actual_size = shot_end - shot_start;
            auto& fb_decoder = *fallback_decoders[thread_idx];
            for (size_t b = 0; b < actual_size; ++b) {
              if (b < low_confidence_shots.size() && low_confidence_shots[b] == 1) {
                num_fallback_triggered++;
                const auto& dets = batch_syndromes[b];
                DecodeResult t_res = fb_decoder.decode_result(dets);
                if (!t_res.low_confidence) {
                  double t_cost = t_res.total_cost;
                  double bp_cost = decoders[thread_idx]->cost_from_errors(predicted_errors_batch[b]);
                  if (t_cost <= bp_cost + 1e-4) {
                    num_fallback_substituted++;
                    std::vector<uint8_t> new_flips(num_observables, 0);
                    for (int obs : t_res.predictions) {
                      if (obs >= 0 && static_cast<size_t>(obs) < num_observables) {
                        new_flips[obs] = 1;
                      }
                    }
                    all_predictions[batch_idx][b] = std::move(new_flips);
                  }
                }
              }
            }
          }
        }
        auto stop_batch = std::chrono::high_resolution_clock::now();
        batch_time_seconds[batch_idx] =
            std::chrono::duration_cast<std::chrono::microseconds>(stop_batch - start_batch)
                .count() /
            1e6;
      },
      [&](size_t batch_idx) {
        total_time_seconds = total_time_seconds + batch_time_seconds[batch_idx];
        stage1_time_seconds += batch_time_seconds[batch_idx];
        if (deferred_cascade) {
          return true;
        }
        size_t shot_start = batch_idx * batch_size;
        size_t shot_end = std::min(shot_start + batch_size, total_shots);
        size_t actual_size = shot_end - shot_start;

        for (size_t b = 0; b < actual_size; ++b) {
          size_t s = shot_start + b;
          const auto& predicted_flips = all_predictions[batch_idx][b];
          stim::simd_bits<64> predicted_obs(num_observables);
          for (size_t o = 0; o < num_observables; ++o) {
            if (predicted_flips[o]) predicted_obs[o] ^= 1;
          }
          if (writer) {
            writer->write_bits(predicted_obs.u8, num_observables);
            writer->write_end();
          }
          if (has_obs && predicted_obs != shots[s].obs_mask) {
            num_errors++;
          }
          processed_shots++;
          if (args.print_stats && processed_shots % print_stride == 0) {
            std::cout << "Processed " << processed_shots << " shots, errors: "
                      << (has_obs ? std::to_string(num_errors.load()) : std::string("N/A"))
                      << std::endl;
          }
        }
        return !has_obs || num_errors < args.max_errors;
      });

  if (deferred_cascade) {
    struct SuspectRef {
      size_t batch_idx;
      size_t b;
      size_t shot_idx;
      uint8_t lc;
    };
    std::vector<SuspectRef> suspect_refs;
    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      size_t shot_start = batch_idx * batch_size;
      size_t shot_end = std::min(shot_start + batch_size, total_shots);
      size_t actual_size = shot_end - shot_start;
      for (size_t b = 0; b < actual_size; ++b) {
        if (b < all_low_conf[batch_idx].size() && all_low_conf[batch_idx][b] != 0) {
          suspect_refs.push_back({batch_idx, b, shot_start + b, all_low_conf[batch_idx][b]});
        }
      }
    }
    num_stage2_triggered = suspect_refs.size();
    for (const auto& ref : suspect_refs) {
      num_stage2_by_reason[std::min<size_t>(ref.lc, num_stage2_by_reason.size() - 1)]++;
    }

    struct Stage3Ref {
      size_t batch_idx;
      size_t b;
      size_t shot_idx;
      uint8_t lc;
      uint8_t m_lc;
      std::vector<uint8_t> stage2_obs;
      double mono_cost;
      double stage1_cost;
    };
    std::vector<Stage3Ref> stage3_refs;

    if (!suspect_refs.empty()) {
      const bool is_large_qldpc = (dem.count_detectors() > 480 && num_observables > 1);
      std::vector<std::unique_ptr<MultiPassBpDecoder>> s2_mp_decoders(args.num_threads);
      const size_t s2_batch_size = std::min<size_t>(
          512, std::max<size_t>(1, (suspect_refs.size() + args.num_threads - 1) / args.num_threads));
      const size_t num_s2_batches = (suspect_refs.size() + s2_batch_size - 1) / s2_batch_size;
      std::vector<std::vector<Stage3Ref>> per_s2_stage3(num_s2_batches);
      std::vector<double> s2_time_seconds(num_s2_batches, 0.0);

      parallel_for_shots_in_order(
          num_s2_batches, args.num_threads,
          [&](size_t thread_idx, size_t s2_idx) {
            size_t k_start = s2_idx * s2_batch_size;
            size_t k_end = std::min(k_start + s2_batch_size, suspect_refs.size());
            std::vector<std::vector<uint64_t>> s2_syndromes;
            s2_syndromes.reserve(k_end - k_start);
            for (size_t k = k_start; k < k_end; ++k) {
              const size_t s = suspect_refs[k].shot_idx;
              s2_syndromes.emplace_back(shots[s].hits.begin(), shots[s].hits.end());
            }
            if ((is_large_qldpc &&
                 (dem.count_detectors() <= 1500 || dem.count_detectors() > 6000)) ||
                (num_observables == 1 && dem.count_detectors() > 1500)) {
              if (!s2_mp_decoders[thread_idx]) {
                BPParams s2_params = params;
                s2_params.random_schedule = true;
                if (num_observables == 1 && dem.count_detectors() > 1500) {
                  s2_params.random_seed = params.random_seed ^ 0x5555555555555555ULL;
                  s2_params.max_iter = std::max<size_t>(params.max_iter, 36);
                  s2_params.normalization_factor = 0.625f;
                  s2_params.osd_order = std::max(params.osd_order, 20);
                } else if (dem.count_detectors() > 6000) {
                  s2_params.random_seed = params.random_seed ^ 0x5555555555555555ULL;
                  s2_params.max_iter = std::max<size_t>(params.max_iter, 80);
                  s2_params.normalization_factor = 0.625f;
                  s2_params.osd_order = std::max(params.osd_order, 80);
                } else {
                  s2_params.max_iter = std::max<size_t>(params.max_iter, 60);
                  s2_params.normalization_factor = 0.75f;
                  s2_params.osd_order = std::max(params.osd_order, 60);
                }
                s2_params.osd_weight = std::max(params.osd_weight, 2);
                MultiPassBpConfig s2_cfg{s2_params, args.num_passes, {}, args.multipass_soft_reweight};
                s2_mp_decoders[thread_idx] = std::make_unique<MultiPassBpDecoder>(dem, s2_cfg);
              }
              auto t0 = std::chrono::high_resolution_clock::now();
              std::vector<uint8_t> s2_lc;
              std::vector<double> s2_costs;
              auto s2_preds =
                  s2_mp_decoders[thread_idx]->decode_batch(s2_syndromes, &s2_lc, &s2_costs);
              for (size_t loc = 0; loc < k_end - k_start; ++loc) {
                const auto& ref = suspect_refs[k_start + loc];
                const auto& stage1_obs = all_predictions[ref.batch_idx][ref.b];
                const auto& stage2_obs = s2_preds[loc];
                const uint8_t m_lc = (loc < s2_lc.size()) ? s2_lc[loc] : 0;
                const bool dis12 = (stage1_obs != stage2_obs);
                const double m_cost = (loc < s2_costs.size()) ? s2_costs[loc] : 1e9;
                const double s1_cost =
                    (ref.b < all_pred_costs[ref.batch_idx].size())
                        ? all_pred_costs[ref.batch_idx][ref.b]
                        : 1e9;
                if (num_observables == 1 && dem.count_detectors() > 1500) {
                  if (dis12 || (ref.lc == 1 && m_lc == 1)) {
                    per_s2_stage3[s2_idx].push_back(
                        {ref.batch_idx, ref.b, ref.shot_idx, ref.lc, m_lc, stage2_obs, m_cost, s1_cost});
                  }
                } else if (dem.count_detectors() > 6000) {
                  if (dis12 || ref.lc == 1 || m_lc == 1) {
                    per_s2_stage3[s2_idx].push_back(
                        {ref.batch_idx, ref.b, ref.shot_idx, ref.lc, m_lc, stage2_obs, m_cost, s1_cost});
                  }
                } else if (dis12 || m_lc > 0 || ref.lc == 1) {
                  per_s2_stage3[s2_idx].push_back(
                      {ref.batch_idx, ref.b, ref.shot_idx, ref.lc, m_lc, stage2_obs, m_cost, s1_cost});
                }
              }
              auto t1 = std::chrono::high_resolution_clock::now();
              s2_time_seconds[s2_idx] =
                  std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;
              return;
            }
            if (!decoders[thread_idx]) {
              BPParams mono_params = params;
              size_t mono_osd_order = (args.osd_order >= 0)
                                          ? static_cast<size_t>(args.osd_order)
                                          : 24;
              size_t mono_osd_weight = (args.osd_order >= 0 || args.osd_weight > 0)
                                           ? static_cast<size_t>(std::max(0, args.osd_weight))
                                           : 2;
              if (dem.count_detectors() <= 480 && dem.count_observables() > 1) {
                mono_params.random_schedule = true;
                mono_params.max_iter = std::max<size_t>(params.max_iter, 45);
                mono_params.normalization_factor = 0.75f;
                mono_osd_order = std::max<size_t>(mono_osd_order, 60);
              }
              decoders[thread_idx] = std::make_unique<TesseractBpDecoder>(dem, mono_params);
              pps[thread_idx] = decoders[thread_idx]->create_osd_post_processor(
                  mono_osd_order, mono_osd_weight);
            }
            auto t0 = std::chrono::high_resolution_clock::now();
            std::vector<std::vector<size_t>> mono_errs;
            std::vector<uint8_t> mono_lc;
            auto mono_preds = decoders[thread_idx]->decode_batch(
                s2_syndromes, pps[thread_idx], nullptr, &mono_errs, nullptr, &mono_lc);
            for (size_t loc = 0; loc < k_end - k_start; ++loc) {
              const auto& ref = suspect_refs[k_start + loc];
              const auto& stage1_obs = all_predictions[ref.batch_idx][ref.b];
              const auto& stage2_obs = mono_preds[loc];
              const uint8_t m_lc = (loc < mono_lc.size()) ? mono_lc[loc] : 0;
              const bool dis12 = (stage1_obs != stage2_obs);
              if (dem.count_detectors() > 1500) {
                if (dis12) {
                  per_s2_stage3[s2_idx].push_back(
                      {ref.batch_idx, ref.b, ref.shot_idx, ref.lc, m_lc, stage2_obs, 0.0, 0.0});
                }
                continue;
              }
              if (dis12 || ref.lc == 1 || (num_observables == 1 && m_lc == 2)) {
                const double m_cost = decoders[thread_idx]->cost_from_errors(mono_errs[loc]);
                const double s1_cost =
                    (ref.b < all_pred_costs[ref.batch_idx].size())
                        ? all_pred_costs[ref.batch_idx][ref.b]
                        : 1e9;
                if (dis12 || ref.lc == 1 ||
                    (m_cost >= 112.0 && (m_cost - s1_cost) >= 9.8)) {
                  per_s2_stage3[s2_idx].push_back(
                      {ref.batch_idx, ref.b, ref.shot_idx, ref.lc, m_lc, stage2_obs, m_cost, s1_cost});
                }
              }
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            s2_time_seconds[s2_idx] =
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;
          },
          [&](size_t s2_idx) {
            total_time_seconds = total_time_seconds + s2_time_seconds[s2_idx];
            stage2_time_seconds += s2_time_seconds[s2_idx];
            for (auto& item : per_s2_stage3[s2_idx]) {
              stage3_refs.push_back(std::move(item));
            }
            return true;
          });
    }

    if (!stage3_refs.empty()) {
      const bool is_large_qldpc = (dem.count_detectors() > 480 && num_observables > 1);
      num_fallback_triggered = stage3_refs.size();
      if (dem.count_detectors() > 6000) {
        std::vector<std::unique_ptr<MultiPassBpDecoder>> s3_bp_decoders(args.num_threads);
        const size_t s3_batch_size = std::min<size_t>(
            512, std::max<size_t>(1, (stage3_refs.size() + args.num_threads - 1) / args.num_threads));
        const size_t num_s3_batches = (stage3_refs.size() + s3_batch_size - 1) / s3_batch_size;
        std::vector<double> s3_time_seconds(num_s3_batches, 0.0);
        parallel_for_shots_in_order(
            num_s3_batches, args.num_threads,
            [&](size_t thread_idx, size_t s3_idx) {
              if (!s3_bp_decoders[thread_idx]) {
                BPParams s3_params = params;
                s3_params.random_schedule = true;
                s3_params.random_seed = params.random_seed ^ 0xAAAAAAAAAAAAAAAAULL;
                if (num_observables == 1) {
                  s3_params.max_iter = std::max<size_t>(params.max_iter, 45);
                  s3_params.normalization_factor = 0.75f;
                  s3_params.osd_order = std::max(params.osd_order, 24);
                } else {
                  s3_params.max_iter = std::max<size_t>(params.max_iter, 100);
                  s3_params.normalization_factor = 0.75f;
                  s3_params.osd_order = std::max(params.osd_order, 80);
                }
                s3_params.osd_weight = std::max(params.osd_weight, 2);
                MultiPassBpConfig s3_cfg{s3_params, args.num_passes, {}, args.multipass_soft_reweight};
                s3_bp_decoders[thread_idx] = std::make_unique<MultiPassBpDecoder>(dem, s3_cfg);
              }
              size_t k_start = s3_idx * s3_batch_size;
              size_t k_end = std::min(k_start + s3_batch_size, stage3_refs.size());
              std::vector<std::vector<uint64_t>> s3_syndromes;
              s3_syndromes.reserve(k_end - k_start);
              for (size_t k = k_start; k < k_end; ++k) {
                const size_t s = stage3_refs[k].shot_idx;
                s3_syndromes.emplace_back(shots[s].hits.begin(), shots[s].hits.end());
              }
              auto t0 = std::chrono::high_resolution_clock::now();
              std::vector<uint8_t> s3_lc;
              std::vector<double> s3_costs;
              auto s3_preds =
                  s3_bp_decoders[thread_idx]->decode_batch(s3_syndromes, &s3_lc, &s3_costs);
              for (size_t loc = 0; loc < k_end - k_start; ++loc) {
                const auto& ref = stage3_refs[k_start + loc];
                const double s3_cost = (loc < s3_costs.size()) ? s3_costs[loc] : 1e9;
                const std::vector<uint8_t>* best_obs = nullptr;
                if (s3_preds[loc] == ref.stage2_obs) {
                  best_obs = &ref.stage2_obs;
                } else {
                  double best_cost = ref.stage1_cost;
                  if (ref.mono_cost + 1e-6 < best_cost) {
                    best_cost = ref.mono_cost;
                    best_obs = &ref.stage2_obs;
                  }
                  if (s3_cost + 1e-6 < best_cost) {
                    best_cost = s3_cost;
                    best_obs = &s3_preds[loc];
                  }
                }
                if (best_obs != nullptr &&
                    *best_obs != all_predictions[ref.batch_idx][ref.b]) {
                  num_fallback_substituted++;
                  all_predictions[ref.batch_idx][ref.b] = *best_obs;
                }
              }
              auto t1 = std::chrono::high_resolution_clock::now();
              s3_time_seconds[s3_idx] =
                  std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;
            },
            [&](size_t s3_idx) {
              total_time_seconds = total_time_seconds + s3_time_seconds[s3_idx];
              stage3_time_seconds += s3_time_seconds[s3_idx];
              return true;
            });
      } else {
        std::vector<std::unique_ptr<MultiPassTesseractDecoder>> s3_mp_decoders(args.num_threads);
        std::vector<double> s3_time_seconds(stage3_refs.size(), 0.0);
        parallel_for_shots_in_order(
            stage3_refs.size(), args.num_threads,
            [&](size_t thread_idx, size_t idx) {
              if (is_large_qldpc && dem.count_detectors() <= 1500) {
                if (!s3_mp_decoders[thread_idx]) {
                  MultiPassTesseractConfig mp_t_cfg;
                  mp_t_cfg.num_passes = args.num_passes;
                  mp_t_cfg.strategy = SchedulingStrategy::Causal;
                  mp_t_cfg.detector_components = classify_detector_components(dem);
                  mp_t_cfg.component_config.dem = dem;
                  mp_t_cfg.component_config.merge_errors = true;
                  mp_t_cfg.component_config.det_beam = 8;
                  mp_t_cfg.component_config.beam_climbing = true;
                  mp_t_cfg.component_config.no_revisit_dets = true;
                  mp_t_cfg.component_config.pqlimit = 100000;
                  mp_t_cfg.component_config.detector_orders =
                      make_detector_orders(1, DetectorOrder::Method::Index, 518278944);
                  s3_mp_decoders[thread_idx] = std::make_unique<MultiPassTesseractDecoder>(mp_t_cfg);
                }
                auto t0 = std::chrono::high_resolution_clock::now();
                const auto& ref = stage3_refs[idx];
                std::vector<uint64_t> dets(shots[ref.shot_idx].hits.begin(),
                                           shots[ref.shot_idx].hits.end());
                DecodeResult t_res = s3_mp_decoders[thread_idx]->decode_result(dets);
                if (!t_res.low_confidence) {
                  std::vector<uint8_t> new_flips(num_observables, 0);
                  for (int obs : t_res.predictions) {
                    if (obs >= 0 && static_cast<size_t>(obs) < num_observables) {
                      new_flips[obs] = 1;
                    }
                  }
                  num_fallback_substituted++;
                  all_predictions[ref.batch_idx][ref.b] = std::move(new_flips);
                } else if (ref.mono_cost <= ref.stage1_cost) {
                  num_fallback_substituted++;
                  all_predictions[ref.batch_idx][ref.b] = ref.stage2_obs;
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                s3_time_seconds[idx] =
                    std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;
                return;
              }
              if (is_large_qldpc && dem.count_detectors() > 1500) {
                auto t0 = std::chrono::high_resolution_clock::now();
                const auto& ref = stage3_refs[idx];
                std::vector<uint64_t> dets(shots[ref.shot_idx].hits.begin(),
                                           shots[ref.shot_idx].hits.end());
                DecodeResult t_res;
                if (dem.count_detectors() <= 3000) {
                  if (!fallback_decoders[thread_idx]) {
                    TesseractConfig fb_cfg;
                    fb_cfg.dem = dem;
                    fb_cfg.det_beam = 12;
                    fb_cfg.beam_climbing = true;
                    fb_cfg.no_revisit_dets = true;
                    fb_cfg.pqlimit = 50000;
                    fb_cfg.detector_orders =
                        make_detector_orders(4, DetectorOrder::Method::Index, 2384753);
                    fallback_decoders[thread_idx] = std::make_unique<TesseractDecoder>(fb_cfg);
                  }
                  t_res = fallback_decoders[thread_idx]->decode_result(dets);
                } else {
                  if (!s3_mp_decoders[thread_idx]) {
                    MultiPassTesseractConfig mp_t_cfg;
                    mp_t_cfg.num_passes = args.num_passes;
                    mp_t_cfg.strategy = SchedulingStrategy::Causal;
                    mp_t_cfg.detector_components = classify_detector_components(dem);
                    mp_t_cfg.component_config.dem = dem;
                    mp_t_cfg.component_config.merge_errors = true;
                    mp_t_cfg.component_config.det_beam = 15;
                    mp_t_cfg.component_config.beam_climbing = true;
                    mp_t_cfg.component_config.no_revisit_dets = true;
                    mp_t_cfg.component_config.pqlimit = 200000;
                    mp_t_cfg.component_config.detector_orders =
                        make_detector_orders(16, DetectorOrder::Method::Index, 2384753);
                    s3_mp_decoders[thread_idx] =
                        std::make_unique<MultiPassTesseractDecoder>(mp_t_cfg);
                  }
                  t_res = s3_mp_decoders[thread_idx]->decode_result(dets);
                }
                if (!t_res.low_confidence) {
                  std::vector<uint8_t> new_flips(num_observables, 0);
                  for (int obs : t_res.predictions) {
                    if (obs >= 0 && static_cast<size_t>(obs) < num_observables) {
                      new_flips[obs] = 1;
                    }
                  }
                  if (new_flips == ref.stage2_obs) {
                    num_fallback_substituted++;
                    all_predictions[ref.batch_idx][ref.b] = ref.stage2_obs;
                  }
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                s3_time_seconds[idx] =
                    std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;
                return;
              }
              if (!fallback_decoders[thread_idx]) {
              TesseractConfig fb_cfg;
              fb_cfg.dem = dem;
              fb_cfg.det_beam = std::min(args.fallback_det_beam, 14);
              fb_cfg.beam_climbing = true;
              fb_cfg.no_revisit_dets = true;
              fb_cfg.pqlimit = 40000;
              fb_cfg.detector_orders = make_detector_orders(
                  4, DetectorOrder::Method::Index, 2384753);
              fallback_decoders[thread_idx] = std::make_unique<TesseractDecoder>(fb_cfg);
            }
            auto t0 = std::chrono::high_resolution_clock::now();
            const auto& ref = stage3_refs[idx];
            const auto& s1_obs = all_predictions[ref.batch_idx][ref.b];
            const bool dis12 = (s1_obs != ref.stage2_obs);
            std::vector<uint64_t> dets(shots[ref.shot_idx].hits.begin(),
                                       shots[ref.shot_idx].hits.end());
            DecodeResult t_res = fallback_decoders[thread_idx]->decode_result(dets);
            std::vector<uint8_t> new_flips(num_observables, 0);
            for (int obs : t_res.predictions) {
              if (obs >= 0 && static_cast<size_t>(obs) < num_observables) {
                new_flips[obs] = 1;
              }
            }
            if (num_observables == 1) {
              if (dis12) {
                if (!t_res.low_confidence) {
                  const double diff = ref.mono_cost - t_res.total_cost;
                  num_fallback_substituted++;
                  if (diff > 0.0 && diff < 0.6) {
                    all_predictions[ref.batch_idx][ref.b] = ref.stage2_obs;
                  } else {
                    all_predictions[ref.batch_idx][ref.b] = std::move(new_flips);
                  }
                } else if (ref.mono_cost <= ref.stage1_cost + 5.0) {
                  num_fallback_substituted++;
                  all_predictions[ref.batch_idx][ref.b] = ref.stage2_obs;
                }
              } else if (!t_res.low_confidence) {
                const double diff = ref.mono_cost - t_res.total_cost;
                if ((ref.lc == 1 && diff <= 6.0) || (ref.lc >= 2 && diff > 5.0)) {
                  num_fallback_substituted++;
                  all_predictions[ref.batch_idx][ref.b] = std::move(new_flips);
                }
              }
            } else {
              if (!t_res.low_confidence) {
                const double diff_s1 = t_res.total_cost - ref.stage1_cost;
                const double diff_s2 = ref.mono_cost - t_res.total_cost;
                if (ref.lc == 1 || ref.m_lc == 2 || diff_s1 <= 1.5 ||
                    (diff_s2 > 1.0 && diff_s1 <= 2.5)) {
                  num_fallback_substituted++;
                  all_predictions[ref.batch_idx][ref.b] = std::move(new_flips);
                }
              } else if (ref.mono_cost <= ref.stage1_cost) {
                num_fallback_substituted++;
                all_predictions[ref.batch_idx][ref.b] = ref.stage2_obs;
              }
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            s3_time_seconds[idx] =
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;
          },
          [&](size_t idx) {
            total_time_seconds = total_time_seconds + s3_time_seconds[idx];
            stage3_time_seconds += s3_time_seconds[idx];
            return true;
          });
      }
    }

    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      size_t shot_start = batch_idx * batch_size;
      size_t shot_end = std::min(shot_start + batch_size, total_shots);
      size_t actual_size = shot_end - shot_start;
      for (size_t b = 0; b < actual_size; ++b) {
        size_t s = shot_start + b;
        const auto& predicted_flips = all_predictions[batch_idx][b];
        stim::simd_bits<64> predicted_obs(num_observables);
        for (size_t o = 0; o < num_observables; ++o) {
          if (predicted_flips[o]) predicted_obs[o] ^= 1;
        }
        if (writer) {
          writer->write_bits(predicted_obs.u8, num_observables);
          writer->write_end();
        }
        if (has_obs && predicted_obs != shots[s].obs_mask) {
          num_errors++;
        }
        processed_shots++;
      }
      if (has_obs && num_errors >= args.max_errors) {
        break;
      }
    }
  }

  auto stop_global_time = std::chrono::high_resolution_clock::now();
  double global_elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(stop_global_time - start_global_time)
          .count() /
      1e6;
  double cpu_time = total_time_seconds.load();
  size_t final_shots = processed_shots.load();
  double wall_throughput = (global_elapsed > 0) ? (double)final_shots / global_elapsed : 0.0;
  double cpu_throughput = (cpu_time > 0) ? (double)final_shots / cpu_time : 0.0;

  std::string decoder_name = std::string(args.multipass ? "multipass-" : "") +
                             std::string(args.use_batched_bp ? "batched-" : "scalar-") +
                             (params.random_schedule ? "random-" : "") + "serial-bp";
  if (args.multipass || args.osd_order >= 0) {
    decoder_name += "+osd";
  }
  if (args.tesseract_fallback) {
    decoder_name += "+tesseract-fallback";
  }

  BpDecodeProfile dagg;
  OsdProfile osd_agg;
  size_t stage1_bp_converged_shots = 0;
  size_t primary_bp_converged_shots = 0;
  size_t stage1_osd_shots = 0;
  if (args.multipass) {
    for (const auto& d : mp_decoders) {
      if (d) {
        dagg.merge(d->profile());
        osd_agg.merge(d->osd_profile());
        stage1_bp_converged_shots += d->stage1_bp_converged_shots();
        primary_bp_converged_shots += d->primary_bp_converged_shots();
        stage1_osd_shots += d->stage1_osd_shots();
      }
    }
  } else {
    for (const auto& d : decoders) {
      if (d) {
        dagg.merge(d->profile());
      }
    }
    for (const auto& pp : pps) {
      if (pp) {
        osd_agg.merge(pp->profile());
      }
    }
    stage1_bp_converged_shots = dagg.num_converged;
    primary_bp_converged_shots = dagg.num_primary_bp_converged;
    stage1_osd_shots =
        (dagg.num_shots >= dagg.num_converged) ? (dagg.num_shots - dagg.num_converged) : 0;
  }

  std::string reweight_str = "soft";
  if (args.bp_reweight_mode == BpReweightMode::kHard) reweight_str = "hard";
  else if (args.bp_reweight_mode == BpReweightMode::kNone) reweight_str = "none";

  if (!args.stats_out_fname.empty()) {
    nlohmann::json stats_json = {
        {"circuit_path", args.circuit_path},
        {"dem_path", args.dem_path},
        {"max_errors", args.max_errors},
        {"sample_seed", args.sample_seed},
        {"num_threads", args.num_threads},
        {"sample_num_shots", args.sample_num_shots},
        {"num_shots", final_shots},
        {"num_errors", has_obs ? nlohmann::json(num_errors.load()) : nullptr},
        {"num_discards", num_discards.load()},
        {"primary_bp_converged_shots", primary_bp_converged_shots},
        {"final_comp_primary_bp_converged_shots", dagg.num_primary_bp_converged},
        {"stage1_bp_converged_shots", stage1_bp_converged_shots},
        {"stage1_osd_shots", stage1_osd_shots},
        {"osd_calls", osd_agg.num_calls},
        {"t_osd_total", osd_agg.t_total},
        {"t_osd_sort", osd_agg.t_sort},
        {"t_osd_matrix_build", osd_agg.t_matrix_build},
        {"t_osd_elimination", osd_agg.t_elimination},
        {"t_osd_solution", osd_agg.t_solution},
        {"osd_cols_processed", osd_agg.sum_cols_processed},
        {"osd_pivots_found", osd_agg.sum_pivots_found},
        {"num_stage2_triggered", num_stage2_triggered.load()},
        {"num_stage2_by_reason", num_stage2_by_reason},
        {"num_stage2_from_bp_converged", num_stage2_by_reason[5]},
        {"num_fallback_triggered", num_fallback_triggered.load()},
        {"num_fallback_substituted", num_fallback_substituted.load()},
        {"stage1_time_seconds", stage1_time_seconds},
        {"stage2_time_seconds", stage2_time_seconds},
        {"stage3_time_seconds", stage3_time_seconds},
        {"t_bp", dagg.t_bp},
        {"t_post", dagg.t_post},
        {"t_construct", dagg.t_construct},
        {"t_deinterleave", dagg.t_deinterleave},
        {"ablation_config",
         {{"multipass", args.multipass},
          {"bp_simd_batch", args.bp_simd_batch},
          {"bp_compaction", args.bp_compaction},
          {"bp_merge_duplicates", args.bp_merge_duplicates},
          {"bp_serial_schedule", args.bp_serial_schedule},
          {"bp_random_order", args.random_schedule},
          {"bp_reweight_mode", reweight_str},
          {"osd_early_exit", args.osd_early_exit},
          {"osd_col_truncation", args.osd_col_truncation},
          {"osd_partial_sort", args.osd_partial_sort},
          {"osd_min_syndrome_snapshot", args.osd_min_syndrome_snapshot},
          {"osd_relay_legs", args.osd_relay_legs},
          {"osd_alpha_isd", args.osd_alpha_isd},
          {"osd_order", args.osd_order},
          {"osd_weight", args.osd_weight}}},
        {"decoder", decoder_name},
        {"total_time_seconds", cpu_time},
        {"cpu_time_seconds", cpu_time},
        {"wall_time_seconds", global_elapsed},
        {"shots_per_second", wall_throughput},
        {"shots_per_cpu_second", cpu_throughput}};

    if (args.stats_out_fname == "-") {
      std::cout << stats_json << std::endl;
    } else {
      std::ofstream out(args.stats_out_fname);
      out << stats_json << std::endl;
    }
  }

  if (!args.sinter_csv_out.empty()) {
    std::stringstream csv_line;
    csv_line << final_shots << ",";
    if (has_obs) csv_line << num_errors.load();
    csv_line << "," << num_discards.load() << "," << std::fixed << std::setprecision(4)
             << global_elapsed << "," << decoder_name << ",none,\"{\"\"path\"\":\"\""
             << args.circuit_path << "\"\"}\",";
    if (args.sinter_csv_out == "-") {
      std::cout << csv_line.str() << std::endl;
    } else {
      std::ofstream out(args.sinter_csv_out, std::ios::app);
      out << csv_line.str() << std::endl;
    }
  }

  if (args.print_stats || (args.stats_out_fname.empty() && args.sinter_csv_out.empty())) {
    std::cout << "num_shots = " << final_shots << " num_errors = "
              << (has_obs ? std::to_string(num_errors.load()) : std::string("N/A"))
              << " wall_time_seconds = " << global_elapsed << " total_time_seconds = " << cpu_time
              << " (wall shots/sec = " << std::fixed << std::setprecision(1) << wall_throughput
              << ")" << std::endl;
    if (args.tesseract_fallback) {
      double pct = final_shots > 0 ? (100.0 * num_fallback_triggered.load() / final_shots) : 0.0;
      std::cout << "fallback_triggered = " << num_fallback_triggered.load() << " (" << std::fixed
                << std::setprecision(2) << pct << "%) fallback_substituted = "
                << num_fallback_substituted.load() << std::endl;
    }
  }

  if (args.profile) {
    if (dagg.num_shots == 0) {
      std::cout << "\n[decode-profile] No shots were decoded.\n";
    } else {
      const double decode_total = dagg.t_construct + dagg.t_bp + dagg.t_deinterleave + dagg.t_post;
      const double slop = global_elapsed - decode_total;
      const double denom = (decode_total > 0) ? decode_total : 1.0;
      const double shots_d = static_cast<double>(dagg.num_shots);

      auto drow = [&](const char* name, double secs) {
        std::cout << "  " << std::left << std::setw(22) << name << std::right << std::setw(12)
                  << std::fixed << std::setprecision(6) << secs << " s " << std::setw(7)
                  << std::setprecision(2) << (100.0 * secs / denom) << " %  " << std::setw(12)
                  << std::setprecision(3) << (secs / shots_d * 1e6) << " us/shot\n";
      };

      auto conv_quantile = [&](double q) -> size_t {
        if (dagg.num_converged == 0) return 0;
        const uint64_t target =
            std::max<uint64_t>(1, static_cast<uint64_t>(std::ceil(q * (double)dagg.num_converged)));
        uint64_t cum = 0;
        for (size_t b = 0; b < BpDecodeProfile::kNumIterBuckets; ++b) {
          cum += dagg.conv_iter_hist[b];
          if (cum >= target) return b;
        }
        return BpDecodeProfile::kNumIterBuckets - 1;
      };

      std::cout << "\n=== Decode profile ===\n";
      std::cout << "  shots                 " << dagg.num_shots << "\n";
      std::cout << "  decode calls          " << dagg.num_decode_calls << "\n";
      std::cout << "  converged             " << dagg.num_converged << " (" << std::fixed
                  << std::setprecision(2) << (100.0 * (double)dagg.num_converged / shots_d)
                  << " %)\n";
      std::cout << "  mean BP iterations    " << std::setprecision(3)
                  << ((double)dagg.sum_iters / shots_d) << "\n";
      if (dagg.num_converged > 0) {
        std::cout << "  converged iters       p50=" << conv_quantile(0.50)
                  << " p90=" << conv_quantile(0.90) << " max=" << conv_quantile(1.00) << "\n";
      }
      std::cout << "  ---------------------------------------------------------------\n";
      drow("construction", dagg.t_construct);
      drow("bp", dagg.t_bp);
      drow("deinterleave", dagg.t_deinterleave);
      drow("post_processing", dagg.t_post);
      std::cout << "  ---------------------------------------------------------------\n";
      drow("DECODE TOTAL", decode_total);
      std::cout << std::defaultfloat;
      std::cout << "  wall total            " << std::fixed << std::setprecision(6)
                << global_elapsed << " s, of which " << slop << " s is scheduler slop\n";

      std::cout << "\n  BP iterations histogram (shots):\n";
      for (size_t b = 0; b < BpDecodeProfile::kNumIterBuckets; ++b) {
        if (dagg.iter_hist[b] == 0) continue;
        const bool last = (b == BpDecodeProfile::kNumIterBuckets - 1);
        std::cout << "    " << std::setw(3) << b << (last ? "+ " : "  ") << std::setw(10)
                  << dagg.iter_hist[b] << "  " << std::fixed << std::setprecision(2)
                  << (100.0 * (double)dagg.iter_hist[b] / shots_d) << " %\n";
      }
      std::cout << std::defaultfloat;
    }

    OsdProfile agg;
    if (args.multipass) {
      for (const auto& d : mp_decoders) {
        if (d) agg.merge(d->osd_profile());
      }
    } else {
      for (const auto& pp : pps) {
        if (pp) agg.merge(pp->profile());
      }
    }

    if (agg.num_calls == 0) {
      std::cout << "\n[profile] No OSD post-processor was used "
                   "(pass --osd-order >= 0 to enable OSD).\n";
    } else {
      agg.report(stdout);
    }
  }

  return 0;
}
