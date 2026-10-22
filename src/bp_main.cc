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
#include "bp/multi_pass_bp_decoder.h"
#include "bp/osd_post_processor.h"
#include "bp/tesseract_bp_decoder.h"
#include "bp/tesseract_post_processor.h"
#include "common.h"
#include "stim.h"
#include "utils.h"

using namespace bp;
using namespace tesseract_decoder;

struct Args {
  std::string circuit_path;
  std::string dem_path;

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
  bool random_schedule = false;
  double osd_truncation_factor = 0.0;
  bool profile = false;
  bool multipass = false;
  size_t num_passes = 2;
  std::string multipass_strategy = "causal";
  bool multipass_soft_reweight = true;
  std::string post_processor = "osd";
  int tesseract_det_beam = 5;
  size_t tesseract_pqlimit = 2000;

  std::string stats_out_fname = "";
  std::string sinter_csv_out = "";
  std::string dump_trace_fname = "";
  std::string trace_dets_csv = "";
  bool print_stats = false;

  bool has_observables() {
    return append_observables || !obs_in_fname.empty() || (sample_num_shots > 0);
  }

  void validate() {
    if (circuit_path.empty() and dem_path.empty()) {
      throw std::invalid_argument("Must provide at least one of --circuit or --dem");
    }
    int num_data_sources =
        int(sample_num_shots > 0) + int(!in_fname.empty()) + int(!trace_dets_csv.empty());
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
    params.random_schedule = random_schedule || schedule == "random-serial" ||
                             schedule == "stochastic-serial" || schedule == "random";
    params.random_seed = sample_seed;
    params.osd_truncation_factor = (float)osd_truncation_factor;
    params.osd_order = osd_order;
    params.osd_weight = osd_weight;
    params.post_processor = post_processor;
    params.tesseract_det_beam = tesseract_det_beam;
    params.tesseract_pqlimit = tesseract_pqlimit;
    params.use_soft_reweighting = multipass_soft_reweight;

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
  program.add_argument("--osd-truncation-factor")
      .help(
          "Multiplier defining the maximum number of free columns for the dense OSD Gaussian "
          "elimination matrix (e.g. 1.2 bounds to 1.2x the number of defect detectors). Use 0.0 to "
          "disable truncation.")
      .default_value(1.2)
      .store_into(args.osd_truncation_factor);
  program.add_argument("--disable-osd-early-exit")
      .help("Disable residual syndrome early stopping in OSD elimination")
      .flag();
  program.add_argument("--profile")
      .help(
          "Print timing breakdowns: first the decode path (construction, BP, de-interleave, "
          "post-processing), then the per-phase OSD breakdown (sort, matrix build, Gaussian "
          "elimination, solution extraction). The decode split is needed to interpret the OSD "
          "percentages, since OSD is often a minority of end-to-end runtime")
      .flag()
      .store_into(args.profile);
  program.add_argument("--batched")
      .help("Use AVX-512 batching across shots")
      .flag()
      .store_into(args.use_batched_bp);
  program.add_argument("--random-schedule")
      .help("Use randomized check node permutation in serial BP to prevent trapping sets")
      .flag()
      .store_into(args.random_schedule);
  program.add_argument("--multipass")
      .help("Enable Multi-Pass Shattered BP decoding")
      .flag()
      .store_into(args.multipass);
  program.add_argument("--num-passes")
      .help("Number of passes in Multi-Pass BP (default: 2)")
      .default_value(size_t(2))
      .store_into(args.num_passes);
  program.add_argument("--multipass-strategy")
      .help("Multi-pass scheduling strategy: 'causal' or 'static' (default: causal)")
      .default_value(std::string("causal"))
      .store_into(args.multipass_strategy);
  program.add_argument("--multipass-soft-reweight")
      .help("Use soft LLR reweighting across passes (enabled by default)")
      .flag();
  program.add_argument("--disable-multipass-soft-reweight")
      .help("Disable soft LLR reweighting across passes")
      .flag();
  program.add_argument("--post-processor")
      .help("Post-processor algorithm: 'osd', 'tesseract', or 'hard' (default: osd)")
      .default_value(std::string("osd"))
      .store_into(args.post_processor);
  program.add_argument("--tesseract-beam")
      .help("Beam width for TesseractPostProcessor (default: 5)")
      .default_value(5)
      .store_into(args.tesseract_det_beam);
  program.add_argument("--tesseract-pqlimit")
      .help("PQ limit for residual Tesseract A* search (default: 2000)")
      .default_value(size_t(2000))
      .store_into(args.tesseract_pqlimit);
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
  program.add_argument("--dump-trace")
      .help("Optional JSON file path to dump a full step-by-step BP + post-processor trace for the first shot (OFF by default)")
      .store_into(args.dump_trace_fname);
  program.add_argument("--trace-dets")
      .help("Comma-separated detector indices for a single trace shot (e.g. '0,3,5')")
      .store_into(args.trace_dets_csv);
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
  if (program.is_used("--disable-multipass-soft-reweight")) {
    args.multipass_soft_reweight = false;
  }
  args.extract(params, dem, shots, writer);
  if (program.is_used("--disable-osd-early-exit")) {
    params.osd_early_exit = false;
  }

  if (!args.trace_dets_csv.empty()) {
    stim::SparseShot shot;
    shot.clear();
    size_t start = 0;
    while (start < args.trace_dets_csv.size()) {
      size_t comma = args.trace_dets_csv.find(',', start);
      std::string tok = (comma == std::string::npos)
                            ? args.trace_dets_csv.substr(start)
                            : args.trace_dets_csv.substr(start, comma - start);
      if (!tok.empty()) {
        shot.hits.push_back(static_cast<uint64_t>(std::stoull(tok)));
      }
      if (comma == std::string::npos) break;
      start = comma + 1;
    }
    shots.push_back(shot);
  }

  if (!args.dump_trace_fname.empty() && !shots.empty()) {
    auto topo_to_json = [](const TannerGraphTopology& t) -> nlohmann::json {
      nlohmann::json j;
      j["num_vars"] = t.num_vars;
      j["num_checks"] = t.num_checks;
      j["num_edges"] = t.num_edges;
      j["num_observables"] = t.num_observables;
      j["priors"] = t.priors;
      j["obs_masks"] = t.obs_masks;
      j["hyperedge_observables"] = t.hyperedge_observables;
      j["edges"] = t.edges;
      return j;
    };
    auto bp_trace_to_json = [&](const BpDecodeTrace& tr) -> nlohmann::json {
      nlohmann::json j;
      j["topology"] = topo_to_json(tr.topology);
      j["detection_events"] = tr.detection_events;
      j["bp_converged"] = tr.bp_converged;
      j["num_iters"] = tr.num_iters;
      j["post_processor_type"] = tr.post_processor_type;
      j["predicted_errors"] = tr.predicted_errors;
      j["predicted_observables"] = tr.predicted_observables;
      nlohmann::json iters = nlohmann::json::array();
      for (const auto& it : tr.iterations) {
        iters.push_back({
            {"iter", it.iter},
            {"var_to_check_messages", it.var_to_check_messages},
            {"check_to_var_messages", it.check_to_var_messages},
            {"posteriors", it.posteriors},
            {"hard_decisions", it.hard_decisions},
            {"check_satisfied", it.check_satisfied},
            {"residual_syndrome", it.residual_syndrome},
            {"converged", it.converged},
        });
      }
      j["iterations"] = std::move(iters);
      j["osd_trace"] = {
          {"triggered", tr.osd_trace.triggered},
          {"residual_syndrome", tr.osd_trace.residual_syndrome},
          {"col_to_var", tr.osd_trace.col_to_var},
          {"pivot_vars", tr.osd_trace.pivot_vars},
          {"free_vars", tr.osd_trace.free_vars},
          {"best_error", tr.osd_trace.best_error},
      };
      j["tesseract_trace"] = {
          {"triggered", tr.tesseract_trace.triggered},
          {"peeled_vars", tr.tesseract_trace.peeled_vars},
          {"active_vars", tr.tesseract_trace.active_vars},
          {"unsolved_dets", tr.tesseract_trace.unsolved_dets},
          {"comp_dets", tr.tesseract_trace.comp_dets},
          {"comp_vars", tr.tesseract_trace.comp_vars},
          {"residual_llr_costs", tr.tesseract_trace.residual_llr_costs},
          {"astar_flipped_vars", tr.tesseract_trace.astar_flipped_vars},
          {"final_flipped_vars", tr.tesseract_trace.final_flipped_vars},
          {"used_osd_fallback", tr.tesseract_trace.used_osd_fallback},
      };
      return j;
    };

    std::vector<uint64_t> dets(shots[0].hits.begin(), shots[0].hits.end());
    nlohmann::json out_j;
    if (args.multipass) {
      tesseract_decoder::SchedulingStrategy mp_strat =
          (args.multipass_strategy == "static")
              ? tesseract_decoder::SchedulingStrategy::Static
              : tesseract_decoder::SchedulingStrategy::Causal;
      MultiPassBpConfig mp_cfg{params, args.num_passes, {}, mp_strat, args.multipass_soft_reweight};
      MultiPassBpDecoder mp_dec(dem, mp_cfg);
      MultiPassBpDecodeTrace mtr = mp_dec.decode_trace(dets);
      out_j["multipass"] = true;
      out_j["component_topologies"] = {topo_to_json(mtr.component_topologies[0]),
                                       topo_to_json(mtr.component_topologies[1])};
      out_j["detector_components"] = mtr.detector_components;
      out_j["pass_schedule"] = mtr.pass_schedule;
      out_j["predicted_observables"] = mtr.predicted_observables;
      nlohmann::json cpasses = nlohmann::json::array();
      for (const auto& cp : mtr.component_passes) {
        cpasses.push_back({
            {"pass", cp.pass},
            {"comp_idx", cp.comp_idx},
            {"trace", bp_trace_to_json(cp.trace)},
        });
      }
      out_j["component_passes"] = std::move(cpasses);
      nlohmann::json reweights = nlohmann::json::array();
      for (const auto& ev : mtr.reweight_events) {
        reweights.push_back({
            {"pass", ev.pass},
            {"source_comp_idx", ev.source_comp_idx},
            {"source_var_idx", ev.source_var_idx},
            {"target_comp_idx", ev.target_comp_idx},
            {"target_var_idx", ev.target_var_idx},
            {"old_prior", ev.old_prior},
            {"new_prior", ev.new_prior},
        });
      }
      out_j["reweight_events"] = std::move(reweights);
    } else {
      TesseractBpDecoder dec(dem, params);
      BpDecodeTrace tr = dec.decode_trace(dets);
      out_j = bp_trace_to_json(tr);
      out_j["multipass"] = false;
    }

    if (args.dump_trace_fname == "-") {
      std::cout << out_j.dump() << std::endl;
    } else {
      std::ofstream ofs(args.dump_trace_fname);
      ofs << out_j.dump();
    }
    if (!args.trace_dets_csv.empty()) {
      return EXIT_SUCCESS;
    }
  }

  std::vector<std::unique_ptr<TesseractBpDecoder>> decoders(args.num_threads);
  std::vector<std::unique_ptr<MultiPassBpDecoder>> mp_decoders(args.num_threads);
  std::vector<std::shared_ptr<PostProcessor>> pps(args.num_threads);

  tesseract_decoder::SchedulingStrategy mp_strat =
      (args.multipass_strategy == "static")
          ? tesseract_decoder::SchedulingStrategy::Static
          : tesseract_decoder::SchedulingStrategy::Causal;
  MultiPassBpConfig mp_config{params, args.num_passes, {}, mp_strat, args.multipass_soft_reweight};

  size_t total_shots = shots.size();
  size_t num_observables = dem.count_observables();
  bool has_obs = args.has_observables();
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
          if (args.multipass) {
            if (!mp_decoders[thread_idx]) {
              mp_decoders[thread_idx] = std::make_unique<MultiPassBpDecoder>(dem, mp_config);
            }
          } else {
            if (!decoders[thread_idx]) {
              decoders[thread_idx] = std::make_unique<TesseractBpDecoder>(dem, params);
              pps[thread_idx] = decoders[thread_idx]->create_post_processor();
            }
          }

          size_t shot_start = batch_idx * 64;
          size_t shot_end = std::min(shot_start + 64, total_shots);
          std::vector<std::vector<uint64_t>> batch_syndromes;
          for (size_t s = shot_start; s < shot_end; ++s) {
            std::vector<uint64_t> dets(shots[s].hits.begin(), shots[s].hits.end());
            batch_syndromes.push_back(dets);
          }

          auto start_batch = std::chrono::high_resolution_clock::now();
          if (args.multipass) {
            all_predictions[batch_idx] =
                mp_decoders[thread_idx]->decode_batch(batch_syndromes);
          } else {
            all_predictions[batch_idx] =
                decoders[thread_idx]->decode_batch(batch_syndromes, pps[thread_idx]);
          }
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
          total_time_seconds = total_time_seconds + batch_time_seconds[batch_idx];
          if (args.print_stats && processed_shots % 1024 == 0) {
            std::cout << "Processed " << processed_shots;
            if (has_obs) {
              std::cout << " shots, errors: " << num_errors;
            } else {
              std::cout << " shots, errors: N/A";
            }
            std::cout << std::endl;
          }
          return !has_obs || num_errors < args.max_errors;
        });
  } else {
    std::vector<std::vector<uint8_t>> all_predictions(total_shots);
    std::vector<double> shot_time_seconds(total_shots);

    parallel_for_shots_in_order(
        total_shots, args.num_threads,
        [&](size_t thread_idx, size_t shot_idx) {
          if (args.multipass) {
            if (!mp_decoders[thread_idx]) {
              mp_decoders[thread_idx] = std::make_unique<MultiPassBpDecoder>(dem, mp_config);
            }
          } else {
            if (!decoders[thread_idx]) {
              decoders[thread_idx] = std::make_unique<TesseractBpDecoder>(dem, params);
              pps[thread_idx] = decoders[thread_idx]->create_post_processor();
            }
          }
          std::vector<uint64_t> dets(shots[shot_idx].hits.begin(), shots[shot_idx].hits.end());
          auto start_shot = std::chrono::high_resolution_clock::now();
          if (args.multipass) {
            all_predictions[shot_idx] = mp_decoders[thread_idx]->decode(dets);
          } else {
            all_predictions[shot_idx] = decoders[thread_idx]->decode(dets, pps[thread_idx]);
          }
          auto stop_shot = std::chrono::high_resolution_clock::now();
          shot_time_seconds[shot_idx] =
              std::chrono::duration_cast<std::chrono::microseconds>(stop_shot - start_shot)
                  .count() /
              1e6;
        },
        [&](size_t shot_idx) {
          std::vector<uint8_t> predicted_flips = all_predictions[shot_idx];
          stim::simd_bits<64> predicted_obs(num_observables);
          for (size_t o = 0; o < num_observables; ++o) {
            if (predicted_flips[o]) predicted_obs[o] ^= 1;
          }
          if (writer) {
            writer->write_bits(predicted_obs.u8, num_observables);
            writer->write_end();
          }
          if (has_obs && predicted_obs != shots[shot_idx].obs_mask) {
            num_errors++;
          }
          processed_shots++;
          total_time_seconds = total_time_seconds + shot_time_seconds[shot_idx];
          if (args.print_stats && processed_shots % 100 == 0) {
            std::cout << "Processed " << processed_shots;
            if (has_obs) {
              std::cout << " shots, errors: " << num_errors;
            } else {
              std::cout << " shots, errors: N/A";
            }
            std::cout << std::endl;
          }
          return !has_obs || num_errors < args.max_errors;
        });
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
                             (params.random_schedule ? "random-" : "") + args.schedule + "-bp";
  if (params.post_processor == "tesseract") {
    decoder_name += "+tesseract";
  } else if (args.osd_order >= 0) {
    decoder_name += "+osd";
  }

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
    if (has_obs) {
      csv_line << num_errors.load();
    }
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
    std::cout << "num_shots = " << final_shots;
    if (has_obs) {
      std::cout << " num_errors = " << num_errors.load();
    } else {
      std::cout << " num_errors = N/A";
    }
    std::cout << " wall_time_seconds = " << global_elapsed << " total_time_seconds = " << cpu_time
              << " (wall shots/sec = " << std::fixed << std::setprecision(1) << wall_throughput
              << ")" << std::endl;
  }

  if (args.profile) {
    BpDecodeProfile dagg;
    if (args.multipass) {
      for (const auto& d : mp_decoders) {
        if (d) dagg.merge(d->profile());
      }
    } else {
      for (const auto& d : decoders) {
        if (d) dagg.merge(d->profile());
      }
    }

    if (dagg.num_shots == 0) {
      std::cout << "\n[decode-profile] No shots were decoded.\n";
    } else {
      // Construction happens once per thread, so with T threads t_construct is
      // summed over T independent builds of the same graph.
      const double decode_total = dagg.t_construct + dagg.t_bp + dagg.t_deinterleave + dagg.t_post;
      // Percentages are taken against accounted decode time, not wall time.
      //
      // Wall time is not a usable denominator here: parallel_for_shots_in_order
      // polls for the next in-order shot with a 100 ms sleep, so wall time is
      // quantized up to the next multiple of 100 ms. The wall-minus-decode
      // residual is dominated by that sleep rather than by real work (actual
      // out-of-decoder work -- syndrome vector building and observable
      // comparison -- measures at only a few ms). Note also that sampling and
      // DEM construction happen in args.extract(), before the wall clock even
      // starts, so they are not in wall time at all.
      const double slop = global_elapsed - decode_total;
      const double denom = (decode_total > 0) ? decode_total : 1.0;
      const double shots = (double)dagg.num_shots;

      auto drow = [&](const char* name, double secs) {
        std::cout << "  " << std::left << std::setw(22) << name << std::right << std::setw(12)
                  << std::fixed << std::setprecision(6) << secs << " s " << std::setw(7)
                  << std::setprecision(2) << (100.0 * secs / denom) << " %  " << std::setw(12)
                  << std::setprecision(3) << (secs / shots * 1e6) << " us/shot\n";
      };

      auto conv_quantile = [&](double q) -> size_t {
        if (dagg.num_converged == 0) return 0;
        const uint64_t target =
            std::max<uint64_t>(1, (uint64_t)std::ceil(q * (double)dagg.num_converged));
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
                << std::setprecision(2) << (100.0 * (double)dagg.num_converged / shots) << " %)\n";
      std::cout << "  mean BP iterations    " << std::setprecision(3)
                << ((double)dagg.sum_iters / shots) << "\n";
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
      std::cout << "  (percentages are of DECODE TOTAL. With T>1 these rows sum CPU time\n"
                << "   across threads, so they exceed wall time.)\n";
      std::cout << "  wall total            " << std::fixed << std::setprecision(6)
                << global_elapsed << " s, of which " << slop << " s is scheduler slop\n"
                << "  (parallel_for_shots_in_order polls with a 100 ms sleep, so wall time\n"
                << "   rounds up to the next 100 ms. Do not read throughput off short runs.)\n";

      std::cout << "\n  BP iterations histogram (shots):\n";
      for (size_t b = 0; b < BpDecodeProfile::kNumIterBuckets; ++b) {
        if (dagg.iter_hist[b] == 0) continue;
        const bool last = (b == BpDecodeProfile::kNumIterBuckets - 1);
        std::cout << "    " << std::setw(3) << b << (last ? "+ " : "  ") << std::setw(10)
                  << dagg.iter_hist[b] << "  " << std::fixed << std::setprecision(2)
                  << (100.0 * (double)dagg.iter_hist[b] / shots) << " %\n";
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
        if (!pp) continue;
        if (auto* osd_pp = dynamic_cast<const OsdPostProcessor*>(pp.get())) {
          agg.merge(osd_pp->profile());
        } else if (auto* tess_pp = dynamic_cast<const TesseractPostProcessor*>(pp.get())) {
          agg.merge(tess_pp->profile());
        }
      }
    }

    if (agg.num_calls == 0) {
      std::cout << "\n[profile] No OSD post-processor was used "
                   "(pass --osd-order >= 0 to enable OSD).\n";
    } else {
      // Phase timings only accumulate on calls that actually ran OSD.
      const double n = (double)std::max<uint64_t>(agg.num_osd_invocations, 1);
      const double n_conv = (double)std::max<uint64_t>(agg.num_converged_early_exits, 1);
      const double measured = agg.t_hard_decision + agg.t_residual + agg.t_sort +
                              agg.t_matrix_build + agg.t_elimination + agg.t_solution;
      const double other = std::max(0.0, agg.t_total - measured);
      const double denom = (agg.t_total > 0) ? agg.t_total : 1.0;

      auto row = [&](const char* name, double secs) {
        std::cout << "  " << std::left << std::setw(22) << name << std::right << std::setw(12)
                  << std::fixed << std::setprecision(6) << secs << " s " << std::setw(7)
                  << std::setprecision(2) << (100.0 * secs / denom) << " %  " << std::setw(12)
                  << std::setprecision(3) << (secs / n * 1e6) << " us/call\n";
      };

      std::cout << "\n=== OSD profile ===\n";
      std::cout << "  calls                 " << agg.num_calls << "\n";
      std::cout << "  converged early exits " << agg.num_converged_early_exits << " (" << std::fixed
                << std::setprecision(6) << agg.t_converged_passthrough << " s, "
                << std::setprecision(3) << (agg.t_converged_passthrough / n_conv * 1e6)
                << " us/exit)\n"
                << std::defaultfloat;
      std::cout << "  OSD invocations       " << agg.num_osd_invocations << "\n";
      if (agg.num_osd_invocations > 0) {
        std::cout << "  detectors (M)         " << (agg.sum_num_detectors / agg.num_osd_invocations)
                  << "\n";
        std::cout << "  errors (N)            " << (agg.sum_num_errors / agg.num_osd_invocations)
                  << "\n";
        std::cout << "  cols after truncation " << (agg.sum_max_osd_cols / agg.num_osd_invocations)
                  << "\n";
        std::cout << "  cols processed        "
                  << (agg.sum_cols_processed / agg.num_osd_invocations) << "\n";
        std::cout << "  pivots found          " << (agg.sum_pivots_found / agg.num_osd_invocations)
                  << "\n";
      }
      std::cout << "  ---------------------------------------------------------------\n";
      row("hard_decision", agg.t_hard_decision);
      row("residual_syndrome", agg.t_residual);
      row("sort", agg.t_sort);
      row("matrix_build", agg.t_matrix_build);
      row("elimination", agg.t_elimination);
      row("solution", agg.t_solution);
      row("other", other);
      std::cout << "  ---------------------------------------------------------------\n";
      row("TOTAL", agg.t_total);
      std::cout << std::defaultfloat;
    }
  }

  return 0;
}
