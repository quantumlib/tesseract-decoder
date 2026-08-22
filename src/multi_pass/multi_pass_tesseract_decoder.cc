#include "multi_pass_tesseract_decoder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <set>
#include <sstream>

#include "../common.h"
#include "dem_decomposition.h"

namespace tesseract {

std::string MultiPassExecutionPlan::str() const {
  std::stringstream ss;
  ss << "Multi-pass execution plan\n"
     << "strategy: " << (strategy == SchedulingStrategy::Static ? "static" : "causal") << '\n'
     << "passes: " << num_passes << '\n'
     << "monolithic input DEM: detectors=" << monolithic_statistics.detector_count
     << ", error_mechanisms=" << monolithic_statistics.error_mechanism_count
     << ", average_detector_degree=" << monolithic_statistics.average_detector_degree << '\n'
     << "components: " << components.size() << '\n';
  for (const auto& component : components) {
    ss << "  component " << component.id << ": label=" << component.classifier_label
       << ", active_detectors=" << component.active_detector_count
       << ", decoder_detectors=" << component.decoder_detector_count
       << ", observable=" << (component.affects_observable ? "yes" : "no")
       << ", error_mechanisms=" << component.error_mechanism_count
       << ", average_active_detector_degree=" << component.average_active_detector_degree << '\n';
  }
  ss << "dependencies:\n";
  if (dependencies.empty()) ss << "  none\n";
  for (const auto& dependency : dependencies) {
    ss << "  component " << dependency.source_component << " -> component "
       << dependency.target_component << ": " << dependency.rule_count << " rules\n";
  }
  ss << "schedule:\n";
  for (size_t pass = 0; pass < pass_schedule.size(); ++pass) {
    ss << "  pass " << pass + 1 << ": [";
    for (size_t i = 0; i < pass_schedule[pass].size(); ++i) {
      if (i) ss << ", ";
      ss << pass_schedule[pass][i];
    }
    ss << "]\n";
  }
  return ss.str();
}

namespace {

struct DetectorMetadata {
  std::map<uint64_t, std::vector<double>> coordinates;
  std::map<uint64_t, std::string> tags;
};

MultiPassExecutionPlan::DemStatistics dem_statistics(size_t detector_count,
                                                     const std::vector<common::Error>& errors) {
  size_t detector_incidences = 0;
  for (const auto& error : errors) {
    detector_incidences += error.symptom.detectors.size();
  }
  double average_detector_degree =
      detector_count == 0 ? 0.0 : (double)detector_incidences / detector_count;
  return {detector_count, errors.size(), average_detector_degree};
}

DetectorMetadata collect_detector_metadata(const stim::DetectorErrorModel& flattened) {
  std::set<uint64_t> detector_ids;
  for (uint64_t d = 0; d < flattened.count_detectors(); ++d) {
    detector_ids.insert(d);
  }

  DetectorMetadata metadata;
  metadata.coordinates = flattened.get_detector_coordinates(detector_ids);
  for (const auto& instruction : flattened.instructions) {
    if (instruction.type == stim::DemInstructionType::DEM_DETECTOR) {
      metadata.tags[instruction.target_data[0].val()] = instruction.tag;
    }
  }
  return metadata;
}

void validate_detector_classes(const std::vector<int>& detector_classes, size_t num_detectors) {
  if (detector_classes.size() != num_detectors) {
    throw std::invalid_argument("Detector classification count does not match the DEM.");
  }

  std::set<int> unique_classes;
  for (size_t d = 0; d < detector_classes.size(); ++d) {
    int classifier_label = detector_classes[d];
    if (classifier_label < 0) {
      throw std::invalid_argument(
          "Detector D" + std::to_string(d) +
          " could not be classified (missing basis annotation or valid coordinates).");
    }
    unique_classes.insert(classifier_label);
  }

  if (unique_classes.size() != 2) {
    throw std::invalid_argument("Multi-pass decoding requires exactly 2 detector components; got " +
                                std::to_string(unique_classes.size()) + ".");
  }
}

}  // namespace

MultiPassTesseractDecoder::MultiPassTesseractDecoder(
    const stim::DetectorErrorModel& dem, size_t num_passes, const DetectorClassifier& classifier,
    const TesseractConfig& base_config, size_t num_det_orders, DetOrder det_order_method,
    uint64_t seed, SchedulingStrategy strategy, bool collect_plan_statistics)
    : num_passes(num_passes),
      strategy(strategy),
      total_global_detectors(dem.count_detectors()),
      base_config(base_config),
      num_det_orders(num_det_orders),
      det_order_method(det_order_method),
      seed(seed),
      collect_plan_statistics(collect_plan_statistics) {
  if (num_passes < 1 || num_passes > 2) {
    throw std::invalid_argument("num_passes must be 1 or 2.");
  }
  initialize(dem, classify_detectors(dem, classifier));
}

MultiPassTesseractDecoder::MultiPassTesseractDecoder(
    const stim::DetectorErrorModel& dem, size_t num_passes,
    const std::vector<int>& detector_classes, const TesseractConfig& base_config,
    size_t num_det_orders, DetOrder det_order_method, uint64_t seed, SchedulingStrategy strategy,
    bool collect_plan_statistics)
    : num_passes(num_passes),
      strategy(strategy),
      total_global_detectors(dem.count_detectors()),
      base_config(base_config),
      num_det_orders(num_det_orders),
      det_order_method(det_order_method),
      seed(seed),
      collect_plan_statistics(collect_plan_statistics) {
  if (num_passes < 1 || num_passes > 2) {
    throw std::invalid_argument("num_passes must be 1 or 2.");
  }
  initialize(dem, detector_classes);
}

std::vector<int> MultiPassTesseractDecoder::classify_detectors(
    const stim::DetectorErrorModel& dem, const DetectorClassifier& classifier) {
  stim::DetectorErrorModel flattened = dem.flattened();
  DetectorMetadata metadata = collect_detector_metadata(flattened);
  std::vector<int> detector_classes(flattened.count_detectors());
  for (size_t d = 0; d < detector_classes.size(); ++d) {
    const std::vector<double>& coordinates = metadata.coordinates[d];
    const std::string& tag = metadata.tags[d];
    detector_classes[d] = classifier((int)d, coordinates, tag);
  }
  validate_detector_classes(detector_classes, flattened.count_detectors());
  return detector_classes;
}

void MultiPassTesseractDecoder::initialize(const stim::DetectorErrorModel& dem,
                                           const std::vector<int>& detector_classes) {
  stim::DetectorErrorModel flattened = dem.flattened();
  total_global_detectors = (size_t)flattened.count_detectors();
  validate_detector_classes(detector_classes, total_global_detectors);
  DetectorMetadata metadata = collect_detector_metadata(flattened);

  if (collect_plan_statistics) {
    std::vector<size_t> error_index_map;
    stim::DetectorErrorModel monolithic_dem =
        common::merge_indistinguishable_errors(flattened, error_index_map);
    monolithic_dem = common::remove_zero_probability_errors(monolithic_dem, error_index_map);
    monolithic_statistics =
        dem_statistics(monolithic_dem.count_detectors(), get_errors_from_dem(monolithic_dem));
  }

  std::set<int> unique_classes;
  unique_classes.insert(detector_classes.begin(), detector_classes.end());

  std::map<int, int> class_to_comp_id;
  int next_comp_id = 0;
  for (int c : unique_classes) class_to_comp_id[c] = next_comp_id++;

  component_decoders.resize(unique_classes.size());
  for (const auto& [classifier_label, component_id] : class_to_comp_id) {
    component_decoders[component_id].classifier_label = classifier_label;
  }

  global_det_to_comp_id.resize(total_global_detectors);
  for (size_t i = 0; i < total_global_detectors; ++i) {
    int component_id = class_to_comp_id.at(detector_classes[i]);
    global_det_to_comp_id[i] = component_id;
    component_decoders[component_id].component_detectors.insert((int)i);
  }

  auto detector_component = [&](int detector) {
    if (detector < 0 || (size_t)detector >= global_det_to_comp_id.size()) {
      throw std::invalid_argument("Detector D" + std::to_string(detector) + " is out of range.");
    }
    return global_det_to_comp_id[detector];
  };

  stim::DetectorErrorModel decomposed =
      decompose_errors_using_detector_assignment(flattened, detector_component, true);

  ImpliedProbsMap raw_correlations = process_dem_correlations(decomposed, global_det_to_comp_id);

  auto component_dems = split_dem_by_component(decomposed, detector_component);

  for (size_t i = 0; i < component_decoders.size(); ++i) {
    auto& cd = component_decoders[i];

    std::vector<size_t> error_index_map;
    stim::DetectorErrorModel component_dem =
        common::merge_indistinguishable_errors(component_dems[i], error_index_map);
    component_dem = common::remove_zero_probability_errors(component_dem, error_index_map);

    for (size_t global_d = 0; global_d < total_global_detectors; ++global_d) {
      cd.global_to_local_det[global_d] = (int)global_d;
    }

    stim::DetectorErrorModel local_dem;
    for (size_t global_d = 0; global_d < total_global_detectors; ++global_d) {
      local_dem.append_detector_instruction(metadata.coordinates[global_d],
                                            stim::DemTarget::relative_detector_id(global_d),
                                            metadata.tags[global_d]);
    }

    for (const auto& inst : component_dem.instructions) {
      if (inst.type == stim::DemInstructionType::DEM_ERROR) {
        bool has_obs = false;
        for (const auto& t : inst.target_data) {
          if (t.is_observable_id()) has_obs = true;
        }
        if (has_obs) cd.affects_observable = true;
        local_dem.append_error_instruction(inst.arg_data[0], inst.target_data, inst.tag);
      } else if (inst.type == stim::DemInstructionType::DEM_LOGICAL_OBSERVABLE) {
        local_dem.append_dem_instruction(inst);
      }
    }

    TesseractConfig config = base_config;
    config.dem = local_dem;
    config.merge_errors = true;
    config.det_orders = build_det_orders(config.dem, num_det_orders, det_order_method, seed);

    cd.decoder = std::make_unique<TesseractDecoder>(config);
    if (base_config.verbose) {
      std::cout << "DEBUG: Component " << i << " initialized with " << cd.decoder->errors.size()
                << " errors and " << config.dem.count_detectors() << " detectors." << std::endl;
    }
    cd.error_index_to_rules.resize(cd.decoder->errors.size());

    for (size_t ei = 0; ei < cd.decoder->errors.size(); ++ei) {
      cd.original_costs.push_back(cd.decoder->errors[ei].likelihood_cost);
      ComponentSymptom global_symptom{cd.decoder->errors[ei].symptom.detectors,
                                      cd.decoder->errors[ei].symptom.observables};
      std::sort(global_symptom.detectors.begin(), global_symptom.detectors.end());
      std::sort(global_symptom.observables.begin(), global_symptom.observables.end());
      cd.symptom_to_error_index[global_symptom].push_back(ei);
    }
  }

  for (const auto& [global_symptom, implied_probs] : raw_correlations) {
    int causal_comp = global_det_to_comp_id[global_symptom.detectors[0]];

    auto it = component_decoders[causal_comp].symptom_to_error_index.find(global_symptom);
    if (it == component_decoders[causal_comp].symptom_to_error_index.end()) continue;

    // Loop through all degenerate causal error indices!
    for (size_t causal_err_idx : it->second) {
      for (const auto& imp : implied_probs) {
        const ComponentSymptom& target_symptom = imp.affected_symptom;
        int target_comp = global_det_to_comp_id[target_symptom.detectors[0]];

        auto t_it = component_decoders[target_comp].symptom_to_error_index.find(target_symptom);
        if (t_it != component_decoders[target_comp].symptom_to_error_index.end()) {
          // Loop through all degenerate target error indices and add rules to
          // each!
          for (size_t target_err_idx : t_it->second) {
            component_decoders[causal_comp].error_index_to_rules[causal_err_idx].push_back(
                {(size_t)target_comp, target_err_idx, imp.probability});
          }
        }
      }
    }
  }

  if (strategy == SchedulingStrategy::Static) {
    build_static_schedule();
  } else if (strategy == SchedulingStrategy::Causal) {
    build_causal_schedule();
  }
}

void MultiPassTesseractDecoder::build_static_schedule() {
  pass_schedule.assign(num_passes, {});
  for (size_t p = 0; p < num_passes; ++p) {
    for (size_t i = 0; i < component_decoders.size(); ++i) {
      pass_schedule[p].push_back(i);
    }
  }
}

void MultiPassTesseractDecoder::build_causal_schedule() {
  size_t num_components = component_decoders.size();
  std::vector<std::set<size_t>> schedule_sets(num_passes);

  // Initial seed: Final pass includes all components that directly affect an
  // observable.
  for (size_t i = 0; i < num_components; ++i) {
    if (component_decoders[i].affects_observable) {
      schedule_sets[num_passes - 1].insert(i);
    }
  }

  // Back-propagate dependencies through passes.
  // A component is needed in pass p if it can reweight a component needed in
  // pass p+1.
  for (int p = (int)num_passes - 2; p >= 0; --p) {
    // Start with everyone needed in the next pass (they might need to re-decode
    // or bias others) Actually, if a component is in pass p+1, it's because it
    // was influenced by pass p.
    for (size_t target_comp_idx : schedule_sets[p + 1]) {
      for (size_t causal_comp_idx = 0; causal_comp_idx < num_components; ++causal_comp_idx) {
        for (const auto& rules : component_decoders[causal_comp_idx].error_index_to_rules) {
          for (const auto& rule : rules) {
            if (rule.target_comp_idx == target_comp_idx) {
              schedule_sets[p].insert(causal_comp_idx);
            }
          }
        }
      }
    }
  }

  // Convert sets to pass_schedule vectors.
  pass_schedule.assign(num_passes, {});
  for (size_t p = 0; p < num_passes; ++p) {
    for (size_t c_idx : schedule_sets[p]) {
      pass_schedule[p].push_back(c_idx);
    }
  }
}

MultiPassExecutionPlan MultiPassTesseractDecoder::get_execution_plan() const {
  if (!collect_plan_statistics) {
    throw std::logic_error("Execution plan statistics were not collected.");
  }
  MultiPassExecutionPlan plan{num_passes, strategy, monolithic_statistics, {}, {}, pass_schedule};
  for (size_t component_id = 0; component_id < component_decoders.size(); ++component_id) {
    const auto& component = component_decoders[component_id];
    auto statistics =
        dem_statistics(component.component_detectors.size(), component.decoder->errors);
    plan.components.push_back({component_id, component.classifier_label, statistics.detector_count,
                               component.decoder->num_detectors, statistics.error_mechanism_count,
                               statistics.average_detector_degree, component.affects_observable});
  }

  std::map<std::pair<size_t, size_t>, size_t> dependency_counts;
  for (size_t source = 0; source < component_decoders.size(); ++source) {
    for (const auto& rules : component_decoders[source].error_index_to_rules) {
      for (const auto& rule : rules) {
        dependency_counts[{source, rule.target_comp_idx}]++;
      }
    }
  }
  for (const auto& [components, rule_count] : dependency_counts) {
    plan.dependencies.push_back({components.first, components.second, rule_count});
  }
  return plan;
}

std::vector<int> MultiPassTesseractDecoder::decode(const std::vector<uint64_t>& detections) {
  return decode_result(detections).predictions;
}

MultiPassDecodeResult MultiPassTesseractDecoder::decode_result(
    const std::vector<uint64_t>& detections) {
  for (uint64_t d : detections) {
    if (d >= total_global_detectors) {
      throw std::invalid_argument("Detector D" + std::to_string(d) +
                                  " is out of range for a model with " +
                                  std::to_string(total_global_detectors) + " detectors.");
    }
  }

  last_shot_num_reweights = 0;

  // 1. Multi-Pass Loop: Sequentially schedules component passes and propagates
  // priors.
  for (size_t pass = 0; pass < num_passes; ++pass) {
    bool is_final_pass = (pass == num_passes - 1);

    // Decode scheduled components for the current pass layer using persistent
    // local buffers.
    for (size_t comp_idx : pass_schedule[pass]) {
      auto& cd = component_decoders[comp_idx];
      std::vector<uint64_t> local_dets;
      for (uint64_t d : detections) {
        if (cd.component_detectors.count(static_cast<int>(d))) {
          local_dets.push_back(d);
        }
      }

      cd.decoder->decode_to_errors(local_dets);
      component_predictions[comp_idx] = cd.decoder->predicted_errors_buffer;
    }

    if (!is_final_pass) {
      // Step A: Apply Damped Fractional Memory to previously modified priors.
      // Smoothly decay current modifications back toward the baseline to
      // prevent message saturation.
      double gamma = 0.5;  // Tunable decay factor: 1.0 is strict isolation, 0.0
                           // is full accumulation.

      for (size_t m_comp_idx : modified_component_indices) {
        auto& cd = component_decoders[m_comp_idx];
        if (!cd.shot_all_modified_error_indices.empty()) {
          for (size_t idx : cd.shot_all_modified_error_indices) {
            double baseline_cost = cd.original_costs[idx];
            double current_cost = cd.decoder->errors[idx].likelihood_cost;
            cd.decoder->errors[idx].likelihood_cost =
                gamma * baseline_cost + (1.0 - gamma) * current_cost;
          }
          cd.decoder->update_internal_costs(cd.shot_all_modified_error_indices);
          // Retain tracking indices so the final Surgical Reset completely
          // clears cross-shot state.
        }
      }

      // Step B: Broadcast reweighting rules derived strictly from the latest
      // predictions.
      for (size_t comp_idx : pass_schedule[pass]) {
        auto& cd = component_decoders[comp_idx];
        for (size_t dem_err_idx : cd.decoder->predicted_errors_buffer) {
          size_t internal_err_idx = cd.decoder->dem_error_to_error.at(dem_err_idx);
          if (internal_err_idx == std::numeric_limits<size_t>::max()) continue;

          for (const auto& rule : cd.error_index_to_rules[internal_err_idx]) {
            auto& target_cd = component_decoders[rule.target_comp_idx];

            modified_component_indices.push_back(rule.target_comp_idx);

            // Apply Max-Prob Rule safely for concurrent rules within this pass
            // layer.
            double current_p = target_cd.decoder->errors[rule.target_error_idx].get_probability();
            if (rule.conditional_prob > current_p) {
              target_cd.decoder->errors[rule.target_error_idx].set_with_probability(
                  std::min(rule.conditional_prob, 0.5));
              target_cd.shot_all_modified_error_indices.push_back(rule.target_error_idx);
              last_shot_num_reweights++;
            }
          }
        }
      }

      // Step C: Deduplicate modified tracking vectors and synchronize internal
      // graph costs.
      std::sort(modified_component_indices.begin(), modified_component_indices.end());
      modified_component_indices.erase(
          std::unique(modified_component_indices.begin(), modified_component_indices.end()),
          modified_component_indices.end());

      for (size_t m_comp_idx : modified_component_indices) {
        auto& cd = component_decoders[m_comp_idx];
        if (!cd.shot_all_modified_error_indices.empty()) {
          std::sort(cd.shot_all_modified_error_indices.begin(),
                    cd.shot_all_modified_error_indices.end());
          cd.shot_all_modified_error_indices.erase(
              std::unique(cd.shot_all_modified_error_indices.begin(),
                          cd.shot_all_modified_error_indices.end()),
              cd.shot_all_modified_error_indices.end());
          cd.decoder->update_internal_costs(cd.shot_all_modified_error_indices);
        }
      }
    }
  }

  // 2. Unified Logical Extraction: Collect final predictions from ALL
  // components that ran during the shot.
  std::set<int> flipped_observables;
  bool aggregate_low_confidence = false;
  double aggregate_cost = 0.0;

  for (const auto& [comp_idx, preds] : component_predictions) {
    auto& cd = component_decoders[comp_idx];
    if (cd.decoder->low_confidence_flag) {
      aggregate_low_confidence = true;
    }
    if (!preds.empty()) {
      std::vector<int> local_flips = cd.decoder->get_flipped_observables(preds);
      for (int obs : local_flips) {
        if (flipped_observables.count(obs))
          flipped_observables.erase(obs);
        else
          flipped_observables.insert(obs);
      }
    }
  }

  for (size_t comp_idx : pass_schedule.back()) {
    auto& cd = component_decoders[comp_idx];
    const auto& preds = component_predictions.at(comp_idx);
    aggregate_cost += cd.decoder->cost_from_errors(preds);
  }

  // 3. Surgical Reset: Restore modified costs to leave the internal structures
  // pristine for the next shot.
  for (size_t m_comp_idx : modified_component_indices) {
    auto& cd = component_decoders[m_comp_idx];
    if (!cd.shot_all_modified_error_indices.empty()) {
      for (size_t idx : cd.shot_all_modified_error_indices) {
        cd.decoder->errors[idx].likelihood_cost = cd.original_costs[idx];
      }
      cd.decoder->update_internal_costs(cd.shot_all_modified_error_indices);
      cd.shot_all_modified_error_indices.clear();
    }
  }

  modified_component_indices.clear();
  final_pass_active_components.clear();

  MultiPassDecodeResult res;
  res.predictions = std::vector<int>(flipped_observables.begin(), flipped_observables.end());
  res.low_confidence = aggregate_low_confidence;
  res.total_cost = aggregate_cost;
  return res;
}

}  // namespace tesseract
