// Copyright 2026 Google LLC
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

#include "multi_pass_tesseract_decoder.h"

#include <algorithm>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "../common.h"
#include "../utils.h"
#include "dem_decomposition.h"

namespace tesseract_decoder {

SchedulingStrategy parse_scheduling_strategy(std::string_view value) {
  if (value == "static") {
    return SchedulingStrategy::Static;
  }
  if (value == "causal") {
    return SchedulingStrategy::Causal;
  }
  throw std::invalid_argument("Invalid multi-pass scheduling strategy '" + std::string(value) +
                              "'; expected 'static' or 'causal'.");
}

const char* scheduling_strategy_name(SchedulingStrategy strategy) {
  switch (strategy) {
    case SchedulingStrategy::Static:
      return "static";
    case SchedulingStrategy::Causal:
      return "causal";
  }
  throw std::invalid_argument("Invalid multi-pass scheduling strategy.");
}

std::string MultiPassExecutionPlan::str() const {
  std::stringstream ss;
  ss << "Multi-pass execution plan\n"
     << "strategy: " << scheduling_strategy_name(strategy) << '\n'
     << "passes: " << num_passes << '\n'
     << "monolithic input DEM: detectors=" << monolithic_statistics.detector_count
     << ", error_mechanisms=" << monolithic_statistics.error_mechanism_count
     << ", average_detector_degree=" << monolithic_statistics.average_detector_degree << '\n'
     << "components: " << components.size() << '\n';
  for (const auto& component : components) {
    ss << "  component " << component.id << ": label=" << component.assignment_label
       << ", active_detectors=" << component.active_detector_count
       << ", decoder_detectors=" << component.decoder_detector_count
       << ", observable=" << (component.affects_observable ? "yes" : "no")
       << ", error_mechanisms=" << component.error_mechanism_count
       << ", average_active_detector_degree=" << component.average_active_detector_degree
       << ", sparsify_reactivate_limit=";
    if (!component.sparsify_errors) {
      ss << "disabled";
    } else {
      ss << component.sparsify_reactivate_limit;
    }
    ss << '\n';
  }
  ss << "dependencies:\n";
  if (dependencies.empty()) {
    ss << "  none\n";
  }
  for (const auto& dependency : dependencies) {
    ss << "  component " << dependency.source_component << " -> component "
       << dependency.target_component << ": " << dependency.rule_count << " rules\n";
  }
  ss << "schedule:\n";
  for (size_t pass = 0; pass < pass_schedule.size(); ++pass) {
    ss << "  pass " << pass + 1 << ": [";
    for (size_t i = 0; i < pass_schedule[pass].size(); ++i) {
      if (i) {
        ss << ", ";
      }
      ss << pass_schedule[pass][i];
    }
    ss << "]\n";
  }
  return ss.str();
}

namespace {

constexpr double MAX_REWEIGHT_PROBABILITY = 0.499;

int canonical_detector_basis_component(int detector, const std::string& tag) {
  const std::string detector_name = "Detector D" + std::to_string(detector);
  if (tag.empty()) {
    throw std::invalid_argument(detector_name +
                                " has no tag; multi-pass CLI input requires a top-level JSON "
                                "basis field equal to \"X\" or \"Z\".");
  }

  nlohmann::json metadata = nlohmann::json::parse(tag, nullptr, false);
  if (metadata.is_discarded()) {
    throw std::invalid_argument(detector_name +
                                " has a non-JSON tag; multi-pass CLI input requires a top-level "
                                "JSON basis field equal to \"X\" or \"Z\".");
  }
  if (!metadata.is_object() || !metadata.contains("basis")) {
    throw std::invalid_argument(detector_name +
                                " tag has no top-level basis field equal to \"X\" or \"Z\".");
  }
  const auto& basis = metadata["basis"];
  if (basis == "X") return 0;
  if (basis == "Z") return 1;
  throw std::invalid_argument(detector_name +
                              " has an invalid top-level basis; expected the string \"X\" or "
                              "\"Z\".");
}

std::vector<int> classify_canonical_detector_bases(const stim::DetectorErrorModel& dem) {
  stim::DetectorErrorModel flattened = dem.flattened();
  std::vector<std::string> detector_tags(flattened.count_detectors());
  std::vector<bool> has_detector_instruction(flattened.count_detectors());

  for (const stim::DemInstruction& instruction : flattened.instructions) {
    if (instruction.type != stim::DemInstructionType::DEM_DETECTOR) continue;
    if (instruction.target_data.size() != 1 ||
        !instruction.target_data[0].is_relative_detector_id()) {
      throw std::invalid_argument("Malformed detector instruction: " + instruction.str());
    }
    size_t detector = instruction.target_data[0].val();
    if (detector >= detector_tags.size()) {
      throw std::invalid_argument("Detector D" + std::to_string(detector) + " is out of range.");
    }
    if (has_detector_instruction[detector]) {
      throw std::invalid_argument("Detector D" + std::to_string(detector) +
                                  " has more than one detector instruction.");
    }
    has_detector_instruction[detector] = true;
    detector_tags[detector] = instruction.tag;
  }

  std::vector<int> detector_components(detector_tags.size());
  for (size_t detector = 0; detector < detector_tags.size(); ++detector) {
    if (!has_detector_instruction[detector]) {
      throw std::invalid_argument("Detector D" + std::to_string(detector) +
                                  " has no detector instruction with a canonical basis tag.");
    }
    detector_components[detector] =
        canonical_detector_basis_component(static_cast<int>(detector), detector_tags[detector]);
  }
  return detector_components;
}

MultiPassExecutionPlan::DemStatistics dem_statistics(size_t detector_count,
                                                     const std::vector<common::Error>& errors) {
  size_t detector_incidences = 0;
  for (const auto& error : errors) {
    detector_incidences += error.symptom.detectors.size();
  }
  double average_detector_degree =
      detector_count == 0 ? 0.0 : static_cast<double>(detector_incidences) / detector_count;
  return {detector_count, errors.size(), average_detector_degree};
}

MultiPassExecutionPlan::DemStatistics dem_statistics(size_t detector_count,
                                                     const TesseractDecoder& decoder) {
  size_t detector_incidences = 0;
  for (size_t error_index = 0; error_index < decoder.error_count(); ++error_index) {
    detector_incidences += decoder.error_symptom(error_index).detectors.size();
  }
  double average_detector_degree =
      detector_count == 0 ? 0.0 : static_cast<double>(detector_incidences) / detector_count;
  return {detector_count, decoder.error_count(), average_detector_degree};
}

}  // namespace

MultiPassTesseractDecoder::MultiPassTesseractDecoder(MultiPassTesseractConfig config)
    : num_passes(config.num_passes), strategy(config.strategy) {
  if (num_passes < 1 || num_passes > 2) {
    throw std::invalid_argument("num_passes must be 1 or 2.");
  }
  if (strategy != SchedulingStrategy::Static && strategy != SchedulingStrategy::Causal) {
    throw std::invalid_argument("Invalid multi-pass scheduling strategy.");
  }
  if (num_passes == 2 && !config.component_config.merge_errors) {
    throw std::invalid_argument(
        "Two-pass decoding requires merge_errors=true because reweighting is defined on "
        "aggregate component symptoms; use one pass to decode unmerged error mechanisms.");
  }
  std::vector<int> detector_components = std::move(config.detector_components);
  if (detector_components.empty()) {
    detector_components = classify_canonical_detector_bases(config.component_config.dem);
  }
  initialize(config.component_config.dem, detector_components, config.component_config);
}

void MultiPassTesseractDecoder::initialize(const stim::DetectorErrorModel& dem,
                                           const std::vector<int>& detector_components,
                                           const TesseractConfig& base_config) {
  TwoComponentDem prepared = prepare_two_component_dem(dem, detector_components);
  global_det_to_comp_id = prepared.detector_components;
  for (size_t d = 0; d < global_det_to_comp_id.size(); ++d) {
    int component = global_det_to_comp_id[d];
    component_decoders[component].active_detector_count++;
  }

  ReweightProbsMap reweight_probabilities;
  if (num_passes == 2) {
    reweight_probabilities = process_dem_correlations(prepared);
  }
  monolithic_dem = std::move(prepared.decomposed_dem);

  std::array<std::map<ComponentSymptom, std::vector<std::pair<size_t, double>>>, 2>
      symptom_to_errors;
  for (size_t component = 0; component < component_decoders.size(); ++component) {
    auto& component_decoder = component_decoders[component];
    component_decoder.assignment_label = prepared.assignment_labels[component];
    TesseractConfig config = base_config;
    config.dem = std::move(prepared.component_dems[component]);
    component_decoder.decoder = std::make_unique<TesseractDecoder>(std::move(config));

    for (size_t error_index = 0; error_index < component_decoder.decoder->error_count();
         ++error_index) {
      const auto& symptom = component_decoder.decoder->error_symptom(error_index);
      component_decoder.affects_observable |= !symptom.observables.empty();
      if (num_passes == 2) {
        ComponentSymptom component_symptom{symptom.detectors, symptom.observables};
        symptom_to_errors[component][std::move(component_symptom)].push_back(
            {component_decoder.decoder->dem_error_index(error_index),
             component_decoder.decoder->error_probability(error_index)});
      }
    }
  }

  for (const auto& [causal_symptom, probabilities] : reweight_probabilities) {
    int causal_component = global_det_to_comp_id.at(causal_symptom.detectors.at(0));
    auto causal_errors = symptom_to_errors[causal_component].find(causal_symptom);
    if (causal_errors == symptom_to_errors[causal_component].end()) {
      continue;
    }

    for (const auto& causal_error : causal_errors->second) {
      size_t causal_dem_error_index = causal_error.first;
      for (const auto& probability : probabilities) {
        int target_component =
            global_det_to_comp_id.at(probability.affected_symptom.detectors.at(0));
        auto target_errors = symptom_to_errors[target_component].find(probability.affected_symptom);
        if (target_errors == symptom_to_errors[target_component].end()) {
          continue;
        }
        for (const auto& [target_dem_error_index, target_probability] : target_errors->second) {
          component_decoders[causal_component].reweight_rules[causal_dem_error_index].push_back(
              {static_cast<size_t>(target_component), target_dem_error_index,
               probability.probability, target_probability});
        }
      }
    }
  }

  if (strategy == SchedulingStrategy::Static) {
    build_static_schedule();
  } else {
    build_causal_schedule();
  }
}

void MultiPassTesseractDecoder::build_static_schedule() {
  pass_schedule.assign(num_passes, {});
  for (auto& pass : pass_schedule) {
    for (size_t component = 0; component < component_decoders.size(); ++component) {
      pass.push_back(component);
    }
  }
}

void MultiPassTesseractDecoder::build_causal_schedule() {
  std::vector<std::set<size_t>> schedule_sets(num_passes);
  for (size_t component = 0; component < component_decoders.size(); ++component) {
    if (component_decoders[component].affects_observable) {
      schedule_sets.back().insert(component);
    }
  }

  for (int pass = static_cast<int>(num_passes) - 2; pass >= 0; --pass) {
    for (size_t target_component : schedule_sets[pass + 1]) {
      for (size_t source_component = 0; source_component < component_decoders.size();
           ++source_component) {
        for (const auto& rule_entry : component_decoders[source_component].reweight_rules) {
          const auto& rules = rule_entry.second;
          for (const auto& rule : rules) {
            if (rule.target_comp_idx == target_component) {
              schedule_sets[pass].insert(source_component);
            }
          }
        }
      }
    }
  }

  pass_schedule.assign(num_passes, {});
  for (size_t pass = 0; pass < num_passes; ++pass) {
    pass_schedule[pass].assign(schedule_sets[pass].begin(), schedule_sets[pass].end());
  }
}

MultiPassExecutionPlan MultiPassTesseractDecoder::get_execution_plan() const {
  std::vector<size_t> error_index_map;
  stim::DetectorErrorModel statistics_dem = monolithic_dem;
  if (component_decoders.front().decoder->config.merge_errors) {
    statistics_dem = common::merge_indistinguishable_errors(statistics_dem, error_index_map);
  }
  statistics_dem = common::remove_zero_probability_errors(statistics_dem, error_index_map);
  auto monolithic_statistics =
      dem_statistics(statistics_dem.count_detectors(), get_errors_from_dem(statistics_dem));

  MultiPassExecutionPlan plan{num_passes, strategy, monolithic_statistics, {}, {}, pass_schedule};
  for (size_t component_id = 0; component_id < component_decoders.size(); ++component_id) {
    const auto& component = component_decoders[component_id];
    auto statistics = dem_statistics(component.active_detector_count, *component.decoder);
    plan.components.push_back(
        {component_id, component.assignment_label, component.active_detector_count,
         component.decoder->config.dem.count_detectors(), statistics.error_mechanism_count,
         statistics.average_detector_degree, component.decoder->config.sparsify_errors,
         component.decoder->config.sparsify_reactivate_limit, component.affects_observable});
  }

  std::map<std::pair<size_t, size_t>, size_t> dependency_counts;
  for (size_t source = 0; source < component_decoders.size(); ++source) {
    for (const auto& rule_entry : component_decoders[source].reweight_rules) {
      const auto& rules = rule_entry.second;
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

DecodeResult MultiPassTesseractDecoder::decode_result(const std::vector<uint64_t>& detections) {
  for (uint64_t detector : detections) {
    if (detector >= global_det_to_comp_id.size()) {
      throw std::invalid_argument("Detector D" + std::to_string(detector) +
                                  " is out of range for a model with " +
                                  std::to_string(global_det_to_comp_id.size()) + " detectors.");
    }
  }

  std::array<DecodeResult, 2> component_results;
  std::array<std::map<size_t, double>, 2> probability_updates;
  bool aggregate_low_confidence = false;

  for (size_t pass = 0; pass < num_passes; ++pass) {
    bool is_final_pass = pass + 1 == num_passes;
    for (size_t component : pass_schedule[pass]) {
      auto& component_decoder = component_decoders.at(component);
      std::vector<uint64_t> component_detections;
      for (uint64_t detector : detections) {
        if (global_det_to_comp_id.at(detector) == static_cast<int>(component)) {
          component_detections.push_back(detector);
        }
      }

      std::vector<ErrorProbabilityUpdate> updates;
      updates.reserve(probability_updates[component].size());
      for (const auto& [dem_error_index, probability] : probability_updates[component]) {
        updates.push_back({dem_error_index, probability});
      }
      component_results[component] =
          updates.empty() ? component_decoder.decoder->decode_result(component_detections)
                          : component_decoder.decoder->decode_result(component_detections, updates);
      aggregate_low_confidence |= component_results[component].low_confidence;
    }

    if (is_final_pass) {
      continue;
    }

    for (size_t source_component : pass_schedule[pass]) {
      const auto& source = component_decoders.at(source_component);
      for (size_t dem_error_index : component_results[source_component].predicted_errors) {
        auto rules = source.reweight_rules.find(dem_error_index);
        if (rules == source.reweight_rules.end()) continue;
        for (const auto& rule : rules->second) {
          double reweight_probability =
              std::min(rule.reweight_probability, MAX_REWEIGHT_PROBABILITY);
          if (reweight_probability > rule.original_probability) {
            double& update = probability_updates[rule.target_comp_idx][rule.target_dem_error_idx];
            update = std::max(update, reweight_probability);
          }
        }
      }
    }
  }

  std::set<int> flipped_observables;
  double aggregate_cost = 0;
  for (size_t component : pass_schedule.back()) {
    const auto& component_result = component_results.at(component);
    for (int observable : component_result.predictions) {
      if (!flipped_observables.erase(observable)) {
        flipped_observables.insert(observable);
      }
    }
    aggregate_cost += component_result.total_cost;
  }

  DecodeResult result;
  result.predictions.assign(flipped_observables.begin(), flipped_observables.end());
  result.low_confidence = aggregate_low_confidence;
  result.total_cost = aggregate_cost;
  return result;
}

}  // namespace tesseract_decoder
