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
#include <limits>
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

std::unique_ptr<Decoder> make_decoder(const TesseractConfig& config) {
  if (config.multipass) {
    return std::make_unique<MultiPassTesseractDecoder>(config);
  }
  return std::make_unique<TesseractDecoder>(config);
}

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
    ss << "  component " << component.id << ": label=" << component.assignment_label
       << ", active_detectors=" << component.active_detector_count
       << ", decoder_detectors=" << component.decoder_detector_count
       << ", observable=" << (component.affects_observable ? "yes" : "no")
       << ", error_mechanisms=" << component.error_mechanism_count
       << ", average_active_detector_degree=" << component.average_active_detector_degree << '\n';
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

template <typename Callback>
class ScopeExit {
 public:
  explicit ScopeExit(Callback callback) : callback(std::move(callback)) {}
  ScopeExit(const ScopeExit&) = delete;
  ScopeExit& operator=(const ScopeExit&) = delete;
  ~ScopeExit() noexcept {
    callback();
  }

 private:
  Callback callback;
};

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

void validate_detector_components(const std::vector<int>& detector_components,
                                  size_t num_detectors) {
  if (detector_components.size() != num_detectors) {
    throw std::invalid_argument(
        "Detector component assignment count does not match the DEM detector count.");
  }

  std::set<int> unique_components;
  for (size_t d = 0; d < detector_components.size(); ++d) {
    int component = detector_components[d];
    if (component < 0) {
      throw std::invalid_argument("Detector D" + std::to_string(d) +
                                  " has an invalid negative component assignment.");
    }
    unique_components.insert(component);
  }

  if (unique_components.size() != 2) {
    throw std::invalid_argument("Multi-pass decoding requires exactly 2 detector components; got " +
                                std::to_string(unique_components.size()) + ".");
  }
}

}  // namespace

MultiPassTesseractDecoder::MultiPassTesseractDecoder(const TesseractConfig& config)
    : MultiPassTesseractDecoder(
          config.dem, config.num_passes,
          config.detector_components.empty() ? classify_canonical_detector_bases(config.dem)
                                             : config.detector_components,
          config, config.num_det_orders, config.det_order_method, config.det_order_seed,
          config.multipass_strategy, config.collect_multipass_plan_statistics) {}

MultiPassTesseractDecoder::MultiPassTesseractDecoder(
    const stim::DetectorErrorModel& dem, size_t num_passes,
    const std::vector<int>& detector_components, const TesseractConfig& base_config,
    size_t num_det_orders, DetOrder det_order_method, uint64_t seed, SchedulingStrategy strategy,
    bool collect_plan_statistics)
    : num_passes(num_passes),
      strategy(strategy),
      total_global_detectors(dem.count_detectors()),
      collect_plan_statistics(collect_plan_statistics) {
  if (num_passes < 1 || num_passes > 2) {
    throw std::invalid_argument("num_passes must be 1 or 2.");
  }
  if (strategy != SchedulingStrategy::Static && strategy != SchedulingStrategy::Causal) {
    throw std::invalid_argument("Invalid multi-pass scheduling strategy.");
  }
  if (num_passes == 2 && !base_config.merge_errors) {
    throw std::invalid_argument(
        "Two-pass decoding requires merge_errors=true because reweighting is defined on "
        "aggregate component symptoms; use one pass to decode unmerged error mechanisms.");
  }
  initialize(dem, detector_components, base_config, num_det_orders, det_order_method, seed);
}

void MultiPassTesseractDecoder::initialize(const stim::DetectorErrorModel& dem,
                                           const std::vector<int>& detector_components,
                                           const TesseractConfig& base_config,
                                           size_t num_det_orders, DetOrder det_order_method,
                                           uint64_t seed) {
  stim::DetectorErrorModel flattened = dem.flattened();
  total_global_detectors = flattened.count_detectors();
  validate_detector_components(detector_components, total_global_detectors);

  if (collect_plan_statistics) {
    std::vector<size_t> error_index_map;
    stim::DetectorErrorModel statistics_dem = flattened;
    if (base_config.merge_errors) {
      statistics_dem = common::merge_indistinguishable_errors(statistics_dem, error_index_map);
    }
    statistics_dem = common::remove_zero_probability_errors(statistics_dem, error_index_map);
    monolithic_statistics =
        dem_statistics(statistics_dem.count_detectors(), get_errors_from_dem(statistics_dem));
  }

  std::set<int> unique_labels(detector_components.begin(), detector_components.end());
  std::map<int, int> label_to_component;
  for (int label : unique_labels) {
    label_to_component.emplace(label, static_cast<int>(label_to_component.size()));
  }

  component_decoders.resize(label_to_component.size());
  for (const auto& [label, component] : label_to_component) {
    component_decoders[component].assignment_label = label;
  }

  global_det_to_comp_id.resize(total_global_detectors);
  for (size_t d = 0; d < total_global_detectors; ++d) {
    int component = label_to_component.at(detector_components[d]);
    global_det_to_comp_id[d] = component;
    component_decoders[component].active_detector_count++;
  }

  stim::DetectorErrorModel decomposed =
      decompose_errors_using_detector_assignment(flattened, global_det_to_comp_id);
  ReweightProbsMap reweight_probabilities;
  if (num_passes == 2) {
    reweight_probabilities = process_dem_correlations(decomposed, global_det_to_comp_id);
  }
  std::map<int, stim::DetectorErrorModel> component_dems =
      split_dem_by_component(decomposed, global_det_to_comp_id);

  for (size_t component = 0; component < component_decoders.size(); ++component) {
    auto& component_decoder = component_decoders[component];
    TesseractConfig config = base_config;
    config.multipass = false;
    config.dem = component_dems.at(component);
    if (config.det_orders.empty()) {
      config.det_orders = build_det_orders(config.dem, num_det_orders, det_order_method, seed);
    }
    component_decoder.decoder = std::make_unique<TesseractDecoder>(std::move(config));
    component_decoder.error_index_to_rules.resize(component_decoder.decoder->errors.size());

    for (size_t error_index = 0; error_index < component_decoder.decoder->errors.size();
         ++error_index) {
      const auto& error = component_decoder.decoder->errors[error_index];
      component_decoder.original_costs.push_back(error.likelihood_cost);
      component_decoder.affects_observable |= !error.symptom.observables.empty();
      ComponentSymptom symptom{error.symptom.detectors, error.symptom.observables};
      component_decoder.symptom_to_error_index[std::move(symptom)].push_back(error_index);
    }
  }

  for (const auto& [causal_symptom, probabilities] : reweight_probabilities) {
    int causal_component = global_det_to_comp_id.at(causal_symptom.detectors.at(0));
    auto causal_errors =
        component_decoders[causal_component].symptom_to_error_index.find(causal_symptom);
    if (causal_errors == component_decoders[causal_component].symptom_to_error_index.end()) {
      continue;
    }

    for (size_t causal_error : causal_errors->second) {
      for (const auto& probability : probabilities) {
        int target_component =
            global_det_to_comp_id.at(probability.affected_symptom.detectors.at(0));
        auto target_errors = component_decoders[target_component].symptom_to_error_index.find(
            probability.affected_symptom);
        if (target_errors == component_decoders[target_component].symptom_to_error_index.end()) {
          continue;
        }
        for (size_t target_error : target_errors->second) {
          component_decoders[causal_component].error_index_to_rules[causal_error].push_back(
              {static_cast<size_t>(target_component), target_error, probability.probability});
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
        for (const auto& rules : component_decoders[source_component].error_index_to_rules) {
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
  if (!collect_plan_statistics) {
    throw std::logic_error("Execution plan statistics were not collected.");
  }
  MultiPassExecutionPlan plan{num_passes, strategy, monolithic_statistics, {}, {}, pass_schedule};
  for (size_t component_id = 0; component_id < component_decoders.size(); ++component_id) {
    const auto& component = component_decoders[component_id];
    auto statistics = dem_statistics(component.active_detector_count, component.decoder->errors);
    plan.components.push_back({component_id, component.assignment_label,
                               component.active_detector_count, component.decoder->num_detectors,
                               statistics.error_mechanism_count, statistics.average_detector_degree,
                               component.affects_observable});
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

void MultiPassTesseractDecoder::restore_modified_costs(
    const std::vector<std::vector<size_t>>& modified_error_indices) {
  for (size_t component = 0; component < modified_error_indices.size(); ++component) {
    if (modified_error_indices[component].empty()) {
      continue;
    }
    auto& component_decoder = component_decoders.at(component);
    for (size_t error_index : modified_error_indices[component]) {
      component_decoder.decoder->errors.at(error_index).likelihood_cost =
          component_decoder.original_costs.at(error_index);
    }
    component_decoder.decoder->update_internal_costs(modified_error_indices[component]);
  }
}

DecoderResult MultiPassTesseractDecoder::decode_result(const std::vector<uint64_t>& detections) {
  last_shot_num_reweights = 0;
  for (uint64_t detector : detections) {
    if (detector >= total_global_detectors) {
      throw std::invalid_argument("Detector D" + std::to_string(detector) +
                                  " is out of range for a model with " +
                                  std::to_string(total_global_detectors) + " detectors.");
    }
  }

  std::vector<std::vector<size_t>> component_predictions(component_decoders.size());
  std::vector<std::vector<size_t>> modified_error_indices(component_decoders.size());
  ScopeExit reset_guard([&] { restore_modified_costs(modified_error_indices); });
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

      component_decoder.decoder->decode_to_errors(component_detections);
      aggregate_low_confidence |= component_decoder.decoder->low_confidence_flag;
      component_predictions[component] = component_decoder.decoder->predicted_errors_buffer;
    }

    if (is_final_pass) {
      continue;
    }

    for (size_t source_component : pass_schedule[pass]) {
      auto& source = component_decoders.at(source_component);
      for (size_t dem_error_index : component_predictions[source_component]) {
        size_t source_error = source.decoder->dem_error_to_error.at(dem_error_index);
        if (source_error == std::numeric_limits<size_t>::max()) {
          throw std::logic_error("A decoded error does not map to a retained component error.");
        }
        for (const auto& rule : source.error_index_to_rules.at(source_error)) {
          auto& target = component_decoders.at(rule.target_comp_idx);
          auto& target_error = target.decoder->errors.at(rule.target_error_idx);
          double reweight_probability =
              std::min(rule.reweight_probability, MAX_REWEIGHT_PROBABILITY);
          if (reweight_probability > target_error.get_probability()) {
            modified_error_indices.at(rule.target_comp_idx).push_back(rule.target_error_idx);
            target_error.set_with_probability(reweight_probability);
            last_shot_num_reweights++;
          }
        }
      }
    }

    for (size_t component = 0; component < modified_error_indices.size(); ++component) {
      auto& modified = modified_error_indices[component];
      if (modified.empty()) {
        continue;
      }
      std::sort(modified.begin(), modified.end());
      modified.erase(std::unique(modified.begin(), modified.end()), modified.end());
      component_decoders[component].decoder->update_internal_costs(modified);
    }
  }

  std::set<int> flipped_observables;
  double aggregate_cost = 0;
  for (size_t component : pass_schedule.back()) {
    const auto& decoder = component_decoders.at(component).decoder;
    const auto& predictions = component_predictions.at(component);
    for (int observable : decoder->get_flipped_observables(predictions)) {
      if (!flipped_observables.erase(observable)) {
        flipped_observables.insert(observable);
      }
    }
    aggregate_cost += decoder->cost_from_errors(predictions);
  }

  DecoderResult result;
  result.predictions.assign(flipped_observables.begin(), flipped_observables.end());
  result.low_confidence = aggregate_low_confidence;
  result.total_cost = aggregate_cost;
  return result;
}

}  // namespace tesseract_decoder
