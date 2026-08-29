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

#include "dem_decomposition.h"

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tesseract_decoder {
namespace {

struct ErrorGroup {
  std::vector<stim::DemTarget> targets;
  std::vector<int> detectors;
  std::vector<int> observables;
  int component;
};

std::vector<int> reduce_symmetric_difference(const std::vector<int>& items) {
  std::set<int> unpaired;
  for (int item : items) {
    if (!unpaired.erase(item)) {
      unpaired.insert(item);
    }
  }
  return {unpaired.begin(), unpaired.end()};
}

std::vector<int> reduce_set_symmetric_difference(const std::vector<std::vector<int>>& sets) {
  std::vector<int> items;
  for (const auto& set : sets) {
    items.insert(items.end(), set.begin(), set.end());
  }
  return reduce_symmetric_difference(items);
}

void validate_detector_components(const stim::DetectorErrorModel& dem,
                                  const std::vector<int>& detector_components) {
  if (detector_components.size() != dem.count_detectors()) {
    throw std::invalid_argument(
        "Detector component assignment count does not match the DEM detector count.");
  }
  for (size_t d = 0; d < detector_components.size(); ++d) {
    if (detector_components[d] < 0) {
      throw std::invalid_argument("Detector D" + std::to_string(d) +
                                  " has an invalid negative component assignment.");
    }
  }
}

int component_for_detector(int detector, const std::vector<int>& detector_components) {
  if (detector < 0 || static_cast<size_t>(detector) >= detector_components.size()) {
    throw std::invalid_argument("Detector D" + std::to_string(detector) + " is out of range.");
  }
  return detector_components[detector];
}

bool has_separator(const stim::DemInstruction& instruction) {
  return std::any_of(instruction.target_data.begin(), instruction.target_data.end(),
                     [](const stim::DemTarget& target) { return target.is_separator(); });
}

std::vector<ErrorGroup> parse_and_validate_groups(const stim::DemInstruction& instruction,
                                                  const std::vector<int>& detector_components,
                                                  bool allow_undecomposed_mixed_group) {
  if (instruction.type != stim::DemInstructionType::DEM_ERROR) {
    throw std::invalid_argument("DEM instruction must be an error.");
  }

  std::vector<ErrorGroup> groups;
  instruction.for_separated_targets([&](std::span<const stim::DemTarget> raw_targets) {
    ErrorGroup group;
    std::vector<int> raw_detectors;
    std::vector<int> raw_observables;
    for (const auto& target : raw_targets) {
      group.targets.push_back(target);
      if (target.is_relative_detector_id()) {
        raw_detectors.push_back(target.val());
      } else if (target.is_observable_id()) {
        raw_observables.push_back(target.val());
      }
    }
    group.detectors = reduce_symmetric_difference(raw_detectors);
    group.observables = reduce_symmetric_difference(raw_observables);

    if (group.detectors.empty()) {
      throw std::invalid_argument("Error instruction `" + instruction.str() +
                                  "` contains a detectorless decomposition group, which cannot "
                                  "be assigned to a component.");
    }
    std::set<int> components;
    for (int detector : group.detectors) {
      components.insert(component_for_detector(detector, detector_components));
    }
    if (components.size() != 1 && !allow_undecomposed_mixed_group) {
      throw std::invalid_argument("Error instruction `" + instruction.str() +
                                  "` contains a decomposition group with detectors from multiple "
                                  "components.");
    }
    group.component = components.size() == 1 ? *components.begin() : -1;
    groups.push_back(std::move(group));
  });
  return groups;
}

void generate_obs_combinations(
    const std::vector<std::set<std::vector<int>>>& obs_options_by_component,
    std::vector<std::vector<int>>& current_combination,
    std::vector<std::vector<std::vector<int>>>& all_combinations, size_t component_index) {
  if (component_index == obs_options_by_component.size()) {
    all_combinations.push_back(current_combination);
    return;
  }
  for (const auto& obs_option : obs_options_by_component[component_index]) {
    current_combination.push_back(obs_option);
    generate_obs_combinations(obs_options_by_component, current_combination, all_combinations,
                              component_index + 1);
    current_combination.pop_back();
  }
}

std::vector<std::vector<int>> get_component_obs_matching_undecomposed_obs(
    const std::vector<std::set<std::vector<int>>>& obs_options_by_component,
    const std::vector<int>& error_obs, int num_missing_components) {
  if (num_missing_components > 1) {
    return {};
  }

  std::vector<std::vector<std::vector<int>>> all_combinations;
  std::vector<std::vector<int>> current_combination;
  generate_obs_combinations(obs_options_by_component, current_combination, all_combinations, 0);

  std::vector<int> reduced_error_obs = reduce_symmetric_difference(error_obs);
  for (const auto& combination : all_combinations) {
    std::vector<int> residual_input = reduced_error_obs;
    std::vector<int> known_obs = reduce_set_symmetric_difference(combination);
    residual_input.insert(residual_input.end(), known_obs.begin(), known_obs.end());
    std::vector<int> residual = reduce_symmetric_difference(residual_input);

    if (residual.empty()) {
      std::vector<std::vector<int>> result = combination;
      result.resize(result.size() + num_missing_components);
      return result;
    }
    if (num_missing_components == 1) {
      std::vector<std::vector<int>> result = combination;
      result.push_back(std::move(residual));
      return result;
    }
  }
  return {};
}

}  // namespace

stim::DetectorErrorModel decompose_errors_using_detector_assignment(
    const stim::DetectorErrorModel& dem, const std::vector<int>& detector_components) {
  stim::DetectorErrorModel flattened = dem.flattened();
  validate_detector_components(flattened, detector_components);

  std::map<std::vector<int>, std::set<std::vector<int>>> single_component_dets_to_obs;
  for (const auto& instruction : flattened.instructions) {
    if (instruction.type != stim::DemInstructionType::DEM_ERROR) {
      continue;
    }
    bool decomposed = has_separator(instruction);
    auto groups = parse_and_validate_groups(instruction, detector_components, !decomposed);
    if (decomposed) {
      for (const auto& group : groups) {
        single_component_dets_to_obs[group.detectors].insert(group.observables);
      }
    } else if (groups[0].component >= 0) {
      single_component_dets_to_obs[groups[0].detectors].insert(groups[0].observables);
    }
  }

  stim::DetectorErrorModel output_dem;
  for (const auto& instruction : flattened.instructions) {
    if (instruction.type != stim::DemInstructionType::DEM_ERROR) {
      output_dem.append_dem_instruction(instruction);
      continue;
    }

    bool decomposed = has_separator(instruction);
    auto groups = parse_and_validate_groups(instruction, detector_components, !decomposed);
    if (decomposed) {
      output_dem.append_dem_instruction(instruction);
      continue;
    }

    const auto& undecomposed = groups[0];
    std::map<int, std::vector<int>> dets_by_component_id;
    for (int detector : undecomposed.detectors) {
      dets_by_component_id[component_for_detector(detector, detector_components)].push_back(
          detector);
    }
    if (dets_by_component_id.size() == 1) {
      output_dem.append_dem_instruction(instruction);
      continue;
    }

    std::vector<std::vector<int>> known_component_dets;
    std::vector<std::set<std::vector<int>>> known_component_obs_options;
    std::vector<std::vector<int>> missing_component_dets;
    for (auto& entry : dets_by_component_id) {
      auto& component_dets = entry.second;
      std::sort(component_dets.begin(), component_dets.end());
      auto known = single_component_dets_to_obs.find(component_dets);
      if (known != single_component_dets_to_obs.end()) {
        known_component_dets.push_back(component_dets);
        known_component_obs_options.push_back(known->second);
      } else {
        missing_component_dets.push_back(component_dets);
      }
    }

    auto component_observables = get_component_obs_matching_undecomposed_obs(
        known_component_obs_options, undecomposed.observables,
        static_cast<int>(missing_component_dets.size()));
    if (component_observables.empty()) {
      throw std::invalid_argument("Error instruction `" + instruction.str() +
                                  "` could not be decomposed consistently.");
    }

    std::vector<std::vector<int>> component_dets = known_component_dets;
    component_dets.insert(component_dets.end(), missing_component_dets.begin(),
                          missing_component_dets.end());
    std::vector<stim::DemTarget> targets;
    for (size_t k = 0; k < component_dets.size(); ++k) {
      for (int detector : component_dets[k]) {
        targets.push_back(stim::DemTarget::relative_detector_id(detector));
      }
      for (int observable : component_observables[k]) {
        targets.push_back(stim::DemTarget::observable_id(observable));
      }
      if (k + 1 < component_dets.size()) {
        targets.push_back(stim::DemTarget::separator());
      }
    }
    output_dem.append_error_instruction(instruction.arg_data[0], targets, instruction.tag);
  }
  return output_dem;
}

std::map<int, stim::DetectorErrorModel> split_dem_by_component(
    const stim::DetectorErrorModel& dem, const std::vector<int>& detector_components) {
  stim::DetectorErrorModel flattened = dem.flattened();
  validate_detector_components(flattened, detector_components);

  std::map<int, stim::DetectorErrorModel> component_dems;
  for (int component : detector_components) {
    component_dems.try_emplace(component);
  }

  for (const auto& instruction : flattened.instructions) {
    if (instruction.type != stim::DemInstructionType::DEM_ERROR) {
      for (auto& entry : component_dems) {
        entry.second.append_dem_instruction(instruction);
      }
      continue;
    }

    auto groups = parse_and_validate_groups(instruction, detector_components, false);
    std::map<int, std::vector<std::vector<stim::DemTarget>>> groups_by_component;
    for (auto& group : groups) {
      groups_by_component[group.component].push_back(std::move(group.targets));
    }

    for (const auto& [component, component_groups] : groups_by_component) {
      std::vector<stim::DemTarget> targets;
      std::vector<int> combined_detectors;
      for (size_t k = 0; k < component_groups.size(); ++k) {
        targets.insert(targets.end(), component_groups[k].begin(), component_groups[k].end());
        for (const auto& target : component_groups[k]) {
          if (target.is_relative_detector_id()) {
            combined_detectors.push_back(target.val());
          }
        }
        if (k + 1 < component_groups.size()) {
          targets.push_back(stim::DemTarget::separator());
        }
      }
      if (reduce_symmetric_difference(combined_detectors).empty()) {
        throw std::invalid_argument("Error instruction `" + instruction.str() +
                                    "` has a detectorless component symptom after combining its "
                                    "decomposition groups.");
      }
      component_dems.at(component).append_error_instruction(instruction.arg_data[0], targets,
                                                            instruction.tag);
    }
  }
  return component_dems;
}

}  // namespace tesseract_decoder
