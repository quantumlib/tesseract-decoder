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
#include <array>
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

struct NormalizedDetectorComponents {
  std::array<int, 2> assignment_labels;
  std::vector<int> detector_components;
};

NormalizedDetectorComponents normalize_detector_components(
    const stim::DetectorErrorModel& dem, const std::vector<int>& detector_components) {
  if (detector_components.size() != dem.count_detectors()) {
    throw std::invalid_argument(
        "Detector component assignment count does not match the DEM detector count.");
  }
  std::set<int> labels;
  for (size_t detector = 0; detector < detector_components.size(); ++detector) {
    if (detector_components[detector] < 0) {
      throw std::invalid_argument("Detector D" + std::to_string(detector) +
                                  " has an invalid negative component assignment.");
    }
    labels.insert(detector_components[detector]);
  }
  if (labels.size() != 2) {
    throw std::invalid_argument("Multi-pass decoding requires exactly 2 detector components; got " +
                                std::to_string(labels.size()) + ".");
  }

  NormalizedDetectorComponents result;
  auto label = labels.begin();
  result.assignment_labels[0] = *label++;
  result.assignment_labels[1] = *label;
  result.detector_components.reserve(detector_components.size());
  for (int component : detector_components) {
    result.detector_components.push_back(component == result.assignment_labels[0] ? 0 : 1);
  }
  return result;
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

struct ObservableAssignmentSearchResult {
  std::array<std::vector<int>, 2> assignment;
  size_t solution_count = 0;
};

ObservableAssignmentSearchResult find_component_obs_matching_undecomposed_obs(
    const std::array<const std::set<std::vector<int>>*, 2>& obs_options_by_component,
    const std::vector<int>& error_obs) {
  ObservableAssignmentSearchResult search_result;
  if (obs_options_by_component[0] == nullptr && obs_options_by_component[1] == nullptr) {
    search_result.solution_count = 2;
    return search_result;
  }

  std::vector<int> reduced_error_obs = reduce_symmetric_difference(error_obs);
  auto record_solution = [&](const std::vector<int>& component_0_obs,
                             const std::vector<int>& component_1_obs) {
    ++search_result.solution_count;
    if (search_result.solution_count == 1) {
      search_result.assignment = {component_0_obs, component_1_obs};
    }
  };

  if (obs_options_by_component[0] == nullptr || obs_options_by_component[1] == nullptr) {
    size_t known_component = obs_options_by_component[0] == nullptr ? 1 : 0;
    size_t missing_component = 1 - known_component;
    for (const auto& known_obs : *obs_options_by_component[known_component]) {
      std::vector<int> residual_input = reduced_error_obs;
      residual_input.insert(residual_input.end(), known_obs.begin(), known_obs.end());
      std::array<std::vector<int>, 2> candidate;
      candidate[known_component] = known_obs;
      candidate[missing_component] = reduce_symmetric_difference(residual_input);
      record_solution(candidate[0], candidate[1]);
      if (search_result.solution_count > 1) break;
    }
    return search_result;
  }

  for (const auto& component_0_obs : *obs_options_by_component[0]) {
    for (const auto& component_1_obs : *obs_options_by_component[1]) {
      std::vector<int> combined_obs = component_0_obs;
      combined_obs.insert(combined_obs.end(), component_1_obs.begin(), component_1_obs.end());
      if (reduce_symmetric_difference(combined_obs) != reduced_error_obs) continue;
      record_solution(component_0_obs, component_1_obs);
      if (search_result.solution_count > 1) return search_result;
    }
  }
  return search_result;
}

stim::DetectorErrorModel decompose_flattened_dem(const stim::DetectorErrorModel& flattened,
                                                 const std::vector<int>& detector_components) {
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
    std::array<std::vector<int>, 2> dets_by_component;
    for (int detector : undecomposed.detectors) {
      dets_by_component[component_for_detector(detector, detector_components)].push_back(detector);
    }
    if (dets_by_component[0].empty() || dets_by_component[1].empty()) {
      output_dem.append_dem_instruction(instruction);
      continue;
    }

    std::array<const std::set<std::vector<int>>*, 2> component_obs_options{};
    for (size_t component = 0; component < 2; ++component) {
      std::sort(dets_by_component[component].begin(), dets_by_component[component].end());
      auto known = single_component_dets_to_obs.find(dets_by_component[component]);
      if (known != single_component_dets_to_obs.end()) {
        component_obs_options[component] = &known->second;
      }
    }

    auto observable_assignment = find_component_obs_matching_undecomposed_obs(
        component_obs_options, undecomposed.observables);
    if (observable_assignment.solution_count == 0) {
      throw std::invalid_argument("Error instruction `" + instruction.str() +
                                  "` has no consistent observable decomposition.");
    }
    if (observable_assignment.solution_count > 1) {
      throw std::invalid_argument(
          "Error instruction `" + instruction.str() +
          "` has multiple consistent observable decompositions; logical observable ownership "
          "is ambiguous.");
    }
    std::vector<stim::DemTarget> targets;
    for (size_t component = 0; component < 2; ++component) {
      for (int detector : dets_by_component[component]) {
        targets.push_back(stim::DemTarget::relative_detector_id(detector));
      }
      for (int observable : observable_assignment.assignment[component]) {
        targets.push_back(stim::DemTarget::observable_id(observable));
      }
      if (component == 0) {
        targets.push_back(stim::DemTarget::separator());
      }
    }
    output_dem.append_error_instruction(instruction.arg_data[0], targets, instruction.tag);
  }
  return output_dem;
}

std::array<stim::DetectorErrorModel, 2> split_flattened_dem_by_component(
    const stim::DetectorErrorModel& dem, const std::vector<int>& detector_components) {
  std::array<stim::DetectorErrorModel, 2> component_dems;
  for (const auto& instruction : dem.instructions) {
    if (instruction.type != stim::DemInstructionType::DEM_ERROR) {
      for (auto& component_dem : component_dems) {
        component_dem.append_dem_instruction(instruction);
      }
      continue;
    }

    auto groups = parse_and_validate_groups(instruction, detector_components, false);
    std::array<std::vector<std::vector<stim::DemTarget>>, 2> groups_by_component;
    for (auto& group : groups) {
      groups_by_component[group.component].push_back(std::move(group.targets));
    }

    for (size_t component = 0; component < 2; ++component) {
      const auto& component_groups = groups_by_component[component];
      if (component_groups.empty()) continue;
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
      component_dems[component].append_error_instruction(instruction.arg_data[0], targets,
                                                         instruction.tag);
    }
  }
  return component_dems;
}

}  // namespace

TwoComponentDem prepare_two_component_dem(const stim::DetectorErrorModel& dem,
                                          const std::vector<int>& detector_components) {
  stim::DetectorErrorModel flattened = dem.flattened();
  auto normalized = normalize_detector_components(flattened, detector_components);
  stim::DetectorErrorModel decomposed =
      decompose_flattened_dem(flattened, normalized.detector_components);
  auto component_dems =
      split_flattened_dem_by_component(decomposed, normalized.detector_components);
  return {normalized.assignment_labels, std::move(normalized.detector_components),
          std::move(decomposed), std::move(component_dems)};
}

stim::DetectorErrorModel decompose_errors_using_detector_assignment(
    const stim::DetectorErrorModel& dem, const std::vector<int>& detector_components) {
  stim::DetectorErrorModel flattened = dem.flattened();
  auto normalized = normalize_detector_components(flattened, detector_components);
  return decompose_flattened_dem(flattened, normalized.detector_components);
}

}  // namespace tesseract_decoder
