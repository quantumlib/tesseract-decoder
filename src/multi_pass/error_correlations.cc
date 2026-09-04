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

#include "error_correlations.h"

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

void toggle(std::set<int>& values, int value) {
  if (!values.erase(value)) {
    values.insert(value);
  }
}

void xor_probability(double& accumulated, double probability) {
  accumulated = accumulated * (1 - probability) + probability * (1 - accumulated);
}

std::vector<int> normalize_detector_components(const stim::DetectorErrorModel& dem,
                                               const std::vector<int>& detector_components) {
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
  int first_label = *labels.begin();
  std::vector<int> normalized;
  normalized.reserve(detector_components.size());
  for (int component : detector_components) {
    normalized.push_back(component == first_label ? 0 : 1);
  }
  return normalized;
}

CorrelationEvidence collect_correlation_evidence_from_flattened_dem(
    const stim::DetectorErrorModel& dem, const std::vector<int>& detector_components) {
  CorrelationEvidence evidence;
  for (const auto& instruction : dem.instructions) {
    if (instruction.type != stim::DemInstructionType::DEM_ERROR) {
      continue;
    }

    std::array<std::pair<std::set<int>, std::set<int>>, 2> symptom_sets_by_component;
    std::array<bool, 2> has_component_symptom{};
    instruction.for_separated_targets([&](std::span<const stim::DemTarget> group) {
      std::set<int> group_detectors;
      std::set<int> group_observables;
      for (const auto& target : group) {
        if (target.is_relative_detector_id()) {
          toggle(group_detectors, target.val());
        } else if (target.is_observable_id()) {
          toggle(group_observables, target.val());
        }
      }
      if (group_detectors.empty()) {
        throw std::invalid_argument("Error instruction `" + instruction.str() +
                                    "` contains a detectorless decomposition group.");
      }

      int component = -1;
      for (int detector : group_detectors) {
        if (detector < 0 || static_cast<size_t>(detector) >= detector_components.size()) {
          throw std::invalid_argument("Invalid component assignment for detector D" +
                                      std::to_string(detector) + '.');
        }
        if (component == -1) {
          component = detector_components[detector];
        } else if (component != detector_components[detector]) {
          throw std::invalid_argument(
              "Error instruction `" + instruction.str() +
              "` contains a decomposition group with detectors from multiple components.");
        }
      }

      has_component_symptom[component] = true;
      auto& [detectors, observables] = symptom_sets_by_component[component];
      for (int detector : group_detectors) toggle(detectors, detector);
      for (int observable : group_observables) toggle(observables, observable);
    });

    std::array<ComponentSymptom, 2> symptoms;
    for (size_t component = 0; component < 2; ++component) {
      if (!has_component_symptom[component]) continue;
      const auto& [detectors, observables] = symptom_sets_by_component[component];
      if (detectors.empty()) {
        throw std::invalid_argument("Error instruction `" + instruction.str() +
                                    "` has a detectorless component symptom after combining its "
                                    "decomposition groups.");
      }
      symptoms[component] = {{detectors.begin(), detectors.end()},
                             {observables.begin(), observables.end()}};
    }

    double probability = instruction.arg_data[0];
    for (size_t component = 0; component < 2; ++component) {
      if (has_component_symptom[component]) {
        xor_probability(evidence.symptom_probabilities[symptoms[component]], probability);
      }
    }
    if (has_component_symptom[0] && has_component_symptom[1]) {
      xor_probability(evidence.paired_mechanism_probabilities[symptoms[0]][symptoms[1]],
                      probability);
      xor_probability(evidence.paired_mechanism_probabilities[symptoms[1]][symptoms[0]],
                      probability);
    }
  }
  return evidence;
}

}  // namespace

bool ComponentSymptom::operator<(const ComponentSymptom& other) const {
  if (detectors != other.detectors) {
    return detectors < other.detectors;
  }
  return observables < other.observables;
}

CorrelationEvidence collect_correlation_evidence(const stim::DetectorErrorModel& dem,
                                                 const std::vector<int>& detector_components) {
  stim::DetectorErrorModel flattened = dem.flattened();
  std::vector<int> normalized = normalize_detector_components(flattened, detector_components);
  return collect_correlation_evidence_from_flattened_dem(flattened, normalized);
}

ReweightProbsMap derive_reweight_probabilities(const CorrelationEvidence& evidence) {
  ReweightProbsMap reweight_probabilities;
  for (const auto& [causal, affected_probabilities] : evidence.paired_mechanism_probabilities) {
    auto marginal = evidence.symptom_probabilities.find(causal);
    if (marginal == evidence.symptom_probabilities.end() || marginal->second <= 0 ||
        marginal->second >= 1) {
      continue;
    }
    for (const auto& [affected, paired_probability] : affected_probabilities) {
      double probability = std::clamp(paired_probability / marginal->second, 0.0, 1.0);
      reweight_probabilities[causal].push_back({affected, probability});
    }
  }
  return reweight_probabilities;
}

ReweightProbsMap process_dem_correlations(const stim::DetectorErrorModel& dem,
                                          const std::vector<int>& detector_components) {
  return derive_reweight_probabilities(collect_correlation_evidence(dem, detector_components));
}

ReweightProbsMap process_dem_correlations(const TwoComponentDem& dem) {
  return derive_reweight_probabilities(
      collect_correlation_evidence_from_flattened_dem(dem.decomposed_dem, dem.detector_components));
}

}  // namespace tesseract_decoder
