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

}  // namespace

bool ComponentSymptom::operator==(const ComponentSymptom& other) const {
  return detectors == other.detectors && observables == other.observables;
}

bool ComponentSymptom::operator<(const ComponentSymptom& other) const {
  if (detectors != other.detectors) {
    return detectors < other.detectors;
  }
  return observables < other.observables;
}

CorrelationEvidence collect_correlation_evidence(const stim::DetectorErrorModel& dem,
                                                 const std::vector<int>& detector_components) {
  stim::DetectorErrorModel flattened = dem.flattened();
  if (detector_components.size() != flattened.count_detectors()) {
    throw std::invalid_argument(
        "Detector component assignment count does not match the DEM detector count.");
  }
  for (size_t detector = 0; detector < detector_components.size(); ++detector) {
    if (detector_components[detector] < 0) {
      throw std::invalid_argument("Detector D" + std::to_string(detector) +
                                  " has an invalid negative component assignment.");
    }
  }

  CorrelationEvidence evidence;
  for (const auto& instruction : flattened.instructions) {
    if (instruction.type != stim::DemInstructionType::DEM_ERROR) {
      continue;
    }

    std::map<int, std::pair<std::set<int>, std::set<int>>> symptom_sets_by_component;
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

      std::set<int> components;
      for (int detector : group_detectors) {
        if (detector < 0 || static_cast<size_t>(detector) >= detector_components.size() ||
            detector_components[detector] < 0) {
          throw std::invalid_argument("Invalid component assignment for detector D" +
                                      std::to_string(detector) + '.');
        }
        components.insert(detector_components[detector]);
      }
      if (components.size() != 1) {
        throw std::invalid_argument("Error instruction `" + instruction.str() +
                                    "` contains a decomposition group with detectors from multiple "
                                    "components.");
      }

      auto& [detectors, observables] = symptom_sets_by_component[*components.begin()];
      for (int detector : group_detectors) toggle(detectors, detector);
      for (int observable : group_observables) toggle(observables, observable);
    });

    std::vector<ComponentSymptom> symptoms;
    for (const auto& entry : symptom_sets_by_component) {
      const auto& symptom_sets = entry.second;
      if (symptom_sets.first.empty()) {
        throw std::invalid_argument("Error instruction `" + instruction.str() +
                                    "` has a detectorless component symptom after combining its "
                                    "decomposition groups.");
      }
      symptoms.push_back({{symptom_sets.first.begin(), symptom_sets.first.end()},
                          {symptom_sets.second.begin(), symptom_sets.second.end()}});
    }

    double probability = instruction.arg_data[0];
    for (const auto& symptom : symptoms) {
      xor_probability(evidence.symptom_probabilities[symptom], probability);
    }
    for (size_t i = 0; i < symptoms.size(); ++i) {
      for (size_t j = 0; j < symptoms.size(); ++j) {
        if (i != j) {
          xor_probability(evidence.paired_mechanism_probabilities[symptoms[i]][symptoms[j]],
                          probability);
        }
      }
    }
  }
  return evidence;
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

}  // namespace tesseract_decoder
