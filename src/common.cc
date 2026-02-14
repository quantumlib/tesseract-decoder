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

#include "common.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

std::string vector_to_string(const std::vector<int>& vec) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    ss << vec[i];
    if (i < vec.size() - 1) {
      ss << " ";
    }
  }

  ss << "]";
  return ss.str();
}

std::string common::Symptom::str() const {
  std::string s = "Symptom{detectors=";
  s += vector_to_string(detectors);
  s += ", observables=";
  s += vector_to_string(observables);
  s += "}";
  return s;
}

common::Error::Error(const stim::DemInstruction& error) {
  if (error.type != stim::DemInstructionType::DEM_ERROR) {
    throw std::invalid_argument(
        "Error must be loaded from an error dem instruction, but received: " + error.str());
  }
  double probability = error.arg_data[0];
  if (probability < 0 || probability > 1) {
    throw std::invalid_argument("Probability must be between 0 and 1, but received: " +
                                std::to_string(probability));
  }

  std::set<int> detectors_set;
  std::set<int> observables_set;

  for (const stim::DemTarget& target : error.target_data) {
    if (target.is_observable_id()) {
      if (observables_set.find(target.val()) != observables_set.end()) {
        observables_set.erase(target.val());
      } else {
        observables_set.insert(target.val());
      }
    } else if (target.is_relative_detector_id()) {
      if (detectors_set.find(target.val()) != detectors_set.end()) {
        detectors_set.erase(target.val());
      } else {
        detectors_set.insert(target.val());
      }
    }
  }
  // Detectors in the set are already sorted order, which we need so that there
  // is a unique canonical representative for each set of detectors.
  std::vector<int> detectors(detectors_set.begin(), detectors_set.end());
  std::vector<int> observables(observables_set.begin(), observables_set.end());
  likelihood_cost = -1 * std::log(probability / (1 - probability));
  symptom.detectors = detectors;
  symptom.observables = observables;
}

std::string common::Error::str() const {
  std::stringstream ss;
  ss << std::fixed << std::setprecision(6) << likelihood_cost;
  return "Error{cost=" + ss.str() + ", symptom=" + symptom.str() + "}";
}

double common::Error::get_probability() const {
  return 1.0 / (1.0 + std::exp(likelihood_cost));
}

void common::Error::set_with_probability(double p) {
  if (p <= 0 || p >= 1) {
    throw std::invalid_argument("Probability must be between 0 and 1.");
  }
  likelihood_cost = -std::log(p / (1.0 - p));
}

std::vector<stim::DemTarget> common::Symptom::as_dem_instruction_targets() const {
  std::vector<stim::DemTarget> targets;
  for (int d : detectors) {
    targets.push_back(stim::DemTarget::relative_detector_id(d));
  }
  for (int o : observables) {
    targets.push_back(stim::DemTarget::observable_id(o));
  }
  return targets;
}

double common::merge_weights(double a, double b) {
  auto sgn = std::copysign(1, a) * std::copysign(1, b);
  auto signed_min = sgn * std::min(std::abs(a), std::abs(b));
  return signed_min + std::log(1 + std::exp(-std::abs(a + b))) -
         std::log(1 + std::exp(-std::abs(a - b)));
}

stim::DetectorErrorModel common::merge_indistinguishable_errors(
    const stim::DetectorErrorModel& dem, std::vector<size_t>& original_indices) {
  stim::DetectorErrorModel out_dem;
  std::vector<size_t> new_mapping;

  std::vector<DemErrorRef> merged_errors;
  std::unordered_map<Symptom, size_t, Symptom::hash> symptom_to_merged_idx;

  size_t current_error_idx = 0;
  for (const stim::DemInstruction& instruction : dem.flattened().instructions) {
    switch (instruction.type) {
      case stim::DemInstructionType::DEM_ERROR: {
        Error error(instruction);
        if (error.symptom.detectors.size() == 0) {
          // TODO: For errors without detectors, the observables should be included if p>0.5
          std::cout << "Warning: the circuit has errors that do not flip any detectors \n";
        }
        auto it = symptom_to_merged_idx.find(error.symptom);
        if (it == symptom_to_merged_idx.end()) {
          symptom_to_merged_idx[error.symptom] = merged_errors.size();
          merged_errors.emplace_back(DemErrorRef{error, original_indices.at(current_error_idx)});
        } else {
          merged_errors[it->second].error.likelihood_cost = merge_weights(
              merged_errors[it->second].error.likelihood_cost, error.likelihood_cost);
        }
        current_error_idx++;
        break;
      }
      case stim::DemInstructionType::DEM_DETECTOR: {
        out_dem.append_dem_instruction(instruction);
        break;
      }
      case stim::DemInstructionType::DEM_LOGICAL_OBSERVABLE: {
        out_dem.append_dem_instruction(instruction);
        break;
      }
      default:
        throw std::invalid_argument("Unrecognized instruction type: " + instruction.str());
    }
  }

  for (const auto& me : merged_errors) {
    out_dem.append_error_instruction(me.error.get_probability(),
                                     me.error.symptom.as_dem_instruction_targets(), "");
    new_mapping.push_back(me.index);
  }

  original_indices = std::move(new_mapping);
  return out_dem;
}


stim::DetectorErrorModel common::remove_zero_probability_errors(
    const stim::DetectorErrorModel& dem, std::vector<size_t>& original_indices) {
  stim::DetectorErrorModel out_dem;
  std::vector<size_t> new_mapping;
  size_t current_error_idx = 0;
  for (const stim::DemInstruction& instruction : dem.flattened().instructions) {
    switch (instruction.type) {
      case stim::DemInstructionType::DEM_ERROR:
        if (instruction.arg_data[0] > 0) {
          out_dem.append_dem_instruction(instruction);
          new_mapping.push_back(original_indices.at(current_error_idx));
        }
        current_error_idx++;
        break;
      case stim::DemInstructionType::DEM_DETECTOR:
        out_dem.append_dem_instruction(instruction);
        break;
      case stim::DemInstructionType::DEM_LOGICAL_OBSERVABLE:
        out_dem.append_dem_instruction(instruction);
        break;
      default:
        throw std::invalid_argument("Unrecognized instruction type: " + instruction.str());
    }
  }
  original_indices = std::move(new_mapping);
  return out_dem;
}

stim::DetectorErrorModel common::dem_from_counts(stim::DetectorErrorModel& orig_dem,
                                                 const std::vector<size_t>& error_counts,
                                                 size_t num_shots) {
  if (orig_dem.count_errors() != error_counts.size()) {
    throw std::invalid_argument(
        "Error hits array must be the same size as the number of errors in the "
        "original DEM.");
  }

  stim::DetectorErrorModel out_dem;
  size_t ei = 0;
  for (const stim::DemInstruction& instruction : orig_dem.flattened().instructions) {
    switch (instruction.type) {
      case stim::DemInstructionType::DEM_ERROR: {
        double est_probability = double(error_counts.at(ei)) / double(num_shots);
        out_dem.append_error_instruction(est_probability, instruction.target_data, instruction.tag);
        ++ei;
        break;
      }
      case stim::DemInstructionType::DEM_DETECTOR: {
        out_dem.append_dem_instruction(instruction);
        break;
      }
      case stim::DemInstructionType::DEM_LOGICAL_OBSERVABLE: {
        out_dem.append_dem_instruction(instruction);
        break;
      }
      default:
        throw std::invalid_argument("Unrecognized instruction type: " + instruction.str());
    }
  }
  return out_dem;
}
