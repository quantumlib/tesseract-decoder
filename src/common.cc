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

std::string common::Symptom::str() {
  std::string s = "Symptom{";
  for (size_t d : detectors) {
    s += "D" + std::to_string(d);
    s += " ";
  }
  s += "}";
  return s;
}

common::Error::Error(const stim::DemInstruction& error) {
  if (error.type != stim::DemInstructionType::DEM_ERROR) {
    throw std::invalid_argument(
        "Error must be loaded from an error dem instruction, but received: " + error.str());
  }
  assert(error.type == stim::DemInstructionType::DEM_ERROR);
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

std::string common::Error::str() {
  return "Error{cost=" + std::to_string(likelihood_cost) + ", symptom=" + symptom.str() + "}";
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
    const stim::DetectorErrorModel& dem) {
  stim::DetectorErrorModel out_dem;

  // Map to track the distinct symptoms
  std::unordered_map<Symptom, Error, Symptom::hash> errors_by_symptom;
  for (const stim::DemInstruction& instruction : dem.flattened().instructions) {
    switch (instruction.type) {
      case stim::DemInstructionType::DEM_ERROR: {
        Error error(instruction);
        if (error.symptom.detectors.size() == 0) {
          throw std::invalid_argument("Errors that do not flip any detectors are not supported.");
        }
        error.likelihood_cost =
            -1 * std::log(instruction.arg_data[0] / (1 - instruction.arg_data[0]));

        if (errors_by_symptom.find(error.symptom) != errors_by_symptom.end()) {
          double existing_cost = errors_by_symptom[error.symptom].likelihood_cost;
          error.likelihood_cost = -merge_weights(-existing_cost, -error.likelihood_cost);
        }
        errors_by_symptom[error.symptom] = error;
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
  for (const auto& it : errors_by_symptom) {
    // Recalculate probability from the final merged cost for the output DEM.
    double final_p = 1 / (1 + std::exp(it.second.likelihood_cost));
    out_dem.append_error_instruction(final_p, it.second.symptom.as_dem_instruction_targets(),
                                     /*tag=*/"");
  }
  return out_dem;
}

stim::DetectorErrorModel common::remove_zero_probability_errors(
    const stim::DetectorErrorModel& dem) {
  stim::DetectorErrorModel out_dem;
  for (const stim::DemInstruction& instruction : dem.flattened().instructions) {
    switch (instruction.type) {
      case stim::DemInstructionType::DEM_ERROR:
        if (instruction.arg_data[0] > 0) {
          out_dem.append_dem_instruction(instruction);
        }
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

  for (const stim::DemInstruction& instruction : orig_dem.flattened().instructions) {
    if (instruction.type == stim::DemInstructionType::DEM_ERROR && instruction.arg_data[0] == 0) {
      throw std::invalid_argument(
          "dem_from_counts requires DEMs without zero-probability errors. Use"
          " remove_zero_probability_errors first.");
    }
  }

  stim::DetectorErrorModel out_dem;
  size_t ei = 0;
  for (const stim::DemInstruction& instruction : orig_dem.flattened().instructions) {
    switch (instruction.type) {
      case stim::DemInstructionType::DEM_ERROR: {
        double est_probability = double(error_counts.at(ei)) / double(num_shots);
        out_dem.append_error_instruction(est_probability, instruction.target_data, /*tag=*/"");
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
