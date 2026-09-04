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

#ifndef ERROR_CORRELATIONS_H
#define ERROR_CORRELATIONS_H

#include <map>
#include <vector>

#include "dem_decomposition.h"
#include "stim.h"

namespace tesseract_decoder {

struct ComponentSymptom {
  std::vector<int> detectors;
  std::vector<int> observables;

  bool operator<(const ComponentSymptom& other) const;
};

/** A correlated-matching-style probability used to reweight an affected symptom. */
struct ReweightProbability {
  ComponentSymptom affected_symptom;
  double probability;
};

/**
 * Evidence used by the multipass reweighting heuristic.
 *
 * `symptom_probabilities` tracks the XOR probability of every component
 * symptom. `paired_mechanism_probabilities[a][b]` tracks only error mechanisms
 * that contain both symptoms. It deliberately omits combinations of independent
 * one-sided mechanisms, so it is not generally the joint probability P(a and b).
 */
struct CorrelationEvidence {
  std::map<ComponentSymptom, double> symptom_probabilities;
  std::map<ComponentSymptom, std::map<ComponentSymptom, double>> paired_mechanism_probabilities;
};

using ReweightProbsMap = std::map<ComponentSymptom, std::vector<ReweightProbability>>;

CorrelationEvidence collect_correlation_evidence(const stim::DetectorErrorModel& dem,
                                                 const std::vector<int>& detector_components);

/**
 * Forms the heuristic ratio paired_mechanism_probability / symptom_probability.
 *
 * This ratio is useful for correlated matching-style reweighting but is not, in
 * general, an exact conditional probability.
 */
ReweightProbsMap derive_reweight_probabilities(const CorrelationEvidence& evidence);

ReweightProbsMap process_dem_correlations(const stim::DetectorErrorModel& dem,
                                          const std::vector<int>& detector_components);

// Uses an already-flattened, validated, and decomposed two-component DEM.
ReweightProbsMap process_dem_correlations(const TwoComponentDem& dem);

}  // namespace tesseract_decoder

#endif  // ERROR_CORRELATIONS_H
