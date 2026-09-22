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

#ifndef DEM_DECOMPOSITION_H
#define DEM_DECOMPOSITION_H

#include <array>
#include <vector>

#include "stim.h"

namespace tesseract_decoder {

struct TwoComponentDem {
  // Component IDs are normalized to 0 and 1. assignment_labels preserves the
  // two caller-supplied labels in ascending order.
  std::array<int, 2> assignment_labels;
  std::vector<int> detector_components;
  stim::DetectorErrorModel decomposed_dem;
  std::array<stim::DetectorErrorModel, 2> component_dems;
};

/**
 * Validates, decomposes, and splits a DEM assigned to exactly two components.
 *
 * The DEM is flattened and the assignment is validated once. This is the
 * normal preprocessing entry point for native multipass decoding.
 */
TwoComponentDem prepare_two_component_dem(const stim::DetectorErrorModel& dem,
                                          const std::vector<int>& detector_components);

/**
 * Decomposes undecomposed errors using an explicit detector-to-component assignment.
 *
 * Existing Stim `^` decomposition groups are preserved after validation. Every
 * existing group must contain detectors from exactly one component. Detectorless
 * groups are rejected because they cannot be assigned unambiguously.
 */
stim::DetectorErrorModel decompose_errors_using_detector_assignment(
    const stim::DetectorErrorModel& dem, const std::vector<int>& detector_components);

}  // namespace tesseract_decoder

#endif  // DEM_DECOMPOSITION_H
