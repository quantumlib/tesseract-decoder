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

#include <map>
#include <vector>

#include "stim.h"

namespace tesseract_decoder {

/**
 * Decomposes undecomposed errors using an explicit detector-to-component assignment.
 *
 * Existing Stim `^` decomposition groups are preserved after validation. Every
 * existing group must contain detectors from exactly one component. Detectorless
 * groups are rejected because they cannot be assigned unambiguously.
 */
stim::DetectorErrorModel decompose_errors_using_detector_assignment(
    const stim::DetectorErrorModel& dem, const std::vector<int>& detector_components);

/**
 * Splits a decomposed DEM into one DEM per explicitly assigned component.
 *
 * All groups from one physical error mechanism that belong to the same component
 * remain in one tagged error instruction.
 */
std::map<int, stim::DetectorErrorModel> split_dem_by_component(
    const stim::DetectorErrorModel& dem, const std::vector<int>& detector_components);

}  // namespace tesseract_decoder

#endif  // DEM_DECOMPOSITION_H
