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

#ifndef __TESSERACT_UTILS_H__
#define __TESSERACT_UTILS_H__

#include <array>
#include <cassert>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "common.h"
#include "stim.h"

constexpr const double EPSILON = 1e-7;

std::vector<std::vector<double>> get_detector_coords(
    stim::DetectorErrorModel& dem);

const double INF = std::numeric_limits<double>::infinity();

bool sampling_from_dem(uint64_t seed, size_t num_shots,
                       stim::DetectorErrorModel dem,
                       std::vector<stim::SparseShot>& shots);

void sample_shots(uint64_t sample_seed, stim::Circuit& circuit,
                  size_t sample_num_shots,
                  std::vector<stim::SparseShot>& shots);

std::vector<common::Error> get_errors_from_dem(
    const stim::DetectorErrorModel& dem);

std::vector<std::string> get_files_recursive(const std::string& directory_path);

#endif  // __TESSERACT_UTILS_H__
