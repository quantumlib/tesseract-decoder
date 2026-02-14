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

#ifndef SIMPLEX_HPP
#define SIMPLEX_HPP
#include <unordered_set>
#include <vector>

#include "common.h"
#include "stim.h"

struct HighsModel;
struct Highs;
enum class HighsStatus;

struct SimplexConfig {
  stim::DetectorErrorModel dem;
  bool parallelize = false;
  size_t window_length = 0;
  size_t window_slide_length = 0;
  bool verbose = false;
  bool merge_errors = true;
  bool windowing_enabled() {
    return (window_length != 0);
  }
  std::string str();
};

struct SimplexDecoder {
  SimplexConfig config;
  // The list of errors used internally by the decoder (merged/filtered).
  // These correspond to the indices in predicted_errors_buffer.
  std::vector<common::Error> errors;
  size_t num_detectors = 0;
  size_t num_observables = 0;
  // A buffer containing the predicted errors from the most recent decode
  // operation. Indices correspond to the internal optimized 'errors' vector.
  std::vector<size_t> predicted_errors_buffer;
  std::vector<std::vector<int>> error_masks;
  std::vector<std::vector<size_t>> start_time_to_errors;
  std::vector<std::vector<size_t>> end_time_to_errors;

  // Maps internal optimized error index to the original flattened DEM's error index.
  std::vector<size_t> original_error_indices;
  // Maps original error index to internal optimized error index.
  std::vector<size_t> original_to_internal;
  stim::DetectorErrorModel original_dem;

  // Returns the index of this error in the original flattened DEM.
  size_t get_original_error_index(size_t internal_index) const {
    return original_error_indices.at(internal_index);
  }

  // Returns the internal index of this error if it is the representative,
  // or size_t::max() if it was merged or removed.
  size_t get_internal_error_index(size_t original_index) const {
    return original_to_internal.at(original_index);
  }

  std::unique_ptr<HighsModel> model;
  std::unique_ptr<Highs> highs;
  std::unique_ptr<HighsStatus> return_status;

  // For consistency with Tesseract, we provide a low confidence flag on Simplex
  // decoder which is always set to false
  const bool low_confidence_flag = false;

  SimplexDecoder(SimplexConfig config);

  // Clears the predicted_errors_buffer and fills it with the decoded errors for
  // these detection events. Indices are internal to the decoder's errors vector.
  void decode_to_errors(const std::vector<uint64_t>& detections);

  // Returns the bitwise XOR of all the observables bitmasks of all errors in
  // the provided buffer (interpreted as internal indices).
  std::vector<int> get_flipped_observables(const std::vector<size_t>& predicted_errors) const;

  // Returns the sum of the likelihood costs (minus-log-likelihood-ratios) of
  // all errors in the provided buffer (interpreted as internal indices).
  double cost_from_errors(const std::vector<size_t>& predicted_errors) const;

  std::vector<int> decode(const std::vector<uint64_t>& detections);

  void decode_shots(std::vector<stim::SparseShot>& shots,
                    std::vector<std::vector<int>>& obs_predicted);

  ~SimplexDecoder();
  void init_ilp();
};

#endif  // SIMPLEX_HPP
