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
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "common.h"
#include "stim.h"

constexpr const double EPSILON = 1e-7;

std::vector<std::vector<double>> get_detector_coords(const stim::DetectorErrorModel& dem);

// Builds an adjacency list graph where two detectors share an edge iff an error
// in the model activates them both.
std::vector<std::vector<size_t>> build_detector_graph(const stim::DetectorErrorModel& dem);

std::vector<std::vector<size_t>> build_det_orders(const stim::DetectorErrorModel& dem,
                                                  size_t num_det_orders, bool det_order_bfs = true,
                                                  uint64_t seed = 0);

const double INF = std::numeric_limits<double>::infinity();

bool sampling_from_dem(uint64_t seed, size_t num_shots, stim::DetectorErrorModel dem,
                       std::vector<stim::SparseShot>& shots);

void sample_shots(uint64_t sample_seed, stim::Circuit& circuit, size_t sample_num_shots,
                  std::vector<stim::SparseShot>& shots);

std::vector<common::Error> get_errors_from_dem(const stim::DetectorErrorModel& dem);

std::vector<std::string> get_files_recursive(const std::string& directory_path);

uint64_t vector_to_u64_mask(const std::vector<int>& v);

struct CallbackStream {
  bool active = false;
  std::function<void(const std::string&)> callback;
  std::stringstream buffer;

  CallbackStream() = default;
  CallbackStream(bool active_, std::function<void(const std::string&)> cb)
      : active(active_), callback(std::move(cb)) {}

  CallbackStream(const CallbackStream& other)
      : active(other.active), callback(other.callback) {}
  CallbackStream& operator=(const CallbackStream& other) {
    active = other.active;
    callback = other.callback;
    buffer.str("");
    buffer.clear();
    return *this;
  }
  CallbackStream(CallbackStream&&) = default;
  CallbackStream& operator=(CallbackStream&&) = default;

  template <typename T>
  CallbackStream& operator<<(const T& value) {
    if (active) buffer << value;
    return *this;
  }

  CallbackStream& operator<<(std::ostream& (*manip)(std::ostream&)) {
    if (active) {
      manip(buffer);
      if (manip == static_cast<std::ostream& (*)(std::ostream&)>(std::endl) ||
          manip == static_cast<std::ostream& (*)(std::ostream&)>(std::flush)) {
        flush();
      }
    }
    return *this;
  }

  void flush() {
    if (active && callback && buffer.tellp() > 0) {
      callback(buffer.str());
      buffer.str("");
      buffer.clear();
    }
  }

  ~CallbackStream() { flush(); }
};

#endif  // __TESSERACT_UTILS_H__
