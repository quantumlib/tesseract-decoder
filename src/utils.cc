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

#include "utils.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <utility>

#include "common.h"
#include "stim.h"

namespace tesseract_decoder {

std::vector<std::vector<double>> get_detector_coords(const stim::DetectorErrorModel& dem) {
  const size_t num_detectors = dem.count_detectors();
  std::vector<std::vector<double>> detector_coords(num_detectors);
  std::vector<char> detector_has_coordinate_instruction(num_detectors, false);
  bool has_any_detector_coordinate_instruction = false;
  for (const stim::DemInstruction& instruction : common::flatten(dem).instructions) {
    switch (instruction.type) {
      case stim::DemInstructionType::DEM_SHIFT_DETECTORS:
        throw std::invalid_argument("DEM_SHIFT_DETECTORS is not supported by this function.");
        break;
      case stim::DemInstructionType::DEM_ERROR: {
        break;
      }
      case stim::DemInstructionType::DEM_DETECTOR: {
        has_any_detector_coordinate_instruction = true;
        const std::vector<double> coord(instruction.arg_data.begin(), instruction.arg_data.end());
        for (const stim::DemTarget& target : instruction.target_data) {
          if (!target.is_relative_detector_id()) {
            continue;
          }
          const size_t detector = target.val();
          if (detector < num_detectors && !detector_has_coordinate_instruction[detector]) {
            detector_coords[detector] = coord;
            detector_has_coordinate_instruction[detector] = true;
          }
        }
        break;
      }
      case stim::DemInstructionType::DEM_LOGICAL_OBSERVABLE:
        break;
      default:
        throw std::invalid_argument(
            "Unexpected DemInstructionType found in the detector error model.");
    }
  }
  if (!has_any_detector_coordinate_instruction) {
    return {};
  }
  return detector_coords;
}

std::vector<std::vector<size_t>> build_detector_graph(const stim::DetectorErrorModel& dem) {
  size_t num_detectors = dem.count_detectors();
  std::vector<std::vector<size_t>> neighbors(num_detectors);
  for (const stim::DemInstruction& instruction : common::flatten(dem).instructions) {
    if (instruction.type != stim::DemInstructionType::DEM_ERROR) {
      continue;
    }
    if (instruction.arg_data[0] < 0) {
      throw std::invalid_argument("Detector error probability must be non-negative.");
    }
    if (instruction.arg_data[0] == 0) {
      continue;
    }
    const common::Error error(instruction);
    const std::vector<int>& dets = error.symptom.detectors;
    for (size_t i = 0; i < dets.size(); ++i) {
      for (size_t j = i + 1; j < dets.size(); ++j) {
        size_t a = dets[i];
        size_t b = dets[j];
        neighbors[a].push_back(b);
        neighbors[b].push_back(a);
      }
    }
  }
  for (auto& neigh : neighbors) {
    std::sort(neigh.begin(), neigh.end());
    neigh.erase(std::unique(neigh.begin(), neigh.end()), neigh.end());
  }
  return neighbors;
}

static std::vector<std::vector<size_t>> build_det_orders_bfs(const stim::DetectorErrorModel& dem,
                                                             size_t num_det_orders,
                                                             std::mt19937_64& rng) {
  std::vector<std::vector<size_t>> det_orders(num_det_orders);
  auto graph = build_detector_graph(dem);
  if (graph.empty()) {
    return det_orders;
  }
  for (size_t order_index = 0; order_index < num_det_orders; ++order_index) {
    std::vector<size_t> detector_at_position;
    detector_at_position.reserve(graph.size());
    std::vector<char> visited(graph.size(), false);
    std::vector<size_t> unvisited_detectors(graph.size());
    std::vector<size_t> unvisited_position(graph.size());
    std::iota(unvisited_detectors.begin(), unvisited_detectors.end(), 0);
    std::iota(unvisited_position.begin(), unvisited_position.end(), 0);

    auto mark_visited = [&](size_t detector) {
      visited[detector] = true;
      const size_t position = unvisited_position[detector];
      const size_t last_detector = unvisited_detectors.back();
      unvisited_detectors[position] = last_detector;
      unvisited_position[last_detector] = position;
      unvisited_detectors.pop_back();
    };

    std::queue<size_t> q;
    while (!unvisited_detectors.empty()) {
      std::uniform_int_distribution<size_t> dist_root(0, unvisited_detectors.size() - 1);
      const size_t start = unvisited_detectors[dist_root(rng)];
      mark_visited(start);
      q.push(start);
      detector_at_position.push_back(start);

      while (!q.empty()) {
        size_t cur = q.front();
        q.pop();
        auto neigh = graph[cur];
        std::shuffle(neigh.begin(), neigh.end(), rng);
        for (size_t n : neigh) {
          if (!visited[n]) {
            mark_visited(n);
            q.push(n);
            detector_at_position.push_back(n);
          }
        }
      }
    }
    det_orders[order_index] = std::move(detector_at_position);
  }
  return det_orders;
}

static std::vector<std::vector<size_t>> build_det_orders_coordinate(
    const stim::DetectorErrorModel& dem, size_t num_det_orders, std::mt19937_64& rng) {
  std::vector<std::vector<size_t>> det_orders(num_det_orders);
  auto detector_coords = get_detector_coords(dem);
  std::vector<double> inner_products(dem.count_detectors());
  std::normal_distribution<double> dist(0, 1);
  size_t num_coordinate_dimensions = 0;
  for (const auto& coord : detector_coords) {
    num_coordinate_dimensions = std::max(num_coordinate_dimensions, coord.size());
  }
  if (num_coordinate_dimensions == 0) {
    for (size_t order_index = 0; order_index < num_det_orders; ++order_index) {
      det_orders[order_index].resize(dem.count_detectors());
      std::iota(det_orders[order_index].begin(), det_orders[order_index].end(), 0);
    }
    return det_orders;
  }
  for (size_t order_index = 0; order_index < num_det_orders; ++order_index) {
    std::vector<double> orientation_vector;
    orientation_vector.reserve(num_coordinate_dimensions);
    for (size_t i = 0; i < num_coordinate_dimensions; ++i) {
      orientation_vector.push_back(dist(rng));
    }
    size_t num_dets = std::min(detector_coords.size(), inner_products.size());
    for (size_t i = 0; i < num_dets; ++i) {
      inner_products[i] = 0;
      for (size_t j = 0; j < detector_coords[i].size(); ++j) {
        inner_products[i] += detector_coords[i][j] * orientation_vector[j];
      }
    }
    std::vector<size_t> detector_at_position;
    detector_at_position.reserve(dem.count_detectors());
    for (size_t detector = 0; detector < detector_coords.size(); ++detector) {
      if (!detector_coords[detector].empty()) {
        detector_at_position.push_back(detector);
      }
    }
    std::stable_sort(
        detector_at_position.begin(), detector_at_position.end(),
        [&](const size_t& i, const size_t& j) { return inner_products[i] > inner_products[j]; });
    for (size_t detector = 0; detector < detector_coords.size(); ++detector) {
      if (detector_coords[detector].empty()) {
        detector_at_position.push_back(detector);
      }
    }
    det_orders[order_index] = std::move(detector_at_position);
  }
  return det_orders;
}

static std::vector<std::vector<size_t>> build_det_orders_index(const stim::DetectorErrorModel& dem,
                                                               size_t num_det_orders,
                                                               std::mt19937_64& rng) {
  std::vector<std::vector<size_t>> det_orders(num_det_orders);
  std::uniform_int_distribution<int> dist_bool(0, 1);
  size_t n = dem.count_detectors();
  for (size_t order_index = 0; order_index < num_det_orders; ++order_index) {
    det_orders[order_index].resize(n);
    if (dist_bool(rng)) {
      for (size_t i = 0; i < n; ++i) {
        det_orders[order_index][i] = n - 1 - i;
      }
    } else {
      std::iota(det_orders[order_index].begin(), det_orders[order_index].end(), 0);
    }
  }
  return det_orders;
}

static std::vector<std::vector<size_t>> generate_detector_orders(
    const stim::DetectorErrorModel& dem, size_t num_det_orders, DetectorOrderMethod method,
    uint64_t seed) {
  std::mt19937_64 rng(seed);
  switch (method) {
    case DetectorOrderMethod::BFS:
      return build_det_orders_bfs(dem, num_det_orders, rng);
    case DetectorOrderMethod::Coordinate:
      return build_det_orders_coordinate(dem, num_det_orders, rng);
    case DetectorOrderMethod::Index:
      return build_det_orders_index(dem, num_det_orders, rng);
    case DetectorOrderMethod::Literal:
      throw std::invalid_argument("Literal detector orders cannot be generated from a DEM.");
  }
  throw std::invalid_argument("Unknown detector order method.");
}

static void validate_detector_order(const std::vector<size_t>& detector_at_position,
                                    size_t num_detectors, size_t order_index) {
  if (detector_at_position.size() != num_detectors) {
    throw std::invalid_argument("Detector order " + std::to_string(order_index) + " has size " +
                                std::to_string(detector_at_position.size()) +
                                ", but the detector error model has " +
                                std::to_string(num_detectors) + " detectors.");
  }

  std::vector<char> seen(num_detectors, false);
  for (size_t position = 0; position < detector_at_position.size(); ++position) {
    const size_t detector = detector_at_position[position];
    if (detector >= num_detectors) {
      throw std::invalid_argument("Detector order " + std::to_string(order_index) +
                                  " contains out-of-range detector ID " + std::to_string(detector) +
                                  " at position " + std::to_string(position) + ".");
    }
    if (seen[detector]) {
      throw std::invalid_argument("Detector order " + std::to_string(order_index) +
                                  " contains detector ID " + std::to_string(detector) +
                                  " more than once.");
    }
    seen[detector] = true;
  }
}

DetectorOrderSpec::DetectorOrderSpec(DetectorOrderMethod method, uint64_t seed)
    : DetectorOrderSpec(method, seed, 0) {}

DetectorOrderSpec::DetectorOrderSpec(DetectorOrderMethod method, uint64_t seed,
                                     size_t sequence_index)
    : method(method),
      seed(seed),
      sequence_index(sequence_index),
      resolved(false),
      detector_order() {
  if (method == DetectorOrderMethod::Literal) {
    throw std::invalid_argument(
        "A literal DetectorOrderSpec must be constructed from a detector order.");
  }
}

DetectorOrderSpec::DetectorOrderSpec(std::vector<size_t> detector_order)
    : method(DetectorOrderMethod::Literal),
      seed(0),
      sequence_index(0),
      resolved(true),
      detector_order(std::move(detector_order)) {}

void DetectorOrderSpec::resolve(const stim::DetectorErrorModel& dem) {
  if (resolved) {
    return;
  }
  auto orders = generate_detector_orders(dem, sequence_index + 1, method, seed);
  detector_order = std::move(orders.back());
  resolved = true;
}

bool DetectorOrderSpec::is_resolved() const {
  return resolved;
}

DetectorOrderMethod DetectorOrderSpec::get_method() const {
  return method;
}

const std::vector<size_t>& DetectorOrderSpec::get_detector_order() const {
  if (!resolved) {
    throw std::logic_error("Detector order has not been resolved against a DEM.");
  }
  return detector_order;
}

std::vector<DetectorOrderSpec> make_detector_order_specs(size_t num_det_orders,
                                                         DetectorOrderMethod method,
                                                         uint64_t seed) {
  if (method == DetectorOrderMethod::Literal) {
    throw std::invalid_argument("Literal detector orders must provide their detector IDs.");
  }
  std::vector<DetectorOrderSpec> result;
  result.reserve(num_det_orders);
  for (size_t order_index = 0; order_index < num_det_orders; ++order_index) {
    result.push_back(DetectorOrderSpec(method, seed, order_index));
  }
  return result;
}

std::vector<DetectorOrderSpec> make_literal_detector_order_specs(
    std::vector<std::vector<size_t>> detector_orders) {
  std::vector<DetectorOrderSpec> result;
  result.reserve(detector_orders.size());
  for (auto& detector_order : detector_orders) {
    result.emplace_back(std::move(detector_order));
  }
  return result;
}

void resolve_detector_order_specs(std::vector<DetectorOrderSpec>& detector_order_specs,
                                  const stim::DetectorErrorModel& dem) {
  bool is_homogeneous_generated_sequence = !detector_order_specs.empty();
  if (is_homogeneous_generated_sequence) {
    const auto& first = detector_order_specs.front();
    is_homogeneous_generated_sequence = !first.resolved && first.sequence_index == 0;
    for (size_t k = 0; is_homogeneous_generated_sequence && k < detector_order_specs.size(); ++k) {
      const auto& spec = detector_order_specs[k];
      is_homogeneous_generated_sequence = !spec.resolved && spec.method == first.method &&
                                          spec.seed == first.seed && spec.sequence_index == k;
    }
  }

  if (is_homogeneous_generated_sequence) {
    auto orders = generate_detector_orders(dem, detector_order_specs.size(),
                                           detector_order_specs.front().method,
                                           detector_order_specs.front().seed);
    for (size_t k = 0; k < detector_order_specs.size(); ++k) {
      detector_order_specs[k].detector_order = std::move(orders[k]);
      detector_order_specs[k].resolved = true;
    }
  } else {
    for (auto& spec : detector_order_specs) {
      spec.resolve(dem);
    }
  }

  for (size_t k = 0; k < detector_order_specs.size(); ++k) {
    validate_detector_order(detector_order_specs[k].get_detector_order(), dem.count_detectors(), k);
  }
}

std::vector<std::vector<size_t>> build_det_orders(const stim::DetectorErrorModel& dem,
                                                  size_t num_det_orders, DetectorOrderMethod method,
                                                  uint64_t seed) {
  return generate_detector_orders(dem, num_det_orders, method, seed);
}

bool sampling_from_dem(uint64_t seed, size_t num_shots, stim::DetectorErrorModel dem,
                       std::vector<stim::SparseShot>& shots) {
  stim::DemSampler<stim::MAX_BITWORD_WIDTH> sampler(dem, std::mt19937_64{seed}, num_shots);
  sampler.resample(false);
  shots.resize(0);
  shots.resize(num_shots);
  for (size_t shot = 0; shot < num_shots; shot++) {
    if (sampler.num_detectors > 0) {
      std::vector<bool> detection_vec(sampler.num_detectors, false);
      size_t stripe = stim::MAX_BITWORD_WIDTH / sampler.num_detectors;
      int det = 0;
      for (size_t i = 0; i < stim::MAX_BITWORD_WIDTH; i++) {
        det ^= (sampler.det_buffer[shot][i]);
        detection_vec[(size_t)i / stripe] = (bool)det;
      }
      for (size_t i = 0; i < sampler.num_detectors; ++i) {
        if (!detection_vec[i]) continue;
        shots[shot].hits.push_back(i);
      }
    }
    if (sampler.num_observables > 0) {
      for (size_t i = 0; i < stim::MAX_BITWORD_WIDTH; i++) {
        shots[shot].obs_mask[i] ^= bool(sampler.obs_buffer[shot][i]);
      }
    }
  }
  return true;
}

void sample_shots(uint64_t sample_seed, stim::Circuit& circuit, size_t sample_num_shots,
                  std::vector<stim::SparseShot>& shots) {
  std::mt19937_64 rng(sample_seed);
  size_t num_detectors = circuit.count_detectors();
  const auto [dets, obs] = stim::sample_batch_detection_events<64>(circuit, sample_num_shots, rng);
  stim::simd_bit_table<64> obs_T = obs.transposed();
  shots.resize(sample_num_shots);
  for (size_t k = 0; k < sample_num_shots; k++) {
    shots[k].obs_mask = obs_T[k];
    for (size_t d = 0; d < num_detectors; d++) {
      if (dets[d][k]) {
        shots[k].hits.push_back(d);
      }
    }
  }
}

std::vector<common::Error> get_errors_from_dem(const stim::DetectorErrorModel& dem) {
  std::vector<common::Error> errors;
  for (const stim::DemInstruction& instruction : dem.instructions) {
    // Ignore zero-probability errors
    if (instruction.type == stim::DemInstructionType::DEM_ERROR and instruction.arg_data[0] > 0)
      errors.emplace_back(instruction);
  }
  return errors;
}

std::vector<std::string> get_files_recursive(const std::string& directory_path) {
  std::vector<std::string> file_paths;
  try {
    for (const auto& entry : std::filesystem::recursive_directory_iterator(directory_path)) {
      if (std::filesystem::is_regular_file(entry)) {
        file_paths.push_back(entry.path().string());
      }
    }
  } catch (const std::filesystem::filesystem_error& ex) {
    std::cerr << "Filesystem error: " << ex.what() << std::endl;
  }
  return file_paths;
}

uint64_t vector_to_u64_mask(const std::vector<int>& v) {
  uint64_t mask = 0;
  for (int i : v) {
    mask ^= (1ULL << i);
  }
  return mask;
}

}  // namespace tesseract_decoder
