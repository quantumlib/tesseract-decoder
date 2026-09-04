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
#include <iterator>
#include <nlohmann/json.hpp>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
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

static std::vector<size_t> build_detector_order_bfs(const std::vector<std::vector<size_t>>& graph,
                                                    std::mt19937_64& rng) {
  if (graph.empty()) {
    return {};
  }

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
  return detector_at_position;
}

static std::vector<size_t> build_detector_order_coordinate(
    size_t num_detectors, const std::vector<std::vector<double>>& detector_coords,
    size_t num_coordinate_dimensions, std::mt19937_64& rng) {
  std::vector<double> inner_products(num_detectors);
  std::normal_distribution<double> dist(0, 1);
  if (num_coordinate_dimensions == 0) {
    std::vector<size_t> detector_at_position(num_detectors);
    std::iota(detector_at_position.begin(), detector_at_position.end(), 0);
    return detector_at_position;
  }

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
  detector_at_position.reserve(num_detectors);
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
  return detector_at_position;
}

static std::vector<size_t> build_detector_order_index(size_t num_detectors, std::mt19937_64& rng) {
  std::uniform_int_distribution<int> dist_bool(0, 1);
  std::vector<size_t> detector_at_position(num_detectors);
  if (dist_bool(rng)) {
    for (size_t i = 0; i < num_detectors; ++i) {
      detector_at_position[i] = num_detectors - 1 - i;
    }
  } else {
    std::iota(detector_at_position.begin(), detector_at_position.end(), 0);
  }
  return detector_at_position;
}

struct DetectorOrderPreparation {
  size_t num_detectors;
  std::vector<std::vector<size_t>> graph;
  std::vector<std::vector<double>> detector_coords;
  size_t num_coordinate_dimensions = 0;
};

static DetectorOrderPreparation prepare_detector_order_generation(
    const stim::DetectorErrorModel& dem, bool prepare_graph, bool prepare_coordinates) {
  DetectorOrderPreparation preparation{.num_detectors = dem.count_detectors()};
  if (prepare_graph) {
    preparation.graph = build_detector_graph(dem);
  }
  if (prepare_coordinates) {
    preparation.detector_coords = get_detector_coords(dem);
    for (const auto& coordinates : preparation.detector_coords) {
      preparation.num_coordinate_dimensions =
          std::max(preparation.num_coordinate_dimensions, coordinates.size());
    }
  }
  return preparation;
}

static std::vector<size_t> generate_detector_order(const DetectorOrderPreparation& preparation,
                                                   DetectorOrder::Method method, uint64_t seed) {
  std::mt19937_64 rng(seed);
  switch (method) {
    case DetectorOrder::Method::BFS:
      return build_detector_order_bfs(preparation.graph, rng);
    case DetectorOrder::Method::Coordinate:
      return build_detector_order_coordinate(preparation.num_detectors, preparation.detector_coords,
                                             preparation.num_coordinate_dimensions, rng);
    case DetectorOrder::Method::Index:
      return build_detector_order_index(preparation.num_detectors, rng);
    case DetectorOrder::Method::Literal:
      throw std::invalid_argument("Literal detector orders cannot be generated from a DEM.");
  }
  throw std::invalid_argument("Unknown detector order method.");
}

static void validate_detector_order(const std::vector<size_t>& detector_at_position,
                                    size_t num_detectors) {
  if (detector_at_position.size() != num_detectors) {
    throw std::invalid_argument(
        "Detector order has size " + std::to_string(detector_at_position.size()) +
        ", but the detector error model has " + std::to_string(num_detectors) + " detectors.");
  }

  std::vector<char> seen(num_detectors, false);
  for (size_t position = 0; position < detector_at_position.size(); ++position) {
    const size_t detector = detector_at_position[position];
    if (detector >= num_detectors) {
      throw std::invalid_argument("Detector order contains out-of-range detector ID " +
                                  std::to_string(detector) + " at position " +
                                  std::to_string(position) + ".");
    }
    if (seen[detector]) {
      throw std::invalid_argument("Detector order contains detector ID " +
                                  std::to_string(detector) + " more than once.");
    }
    seen[detector] = true;
  }
}

DetectorOrder::DetectorOrder(Method method, uint64_t seed)
    : method(method), seed(seed), resolved(false), order() {
  if (method == Method::Literal) {
    throw std::invalid_argument("A literal DetectorOrder must be constructed from an order.");
  }
}

DetectorOrder::DetectorOrder(std::vector<size_t> order)
    : method(Method::Literal), seed(0), resolved(true), order(std::move(order)) {}

void DetectorOrder::resolve(const stim::DetectorErrorModel& dem) {
  if (method != Method::Literal) {
    const auto preparation =
        prepare_detector_order_generation(dem, method == Method::BFS, method == Method::Coordinate);
    order = generate_detector_order(preparation, method, seed);
    resolved = true;
  }
  validate_detector_order(order, dem.count_detectors());
}

bool DetectorOrder::is_resolved() const {
  return resolved;
}

DetectorOrder::Method DetectorOrder::get_method() const {
  return method;
}

const std::vector<size_t>& DetectorOrder::get_order() const {
  if (!resolved) {
    throw std::logic_error("Detector order has not been resolved against a DEM.");
  }
  return order;
}

void DetectorOrderSources::add_generated(DetectorOrder::Method method) {
  if (method == DetectorOrder::Method::Literal) {
    throw std::invalid_argument("Literal detector orders must be added from a file.");
  }
  sources.push_back(GeneratedSource{method});
}

void DetectorOrderSources::add_file(std::string path) {
  sources.push_back(FileSource{std::move(path)});
}

bool DetectorOrderSources::empty() const {
  return sources.empty();
}

bool DetectorOrderSources::uses_generated_orders() const {
  return std::any_of(sources.begin(), sources.end(), [](const auto& source) {
    return std::holds_alternative<GeneratedSource>(source);
  });
}

std::vector<DetectorOrder> DetectorOrderSources::make_orders(const stim::DetectorErrorModel& dem,
                                                             size_t orders_per_generated_source,
                                                             uint64_t seed) const {
  std::vector<DetectorOrder> result;
  for (const auto& source : sources) {
    std::vector<DetectorOrder> source_orders;
    if (const auto* generated = std::get_if<GeneratedSource>(&source)) {
      source_orders = make_detector_orders(orders_per_generated_source, generated->method, seed);
    } else {
      source_orders = load_detector_orders(std::get<FileSource>(source).path, dem);
    }
    result.insert(result.end(), std::make_move_iterator(source_orders.begin()),
                  std::make_move_iterator(source_orders.end()));
  }
  return result;
}

std::vector<std::string> DetectorOrderSources::file_paths() const {
  std::vector<std::string> result;
  for (const auto& source : sources) {
    if (const auto* file = std::get_if<FileSource>(&source)) {
      result.push_back(file->path);
    }
  }
  return result;
}

std::vector<DetectorOrder> make_detector_orders(size_t num_det_orders, DetectorOrder::Method method,
                                                uint64_t seed) {
  if (method == DetectorOrder::Method::Literal) {
    throw std::invalid_argument("Literal detector orders cannot be generated.");
  }
  if (num_det_orders == 0) {
    throw std::invalid_argument("The number of detector orders must be at least 1.");
  }
  std::vector<DetectorOrder> result;
  result.reserve(num_det_orders);
  for (size_t order_index = 0; order_index < num_det_orders; ++order_index) {
    result.emplace_back(method, seed + order_index);
  }
  return result;
}

std::vector<DetectorOrder> make_literal_detector_orders(
    std::vector<std::vector<size_t>> detector_orders) {
  std::vector<DetectorOrder> result;
  result.reserve(detector_orders.size());
  for (auto& order : detector_orders) {
    result.emplace_back(std::move(order));
  }
  return result;
}

std::vector<DetectorOrder> load_detector_orders(const std::string& path,
                                                const stim::DetectorErrorModel& dem) {
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::invalid_argument("Could not open the file: " + path);
  }
  nlohmann::json orders = nlohmann::json::parse(input, nullptr, false);
  if (orders.is_discarded()) {
    throw std::invalid_argument("Detector orders file is not valid JSON: " + path);
  }
  if (!orders.is_array() || orders.empty()) {
    throw std::invalid_argument("Detector orders file must contain at least one order.");
  }
  for (const auto& order : orders) {
    if (!order.is_array()) {
      throw std::invalid_argument("Each detector order must be an array.");
    }
    for (const auto& detector : order) {
      if (!detector.is_number_unsigned()) {
        throw std::invalid_argument("Detector IDs must be nonnegative integers.");
      }
    }
  }
  auto detector_orders =
      make_literal_detector_orders(orders.get<std::vector<std::vector<size_t>>>());
  resolve_detector_orders(detector_orders, dem);
  return detector_orders;
}

void resolve_detector_orders(std::vector<DetectorOrder>& detector_orders,
                             const stim::DetectorErrorModel& dem) {
  bool prepare_graph = false;
  bool prepare_coordinates = false;
  for (const auto& order : detector_orders) {
    prepare_graph |= order.method == DetectorOrder::Method::BFS;
    prepare_coordinates |= order.method == DetectorOrder::Method::Coordinate;
  }
  const auto preparation =
      prepare_detector_order_generation(dem, prepare_graph, prepare_coordinates);
  for (auto& order : detector_orders) {
    if (order.method != DetectorOrder::Method::Literal) {
      order.order = generate_detector_order(preparation, order.method, order.seed);
      order.resolved = true;
    }
    validate_detector_order(order.order, preparation.num_detectors);
  }
}

std::vector<std::vector<size_t>> build_det_orders(const stim::DetectorErrorModel& dem,
                                                  size_t num_det_orders,
                                                  DetectorOrder::Method method, uint64_t seed) {
  auto detector_orders = make_detector_orders(num_det_orders, method, seed);
  resolve_detector_orders(detector_orders, dem);
  std::vector<std::vector<size_t>> result;
  result.reserve(detector_orders.size());
  for (const auto& order : detector_orders) {
    result.push_back(order.get_order());
  }
  return result;
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
