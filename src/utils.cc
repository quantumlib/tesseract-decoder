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
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <queue>
#include <random>
#include <string>

#include "common.h"
#include "stim.h"

static std::invalid_argument gari_layout_error(const std::string& path, const std::string& detail) {
  return std::invalid_argument("Invalid GARI layout '" + path + "': " + detail);
}

static size_t read_gari_layout_size(const nlohmann::json& value, const std::string& path) {
  if (!value.is_number_integer()) {
    throw gari_layout_error(path, "detector counts and mapping entries must be integers.");
  }
  if (value.is_number_unsigned()) {
    uint64_t result = value.get<uint64_t>();
    if (result > std::numeric_limits<size_t>::max()) {
      throw gari_layout_error(path, "integer is too large.");
    }
    return static_cast<size_t>(result);
  }
  int64_t result = value.get<int64_t>();
  if (result < 0) {
    throw gari_layout_error(path, "detector counts and mapping entries must be nonnegative.");
  }
  return static_cast<size_t>(result);
}

GariLayout load_gari_layout(const std::string& path, size_t expected_gari_detector_count) {
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::invalid_argument("Could not open GARI layout: " + path);
  }

  try {
    nlohmann::json document;
    input >> document;
    if (document.at("schema").get<std::string>() != "tesseract.gari_layout.v1") {
      throw gari_layout_error(path, "unsupported schema.");
    }
    if (document.at("detector_order").get<std::string>() != "physical_then_virtual") {
      throw gari_layout_error(path, "unsupported detector order.");
    }

    size_t source_detector_count =
        read_gari_layout_size(document.at("source_detector_count"), path);
    size_t gari_detector_count = read_gari_layout_size(document.at("gari_detector_count"), path);
    const auto& mapping = document.at("source_to_gari");
    if (gari_detector_count != expected_gari_detector_count ||
        source_detector_count > gari_detector_count || !mapping.is_array() ||
        mapping.size() != source_detector_count) {
      throw gari_layout_error(path, "detector counts do not agree with the mapping.");
    }

    GariLayout layout;
    layout.path = path;
    layout.gari_detector_count = gari_detector_count;
    std::vector<bool> used(source_detector_count);
    layout.source_to_gari.reserve(source_detector_count);
    for (const auto& entry : mapping) {
      size_t target = read_gari_layout_size(entry, path);
      if (target >= source_detector_count || used[target]) {
        throw gari_layout_error(path, "source_to_gari must permute the physical detector rows.");
      }
      used[target] = true;
      layout.source_to_gari.push_back(target);
    }
    return layout;
  } catch (const nlohmann::json::exception& ex) {
    throw gari_layout_error(path, ex.what());
  }
}

void GariLayout::map_hits(std::vector<uint64_t>& hits) const {
  for (uint64_t& source : hits) {
    if (source >= source_to_gari.size()) {
      throw gari_layout_error(path, "source detector index is out of range.");
    }
    source = source_to_gari[source];
  }
  std::sort(hits.begin(), hits.end());
}

void GariLayout::map_shots(std::vector<stim::SparseShot>& shots) const {
  for (auto& shot : shots) {
    map_hits(shot.hits);
  }
}

void GariLayout::validate_source(const stim::Circuit& circuit,
                                 const stim::DetectorErrorModel& gari_dem) const {
  if (circuit.count_detectors() != source_detector_count()) {
    throw gari_layout_error(path, "source_detector_count does not match the circuit.");
  }
  if (circuit.count_observables() != gari_dem.count_observables()) {
    throw gari_layout_error(path, "the circuit and DEM observable counts differ.");
  }
}

std::vector<std::vector<double>> get_detector_coords(const stim::DetectorErrorModel& dem) {
  std::vector<std::vector<double>> detector_coords;
  for (const stim::DemInstruction& instruction : dem.flattened().instructions) {
    switch (instruction.type) {
      case stim::DemInstructionType::DEM_SHIFT_DETECTORS:
        throw std::invalid_argument("DEM_SHIFT_DETECTORS is not supported by this function.");
        break;
      case stim::DemInstructionType::DEM_ERROR: {
        break;
      }
      case stim::DemInstructionType::DEM_DETECTOR: {
        std::vector<double> coord;
        for (const double& t : instruction.arg_data) {
          coord.push_back(t);
        }
        detector_coords.push_back(coord);
        break;
      }
      case stim::DemInstructionType::DEM_LOGICAL_OBSERVABLE:
        break;
      default:
        throw std::invalid_argument(
            "Unexpected DemInstructionType found in the detector error model.");
    }
  }
  return detector_coords;
}

std::vector<std::vector<size_t>> build_detector_graph(const stim::DetectorErrorModel& dem) {
  size_t num_detectors = dem.count_detectors();
  std::vector<std::vector<size_t>> neighbors(num_detectors);
  for (const stim::DemInstruction& instruction : dem.flattened().instructions) {
    if (instruction.type != stim::DemInstructionType::DEM_ERROR) {
      continue;
    }
    std::vector<int> dets;
    for (const stim::DemTarget& target : instruction.target_data) {
      if (target.is_relative_detector_id()) {
        dets.push_back(target.val());
      }
    }
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
  std::uniform_int_distribution<size_t> dist_det(0, graph.size() - 1);
  for (size_t det_order = 0; det_order < num_det_orders; ++det_order) {
    std::vector<size_t> perm;
    perm.reserve(graph.size());
    std::vector<char> visited(graph.size(), false);
    std::queue<size_t> q;
    size_t start = dist_det(rng);
    while (perm.size() < graph.size()) {
      if (!visited[start]) {
        visited[start] = true;
        q.push(start);
        perm.push_back(start);
      }
      while (!q.empty()) {
        size_t cur = q.front();
        q.pop();
        auto neigh = graph[cur];
        std::shuffle(neigh.begin(), neigh.end(), rng);
        for (size_t n : neigh) {
          if (!visited[n]) {
            visited[n] = true;
            q.push(n);
            perm.push_back(n);
          }
        }
      }
      if (perm.size() < graph.size()) {
        do {
          start = dist_det(rng);
        } while (visited[start]);
      }
    }
    std::vector<size_t> inv_perm(graph.size());
    for (size_t i = 0; i < perm.size(); ++i) {
      inv_perm[perm[i]] = i;
    }
    det_orders[det_order] = inv_perm;
  }
  return det_orders;
}

static std::vector<std::vector<size_t>> build_det_orders_coordinate(
    const stim::DetectorErrorModel& dem, size_t num_det_orders, std::mt19937_64& rng) {
  std::vector<std::vector<size_t>> det_orders(num_det_orders);
  auto detector_coords = get_detector_coords(dem);
  std::vector<double> inner_products(dem.count_detectors());
  std::normal_distribution<double> dist(0, 1);
  if (detector_coords.empty() || detector_coords.at(0).empty()) {
    for (size_t det_order = 0; det_order < num_det_orders; ++det_order) {
      det_orders[det_order].resize(dem.count_detectors());
      std::iota(det_orders[det_order].begin(), det_orders[det_order].end(), 0);
    }
    return det_orders;
  }
  for (size_t det_order = 0; det_order < num_det_orders; ++det_order) {
    std::vector<double> orientation_vector;
    for (size_t i = 0; i < detector_coords.at(0).size(); ++i) {
      orientation_vector.push_back(dist(rng));
    }
    for (size_t i = 0; i < detector_coords.size(); ++i) {
      inner_products[i] = 0;
      for (size_t j = 0; j < orientation_vector.size(); ++j) {
        inner_products[i] += detector_coords[i][j] * orientation_vector[j];
      }
    }
    std::vector<size_t> perm(dem.count_detectors());
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](const size_t& i, const size_t& j) {
      return inner_products[i] > inner_products[j];
    });
    std::vector<size_t> inv_perm(dem.count_detectors());
    for (size_t i = 0; i < perm.size(); ++i) {
      inv_perm[perm[i]] = i;
    }
    det_orders[det_order] = inv_perm;
  }
  return det_orders;
}

static std::vector<std::vector<size_t>> build_det_orders_index(const stim::DetectorErrorModel& dem,
                                                               size_t num_det_orders,
                                                               std::mt19937_64& rng) {
  std::vector<std::vector<size_t>> det_orders(num_det_orders);
  std::uniform_int_distribution<int> dist_bool(0, 1);
  size_t n = dem.count_detectors();
  for (size_t det_order = 0; det_order < num_det_orders; ++det_order) {
    det_orders[det_order].resize(n);
    if (dist_bool(rng)) {
      for (size_t i = 0; i < n; ++i) {
        det_orders[det_order][i] = n - 1 - i;
      }
    } else {
      std::iota(det_orders[det_order].begin(), det_orders[det_order].end(), 0);
    }
  }
  return det_orders;
}

std::vector<std::vector<size_t>> build_det_orders(const stim::DetectorErrorModel& dem,
                                                  size_t num_det_orders, DetOrder method,
                                                  uint64_t seed) {
  std::mt19937_64 rng(seed);
  switch (method) {
    case DetOrder::DetBFS:
      return build_det_orders_bfs(dem, num_det_orders, rng);
    case DetOrder::DetCoordinate:
      return build_det_orders_coordinate(dem, num_det_orders, rng);
    case DetOrder::DetIndex:
      return build_det_orders_index(dem, num_det_orders, rng);
  }
  throw std::invalid_argument("Unknown det order method");
}

std::vector<std::vector<size_t>> build_gari_detector_orders(const stim::Circuit& source_circuit,
                                                            const GariLayout& layout,
                                                            size_t num_det_orders, DetOrder method,
                                                            uint64_t seed) {
  if (num_det_orders == 0) {
    return {};
  }
  if (source_circuit.count_detectors() != layout.source_detector_count()) {
    throw gari_layout_error(layout.path,
                            "source_detector_count does not match the ordering circuit.");
  }
  if (layout.source_detector_count() > layout.gari_detector_count) {
    throw gari_layout_error(layout.path, "source detector count exceeds GARI detector count.");
  }

  std::vector<std::vector<size_t>> source_orders;
  if (layout.source_detector_count() == 0) {
    source_orders.resize(num_det_orders);
  } else {
    stim::DetectorErrorModel source_dem = stim::ErrorAnalyzer::circuit_to_detector_error_model(
        source_circuit, /*decompose_errors=*/false, /*fold_loops=*/true,
        /*allow_gauge_detectors=*/true,
        /*approximate_disjoint_errors_threshold=*/1,
        /*ignore_decomposition_failures=*/false,
        /*block_decomposition_from_introducing_remnant_edges=*/false);
    source_orders = build_det_orders(source_dem, num_det_orders, method, seed);
  }

  std::vector<std::vector<size_t>> gari_orders;
  gari_orders.reserve(source_orders.size());
  for (const auto& source_order : source_orders) {
    if (source_order.size() != layout.source_detector_count()) {
      throw gari_layout_error(layout.path, "source detector order has the wrong size.");
    }
    std::vector<size_t> gari_order;
    gari_order.reserve(layout.gari_detector_count);
    // Existing orders contain source detector IDs. Map those IDs to physical
    // GARI rows, then visit the virtual rows in their natural order.
    for (size_t source : source_order) {
      gari_order.push_back(layout.source_to_gari.at(source));
    }
    for (size_t detector = layout.source_detector_count(); detector < layout.gari_detector_count;
         ++detector) {
      gari_order.push_back(detector);
    }
    gari_orders.push_back(std::move(gari_order));
  }
  return gari_orders;
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
