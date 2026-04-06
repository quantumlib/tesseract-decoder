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

#include "tesseract_trellis.h"

#include <algorithm>
#include <boost/functional/hash.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "utils.h"

namespace std {
template <>
struct hash<boost::dynamic_bitset<>> {
  size_t operator()(const boost::dynamic_bitset<>& bs) const {
    return boost::hash_value(bs);
  }
};
}  // namespace std

namespace {

struct Fault {
  size_t error_index;
  double log_q;
  double log_p;
  uint64_t obs_mask;
  std::vector<int> detectors;
};

struct LayerFault {
  size_t error_index;
  double log_q;
  double log_p;
  uint64_t obs_mask;
  boost::dynamic_bitset<> local_det_mask;
  boost::dynamic_bitset<> retiring_mask;
  boost::dynamic_bitset<> expected_retiring_bits;
  std::vector<size_t> surviving_local_indices;
};

struct SmallLayerFault {
  size_t error_index;
  double q;
  double p;
  uint64_t obs_flip_bit;
  uint64_t local_det_mask;
  uint64_t retiring_mask;
  uint64_t expected_retiring_bits;
  std::vector<uint8_t> surviving_local_indices;
};

struct PackedMass {
  uint64_t key;
  double mass;
};

struct StateMass {
  uint64_t state;
  double mass;
};

struct ObsAggregate {
  double log_mass = -INF;
};

struct FrontierAggregate {
  double total_log_mass = -INF;
  std::unordered_map<uint64_t, ObsAggregate> obs_entries;
};

double logsumexp2(double a, double b) {
  if (a == -INF) return b;
  if (b == -INF) return a;
  if (a < b) std::swap(a, b);
  return a + std::log1p(std::exp(b - a));
}

void add_obs_mass(FrontierAggregate& aggregate, uint64_t obs_mask, double log_mass) {
  aggregate.total_log_mass = logsumexp2(aggregate.total_log_mass, log_mass);
  auto& obs = aggregate.obs_entries[obs_mask];
  obs.log_mass = logsumexp2(obs.log_mass, log_mass);
}

std::vector<Fault> parse_faults(const std::vector<common::Error>& errors, size_t num_observables) {
  std::vector<Fault> faults;
  faults.reserve(errors.size());
  for (size_t error_index = 0; error_index < errors.size(); ++error_index) {
    const auto& error = errors[error_index];
    const double p = error.get_probability();
    if (p <= 0) continue;
    Fault fault;
    fault.error_index = error_index;
    fault.log_q = std::log1p(-p);
    fault.log_p = std::log(p);
    fault.obs_mask = 0;
    for (int obs : error.symptom.observables) {
      if (obs >= 64) {
        throw std::invalid_argument("tesseract_trellis currently supports at most 64 observables");
      }
      if (size_t(obs) >= num_observables) {
        throw std::invalid_argument("Observable index out of range in DEM");
      }
      fault.obs_mask ^= uint64_t{1} << obs;
    }
    fault.detectors = error.symptom.detectors;
    faults.push_back(std::move(fault));
  }
  return faults;
}

boost::dynamic_bitset<> project_state(const boost::dynamic_bitset<>& state,
                                      const std::vector<size_t>& surviving_local_indices) {
  boost::dynamic_bitset<> out(surviving_local_indices.size());
  for (size_t k = 0; k < surviving_local_indices.size(); ++k) {
    out[k] = state[surviving_local_indices[k]];
  }
  return out;
}

std::vector<LayerFault> build_layer_faults(const std::vector<Fault>& faults, size_t num_detectors,
                                           const std::vector<uint64_t>& detections,
                                           size_t* max_frontier_width_seen) {
  std::vector<size_t> last_seen(num_detectors, std::numeric_limits<size_t>::max());
  for (size_t i = 0; i < faults.size(); ++i) {
    for (int d : faults[i].detectors) {
      last_seen[d] = i;
    }
  }

  boost::dynamic_bitset<> actual_dets(num_detectors);
  for (uint64_t d : detections) {
    if (d >= num_detectors) {
      throw std::runtime_error("Detector index out of range.");
    }
    actual_dets.flip(d);
  }

  std::vector<int> active_detectors;
  active_detectors.reserve(num_detectors);
  std::vector<int> global_to_local(num_detectors, -1);
  std::vector<LayerFault> layers;
  layers.reserve(faults.size());
  *max_frontier_width_seen = 0;

  for (size_t i = 0; i < faults.size(); ++i) {
    for (int d : faults[i].detectors) {
      if (global_to_local[d] == -1) {
        global_to_local[d] = active_detectors.size();
        active_detectors.push_back(d);
      }
    }

    *max_frontier_width_seen = std::max(*max_frontier_width_seen, active_detectors.size());
    LayerFault layer{
        .error_index = faults[i].error_index,
        .log_q = faults[i].log_q,
        .log_p = faults[i].log_p,
        .obs_mask = faults[i].obs_mask,
        .local_det_mask = boost::dynamic_bitset<>(active_detectors.size()),
        .retiring_mask = boost::dynamic_bitset<>(active_detectors.size()),
        .expected_retiring_bits = boost::dynamic_bitset<>(active_detectors.size()),
        .surviving_local_indices = {},
    };

    for (int d : faults[i].detectors) {
      layer.local_det_mask.set(global_to_local[d]);
    }

    for (size_t local = 0; local < active_detectors.size(); ++local) {
      const int d = active_detectors[local];
      if (last_seen[d] == i) {
        layer.retiring_mask.set(local);
        layer.expected_retiring_bits[local] = actual_dets[d];
      } else {
        layer.surviving_local_indices.push_back(local);
      }
    }

    std::vector<int> next_active;
    next_active.reserve(layer.surviving_local_indices.size());
    std::fill(global_to_local.begin(), global_to_local.end(), -1);
    for (size_t next_local = 0; next_local < layer.surviving_local_indices.size(); ++next_local) {
      int d = active_detectors[layer.surviving_local_indices[next_local]];
      global_to_local[d] = next_local;
      next_active.push_back(d);
    }
    active_detectors = std::move(next_active);
    layers.push_back(std::move(layer));
  }
  return layers;
}

bool try_build_small_layer_faults(const std::vector<Fault>& faults, size_t num_detectors,
                                  const std::vector<uint64_t>& detections,
                                  std::vector<SmallLayerFault>* layers,
                                  size_t* max_frontier_width_seen) {
  std::vector<size_t> last_seen(num_detectors, std::numeric_limits<size_t>::max());
  for (size_t i = 0; i < faults.size(); ++i) {
    for (int d : faults[i].detectors) {
      last_seen[d] = i;
    }
  }

  boost::dynamic_bitset<> actual_dets(num_detectors);
  for (uint64_t d : detections) {
    if (d >= num_detectors) {
      throw std::runtime_error("Detector index out of range.");
    }
    actual_dets.flip(d);
  }

  std::vector<int> active_detectors;
  active_detectors.reserve(num_detectors);
  std::vector<int> global_to_local(num_detectors, -1);
  layers->clear();
  layers->reserve(faults.size());
  *max_frontier_width_seen = 0;

  for (size_t i = 0; i < faults.size(); ++i) {
    for (int d : faults[i].detectors) {
      if (global_to_local[d] == -1) {
        global_to_local[d] = active_detectors.size();
        active_detectors.push_back(d);
      }
    }

    *max_frontier_width_seen = std::max(*max_frontier_width_seen, active_detectors.size());
    if (*max_frontier_width_seen > 63) {
      return false;
    }

    SmallLayerFault layer{
        .error_index = faults[i].error_index,
        .q = std::exp(faults[i].log_q),
        .p = std::exp(faults[i].log_p),
        .obs_flip_bit = faults[i].obs_mask & 1,
        .local_det_mask = 0,
        .retiring_mask = 0,
        .expected_retiring_bits = 0,
        .surviving_local_indices = {},
    };
    for (int d : faults[i].detectors) {
      layer.local_det_mask ^= uint64_t{1} << global_to_local[d];
    }
    for (size_t local = 0; local < active_detectors.size(); ++local) {
      const int d = active_detectors[local];
      if (last_seen[d] == i) {
        layer.retiring_mask ^= uint64_t{1} << local;
        if (actual_dets[d]) {
          layer.expected_retiring_bits ^= uint64_t{1} << local;
        }
      } else {
        layer.surviving_local_indices.push_back(local);
      }
    }

    std::vector<int> next_active;
    next_active.reserve(layer.surviving_local_indices.size());
    std::fill(global_to_local.begin(), global_to_local.end(), -1);
    for (size_t next_local = 0; next_local < layer.surviving_local_indices.size(); ++next_local) {
      int d = active_detectors[layer.surviving_local_indices[next_local]];
      global_to_local[d] = next_local;
      next_active.push_back(d);
    }
    active_detectors = std::move(next_active);
    layers->push_back(std::move(layer));
  }

  return true;
}

uint64_t project_small_state(uint64_t state, const std::vector<uint8_t>& surviving_local_indices) {
  uint64_t out = 0;
  for (size_t k = 0; k < surviving_local_indices.size(); ++k) {
    out |= ((state >> surviving_local_indices[k]) & 1ULL) << k;
  }
  return out;
}

uint64_t pack_small_key(uint64_t state, uint64_t obs_flip_bit) {
  return (state << 1) | (obs_flip_bit & 1ULL);
}

uint64_t unpack_small_state(uint64_t packed_key) {
  return packed_key >> 1;
}

uint64_t unpack_small_obs(uint64_t packed_key) {
  return packed_key & 1ULL;
}

void normalize_items(std::vector<PackedMass>& items) {
  double total_mass = 0.0;
  for (const auto& item : items) {
    total_mass += item.mass;
  }
  if (total_mass == 0.0) {
    items.clear();
    return;
  }
  double inv = 1.0 / total_mass;
  for (auto& item : items) {
    item.mass *= inv;
  }
}

std::vector<PackedMass> merge_equal_keys(std::vector<PackedMass>& items) {
  if (items.empty()) {
    return {};
  }
  std::sort(items.begin(), items.end(), [](const PackedMass& a, const PackedMass& b) {
    return a.key < b.key;
  });
  std::vector<PackedMass> merged;
  merged.reserve(items.size());
  uint64_t cur_key = items[0].key;
  double cur_mass = items[0].mass;
  for (size_t i = 1; i < items.size(); ++i) {
    if (items[i].key == cur_key) {
      cur_mass += items[i].mass;
    } else {
      merged.push_back({cur_key, cur_mass});
      cur_key = items[i].key;
      cur_mass = items[i].mass;
    }
  }
  merged.push_back({cur_key, cur_mass});
  return merged;
}

std::vector<StateMass> accumulate_state_masses_from_entries(const std::vector<PackedMass>& entries) {
  std::vector<StateMass> totals;
  if (entries.empty()) {
    return totals;
  }
  totals.reserve(entries.size());
  uint64_t cur_state = unpack_small_state(entries[0].key);
  double cur_mass = entries[0].mass;
  for (size_t i = 1; i < entries.size(); ++i) {
    uint64_t s = unpack_small_state(entries[i].key);
    if (s == cur_state) {
      cur_mass += entries[i].mass;
    } else {
      totals.push_back({cur_state, cur_mass});
      cur_state = s;
      cur_mass = entries[i].mass;
    }
  }
  totals.push_back({cur_state, cur_mass});
  return totals;
}

void keep_top_states(std::vector<PackedMass>& entries, size_t beam_width) {
  if (entries.empty()) {
    return;
  }
  auto totals = accumulate_state_masses_from_entries(entries);
  if (totals.size() <= beam_width) {
    return;
  }
  std::nth_element(totals.begin(), totals.begin() + beam_width, totals.end(),
                   [](const StateMass& a, const StateMass& b) { return a.mass > b.mass; });
  totals.resize(beam_width);
  std::sort(totals.begin(), totals.end(), [](const StateMass& a, const StateMass& b) {
    return a.state < b.state;
  });

  std::vector<PackedMass> kept;
  kept.reserve(entries.size());
  size_t ti = 0;
  for (const auto& item : entries) {
    uint64_t s = unpack_small_state(item.key);
    while (ti < totals.size() && totals[ti].state < s) {
      ++ti;
    }
    if (ti < totals.size() && totals[ti].state == s) {
      kept.push_back(item);
    }
  }
  entries = std::move(kept);
}

void keep_top_branch_entries(std::vector<PackedMass>& entries, size_t beam_width) {
  if (entries.size() <= beam_width) {
    return;
  }
  std::nth_element(entries.begin(), entries.begin() + beam_width, entries.end(),
                   [](const PackedMass& a, const PackedMass& b) { return a.mass > b.mass; });
  entries.resize(beam_width);
}

}  // namespace

TesseractTrellisDecoder::TesseractTrellisDecoder(TesseractTrellisConfig config_)
    : config(std::move(config_)) {
  std::vector<size_t> dem_error_map(config.dem.flattened().count_errors());
  std::iota(dem_error_map.begin(), dem_error_map.end(), 0);
  dem_error_to_error = std::move(dem_error_map);
  error_to_dem_error = common::invert_error_map(dem_error_to_error, config.dem.count_errors());
  errors = get_errors_from_dem(config.dem.flattened());
  num_detectors = config.dem.count_detectors();
  num_observables = config.dem.count_observables();
}

void TesseractTrellisDecoder::decode_shot(const std::vector<uint64_t>& detections) {
  low_confidence_flag = false;
  num_states_expanded = 0;
  num_states_merged = 0;
  max_beam_size_seen = 0;
  max_frontier_width_seen = 0;
  time_expand_seconds = 0;
  time_collapse_seconds = 0;
  time_truncate_seconds = 0;
  time_reconstruct_seconds = 0;
  predicted_obs_mask = 0;
  total_mass_obs0 = 0;
  total_mass_obs1 = 0;

  auto faults = parse_faults(errors, num_observables);

  std::unordered_set<int> all_possible_dets;
  for (const auto& error : errors) {
    for (int d : error.symptom.detectors) {
      all_possible_dets.insert(d);
    }
  }
  for (uint64_t d : detections) {
    if (!all_possible_dets.contains(int(d))) {
      low_confidence_flag = true;
      return;
    }
  }

  std::vector<SmallLayerFault> small_layers;
  if (num_observables <= 1 &&
      try_build_small_layer_faults(faults, num_detectors, detections, &small_layers,
                                   &max_frontier_width_seen)) {
    std::vector<PackedMass> beam_entries;
    beam_entries.push_back({pack_small_key(0, 0), 1.0});
    max_beam_size_seen = 1;

    for (size_t layer_index = 0; layer_index < small_layers.size(); ++layer_index) {
      const auto& layer = small_layers[layer_index];
      auto t0 = std::chrono::high_resolution_clock::now();
      std::vector<PackedMass> expanded_entries;
      expanded_entries.reserve(beam_entries.size() * 2);
      for (const auto& item : beam_entries) {
        ++num_states_expanded;
        uint64_t base_state = unpack_small_state(item.key);
        expanded_entries.push_back({pack_small_key(base_state, unpack_small_obs(item.key)),
                                    item.mass * layer.q});
        uint64_t present_key =
            pack_small_key(base_state ^ layer.local_det_mask,
                           unpack_small_obs(item.key) ^ layer.obs_flip_bit);
        expanded_entries.push_back({present_key, item.mass * layer.p});
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      time_expand_seconds +=
          std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;

      std::vector<PackedMass> next_entries;
      next_entries.reserve(expanded_entries.size());
      for (const auto& item : expanded_entries) {
        uint64_t state = unpack_small_state(item.key);
        if (((state ^ layer.expected_retiring_bits) & layer.retiring_mask) != 0) {
          continue;
        }
        uint64_t projected_state = project_small_state(state, layer.surviving_local_indices);
        uint64_t projected_key = pack_small_key(projected_state, unpack_small_obs(item.key));
        next_entries.push_back({projected_key, item.mass});
      }
      auto t1b = std::chrono::high_resolution_clock::now();
      time_collapse_seconds +=
          std::chrono::duration_cast<std::chrono::microseconds>(t1b - t1).count() / 1e6;

      beam_entries = std::move(next_entries);
      bool at_checkpoint = ((layer_index + 1) % config.merge_interval == 0) ||
                           (layer_index + 1 == small_layers.size());
      if (!at_checkpoint) {
        max_beam_size_seen = std::max(max_beam_size_seen, beam_entries.size());
        if (beam_entries.empty()) {
          low_confidence_flag = true;
          return;
        }
        continue;
      }

      auto t2a = std::chrono::high_resolution_clock::now();
      if (config.prune_mode != TesseractTrellisPruneMode::kNoMerge) {
        beam_entries = merge_equal_keys(beam_entries);
      }
      auto t2 = std::chrono::high_resolution_clock::now();
      time_collapse_seconds +=
          std::chrono::duration_cast<std::chrono::microseconds>(t2 - t2a).count() / 1e6;

      if (config.prune_mode == TesseractTrellisPruneMode::kMergedStates) {
        keep_top_states(beam_entries, config.beam_width);
      } else if (config.prune_mode == TesseractTrellisPruneMode::kBranchEntries ||
                 config.prune_mode == TesseractTrellisPruneMode::kNoMerge) {
        keep_top_branch_entries(beam_entries, config.beam_width);
      }
      normalize_items(beam_entries);
      if (beam_entries.empty()) {
        low_confidence_flag = true;
        return;
      }
      if (config.prune_mode == TesseractTrellisPruneMode::kNoMerge) {
        num_states_merged += beam_entries.size();
        max_beam_size_seen = std::max(max_beam_size_seen, beam_entries.size());
      } else {
        auto post_totals = accumulate_state_masses_from_entries(beam_entries);
        num_states_merged += post_totals.size();
        max_beam_size_seen = std::max(max_beam_size_seen, post_totals.size());
      }
      auto t3 = std::chrono::high_resolution_clock::now();
      time_truncate_seconds +=
          std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / 1e6;
    }

    auto tr0 = std::chrono::high_resolution_clock::now();
    for (const auto& [packed_key, mass] : beam_entries) {
      if (unpack_small_state(packed_key) != 0) {
        continue;
      }
      if (unpack_small_obs(packed_key) == 0) {
        total_mass_obs0 += mass;
      } else {
        total_mass_obs1 += mass;
      }
    }
    if (total_mass_obs0 == 0.0 && total_mass_obs1 == 0.0) {
      low_confidence_flag = true;
      return;
    }
    predicted_obs_mask = total_mass_obs1 > total_mass_obs0 ? 1 : 0;
    auto tr1 = std::chrono::high_resolution_clock::now();
    time_reconstruct_seconds +=
        std::chrono::duration_cast<std::chrono::microseconds>(tr1 - tr0).count() / 1e6;
  } else {
    auto layers = build_layer_faults(faults, num_detectors, detections, &max_frontier_width_seen);
    std::unordered_map<boost::dynamic_bitset<>, FrontierAggregate> beam;
    FrontierAggregate init;
    add_obs_mass(init, 0, 0.0);
    beam.emplace(boost::dynamic_bitset<>(0), std::move(init));
    max_beam_size_seen = 1;

    for (const auto& layer : layers) {
      auto t0 = std::chrono::high_resolution_clock::now();
      std::unordered_map<boost::dynamic_bitset<>, FrontierAggregate> expanded;
      expanded.reserve(beam.size() * 2 + 1);

      for (const auto& [state, aggregate] : beam) {
        ++num_states_expanded;
        boost::dynamic_bitset<> base_state = state;
        base_state.resize(layer.local_det_mask.size());
        for (const auto& [obs_mask, obs] : aggregate.obs_entries) {
          auto& absent_bucket = expanded[base_state];
          add_obs_mass(absent_bucket, obs_mask, obs.log_mass + layer.log_q);

          boost::dynamic_bitset<> present_state = base_state ^ layer.local_det_mask;
          auto& present_bucket = expanded[present_state];
          add_obs_mass(present_bucket, obs_mask ^ layer.obs_mask, obs.log_mass + layer.log_p);
        }
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      time_expand_seconds +=
          std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;

      std::unordered_map<boost::dynamic_bitset<>, FrontierAggregate> collapsed;
      collapsed.reserve(expanded.size());
      for (auto& [state, aggregate] : expanded) {
        if (((state & layer.retiring_mask) ^ layer.expected_retiring_bits).any()) {
          continue;
        }
        boost::dynamic_bitset<> projected = project_state(state, layer.surviving_local_indices);
        auto& out = collapsed[projected];
        for (const auto& [obs_mask, obs] : aggregate.obs_entries) {
          add_obs_mass(out, obs_mask, obs.log_mass);
        }
      }
      auto t2 = std::chrono::high_resolution_clock::now();
      time_collapse_seconds +=
          std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1e6;

      num_states_merged += collapsed.size();
      if (collapsed.empty()) {
        low_confidence_flag = true;
        return;
      }

      std::vector<std::pair<boost::dynamic_bitset<>, FrontierAggregate>> next_beam;
      next_beam.reserve(collapsed.size());
      for (auto& item : collapsed) {
        next_beam.push_back(std::move(item));
      }
      if (next_beam.size() > config.beam_width) {
        std::nth_element(next_beam.begin(), next_beam.begin() + config.beam_width, next_beam.end(),
                         [](const auto& a, const auto& b) {
                           return a.second.total_log_mass > b.second.total_log_mass;
                         });
        next_beam.resize(config.beam_width);
      }
      max_beam_size_seen = std::max(max_beam_size_seen, next_beam.size());

      beam.clear();
      beam.reserve(next_beam.size());
      for (auto& item : next_beam) {
        beam.emplace(std::move(item.first), std::move(item.second));
      }
      auto t3 = std::chrono::high_resolution_clock::now();
      time_truncate_seconds +=
          std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / 1e6;
    }

    auto it = beam.find(boost::dynamic_bitset<>(0));
    if (it == beam.end() || it->second.obs_entries.empty()) {
      low_confidence_flag = true;
      return;
    }

    const auto& final_entry = it->second;
    auto tr0 = std::chrono::high_resolution_clock::now();
    if (final_entry.obs_entries.empty()) {
      low_confidence_flag = true;
      return;
    }
    auto it0 = final_entry.obs_entries.find(0);
    auto it1 = final_entry.obs_entries.find(1);
    total_mass_obs0 = it0 == final_entry.obs_entries.end() ? 0.0 : std::exp(it0->second.log_mass);
    total_mass_obs1 = it1 == final_entry.obs_entries.end() ? 0.0 : std::exp(it1->second.log_mass);
    if (total_mass_obs0 == 0.0 && total_mass_obs1 == 0.0) {
      low_confidence_flag = true;
      return;
    }
    predicted_obs_mask = total_mass_obs1 > total_mass_obs0 ? 1 : 0;
    auto tr1 = std::chrono::high_resolution_clock::now();
    time_reconstruct_seconds +=
        std::chrono::duration_cast<std::chrono::microseconds>(tr1 - tr0).count() / 1e6;
  }

  if (config.verbose) {
    std::cout << "trellis beam_width=" << config.beam_width
              << " frontier_width=" << max_frontier_width_seen
              << " states_expanded=" << num_states_expanded
              << " states_merged=" << num_states_merged
              << " max_beam=" << max_beam_size_seen << std::endl;
  }
}

std::vector<int> TesseractTrellisDecoder::decode(const std::vector<uint64_t>& detections) {
  decode_shot(detections);
  return predicted_obs_mask ? std::vector<int>{0} : std::vector<int>{};
}

void TesseractTrellisDecoder::decode_shots(std::vector<stim::SparseShot>& shots,
                                           std::vector<std::vector<int>>& obs_predicted) {
  obs_predicted.resize(shots.size());
  for (size_t i = 0; i < shots.size(); ++i) {
    obs_predicted[i] = decode(shots[i].hits);
  }
}
