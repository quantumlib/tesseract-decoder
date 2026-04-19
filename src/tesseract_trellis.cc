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
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#if defined(__BMI2__) && \
    (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
#include <immintrin.h>
#endif
#include <iostream>
#include <limits>
#include <numeric>
#include <utility>

#include "utils.h"

namespace {

struct Fault {
  size_t error_index;
  double likelihood_cost;
  double log_q;
  double log_p;
  uint64_t obs_mask;
  std::vector<int> detectors;
};

struct PackedMass {
  uint64_t key;
  double mass;
  double penalty;
};

struct SmallStateGroup {
  uint64_t state;
  double mass;
  double score;
  size_t begin;
  size_t end;
};

struct WidePackedMass {
  std::vector<uint64_t> state_words;
  uint64_t obs_mask;
  double mass;
  double penalty;
};

struct WideStateMass {
  std::vector<uint64_t> state_words;
  double mass;
  double penalty;
};

struct BranchPenaltyUpdate {
  bool absent_valid = true;
  bool present_valid = true;
  double absent_penalty = 0.0;
  double present_penalty = 0.0;
};

struct FinalizeKeptStateStatsOnExit {
  TesseractTrellisDecoder* decoder;

  ~FinalizeKeptStateStatsOnExit();
};

std::vector<Fault> parse_faults(const std::vector<common::Error>& errors, size_t num_observables) {
  std::vector<Fault> faults;
  faults.reserve(errors.size());
  for (size_t error_index = 0; error_index < errors.size(); ++error_index) {
    const auto& error = errors[error_index];
    const double p = error.get_probability();
    if (p <= 0) {
      continue;
    }
    Fault fault;
    fault.error_index = error_index;
    fault.likelihood_cost = error.likelihood_cost;
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

bool build_small_layer_templates(const std::vector<Fault>& faults, size_t num_detectors,
                                 std::vector<TesseractTrellisSmallLayerTemplate>* layers,
                                 size_t* max_frontier_width_seen) {
  std::vector<size_t> last_seen(num_detectors, std::numeric_limits<size_t>::max());
  for (size_t i = 0; i < faults.size(); ++i) {
    for (int d : faults[i].detectors) {
      last_seen[d] = i;
    }
  }

  std::vector<int> active_detectors;
  active_detectors.reserve(num_detectors);
  std::vector<int> global_to_local(num_detectors, -1);
  layers->clear();
  layers->reserve(faults.size());
  *max_frontier_width_seen = 0;

  for (size_t i = 0; i < faults.size(); ++i) {
    const size_t previous_width = active_detectors.size();
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

    TesseractTrellisSmallLayerTemplate layer{
        .q = std::exp(faults[i].log_q),
        .p = std::exp(faults[i].log_p),
        .obs_flip_bit = faults[i].obs_mask & 1,
        .local_det_mask = 0,
        .retiring_mask = 0,
        .surviving_mask = 0,
        .projected_fault_mask = 0,
        .previous_width = previous_width,
        .surviving_local_indices = {},
        .current_active_detectors = active_detectors,
        .next_frontier_costs = {},
        .detcost_transition = {},
    };
    for (int d : faults[i].detectors) {
      layer.local_det_mask ^= uint64_t{1} << global_to_local[d];
    }
    for (size_t local = 0; local < active_detectors.size(); ++local) {
      const int d = active_detectors[local];
      if (last_seen[d] == i) {
        layer.retiring_mask ^= uint64_t{1} << local;
      } else {
        layer.surviving_local_indices.push_back((uint8_t)local);
      }
    }
    uint64_t live_mask = (uint64_t{1} << active_detectors.size()) - 1;
    layer.surviving_mask = live_mask & ~layer.retiring_mask;

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

void build_small_detector_layer_refs(
    const std::vector<TesseractTrellisSmallLayerTemplate>& layers, size_t num_detectors,
    std::vector<std::vector<TesseractTrellisSmallDetectorLayerRef>>* refs) {
  refs->assign(num_detectors, {});
  for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
    const auto& layer = layers[layer_index];
    for (size_t local = 0; local < layer.current_active_detectors.size(); ++local) {
      int detector = layer.current_active_detectors[local];
      (*refs)[(size_t)detector].push_back(
          {static_cast<uint32_t>(layer_index), static_cast<uint8_t>(local)});
    }
  }
}

void build_wide_layer_templates(const std::vector<Fault>& faults, size_t num_detectors,
                                std::vector<TesseractTrellisWideLayerTemplate>* layers,
                                size_t* max_frontier_width_seen) {
  std::vector<size_t> last_seen(num_detectors, std::numeric_limits<size_t>::max());
  for (size_t i = 0; i < faults.size(); ++i) {
    for (int d : faults[i].detectors) {
      last_seen[d] = i;
    }
  }

  std::vector<int> active_detectors;
  active_detectors.reserve(num_detectors);
  std::vector<int> global_to_local(num_detectors, -1);
  layers->clear();
  layers->reserve(faults.size());
  *max_frontier_width_seen = 0;

  for (size_t i = 0; i < faults.size(); ++i) {
    const size_t previous_width = active_detectors.size();
    for (int d : faults[i].detectors) {
      if (global_to_local[d] == -1) {
        global_to_local[d] = active_detectors.size();
        active_detectors.push_back(d);
      }
    }

    *max_frontier_width_seen = std::max(*max_frontier_width_seen, active_detectors.size());
    TesseractTrellisWideLayerTemplate layer{
        .q = std::exp(faults[i].log_q),
        .p = std::exp(faults[i].log_p),
        .obs_mask = faults[i].obs_mask,
        .previous_width = previous_width,
        .surviving_local_indices = {},
        .current_active_detectors = active_detectors,
        .projected_fault_mask_words = {},
        .next_frontier_costs = {},
        .detcost_transition = {},
    };

    for (size_t local = 0; local < active_detectors.size(); ++local) {
      const int d = active_detectors[local];
      if (last_seen[d] != i) {
        layer.surviving_local_indices.push_back((uint32_t)local);
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
}

template <typename LayerT>
void build_future_detcost_transitions(const std::vector<Fault>& faults, size_t num_detectors,
                                      std::vector<LayerT>* layers,
                                      std::vector<double>* initial_future_detcost) {
  std::vector<double> current_row(num_detectors, INF);
  for (size_t fault_index = faults.size(); fault_index-- > 0;) {
    auto& layer = (*layers)[fault_index];
    const auto& fault = faults[fault_index];

    layer.next_frontier_costs.resize(layer.surviving_local_indices.size(), INF);
    for (size_t next_local = 0; next_local < layer.surviving_local_indices.size(); ++next_local) {
      size_t current_local = (size_t)layer.surviving_local_indices[next_local];
      int global_detector = layer.current_active_detectors[current_local];
      layer.next_frontier_costs[next_local] = current_row[(size_t)global_detector];
    }

    std::vector<int32_t> current_to_next(layer.current_active_detectors.size(), -1);
    for (size_t next_local = 0; next_local < layer.surviving_local_indices.size(); ++next_local) {
      current_to_next[(size_t)layer.surviving_local_indices[next_local]] = (int32_t)next_local;
    }

    auto& transition = layer.detcost_transition;
    transition.fault_local_indices.clear();
    transition.next_local_indices.clear();
    transition.current_costs.clear();
    transition.next_costs.clear();
    transition.fault_local_indices.reserve(fault.detectors.size());
    transition.next_local_indices.reserve(fault.detectors.size());
    transition.current_costs.reserve(fault.detectors.size());
    transition.next_costs.reserve(fault.detectors.size());

    if (!fault.detectors.empty()) {
      double ecost = fault.likelihood_cost / fault.detectors.size();
      for (int detector : fault.detectors) {
        auto it = std::find(layer.current_active_detectors.begin(),
                            layer.current_active_detectors.end(), detector);
        if (it == layer.current_active_detectors.end()) {
          throw std::runtime_error("Missing detector in active frontier while preparing detcost.");
        }
        uint32_t local = (uint32_t)std::distance(layer.current_active_detectors.begin(), it);
        double next_cost = current_row[(size_t)detector];
        double current_cost = std::min(ecost, next_cost);
        transition.fault_local_indices.push_back(local);
        transition.next_local_indices.push_back(current_to_next[local]);
        transition.current_costs.push_back(current_cost);
        transition.next_costs.push_back(next_cost);
        current_row[(size_t)detector] = current_cost;
      }
    }
  }

  if (initial_future_detcost != nullptr) {
    *initial_future_detcost = std::move(current_row);
  }
}

size_t num_state_words(size_t num_bits) {
  return (num_bits + 63) >> 6;
}

uint64_t compact_bits_u64(uint64_t value, uint64_t mask) {
#if defined(__BMI2__) && \
    (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
  return _pext_u64(value, mask);
#else
  uint64_t out = 0;
  uint64_t out_bit = 1;
  while (mask) {
    uint64_t keep = mask & -mask;
    if (value & keep) {
      out |= out_bit;
    }
    mask ^= keep;
    out_bit <<= 1;
  }
  return out;
#endif
}

bool get_state_bit(const std::vector<uint64_t>& state_words, size_t bit, size_t logical_width) {
  if (bit >= logical_width) {
    return false;
  }
  size_t word = bit >> 6;
  if (word >= state_words.size()) {
    return false;
  }
  return (state_words[word] >> (bit & 63)) & 1ULL;
}

void xor_state_words(std::vector<uint64_t>& state_words, const std::vector<uint64_t>& mask_words) {
  if (state_words.size() < mask_words.size()) {
    state_words.resize(mask_words.size(), 0);
  }
  for (size_t k = 0; k < mask_words.size(); ++k) {
    state_words[k] ^= mask_words[k];
  }
}

std::vector<uint64_t> project_wide_state(const std::vector<uint64_t>& state_words,
                                         size_t logical_width,
                                         const std::vector<uint32_t>& surviving_local_indices) {
  std::vector<uint64_t> out(num_state_words(surviving_local_indices.size()), 0);
  for (size_t next_local = 0; next_local < surviving_local_indices.size(); ++next_local) {
    size_t current_local = (size_t)surviving_local_indices[next_local];
    if (get_state_bit(state_words, current_local, logical_width)) {
      out[next_local >> 6] ^= uint64_t{1} << (next_local & 63);
    }
  }
  return out;
}

bool wide_state_less(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
  if (a.size() != b.size()) {
    return a.size() < b.size();
  }
  for (size_t k = a.size(); k-- > 0;) {
    if (a[k] != b[k]) {
      return a[k] < b[k];
    }
  }
  return false;
}

bool wide_state_zero(const std::vector<uint64_t>& state_words) {
  for (uint64_t word : state_words) {
    if (word != 0) {
      return false;
    }
  }
  return true;
}

uint64_t project_small_state(uint64_t state, uint64_t surviving_mask) {
  return compact_bits_u64(state, surviving_mask);
}

double compute_initial_penalty_for_target_bits(uint64_t target_bits,
                                               const std::vector<int>& active_detectors,
                                               const std::vector<double>& initial_future_detcost) {
  double total = 0.0;
  while (target_bits) {
    uint64_t low_bit = target_bits & -target_bits;
    size_t local = (size_t)std::countr_zero(low_bit);
    target_bits ^= low_bit;
    double best = initial_future_detcost[(size_t)active_detectors[local]];
    if (best == INF) {
      return INF;
    }
    total += best;
  }
  return total;
}

double compute_initial_penalty_for_active_detectors(
    const std::vector<int>& active_detectors, const boost::dynamic_bitset<>& actual_dets,
    const std::vector<double>& initial_future_detcost) {
  double total = 0.0;
  for (int detector : active_detectors) {
    if (!actual_dets[(size_t)detector]) {
      continue;
    }
    double best = initial_future_detcost[(size_t)detector];
    if (best == INF) {
      return INF;
    }
    total += best;
  }
  return total;
}

BranchPenaltyUpdate compute_small_branch_update(uint64_t base_state, size_t previous_width,
                                                double current_penalty,
                                                uint64_t current_target_bits,
                                                const TesseractTrellisDetcostTransition& transition,
                                                bool compute_penalties) {
  BranchPenaltyUpdate update;
  update.absent_penalty = compute_penalties ? current_penalty : 0.0;
  update.present_penalty = compute_penalties ? current_penalty : 0.0;

  for (size_t k = 0; k < transition.fault_local_indices.size(); ++k) {
    size_t local = transition.fault_local_indices[k];
    bool state_bit = local < previous_width && ((base_state >> local) & 1ULL);
    bool target_bit = (current_target_bits >> local) & 1ULL;
    bool mismatch = state_bit ^ target_bit;
    int32_t next_local = transition.next_local_indices[k];

    if (next_local < 0) {
      if (mismatch) {
        update.absent_valid = false;
      } else {
        update.present_valid = false;
      }
    }

    if (!compute_penalties) {
      continue;
    }

    double prev_contrib = (local < previous_width && mismatch) ? transition.current_costs[k] : 0.0;
    double absent_contrib = (next_local >= 0 && mismatch) ? transition.next_costs[k] : 0.0;
    double present_contrib = (next_local >= 0 && !mismatch) ? transition.next_costs[k] : 0.0;
    update.absent_penalty += absent_contrib - prev_contrib;
    update.present_penalty += present_contrib - prev_contrib;
  }

  return update;
}

BranchPenaltyUpdate compute_wide_branch_update(const std::vector<uint64_t>& base_state_words,
                                               size_t previous_width, double current_penalty,
                                               const std::vector<int>& current_active_detectors,
                                               const boost::dynamic_bitset<>& actual_dets,
                                               const TesseractTrellisDetcostTransition& transition,
                                               bool compute_penalties) {
  BranchPenaltyUpdate update;
  update.absent_penalty = compute_penalties ? current_penalty : 0.0;
  update.present_penalty = compute_penalties ? current_penalty : 0.0;

  for (size_t k = 0; k < transition.fault_local_indices.size(); ++k) {
    size_t local = transition.fault_local_indices[k];
    bool state_bit = get_state_bit(base_state_words, local, previous_width);
    bool target_bit = actual_dets[(size_t)current_active_detectors[local]];
    bool mismatch = state_bit ^ target_bit;
    int32_t next_local = transition.next_local_indices[k];

    if (next_local < 0) {
      if (mismatch) {
        update.absent_valid = false;
      } else {
        update.present_valid = false;
      }
    }

    if (!compute_penalties) {
      continue;
    }

    double prev_contrib = (local < previous_width && mismatch) ? transition.current_costs[k] : 0.0;
    double absent_contrib = (next_local >= 0 && mismatch) ? transition.next_costs[k] : 0.0;
    double present_contrib = (next_local >= 0 && !mismatch) ? transition.next_costs[k] : 0.0;
    update.absent_penalty += absent_contrib - prev_contrib;
    update.present_penalty += present_contrib - prev_contrib;
  }

  return update;
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

void normalize_items(std::vector<WidePackedMass>& items) {
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

void radix_sort_packed_masses_by_key(std::vector<PackedMass>& items) {
  if (items.size() <= 1) {
    return;
  }

  thread_local std::vector<PackedMass> buffer;
  buffer.resize(items.size());

  PackedMass* src = items.data();
  PackedMass* dst = buffer.data();
  constexpr size_t RADIX = 256;
  std::array<size_t, RADIX> counts;

  for (size_t shift = 0; shift < 64; shift += 8) {
    counts.fill(0);
    for (size_t k = 0; k < items.size(); ++k) {
      ++counts[(src[k].key >> shift) & 0xFF];
    }

    size_t total = 0;
    for (size_t k = 0; k < RADIX; ++k) {
      size_t count = counts[k];
      counts[k] = total;
      total += count;
    }

    for (size_t k = 0; k < items.size(); ++k) {
      dst[counts[(src[k].key >> shift) & 0xFF]++] = src[k];
    }
    std::swap(src, dst);
  }
}

void merge_equal_keys_inplace(std::vector<PackedMass>& items) {
  if (items.empty()) {
    return;
  }
  radix_sort_packed_masses_by_key(items);
  size_t out = 0;
  for (size_t i = 1; i < items.size(); ++i) {
    if (items[i].key == items[out].key) {
      items[out].mass += items[i].mass;
    } else {
      ++out;
      if (out != i) {
        items[out] = std::move(items[i]);
      }
    }
  }
  items.resize(out + 1);
}

void merge_equal_keys_inplace(std::vector<WidePackedMass>& items) {
  if (items.empty()) {
    return;
  }
  std::sort(items.begin(), items.end(), [](const WidePackedMass& a, const WidePackedMass& b) {
    if (wide_state_less(a.state_words, b.state_words)) {
      return true;
    }
    if (wide_state_less(b.state_words, a.state_words)) {
      return false;
    }
    return a.obs_mask < b.obs_mask;
  });

  size_t out = 0;
  for (size_t i = 1; i < items.size(); ++i) {
    if (items[i].obs_mask == items[out].obs_mask &&
        items[i].state_words == items[out].state_words) {
      items[out].mass += items[i].mass;
    } else {
      ++out;
      if (out != i) {
        items[out] = std::move(items[i]);
      }
    }
  }
  items.resize(out + 1);
}

std::vector<WideStateMass> accumulate_state_masses_from_entries(
    const std::vector<WidePackedMass>& entries) {
  std::vector<WideStateMass> totals;
  if (entries.empty()) {
    return totals;
  }
  totals.reserve(entries.size());
  WideStateMass current{entries[0].state_words, entries[0].mass, entries[0].penalty};
  for (size_t i = 1; i < entries.size(); ++i) {
    if (entries[i].state_words == current.state_words) {
      current.mass += entries[i].mass;
    } else {
      totals.push_back(std::move(current));
      current = {entries[i].state_words, entries[i].mass, entries[i].penalty};
    }
  }
  totals.push_back(std::move(current));
  return totals;
}

double score_mass_and_penalty(double mass, double penalty,
                              TesseractTrellisRankingMode ranking_mode) {
  if (ranking_mode == TesseractTrellisRankingMode::MassOnly) {
    return mass;
  }
  if (penalty == INF || mass == 0.0) {
    return -INF;
  }
  return std::log(mass) - penalty;
}

double branch_score(const PackedMass& item, TesseractTrellisRankingMode ranking_mode) {
  return score_mass_and_penalty(item.mass, item.penalty, ranking_mode);
}

double branch_score(const WidePackedMass& item, TesseractTrellisRankingMode ranking_mode) {
  return score_mass_and_penalty(item.mass, item.penalty, ranking_mode);
}

void reset_kept_state_stats(TesseractTrellisDecoder* decoder) {
  decoder->kept_state_sample_count = 0;
  decoder->kept_state_min = 0;
  decoder->kept_state_median = 0;
  decoder->kept_state_mean = 0;
  decoder->kept_state_max = 0;
  if (!decoder->config.track_kept_state_stats) {
    return;
  }

  const size_t histogram_size = decoder->config.beam_width + 1;
  if (decoder->kept_state_histogram_scratch.size() != histogram_size) {
    decoder->kept_state_histogram_scratch.assign(histogram_size, 0);
  } else {
    std::fill(decoder->kept_state_histogram_scratch.begin(),
              decoder->kept_state_histogram_scratch.end(), 0);
  }
}

void record_kept_state_count(TesseractTrellisDecoder* decoder, size_t kept_states) {
  if (!decoder->config.track_kept_state_stats) {
    return;
  }

  kept_states = std::min(kept_states, decoder->config.beam_width);
  if (decoder->kept_state_sample_count == 0) {
    decoder->kept_state_min = kept_states;
    decoder->kept_state_max = kept_states;
  } else {
    decoder->kept_state_min = std::min(decoder->kept_state_min, kept_states);
    decoder->kept_state_max = std::max(decoder->kept_state_max, kept_states);
  }
  ++decoder->kept_state_sample_count;
  decoder->kept_state_mean += kept_states;
  ++decoder->kept_state_histogram_scratch[kept_states];
}

void finalize_kept_state_stats(TesseractTrellisDecoder* decoder) {
  if (!decoder->config.track_kept_state_stats || decoder->kept_state_sample_count == 0) {
    return;
  }

  decoder->kept_state_mean /= decoder->kept_state_sample_count;
  const size_t lower_target = (decoder->kept_state_sample_count - 1) >> 1;
  const size_t upper_target = decoder->kept_state_sample_count >> 1;
  size_t seen = 0;
  size_t lower = 0;
  size_t upper = 0;
  bool lower_found = false;
  for (size_t kept_states = 0; kept_states < decoder->kept_state_histogram_scratch.size();
       ++kept_states) {
    seen += decoder->kept_state_histogram_scratch[kept_states];
    if (!lower_found && seen > lower_target) {
      lower = kept_states;
      lower_found = true;
    }
    if (seen > upper_target) {
      upper = kept_states;
      break;
    }
  }
  decoder->kept_state_median = 0.5 * (lower + upper);
}

FinalizeKeptStateStatsOnExit::~FinalizeKeptStateStatsOnExit() {
  finalize_kept_state_stats(decoder);
}

bool small_state_group_score_greater(const SmallStateGroup& a, const SmallStateGroup& b) {
  if (a.score != b.score) {
    return a.score > b.score;
  }
  return a.state < b.state;
}

size_t trim_small_state_groups_by_beam_and_mass(std::vector<SmallStateGroup>* groups,
                                                size_t beam_width, double beam_eps) {
  if (groups->empty()) {
    return 0;
  }

  double total_mass = 0.0;
  if (beam_eps > 0.0) {
    for (const auto& group : *groups) {
      total_mass += group.mass;
    }
  }

  if (groups->size() > beam_width) {
    std::nth_element(
        groups->begin(), groups->begin() + beam_width, groups->end(),
        [](const SmallStateGroup& a, const SmallStateGroup& b) { return a.score > b.score; });
    groups->resize(beam_width);
  } else if (beam_eps <= 0.0) {
    return groups->size();
  }

  if (beam_eps <= 0.0 || total_mass <= 0.0) {
    return groups->size();
  }

  std::sort(groups->begin(), groups->end(), small_state_group_score_greater);
  const double retained_target_mass = total_mass * (1.0 - beam_eps);
  double retained_mass = 0.0;
  size_t keep_count = 0;
  while (keep_count < groups->size()) {
    retained_mass += (*groups)[keep_count].mass;
    ++keep_count;
    if (retained_mass >= retained_target_mass) {
      break;
    }
  }
  groups->resize(keep_count);
  std::sort(groups->begin(), groups->end(),
            [](const SmallStateGroup& a, const SmallStateGroup& b) { return a.begin < b.begin; });
  return groups->size();
}

std::vector<SmallStateGroup> collect_small_state_groups(const std::vector<PackedMass>& entries,
                                                        TesseractTrellisRankingMode ranking_mode) {
  std::vector<SmallStateGroup> groups;
  if (entries.empty()) {
    return groups;
  }
  groups.reserve(entries.size());
  size_t begin = 0;
  while (begin < entries.size()) {
    uint64_t state = unpack_small_state(entries[begin].key);
    double mass = 0.0;
    size_t end = begin;
    while (end < entries.size() && unpack_small_state(entries[end].key) == state) {
      mass += entries[end].mass;
      ++end;
    }
    groups.push_back({state, mass,
                      score_mass_and_penalty(mass, entries[begin].penalty, ranking_mode), begin,
                      end});
    begin = end;
  }
  return groups;
}

double state_score(const WideStateMass& item, TesseractTrellisRankingMode ranking_mode) {
  return score_mass_and_penalty(item.mass, item.penalty, ranking_mode);
}

size_t keep_top_states(std::vector<PackedMass>& entries, size_t beam_width, double beam_eps,
                       TesseractTrellisRankingMode ranking_mode) {
  if (entries.empty()) {
    return 0;
  }
  auto groups = collect_small_state_groups(entries, ranking_mode);
  const size_t kept_group_count =
      trim_small_state_groups_by_beam_and_mass(&groups, beam_width, beam_eps);

  std::vector<PackedMass> kept;
  size_t kept_entries = 0;
  for (const auto& group : groups) {
    kept_entries += group.end - group.begin;
  }
  kept.reserve(kept_entries);
  for (const auto& group : groups) {
    for (size_t k = group.begin; k < group.end; ++k) {
      kept.push_back(std::move(entries[k]));
    }
  }
  entries = std::move(kept);
  return kept_group_count;
}

size_t keep_top_states(std::vector<WidePackedMass>& entries, size_t beam_width, double beam_eps,
                       TesseractTrellisRankingMode ranking_mode) {
  if (entries.empty()) {
    return 0;
  }
  auto totals = accumulate_state_masses_from_entries(entries);
  double total_mass = 0.0;
  if (beam_eps > 0.0) {
    for (const auto& item : totals) {
      total_mass += item.mass;
    }
  }

  if (totals.size() > beam_width) {
    std::nth_element(totals.begin(), totals.begin() + beam_width, totals.end(),
                     [ranking_mode](const WideStateMass& a, const WideStateMass& b) {
                       return state_score(a, ranking_mode) > state_score(b, ranking_mode);
                     });
    totals.resize(beam_width);
  } else if (beam_eps <= 0.0) {
    return totals.size();
  }

  if (beam_eps > 0.0 && total_mass > 0.0) {
    std::sort(totals.begin(), totals.end(),
              [ranking_mode](const WideStateMass& a, const WideStateMass& b) {
                double sa = state_score(a, ranking_mode);
                double sb = state_score(b, ranking_mode);
                if (sa != sb) {
                  return sa > sb;
                }
                return wide_state_less(a.state_words, b.state_words);
              });
    const double retained_target_mass = total_mass * (1.0 - beam_eps);
    double retained_mass = 0.0;
    size_t keep_count = 0;
    while (keep_count < totals.size()) {
      retained_mass += totals[keep_count].mass;
      ++keep_count;
      if (retained_mass >= retained_target_mass) {
        break;
      }
    }
    totals.resize(keep_count);
  }

  std::sort(totals.begin(), totals.end(), [](const WideStateMass& a, const WideStateMass& b) {
    return wide_state_less(a.state_words, b.state_words);
  });

  std::vector<WidePackedMass> kept;
  kept.reserve(entries.size());
  size_t ti = 0;
  for (auto& item : entries) {
    while (ti < totals.size() && wide_state_less(totals[ti].state_words, item.state_words)) {
      ++ti;
    }
    if (ti < totals.size() && item.state_words == totals[ti].state_words) {
      kept.push_back(std::move(item));
    }
  }
  entries = std::move(kept);
  return totals.size();
}

void keep_best_state_representatives(std::vector<PackedMass>& entries, size_t beam_width,
                                     TesseractTrellisRankingMode ranking_mode) {
  if (entries.empty()) {
    return;
  }
  if (beam_width == 0) {
    entries.clear();
    return;
  }

  std::vector<size_t> representative_indices;
  representative_indices.reserve(entries.size());
  size_t begin = 0;
  while (begin < entries.size()) {
    uint64_t state = unpack_small_state(entries[begin].key);
    size_t best = begin;
    double best_score = branch_score(entries[begin], ranking_mode);
    size_t end = begin + 1;
    while (end < entries.size() && unpack_small_state(entries[end].key) == state) {
      double score = branch_score(entries[end], ranking_mode);
      if (score > best_score) {
        best = end;
        best_score = score;
      }
      ++end;
    }
    representative_indices.push_back(best);
    begin = end;
  }

  if (representative_indices.size() > beam_width) {
    std::nth_element(representative_indices.begin(), representative_indices.begin() + beam_width,
                     representative_indices.end(), [&entries, ranking_mode](size_t a, size_t b) {
                       double sa = branch_score(entries[a], ranking_mode);
                       double sb = branch_score(entries[b], ranking_mode);
                       if (sa != sb) {
                         return sa > sb;
                       }
                       return a < b;
                     });
    representative_indices.resize(beam_width);
  }
  std::sort(representative_indices.begin(), representative_indices.end());

  std::vector<PackedMass> kept;
  kept.reserve(representative_indices.size());
  for (size_t idx : representative_indices) {
    kept.push_back(entries[idx]);
  }
  entries = std::move(kept);
}

void keep_best_state_representatives(std::vector<WidePackedMass>& entries, size_t beam_width,
                                     TesseractTrellisRankingMode ranking_mode) {
  if (entries.empty()) {
    return;
  }
  if (beam_width == 0) {
    entries.clear();
    return;
  }

  std::vector<size_t> representative_indices;
  representative_indices.reserve(entries.size());
  size_t begin = 0;
  while (begin < entries.size()) {
    size_t best = begin;
    double best_score = branch_score(entries[begin], ranking_mode);
    size_t end = begin + 1;
    while (end < entries.size() && entries[end].state_words == entries[begin].state_words) {
      double score = branch_score(entries[end], ranking_mode);
      if (score > best_score) {
        best = end;
        best_score = score;
      }
      ++end;
    }
    representative_indices.push_back(best);
    begin = end;
  }

  if (representative_indices.size() > beam_width) {
    std::nth_element(representative_indices.begin(), representative_indices.begin() + beam_width,
                     representative_indices.end(), [&entries, ranking_mode](size_t a, size_t b) {
                       double sa = branch_score(entries[a], ranking_mode);
                       double sb = branch_score(entries[b], ranking_mode);
                       if (sa != sb) {
                         return sa > sb;
                       }
                       return a < b;
                     });
    representative_indices.resize(beam_width);
  }
  std::sort(representative_indices.begin(), representative_indices.end());

  std::vector<WidePackedMass> kept;
  kept.reserve(representative_indices.size());
  for (size_t idx : representative_indices) {
    kept.push_back(std::move(entries[idx]));
  }
  entries = std::move(kept);
}

void keep_top_branch_entries(std::vector<PackedMass>& entries, size_t beam_width,
                             TesseractTrellisRankingMode ranking_mode) {
  if (entries.size() <= beam_width) {
    return;
  }
  std::nth_element(entries.begin(), entries.begin() + beam_width, entries.end(),
                   [ranking_mode](const PackedMass& a, const PackedMass& b) {
                     return branch_score(a, ranking_mode) > branch_score(b, ranking_mode);
                   });
  entries.resize(beam_width);
}

void keep_top_branch_entries(std::vector<WidePackedMass>& entries, size_t beam_width,
                             TesseractTrellisRankingMode ranking_mode) {
  if (entries.size() <= beam_width) {
    return;
  }
  std::nth_element(entries.begin(), entries.begin() + beam_width, entries.end(),
                   [ranking_mode](const WidePackedMass& a, const WidePackedMass& b) {
                     return branch_score(a, ranking_mode) > branch_score(b, ranking_mode);
                   });
  entries.resize(beam_width);
}

void prepare_projected_fault_masks(std::vector<TesseractTrellisSmallLayerTemplate>* layers) {
  for (auto& layer : *layers) {
    layer.projected_fault_mask = 0;
    for (int32_t next_local : layer.detcost_transition.next_local_indices) {
      if (next_local >= 0) {
        layer.projected_fault_mask ^= uint64_t{1} << next_local;
      }
    }
  }
}

void prepare_projected_fault_masks(std::vector<TesseractTrellisWideLayerTemplate>* layers) {
  for (auto& layer : *layers) {
    layer.projected_fault_mask_words.assign(num_state_words(layer.surviving_local_indices.size()),
                                            0);
    for (int32_t next_local : layer.detcost_transition.next_local_indices) {
      if (next_local >= 0) {
        size_t local = (size_t)next_local;
        layer.projected_fault_mask_words[local >> 6] ^= uint64_t{1} << (local & 63);
      }
    }
  }
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

  all_possible_detectors = boost::dynamic_bitset<>(num_detectors);
  for (const auto& error : errors) {
    for (int d : error.symptom.detectors) {
      all_possible_detectors[(size_t)d] = true;
    }
  }

  auto faults = parse_faults(errors, num_observables);

  size_t wide_frontier_width = 0;
  build_wide_layer_templates(faults, num_detectors, &wide_layer_templates, &wide_frontier_width);
  build_future_detcost_transitions(faults, num_detectors, &wide_layer_templates,
                                   &initial_future_detcost);
  prepare_projected_fault_masks(&wide_layer_templates);

  size_t small_frontier_width = 0;
  has_small_layer_templates =
      num_observables <= 1 &&
      build_small_layer_templates(faults, num_detectors, &small_layer_templates,
                                  &small_frontier_width);
  if (has_small_layer_templates) {
    build_future_detcost_transitions(faults, num_detectors, &small_layer_templates, nullptr);
    prepare_projected_fault_masks(&small_layer_templates);
    build_small_detector_layer_refs(small_layer_templates, num_detectors,
                                    &small_detector_layer_refs);
    scratch_small_current_target_bits.assign(small_layer_templates.size(), 0);
    scratch_small_expected_retiring_bits.assign(small_layer_templates.size(), 0);
  }
}

void TesseractTrellisDecoder::decode_shot(const std::vector<uint64_t>& detections) {
  low_confidence_flag = false;
  num_states_expanded = 0;
  num_states_merged = 0;
  max_beam_size_seen = 0;
  max_frontier_width_seen = 0;
  reset_kept_state_stats(this);
  time_expand_seconds = 0;
  time_collapse_seconds = 0;
  time_truncate_seconds = 0;
  time_reconstruct_seconds = 0;
  predicted_obs_mask = 0;
  total_mass_obs0 = 0;
  total_mass_obs1 = 0;
  FinalizeKeptStateStatsOnExit kept_state_stats_guard{this};

  if (has_small_layer_templates) {
    std::fill(scratch_small_current_target_bits.begin(), scratch_small_current_target_bits.end(),
              0);
    std::fill(scratch_small_expected_retiring_bits.begin(),
              scratch_small_expected_retiring_bits.end(), 0);

    for (uint64_t d : detections) {
      if (d >= num_detectors || !all_possible_detectors[d]) {
        low_confidence_flag = true;
        return;
      }
      for (const auto& ref : small_detector_layer_refs[(size_t)d]) {
        scratch_small_current_target_bits[ref.layer_index] ^= uint64_t{1} << ref.local_index;
      }
    }

    for (size_t layer_index = 0; layer_index < small_layer_templates.size(); ++layer_index) {
      const auto& layer = small_layer_templates[layer_index];
      max_frontier_width_seen =
          std::max(max_frontier_width_seen, layer.current_active_detectors.size());
      scratch_small_expected_retiring_bits[layer_index] =
          scratch_small_current_target_bits[layer_index] & layer.retiring_mask;
    }

    double initial_penalty = 0.0;
    if (config.ranking_mode == TesseractTrellisRankingMode::FutureDetcostRanked &&
        !small_layer_templates.empty()) {
      initial_penalty = compute_initial_penalty_for_target_bits(
          scratch_small_current_target_bits.front(),
          small_layer_templates.front().current_active_detectors, initial_future_detcost);
    }

    std::vector<PackedMass> beam_entries;
    std::vector<PackedMass> next_entries;
    beam_entries.reserve(config.beam_width * 2 + 2);
    next_entries.reserve(config.beam_width * 4 + 4);
    beam_entries.push_back({pack_small_key(0, 0), 1.0, initial_penalty});
    max_beam_size_seen = 1;

    const bool compute_penalties =
        config.ranking_mode == TesseractTrellisRankingMode::FutureDetcostRanked;
    for (size_t layer_index = 0; layer_index < small_layer_templates.size(); ++layer_index) {
      const auto& layer = small_layer_templates[layer_index];
      const uint64_t current_target_bits = scratch_small_current_target_bits[layer_index];
      const uint64_t expected_retiring_bits = scratch_small_expected_retiring_bits[layer_index];

      auto t0 = std::chrono::high_resolution_clock::now();
      next_entries.clear();
      next_entries.reserve(beam_entries.size() * 2);
      for (const auto& item : beam_entries) {
        ++num_states_expanded;
        const uint64_t base_state = unpack_small_state(item.key);
        const uint64_t base_obs = unpack_small_obs(item.key);

        BranchPenaltyUpdate update;
        if (compute_penalties) {
          update = compute_small_branch_update(base_state, layer.previous_width, item.penalty,
                                               current_target_bits, layer.detcost_transition, true);
        } else {
          update.absent_valid =
              (((base_state ^ expected_retiring_bits) & layer.retiring_mask) == 0);
          uint64_t toggled_state = base_state ^ layer.local_det_mask;
          update.present_valid =
              (((toggled_state ^ expected_retiring_bits) & layer.retiring_mask) == 0);
        }

        if (!update.absent_valid && !update.present_valid) {
          continue;
        }

        uint64_t projected_state = project_small_state(base_state, layer.surviving_mask);
        if (update.absent_valid && layer.q != 0.0) {
          next_entries.push_back({pack_small_key(projected_state, base_obs), item.mass * layer.q,
                                  update.absent_penalty});
        }
        if (update.present_valid && layer.p != 0.0) {
          next_entries.push_back({pack_small_key(projected_state ^ layer.projected_fault_mask,
                                                 base_obs ^ layer.obs_flip_bit),
                                  item.mass * layer.p, update.present_penalty});
        }
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      time_expand_seconds +=
          std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;

      beam_entries.swap(next_entries);
      bool at_checkpoint = ((layer_index + 1) % config.merge_interval == 0) ||
                           (layer_index + 1 == small_layer_templates.size());
      if (!at_checkpoint) {
        normalize_items(beam_entries);
        max_beam_size_seen = std::max(max_beam_size_seen, beam_entries.size());
        if (beam_entries.empty()) {
          low_confidence_flag = true;
          return;
        }
        continue;
      }

      auto t2a = std::chrono::high_resolution_clock::now();
      if (config.prune_mode != TesseractTrellisPruneMode::NoMerge) {
        merge_equal_keys_inplace(beam_entries);
      }
      auto t2 = std::chrono::high_resolution_clock::now();
      time_collapse_seconds +=
          std::chrono::duration_cast<std::chrono::microseconds>(t2 - t2a).count() / 1e6;

      size_t kept_states = 0;
      if (config.prune_mode == TesseractTrellisPruneMode::MergedStates) {
        kept_states =
            keep_top_states(beam_entries, config.beam_width, config.beam_eps, config.ranking_mode);
      } else if (config.prune_mode == TesseractTrellisPruneMode::KeepBest) {
        keep_best_state_representatives(beam_entries, config.beam_width, config.ranking_mode);
      } else if (config.prune_mode == TesseractTrellisPruneMode::BranchEntries ||
                 config.prune_mode == TesseractTrellisPruneMode::NoMerge) {
        keep_top_branch_entries(beam_entries, config.beam_width, config.ranking_mode);
      }
      normalize_items(beam_entries);
      const size_t kept_state_sample =
          beam_entries.empty() ? 0
                               : (config.prune_mode == TesseractTrellisPruneMode::MergedStates
                                      ? kept_states
                                      : beam_entries.size());
      record_kept_state_count(this, kept_state_sample);
      if (beam_entries.empty()) {
        low_confidence_flag = true;
        return;
      }
      if (config.prune_mode == TesseractTrellisPruneMode::MergedStates) {
        num_states_merged += kept_states;
        max_beam_size_seen = std::max(max_beam_size_seen, kept_states);
      } else if (config.prune_mode == TesseractTrellisPruneMode::KeepBest) {
        num_states_merged += beam_entries.size();
        max_beam_size_seen = std::max(max_beam_size_seen, beam_entries.size());
      } else {
        num_states_merged += beam_entries.size();
        max_beam_size_seen = std::max(max_beam_size_seen, beam_entries.size());
      }
      auto t3 = std::chrono::high_resolution_clock::now();
      time_truncate_seconds +=
          std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / 1e6;
    }

    auto tr0 = std::chrono::high_resolution_clock::now();
    for (const auto& [packed_key, mass, penalty] : beam_entries) {
      (void)penalty;
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
    boost::dynamic_bitset<> actual_dets(num_detectors);
    for (uint64_t d : detections) {
      if (d >= num_detectors || !all_possible_detectors[d]) {
        low_confidence_flag = true;
        return;
      }
      actual_dets.flip((size_t)d);
    }
    max_frontier_width_seen = 0;
    for (const auto& layer : wide_layer_templates) {
      max_frontier_width_seen =
          std::max(max_frontier_width_seen, layer.current_active_detectors.size());
    }

    double initial_penalty = 0.0;
    if (config.ranking_mode == TesseractTrellisRankingMode::FutureDetcostRanked &&
        !wide_layer_templates.empty()) {
      initial_penalty = compute_initial_penalty_for_active_detectors(
          wide_layer_templates.front().current_active_detectors, actual_dets,
          initial_future_detcost);
    }

    std::vector<WidePackedMass> beam_entries;
    std::vector<WidePackedMass> next_entries;
    beam_entries.reserve(config.beam_width * 2 + 2);
    next_entries.reserve(config.beam_width * 4 + 4);
    beam_entries.push_back({{}, 0, 1.0, initial_penalty});
    max_beam_size_seen = 1;

    for (size_t layer_index = 0; layer_index < wide_layer_templates.size(); ++layer_index) {
      const auto& layer = wide_layer_templates[layer_index];
      const bool compute_penalties =
          config.ranking_mode == TesseractTrellisRankingMode::FutureDetcostRanked;

      auto t0 = std::chrono::high_resolution_clock::now();
      next_entries.clear();
      next_entries.reserve(beam_entries.size() * 2);

      if (config.verbose) {
        std::cout << "expanding layer " << layer_index << " / " << (wide_layer_templates.size() - 1)
                  << std::endl;
        std::cout << "states to expand = " << beam_entries.size() << std::endl;
      }
      for (const auto& item : beam_entries) {
        ++num_states_expanded;
        BranchPenaltyUpdate update = compute_wide_branch_update(
            item.state_words, layer.previous_width, item.penalty, layer.current_active_detectors,
            actual_dets, layer.detcost_transition, compute_penalties);

        if (!update.absent_valid && !update.present_valid) {
          continue;
        }

        std::vector<uint64_t> projected_state = project_wide_state(
            item.state_words, layer.previous_width, layer.surviving_local_indices);
        if (update.absent_valid && layer.q != 0.0) {
          next_entries.push_back(
              {projected_state, item.obs_mask, item.mass * layer.q, update.absent_penalty});
        }
        if (update.present_valid && layer.p != 0.0) {
          std::vector<uint64_t> projected_toggled = projected_state;
          xor_state_words(projected_toggled, layer.projected_fault_mask_words);
          next_entries.push_back({std::move(projected_toggled), item.obs_mask ^ layer.obs_mask,
                                  item.mass * layer.p, update.present_penalty});
        }
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      time_expand_seconds +=
          std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;

      beam_entries.swap(next_entries);
      bool at_checkpoint = ((layer_index + 1) % config.merge_interval == 0) ||
                           (layer_index + 1 == wide_layer_templates.size());
      if (!at_checkpoint) {
        normalize_items(beam_entries);
        max_beam_size_seen = std::max(max_beam_size_seen, beam_entries.size());
        if (beam_entries.empty()) {
          low_confidence_flag = true;
          return;
        }
        continue;
      }

      auto t2a = std::chrono::high_resolution_clock::now();
      if (config.prune_mode != TesseractTrellisPruneMode::NoMerge) {
        merge_equal_keys_inplace(beam_entries);
      }
      auto t2 = std::chrono::high_resolution_clock::now();
      time_collapse_seconds +=
          std::chrono::duration_cast<std::chrono::microseconds>(t2 - t2a).count() / 1e6;

      size_t kept_states = 0;
      if (config.prune_mode == TesseractTrellisPruneMode::MergedStates) {
        kept_states =
            keep_top_states(beam_entries, config.beam_width, config.beam_eps, config.ranking_mode);
      } else if (config.prune_mode == TesseractTrellisPruneMode::KeepBest) {
        keep_best_state_representatives(beam_entries, config.beam_width, config.ranking_mode);
      } else if (config.prune_mode == TesseractTrellisPruneMode::BranchEntries ||
                 config.prune_mode == TesseractTrellisPruneMode::NoMerge) {
        keep_top_branch_entries(beam_entries, config.beam_width, config.ranking_mode);
      }
      normalize_items(beam_entries);
      const size_t kept_state_sample =
          beam_entries.empty() ? 0
                               : (config.prune_mode == TesseractTrellisPruneMode::MergedStates
                                      ? kept_states
                                      : beam_entries.size());
      record_kept_state_count(this, kept_state_sample);
      if (beam_entries.empty()) {
        low_confidence_flag = true;
        return;
      }
      if (config.prune_mode == TesseractTrellisPruneMode::NoMerge) {
        num_states_merged += beam_entries.size();
        max_beam_size_seen = std::max(max_beam_size_seen, beam_entries.size());
      } else if (config.prune_mode == TesseractTrellisPruneMode::KeepBest) {
        num_states_merged += beam_entries.size();
        max_beam_size_seen = std::max(max_beam_size_seen, beam_entries.size());
      } else {
        num_states_merged += kept_states;
        max_beam_size_seen = std::max(max_beam_size_seen, kept_states);
      }
      auto t3 = std::chrono::high_resolution_clock::now();
      time_truncate_seconds +=
          std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / 1e6;
    }

    auto tr0 = std::chrono::high_resolution_clock::now();
    for (const auto& item : beam_entries) {
      if (!wide_state_zero(item.state_words)) {
        continue;
      }
      if (item.obs_mask == 0) {
        total_mass_obs0 += item.mass;
      } else if (item.obs_mask == 1) {
        total_mass_obs1 += item.mass;
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
  }

  if (config.verbose) {
    std::cout << "trellis beam_width=" << config.beam_width
              << " frontier_width=" << max_frontier_width_seen
              << " states_expanded=" << num_states_expanded
              << " states_merged=" << num_states_merged << " max_beam=" << max_beam_size_seen
              << std::endl;
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
