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
#include <bit>
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
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>

#include "utils.h"

struct TesseractTrellisWideKernelBase {
  virtual ~TesseractTrellisWideKernelBase() = default;
  virtual void decode_shot(TesseractTrellisDecoder* decoder,
                           const std::vector<uint64_t>& detections) const = 0;
  virtual std::vector<double> decode_shot_with_observable_logit_gradient(
      TesseractTrellisDecoder* decoder, const std::vector<uint64_t>& detections) const = 0;
  virtual std::vector<double> decode_shot_with_syndrome_log_probability_gradient(
      TesseractTrellisDecoder* decoder, const std::vector<uint64_t>& detections) const = 0;
};

namespace {

constexpr size_t kMaxCompiledWideStateWords = 4;

#if defined(__GNUC__) || defined(__clang__)
#define TESSERACT_ALWAYS_INLINE inline __attribute__((always_inline))
#define TESSERACT_HOT __attribute__((hot))
#else
#define TESSERACT_ALWAYS_INLINE inline
#define TESSERACT_HOT
#endif

struct Fault {
  size_t error_index;
  double likelihood_cost;
  double log_q;
  double log_p;
  uint64_t obs_mask;
  std::vector<int> detectors;
};

template <size_t Words>
using FixedWideStateWords = std::array<uint64_t, Words>;

template <size_t Words>
struct FixedWideStateEntry {
  FixedWideStateWords<Words> state_words{};
  double mass0 = 0.0;
  double mass1 = 0.0;
  double penalty = 0.0;
  double score = -INF;
};

template <size_t Words>
struct FixedWidePairBucket {
  FixedWideStateWords<Words> key{};
  double mass0[2]{};
  double mass1[2]{};
  double penalty[2]{};
  uint8_t used_mask = 0;
  bool occupied = false;
};

template <size_t Words>
struct CompiledWideLayerTemplate {
  double q = 0.0;
  double p = 0.0;
  bool toggles_observable = false;
  bool has_retiring_terms = false;
  size_t surviving_term_count = 0;
  std::array<uint64_t, Words> surviving_masks{};
  std::array<uint8_t, Words> projection_dst_words{};
  std::array<uint8_t, Words> projection_dst_offsets{};
  std::array<uint64_t, Words> projected_fault_mask_words{};
  std::vector<uint32_t> fault_target_word_indices;
  std::vector<uint64_t> fault_target_bit_masks;
  std::vector<uint8_t> fault_word_indices;
  std::vector<uint64_t> fault_bit_masks;
  std::vector<uint8_t> fault_was_active_before;
  std::vector<double> current_costs;
  std::vector<double> next_costs;
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

struct ActiveDetcostEvent {
  int detector = -1;
  size_t valid_low = 0;
  double cost = INF;
};

struct ActiveDetcostHeapEntry {
  size_t valid_low = 0;
  double cost = INF;

  bool operator>(const ActiveDetcostHeapEntry& other) const {
    return cost > other.cost || (cost == other.cost && valid_low > other.valid_low);
  }
};

template <typename LayerT>
void build_future_active_detcost_transitions(const std::vector<Fault>& faults, size_t num_detectors,
                                             std::vector<LayerT>* layers,
                                             std::vector<double>* initial_future_detcost) {
  std::vector<size_t> first_seen(num_detectors, std::numeric_limits<size_t>::max());
  for (size_t fault_index = 0; fault_index < faults.size(); ++fault_index) {
    for (int detector : faults[fault_index].detectors) {
      size_t& seen = first_seen[(size_t)detector];
      if (seen == std::numeric_limits<size_t>::max()) {
        seen = fault_index;
      }
    }
  }

  std::vector<std::vector<ActiveDetcostEvent>> events_by_high(faults.size());
  std::vector<std::pair<size_t, int>> first_seen_and_detector;
  std::vector<int> active_detectors;
  for (size_t fault_index = 0; fault_index < faults.size(); ++fault_index) {
    const auto& fault = faults[fault_index];
    if (fault.detectors.empty()) {
      continue;
    }

    first_seen_and_detector.clear();
    first_seen_and_detector.reserve(fault.detectors.size());
    for (int detector : fault.detectors) {
      size_t seen = first_seen[(size_t)detector];
      if (seen == std::numeric_limits<size_t>::max() || seen > fault_index) {
        throw std::runtime_error("Invalid first-seen detector state while preparing detcost.");
      }
      first_seen_and_detector.push_back({seen, detector});
    }
    std::sort(first_seen_and_detector.begin(), first_seen_and_detector.end());
    first_seen_and_detector.erase(
        std::unique(first_seen_and_detector.begin(), first_seen_and_detector.end(),
                    [](const auto& a, const auto& b) { return a.second == b.second; }),
        first_seen_and_detector.end());

    active_detectors.clear();
    for (size_t pos = 0; pos < first_seen_and_detector.size();) {
      const size_t low = first_seen_and_detector[pos].first;
      while (pos < first_seen_and_detector.size() && first_seen_and_detector[pos].first == low) {
        active_detectors.push_back(first_seen_and_detector[pos].second);
        ++pos;
      }
      size_t high = fault_index;
      if (pos < first_seen_and_detector.size()) {
        high = std::min(high, first_seen_and_detector[pos].first - 1);
      }
      if (low > high || active_detectors.empty()) {
        continue;
      }
      const double cost = fault.likelihood_cost / active_detectors.size();
      auto& events = events_by_high[high];
      events.reserve(events.size() + active_detectors.size());
      for (int detector : active_detectors) {
        events.push_back({detector, low, cost});
      }
    }
  }

  std::vector<std::priority_queue<ActiveDetcostHeapEntry, std::vector<ActiveDetcostHeapEntry>,
                                  std::greater<ActiveDetcostHeapEntry>>>
      heaps(num_detectors);
  std::vector<std::vector<double>> future_costs_by_layer(faults.size());

  for (size_t layer_index = faults.size(); layer_index-- > 0;) {
    for (const auto& event : events_by_high[layer_index]) {
      heaps[(size_t)event.detector].push({event.valid_low, event.cost});
    }

    const auto& active = (*layers)[layer_index].current_active_detectors;
    auto& costs = future_costs_by_layer[layer_index];
    costs.resize(active.size(), INF);
    for (size_t local = 0; local < active.size(); ++local) {
      auto& heap = heaps[(size_t)active[local]];
      while (!heap.empty() && heap.top().valid_low > layer_index) {
        heap.pop();
      }
      if (!heap.empty()) {
        costs[local] = heap.top().cost;
      }
    }
  }

  std::vector<double> initial(num_detectors, INF);
  if (!layers->empty()) {
    const auto& active = layers->front().current_active_detectors;
    const auto& costs = future_costs_by_layer.front();
    for (size_t local = 0; local < active.size(); ++local) {
      initial[(size_t)active[local]] = costs[local];
    }
  }

  for (size_t fault_index = 0; fault_index < faults.size(); ++fault_index) {
    auto& layer = (*layers)[fault_index];
    const auto& fault = faults[fault_index];

    std::vector<int32_t> current_to_next(layer.current_active_detectors.size(), -1);
    for (size_t next_local = 0; next_local < layer.surviving_local_indices.size(); ++next_local) {
      current_to_next[(size_t)layer.surviving_local_indices[next_local]] = (int32_t)next_local;
    }

    layer.next_frontier_costs.resize(layer.surviving_local_indices.size(), INF);
    if (fault_index + 1 < faults.size()) {
      const auto& next_costs = future_costs_by_layer[fault_index + 1];
      for (size_t next_local = 0; next_local < layer.surviving_local_indices.size(); ++next_local) {
        if (next_local < next_costs.size()) {
          layer.next_frontier_costs[next_local] = next_costs[next_local];
        }
      }
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

    for (int detector : fault.detectors) {
      auto it = std::find(layer.current_active_detectors.begin(),
                          layer.current_active_detectors.end(), detector);
      if (it == layer.current_active_detectors.end()) {
        throw std::runtime_error("Missing detector in active frontier while preparing detcost.");
      }
      uint32_t local = (uint32_t)std::distance(layer.current_active_detectors.begin(), it);
      int32_t next_local = current_to_next[local];
      double next_cost = INF;
      if (next_local >= 0 && fault_index + 1 < faults.size() &&
          (size_t)next_local < future_costs_by_layer[fault_index + 1].size()) {
        next_cost = future_costs_by_layer[fault_index + 1][(size_t)next_local];
      }
      transition.fault_local_indices.push_back(local);
      transition.next_local_indices.push_back(next_local);
      transition.current_costs.push_back(future_costs_by_layer[fault_index][local]);
      transition.next_costs.push_back(next_cost);
    }
  }

  if (initial_future_detcost != nullptr) {
    *initial_future_detcost = std::move(initial);
  }
}

template <typename LayerT>
void scale_future_detcost_transitions(std::vector<LayerT>* layers,
                                      std::vector<double>* initial_future_detcost, double scale) {
  if (scale == 1.0) {
    return;
  }
  auto scale_value = [scale](double value) { return std::isfinite(value) ? value * scale : value; };
  for (auto& value : *initial_future_detcost) {
    value = scale_value(value);
  }
  for (auto& layer : *layers) {
    for (auto& value : layer.next_frontier_costs) {
      value = scale_value(value);
    }
    for (auto& value : layer.detcost_transition.current_costs) {
      value = scale_value(value);
    }
    for (auto& value : layer.detcost_transition.next_costs) {
      value = scale_value(value);
    }
  }
}

size_t num_state_words(size_t num_bits) {
  return (num_bits + 63) >> 6;
}

TESSERACT_ALWAYS_INLINE size_t detector_word_index(size_t detector) {
  return detector >> 6;
}

TESSERACT_ALWAYS_INLINE uint64_t detector_word_mask(size_t detector) {
  return uint64_t{1} << (detector & 63);
}

TESSERACT_ALWAYS_INLINE uint64_t compact_bits_u64(uint64_t value, uint64_t mask) {
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

double compute_initial_penalty_for_active_detectors(
    const std::vector<uint32_t>& active_detector_word_indices,
    const std::vector<uint64_t>& active_detector_bit_masks,
    const std::vector<double>& active_detector_costs,
    const std::vector<uint64_t>& actual_detector_words) {
  double total = 0.0;
  for (size_t k = 0; k < active_detector_costs.size(); ++k) {
    if ((actual_detector_words[active_detector_word_indices[k]] & active_detector_bit_masks[k]) ==
        0) {
      continue;
    }
    double best = active_detector_costs[k];
    if (best == INF) {
      return INF;
    }
    total += best;
  }
  return total;
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

template <size_t Words>
TESSERACT_ALWAYS_INLINE double total_entry_mass(const FixedWideStateEntry<Words>& entry) {
  return entry.mass0 + entry.mass1;
}

TESSERACT_ALWAYS_INLINE uint64_t mix_splitmix64(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
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

template <size_t Words>
bool fixed_wide_state_less(const FixedWideStateWords<Words>& a,
                           const FixedWideStateWords<Words>& b) {
  for (size_t k = Words; k-- > 0;) {
    if (a[k] != b[k]) {
      return a[k] < b[k];
    }
  }
  return false;
}

template <size_t Words>
bool fixed_wide_state_zero(const FixedWideStateWords<Words>& state_words) {
  for (size_t k = 0; k < Words; ++k) {
    if (state_words[k] != 0) {
      return false;
    }
  }
  return true;
}

template <size_t Words>
void xor_compiled_wide_state(FixedWideStateWords<Words>* state_words,
                             const std::array<uint64_t, Words>& mask_words) {
  for (size_t k = 0; k < Words; ++k) {
    (*state_words)[k] ^= mask_words[k];
  }
}

template <size_t Words>
TESSERACT_ALWAYS_INLINE uint64_t
hash_fixed_wide_state(const FixedWideStateWords<Words>& state_words) {
  uint64_t hash = 0x123456789abcdef0ULL;
  for (size_t k = 0; k < Words; ++k) {
    hash ^= mix_splitmix64(state_words[k] + 0x9e3779b97f4a7c15ULL * (k + 1));
    hash = std::rotl(hash, 21);
  }
  return hash;
}

template <size_t Words>
struct FixedWideStateHash {
  size_t operator()(const FixedWideStateWords<Words>& state_words) const {
    return static_cast<size_t>(hash_fixed_wide_state(state_words));
  }
};

template <size_t Words>
struct GradientCandidateTransition {
  size_t parent_index;
  FixedWideStateWords<Words> child_state;
  bool present;
};

struct GradientTransition {
  size_t parent_index;
  size_t child_index;
  bool present;
};

struct GradientLayerTape {
  double normalizer;
  std::vector<GradientTransition> transitions;
};

template <size_t Words>
void ensure_pair_bucket_capacity(std::vector<FixedWidePairBucket<Words>>* buckets,
                                 size_t num_parents) {
  const size_t required = std::bit_ceil(std::max<size_t>(16, num_parents * 2));
  if (buckets->size() < required) {
    buckets->resize(required);
  }
}

template <size_t Words>
void clear_pair_buckets(std::vector<FixedWidePairBucket<Words>>* buckets,
                        std::vector<size_t>* used_bucket_indices) {
  for (size_t index : *used_bucket_indices) {
    (*buckets)[index].occupied = false;
    (*buckets)[index].used_mask = 0;
  }
  used_bucket_indices->clear();
}

template <size_t Words>
TESSERACT_ALWAYS_INLINE size_t find_or_insert_pair_bucket(
    std::vector<FixedWidePairBucket<Words>>* buckets, std::vector<size_t>* used_bucket_indices,
    const FixedWideStateWords<Words>& key) {
  const size_t mask = buckets->size() - 1;
  size_t index = hash_fixed_wide_state(key) & mask;
  while ((*buckets)[index].occupied) {
    if ((*buckets)[index].key == key) {
      return index;
    }
    index = (index + 1) & mask;
  }

  auto& bucket = (*buckets)[index];
  bucket.occupied = true;
  bucket.key = key;
  bucket.used_mask = 0;
  used_bucket_indices->push_back(index);
  return index;
}

template <size_t Words>
TESSERACT_ALWAYS_INLINE void accumulate_pair_bucket_slot(FixedWidePairBucket<Words>* bucket,
                                                         uint8_t slot, double mass0, double mass1,
                                                         double penalty) {
  const uint8_t bit = (uint8_t)(1u << slot);
  if ((bucket->used_mask & bit) == 0) {
    bucket->mass0[slot] = mass0;
    bucket->mass1[slot] = mass1;
    bucket->penalty[slot] = penalty;
    bucket->used_mask |= bit;
  } else {
    bucket->mass0[slot] += mass0;
    bucket->mass1[slot] += mass1;
  }
}

template <size_t Words>
FixedWideStateWords<Words> project_compiled_wide_state(
    const FixedWideStateWords<Words>& state_words, const CompiledWideLayerTemplate<Words>& layer) {
  FixedWideStateWords<Words> out{};
  for (size_t src_word = 0; src_word < Words; ++src_word) {
    const uint64_t mask = layer.surviving_masks[src_word];
    if (mask == 0) {
      continue;
    }
    const uint64_t packed = compact_bits_u64(state_words[src_word], mask);
    const size_t dst_word = layer.projection_dst_words[src_word];
    const uint8_t shift = layer.projection_dst_offsets[src_word];
    out[dst_word] |= packed << shift;
    if constexpr (Words > 1) {
      if (shift != 0 && dst_word + 1 < Words) {
        out[dst_word + 1] |= packed >> (64 - shift);
      }
    }
  }
  return out;
}

template <bool ComputePenalties, bool CheckRetiringTerms, size_t Words>
TESSERACT_ALWAYS_INLINE BranchPenaltyUpdate compute_compiled_wide_branch_update(
    const FixedWideStateWords<Words>& base_state_words, double current_penalty,
    const std::vector<uint64_t>& actual_detector_words,
    const CompiledWideLayerTemplate<Words>& layer) {
  BranchPenaltyUpdate update;
  update.absent_penalty = ComputePenalties ? current_penalty : 0.0;
  update.present_penalty = ComputePenalties ? current_penalty : 0.0;

  for (size_t k = 0; k < layer.surviving_term_count; ++k) {
    const bool state_bit =
        layer.fault_was_active_before[k] &&
        ((base_state_words[layer.fault_word_indices[k]] & layer.fault_bit_masks[k]) != 0);
    const bool target_bit = (actual_detector_words[layer.fault_target_word_indices[k]] &
                             layer.fault_target_bit_masks[k]) != 0;
    const bool mismatch = state_bit ^ target_bit;

    if constexpr (ComputePenalties) {
      const double prev_contrib =
          (layer.fault_was_active_before[k] && mismatch) ? layer.current_costs[k] : 0.0;
      const double next_contrib = mismatch ? layer.next_costs[k] : 0.0;
      update.absent_penalty += next_contrib - prev_contrib;
      update.present_penalty += (layer.next_costs[k] - next_contrib) - prev_contrib;
    }
  }

  if constexpr (CheckRetiringTerms) {
    for (size_t k = layer.surviving_term_count; k < layer.fault_target_word_indices.size(); ++k) {
      const bool state_bit =
          layer.fault_was_active_before[k] &&
          ((base_state_words[layer.fault_word_indices[k]] & layer.fault_bit_masks[k]) != 0);
      const bool target_bit = (actual_detector_words[layer.fault_target_word_indices[k]] &
                               layer.fault_target_bit_masks[k]) != 0;
      const bool mismatch = state_bit ^ target_bit;

      if (mismatch) {
        update.absent_valid = false;
      } else {
        update.present_valid = false;
      }

      if constexpr (ComputePenalties) {
        const double prev_contrib =
            (layer.fault_was_active_before[k] && mismatch) ? layer.current_costs[k] : 0.0;
        update.absent_penalty -= prev_contrib;
        update.present_penalty -= prev_contrib;
      }
    }
  }

  return update;
}

template <size_t Words, bool ComputePenalties, bool CheckRetiringTerms>
void expand_compiled_layer_into_pair_buckets(
    const std::vector<FixedWideStateEntry<Words>>& beam_entries,
    std::vector<FixedWidePairBucket<Words>>* pair_buckets, std::vector<size_t>* used_bucket_indices,
    const std::vector<uint64_t>& actual_detector_words,
    const CompiledWideLayerTemplate<Words>& layer, TesseractTrellisDecoder* decoder) {
  for (const auto& item : beam_entries) {
    ++decoder->num_states_expanded;
    BranchPenaltyUpdate update =
        compute_compiled_wide_branch_update<ComputePenalties, CheckRetiringTerms>(
            item.state_words, item.penalty, actual_detector_words, layer);

    if (!update.absent_valid && !update.present_valid) {
      continue;
    }

    FixedWideStateWords<Words> projected_state =
        project_compiled_wide_state(item.state_words, layer);
    FixedWideStateWords<Words> projected_toggled = projected_state;
    xor_compiled_wide_state(&projected_toggled, layer.projected_fault_mask_words);
    const bool projected_is_key = !fixed_wide_state_less(projected_toggled, projected_state);
    const auto& bucket_key = projected_is_key ? projected_state : projected_toggled;
    const uint8_t absent_slot = projected_is_key ? 0 : 1;
    const uint8_t present_slot = projected_toggled == bucket_key ? 0 : 1;
    const size_t bucket_index =
        find_or_insert_pair_bucket(pair_buckets, used_bucket_indices, bucket_key);
    auto& bucket = (*pair_buckets)[bucket_index];
    const bool keep_absent = update.absent_valid && layer.q != 0.0;
    const bool keep_present = update.present_valid && layer.p != 0.0;

    if (keep_absent) {
      accumulate_pair_bucket_slot(&bucket, absent_slot, item.mass0 * layer.q, item.mass1 * layer.q,
                                  update.absent_penalty);
    }
    if (keep_present) {
      if (layer.toggles_observable) {
        accumulate_pair_bucket_slot(&bucket, present_slot, item.mass1 * layer.p,
                                    item.mass0 * layer.p, update.present_penalty);
      } else {
        accumulate_pair_bucket_slot(&bucket, present_slot, item.mass0 * layer.p,
                                    item.mass1 * layer.p, update.present_penalty);
      }
    }
  }
}

template <size_t Words>
double normalize_compiled_items(std::vector<FixedWideStateEntry<Words>>* items) {
  double total = 0.0;
  for (const auto& item : *items) {
    total += total_entry_mass(item);
  }
  if (total == 0.0) {
    return 0.0;
  }
  for (auto& item : *items) {
    item.mass0 /= total;
    item.mass1 /= total;
  }
  return total;
}

template <size_t Words>
void merge_equal_compiled_keys_inplace(std::vector<FixedWideStateEntry<Words>>* items) {
  if (items->empty()) {
    return;
  }
  std::sort(items->begin(), items->end(),
            [](const FixedWideStateEntry<Words>& a, const FixedWideStateEntry<Words>& b) {
              return fixed_wide_state_less(a.state_words, b.state_words);
            });

  size_t out = 0;
  for (size_t i = 1; i < items->size(); ++i) {
    if ((*items)[i].state_words == (*items)[out].state_words) {
      (*items)[out].mass0 += (*items)[i].mass0;
      (*items)[out].mass1 += (*items)[i].mass1;
    } else {
      ++out;
      if (out != i) {
        (*items)[out] = std::move((*items)[i]);
      }
    }
  }
  items->resize(out + 1);
}

template <size_t Words>
bool compiled_state_score_greater(const FixedWideStateEntry<Words>& a,
                                  const FixedWideStateEntry<Words>& b) {
  if (a.score != b.score) {
    return a.score > b.score;
  }
  return fixed_wide_state_less(a.state_words, b.state_words);
}

template <size_t Words>
size_t keep_top_compiled_states(std::vector<FixedWideStateEntry<Words>>* entries, size_t beam_width,
                                double beam_eps, TesseractTrellisRankingMode ranking_mode) {
  if (entries->empty()) {
    return 0;
  }

  double total_mass = 0.0;
  for (auto& entry : *entries) {
    const double mass = total_entry_mass(entry);
    entry.score = score_mass_and_penalty(mass, entry.penalty, ranking_mode);
    if (beam_eps > 0.0) {
      total_mass += mass;
    }
  }

  if (entries->size() > beam_width) {
    std::nth_element(entries->begin(), entries->begin() + beam_width, entries->end(),
                     [](const FixedWideStateEntry<Words>& a, const FixedWideStateEntry<Words>& b) {
                       return compiled_state_score_greater(a, b);
                     });
    entries->resize(beam_width);
  } else if (beam_eps <= 0.0) {
    return entries->size();
  }

  if (beam_eps <= 0.0 || total_mass <= 0.0) {
    return entries->size();
  }

  std::sort(entries->begin(), entries->end(),
            [](const FixedWideStateEntry<Words>& a, const FixedWideStateEntry<Words>& b) {
              return compiled_state_score_greater(a, b);
            });
  const double retained_target_mass = total_mass * (1.0 - beam_eps);
  double retained_mass = 0.0;
  size_t keep_count = 0;
  while (keep_count < entries->size()) {
    retained_mass += total_entry_mass((*entries)[keep_count]);
    ++keep_count;
    if (retained_mass >= retained_target_mass) {
      break;
    }
  }
  entries->resize(keep_count);
  return keep_count;
}

template <size_t Words>
std::vector<CompiledWideLayerTemplate<Words>> compile_wide_layers(
    const std::vector<TesseractTrellisWideLayerTemplate>& layers) {
  std::vector<CompiledWideLayerTemplate<Words>> compiled_layers;
  compiled_layers.reserve(layers.size());
  for (const auto& layer : layers) {
    if (num_state_words(layer.current_active_detectors.size()) > Words ||
        layer.projected_fault_mask_words.size() > Words) {
      throw std::invalid_argument("Compiled wide kernel word count is smaller than the frontier.");
    }

    CompiledWideLayerTemplate<Words> compiled;
    compiled.q = layer.q;
    compiled.p = layer.p;
    if (layer.obs_mask > 1) {
      throw std::invalid_argument("tesseract_trellis currently supports at most one observable");
    }
    compiled.toggles_observable = layer.obs_mask != 0;

    std::array<uint64_t, Words> surviving_masks{};
    for (uint32_t current_local : layer.surviving_local_indices) {
      surviving_masks[current_local >> 6] |= uint64_t{1} << (current_local & 63);
    }
    size_t next_offset = 0;
    for (size_t src_word = 0; src_word < Words; ++src_word) {
      compiled.surviving_masks[src_word] = surviving_masks[src_word];
      compiled.projection_dst_words[src_word] = static_cast<uint8_t>(next_offset >> 6);
      compiled.projection_dst_offsets[src_word] = static_cast<uint8_t>(next_offset & 63);
      next_offset += std::popcount(surviving_masks[src_word]);
    }

    for (size_t k = 0; k < layer.projected_fault_mask_words.size(); ++k) {
      compiled.projected_fault_mask_words[k] = layer.projected_fault_mask_words[k];
    }

    const auto& transition = layer.detcost_transition;
    const size_t term_count = transition.fault_local_indices.size();
    compiled.fault_target_word_indices.reserve(term_count);
    compiled.fault_target_bit_masks.reserve(term_count);
    compiled.fault_word_indices.reserve(term_count);
    compiled.fault_bit_masks.reserve(term_count);
    compiled.fault_was_active_before.reserve(term_count);
    compiled.current_costs.reserve(term_count);
    compiled.next_costs.reserve(term_count);
    auto append_term = [&](size_t idx) {
      const uint32_t local = transition.fault_local_indices[idx];
      const uint32_t detector = (uint32_t)layer.current_active_detectors[local];
      compiled.fault_target_word_indices.push_back((uint32_t)detector_word_index(detector));
      compiled.fault_target_bit_masks.push_back(detector_word_mask(detector));
      compiled.fault_word_indices.push_back(static_cast<uint8_t>(local >> 6));
      compiled.fault_bit_masks.push_back(uint64_t{1} << (local & 63));
      compiled.fault_was_active_before.push_back(local < layer.previous_width);
      compiled.current_costs.push_back(transition.current_costs[idx]);
      compiled.next_costs.push_back(transition.next_costs[idx]);
    };
    for (size_t idx = 0; idx < term_count; ++idx) {
      if (transition.next_local_indices[idx] >= 0) {
        append_term(idx);
      }
    }
    compiled.surviving_term_count = compiled.fault_target_word_indices.size();
    compiled.has_retiring_terms = compiled.surviving_term_count != term_count;
    for (size_t idx = 0; idx < term_count; ++idx) {
      if (transition.next_local_indices[idx] < 0) {
        append_term(idx);
      }
    }

    compiled_layers.push_back(std::move(compiled));
  }
  return compiled_layers;
}

template <size_t Words>
struct CompiledWideKernel final : TesseractTrellisWideKernelBase {
  explicit CompiledWideKernel(std::vector<CompiledWideLayerTemplate<Words>> layers_,
                              std::vector<uint32_t> initial_detector_word_indices_,
                              std::vector<uint64_t> initial_detector_bit_masks_,
                              std::vector<double> initial_detector_costs_,
                              size_t max_frontier_width_)
      : layers(std::move(layers_)),
        initial_detector_word_indices(std::move(initial_detector_word_indices_)),
        initial_detector_bit_masks(std::move(initial_detector_bit_masks_)),
        initial_detector_costs(std::move(initial_detector_costs_)),
        max_frontier_width(max_frontier_width_) {}

  void decode_shot(TesseractTrellisDecoder* decoder,
                   const std::vector<uint64_t>& detections) const override {
    auto& actual_detector_words = decoder->actual_detector_words_scratch;
    std::fill(actual_detector_words.begin(), actual_detector_words.end(), 0);
    for (uint64_t d : detections) {
      if (d >= decoder->num_detectors) {
        decoder->low_confidence_flag = true;
        return;
      }
      const size_t word = detector_word_index((size_t)d);
      const uint64_t mask = detector_word_mask((size_t)d);
      if ((decoder->all_possible_detector_words[word] & mask) == 0) {
        decoder->low_confidence_flag = true;
        return;
      }
      actual_detector_words[word] ^= mask;
    }

    decoder->max_frontier_width_seen = max_frontier_width;

    double initial_penalty = 0.0;
    if (decoder->config.ranking_mode != TesseractTrellisRankingMode::MassOnly && !layers.empty()) {
      initial_penalty = compute_initial_penalty_for_active_detectors(
          initial_detector_word_indices, initial_detector_bit_masks, initial_detector_costs,
          actual_detector_words);
    }

    std::vector<FixedWideStateEntry<Words>> beam_entries;
    std::vector<FixedWideStateEntry<Words>> next_entries;
    std::vector<FixedWidePairBucket<Words>> pair_buckets;
    std::vector<size_t> used_bucket_indices;
    beam_entries.reserve(decoder->config.beam_width * 2 + 2);
    next_entries.reserve(decoder->config.beam_width * 4 + 4);
    beam_entries.push_back({{}, 1.0, 0.0, initial_penalty});
    decoder->max_beam_size_seen = 1;

    const bool compute_penalties =
        decoder->config.ranking_mode != TesseractTrellisRankingMode::MassOnly;
    for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
      const auto& layer = layers[layer_index];

      ensure_pair_bucket_capacity(&pair_buckets, beam_entries.size());
      clear_pair_buckets(&pair_buckets, &used_bucket_indices);

      auto t0 = std::chrono::high_resolution_clock::now();

      if (decoder->config.verbose) {
        std::cout << "expanding layer " << layer_index << " / " << (layers.size() - 1) << std::endl;
        std::cout << "states to expand = " << beam_entries.size() << std::endl;
      }
      if (compute_penalties) {
        if (layer.has_retiring_terms) {
          expand_compiled_layer_into_pair_buckets<Words, true, true>(
              beam_entries, &pair_buckets, &used_bucket_indices, actual_detector_words, layer,
              decoder);
        } else {
          expand_compiled_layer_into_pair_buckets<Words, true, false>(
              beam_entries, &pair_buckets, &used_bucket_indices, actual_detector_words, layer,
              decoder);
        }
      } else if (layer.has_retiring_terms) {
        expand_compiled_layer_into_pair_buckets<Words, false, true>(
            beam_entries, &pair_buckets, &used_bucket_indices, actual_detector_words, layer,
            decoder);
      } else {
        expand_compiled_layer_into_pair_buckets<Words, false, false>(
            beam_entries, &pair_buckets, &used_bucket_indices, actual_detector_words, layer,
            decoder);
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      decoder->time_expand_seconds +=
          std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;

      auto t2a = std::chrono::high_resolution_clock::now();
      next_entries.clear();
      next_entries.reserve(used_bucket_indices.size() * 2);
      for (size_t index : used_bucket_indices) {
        auto& bucket = pair_buckets[index];
        if ((bucket.used_mask & 1u) != 0) {
          next_entries.push_back({bucket.key, bucket.mass0[0], bucket.mass1[0], bucket.penalty[0]});
        }
        if ((bucket.used_mask & 2u) != 0) {
          auto other_state = bucket.key;
          xor_compiled_wide_state(&other_state, layer.projected_fault_mask_words);
          next_entries.push_back(
              {std::move(other_state), bucket.mass0[1], bucket.mass1[1], bucket.penalty[1]});
        }
      }
      beam_entries.swap(next_entries);
      auto t2 = std::chrono::high_resolution_clock::now();
      decoder->time_collapse_seconds +=
          std::chrono::duration_cast<std::chrono::microseconds>(t2 - t2a).count() / 1e6;

      const size_t kept_states =
          keep_top_compiled_states(&beam_entries, decoder->config.beam_width,
                                   decoder->config.beam_eps, decoder->config.ranking_mode);
      const double normalizer = normalize_compiled_items(&beam_entries);
      record_kept_state_count(decoder, beam_entries.empty() ? 0 : kept_states);
      if (beam_entries.empty() || !std::isfinite(normalizer) || normalizer <= 0.0) {
        decoder->low_confidence_flag = true;
        return;
      }
      decoder->syndrome_log_probability += std::log(normalizer);
      decoder->num_states_merged += kept_states;
      decoder->max_beam_size_seen = std::max(decoder->max_beam_size_seen, kept_states);
      auto t3 = std::chrono::high_resolution_clock::now();
      decoder->time_truncate_seconds +=
          std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / 1e6;
    }

    auto tr0 = std::chrono::high_resolution_clock::now();
    for (const auto& item : beam_entries) {
      if (!fixed_wide_state_zero(item.state_words)) {
        continue;
      }
      decoder->total_mass_obs0 += item.mass0;
      decoder->total_mass_obs1 += item.mass1;
    }
    if (decoder->total_mass_obs0 == 0.0 && decoder->total_mass_obs1 == 0.0) {
      decoder->low_confidence_flag = true;
      return;
    }
    decoder->syndrome_log_probability +=
        std::log(decoder->total_mass_obs0 + decoder->total_mass_obs1);
    decoder->predicted_obs_mask = decoder->total_mass_obs1 > decoder->total_mass_obs0 ? 1 : 0;
    auto tr1 = std::chrono::high_resolution_clock::now();
    decoder->time_reconstruct_seconds +=
        std::chrono::duration_cast<std::chrono::microseconds>(tr1 - tr0).count() / 1e6;
  }

  std::vector<double> decode_shot_with_observable_logit_gradient(
      TesseractTrellisDecoder* decoder, const std::vector<uint64_t>& detections) const override {
    return decode_shot_with_gradient(decoder, detections, false);
  }

  std::vector<double> decode_shot_with_syndrome_log_probability_gradient(
      TesseractTrellisDecoder* decoder, const std::vector<uint64_t>& detections) const override {
    return decode_shot_with_gradient(decoder, detections, true);
  }

  std::vector<double> decode_shot_with_gradient(TesseractTrellisDecoder* decoder,
                                                const std::vector<uint64_t>& detections,
                                                bool syndrome_objective) const {
    auto invalid_gradient = [&]() {
      return std::vector<double>(layers.size(), std::numeric_limits<double>::quiet_NaN());
    };
    auto& actual_detector_words = decoder->actual_detector_words_scratch;
    std::fill(actual_detector_words.begin(), actual_detector_words.end(), 0);
    for (uint64_t d : detections) {
      if (d >= decoder->num_detectors) {
        decoder->low_confidence_flag = true;
        return invalid_gradient();
      }
      const size_t word = detector_word_index((size_t)d);
      const uint64_t mask = detector_word_mask((size_t)d);
      if ((decoder->all_possible_detector_words[word] & mask) == 0) {
        decoder->low_confidence_flag = true;
        return invalid_gradient();
      }
      actual_detector_words[word] ^= mask;
    }

    decoder->max_frontier_width_seen = max_frontier_width;
    double initial_penalty = 0.0;
    if (decoder->config.ranking_mode != TesseractTrellisRankingMode::MassOnly && !layers.empty()) {
      initial_penalty = compute_initial_penalty_for_active_detectors(
          initial_detector_word_indices, initial_detector_bit_masks, initial_detector_costs,
          actual_detector_words);
    }

    using Entry = FixedWideStateEntry<Words>;
    using State = FixedWideStateWords<Words>;
    std::vector<std::vector<Entry>> states_by_layer;
    states_by_layer.reserve(layers.size() + 1);
    states_by_layer.push_back({Entry{{}, 1.0, 0.0, initial_penalty}});
    std::vector<GradientLayerTape> tape;
    tape.reserve(layers.size());
    std::vector<FixedWidePairBucket<Words>> pair_buckets;
    std::vector<size_t> used_bucket_indices;
    std::vector<Entry> next_entries;
    std::vector<GradientCandidateTransition<Words>> candidate_transitions;
    decoder->max_beam_size_seen = 1;

    const bool compute_penalties =
        decoder->config.ranking_mode != TesseractTrellisRankingMode::MassOnly;
    for (const auto& layer : layers) {
      const auto& parent_entries = states_by_layer.back();
      ensure_pair_bucket_capacity(&pair_buckets, parent_entries.size());
      clear_pair_buckets(&pair_buckets, &used_bucket_indices);
      candidate_transitions.clear();
      candidate_transitions.reserve(parent_entries.size() * 2);

      for (size_t parent_index = 0; parent_index < parent_entries.size(); ++parent_index) {
        const auto& item = parent_entries[parent_index];
        ++decoder->num_states_expanded;
        BranchPenaltyUpdate update;
        if (compute_penalties) {
          if (layer.has_retiring_terms) {
            update = compute_compiled_wide_branch_update<true, true>(item.state_words, item.penalty,
                                                                     actual_detector_words, layer);
          } else {
            update = compute_compiled_wide_branch_update<true, false>(
                item.state_words, item.penalty, actual_detector_words, layer);
          }
        } else if (layer.has_retiring_terms) {
          update = compute_compiled_wide_branch_update<false, true>(item.state_words, item.penalty,
                                                                    actual_detector_words, layer);
        } else {
          update = compute_compiled_wide_branch_update<false, false>(item.state_words, item.penalty,
                                                                     actual_detector_words, layer);
        }
        if (!update.absent_valid && !update.present_valid) {
          continue;
        }

        State projected_state = project_compiled_wide_state(item.state_words, layer);
        State projected_toggled = projected_state;
        xor_compiled_wide_state(&projected_toggled, layer.projected_fault_mask_words);
        const bool projected_is_key = !fixed_wide_state_less(projected_toggled, projected_state);
        const auto& bucket_key = projected_is_key ? projected_state : projected_toggled;
        const uint8_t absent_slot = projected_is_key ? 0 : 1;
        const uint8_t present_slot = projected_toggled == bucket_key ? 0 : 1;
        const size_t bucket_index =
            find_or_insert_pair_bucket(&pair_buckets, &used_bucket_indices, bucket_key);
        auto& bucket = pair_buckets[bucket_index];
        const bool keep_absent = update.absent_valid && layer.q != 0.0;
        const bool keep_present = update.present_valid && layer.p != 0.0;

        if (keep_absent) {
          accumulate_pair_bucket_slot(&bucket, absent_slot, item.mass0 * layer.q,
                                      item.mass1 * layer.q, update.absent_penalty);
          candidate_transitions.push_back({parent_index, projected_state, false});
        }
        if (keep_present) {
          if (layer.toggles_observable) {
            accumulate_pair_bucket_slot(&bucket, present_slot, item.mass1 * layer.p,
                                        item.mass0 * layer.p, update.present_penalty);
          } else {
            accumulate_pair_bucket_slot(&bucket, present_slot, item.mass0 * layer.p,
                                        item.mass1 * layer.p, update.present_penalty);
          }
          candidate_transitions.push_back({parent_index, projected_toggled, true});
        }
      }

      next_entries.clear();
      next_entries.reserve(used_bucket_indices.size() * 2);
      for (size_t index : used_bucket_indices) {
        auto& bucket = pair_buckets[index];
        if ((bucket.used_mask & 1u) != 0) {
          next_entries.push_back({bucket.key, bucket.mass0[0], bucket.mass1[0], bucket.penalty[0]});
        }
        if ((bucket.used_mask & 2u) != 0) {
          auto other_state = bucket.key;
          xor_compiled_wide_state(&other_state, layer.projected_fault_mask_words);
          next_entries.push_back(
              {std::move(other_state), bucket.mass0[1], bucket.mass1[1], bucket.penalty[1]});
        }
      }

      const size_t kept_states =
          keep_top_compiled_states(&next_entries, decoder->config.beam_width,
                                   decoder->config.beam_eps, decoder->config.ranking_mode);
      double normalizer = 0.0;
      for (const auto& item : next_entries) {
        normalizer += total_entry_mass(item);
      }
      record_kept_state_count(decoder, next_entries.empty() ? 0 : kept_states);
      if (next_entries.empty() || !std::isfinite(normalizer) || normalizer <= 0.0) {
        decoder->low_confidence_flag = true;
        return invalid_gradient();
      }
      for (auto& item : next_entries) {
        item.mass0 /= normalizer;
        item.mass1 /= normalizer;
      }
      if (syndrome_objective) {
        decoder->syndrome_log_probability += std::log(normalizer);
      }
      decoder->num_states_merged += kept_states;
      decoder->max_beam_size_seen = std::max(decoder->max_beam_size_seen, kept_states);

      std::unordered_map<State, size_t, FixedWideStateHash<Words>> child_indices;
      child_indices.reserve(next_entries.size() * 2);
      for (size_t child_index = 0; child_index < next_entries.size(); ++child_index) {
        child_indices.emplace(next_entries[child_index].state_words, child_index);
      }
      std::vector<GradientTransition> retained_transitions;
      retained_transitions.reserve(candidate_transitions.size());
      for (const auto& transition : candidate_transitions) {
        auto child = child_indices.find(transition.child_state);
        if (child != child_indices.end()) {
          retained_transitions.push_back(
              {transition.parent_index, child->second, transition.present});
        }
      }
      tape.push_back({normalizer, std::move(retained_transitions)});
      states_by_layer.push_back(next_entries);
    }

    const auto& final_entries = states_by_layer.back();
    for (const auto& item : final_entries) {
      if (!fixed_wide_state_zero(item.state_words)) {
        continue;
      }
      decoder->total_mass_obs0 += item.mass0;
      decoder->total_mass_obs1 += item.mass1;
    }
    const double final_syndrome_mass = decoder->total_mass_obs0 + decoder->total_mass_obs1;
    const bool invalid_observable_mass =
        !syndrome_objective && (decoder->total_mass_obs0 <= 0.0 || decoder->total_mass_obs1 <= 0.0);
    if (!std::isfinite(decoder->total_mass_obs0) || !std::isfinite(decoder->total_mass_obs1) ||
        !std::isfinite(final_syndrome_mass) || final_syndrome_mass <= 0.0 ||
        invalid_observable_mass) {
      decoder->low_confidence_flag = true;
      return invalid_gradient();
    }
    if (syndrome_objective) {
      decoder->syndrome_log_probability += std::log(final_syndrome_mass);
    }
    decoder->predicted_obs_mask = decoder->total_mass_obs1 > decoder->total_mass_obs0 ? 1 : 0;

    using Adjoint = std::array<double, 2>;
    std::vector<Adjoint> adjoints(final_entries.size(), Adjoint{});
    for (size_t i = 0; i < final_entries.size(); ++i) {
      if (fixed_wide_state_zero(final_entries[i].state_words)) {
        if (syndrome_objective) {
          adjoints[i][0] = 1.0 / final_syndrome_mass;
          adjoints[i][1] = 1.0 / final_syndrome_mass;
        } else {
          adjoints[i][0] = -1.0 / decoder->total_mass_obs0;
          adjoints[i][1] = 1.0 / decoder->total_mass_obs1;
        }
      }
    }

    std::vector<double> gradients(layers.size(), 0.0);
    for (size_t layer_index = layers.size(); layer_index-- > 0;) {
      const auto& layer = layers[layer_index];
      const auto& children = states_by_layer[layer_index + 1];
      const auto& parents = states_by_layer[layer_index];
      const auto& layer_tape = tape[layer_index];
      double normalization_adjoint = 0.0;
      for (size_t i = 0; i < children.size(); ++i) {
        normalization_adjoint +=
            adjoints[i][0] * children[i].mass0 + adjoints[i][1] * children[i].mass1;
      }

      std::vector<Adjoint> pre_normalization_adjoints(children.size(), Adjoint{});
      for (size_t i = 0; i < children.size(); ++i) {
        pre_normalization_adjoints[i][0] =
            (adjoints[i][0] - normalization_adjoint + (syndrome_objective ? 1.0 : 0.0)) /
            layer_tape.normalizer;
        pre_normalization_adjoints[i][1] =
            (adjoints[i][1] - normalization_adjoint + (syndrome_objective ? 1.0 : 0.0)) /
            layer_tape.normalizer;
      }

      std::vector<Adjoint> parent_adjoints(parents.size(), Adjoint{});
      double probability_gradient = 0.0;
      for (const auto& transition : layer_tape.transitions) {
        const auto& parent = parents[transition.parent_index];
        const auto& child_adjoint = pre_normalization_adjoints[transition.child_index];
        auto& parent_adjoint = parent_adjoints[transition.parent_index];
        if (!transition.present) {
          probability_gradient -= child_adjoint[0] * parent.mass0 + child_adjoint[1] * parent.mass1;
          parent_adjoint[0] += child_adjoint[0] * layer.q;
          parent_adjoint[1] += child_adjoint[1] * layer.q;
        } else if (layer.toggles_observable) {
          probability_gradient += child_adjoint[0] * parent.mass1 + child_adjoint[1] * parent.mass0;
          parent_adjoint[0] += child_adjoint[1] * layer.p;
          parent_adjoint[1] += child_adjoint[0] * layer.p;
        } else {
          probability_gradient += child_adjoint[0] * parent.mass0 + child_adjoint[1] * parent.mass1;
          parent_adjoint[0] += child_adjoint[0] * layer.p;
          parent_adjoint[1] += child_adjoint[1] * layer.p;
        }
      }
      gradients[layer_index] = probability_gradient * layer.p * layer.q;
      adjoints = std::move(parent_adjoints);
    }
    return gradients;
  }

  std::vector<CompiledWideLayerTemplate<Words>> layers;
  std::vector<uint32_t> initial_detector_word_indices;
  std::vector<uint64_t> initial_detector_bit_masks;
  std::vector<double> initial_detector_costs;
  size_t max_frontier_width;
};

std::unique_ptr<TesseractTrellisWideKernelBase> build_compiled_wide_kernel(
    const std::vector<TesseractTrellisWideLayerTemplate>& layers, size_t max_frontier_width,
    const std::vector<double>& initial_future_detcost) {
  const size_t required_words = std::max<size_t>(1, num_state_words(max_frontier_width));
  if (required_words > kMaxCompiledWideStateWords) {
    throw std::invalid_argument("Wide trellis frontier requires " + std::to_string(required_words) +
                                " words, but only " + std::to_string(kMaxCompiledWideStateWords) +
                                " compiled words are enabled.");
  }

  std::vector<uint32_t> initial_detector_word_indices;
  std::vector<uint64_t> initial_detector_bit_masks;
  std::vector<double> initial_detector_costs;
  if (!layers.empty()) {
    const auto& initial_active_detectors = layers.front().current_active_detectors;
    initial_detector_word_indices.reserve(initial_active_detectors.size());
    initial_detector_bit_masks.reserve(initial_active_detectors.size());
    initial_detector_costs.reserve(initial_active_detectors.size());
    for (int detector : initial_active_detectors) {
      initial_detector_word_indices.push_back((uint32_t)detector_word_index((size_t)detector));
      initial_detector_bit_masks.push_back(detector_word_mask((size_t)detector));
      initial_detector_costs.push_back(initial_future_detcost[(size_t)detector]);
    }
  }
  switch (required_words) {
    case 1:
      return std::make_unique<CompiledWideKernel<1>>(
          compile_wide_layers<1>(layers), initial_detector_word_indices, initial_detector_bit_masks,
          initial_detector_costs, max_frontier_width);
    case 2:
      return std::make_unique<CompiledWideKernel<2>>(
          compile_wide_layers<2>(layers), initial_detector_word_indices, initial_detector_bit_masks,
          initial_detector_costs, max_frontier_width);
    case 3:
      return std::make_unique<CompiledWideKernel<3>>(
          compile_wide_layers<3>(layers), initial_detector_word_indices, initial_detector_bit_masks,
          initial_detector_costs, max_frontier_width);
    case 4:
      return std::make_unique<CompiledWideKernel<4>>(
          compile_wide_layers<4>(layers), initial_detector_word_indices, initial_detector_bit_masks,
          initial_detector_costs, max_frontier_width);
    default:
      throw std::invalid_argument("Unsupported compiled wide trellis word count.");
  }
}

}  // namespace

TesseractTrellisDecoder::~TesseractTrellisDecoder() = default;

TesseractTrellisDecoder::TesseractTrellisDecoder(TesseractTrellisConfig config_)
    : config(std::move(config_)) {
  // Maps original flattened DEM error indices to currently preprocessed indices.
  std::vector<size_t> dem_error_map(config.dem.flattened().count_errors());
  std::iota(dem_error_map.begin(), dem_error_map.end(), 0);

  if (config.merge_errors) {
    std::vector<size_t> merge_map;
    config.dem = common::merge_indistinguishable_errors(config.dem, merge_map);
    common::chain_error_maps(dem_error_map, merge_map);
  }

  std::vector<size_t> nonzero_map;
  config.dem = common::remove_zero_probability_errors(config.dem, nonzero_map);
  common::chain_error_maps(dem_error_map, nonzero_map);

  dem_error_to_error = std::move(dem_error_map);
  error_to_dem_error = common::invert_error_map(dem_error_to_error, config.dem.count_errors());

  errors = get_errors_from_dem(config.dem.flattened());
  num_detectors = config.dem.count_detectors();
  num_observables = config.dem.count_observables();
  if (num_observables > 1) {
    throw std::invalid_argument("tesseract_trellis currently supports at most one observable");
  }

  all_possible_detector_words.assign(num_state_words(num_detectors), 0);
  actual_detector_words_scratch.assign(all_possible_detector_words.size(), 0);
  for (const auto& error : errors) {
    for (int d : error.symptom.detectors) {
      all_possible_detector_words[detector_word_index((size_t)d)] |= detector_word_mask((size_t)d);
    }
  }

  auto faults = parse_faults(errors, num_observables);

  size_t wide_frontier_width = 0;
  build_wide_layer_templates(faults, num_detectors, &wide_layer_templates, &wide_frontier_width);
  if (config.ranking_mode == TesseractTrellisRankingMode::FutureActiveDetcostRanked) {
    build_future_active_detcost_transitions(faults, num_detectors, &wide_layer_templates,
                                            &initial_future_detcost);
  } else {
    build_future_detcost_transitions(faults, num_detectors, &wide_layer_templates,
                                     &initial_future_detcost);
  }
  if (config.ranking_mode != TesseractTrellisRankingMode::MassOnly) {
    scale_future_detcost_transitions(&wide_layer_templates, &initial_future_detcost,
                                     config.future_detcost_scale);
  }
  prepare_projected_fault_masks(&wide_layer_templates);
  wide_kernel =
      build_compiled_wide_kernel(wide_layer_templates, wide_frontier_width, initial_future_detcost);
  wide_layer_templates.clear();
}

TESSERACT_HOT void TesseractTrellisDecoder::decode_shot(const std::vector<uint64_t>& detections) {
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
  syndrome_log_probability = 0;
  FinalizeKeptStateStatsOnExit kept_state_stats_guard{this};
  wide_kernel->decode_shot(this, detections);

  if (config.verbose) {
    std::cout << "trellis beam_width=" << config.beam_width
              << " frontier_width=" << max_frontier_width_seen
              << " states_expanded=" << num_states_expanded
              << " states_merged=" << num_states_merged << " max_beam=" << max_beam_size_seen
              << std::endl;
  }
}

std::vector<double> TesseractTrellisDecoder::decode_shot_with_observable_logit_gradient(
    const std::vector<uint64_t>& detections) {
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
  syndrome_log_probability = 0;
  FinalizeKeptStateStatsOnExit kept_state_stats_guard{this};
  return wide_kernel->decode_shot_with_observable_logit_gradient(this, detections);
}

std::vector<double> TesseractTrellisDecoder::decode_shot_with_syndrome_log_probability_gradient(
    const std::vector<uint64_t>& detections) {
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
  syndrome_log_probability = 0;
  FinalizeKeptStateStatsOnExit kept_state_stats_guard{this};
  return wide_kernel->decode_shot_with_syndrome_log_probability_gradient(this, detections);
}

double TesseractTrellisDecoder::observable_probability() const {
  const double total_mass = total_mass_obs0 + total_mass_obs1;
  if (!std::isfinite(total_mass) || total_mass <= 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return total_mass_obs1 / total_mass;
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
