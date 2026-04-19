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
#include <bit>
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
#include <string>
#include <utility>

#include "utils.h"

struct TesseractTrellisWideKernelBase {
  virtual ~TesseractTrellisWideKernelBase() = default;
  virtual void decode_shot(TesseractTrellisDecoder* decoder,
                           const std::vector<uint64_t>& detections) const = 0;
};

namespace {

constexpr size_t kMaxCompiledWideStateWords = 4;

#if defined(__GNUC__) || defined(__clang__)
#define TESSERACT_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define TESSERACT_ALWAYS_INLINE inline
#endif

struct Fault {
  size_t error_index;
  double likelihood_cost;
  double log_q;
  double log_p;
  uint64_t obs_mask;
  std::vector<int> detectors;
};

struct WideStateGroup {
  double mass;
  double score;
  size_t begin;
  size_t end;
};

template <size_t Words>
using FixedWideStateWords = std::array<uint64_t, Words>;

template <size_t Words>
struct FixedWidePackedMass {
  FixedWideStateWords<Words> state_words{};
  uint64_t obs_mask = 0;
  double mass = 0.0;
  double penalty = 0.0;
};

template <size_t Words>
struct CompiledWideLayerTemplate {
  double q = 0.0;
  double p = 0.0;
  uint64_t obs_mask = 0;
  std::array<uint64_t, Words> surviving_masks{};
  std::array<uint8_t, Words> projection_dst_words{};
  std::array<uint8_t, Words> projection_dst_offsets{};
  std::array<uint64_t, Words> projected_fault_mask_words{};
  std::vector<uint32_t> fault_target_word_indices;
  std::vector<uint64_t> fault_target_bit_masks;
  std::vector<uint8_t> fault_word_indices;
  std::vector<uint64_t> fault_bit_masks;
  std::vector<uint8_t> fault_was_active_before;
  std::vector<int32_t> next_local_indices;
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
bool fixed_wide_state_less(const FixedWideStateWords<Words>& a, const FixedWideStateWords<Words>& b) {
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

template <size_t Words>
BranchPenaltyUpdate compute_compiled_wide_branch_update(
    const FixedWideStateWords<Words>& base_state_words, double current_penalty,
    const std::vector<uint64_t>& actual_detector_words,
    const CompiledWideLayerTemplate<Words>& layer,
    bool compute_penalties) {
  BranchPenaltyUpdate update;
  update.absent_penalty = compute_penalties ? current_penalty : 0.0;
  update.present_penalty = compute_penalties ? current_penalty : 0.0;

  for (size_t k = 0; k < layer.fault_target_word_indices.size(); ++k) {
    const bool state_bit =
        layer.fault_was_active_before[k] &&
        ((base_state_words[layer.fault_word_indices[k]] & layer.fault_bit_masks[k]) != 0);
    const bool target_bit =
        (actual_detector_words[layer.fault_target_word_indices[k]] &
         layer.fault_target_bit_masks[k]) != 0;
    const bool mismatch = state_bit ^ target_bit;
    const int32_t next_local = layer.next_local_indices[k];

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

    const double prev_contrib =
        (layer.fault_was_active_before[k] && mismatch) ? layer.current_costs[k] : 0.0;
    const double absent_contrib = (next_local >= 0 && mismatch) ? layer.next_costs[k] : 0.0;
    const double present_contrib = (next_local >= 0 && !mismatch) ? layer.next_costs[k] : 0.0;
    update.absent_penalty += absent_contrib - prev_contrib;
    update.present_penalty += present_contrib - prev_contrib;
  }

  return update;
}

template <size_t Words>
void normalize_compiled_items(std::vector<FixedWidePackedMass<Words>>* items) {
  double total = 0.0;
  for (const auto& item : *items) {
    total += item.mass;
  }
  if (total == 0.0) {
    return;
  }
  for (auto& item : *items) {
    item.mass /= total;
  }
}

template <size_t Words>
void merge_equal_compiled_keys_inplace(std::vector<FixedWidePackedMass<Words>>* items) {
  if (items->empty()) {
    return;
  }
  std::sort(items->begin(), items->end(),
            [](const FixedWidePackedMass<Words>& a, const FixedWidePackedMass<Words>& b) {
              if (fixed_wide_state_less(a.state_words, b.state_words)) {
                return true;
              }
              if (fixed_wide_state_less(b.state_words, a.state_words)) {
                return false;
              }
              return a.obs_mask < b.obs_mask;
            });

  size_t out = 0;
  for (size_t i = 1; i < items->size(); ++i) {
    if ((*items)[i].obs_mask == (*items)[out].obs_mask &&
        (*items)[i].state_words == (*items)[out].state_words) {
      (*items)[out].mass += (*items)[i].mass;
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
bool compiled_wide_state_group_score_greater(const std::vector<FixedWidePackedMass<Words>>& entries,
                                             const WideStateGroup& a, const WideStateGroup& b) {
  if (a.score != b.score) {
    return a.score > b.score;
  }
  return fixed_wide_state_less(entries[a.begin].state_words, entries[b.begin].state_words);
}

template <size_t Words>
size_t trim_compiled_wide_state_groups_by_beam_and_mass(
    const std::vector<FixedWidePackedMass<Words>>& entries, std::vector<WideStateGroup>* groups,
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
    std::nth_element(groups->begin(), groups->begin() + beam_width, groups->end(),
                     [&entries](const WideStateGroup& a, const WideStateGroup& b) {
                       return compiled_wide_state_group_score_greater(entries, a, b);
                     });
    groups->resize(beam_width);
  } else if (beam_eps <= 0.0) {
    return groups->size();
  }

  if (beam_eps <= 0.0 || total_mass <= 0.0) {
    return groups->size();
  }

  std::sort(groups->begin(), groups->end(),
            [&entries](const WideStateGroup& a, const WideStateGroup& b) {
              return compiled_wide_state_group_score_greater(entries, a, b);
            });
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
            [](const WideStateGroup& a, const WideStateGroup& b) { return a.begin < b.begin; });
  return groups->size();
}

template <size_t Words>
std::vector<WideStateGroup> collect_compiled_wide_state_groups(
    const std::vector<FixedWidePackedMass<Words>>& entries,
    TesseractTrellisRankingMode ranking_mode) {
  std::vector<WideStateGroup> groups;
  if (entries.empty()) {
    return groups;
  }
  groups.reserve(entries.size());
  size_t begin = 0;
  while (begin < entries.size()) {
    double mass = 0.0;
    size_t end = begin;
    while (end < entries.size() && entries[end].state_words == entries[begin].state_words) {
      mass += entries[end].mass;
      ++end;
    }
    groups.push_back(
        {mass, score_mass_and_penalty(mass, entries[begin].penalty, ranking_mode), begin, end});
    begin = end;
  }
  return groups;
}

template <size_t Words>
size_t keep_top_compiled_states(std::vector<FixedWidePackedMass<Words>>* entries,
                                size_t beam_width, double beam_eps,
                                TesseractTrellisRankingMode ranking_mode) {
  if (entries->empty()) {
    return 0;
  }
  auto groups = collect_compiled_wide_state_groups(*entries, ranking_mode);
  const size_t kept_group_count =
      trim_compiled_wide_state_groups_by_beam_and_mass(*entries, &groups, beam_width, beam_eps);

  std::vector<FixedWidePackedMass<Words>> kept;
  size_t kept_entries = 0;
  for (const auto& group : groups) {
    kept_entries += group.end - group.begin;
  }
  kept.reserve(kept_entries);
  for (const auto& group : groups) {
    for (size_t k = group.begin; k < group.end; ++k) {
      kept.push_back(std::move((*entries)[k]));
    }
  }
  *entries = std::move(kept);
  return kept_group_count;
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
    compiled.obs_mask = layer.obs_mask;

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
    compiled.fault_target_word_indices.reserve(transition.fault_local_indices.size());
    compiled.fault_target_bit_masks.reserve(transition.fault_local_indices.size());
    compiled.fault_word_indices.reserve(transition.fault_local_indices.size());
    compiled.fault_bit_masks.reserve(transition.fault_local_indices.size());
    compiled.fault_was_active_before.reserve(transition.fault_local_indices.size());
    compiled.next_local_indices = transition.next_local_indices;
    compiled.current_costs = transition.current_costs;
    compiled.next_costs = transition.next_costs;
    for (uint32_t local : transition.fault_local_indices) {
      const uint32_t detector = (uint32_t)layer.current_active_detectors[local];
      compiled.fault_target_word_indices.push_back((uint32_t)detector_word_index(detector));
      compiled.fault_target_bit_masks.push_back(detector_word_mask(detector));
      compiled.fault_word_indices.push_back(static_cast<uint8_t>(local >> 6));
      compiled.fault_bit_masks.push_back(uint64_t{1} << (local & 63));
      compiled.fault_was_active_before.push_back(local < layer.previous_width);
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
    if (decoder->config.ranking_mode == TesseractTrellisRankingMode::FutureDetcostRanked &&
        !layers.empty()) {
      initial_penalty = compute_initial_penalty_for_active_detectors(initial_detector_word_indices,
                                                                     initial_detector_bit_masks,
                                                                     initial_detector_costs,
                                                                     actual_detector_words);
    }

    std::vector<FixedWidePackedMass<Words>> beam_entries;
    std::vector<FixedWidePackedMass<Words>> next_entries;
    beam_entries.reserve(decoder->config.beam_width * 2 + 2);
    next_entries.reserve(decoder->config.beam_width * 4 + 4);
    beam_entries.push_back({{}, 0, 1.0, initial_penalty});
    decoder->max_beam_size_seen = 1;

    const bool compute_penalties =
        decoder->config.ranking_mode == TesseractTrellisRankingMode::FutureDetcostRanked;
    for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
      const auto& layer = layers[layer_index];

      auto t0 = std::chrono::high_resolution_clock::now();
      next_entries.clear();
      next_entries.reserve(beam_entries.size() * 2);

      if (decoder->config.verbose) {
        std::cout << "expanding layer " << layer_index << " / " << (layers.size() - 1)
                  << std::endl;
        std::cout << "states to expand = " << beam_entries.size() << std::endl;
      }
      for (const auto& item : beam_entries) {
        ++decoder->num_states_expanded;
        BranchPenaltyUpdate update = compute_compiled_wide_branch_update(
            item.state_words, item.penalty, actual_detector_words, layer, compute_penalties);

        if (!update.absent_valid && !update.present_valid) {
          continue;
        }

        FixedWideStateWords<Words> projected_state =
            project_compiled_wide_state(item.state_words, layer);
        const bool keep_absent = update.absent_valid && layer.q != 0.0;
        const bool keep_present = update.present_valid && layer.p != 0.0;
        if (keep_absent && keep_present) {
          FixedWideStateWords<Words> projected_toggled = projected_state;
          xor_compiled_wide_state(&projected_toggled, layer.projected_fault_mask_words);
          next_entries.push_back(
              {std::move(projected_state), item.obs_mask, item.mass * layer.q, update.absent_penalty});
          next_entries.push_back({std::move(projected_toggled), item.obs_mask ^ layer.obs_mask,
                                  item.mass * layer.p, update.present_penalty});
        } else if (keep_absent) {
          next_entries.push_back(
              {std::move(projected_state), item.obs_mask, item.mass * layer.q, update.absent_penalty});
        } else if (keep_present) {
          xor_compiled_wide_state(&projected_state, layer.projected_fault_mask_words);
          next_entries.push_back({std::move(projected_state), item.obs_mask ^ layer.obs_mask,
                                  item.mass * layer.p, update.present_penalty});
        }
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      decoder->time_expand_seconds +=
          std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;

      beam_entries.swap(next_entries);
      auto t2a = std::chrono::high_resolution_clock::now();
      merge_equal_compiled_keys_inplace(&beam_entries);
      auto t2 = std::chrono::high_resolution_clock::now();
      decoder->time_collapse_seconds +=
          std::chrono::duration_cast<std::chrono::microseconds>(t2 - t2a).count() / 1e6;

      const size_t kept_states = keep_top_compiled_states(
          &beam_entries, decoder->config.beam_width, decoder->config.beam_eps,
          decoder->config.ranking_mode);
      normalize_compiled_items(&beam_entries);
      record_kept_state_count(decoder, beam_entries.empty() ? 0 : kept_states);
      if (beam_entries.empty()) {
        decoder->low_confidence_flag = true;
        return;
      }
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
      if (item.obs_mask == 0) {
        decoder->total_mass_obs0 += item.mass;
      } else if (item.obs_mask == 1) {
        decoder->total_mass_obs1 += item.mass;
      }
    }
    if (decoder->total_mass_obs0 == 0.0 && decoder->total_mass_obs1 == 0.0) {
      decoder->low_confidence_flag = true;
      return;
    }
    decoder->predicted_obs_mask = decoder->total_mass_obs1 > decoder->total_mass_obs0 ? 1 : 0;
    auto tr1 = std::chrono::high_resolution_clock::now();
    decoder->time_reconstruct_seconds +=
        std::chrono::duration_cast<std::chrono::microseconds>(tr1 - tr0).count() / 1e6;
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
                                " words, but only " +
                                std::to_string(kMaxCompiledWideStateWords) +
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
  std::vector<size_t> dem_error_map(config.dem.flattened().count_errors());
  std::iota(dem_error_map.begin(), dem_error_map.end(), 0);
  dem_error_to_error = std::move(dem_error_map);
  error_to_dem_error = common::invert_error_map(dem_error_to_error, config.dem.count_errors());
  errors = get_errors_from_dem(config.dem.flattened());
  num_detectors = config.dem.count_detectors();
  num_observables = config.dem.count_observables();

  all_possible_detector_words.assign(num_state_words(num_detectors), 0);
  actual_detector_words_scratch.assign(all_possible_detector_words.size(), 0);
  for (const auto& error : errors) {
    for (int d : error.symptom.detectors) {
      all_possible_detector_words[detector_word_index((size_t)d)] |=
          detector_word_mask((size_t)d);
    }
  }

  auto faults = parse_faults(errors, num_observables);

  size_t wide_frontier_width = 0;
  build_wide_layer_templates(faults, num_detectors, &wide_layer_templates, &wide_frontier_width);
  std::vector<double> initial_future_detcost;
  build_future_detcost_transitions(faults, num_detectors, &wide_layer_templates,
                                   &initial_future_detcost);
  prepare_projected_fault_masks(&wide_layer_templates);
  wide_kernel =
      build_compiled_wide_kernel(wide_layer_templates, wide_frontier_width, initial_future_detcost);
}

__attribute__((hot)) void TesseractTrellisDecoder::decode_shot(
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
