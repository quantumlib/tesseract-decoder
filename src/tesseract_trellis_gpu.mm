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

#include "tesseract_trellis_gpu.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common.h"

namespace {

constexpr size_t kMaxCompiledWideStateWords = 4;
constexpr double kInf = std::numeric_limits<double>::infinity();

struct GpuTransitionTerm {
  uint64_t fault_target_bit_mask;
  uint64_t fault_bit_mask;
  float current_cost;
  float next_cost;
  uint32_t fault_target_word_index;
  uint32_t fault_word_index;
  uint32_t fault_was_active_before;
  uint32_t pad;
};

struct GpuLayerConfig {
  uint64_t surviving_masks[kMaxCompiledWideStateWords];
  uint64_t projected_fault_mask_words[kMaxCompiledWideStateWords];
  float q;
  float p;
  float log_q;
  float log_p;
  uint32_t toggles_observable;
  uint32_t compute_penalties;
  uint32_t surviving_term_count;
  uint32_t num_terms;
  uint32_t num_words;
  uint32_t has_retiring_terms;
  uint32_t term_offset;
  uint32_t projection_dst_words[kMaxCompiledWideStateWords];
  uint32_t projection_dst_offsets[kMaxCompiledWideStateWords];
};

struct MetalBeamEntry {
  uint64_t state_words[kMaxCompiledWideStateWords];
  float log_mass0;
  float log_mass1;
  float penalty;
  float log_total_mass;
};

struct MetalChildCandidate {
  uint64_t state_words[kMaxCompiledWideStateWords];
  float log_mass0;
  float log_mass1;
  float log_total_mass;
  float penalty;
  float score;
  uint32_t valid;
};

struct MetalFinalObsMass {
  float log_mass0;
  float log_mass1;
  uint32_t valid;
  uint32_t pad;
};

struct MetalBatchLaunchConfig {
  uint32_t num_shots;
  uint32_t beam_width;
  uint32_t detector_word_count;
  uint32_t ranking_mode;
};

struct MetalPersistentLaunchConfig {
  uint32_t num_shots;
  uint32_t beam_width;
  uint32_t detector_word_count;
  uint32_t layer_start;
  uint32_t num_layers;
  uint32_t ranking_mode;
  uint32_t dynamic_beam_enabled;
  uint32_t dynamic_initial_beam_width;
  float dynamic_confidence_threshold;
  uint32_t pad0;
};

struct GpuBeamEntry {
  std::array<uint64_t, kMaxCompiledWideStateWords> state_words{};
  double mass0 = 0.0;
  double mass1 = 0.0;
  double penalty = 0.0;
  double score = -kInf;
};

struct GpuStateBucket {
  std::array<uint64_t, kMaxCompiledWideStateWords> key{};
  double mass0 = 0.0;
  double mass1 = 0.0;
  double penalty = 0.0;
  bool occupied = false;
};

struct GpuCompiledLayer {
  GpuLayerConfig config{};
  std::vector<GpuTransitionTerm> terms;
  id<MTLBuffer> terms_buffer = nil;
};

struct PerShotScratch {
  std::vector<uint64_t> detector_words;
  std::vector<GpuBeamEntry> beam_entries;
  std::vector<GpuBeamEntry> next_entries;
  bool invalid = false;
  double initial_penalty = 0.0;
};

struct MetalContext {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> command_queue = nil;
  id<MTLComputePipelineState> pipeline = nil;
  id<MTLComputePipelineState> batch_pipeline = nil;
  id<MTLComputePipelineState> batch_select_pipeline = nil;
  id<MTLComputePipelineState> persistent_pipeline = nil;
  id<MTLComputePipelineState> persistent_histogram_pipeline = nil;
  id<MTLComputePipelineState> persistent_streaming_histogram_pipeline = nil;
  id<MTLComputePipelineState> final_obs_pipeline = nil;
  id<MTLBuffer> detector_words_buffer = nil;
  id<MTLBuffer> input_buffer = nil;
  id<MTLBuffer> output_buffer = nil;
  id<MTLBuffer> batch_detector_words_buffer = nil;
  id<MTLBuffer> batch_parent_counts_buffer = nil;
  id<MTLBuffer> batch_beam_limits_buffer = nil;
  id<MTLBuffer> batch_beam_growth_counts_buffer = nil;
  id<MTLBuffer> batch_input_buffer = nil;
  id<MTLBuffer> batch_next_buffer = nil;
  id<MTLBuffer> batch_output_buffer = nil;
  id<MTLBuffer> batch_final_obs_buffer = nil;
  id<MTLBuffer> persistent_layers_buffer = nil;
  id<MTLBuffer> persistent_terms_buffer = nil;
  size_t batch_capacity = 0;
  size_t detector_word_count = 0;
};

size_t num_state_words(size_t num_bits) {
  return (num_bits + 63) >> 6;
}

size_t detector_word_index(size_t detector) {
  return detector >> 6;
}

uint64_t detector_word_mask(size_t detector) {
  return uint64_t{1} << (detector & 63);
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
    if (best == kInf) {
      return kInf;
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
  if (penalty == kInf || mass == 0.0) {
    return -kInf;
  }
  return std::log(mass) - penalty;
}

double total_entry_mass(const GpuBeamEntry& entry) {
  return entry.mass0 + entry.mass1;
}

float metal_log_mass(double mass) {
  return mass > 0.0 ? static_cast<float>(std::log(mass))
                    : -std::numeric_limits<float>::infinity();
}

double exp_shifted(float log_value, float log_shift) {
  if (!std::isfinite(log_value) || !std::isfinite(log_shift)) {
    return 0.0;
  }
  return std::exp(static_cast<double>(log_value - log_shift));
}

bool fixed_wide_state_less(const std::array<uint64_t, kMaxCompiledWideStateWords>& a,
                           const std::array<uint64_t, kMaxCompiledWideStateWords>& b,
                           size_t num_words) {
  for (size_t k = num_words; k-- > 0;) {
    if (a[k] != b[k]) {
      return a[k] < b[k];
    }
  }
  return false;
}

bool fixed_wide_state_zero(const std::array<uint64_t, kMaxCompiledWideStateWords>& state_words,
                           size_t num_words) {
  for (size_t k = 0; k < num_words; ++k) {
    if (state_words[k] != 0) {
      return false;
    }
  }
  return true;
}

void reset_kept_state_stats(TesseractTrellisGpuDecoder* decoder, std::vector<uint32_t>* histogram) {
  decoder->kept_state_sample_count = 0;
  decoder->kept_state_min = 0;
  decoder->kept_state_median = 0;
  decoder->kept_state_mean = 0;
  decoder->kept_state_max = 0;
  if (!decoder->config.track_kept_state_stats) {
    return;
  }
  histogram->assign(decoder->config.beam_width + 1, 0);
}

void record_kept_state_count(TesseractTrellisGpuDecoder* decoder, std::vector<uint32_t>* histogram,
                             size_t kept_states) {
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
  ++(*histogram)[kept_states];
}

void finalize_kept_state_stats(TesseractTrellisGpuDecoder* decoder,
                               const std::vector<uint32_t>& histogram) {
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
  for (size_t kept_states = 0; kept_states < histogram.size(); ++kept_states) {
    seen += histogram[kept_states];
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

void reset_dynamic_beam_stats(TesseractTrellisGpuDecoder* decoder) {
  decoder->gpu_dynamic_beam_limit_sample_count = 0;
  decoder->gpu_dynamic_beam_limit_min = 0;
  decoder->gpu_dynamic_beam_limit_median = 0;
  decoder->gpu_dynamic_beam_limit_mean = 0;
  decoder->gpu_dynamic_beam_limit_max = 0;
  decoder->gpu_dynamic_beam_grow_events = 0;
}

void finalize_dynamic_beam_stats(TesseractTrellisGpuDecoder* decoder,
                                 std::vector<uint32_t> beam_limit_samples) {
  decoder->gpu_dynamic_beam_limit_sample_count = beam_limit_samples.size();
  if (beam_limit_samples.empty()) {
    return;
  }
  auto [min_it, max_it] =
      std::minmax_element(beam_limit_samples.begin(), beam_limit_samples.end());
  decoder->gpu_dynamic_beam_limit_min = *min_it;
  decoder->gpu_dynamic_beam_limit_max = *max_it;
  uint64_t sum = 0;
  for (uint32_t v : beam_limit_samples) {
    sum += v;
  }
  decoder->gpu_dynamic_beam_limit_mean =
      static_cast<double>(sum) / static_cast<double>(beam_limit_samples.size());
  const size_t mid = beam_limit_samples.size() / 2;
  std::nth_element(beam_limit_samples.begin(), beam_limit_samples.begin() + mid,
                   beam_limit_samples.end());
  if ((beam_limit_samples.size() & 1u) != 0) {
    decoder->gpu_dynamic_beam_limit_median = beam_limit_samples[mid];
  } else {
    const uint32_t hi = beam_limit_samples[mid];
    std::nth_element(beam_limit_samples.begin(), beam_limit_samples.begin() + mid - 1,
                     beam_limit_samples.end());
    decoder->gpu_dynamic_beam_limit_median = 0.5 * (beam_limit_samples[mid - 1] + hi);
  }
}

void normalize_compiled_items(std::vector<GpuBeamEntry>* items) {
  double total = 0.0;
  for (const auto& item : *items) {
    total += total_entry_mass(item);
  }
  if (total == 0.0) {
    return;
  }
  for (auto& item : *items) {
    item.mass0 /= total;
    item.mass1 /= total;
  }
}

uint64_t mix_splitmix64(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

uint64_t hash_state_words(const std::array<uint64_t, kMaxCompiledWideStateWords>& state_words,
                          size_t num_words) {
  uint64_t hash = 0x123456789abcdef0ULL;
  for (size_t k = 0; k < num_words; ++k) {
    hash ^= mix_splitmix64(state_words[k] + 0x9e3779b97f4a7c15ULL * (k + 1));
    hash = std::rotl(hash, 21);
  }
  return hash;
}

void ensure_state_bucket_capacity(std::vector<GpuStateBucket>* buckets, size_t num_items) {
  const size_t required = std::bit_ceil(std::max<size_t>(16, num_items * 2));
  if (buckets->size() < required) {
    buckets->resize(required);
  }
}

void clear_state_buckets(std::vector<GpuStateBucket>* buckets, std::vector<size_t>* used_indices) {
  for (size_t index : *used_indices) {
    (*buckets)[index].occupied = false;
  }
  used_indices->clear();
}

size_t find_or_insert_state_bucket(std::vector<GpuStateBucket>* buckets,
                                   std::vector<size_t>* used_indices,
                                   const std::array<uint64_t, kMaxCompiledWideStateWords>& key,
                                   size_t num_words) {
  const size_t mask = buckets->size() - 1;
  size_t index = hash_state_words(key, num_words) & mask;
  while ((*buckets)[index].occupied) {
    if ((*buckets)[index].key == key) {
      return index;
    }
    index = (index + 1) & mask;
  }

  auto& bucket = (*buckets)[index];
  bucket.occupied = true;
  bucket.key = key;
  bucket.mass0 = 0.0;
  bucket.mass1 = 0.0;
  bucket.penalty = kInf;
  used_indices->push_back(index);
  return index;
}

void collapse_child_candidates_into_entries(
    const MetalChildCandidate* child_ptr, size_t child_count, size_t num_words,
    std::vector<GpuStateBucket>* buckets, std::vector<size_t>* used_bucket_indices,
    std::vector<GpuBeamEntry>* out_entries) {
  ensure_state_bucket_capacity(buckets, child_count);
  clear_state_buckets(buckets, used_bucket_indices);
  float log_shift = -std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < child_count; ++i) {
    if (!child_ptr[i].valid) {
      continue;
    }
    log_shift = std::max(log_shift, child_ptr[i].log_mass0);
    log_shift = std::max(log_shift, child_ptr[i].log_mass1);
  }
  for (size_t i = 0; i < child_count; ++i) {
    if (!child_ptr[i].valid) {
      continue;
    }
    std::array<uint64_t, kMaxCompiledWideStateWords> key{};
    for (size_t k = 0; k < kMaxCompiledWideStateWords; ++k) {
      key[k] = child_ptr[i].state_words[k];
    }
    const size_t bucket_index =
        find_or_insert_state_bucket(buckets, used_bucket_indices, key, num_words);
    auto& bucket = (*buckets)[bucket_index];
    bucket.mass0 += exp_shifted(child_ptr[i].log_mass0, log_shift);
    bucket.mass1 += exp_shifted(child_ptr[i].log_mass1, log_shift);
    bucket.penalty = std::min(bucket.penalty, (double)child_ptr[i].penalty);
  }

  out_entries->clear();
  out_entries->reserve(used_bucket_indices->size());
  for (size_t bucket_index : *used_bucket_indices) {
    const auto& bucket = (*buckets)[bucket_index];
    out_entries->push_back({
        .state_words = bucket.key,
        .mass0 = bucket.mass0,
        .mass1 = bucket.mass1,
        .penalty = bucket.penalty,
    });
  }
}

void append_child_candidates_as_entries(const MetalChildCandidate* child_ptr, size_t child_count,
                                        std::vector<GpuBeamEntry>* out_entries) {
  out_entries->clear();
  out_entries->reserve(child_count);
  float log_shift = -std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < child_count; ++i) {
    if (!child_ptr[i].valid) {
      continue;
    }
    log_shift = std::max(log_shift, child_ptr[i].log_mass0);
    log_shift = std::max(log_shift, child_ptr[i].log_mass1);
  }
  for (size_t i = 0; i < child_count; ++i) {
    if (!child_ptr[i].valid) {
      continue;
    }
    GpuBeamEntry entry{};
    for (size_t k = 0; k < kMaxCompiledWideStateWords; ++k) {
      entry.state_words[k] = child_ptr[i].state_words[k];
    }
    entry.mass0 = exp_shifted(child_ptr[i].log_mass0, log_shift);
    entry.mass1 = exp_shifted(child_ptr[i].log_mass1, log_shift);
    entry.penalty = child_ptr[i].penalty;
    out_entries->push_back(entry);
  }
}

bool should_exact_merge_layer(size_t merge_period, size_t layer_index, size_t num_layers) {
  (void)num_layers;
  if (merge_period <= 1) {
    return true;
  }
  return ((layer_index + 1) % merge_period) == 0;
}

bool can_use_gpu_topk_without_merge(const TesseractTrellisConfig& config) {
  return config.beam_width <= 100000;
}

bool can_use_persistent_gpu_segment(const TesseractTrellisConfig& config) {
  return can_use_gpu_topk_without_merge(config);
}

bool gpu_dynamic_beam_enabled(const TesseractTrellisConfig& config) {
  return config.gpu_dynamic_initial_beam_width > 0 &&
         config.gpu_dynamic_initial_beam_width < config.beam_width;
}

uint32_t metal_ranking_mode(TesseractTrellisRankingMode ranking_mode) {
  if (ranking_mode == TesseractTrellisRankingMode::MassOnly) {
    return 0;
  }
  if (ranking_mode == TesseractTrellisRankingMode::FutureDetcostRanked ||
      ranking_mode == TesseractTrellisRankingMode::FutureActiveDetcostRanked) {
    return 1;
  }
  return 0;
}

NSUInteger preferred_sort_threadgroup_width(NSUInteger max_threads_per_group, size_t beam_width) {
  const size_t child_count = std::max<size_t>(16, std::bit_ceil(std::min<size_t>(beam_width * 2, 1024)));
  return std::min<NSUInteger>(max_threads_per_group, child_count);
}

bool compiled_state_score_greater(const GpuBeamEntry& a, const GpuBeamEntry& b, size_t num_words) {
  if (a.score != b.score) {
    return a.score > b.score;
  }
  return fixed_wide_state_less(a.state_words, b.state_words, num_words);
}

size_t keep_top_compiled_states(std::vector<GpuBeamEntry>* entries, size_t beam_width,
                                TesseractTrellisRankingMode ranking_mode, size_t num_words) {
  if (entries->empty()) {
    return 0;
  }

  for (auto& entry : *entries) {
    const double mass = total_entry_mass(entry);
    entry.score = score_mass_and_penalty(mass, entry.penalty, ranking_mode);
  }

  if (entries->size() > beam_width) {
    std::nth_element(entries->begin(), entries->begin() + beam_width, entries->end(),
                     [&](const GpuBeamEntry& a, const GpuBeamEntry& b) {
                       return compiled_state_score_greater(a, b, num_words);
                     });
    entries->resize(beam_width);
  }
  return entries->size();
}

std::vector<GpuCompiledLayer> compile_gpu_layers(
    const std::vector<TesseractTrellisWideLayerTemplate>& layers, size_t required_words) {
  std::vector<GpuCompiledLayer> compiled_layers;
  compiled_layers.reserve(layers.size());
  for (const auto& layer : layers) {
    GpuCompiledLayer compiled;
    compiled.config.q = static_cast<float>(layer.q);
    compiled.config.p = static_cast<float>(layer.p);
    compiled.config.log_q =
        layer.q > 0.0 ? static_cast<float>(std::log(layer.q)) : -std::numeric_limits<float>::infinity();
    compiled.config.log_p =
        layer.p > 0.0 ? static_cast<float>(std::log(layer.p)) : -std::numeric_limits<float>::infinity();
    compiled.config.num_words = (uint32_t)required_words;
    compiled.config.toggles_observable = layer.obs_mask != 0;

    if (layer.obs_mask > 1) {
      throw std::invalid_argument("tesseract_trellis_gpu currently supports at most 1 observable");
    }

    std::array<uint64_t, kMaxCompiledWideStateWords> surviving_masks{};
    for (uint32_t current_local : layer.surviving_local_indices) {
      surviving_masks[current_local >> 6] |= uint64_t{1} << (current_local & 63);
    }
    size_t next_offset = 0;
    for (size_t src_word = 0; src_word < required_words; ++src_word) {
      compiled.config.surviving_masks[src_word] = surviving_masks[src_word];
      compiled.config.projection_dst_words[src_word] = static_cast<uint32_t>(next_offset >> 6);
      compiled.config.projection_dst_offsets[src_word] = static_cast<uint32_t>(next_offset & 63);
      next_offset += std::popcount(surviving_masks[src_word]);
    }

    for (size_t k = 0; k < layer.projected_fault_mask_words.size(); ++k) {
      compiled.config.projected_fault_mask_words[k] = layer.projected_fault_mask_words[k];
    }

    const auto& transition = layer.detcost_transition;
    auto append_term = [&](size_t idx) {
      const uint32_t local = transition.fault_local_indices[idx];
      const uint32_t detector = (uint32_t)layer.current_active_detectors[local];
      compiled.terms.push_back(GpuTransitionTerm{
          .fault_target_bit_mask = detector_word_mask(detector),
          .fault_bit_mask = uint64_t{1} << (local & 63),
          .current_cost = static_cast<float>(transition.current_costs[idx]),
          .next_cost = static_cast<float>(transition.next_costs[idx]),
          .fault_target_word_index = (uint32_t)detector_word_index(detector),
          .fault_word_index = (uint32_t)(local >> 6),
          .fault_was_active_before = local < layer.previous_width ? 1u : 0u,
          .pad = 0u,
      });
    };
    for (size_t idx = 0; idx < transition.fault_local_indices.size(); ++idx) {
      if (transition.next_local_indices[idx] >= 0) {
        append_term(idx);
      }
    }
    compiled.config.surviving_term_count = (uint32_t)compiled.terms.size();
    for (size_t idx = 0; idx < transition.fault_local_indices.size(); ++idx) {
      if (transition.next_local_indices[idx] < 0) {
        append_term(idx);
      }
    }
    compiled.config.num_terms = (uint32_t)compiled.terms.size();
    compiled.config.has_retiring_terms =
        compiled.config.surviving_term_count != compiled.config.num_terms;
    compiled.config.term_offset = 0;
    compiled_layers.push_back(std::move(compiled));
  }
  return compiled_layers;
}

NSString* metal_shader_source() {
  return @R"METAL(
#include <metal_stdlib>
using namespace metal;

constant uint kMaxWords = 4;

struct GpuTransitionTerm {
  ulong fault_target_bit_mask;
  ulong fault_bit_mask;
  float current_cost;
  float next_cost;
  uint fault_target_word_index;
  uint fault_word_index;
  uint fault_was_active_before;
  uint pad;
};

struct GpuLayerConfig {
  ulong surviving_masks[kMaxWords];
  ulong projected_fault_mask_words[kMaxWords];
  float q;
  float p;
  float log_q;
  float log_p;
  uint toggles_observable;
  uint compute_penalties;
  uint surviving_term_count;
  uint num_terms;
  uint num_words;
  uint has_retiring_terms;
  uint term_offset;
  uint projection_dst_words[kMaxWords];
  uint projection_dst_offsets[kMaxWords];
};

struct MetalBeamEntry {
  ulong state_words[kMaxWords];
  float log_mass0;
  float log_mass1;
  float penalty;
  float log_total_mass;
};

struct MetalChildCandidate {
  ulong state_words[kMaxWords];
  float log_mass0;
  float log_mass1;
  float log_total_mass;
  float penalty;
  float score;
  uint valid;
};

struct MetalFinalObsMass {
  float log_mass0;
  float log_mass1;
  uint valid;
  uint pad;
};

struct MetalBatchLaunchConfig {
  uint num_shots;
  uint beam_width;
  uint detector_word_count;
  uint ranking_mode;
};

struct MetalPersistentLaunchConfig {
  uint num_shots;
  uint beam_width;
  uint detector_word_count;
  uint layer_start;
  uint num_layers;
  uint ranking_mode;
  uint dynamic_beam_enabled;
  uint dynamic_initial_beam_width;
  float dynamic_confidence_threshold;
  uint pad0;
};

inline ulong compact_bits_u64(ulong value, ulong mask) {
  ulong out = 0;
  ulong out_bit = 1;
  while (mask != 0) {
    ulong keep = mask & (~mask + 1);
    if ((value & keep) != 0) {
      out |= out_bit;
    }
    mask ^= keep;
    out_bit <<= 1;
  }
  return out;
}

inline void project_state(thread ulong out_words[kMaxWords],
                          const thread ulong in_words[kMaxWords],
                          constant GpuLayerConfig& layer) {
  for (uint i = 0; i < kMaxWords; i++) {
    out_words[i] = 0;
  }
  for (uint src_word = 0; src_word < layer.num_words; src_word++) {
    ulong mask = layer.surviving_masks[src_word];
    if (mask == 0) {
      continue;
    }
    ulong packed = compact_bits_u64(in_words[src_word], mask);
    uint dst_word = layer.projection_dst_words[src_word];
    uint shift = layer.projection_dst_offsets[src_word];
    out_words[dst_word] |= packed << shift;
    if (shift != 0 && dst_word + 1 < layer.num_words) {
      out_words[dst_word + 1] |= packed >> (64 - shift);
    }
  }
}

inline void project_state_local(thread ulong out_words[kMaxWords],
                                const thread ulong in_words[kMaxWords],
                                const thread GpuLayerConfig& layer) {
  for (uint i = 0; i < kMaxWords; i++) {
    out_words[i] = 0;
  }
  for (uint src_word = 0; src_word < layer.num_words; src_word++) {
    ulong mask = layer.surviving_masks[src_word];
    if (mask == 0) {
      continue;
    }
    ulong packed = compact_bits_u64(in_words[src_word], mask);
    uint dst_word = layer.projection_dst_words[src_word];
    uint shift = layer.projection_dst_offsets[src_word];
    out_words[dst_word] |= packed << shift;
    if (shift != 0 && dst_word + 1 < layer.num_words) {
      out_words[dst_word + 1] |= packed >> (64 - shift);
    }
  }
}

inline void copy_words(thread ulong dst[kMaxWords], const thread ulong src[kMaxWords]) {
  for (uint i = 0; i < kMaxWords; i++) {
    dst[i] = src[i];
  }
}

inline void xor_words(thread ulong dst[kMaxWords], constant ulong* mask_words, uint num_words) {
  for (uint i = 0; i < num_words; i++) {
    dst[i] ^= mask_words[i];
  }
}

inline void xor_words_local(thread ulong dst[kMaxWords], const thread ulong mask_words[kMaxWords],
                            uint num_words) {
  for (uint i = 0; i < num_words; i++) {
    dst[i] ^= mask_words[i];
  }
}

inline float logaddexp_metal(float a, float b) {
  if (!isfinite(a)) {
    return b;
  }
  if (!isfinite(b)) {
    return a;
  }
  const float m = max(a, b);
  return m + log(exp(a - m) + exp(b - m));
}

inline float score_total_mass_and_penalty_metal(float log_total_mass,
                                                float penalty,
                                                uint ranking_mode) {
  if (!isfinite(log_total_mass)) {
    return -INFINITY;
  }
  if (ranking_mode == 0u) {
    return log_total_mass;
  }
  if (!isfinite(penalty)) {
    return -INFINITY;
  }
  return log_total_mass - penalty;
}

inline float confidence_margin_from_log_masses(float log_mass0, float log_mass1) {
  if (!isfinite(log_mass0) && !isfinite(log_mass1)) {
    return 0.0f;
  }
  const float log_total = logaddexp_metal(log_mass0, log_mass1);
  if (!isfinite(log_total)) {
    return 0.0f;
  }
  const float mass0 = exp(log_mass0 - log_total);
  const float mass1 = exp(log_mass1 - log_total);
  return fabs(mass1 - mass0);
}

inline uint active_beam_limit_for_shot(device const uint* beam_limits,
                                       constant MetalPersistentLaunchConfig& launch,
                                       uint shot) {
  if (launch.dynamic_beam_enabled == 0u) {
    return launch.beam_width;
  }
  return min(launch.beam_width, max(1u, beam_limits[shot]));
}

inline uint maybe_grow_active_beam_limit_for_cutoff(uint valid_count,
                                                    uint active_beam_limit,
                                                    uint cutoff_count,
                                                    device uint* beam_limits,
                                                    device uint* beam_growth_counts,
                                                    constant MetalPersistentLaunchConfig& launch,
                                                    uint shot) {
  if (launch.dynamic_beam_enabled == 0u || active_beam_limit >= launch.beam_width ||
      valid_count <= active_beam_limit) {
    return active_beam_limit;
  }
  const float cutoff_fraction =
      min(1.0f, float(cutoff_count) / float(max(1u, active_beam_limit)));
  const float confidence = 1.0f - cutoff_fraction;
  if (!(confidence <= launch.dynamic_confidence_threshold)) {
    return active_beam_limit;
  }
  const uint new_limit = min(launch.beam_width, max(active_beam_limit + 1u, active_beam_limit * 2u));
  beam_limits[shot] = new_limit;
  beam_growth_counts[shot] += 1u;
  return new_limit;
}

inline bool state_words_less_desc(device const MetalChildCandidate& a,
                                  device const MetalChildCandidate& b) {
  for (uint k = kMaxWords; k-- > 0;) {
    if (a.state_words[k] != b.state_words[k]) {
      return a.state_words[k] > b.state_words[k];
    }
  }
  return false;
}

inline bool thread_beam_entry_state_is_zero(thread const MetalBeamEntry& entry) {
  for (uint k = 0; k < kMaxWords; ++k) {
    if (entry.state_words[k] != 0ul) {
      return false;
    }
  }
  return true;
}

inline bool child_less_for_sort(device const MetalChildCandidate* children,
                                threadgroup const float* scores, ushort a, ushort b,
                                uint shot_base) {
  const float sa = scores[a];
  const float sb = scores[b];
  if (sa != sb) {
    return sa < sb;
  }
  return state_words_less_desc(children[shot_base + a], children[shot_base + b]);
}

struct TransitionEffects {
  bool absent_valid;
  bool present_valid;
  float absent_penalty;
  float present_penalty;
};

inline TransitionEffects compute_transition_effects_local(
    thread const MetalBeamEntry& parent,
    device const ulong* actual_detector_words,
    device const GpuTransitionTerm* terms,
    thread const GpuLayerConfig& layer) {
  TransitionEffects effects;
  effects.absent_valid = true;
  effects.present_valid = true;
  effects.absent_penalty = layer.compute_penalties ? parent.penalty : 0.0f;
  effects.present_penalty = layer.compute_penalties ? parent.penalty : 0.0f;

  if (layer.compute_penalties != 0) {
    for (uint k = 0; k < layer.surviving_term_count; k++) {
      GpuTransitionTerm term = terms[k];
      bool state_bit = term.fault_was_active_before != 0 &&
                       ((parent.state_words[term.fault_word_index] & term.fault_bit_mask) != 0);
      bool target_bit =
          (actual_detector_words[term.fault_target_word_index] & term.fault_target_bit_mask) != 0;
      bool mismatch = state_bit ^ target_bit;
      float prev_contrib =
          (term.fault_was_active_before != 0 && mismatch) ? term.current_cost : 0.0f;
      float next_contrib = mismatch ? term.next_cost : 0.0f;
      effects.absent_penalty += next_contrib - prev_contrib;
      effects.present_penalty += (term.next_cost - next_contrib) - prev_contrib;
    }
    for (uint k = layer.surviving_term_count; k < layer.num_terms; k++) {
      GpuTransitionTerm term = terms[k];
      bool state_bit = term.fault_was_active_before != 0 &&
                       ((parent.state_words[term.fault_word_index] & term.fault_bit_mask) != 0);
      bool target_bit =
          (actual_detector_words[term.fault_target_word_index] & term.fault_target_bit_mask) != 0;
      bool mismatch = state_bit ^ target_bit;
      float prev_contrib =
          (term.fault_was_active_before != 0 && mismatch) ? term.current_cost : 0.0f;
      effects.absent_penalty -= prev_contrib;
      effects.present_penalty -= prev_contrib;
      if (mismatch) {
        effects.absent_valid = false;
      } else {
        effects.present_valid = false;
      }
    }
  } else if (layer.has_retiring_terms != 0) {
    for (uint k = layer.surviving_term_count; k < layer.num_terms; k++) {
      GpuTransitionTerm term = terms[k];
      bool state_bit = term.fault_was_active_before != 0 &&
                       ((parent.state_words[term.fault_word_index] & term.fault_bit_mask) != 0);
      bool target_bit =
          (actual_detector_words[term.fault_target_word_index] & term.fault_target_bit_mask) != 0;
      bool mismatch = state_bit ^ target_bit;
      if (mismatch) {
        effects.absent_valid = false;
      } else {
        effects.present_valid = false;
      }
    }
  }

  return effects;
}

kernel void expand_trellis_layer(
    device const MetalBeamEntry* beam_entries [[buffer(0)]],
    device MetalChildCandidate* children [[buffer(1)]],
    device const ulong* actual_detector_words [[buffer(2)]],
    device const GpuTransitionTerm* terms [[buffer(3)]],
    constant GpuLayerConfig& layer [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  MetalBeamEntry parent = beam_entries[tid];

  bool absent_valid = true;
  bool present_valid = true;
  float absent_penalty = layer.compute_penalties ? parent.penalty : 0.0f;
  float present_penalty = layer.compute_penalties ? parent.penalty : 0.0f;

  for (uint k = 0; k < layer.num_terms; k++) {
    GpuTransitionTerm term = terms[k];
    bool state_bit = term.fault_was_active_before != 0 &&
                     ((parent.state_words[term.fault_word_index] & term.fault_bit_mask) != 0);
    bool target_bit =
        (actual_detector_words[term.fault_target_word_index] & term.fault_target_bit_mask) != 0;
    bool mismatch = state_bit ^ target_bit;

    if (layer.compute_penalties != 0) {
      float prev_contrib =
          (term.fault_was_active_before != 0 && mismatch) ? term.current_cost : 0.0f;
      if (k < layer.surviving_term_count) {
        float next_contrib = mismatch ? term.next_cost : 0.0f;
        absent_penalty += next_contrib - prev_contrib;
        present_penalty += (term.next_cost - next_contrib) - prev_contrib;
      } else {
        absent_penalty -= prev_contrib;
        present_penalty -= prev_contrib;
      }
    }

    if (k >= layer.surviving_term_count) {
      if (mismatch) {
        absent_valid = false;
      } else {
        present_valid = false;
      }
    }
  }

  thread ulong projected_state[kMaxWords];
  project_state(projected_state, parent.state_words, layer);
  thread ulong toggled_state[kMaxWords];
  copy_words(toggled_state, projected_state);
  xor_words(toggled_state, layer.projected_fault_mask_words, layer.num_words);

  MetalChildCandidate absent_child;
  MetalChildCandidate present_child;
  for (uint i = 0; i < kMaxWords; i++) {
    absent_child.state_words[i] = projected_state[i];
    present_child.state_words[i] = toggled_state[i];
  }
  absent_child.log_mass0 = parent.log_mass0 + layer.log_q;
  absent_child.log_mass1 = parent.log_mass1 + layer.log_q;
  absent_child.log_total_mass = parent.log_total_mass + layer.log_q;
  absent_child.penalty = absent_penalty;
  absent_child.score = 0.0f;
  absent_child.valid = absent_valid && layer.q != 0.0f ? 1u : 0u;

  if (layer.toggles_observable != 0) {
    present_child.log_mass0 = parent.log_mass1 + layer.log_p;
    present_child.log_mass1 = parent.log_mass0 + layer.log_p;
  } else {
    present_child.log_mass0 = parent.log_mass0 + layer.log_p;
    present_child.log_mass1 = parent.log_mass1 + layer.log_p;
  }
  present_child.log_total_mass = parent.log_total_mass + layer.log_p;
  present_child.penalty = present_penalty;
  present_child.score = 0.0f;
  present_child.valid = present_valid && layer.p != 0.0f ? 1u : 0u;

  children[tid * 2] = absent_child;
  children[tid * 2 + 1] = present_child;
}

kernel void expand_trellis_layer_batched(
    device const MetalBeamEntry* beam_entries [[buffer(0)]],
    device MetalChildCandidate* children [[buffer(1)]],
    device const ulong* detector_words [[buffer(2)]],
    device const uint* parent_counts [[buffer(3)]],
    device const GpuTransitionTerm* terms [[buffer(4)]],
    constant GpuLayerConfig& layer [[buffer(5)]],
    constant MetalBatchLaunchConfig& launch [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
  const uint shot = tid / launch.beam_width;
  const uint local_parent = tid % launch.beam_width;
  if (shot >= launch.num_shots || local_parent >= parent_counts[shot]) {
    return;
  }

  const uint parent_index = shot * launch.beam_width + local_parent;
  const uint child_index = shot * launch.beam_width * 2 + local_parent * 2;
  MetalBeamEntry parent = beam_entries[parent_index];
  device const ulong* actual_detector_words = detector_words + shot * launch.detector_word_count;

  bool absent_valid = true;
  bool present_valid = true;
  float absent_penalty = layer.compute_penalties ? parent.penalty : 0.0f;
  float present_penalty = layer.compute_penalties ? parent.penalty : 0.0f;

  for (uint k = 0; k < layer.num_terms; k++) {
    GpuTransitionTerm term = terms[k];
    bool state_bit = term.fault_was_active_before != 0 &&
                     ((parent.state_words[term.fault_word_index] & term.fault_bit_mask) != 0);
    bool target_bit =
        (actual_detector_words[term.fault_target_word_index] & term.fault_target_bit_mask) != 0;
    bool mismatch = state_bit ^ target_bit;

    if (layer.compute_penalties != 0) {
      float prev_contrib =
          (term.fault_was_active_before != 0 && mismatch) ? term.current_cost : 0.0f;
      if (k < layer.surviving_term_count) {
        float next_contrib = mismatch ? term.next_cost : 0.0f;
        absent_penalty += next_contrib - prev_contrib;
        present_penalty += (term.next_cost - next_contrib) - prev_contrib;
      } else {
        absent_penalty -= prev_contrib;
        present_penalty -= prev_contrib;
      }
    }

    if (k >= layer.surviving_term_count) {
      if (mismatch) {
        absent_valid = false;
      } else {
        present_valid = false;
      }
    }
  }

  thread ulong projected_state[kMaxWords];
  project_state(projected_state, parent.state_words, layer);
  thread ulong toggled_state[kMaxWords];
  copy_words(toggled_state, projected_state);
  xor_words(toggled_state, layer.projected_fault_mask_words, layer.num_words);

  MetalChildCandidate absent_child;
  MetalChildCandidate present_child;
  for (uint i = 0; i < kMaxWords; i++) {
    absent_child.state_words[i] = projected_state[i];
    present_child.state_words[i] = toggled_state[i];
  }
  absent_child.log_mass0 = parent.log_mass0 + layer.log_q;
  absent_child.log_mass1 = parent.log_mass1 + layer.log_q;
  absent_child.log_total_mass = parent.log_total_mass + layer.log_q;
  absent_child.penalty = absent_penalty;
  absent_child.score = 0.0f;
  absent_child.valid = absent_valid && layer.q != 0.0f ? 1u : 0u;

  if (layer.toggles_observable != 0) {
    present_child.log_mass0 = parent.log_mass1 + layer.log_p;
    present_child.log_mass1 = parent.log_mass0 + layer.log_p;
  } else {
    present_child.log_mass0 = parent.log_mass0 + layer.log_p;
    present_child.log_mass1 = parent.log_mass1 + layer.log_p;
  }
  present_child.log_total_mass = parent.log_total_mass + layer.log_p;
  present_child.penalty = present_penalty;
  present_child.score = 0.0f;
  present_child.valid = present_valid && layer.p != 0.0f ? 1u : 0u;

  children[child_index] = absent_child;
  children[child_index + 1] = present_child;
}

kernel void select_top_children_batched(
    device const MetalChildCandidate* children [[buffer(0)]],
    device MetalBeamEntry* next_beam_entries [[buffer(1)]],
    device uint* parent_counts [[buffer(2)]],
    constant MetalBatchLaunchConfig& launch [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint shot [[threadgroup_position_in_grid]]) {
  if (shot >= launch.num_shots) {
    return;
  }
  constexpr uint kMaxBatchChildren = 2048;
  threadgroup ushort order[kMaxBatchChildren];
  threadgroup float scores[kMaxBatchChildren];

  const uint parent_count = parent_counts[shot];
  const uint child_count = min(parent_count * 2u, launch.beam_width * 2u);
  uint sort_count = 1;
  while (sort_count < max(child_count, 1u)) {
    sort_count <<= 1u;
  }
  const uint shot_base = shot * launch.beam_width * 2u;

  for (uint i = tid; i < sort_count; i += threads_per_group) {
    order[i] = static_cast<ushort>(i);
    if (i < child_count && children[shot_base + i].valid != 0u) {
      scores[i] = score_total_mass_and_penalty_metal(
          children[shot_base + i].log_total_mass, children[shot_base + i].penalty,
          launch.ranking_mode);
    } else {
      scores[i] = -INFINITY;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint k = 2; k <= sort_count; k <<= 1u) {
    for (uint j = k >> 1u; j > 0; j >>= 1u) {
      for (uint i = tid; i < sort_count; i += threads_per_group) {
        const uint ixj = i ^ j;
        if (ixj > i) {
          const bool ascending = (i & k) == 0u;
          const bool left_less = child_less_for_sort(children, scores, order[i], order[ixj], shot_base);
          if ((ascending && !left_less) || (!ascending && left_less)) {
            const ushort tmp = order[i];
            order[i] = order[ixj];
            order[ixj] = tmp;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

    if (tid == 0) {
      const uint keep_count = min(launch.beam_width, child_count);
      uint valid_kept = 0;
      for (uint out = 0; out < keep_count; ++out) {
        const uint src_index = order[sort_count - 1u - out];
        if (scores[src_index] == -INFINITY) {
          break;
        }
        ++valid_kept;
      }
      if (valid_kept == 0) {
        parent_counts[shot] = 0;
        return;
      }
      const uint beam_base = shot * launch.beam_width;
      for (uint out = 0; out < valid_kept; ++out) {
      const uint src_index = order[sort_count - 1u - out];
        const auto child = children[shot_base + src_index];
        for (uint k = 0; k < kMaxWords; ++k) {
          next_beam_entries[beam_base + out].state_words[k] = child.state_words[k];
        }
        next_beam_entries[beam_base + out].log_mass0 = child.log_mass0;
        next_beam_entries[beam_base + out].log_mass1 = child.log_mass1;
        next_beam_entries[beam_base + out].log_total_mass = child.log_total_mass;
        next_beam_entries[beam_base + out].penalty = child.penalty;
      }
      parent_counts[shot] = valid_kept;
    }
}

kernel void run_trellis_layers_persistent(
    device MetalBeamEntry* beam_a [[buffer(0)]],
    device MetalBeamEntry* beam_b [[buffer(1)]],
    device MetalChildCandidate* children [[buffer(2)]],
    device const ulong* detector_words [[buffer(3)]],
    device const GpuLayerConfig* layers [[buffer(4)]],
    device const GpuTransitionTerm* terms [[buffer(5)]],
    device uint* parent_counts [[buffer(6)]],
    constant MetalPersistentLaunchConfig& launch [[buffer(7)]],
    device uint* beam_limits [[buffer(8)]],
    device uint* beam_growth_counts [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint shot [[threadgroup_position_in_grid]]) {
  if (shot >= launch.num_shots) {
    return;
  }
  constexpr uint kMaxBatchChildren = 2048;
  threadgroup ushort order[kMaxBatchChildren];
  threadgroup float scores[kMaxBatchChildren];
  threadgroup uint active_beam_limit_shared;

  uint current_count = parent_counts[shot];
  if (current_count == 0) {
    return;
  }
  bool current_is_a = true;
  const uint shot_word_base = shot * launch.detector_word_count;
  device const ulong* actual_detector_words = detector_words + shot_word_base;
  const uint shot_child_base = shot * launch.beam_width * 2u;
  const uint shot_beam_base = shot * launch.beam_width;

  for (uint local_layer_index = 0; local_layer_index < launch.num_layers; ++local_layer_index) {
    const GpuLayerConfig layer = layers[launch.layer_start + local_layer_index];
    device MetalBeamEntry* current_beam = current_is_a ? beam_a : beam_b;
    device MetalBeamEntry* next_beam = current_is_a ? beam_b : beam_a;
    const uint child_count = min(current_count * 2u, launch.beam_width * 2u);

    for (uint local_parent = tid; local_parent < current_count; local_parent += threads_per_group) {
      const uint parent_index = shot_beam_base + local_parent;
      const uint child_index = shot_child_base + local_parent * 2u;
      MetalBeamEntry parent = current_beam[parent_index];

      const TransitionEffects effects = compute_transition_effects_local(
          parent, actual_detector_words, terms + layer.term_offset, layer);
      const bool absent_valid = effects.absent_valid;
      const bool present_valid = effects.present_valid;
      const float absent_penalty = effects.absent_penalty;
      const float present_penalty = effects.present_penalty;

      thread ulong projected_state[kMaxWords];
      project_state_local(projected_state, parent.state_words, layer);
      thread ulong toggled_state[kMaxWords];
      copy_words(toggled_state, projected_state);
      xor_words_local(toggled_state, layer.projected_fault_mask_words, layer.num_words);

      MetalChildCandidate absent_child;
      MetalChildCandidate present_child;
      for (uint i = 0; i < kMaxWords; i++) {
        absent_child.state_words[i] = projected_state[i];
        present_child.state_words[i] = toggled_state[i];
      }
      absent_child.log_mass0 = parent.log_mass0 + layer.log_q;
      absent_child.log_mass1 = parent.log_mass1 + layer.log_q;
      absent_child.log_total_mass = parent.log_total_mass + layer.log_q;
      absent_child.penalty = absent_penalty;
      absent_child.valid = absent_valid && layer.q != 0.0f ? 1u : 0u;
      if (absent_child.valid != 0u) {
        const float score = score_total_mass_and_penalty_metal(
            absent_child.log_total_mass, absent_child.penalty, launch.ranking_mode);
        absent_child.score = score;
      } else {
        absent_child.score = -INFINITY;
      }

      if (layer.toggles_observable != 0) {
        present_child.log_mass0 = parent.log_mass1 + layer.log_p;
        present_child.log_mass1 = parent.log_mass0 + layer.log_p;
      } else {
        present_child.log_mass0 = parent.log_mass0 + layer.log_p;
        present_child.log_mass1 = parent.log_mass1 + layer.log_p;
      }
      present_child.log_total_mass = parent.log_total_mass + layer.log_p;
      present_child.penalty = present_penalty;
      present_child.valid = present_valid && layer.p != 0.0f ? 1u : 0u;
      if (present_child.valid != 0u) {
        const float score = score_total_mass_and_penalty_metal(
            present_child.log_total_mass, present_child.penalty, launch.ranking_mode);
        present_child.score = score;
      } else {
        present_child.score = -INFINITY;
      }

      children[child_index] = absent_child;
      children[child_index + 1u] = present_child;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid == 0) {
      active_beam_limit_shared = active_beam_limit_for_shot(beam_limits, launch, shot);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint padded_sort_count = 1u;
    while (padded_sort_count < max(child_count, 1u)) {
      padded_sort_count <<= 1u;
    }
    padded_sort_count = min(padded_sort_count, uint(kMaxBatchChildren));

    for (uint i = tid; i < padded_sort_count; i += threads_per_group) {
      order[i] = static_cast<ushort>(i);
      if (i < child_count) {
        scores[i] = children[shot_child_base + i].score;
      } else {
        scores[i] = -INFINITY;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2; k <= padded_sort_count; k <<= 1u) {
      for (uint j = k >> 1u; j > 0; j >>= 1u) {
        for (uint i = tid; i < padded_sort_count; i += threads_per_group) {
          const uint ixj = i ^ j;
          if (ixj > i) {
            const bool ascending = (i & k) == 0u;
            const bool left_less =
                child_less_for_sort(children, scores, order[i], order[ixj], shot_child_base);
            if ((ascending && !left_less) || (!ascending && left_less)) {
              const ushort tmp = order[i];
              order[i] = order[ixj];
              order[ixj] = tmp;
            }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }

    if (tid == 0) {
      const uint keep_count = min(active_beam_limit_shared, child_count);
      uint valid_kept = 0;
      for (uint out = 0; out < keep_count; ++out) {
        const uint src_index = order[padded_sort_count - 1u - out];
        if (scores[src_index] == -INFINITY) {
          break;
        }
        ++valid_kept;
      }
      if (valid_kept == 0) {
        parent_counts[shot] = 0;
      } else {
        for (uint out = 0; out < valid_kept; ++out) {
          const uint src_index = order[padded_sort_count - 1u - out];
          const auto child = children[shot_child_base + src_index];
          for (uint k = 0; k < kMaxWords; ++k) {
            next_beam[shot_beam_base + out].state_words[k] = child.state_words[k];
          }
          next_beam[shot_beam_base + out].log_mass0 = child.log_mass0;
          next_beam[shot_beam_base + out].log_mass1 = child.log_mass1;
          next_beam[shot_beam_base + out].log_total_mass = child.log_total_mass;
          next_beam[shot_beam_base + out].penalty = child.penalty;
        }
        parent_counts[shot] = min(active_beam_limit_shared, valid_kept);
      }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    current_count = parent_counts[shot];
    if (current_count == 0) {
      break;
    }
    current_is_a = !current_is_a;
  }

  if (!current_is_a) {
    for (uint i = tid; i < current_count; i += threads_per_group) {
    beam_a[shot_beam_base + i] = beam_b[shot_beam_base + i];
    }
  }
}

kernel void run_trellis_layers_persistent_histogram(
    device MetalBeamEntry* beam_a [[buffer(0)]],
    device MetalBeamEntry* beam_b [[buffer(1)]],
    device MetalChildCandidate* children [[buffer(2)]],
    device const ulong* detector_words [[buffer(3)]],
    device const GpuLayerConfig* layers [[buffer(4)]],
    device const GpuTransitionTerm* terms [[buffer(5)]],
    device uint* parent_counts [[buffer(6)]],
    constant MetalPersistentLaunchConfig& launch [[buffer(7)]],
    device uint* beam_limits [[buffer(8)]],
    device uint* beam_growth_counts [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint shot [[threadgroup_position_in_grid]]) {
  if (shot >= launch.num_shots) {
    return;
  }
  constexpr uint kMaxBatchChildren = 2048;
  constexpr uint kScoreBins = 256;
  threadgroup float scores[kMaxBatchChildren];
  threadgroup atomic_uint hist[kScoreBins];
  threadgroup float local_min_scores[1024];
  threadgroup float local_max_scores[1024];
  threadgroup uint local_valid_counts[1024];
  threadgroup float min_score_shared;
  threadgroup float max_score_shared;
  threadgroup uint valid_count_shared;
  threadgroup uint active_beam_limit_shared;
  threadgroup uint cutoff_bin_shared;
  threadgroup uint better_count_shared;
  threadgroup atomic_uint kept_counter;
  threadgroup atomic_uint cutoff_counter;

  uint current_count = parent_counts[shot];
  if (current_count == 0) {
    return;
  }
  bool current_is_a = true;
  const uint shot_word_base = shot * launch.detector_word_count;
  device const ulong* actual_detector_words = detector_words + shot_word_base;
  const uint shot_child_base = shot * launch.beam_width * 2u;
  const uint shot_beam_base = shot * launch.beam_width;

  for (uint local_layer_index = 0; local_layer_index < launch.num_layers; ++local_layer_index) {
    const GpuLayerConfig layer = layers[launch.layer_start + local_layer_index];
    device MetalBeamEntry* current_beam = current_is_a ? beam_a : beam_b;
    device MetalBeamEntry* next_beam = current_is_a ? beam_b : beam_a;
    const uint child_count = min(current_count * 2u, launch.beam_width * 2u);
    float thread_min_score = INFINITY;
    float thread_max_score = -INFINITY;
    uint thread_valid_count = 0;

    for (uint local_parent = tid; local_parent < current_count; local_parent += threads_per_group) {
      const uint parent_index = shot_beam_base + local_parent;
      const uint child_index = shot_child_base + local_parent * 2u;
      MetalBeamEntry parent = current_beam[parent_index];

      const TransitionEffects effects = compute_transition_effects_local(
          parent, actual_detector_words, terms + layer.term_offset, layer);
      const bool absent_valid = effects.absent_valid;
      const bool present_valid = effects.present_valid;
      const float absent_penalty = effects.absent_penalty;
      const float present_penalty = effects.present_penalty;

      thread ulong projected_state[kMaxWords];
      project_state_local(projected_state, parent.state_words, layer);
      thread ulong toggled_state[kMaxWords];
      copy_words(toggled_state, projected_state);
      xor_words_local(toggled_state, layer.projected_fault_mask_words, layer.num_words);

      MetalChildCandidate absent_child;
      MetalChildCandidate present_child;
      for (uint i = 0; i < kMaxWords; i++) {
        absent_child.state_words[i] = projected_state[i];
        present_child.state_words[i] = toggled_state[i];
      }
      absent_child.log_mass0 = parent.log_mass0 + layer.log_q;
      absent_child.log_mass1 = parent.log_mass1 + layer.log_q;
      absent_child.log_total_mass = parent.log_total_mass + layer.log_q;
      absent_child.penalty = absent_penalty;
      absent_child.valid = absent_valid && layer.q != 0.0f ? 1u : 0u;
      if (absent_child.valid != 0u) {
        const float score = score_total_mass_and_penalty_metal(
            absent_child.log_total_mass, absent_child.penalty, launch.ranking_mode);
        absent_child.score = score;
        scores[local_parent * 2u] = score;
        if (score > -INFINITY) {
          thread_min_score = min(thread_min_score, score);
          thread_max_score = max(thread_max_score, score);
          thread_valid_count++;
        }
      } else {
        absent_child.score = -INFINITY;
        scores[local_parent * 2u] = -INFINITY;
      }

      if (layer.toggles_observable != 0) {
        present_child.log_mass0 = parent.log_mass1 + layer.log_p;
        present_child.log_mass1 = parent.log_mass0 + layer.log_p;
      } else {
        present_child.log_mass0 = parent.log_mass0 + layer.log_p;
        present_child.log_mass1 = parent.log_mass1 + layer.log_p;
      }
      present_child.log_total_mass = parent.log_total_mass + layer.log_p;
      present_child.penalty = present_penalty;
      present_child.valid = present_valid && layer.p != 0.0f ? 1u : 0u;
      if (present_child.valid != 0u) {
        const float score = score_total_mass_and_penalty_metal(
            present_child.log_total_mass, present_child.penalty, launch.ranking_mode);
        present_child.score = score;
        scores[local_parent * 2u + 1u] = score;
        if (score > -INFINITY) {
          thread_min_score = min(thread_min_score, score);
          thread_max_score = max(thread_max_score, score);
          thread_valid_count++;
        }
      } else {
        present_child.score = -INFINITY;
        scores[local_parent * 2u + 1u] = -INFINITY;
      }

      children[child_index] = absent_child;
      children[child_index + 1u] = present_child;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    local_min_scores[tid] = thread_min_score;
    local_max_scores[tid] = thread_max_score;
    local_valid_counts[tid] = thread_valid_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_group >> 1u; stride > 0; stride >>= 1u) {
      if (tid < stride) {
        local_min_scores[tid] = min(local_min_scores[tid], local_min_scores[tid + stride]);
        local_max_scores[tid] = max(local_max_scores[tid], local_max_scores[tid + stride]);
        local_valid_counts[tid] += local_valid_counts[tid + stride];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
      min_score_shared = local_min_scores[0];
      max_score_shared = local_max_scores[0];
      valid_count_shared = local_valid_counts[0];
      uint active_beam_limit = active_beam_limit_for_shot(beam_limits, launch, shot);
      if (!(max_score_shared > min_score_shared)) {
        active_beam_limit = maybe_grow_active_beam_limit_for_cutoff(
            valid_count_shared, active_beam_limit, valid_count_shared, beam_limits,
            beam_growth_counts, launch, shot);
      }
      active_beam_limit_shared = active_beam_limit;
      atomic_store_explicit(&kept_counter, 0u, memory_order_relaxed);
      atomic_store_explicit(&cutoff_counter, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid_count_shared == 0) {
      if (tid == 0) {
        parent_counts[shot] = 0;
      }
    } else if (valid_count_shared <= active_beam_limit_shared || !(max_score_shared > min_score_shared)) {
      for (uint i = tid; i < child_count; i += threads_per_group) {
        if (scores[i] == -INFINITY) {
          continue;
        }
        const uint out = atomic_fetch_add_explicit(&kept_counter, 1u, memory_order_relaxed);
        if (out < active_beam_limit_shared) {
          const auto child = children[shot_child_base + i];
          for (uint k = 0; k < kMaxWords; ++k) {
            next_beam[shot_beam_base + out].state_words[k] = child.state_words[k];
          }
          next_beam[shot_beam_base + out].log_mass0 = child.log_mass0;
          next_beam[shot_beam_base + out].log_mass1 = child.log_mass1;
          next_beam[shot_beam_base + out].log_total_mass = child.log_total_mass;
          next_beam[shot_beam_base + out].penalty = child.penalty;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid == 0) {
        parent_counts[shot] = min(active_beam_limit_shared,
                                  atomic_load_explicit(&kept_counter, memory_order_relaxed));
      }
    } else {
      if (tid < kScoreBins) {
        atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const float score_scale = float(kScoreBins - 1u) / (max_score_shared - min_score_shared);
      for (uint i = tid; i < child_count; i += threads_per_group) {
        const float score = scores[i];
        if (score == -INFINITY) {
          continue;
        }
        const uint bin = min(kScoreBins - 1u, uint((score - min_score_shared) * score_scale));
        atomic_fetch_add_explicit(&hist[bin], 1u, memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (tid == 0) {
        uint active_beam_limit = active_beam_limit_shared;
        uint better_count = 0;
        uint cutoff_count = 0;
        uint cutoff_bin = 0;
        for (uint attempt = 0; attempt < 2u; ++attempt) {
          better_count = 0;
          cutoff_count = 0;
          cutoff_bin = 0;
          for (uint rev = 0; rev < kScoreBins; ++rev) {
            const uint bin = (kScoreBins - 1u) - rev;
            const uint count = atomic_load_explicit(&hist[bin], memory_order_relaxed);
            if (better_count + count >= active_beam_limit) {
              cutoff_bin = bin;
              cutoff_count = count;
              break;
            }
            better_count += count;
          }
          const uint grown_limit = maybe_grow_active_beam_limit_for_cutoff(
              valid_count_shared, active_beam_limit, cutoff_count, beam_limits, beam_growth_counts,
              launch, shot);
          if (grown_limit == active_beam_limit) {
            break;
          }
          active_beam_limit = grown_limit;
        }
        active_beam_limit_shared = active_beam_limit;
        cutoff_bin_shared = cutoff_bin;
        better_count_shared = better_count;
        atomic_store_explicit(&kept_counter, 0u, memory_order_relaxed);
        atomic_store_explicit(&cutoff_counter, 0u, memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      for (uint i = tid; i < child_count; i += threads_per_group) {
        const float score = scores[i];
        if (score == -INFINITY) {
          continue;
        }
        const uint bin = min(kScoreBins - 1u, uint((score - min_score_shared) * score_scale));
        if (bin > cutoff_bin_shared) {
          const uint out = atomic_fetch_add_explicit(&kept_counter, 1u, memory_order_relaxed);
          if (out < active_beam_limit_shared) {
            const auto child = children[shot_child_base + i];
            for (uint k = 0; k < kMaxWords; ++k) {
              next_beam[shot_beam_base + out].state_words[k] = child.state_words[k];
            }
            next_beam[shot_beam_base + out].log_mass0 = child.log_mass0;
            next_beam[shot_beam_base + out].log_mass1 = child.log_mass1;
            next_beam[shot_beam_base + out].log_total_mass = child.log_total_mass;
            next_beam[shot_beam_base + out].penalty = child.penalty;
          }
        } else if (bin == cutoff_bin_shared) {
          const uint cutoff_out =
              atomic_fetch_add_explicit(&cutoff_counter, 1u, memory_order_relaxed);
          const uint out = better_count_shared + cutoff_out;
          if (out < active_beam_limit_shared) {
            const auto child = children[shot_child_base + i];
            for (uint k = 0; k < kMaxWords; ++k) {
              next_beam[shot_beam_base + out].state_words[k] = child.state_words[k];
            }
            next_beam[shot_beam_base + out].log_mass0 = child.log_mass0;
            next_beam[shot_beam_base + out].log_mass1 = child.log_mass1;
            next_beam[shot_beam_base + out].log_total_mass = child.log_total_mass;
            next_beam[shot_beam_base + out].penalty = child.penalty;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const uint kept_from_better = atomic_load_explicit(&kept_counter, memory_order_relaxed);
      const uint cutoff_count = atomic_load_explicit(&cutoff_counter, memory_order_relaxed);
      if (tid == 0) {
        parent_counts[shot] = min(active_beam_limit_shared, kept_from_better + cutoff_count);
      }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    current_count = parent_counts[shot];
    if (current_count == 0) {
      break;
    }
    current_is_a = !current_is_a;
  }

  if (!current_is_a) {
    for (uint i = tid; i < current_count; i += threads_per_group) {
      beam_a[shot_beam_base + i] = beam_b[shot_beam_base + i];
    }
  }
}

kernel void run_trellis_layers_persistent_streaming_histogram(
    device MetalBeamEntry* beam_a [[buffer(0)]],
    device MetalBeamEntry* beam_b [[buffer(1)]],
    device MetalChildCandidate* children [[buffer(2)]],
    device const ulong* detector_words [[buffer(3)]],
    device const GpuLayerConfig* layers [[buffer(4)]],
    device const GpuTransitionTerm* terms [[buffer(5)]],
    device uint* parent_counts [[buffer(6)]],
    constant MetalPersistentLaunchConfig& launch [[buffer(7)]],
    device uint* beam_limits [[buffer(8)]],
    device uint* beam_growth_counts [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint shot [[threadgroup_position_in_grid]]) {
  if (shot >= launch.num_shots) {
    return;
  }
  constexpr uint kScoreBins = 256;
  threadgroup atomic_uint hist[kScoreBins];
  threadgroup float local_min_scores[1024];
  threadgroup float local_max_scores[1024];
  threadgroup uint local_valid_counts[1024];
  threadgroup float min_score_shared;
  threadgroup float max_score_shared;
  threadgroup uint valid_count_shared;
  threadgroup uint active_beam_limit_shared;
  threadgroup uint cutoff_bin_shared;
  threadgroup uint better_count_shared;
  threadgroup atomic_uint kept_counter;
  threadgroup atomic_uint cutoff_counter;

  uint current_count = parent_counts[shot];
  if (current_count == 0) {
    return;
  }
  bool current_is_a = true;
  const uint shot_word_base = shot * launch.detector_word_count;
  device const ulong* actual_detector_words = detector_words + shot_word_base;
  const uint shot_child_base = shot * launch.beam_width * 2u;
  const uint shot_beam_base = shot * launch.beam_width;

  for (uint local_layer_index = 0; local_layer_index < launch.num_layers; ++local_layer_index) {
    const GpuLayerConfig layer = layers[launch.layer_start + local_layer_index];
    device MetalBeamEntry* current_beam = current_is_a ? beam_a : beam_b;
    device MetalBeamEntry* next_beam = current_is_a ? beam_b : beam_a;
    const uint child_count = min(current_count * 2u, launch.beam_width * 2u);
    float thread_min_score = INFINITY;
    float thread_max_score = -INFINITY;
    uint thread_valid_count = 0;

    for (uint local_parent = tid; local_parent < current_count; local_parent += threads_per_group) {
      const uint parent_index = shot_beam_base + local_parent;
      const uint child_index = shot_child_base + local_parent * 2u;
      MetalBeamEntry parent = current_beam[parent_index];

      const TransitionEffects effects = compute_transition_effects_local(
          parent, actual_detector_words, terms + layer.term_offset, layer);
      const bool absent_valid = effects.absent_valid;
      const bool present_valid = effects.present_valid;
      const float absent_penalty = effects.absent_penalty;
      const float present_penalty = effects.present_penalty;

      thread ulong projected_state[kMaxWords];
      project_state_local(projected_state, parent.state_words, layer);
      thread ulong toggled_state[kMaxWords];
      copy_words(toggled_state, projected_state);
      xor_words_local(toggled_state, layer.projected_fault_mask_words, layer.num_words);

      MetalChildCandidate absent_child;
      MetalChildCandidate present_child;
      for (uint i = 0; i < kMaxWords; i++) {
        absent_child.state_words[i] = projected_state[i];
        present_child.state_words[i] = toggled_state[i];
      }
      absent_child.log_mass0 = parent.log_mass0 + layer.log_q;
      absent_child.log_mass1 = parent.log_mass1 + layer.log_q;
      absent_child.log_total_mass = parent.log_total_mass + layer.log_q;
      absent_child.penalty = absent_penalty;
      absent_child.valid = absent_valid && layer.q != 0.0f ? 1u : 0u;
      if (absent_child.valid != 0u) {
        const float score = score_total_mass_and_penalty_metal(
            absent_child.log_total_mass, absent_child.penalty, launch.ranking_mode);
        absent_child.score = score;
        if (score > -INFINITY) {
          thread_min_score = min(thread_min_score, score);
          thread_max_score = max(thread_max_score, score);
          thread_valid_count++;
        }
      } else {
        absent_child.score = -INFINITY;
      }

      if (layer.toggles_observable != 0) {
        present_child.log_mass0 = parent.log_mass1 + layer.log_p;
        present_child.log_mass1 = parent.log_mass0 + layer.log_p;
      } else {
        present_child.log_mass0 = parent.log_mass0 + layer.log_p;
        present_child.log_mass1 = parent.log_mass1 + layer.log_p;
      }
      present_child.log_total_mass = parent.log_total_mass + layer.log_p;
      present_child.penalty = present_penalty;
      present_child.valid = present_valid && layer.p != 0.0f ? 1u : 0u;
      if (present_child.valid != 0u) {
        const float score = score_total_mass_and_penalty_metal(
            present_child.log_total_mass, present_child.penalty, launch.ranking_mode);
        present_child.score = score;
        if (score > -INFINITY) {
          thread_min_score = min(thread_min_score, score);
          thread_max_score = max(thread_max_score, score);
          thread_valid_count++;
        }
      } else {
        present_child.score = -INFINITY;
      }

      children[child_index] = absent_child;
      children[child_index + 1u] = present_child;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    local_min_scores[tid] = thread_min_score;
    local_max_scores[tid] = thread_max_score;
    local_valid_counts[tid] = thread_valid_count;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    for (uint stride = threads_per_group >> 1u; stride > 0; stride >>= 1u) {
      if (tid < stride) {
        local_min_scores[tid] = min(local_min_scores[tid], local_min_scores[tid + stride]);
        local_max_scores[tid] = max(local_max_scores[tid], local_max_scores[tid + stride]);
        local_valid_counts[tid] += local_valid_counts[tid + stride];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
      min_score_shared = local_min_scores[0];
      max_score_shared = local_max_scores[0];
      valid_count_shared = local_valid_counts[0];
      uint active_beam_limit = active_beam_limit_for_shot(beam_limits, launch, shot);
      if (!(max_score_shared > min_score_shared)) {
        active_beam_limit = maybe_grow_active_beam_limit_for_cutoff(
            valid_count_shared, active_beam_limit, valid_count_shared, beam_limits,
            beam_growth_counts, launch, shot);
      }
      active_beam_limit_shared = active_beam_limit;
      atomic_store_explicit(&kept_counter, 0u, memory_order_relaxed);
      atomic_store_explicit(&cutoff_counter, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid_count_shared == 0) {
      if (tid == 0) {
        parent_counts[shot] = 0;
      }
    } else if (valid_count_shared <= active_beam_limit_shared || !(max_score_shared > min_score_shared)) {
      for (uint i = tid; i < child_count; i += threads_per_group) {
        if (children[shot_child_base + i].score == -INFINITY) {
          continue;
        }
        const uint out = atomic_fetch_add_explicit(&kept_counter, 1u, memory_order_relaxed);
        if (out < active_beam_limit_shared) {
          const auto child = children[shot_child_base + i];
          for (uint k = 0; k < kMaxWords; ++k) {
            next_beam[shot_beam_base + out].state_words[k] = child.state_words[k];
          }
          next_beam[shot_beam_base + out].log_mass0 = child.log_mass0;
          next_beam[shot_beam_base + out].log_mass1 = child.log_mass1;
          next_beam[shot_beam_base + out].log_total_mass = child.log_total_mass;
          next_beam[shot_beam_base + out].penalty = child.penalty;
        }
      }
      threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
      if (tid == 0) {
        parent_counts[shot] = min(active_beam_limit_shared,
                                  atomic_load_explicit(&kept_counter, memory_order_relaxed));
      }
    } else {
      if (tid < kScoreBins) {
        atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const float score_scale = float(kScoreBins - 1u) / (max_score_shared - min_score_shared);
      for (uint i = tid; i < child_count; i += threads_per_group) {
        const float score = children[shot_child_base + i].score;
        if (score == -INFINITY) {
          continue;
        }
        const uint bin = min(kScoreBins - 1u, uint((score - min_score_shared) * score_scale));
        atomic_fetch_add_explicit(&hist[bin], 1u, memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (tid == 0) {
        uint active_beam_limit = active_beam_limit_shared;
        uint better_count = 0;
        uint cutoff_count = 0;
        uint cutoff_bin = 0;
        for (uint attempt = 0; attempt < 2u; ++attempt) {
          better_count = 0;
          cutoff_count = 0;
          cutoff_bin = 0;
          for (uint rev = 0; rev < kScoreBins; ++rev) {
            const uint bin = (kScoreBins - 1u) - rev;
            const uint count = atomic_load_explicit(&hist[bin], memory_order_relaxed);
            if (better_count + count >= active_beam_limit) {
              cutoff_bin = bin;
              cutoff_count = count;
              break;
            }
            better_count += count;
          }
          const uint grown_limit = maybe_grow_active_beam_limit_for_cutoff(
              valid_count_shared, active_beam_limit, cutoff_count, beam_limits, beam_growth_counts,
              launch, shot);
          if (grown_limit == active_beam_limit) {
            break;
          }
          active_beam_limit = grown_limit;
        }
        active_beam_limit_shared = active_beam_limit;
        cutoff_bin_shared = cutoff_bin;
        better_count_shared = better_count;
        atomic_store_explicit(&kept_counter, 0u, memory_order_relaxed);
        atomic_store_explicit(&cutoff_counter, 0u, memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      for (uint i = tid; i < child_count; i += threads_per_group) {
        const float score = children[shot_child_base + i].score;
        if (score == -INFINITY) {
          continue;
        }
        const uint bin = min(kScoreBins - 1u, uint((score - min_score_shared) * score_scale));
        if (bin > cutoff_bin_shared) {
          const uint out = atomic_fetch_add_explicit(&kept_counter, 1u, memory_order_relaxed);
          if (out < active_beam_limit_shared) {
            const auto child = children[shot_child_base + i];
            for (uint k = 0; k < kMaxWords; ++k) {
              next_beam[shot_beam_base + out].state_words[k] = child.state_words[k];
            }
            next_beam[shot_beam_base + out].log_mass0 = child.log_mass0;
            next_beam[shot_beam_base + out].log_mass1 = child.log_mass1;
            next_beam[shot_beam_base + out].log_total_mass = child.log_total_mass;
            next_beam[shot_beam_base + out].penalty = child.penalty;
          }
        } else if (bin == cutoff_bin_shared) {
          const uint cutoff_out =
              atomic_fetch_add_explicit(&cutoff_counter, 1u, memory_order_relaxed);
          const uint out = better_count_shared + cutoff_out;
          if (out < active_beam_limit_shared) {
            const auto child = children[shot_child_base + i];
            for (uint k = 0; k < kMaxWords; ++k) {
              next_beam[shot_beam_base + out].state_words[k] = child.state_words[k];
            }
            next_beam[shot_beam_base + out].log_mass0 = child.log_mass0;
            next_beam[shot_beam_base + out].log_mass1 = child.log_mass1;
            next_beam[shot_beam_base + out].log_total_mass = child.log_total_mass;
            next_beam[shot_beam_base + out].penalty = child.penalty;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

      const uint kept_from_better = atomic_load_explicit(&kept_counter, memory_order_relaxed);
      const uint cutoff_count = atomic_load_explicit(&cutoff_counter, memory_order_relaxed);
      if (tid == 0) {
        parent_counts[shot] = min(active_beam_limit_shared, kept_from_better + cutoff_count);
      }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    current_count = parent_counts[shot];
    if (current_count == 0) {
      break;
    }
    current_is_a = !current_is_a;
  }

  if (!current_is_a) {
    for (uint i = tid; i < current_count; i += threads_per_group) {
      beam_a[shot_beam_base + i] = beam_b[shot_beam_base + i];
    }
  }
}

kernel void reduce_final_observable_masses_batched(
    device const MetalBeamEntry* beam_entries [[buffer(0)]],
    device const uint* parent_counts [[buffer(1)]],
    device MetalFinalObsMass* final_obs [[buffer(2)]],
    constant MetalBatchLaunchConfig& launch [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint shot [[threadgroup_position_in_grid]]) {
  if (shot >= launch.num_shots) {
    return;
  }
  threadgroup float local_max0[1024];
  threadgroup float local_max1[1024];
  threadgroup float local_sum0[1024];
  threadgroup float local_sum1[1024];

  const uint beam_base = shot * launch.beam_width;
  const uint count = min(parent_counts[shot], launch.beam_width);
  float thread_max0 = -INFINITY;
  float thread_max1 = -INFINITY;
  for (uint i = tid; i < count; i += threads_per_group) {
    const MetalBeamEntry entry = beam_entries[beam_base + i];
    if (!thread_beam_entry_state_is_zero(entry)) {
      continue;
    }
    thread_max0 = max(thread_max0, entry.log_mass0);
    thread_max1 = max(thread_max1, entry.log_mass1);
  }
  local_max0[tid] = thread_max0;
  local_max1[tid] = thread_max1;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = threads_per_group >> 1u; stride > 0; stride >>= 1u) {
    if (tid < stride) {
      local_max0[tid] = max(local_max0[tid], local_max0[tid + stride]);
      local_max1[tid] = max(local_max1[tid], local_max1[tid + stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float max0 = local_max0[0];
  const float max1 = local_max1[0];
  float thread_sum0 = 0.0f;
  float thread_sum1 = 0.0f;
  for (uint i = tid; i < count; i += threads_per_group) {
    const MetalBeamEntry entry = beam_entries[beam_base + i];
    if (!thread_beam_entry_state_is_zero(entry)) {
      continue;
    }
    if (isfinite(max0) && isfinite(entry.log_mass0)) {
      thread_sum0 += exp(entry.log_mass0 - max0);
    }
    if (isfinite(max1) && isfinite(entry.log_mass1)) {
      thread_sum1 += exp(entry.log_mass1 - max1);
    }
  }
  local_sum0[tid] = thread_sum0;
  local_sum1[tid] = thread_sum1;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = threads_per_group >> 1u; stride > 0; stride >>= 1u) {
    if (tid < stride) {
      local_sum0[tid] += local_sum0[tid + stride];
      local_sum1[tid] += local_sum1[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0) {
    MetalFinalObsMass out;
    out.log_mass0 = (local_sum0[0] > 0.0f && isfinite(max0)) ? max0 + log(local_sum0[0]) : -INFINITY;
    out.log_mass1 = (local_sum1[0] > 0.0f && isfinite(max1)) ? max1 + log(local_sum1[0]) : -INFINITY;
    out.valid = (isfinite(out.log_mass0) || isfinite(out.log_mass1)) ? 1u : 0u;
    out.pad = 0u;
    final_obs[shot] = out;
  }
}

)METAL";
}

std::string ns_error_string(NSError* error) {
  if (error == nil) {
    return "";
  }
  return std::string([[error localizedDescription] UTF8String]);
}

std::unique_ptr<MetalContext> create_metal_context(size_t detector_word_count, size_t beam_width,
                                                   std::string* error_message) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      *error_message = "Metal device not available";
      return nullptr;
    }

    NSError* error = nil;
    id<MTLLibrary> library =
        [device newLibraryWithSource:metal_shader_source() options:nil error:&error];
    if (library == nil) {
      *error_message = "Metal shader compilation failed: " + ns_error_string(error);
      return nullptr;
    }

    id<MTLFunction> function = [library newFunctionWithName:@"expand_trellis_layer"];
    id<MTLFunction> batch_function = [library newFunctionWithName:@"expand_trellis_layer_batched"];
    id<MTLFunction> batch_select_function =
        [library newFunctionWithName:@"select_top_children_batched"];
    id<MTLFunction> persistent_function =
        [library newFunctionWithName:@"run_trellis_layers_persistent"];
    id<MTLFunction> persistent_histogram_function =
        [library newFunctionWithName:@"run_trellis_layers_persistent_histogram"];
    id<MTLFunction> persistent_streaming_histogram_function =
        [library newFunctionWithName:@"run_trellis_layers_persistent_streaming_histogram"];
    id<MTLFunction> final_obs_function =
        [library newFunctionWithName:@"reduce_final_observable_masses_batched"];
    if (function == nil || batch_function == nil || batch_select_function == nil ||
        persistent_function == nil || persistent_histogram_function == nil ||
        persistent_streaming_histogram_function == nil || final_obs_function == nil) {
      *error_message = "Metal shader function expand_trellis_layer not found";
      return nullptr;
    }

    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
      *error_message = "Metal pipeline creation failed: " + ns_error_string(error);
      return nullptr;
    }
    id<MTLComputePipelineState> batch_pipeline =
        [device newComputePipelineStateWithFunction:batch_function error:&error];
    if (batch_pipeline == nil) {
      *error_message = "Metal batch pipeline creation failed: " + ns_error_string(error);
      return nullptr;
    }
    id<MTLComputePipelineState> batch_select_pipeline =
        [device newComputePipelineStateWithFunction:batch_select_function error:&error];
    if (batch_select_pipeline == nil) {
      *error_message = "Metal batch select pipeline creation failed: " + ns_error_string(error);
      return nullptr;
    }
    id<MTLComputePipelineState> persistent_pipeline =
        [device newComputePipelineStateWithFunction:persistent_function error:&error];
    if (persistent_pipeline == nil) {
      *error_message = "Metal persistent pipeline creation failed: " + ns_error_string(error);
      return nullptr;
    }
    id<MTLComputePipelineState> persistent_histogram_pipeline =
        [device newComputePipelineStateWithFunction:persistent_histogram_function error:&error];
    if (persistent_histogram_pipeline == nil) {
      *error_message = "Metal persistent histogram pipeline creation failed: " + ns_error_string(error);
      return nullptr;
    }
    id<MTLComputePipelineState> persistent_streaming_histogram_pipeline =
        [device newComputePipelineStateWithFunction:persistent_streaming_histogram_function
                                              error:&error];
    if (persistent_streaming_histogram_pipeline == nil) {
      *error_message =
          "Metal persistent streaming histogram pipeline creation failed: " + ns_error_string(error);
      return nullptr;
    }
    id<MTLComputePipelineState> final_obs_pipeline =
        [device newComputePipelineStateWithFunction:final_obs_function error:&error];
    if (final_obs_pipeline == nil) {
      *error_message = "Metal final observable pipeline creation failed: " + ns_error_string(error);
      return nullptr;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
      *error_message = "Metal command queue creation failed";
      return nullptr;
    }

    auto ctx = std::make_unique<MetalContext>();
    ctx->device = device;
    ctx->command_queue = queue;
    ctx->pipeline = pipeline;
    ctx->batch_pipeline = batch_pipeline;
    ctx->batch_select_pipeline = batch_select_pipeline;
    ctx->persistent_pipeline = persistent_pipeline;
    ctx->persistent_histogram_pipeline = persistent_histogram_pipeline;
    ctx->persistent_streaming_histogram_pipeline = persistent_streaming_histogram_pipeline;
    ctx->final_obs_pipeline = final_obs_pipeline;
    ctx->detector_word_count = detector_word_count;
    ctx->detector_words_buffer =
        [device newBufferWithLength:sizeof(uint64_t) * detector_word_count
                            options:MTLResourceStorageModeShared];
    ctx->input_buffer =
        [device newBufferWithLength:sizeof(MetalBeamEntry) * std::max<size_t>(1, beam_width)
                            options:MTLResourceStorageModeShared];
    ctx->output_buffer =
        [device newBufferWithLength:sizeof(MetalChildCandidate) * std::max<size_t>(2, beam_width * 2)
                            options:MTLResourceStorageModeShared];
    if (ctx->detector_words_buffer == nil || ctx->input_buffer == nil || ctx->output_buffer == nil) {
      *error_message = "Metal buffer allocation failed";
      return nullptr;
    }

    return ctx;
  }
}

bool ensure_batch_capacity(MetalContext* ctx, size_t num_shots, size_t beam_width,
                           std::string* error_message) {
  if (num_shots <= ctx->batch_capacity && ctx->batch_input_buffer != nil &&
      ctx->batch_next_buffer != nil && ctx->batch_output_buffer != nil &&
      ctx->batch_detector_words_buffer != nil &&
      ctx->batch_parent_counts_buffer != nil && ctx->batch_beam_limits_buffer != nil &&
      ctx->batch_beam_growth_counts_buffer != nil && ctx->batch_final_obs_buffer != nil) {
    return true;
  }
  ctx->batch_capacity = std::max(ctx->batch_capacity, num_shots);
  ctx->batch_detector_words_buffer =
      [ctx->device newBufferWithLength:sizeof(uint64_t) * ctx->detector_word_count * ctx->batch_capacity
                               options:MTLResourceStorageModeShared];
  ctx->batch_parent_counts_buffer =
      [ctx->device newBufferWithLength:sizeof(uint32_t) * ctx->batch_capacity
                               options:MTLResourceStorageModeShared];
  ctx->batch_beam_limits_buffer =
      [ctx->device newBufferWithLength:sizeof(uint32_t) * ctx->batch_capacity
                               options:MTLResourceStorageModeShared];
  ctx->batch_beam_growth_counts_buffer =
      [ctx->device newBufferWithLength:sizeof(uint32_t) * ctx->batch_capacity
                               options:MTLResourceStorageModeShared];
  ctx->batch_input_buffer =
      [ctx->device newBufferWithLength:sizeof(MetalBeamEntry) * beam_width * ctx->batch_capacity
                               options:MTLResourceStorageModeShared];
  ctx->batch_next_buffer =
      [ctx->device newBufferWithLength:sizeof(MetalBeamEntry) * beam_width * ctx->batch_capacity
                               options:MTLResourceStorageModeShared];
  ctx->batch_output_buffer =
      [ctx->device
          newBufferWithLength:sizeof(MetalChildCandidate) * beam_width * 2 * ctx->batch_capacity
                     options:MTLResourceStorageModeShared];
  ctx->batch_final_obs_buffer =
      [ctx->device newBufferWithLength:sizeof(MetalFinalObsMass) * ctx->batch_capacity
                               options:MTLResourceStorageModeShared];
  if (ctx->batch_detector_words_buffer == nil || ctx->batch_parent_counts_buffer == nil ||
      ctx->batch_beam_limits_buffer == nil || ctx->batch_beam_growth_counts_buffer == nil ||
      ctx->batch_input_buffer == nil || ctx->batch_next_buffer == nil ||
      ctx->batch_output_buffer == nil || ctx->batch_final_obs_buffer == nil) {
    *error_message = "Metal batch buffer allocation failed";
    return false;
  }
  return true;
}

void ensure_shot_scratch_capacity(std::vector<PerShotScratch>* scratch, size_t num_shots,
                                  size_t detector_word_count, size_t beam_width) {
  if (scratch->size() < num_shots) {
    scratch->resize(num_shots);
  }
  for (size_t k = 0; k < num_shots; ++k) {
    auto& s = (*scratch)[k];
    if (s.detector_words.size() != detector_word_count) {
      s.detector_words.assign(detector_word_count, 0);
    } else {
      std::fill(s.detector_words.begin(), s.detector_words.end(), 0);
    }
    s.beam_entries.clear();
    s.next_entries.clear();
    s.beam_entries.reserve(beam_width * 2 + 2);
    s.next_entries.reserve(beam_width * 2 + 2);
    s.invalid = false;
    s.initial_penalty = 0.0;
  }
}

void copy_stats_from_cpu(TesseractTrellisGpuDecoder* dst, const TesseractTrellisDecoder& src) {
  dst->low_confidence_flag = src.low_confidence_flag;
  dst->num_states_expanded = src.num_states_expanded;
  dst->num_states_merged = src.num_states_merged;
  dst->max_beam_size_seen = src.max_beam_size_seen;
  dst->max_frontier_width_seen = src.max_frontier_width_seen;
  dst->kept_state_sample_count = src.kept_state_sample_count;
  dst->kept_state_min = src.kept_state_min;
  dst->kept_state_median = src.kept_state_median;
  dst->kept_state_mean = src.kept_state_mean;
  dst->kept_state_max = src.kept_state_max;
  dst->time_expand_seconds = src.time_expand_seconds;
  dst->time_collapse_seconds = src.time_collapse_seconds;
  dst->time_truncate_seconds = src.time_truncate_seconds;
  dst->time_reconstruct_seconds = src.time_reconstruct_seconds;
  dst->predicted_obs_mask = src.predicted_obs_mask;
  dst->total_mass_obs0 = src.total_mass_obs0;
  dst->total_mass_obs1 = src.total_mass_obs1;
  dst->merge_calls = 0;
  dst->merge_input_candidates = 0;
  dst->merge_output_candidates = 0;
  dst->merge_duplicate_layers = 0;
  dst->merge_skipped_layers = 0;
  dst->gpu_dynamic_beam_limit_sample_count = 0;
  dst->gpu_dynamic_beam_limit_min = 0;
  dst->gpu_dynamic_beam_limit_median = 0;
  dst->gpu_dynamic_beam_limit_mean = 0;
  dst->gpu_dynamic_beam_limit_max = 0;
  dst->gpu_dynamic_beam_grow_events = 0;
}

}  // namespace

struct TesseractTrellisGpuDecoder::Impl {
  explicit Impl(TesseractTrellisConfig config)
      : cpu_decoder(std::move(config)), num_words(0), max_frontier_width(0), gpu_enabled(false) {}

  TesseractTrellisDecoder cpu_decoder;
  size_t num_words;
  size_t max_frontier_width;
  bool gpu_enabled;
  std::string backend;
  std::vector<GpuCompiledLayer> layers;
  std::vector<uint32_t> kept_state_histogram;
  std::vector<GpuStateBucket> collapse_buckets;
  std::vector<size_t> used_bucket_indices;
  std::vector<uint32_t> initial_detector_word_indices;
  std::vector<uint64_t> initial_detector_bit_masks;
  std::vector<double> initial_detector_costs;
  std::vector<PerShotScratch> shot_scratch;
  std::vector<uint32_t> dynamic_beam_limit_samples;
  std::unique_ptr<MetalContext> metal;
};

TesseractTrellisGpuDecoder::TesseractTrellisGpuDecoder(TesseractTrellisConfig config_)
    : config(std::move(config_)), impl(std::make_unique<Impl>(config)) {
  for (const auto& layer : impl->cpu_decoder.wide_layer_templates) {
    impl->max_frontier_width =
        std::max(impl->max_frontier_width, layer.current_active_detectors.size());
  }
  impl->num_words = std::max<size_t>(1, num_state_words(impl->max_frontier_width));
  if (impl->num_words > kMaxCompiledWideStateWords) {
    std::ostringstream ss;
    ss << "cpu-fallback (frontier requires " << impl->num_words << " state words)";
    impl->backend = ss.str();
    return;
  }

  try {
    std::vector<TesseractTrellisWideLayerTemplate> layers = impl->cpu_decoder.wide_layer_templates;
    const std::vector<double>& initial_future_detcost = impl->cpu_decoder.initial_future_detcost;
    impl->layers = compile_gpu_layers(layers, impl->num_words);
    if (!layers.empty()) {
      const auto& initial_active_detectors = layers.front().current_active_detectors;
      impl->initial_detector_word_indices.reserve(initial_active_detectors.size());
      impl->initial_detector_bit_masks.reserve(initial_active_detectors.size());
      impl->initial_detector_costs.reserve(initial_active_detectors.size());
      for (int detector : initial_active_detectors) {
        impl->initial_detector_word_indices.push_back((uint32_t)detector_word_index((size_t)detector));
        impl->initial_detector_bit_masks.push_back(detector_word_mask((size_t)detector));
        impl->initial_detector_costs.push_back(initial_future_detcost[(size_t)detector]);
      }
    }

    std::string error;
    impl->metal = create_metal_context(impl->cpu_decoder.actual_detector_words_scratch.size(),
                                       config.beam_width, &error);
    if (!impl->metal) {
      impl->backend = "cpu-fallback (" + error + ")";
      return;
    }

    for (auto& layer : impl->layers) {
      layer.config.compute_penalties = metal_ranking_mode(config.ranking_mode);
      const size_t bytes = std::max<size_t>(1, layer.terms.size()) * sizeof(GpuTransitionTerm);
      layer.terms_buffer =
          [impl->metal->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
      if (layer.terms_buffer == nil) {
        impl->backend = "cpu-fallback (Metal term buffer allocation failed)";
        impl->metal.reset();
        return;
      }
      if (!layer.terms.empty()) {
        std::memcpy([layer.terms_buffer contents], layer.terms.data(),
                    layer.terms.size() * sizeof(GpuTransitionTerm));
      }
    }

    size_t total_terms = 0;
    for (auto& layer : impl->layers) {
      layer.config.term_offset = static_cast<uint32_t>(total_terms);
      total_terms += layer.terms.size();
    }
    const size_t layer_bytes = std::max<size_t>(1, impl->layers.size()) * sizeof(GpuLayerConfig);
    impl->metal->persistent_layers_buffer =
        [impl->metal->device newBufferWithLength:layer_bytes options:MTLResourceStorageModeShared];
    const size_t term_bytes = std::max<size_t>(1, total_terms) * sizeof(GpuTransitionTerm);
    impl->metal->persistent_terms_buffer =
        [impl->metal->device newBufferWithLength:term_bytes options:MTLResourceStorageModeShared];
    if (impl->metal->persistent_layers_buffer == nil || impl->metal->persistent_terms_buffer == nil) {
      impl->backend = "cpu-fallback (Metal persistent buffer allocation failed)";
      impl->metal.reset();
      return;
    }
    auto* persistent_layer_ptr =
        static_cast<GpuLayerConfig*>([impl->metal->persistent_layers_buffer contents]);
    auto* persistent_term_ptr =
        static_cast<GpuTransitionTerm*>([impl->metal->persistent_terms_buffer contents]);
    size_t term_offset = 0;
    for (size_t k = 0; k < impl->layers.size(); ++k) {
      persistent_layer_ptr[k] = impl->layers[k].config;
      if (!impl->layers[k].terms.empty()) {
        std::memcpy(persistent_term_ptr + term_offset, impl->layers[k].terms.data(),
                    impl->layers[k].terms.size() * sizeof(GpuTransitionTerm));
        term_offset += impl->layers[k].terms.size();
      }
    }

    impl->gpu_enabled = true;
    impl->backend = "metal-gpu";
  } catch (const std::exception& ex) {
    impl->backend = std::string("cpu-fallback (") + ex.what() + ")";
  }
}

TesseractTrellisGpuDecoder::~TesseractTrellisGpuDecoder() = default;

void TesseractTrellisGpuDecoder::decode_shot(const std::vector<uint64_t>& detections) {
  if (!impl->gpu_enabled) {
    impl->cpu_decoder.decode_shot(detections);
    copy_stats_from_cpu(this, impl->cpu_decoder);
    return;
  }

  low_confidence_flag = false;
  num_states_expanded = 0;
  num_states_merged = 0;
  max_beam_size_seen = 0;
  max_frontier_width_seen = impl->max_frontier_width;
  time_expand_seconds = 0;
  time_collapse_seconds = 0;
  time_truncate_seconds = 0;
  time_reconstruct_seconds = 0;
  predicted_obs_mask = 0;
  total_mass_obs0 = 0;
  total_mass_obs1 = 0;
  merge_calls = 0;
  merge_input_candidates = 0;
  merge_output_candidates = 0;
  merge_duplicate_layers = 0;
  merge_skipped_layers = 0;
  reset_dynamic_beam_stats(this);
  reset_kept_state_stats(this, &impl->kept_state_histogram);

  auto& actual_detector_words = impl->cpu_decoder.actual_detector_words_scratch;
  std::fill(actual_detector_words.begin(), actual_detector_words.end(), 0);
  for (uint64_t d : detections) {
    if (d >= impl->cpu_decoder.num_detectors) {
      low_confidence_flag = true;
      finalize_kept_state_stats(this, impl->kept_state_histogram);
      return;
    }
    const size_t word = detector_word_index((size_t)d);
    const uint64_t mask = detector_word_mask((size_t)d);
    if ((impl->cpu_decoder.all_possible_detector_words[word] & mask) == 0) {
      low_confidence_flag = true;
      finalize_kept_state_stats(this, impl->kept_state_histogram);
      return;
    }
    actual_detector_words[word] ^= mask;
  }

  std::memcpy([impl->metal->detector_words_buffer contents], actual_detector_words.data(),
              actual_detector_words.size() * sizeof(uint64_t));

  double initial_penalty = 0.0;
  if (config.ranking_mode != TesseractTrellisRankingMode::MassOnly && !impl->layers.empty()) {
    initial_penalty = compute_initial_penalty_for_active_detectors(
        impl->initial_detector_word_indices, impl->initial_detector_bit_masks,
        impl->initial_detector_costs, actual_detector_words);
  }

  std::vector<GpuBeamEntry> beam_entries;
  std::vector<GpuBeamEntry> next_entries;
  beam_entries.reserve(config.beam_width * 2 + 2);
  next_entries.reserve(config.beam_width * 2 + 2);
  beam_entries.push_back({{}, 1.0, 0.0, initial_penalty});
  max_beam_size_seen = 1;

  for (size_t layer_index = 0; layer_index < impl->layers.size(); ++layer_index) {
    const auto& layer = impl->layers[layer_index];
    const size_t parent_count = beam_entries.size();
    num_states_expanded += parent_count;

    auto t0 = std::chrono::high_resolution_clock::now();
    auto* beam_ptr = static_cast<MetalBeamEntry*>([impl->metal->input_buffer contents]);
    for (size_t i = 0; i < parent_count; ++i) {
      for (size_t k = 0; k < kMaxCompiledWideStateWords; ++k) {
        beam_ptr[i].state_words[k] = beam_entries[i].state_words[k];
      }
      beam_ptr[i].log_mass0 = metal_log_mass(beam_entries[i].mass0);
      beam_ptr[i].log_mass1 = metal_log_mass(beam_entries[i].mass1);
      beam_ptr[i].penalty = static_cast<float>(beam_entries[i].penalty);
      beam_ptr[i].log_total_mass = metal_log_mass(total_entry_mass(beam_entries[i]));
    }

    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = [impl->metal->command_queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      [encoder setComputePipelineState:impl->metal->pipeline];
      [encoder setBuffer:impl->metal->input_buffer offset:0 atIndex:0];
      [encoder setBuffer:impl->metal->output_buffer offset:0 atIndex:1];
      [encoder setBuffer:impl->metal->detector_words_buffer offset:0 atIndex:2];
      [encoder setBuffer:layer.terms_buffer offset:0 atIndex:3];
      [encoder setBytes:&layer.config length:sizeof(GpuLayerConfig) atIndex:4];

      const NSUInteger grid_size = parent_count;
      const NSUInteger width =
          std::min<NSUInteger>(impl->metal->pipeline.maxTotalThreadsPerThreadgroup, 64);
      [encoder dispatchThreads:MTLSizeMake(grid_size, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
      [encoder endEncoding];
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
      if (command_buffer.error != nil) {
        impl->gpu_enabled = false;
        impl->backend = "cpu-fallback (Metal execution failed: " + ns_error_string(command_buffer.error) + ")";
        impl->cpu_decoder.decode_shot(detections);
        copy_stats_from_cpu(this, impl->cpu_decoder);
        return;
      }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    time_expand_seconds +=
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;

    auto t2a = std::chrono::high_resolution_clock::now();
    next_entries.clear();
    const auto* child_ptr =
        static_cast<const MetalChildCandidate*>([impl->metal->output_buffer contents]);
    if (should_exact_merge_layer(config.gpu_merge_period, layer_index, impl->layers.size())) {
      merge_calls += 1;
      merge_input_candidates += parent_count * 2;
      collapse_child_candidates_into_entries(child_ptr, parent_count * 2, impl->num_words,
                                             &impl->collapse_buckets, &impl->used_bucket_indices,
                                             &next_entries);
      merge_output_candidates += next_entries.size();
      if (next_entries.size() < parent_count * 2) {
        merge_duplicate_layers += 1;
      }
    } else {
      merge_skipped_layers += 1;
      append_child_candidates_as_entries(child_ptr, parent_count * 2, &next_entries);
    }
    beam_entries.swap(next_entries);
    auto t2 = std::chrono::high_resolution_clock::now();
    time_collapse_seconds +=
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t2a).count() / 1e6;

    const size_t kept_states =
        keep_top_compiled_states(&beam_entries, config.beam_width, config.ranking_mode,
                                 impl->num_words);
    normalize_compiled_items(&beam_entries);
    record_kept_state_count(this, &impl->kept_state_histogram, beam_entries.empty() ? 0 : kept_states);
    if (beam_entries.empty()) {
      low_confidence_flag = true;
      finalize_kept_state_stats(this, impl->kept_state_histogram);
      return;
    }
    num_states_merged += kept_states;
    max_beam_size_seen = std::max(max_beam_size_seen, kept_states);
    auto t3 = std::chrono::high_resolution_clock::now();
    time_truncate_seconds +=
        std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / 1e6;

    if (config.verbose) {
      std::cout << "gpu layer " << layer_index << " / " << (impl->layers.size() - 1)
                << " states=" << beam_entries.size() << std::endl;
    }
  }

  auto tr0 = std::chrono::high_resolution_clock::now();
  for (const auto& item : beam_entries) {
    if (!fixed_wide_state_zero(item.state_words, impl->num_words)) {
      continue;
    }
    total_mass_obs0 += item.mass0;
    total_mass_obs1 += item.mass1;
  }
  if (total_mass_obs0 == 0.0 && total_mass_obs1 == 0.0) {
    low_confidence_flag = true;
    finalize_kept_state_stats(this, impl->kept_state_histogram);
    return;
  }
  predicted_obs_mask = total_mass_obs1 > total_mass_obs0 ? 1 : 0;
  auto tr1 = std::chrono::high_resolution_clock::now();
  time_reconstruct_seconds +=
      std::chrono::duration_cast<std::chrono::microseconds>(tr1 - tr0).count() / 1e6;
  finalize_kept_state_stats(this, impl->kept_state_histogram);
}

std::vector<int> TesseractTrellisGpuDecoder::decode(const std::vector<uint64_t>& detections) {
  decode_shot(detections);
  return predicted_obs_mask ? std::vector<int>{0} : std::vector<int>{};
}

void TesseractTrellisGpuDecoder::decode_shots(std::vector<stim::SparseShot>& shots,
                                              std::vector<std::vector<int>>& obs_predicted) {
  obs_predicted.resize(shots.size());
  for (size_t i = 0; i < shots.size(); ++i) {
    obs_predicted[i] = decode(shots[i].hits);
  }
}

void TesseractTrellisGpuDecoder::decode_shots_batched(
    std::vector<stim::SparseShot>& shots, std::vector<uint64_t>& obs_predicted_masks,
    size_t shot_batch_size, std::vector<double>* obs0_masses, std::vector<double>* obs1_masses) {
  if (!using_gpu()) {
    merge_calls = 0;
    merge_input_candidates = 0;
    merge_output_candidates = 0;
    merge_duplicate_layers = 0;
    merge_skipped_layers = 0;
    reset_dynamic_beam_stats(this);
    obs_predicted_masks.resize(shots.size());
    if (obs0_masses != nullptr) {
      obs0_masses->resize(shots.size());
    }
    if (obs1_masses != nullptr) {
      obs1_masses->resize(shots.size());
    }
    for (size_t i = 0; i < shots.size(); ++i) {
      decode_shot(shots[i].hits);
      obs_predicted_masks[i] = predicted_obs_mask;
      if (obs0_masses != nullptr) {
        (*obs0_masses)[i] = total_mass_obs0;
      }
      if (obs1_masses != nullptr) {
        (*obs1_masses)[i] = total_mass_obs1;
      }
    }
    return;
  }
  shot_batch_size = std::max<size_t>(1, shot_batch_size);
  const size_t estimated_gpu_bytes_per_shot =
      config.beam_width * (2 * sizeof(MetalBeamEntry) + 2 * sizeof(MetalChildCandidate)) +
      impl->metal->detector_word_count * sizeof(uint64_t) + sizeof(uint32_t);
  constexpr size_t kTargetBatchBytes = 1536ull << 20;
  if (estimated_gpu_bytes_per_shot > 0) {
    const size_t memory_limited_batch_size =
        std::max<size_t>(1, kTargetBatchBytes / estimated_gpu_bytes_per_shot);
    shot_batch_size = std::min(shot_batch_size, memory_limited_batch_size);
  }
  obs_predicted_masks.resize(shots.size());
  if (obs0_masses != nullptr) {
    obs0_masses->assign(shots.size(), 0.0);
  }
  if (obs1_masses != nullptr) {
    obs1_masses->assign(shots.size(), 0.0);
  }
  merge_calls = 0;
  merge_input_candidates = 0;
  merge_output_candidates = 0;
  merge_duplicate_layers = 0;
  merge_skipped_layers = 0;
  reset_dynamic_beam_stats(this);
  impl->dynamic_beam_limit_samples.clear();
  for (size_t start = 0; start < shots.size(); start += shot_batch_size) {
    const size_t batch_size = std::min(shot_batch_size, shots.size() - start);
    std::string error;
    if (!ensure_batch_capacity(impl->metal.get(), batch_size, config.beam_width, &error)) {
      impl->gpu_enabled = false;
      impl->backend = "cpu-fallback (" + error + ")";
      for (size_t i = start; i < start + batch_size; ++i) {
        decode_shot(shots[i].hits);
        obs_predicted_masks[i] = predicted_obs_mask;
      }
      continue;
    }
    ensure_shot_scratch_capacity(&impl->shot_scratch, batch_size, impl->metal->detector_word_count,
                                 config.beam_width);

    for (size_t local_shot = 0; local_shot < batch_size; ++local_shot) {
      const auto& shot = shots[start + local_shot];
      auto& scratch = impl->shot_scratch[local_shot];
      auto& detector_words = scratch.detector_words;
      for (uint64_t d : shot.hits) {
        if (d >= impl->cpu_decoder.num_detectors) {
          scratch.invalid = true;
          break;
        }
        const size_t word = detector_word_index((size_t)d);
        const uint64_t mask = detector_word_mask((size_t)d);
        if ((impl->cpu_decoder.all_possible_detector_words[word] & mask) == 0) {
          scratch.invalid = true;
          break;
        }
        detector_words[word] ^= mask;
      }
      if (!scratch.invalid && config.ranking_mode != TesseractTrellisRankingMode::MassOnly &&
          !impl->layers.empty()) {
        scratch.initial_penalty = compute_initial_penalty_for_active_detectors(
            impl->initial_detector_word_indices, impl->initial_detector_bit_masks,
            impl->initial_detector_costs, detector_words);
      }
      if (!scratch.invalid) {
        scratch.beam_entries.push_back({{}, 1.0, 0.0, scratch.initial_penalty});
      }
    }

    auto* detector_words_ptr =
        static_cast<uint64_t*>([impl->metal->batch_detector_words_buffer contents]);
    for (size_t local_shot = 0; local_shot < batch_size; ++local_shot) {
      std::memcpy(detector_words_ptr + local_shot * impl->metal->detector_word_count,
                  impl->shot_scratch[local_shot].detector_words.data(),
                  impl->metal->detector_word_count * sizeof(uint64_t));
    }

    auto* parent_counts_ptr =
        static_cast<uint32_t*>([impl->metal->batch_parent_counts_buffer contents]);
    auto* beam_limits_ptr =
        static_cast<uint32_t*>([impl->metal->batch_beam_limits_buffer contents]);
    auto* beam_growth_counts_ptr =
        static_cast<uint32_t*>([impl->metal->batch_beam_growth_counts_buffer contents]);
    auto* beam_ptr = static_cast<MetalBeamEntry*>([impl->metal->batch_input_buffer contents]);
    const bool dynamic_beam_enabled = gpu_dynamic_beam_enabled(config);
    const uint32_t initial_beam_limit = static_cast<uint32_t>(
        dynamic_beam_enabled ? std::min(config.beam_width, config.gpu_dynamic_initial_beam_width)
                             : config.beam_width);
    const MetalBatchLaunchConfig launch = {
        .num_shots = static_cast<uint32_t>(batch_size),
        .beam_width = static_cast<uint32_t>(config.beam_width),
        .detector_word_count = static_cast<uint32_t>(impl->metal->detector_word_count),
        .ranking_mode = metal_ranking_mode(config.ranking_mode),
    };
    const NSUInteger expand_width =
        std::min<NSUInteger>(impl->metal->batch_pipeline.maxTotalThreadsPerThreadgroup, 64);
    const NSUInteger expand_grid_size = batch_size * config.beam_width;
    auto upload_cpu_beam_state = [&]() {
      for (size_t local_shot = 0; local_shot < batch_size; ++local_shot) {
        auto& scratch = impl->shot_scratch[local_shot];
        parent_counts_ptr[local_shot] =
            scratch.invalid ? 0 : static_cast<uint32_t>(scratch.beam_entries.size());
        beam_limits_ptr[local_shot] = initial_beam_limit;
        beam_growth_counts_ptr[local_shot] = 0;
        for (size_t i = 0; i < scratch.beam_entries.size(); ++i) {
          const size_t dst = local_shot * config.beam_width + i;
          for (size_t k = 0; k < kMaxCompiledWideStateWords; ++k) {
            beam_ptr[dst].state_words[k] = scratch.beam_entries[i].state_words[k];
          }
          beam_ptr[dst].log_mass0 = metal_log_mass(scratch.beam_entries[i].mass0);
          beam_ptr[dst].log_mass1 = metal_log_mass(scratch.beam_entries[i].mass1);
          beam_ptr[dst].penalty = static_cast<float>(scratch.beam_entries[i].penalty);
          beam_ptr[dst].log_total_mass = metal_log_mass(total_entry_mass(scratch.beam_entries[i]));
        }
      }
    };

    bool gpu_state_resident = false;
    size_t layer_index = 0;
    while (layer_index < impl->layers.size()) {
      const bool exact_merge = should_exact_merge_layer(config.gpu_merge_period, layer_index,
                                                        impl->layers.size());
      const bool gpu_select_only = !exact_merge && can_use_gpu_topk_without_merge(config);

      if (gpu_select_only && can_use_persistent_gpu_segment(config)) {
        if (!gpu_state_resident) {
          upload_cpu_beam_state();
          gpu_state_resident = true;
        }
        size_t run_end = layer_index;
        while (run_end < impl->layers.size() &&
               !should_exact_merge_layer(config.gpu_merge_period, run_end, impl->layers.size())) {
          run_end += 1;
        }
        const size_t run_length = run_end - layer_index;
        @autoreleasepool {
          id<MTLCommandBuffer> command_buffer = [impl->metal->command_queue commandBuffer];
          const MetalPersistentLaunchConfig persistent_launch = {
              .num_shots = static_cast<uint32_t>(batch_size),
              .beam_width = static_cast<uint32_t>(config.beam_width),
              .detector_word_count = static_cast<uint32_t>(impl->metal->detector_word_count),
              .layer_start = static_cast<uint32_t>(layer_index),
              .num_layers = static_cast<uint32_t>(run_length),
              .ranking_mode = metal_ranking_mode(config.ranking_mode),
              .dynamic_beam_enabled = dynamic_beam_enabled ? 1u : 0u,
              .dynamic_initial_beam_width = initial_beam_limit,
              .dynamic_confidence_threshold =
                  static_cast<float>(config.gpu_dynamic_confidence_threshold),
              .pad0 = 0,
          };
          id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
          id<MTLComputePipelineState> persistent_pipeline = impl->metal->persistent_pipeline;
          if (config.beam_width > 1024) {
            persistent_pipeline = impl->metal->persistent_streaming_histogram_pipeline;
          } else if (config.beam_width >= 96) {
            persistent_pipeline = impl->metal->persistent_histogram_pipeline;
          }
          [encoder setComputePipelineState:persistent_pipeline];
          [encoder setBuffer:impl->metal->batch_input_buffer offset:0 atIndex:0];
          [encoder setBuffer:impl->metal->batch_next_buffer offset:0 atIndex:1];
          [encoder setBuffer:impl->metal->batch_output_buffer offset:0 atIndex:2];
          [encoder setBuffer:impl->metal->batch_detector_words_buffer offset:0 atIndex:3];
          [encoder setBuffer:impl->metal->persistent_layers_buffer offset:0 atIndex:4];
          [encoder setBuffer:impl->metal->persistent_terms_buffer offset:0 atIndex:5];
          [encoder setBuffer:impl->metal->batch_parent_counts_buffer offset:0 atIndex:6];
          [encoder setBytes:&persistent_launch length:sizeof(MetalPersistentLaunchConfig) atIndex:7];
          [encoder setBuffer:impl->metal->batch_beam_limits_buffer offset:0 atIndex:8];
          [encoder setBuffer:impl->metal->batch_beam_growth_counts_buffer offset:0 atIndex:9];
          const size_t threadgroup_width_beam =
              dynamic_beam_enabled ? static_cast<size_t>(initial_beam_limit) : config.beam_width;
          const NSUInteger persistent_width = preferred_sort_threadgroup_width(
              persistent_pipeline.maxTotalThreadsPerThreadgroup, threadgroup_width_beam);
          [encoder dispatchThreadgroups:MTLSizeMake(batch_size, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(persistent_width, 1, 1)];
          [encoder endEncoding];
          [command_buffer commit];
          [command_buffer waitUntilCompleted];
          if (command_buffer.error != nil) {
            impl->gpu_enabled = false;
            impl->backend = "cpu-fallback (Metal batch execution failed: " +
                            ns_error_string(command_buffer.error) + ")";
            for (size_t i = start; i < start + batch_size; ++i) {
              decode_shot(shots[i].hits);
              obs_predicted_masks[i] = predicted_obs_mask;
            }
            return;
          }
        }
        merge_skipped_layers += batch_size * run_length;
        layer_index = run_end;
        continue;
      }

      if (!gpu_state_resident) {
        upload_cpu_beam_state();
      }
      const auto& layer = impl->layers[layer_index];
      @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = [impl->metal->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:impl->metal->batch_pipeline];
        [encoder setBuffer:impl->metal->batch_input_buffer offset:0 atIndex:0];
        [encoder setBuffer:impl->metal->batch_output_buffer offset:0 atIndex:1];
        [encoder setBuffer:impl->metal->batch_detector_words_buffer offset:0 atIndex:2];
        [encoder setBuffer:impl->metal->batch_parent_counts_buffer offset:0 atIndex:3];
        [encoder setBuffer:layer.terms_buffer offset:0 atIndex:4];
        [encoder setBytes:&layer.config length:sizeof(GpuLayerConfig) atIndex:5];
        [encoder setBytes:&launch length:sizeof(MetalBatchLaunchConfig) atIndex:6];
        [encoder dispatchThreads:MTLSizeMake(expand_grid_size, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(expand_width, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.error != nil) {
          impl->gpu_enabled = false;
          impl->backend =
              "cpu-fallback (Metal batch execution failed: " + ns_error_string(command_buffer.error) +
              ")";
          for (size_t i = start; i < start + batch_size; ++i) {
            decode_shot(shots[i].hits);
            obs_predicted_masks[i] = predicted_obs_mask;
          }
          return;
        }
      }

      const auto* child_ptr =
          static_cast<const MetalChildCandidate*>([impl->metal->batch_output_buffer contents]);
      for (size_t local_shot = 0; local_shot < batch_size; ++local_shot) {
        auto& scratch = impl->shot_scratch[local_shot];
        if (gpu_state_resident && parent_counts_ptr[local_shot] == 0) {
          scratch.invalid = true;
          scratch.beam_entries.clear();
          scratch.next_entries.clear();
          continue;
        }
        if (!gpu_state_resident && scratch.invalid) {
          continue;
        }
        scratch.next_entries.clear();
        const size_t parent_count =
            gpu_state_resident ? parent_counts_ptr[local_shot] : scratch.beam_entries.size();
        if (exact_merge) {
          merge_calls += 1;
          merge_input_candidates += parent_count * 2;
          collapse_child_candidates_into_entries(
              child_ptr + local_shot * config.beam_width * 2, parent_count * 2, impl->num_words,
              &impl->collapse_buckets, &impl->used_bucket_indices, &scratch.next_entries);
          merge_output_candidates += scratch.next_entries.size();
          if (scratch.next_entries.size() < parent_count * 2) {
            merge_duplicate_layers += 1;
          }
        } else {
          merge_skipped_layers += 1;
          append_child_candidates_as_entries(
              child_ptr + local_shot * config.beam_width * 2, parent_count * 2,
              &scratch.next_entries);
        }
        const size_t active_beam_width =
            dynamic_beam_enabled
                ? std::max<size_t>(1, std::min<size_t>(config.beam_width, beam_limits_ptr[local_shot]))
                : config.beam_width;
        const size_t kept_states = keep_top_compiled_states(
            &scratch.next_entries, active_beam_width, config.ranking_mode, impl->num_words);
        normalize_compiled_items(&scratch.next_entries);
        (void)kept_states;
        if (scratch.next_entries.empty()) {
          scratch.invalid = true;
        } else {
          scratch.invalid = false;
        }
      }
      for (size_t local_shot = 0; local_shot < batch_size; ++local_shot) {
        auto& scratch = impl->shot_scratch[local_shot];
        scratch.beam_entries.swap(scratch.next_entries);
      }
      gpu_state_resident = false;
      layer_index += 1;
    }

    bool final_obs_resolved_on_gpu = false;
    if (gpu_state_resident) {
      @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = [impl->metal->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:impl->metal->final_obs_pipeline];
        [encoder setBuffer:impl->metal->batch_input_buffer offset:0 atIndex:0];
        [encoder setBuffer:impl->metal->batch_parent_counts_buffer offset:0 atIndex:1];
        [encoder setBuffer:impl->metal->batch_final_obs_buffer offset:0 atIndex:2];
        [encoder setBytes:&launch length:sizeof(MetalBatchLaunchConfig) atIndex:3];
        const NSUInteger final_width = preferred_sort_threadgroup_width(
            impl->metal->final_obs_pipeline.maxTotalThreadsPerThreadgroup, config.beam_width);
        [encoder dispatchThreadgroups:MTLSizeMake(batch_size, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(final_width, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.error != nil) {
          impl->gpu_enabled = false;
          impl->backend = "cpu-fallback (Metal final reduction failed: " +
                          ns_error_string(command_buffer.error) + ")";
          for (size_t i = start; i < start + batch_size; ++i) {
            decode_shot(shots[i].hits);
            obs_predicted_masks[i] = predicted_obs_mask;
            if (obs0_masses != nullptr) {
              (*obs0_masses)[i] = total_mass_obs0;
            }
            if (obs1_masses != nullptr) {
              (*obs1_masses)[i] = total_mass_obs1;
            }
          }
          return;
        }
      }
      const auto* final_obs_ptr =
          static_cast<const MetalFinalObsMass*>([impl->metal->batch_final_obs_buffer contents]);
      for (size_t local_shot = 0; local_shot < batch_size; ++local_shot) {
        const auto& final_obs = final_obs_ptr[local_shot];
        if (final_obs.valid == 0) {
          obs_predicted_masks[start + local_shot] = 0;
          if (obs0_masses != nullptr) {
            (*obs0_masses)[start + local_shot] = 0.0;
          }
          if (obs1_masses != nullptr) {
            (*obs1_masses)[start + local_shot] = 0.0;
          }
          continue;
        }
        const float log_shift = std::max(final_obs.log_mass0, final_obs.log_mass1);
        const double total_mass_obs0 = exp_shifted(final_obs.log_mass0, log_shift);
        const double total_mass_obs1 = exp_shifted(final_obs.log_mass1, log_shift);
        if (obs0_masses != nullptr) {
          (*obs0_masses)[start + local_shot] = total_mass_obs0;
        }
        if (obs1_masses != nullptr) {
          (*obs1_masses)[start + local_shot] = total_mass_obs1;
        }
        obs_predicted_masks[start + local_shot] = final_obs.log_mass1 > final_obs.log_mass0 ? 1 : 0;
      }
      final_obs_resolved_on_gpu = true;
    }

    if (!final_obs_resolved_on_gpu) {
      for (size_t local_shot = 0; local_shot < batch_size; ++local_shot) {
        auto& scratch = impl->shot_scratch[local_shot];
        if (scratch.invalid) {
          obs_predicted_masks[start + local_shot] = 0;
          if (obs0_masses != nullptr) {
            (*obs0_masses)[start + local_shot] = 0.0;
          }
          if (obs1_masses != nullptr) {
            (*obs1_masses)[start + local_shot] = 0.0;
          }
          continue;
        }
        double total_mass_obs0 = 0.0;
        double total_mass_obs1 = 0.0;
        for (const auto& item : scratch.beam_entries) {
          if (!fixed_wide_state_zero(item.state_words, impl->num_words)) {
            continue;
          }
          total_mass_obs0 += item.mass0;
          total_mass_obs1 += item.mass1;
        }
        if (obs0_masses != nullptr) {
          (*obs0_masses)[start + local_shot] = total_mass_obs0;
        }
        if (obs1_masses != nullptr) {
          (*obs1_masses)[start + local_shot] = total_mass_obs1;
        }
        obs_predicted_masks[start + local_shot] = total_mass_obs1 > total_mass_obs0 ? 1 : 0;
      }
    }
    if (dynamic_beam_enabled) {
      impl->dynamic_beam_limit_samples.reserve(impl->dynamic_beam_limit_samples.size() + batch_size);
      for (size_t local_shot = 0; local_shot < batch_size; ++local_shot) {
        impl->dynamic_beam_limit_samples.push_back(beam_limits_ptr[local_shot]);
        gpu_dynamic_beam_grow_events += beam_growth_counts_ptr[local_shot];
      }
    }
  }
  finalize_dynamic_beam_stats(this, impl->dynamic_beam_limit_samples);
}

bool TesseractTrellisGpuDecoder::using_gpu() const {
  return impl->gpu_enabled;
}

const std::string& TesseractTrellisGpuDecoder::backend_description() const {
  return impl->backend;
}
