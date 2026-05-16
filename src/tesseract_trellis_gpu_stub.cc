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

#include <utility>

struct TesseractTrellisGpuDecoder::Impl {
  explicit Impl(TesseractTrellisConfig config) : cpu_decoder(std::move(config)) {}

  TesseractTrellisDecoder cpu_decoder;
  std::string backend =
      "cpu-fallback (GPU prototype is blocked here: Bazel Apple Metal toolchain support is not configured in this workspace)";
};

namespace {

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

TesseractTrellisGpuDecoder::TesseractTrellisGpuDecoder(TesseractTrellisConfig config_)
    : config(std::move(config_)), impl(std::make_unique<Impl>(config)) {}

TesseractTrellisGpuDecoder::~TesseractTrellisGpuDecoder() = default;

void TesseractTrellisGpuDecoder::decode_shot(const std::vector<uint64_t>& detections) {
  impl->cpu_decoder.decode_shot(detections);
  copy_stats_from_cpu(this, impl->cpu_decoder);
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
  (void)shot_batch_size;
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
}

bool TesseractTrellisGpuDecoder::using_gpu() const {
  return false;
}

const std::string& TesseractTrellisGpuDecoder::backend_description() const {
  return impl->backend;
}
