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

#ifndef TESSERACT_TRELLIS_GPU_DECODER_H
#define TESSERACT_TRELLIS_GPU_DECODER_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "stim.h"
#include "tesseract_trellis.h"

struct TesseractTrellisGpuDecoder {
  explicit TesseractTrellisGpuDecoder(TesseractTrellisConfig config);
  ~TesseractTrellisGpuDecoder();

  void decode_shot(const std::vector<uint64_t>& detections);
  std::vector<int> decode(const std::vector<uint64_t>& detections);
  void decode_shots(std::vector<stim::SparseShot>& shots,
                    std::vector<std::vector<int>>& obs_predicted);
  void decode_shots_batched(std::vector<stim::SparseShot>& shots,
                            std::vector<uint64_t>& obs_predicted_masks,
                            size_t shot_batch_size,
                            std::vector<double>* obs0_masses = nullptr,
                            std::vector<double>* obs1_masses = nullptr);

  bool using_gpu() const;
  const std::string& backend_description() const;

  TesseractTrellisConfig config;
  bool low_confidence_flag = false;
  size_t num_states_expanded = 0;
  size_t num_states_merged = 0;
  size_t max_beam_size_seen = 0;
  size_t max_frontier_width_seen = 0;
  size_t kept_state_sample_count = 0;
  size_t kept_state_min = 0;
  double kept_state_median = 0;
  double kept_state_mean = 0;
  size_t kept_state_max = 0;
  double time_expand_seconds = 0;
  double time_collapse_seconds = 0;
  double time_truncate_seconds = 0;
  double time_reconstruct_seconds = 0;
  uint64_t predicted_obs_mask = 0;
  double total_mass_obs0 = 0;
  double total_mass_obs1 = 0;
  size_t merge_calls = 0;
  size_t merge_input_candidates = 0;
  size_t merge_output_candidates = 0;
  size_t merge_duplicate_layers = 0;
  size_t merge_skipped_layers = 0;
  size_t gpu_dynamic_beam_limit_sample_count = 0;
  size_t gpu_dynamic_beam_limit_min = 0;
  double gpu_dynamic_beam_limit_median = 0;
  double gpu_dynamic_beam_limit_mean = 0;
  size_t gpu_dynamic_beam_limit_max = 0;
  size_t gpu_dynamic_beam_grow_events = 0;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

#endif  // TESSERACT_TRELLIS_GPU_DECODER_H
