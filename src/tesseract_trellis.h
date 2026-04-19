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

#ifndef TESSERACT_TRELLIS_DECODER_H
#define TESSERACT_TRELLIS_DECODER_H

#include <cstdint>
#include <memory>
#include <vector>

#include "common.h"
#include "stim.h"

struct TesseractTrellisWideKernelBase;

enum class TesseractTrellisRankingMode {
  MassOnly,
  FutureDetcostRanked,
};

struct TesseractTrellisDetcostTransition {
  std::vector<uint32_t> fault_local_indices;
  std::vector<int32_t> next_local_indices;
  std::vector<double> current_costs;
  std::vector<double> next_costs;
};

struct TesseractTrellisWideLayerTemplate {
  double q = 0;
  double p = 0;
  uint64_t obs_mask = 0;
  size_t previous_width = 0;
  std::vector<uint32_t> surviving_local_indices;
  std::vector<int> current_active_detectors;
  std::vector<uint64_t> projected_fault_mask_words;
  std::vector<double> next_frontier_costs;
  TesseractTrellisDetcostTransition detcost_transition;
};

struct TesseractTrellisConfig {
  stim::DetectorErrorModel dem;
  size_t beam_width = 1024;
  double beam_eps = 0.0;
  bool verbose = false;
  bool track_kept_state_stats = false;
  TesseractTrellisRankingMode ranking_mode = TesseractTrellisRankingMode::MassOnly;
};

struct TesseractTrellisDecoder {
  explicit TesseractTrellisDecoder(TesseractTrellisConfig config);
  ~TesseractTrellisDecoder();

  void decode_shot(const std::vector<uint64_t>& detections);
  std::vector<int> decode(const std::vector<uint64_t>& detections);
  void decode_shots(std::vector<stim::SparseShot>& shots,
                    std::vector<std::vector<int>>& obs_predicted);

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

  std::vector<size_t> dem_error_to_error;
  std::vector<size_t> error_to_dem_error;
  std::vector<common::Error> errors;
  size_t num_observables = 0;
  size_t num_detectors = 0;
  std::vector<uint64_t> all_possible_detector_words;
  std::vector<uint64_t> actual_detector_words_scratch;
  std::vector<TesseractTrellisWideLayerTemplate> wide_layer_templates;
  std::unique_ptr<TesseractTrellisWideKernelBase> wide_kernel;
  std::vector<uint32_t> kept_state_histogram_scratch;
};

#endif  // TESSERACT_TRELLIS_DECODER_H
