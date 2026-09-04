// Copyright 2026 Google LLC
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

#ifndef MULTI_PASS_TESSERACT_DECODER_H
#define MULTI_PASS_TESSERACT_DECODER_H

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../tesseract.h"
#include "error_correlations.h"
#include "stim.h"

namespace tesseract_decoder {

class MultiPassTestPeer;

enum class SchedulingStrategy {
  Static,  // Schedules both components in every pass.
  Causal   // Derives each pass from component dependencies.
};

struct MultiPassTesseractConfig {
  // Settings applied independently to each component decoder. The DEM is the
  // monolithic input DEM and is replaced with each component DEM internally.
  TesseractConfig component_config;
  size_t num_passes = 2;
  std::vector<int> detector_components;
  SchedulingStrategy strategy = SchedulingStrategy::Causal;
};

struct MultiPassExecutionPlan {
  struct DemStatistics {
    size_t detector_count;
    size_t error_mechanism_count;
    double average_detector_degree;
  };

  struct Component {
    size_t id;
    int assignment_label;
    size_t active_detector_count;
    size_t decoder_detector_count;
    size_t error_mechanism_count;
    double average_active_detector_degree;
    bool sparsify_errors;
    int sparsify_reactivate_limit;
    bool affects_observable;
  };

  struct Dependency {
    size_t source_component;
    size_t target_component;
    size_t rule_count;
  };

  size_t num_passes;
  SchedulingStrategy strategy;
  DemStatistics monolithic_statistics;
  std::vector<Component> components;
  std::vector<Dependency> dependencies;
  std::vector<std::vector<size_t>> pass_schedule;

  std::string str() const;
};

/**
 * Decodes a detector error model by splitting it into exactly two detector components.
 *
 * One or two passes are supported. `detector_components[d]` assigns detector Dd to a component.
 * Every assignment must be nonnegative, and exactly two distinct labels must be present.
 * Two-pass reweighting requires `component_config.merge_errors=true`; one-pass decoding also
 * supports unmerged error mechanisms.
 */
class MultiPassTesseractDecoder : public Decoder {
 public:
  explicit MultiPassTesseractDecoder(MultiPassTesseractConfig config);

  /** Computes and returns the component schedule and statistics. */
  MultiPassExecutionPlan get_execution_plan() const;
  std::vector<int> decode(const std::vector<uint64_t>& detections);
  /** Returns predictions and the cost of predictions made during the final pass. */
  DecodeResult decode_result(const std::vector<uint64_t>& detections) override;

  size_t get_last_shot_num_reweights() const {
    return last_shot_num_reweights;
  }
  size_t num_components() const {
    return component_decoders.size();
  }

 private:
  struct LocalReweightRule {
    size_t target_comp_idx;
    size_t target_error_idx;
    double reweight_probability;
  };

  struct ComponentDecoder {
    std::unique_ptr<TesseractDecoder> decoder;
    int assignment_label = -1;
    size_t active_detector_count = 0;
    std::vector<double> original_costs;
    std::map<ComponentSymptom, std::vector<size_t>> symptom_to_error_index;
    std::vector<std::vector<LocalReweightRule>> error_index_to_rules;
    bool affects_observable = false;
  };

  size_t num_passes;
  SchedulingStrategy strategy;
  size_t total_global_detectors;
  size_t last_shot_num_reweights = 0;
  // Retained so execution-plan statistics can be computed on demand.
  stim::DetectorErrorModel monolithic_dem;
  std::vector<ComponentDecoder> component_decoders;
  std::vector<std::vector<size_t>> pass_schedule;
  std::vector<int> global_det_to_comp_id;

  void initialize(const stim::DetectorErrorModel& dem, const std::vector<int>& detector_components,
                  const TesseractConfig& base_config);
  void build_static_schedule();
  void build_causal_schedule();
  void restore_modified_costs(const std::vector<std::vector<size_t>>& modified_error_indices);

  friend class MultiPassTestPeer;
};

}  // namespace tesseract_decoder

#endif  // MULTI_PASS_TESSERACT_DECODER_H
