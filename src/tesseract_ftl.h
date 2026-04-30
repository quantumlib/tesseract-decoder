
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

#ifndef TESSERACT_FTL_DECODER_H
#define TESSERACT_FTL_DECODER_H

#include <boost/dynamic_bitset.hpp>
#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common.h"
#include "stim.h"
#include "tesseract.h"
#include "utils.h"
#include "visualization.h"

constexpr size_t DEFAULT_FTL_SUBSET_DETCOST_SIZE = 0;

enum class FTLDetectorChoicePolicy : uint8_t {
  kOrder = 0,
  kFewestIncidentErrors = 1,
  kLargestBudget = 2,
  kLargestBudgetPerIncident = 3,
};

enum class FTLErrorOrderPolicy : uint8_t {
  kStatic = 0,
  kReducedCost = 1,
};

struct TesseractFTLConfig {
  stim::DetectorErrorModel dem;
  int det_beam = DEFAULT_DET_BEAM;
  bool beam_climbing = false;
  bool no_revisit_dets = true;

  bool verbose = false;
  bool merge_errors = true;
  size_t pqlimit = DEFAULT_PQLIMIT;
  std::vector<std::vector<size_t>> det_orders;
  double det_penalty = 0;
  bool create_visualization = false;
  bool ignore_blocked_errors_in_heuristic = false;
  size_t num_min_dets_to_consider = 1;
  FTLDetectorChoicePolicy detector_choice_policy = FTLDetectorChoicePolicy::kOrder;
  FTLErrorOrderPolicy error_order_policy = FTLErrorOrderPolicy::kStatic;
  size_t root_det_order_count = 1;
  size_t root_det_order_depth = 0;
  size_t exact_child_refine_count = 0;

  // 0 = delegate to the original Tesseract detcost heuristic.
  // 1 = use the singleton fractional lower bound implemented in this file.
  size_t subset_detcost_size = DEFAULT_FTL_SUBSET_DETCOST_SIZE;

  std::string str();
};

enum class FTLHeuristicSource : uint8_t { kPlain = 0, kProjected = 1, kExact = 2 };

struct TesseractFTLStats {
  size_t num_pq_pushed = 0;
  size_t num_nodes_popped = 0;
  size_t max_queue_size = 0;

  size_t heuristic_calls = 0;
  size_t plain_heuristic_calls = 0;
  size_t projection_heuristic_calls = 0;
  size_t exact_refinement_calls = 0;
  size_t lp_calls = 0;
  size_t lp_reinserts = 0;
  size_t projected_nodes_generated = 0;
  size_t projected_nodes_refined = 0;
  double total_lp_refinement_gain = 0.0;
  double max_lp_refinement_gain = 0.0;
  double lp_total_seconds = 0.0;
  double chain_replay_total_seconds = 0.0;
  double component_build_total_seconds = 0.0;
  double component_candidate_total_seconds = 0.0;
  double component_union_total_seconds = 0.0;
  double component_dedup_total_seconds = 0.0;
  double component_finalize_total_seconds = 0.0;
  double simplex_total_seconds = 0.0;
  double projection_total_seconds = 0.0;
  size_t component_build_calls = 0;
  size_t simplex_calls = 0;
  size_t projection_calls = 0;
  size_t detector_choice_calls = 0;
  size_t error_ordering_calls = 0;
  size_t total_active_detectors_popped = 0;
  size_t total_root_order_candidates = 0;
  size_t total_min_detector_candidates = 0;
  size_t total_min_detectors_selected = 0;
  size_t total_min_detector_available_errors = 0;
  size_t total_min_detector_blocked_errors = 0;
  size_t total_child_candidates_considered = 0;
  size_t total_children_generated = 0;
  size_t total_children_beam_pruned = 0;
  size_t total_children_infeasible = 0;
  double total_selected_min_detector_budget = 0.0;
  size_t exact_child_pre_refinements = 0;

  void clear();
  void accumulate(const TesseractFTLStats& other);
};

struct TesseractFTLDecoder {
  TesseractFTLConfig config;
  Visualizer visualizer;

  explicit TesseractFTLDecoder(TesseractFTLConfig config);
  ~TesseractFTLDecoder();

  // Clears the predicted_errors_buffer and fills it with the decoded errors for
  // these detection events.
  void decode_to_errors(const std::vector<uint64_t>& detections);

  // Clears the predicted_errors_buffer and fills it with the decoded errors for
  // these detection events, using a specified detector ordering index.
  void decode_to_errors(const std::vector<uint64_t>& detections, size_t detector_order,
                        size_t detector_beam);

  // Returns the bitwise XOR of the observables flipped by the errors in the given array, indexed by
  // the original flattened DEM error indices.
  std::vector<int> get_flipped_observables(const std::vector<size_t>& predicted_errors) const;

  // Returns the sum of likelihood costs of the errors in the given array, indexed by the original
  // flattened DEM error indices.
  double cost_from_errors(const std::vector<size_t>& predicted_errors) const;

  std::vector<int> decode(const std::vector<uint64_t>& detections);
  void decode_shots(std::vector<stim::SparseShot>& shots,
                    std::vector<std::vector<int>>& obs_predicted);

  bool low_confidence_flag = false;
  std::vector<size_t> predicted_errors_buffer;
  std::vector<size_t> dem_error_to_error;
  std::vector<size_t> error_to_dem_error;
  std::vector<common::Error> errors;
  size_t num_observables = 0;
  size_t num_detectors = 0;
  TesseractFTLStats stats;

  struct SingletonPatternConstraint {
    std::vector<int> local_detectors;
    double rhs = 0.0;
  };

 private:
  struct ErrorCost {
    double likelihood_cost = 0;
    double min_cost = 0;
  };

  struct FTLNode {
    double f_cost = 0.0;
    double g_cost = 0.0;
    double h_cost = 0.0;
    size_t num_dets = 0;
    size_t depth = 0;
    int64_t error_chain_idx = -1;
    int64_t detector_state_idx = -1;
    int64_t warm_solution_idx = -1;
    int64_t exact_solution_idx = -1;
    bool exact_refined = false;
    FTLHeuristicSource heuristic_source = FTLHeuristicSource::kPlain;

    bool operator>(const FTLNode& other) const;
  };
  struct SingletonLPComponent {
    std::vector<int> detectors;
    std::vector<SingletonPatternConstraint> constraints;
    std::vector<int> cheapest_constraint_for_local_detector;
  };

  struct ExactSubsetSolution {
    double value = 0.0;
    size_t num_active_subsets = 0;
    size_t num_components = 0;
    size_t num_variables = 0;
    size_t num_constraints = 0;
    std::vector<int> active_detectors;
    std::vector<double> detector_budgets;
  };

  struct SingletonBuildResult {
    bool feasible = true;
    std::vector<SingletonLPComponent> components;
  };

  struct DynamicBitsetHash {
    size_t operator()(const boost::dynamic_bitset<>& bs) const;
  };

  std::vector<std::vector<int>> d2e;
  std::vector<std::vector<int>> edets;
  size_t num_errors = 0;
  std::vector<ErrorCost> error_costs;
  std::vector<common::ErrorChainNode> error_chain_arena;
  std::vector<boost::dynamic_bitset<>> detector_state_arena;
  std::vector<ExactSubsetSolution> exact_solution_arena;
  std::unordered_map<boost::dynamic_bitset<>, int64_t, DynamicBitsetHash> exact_solution_cache;
  mutable std::vector<uint64_t> candidate_error_marks;
  mutable uint64_t candidate_error_mark_epoch = 1;

  // If subset_detcost_size == 0, delegate to the original Tesseract decoder.
  std::unique_ptr<TesseractDecoder> plain_delegate;

  void initialize_structures(size_t num_detectors);

  void flip_detectors_and_block_errors(size_t detector_order, int64_t error_chain_idx,
                                       boost::dynamic_bitset<>& detectors,
                                       std::vector<uint8_t>& blocked_flags) const;

  SingletonBuildResult build_singleton_components(const boost::dynamic_bitset<>& detectors,
                                                  const std::vector<uint8_t>& blocked_flags);

  ExactSubsetSolution solve_exact_subset_lp(const boost::dynamic_bitset<>& detectors,
                                            const std::vector<uint8_t>& blocked_flags,
                                            int64_t warm_solution_idx);

  double project_from_exact_solution(const ExactSubsetSolution& solution,
                                     const boost::dynamic_bitset<>& detectors,
                                     const std::vector<uint8_t>& blocked_flags);

  std::vector<size_t> select_min_detectors(const boost::dynamic_bitset<>& detectors,
                                           const std::vector<uint8_t>& blocked_flags,
                                           size_t detector_order, size_t depth,
                                           const ExactSubsetSolution& exact_solution);

  std::vector<int> order_candidate_errors(size_t min_detector,
                                          const boost::dynamic_bitset<>& detectors,
                                          const std::vector<uint8_t>& blocked_flags,
                                          const ExactSubsetSolution& exact_solution);

  void reset_decode_state();
};

#endif  // TESSERACT_FTL_DECODER_H
