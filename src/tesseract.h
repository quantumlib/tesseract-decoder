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

#ifndef TESSERACT_DECODER_H
#define TESSERACT_DECODER_H

#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <cstdint>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.h"
#include "decoder.h"
#include "stim.h"
#include "utils.h"
#include "visualization.h"

namespace tesseract_decoder {

constexpr size_t INF_DET_BEAM = std::numeric_limits<uint16_t>::max();
constexpr int DEFAULT_DET_BEAM = 5;
constexpr size_t DEFAULT_PQLIMIT = 200000;

int suggest_sparsify_reactivate_limit(size_t num_detectors, int sparsify_base_degree);

struct TesseractConfig {
  stim::DetectorErrorModel dem;
  int det_beam = DEFAULT_DET_BEAM;
  bool beam_climbing = false;
  bool no_revisit_dets = true;

  bool verbose = false;
  bool merge_errors = true;
  size_t pqlimit = DEFAULT_PQLIMIT;

  // Each entry resolves to one detector traversal permutation once the
  // decoder's concrete DEM is known.
  std::vector<DetectorOrder> detector_orders{DetectorOrder()};
  double det_penalty = 0;
  bool create_visualization = false;

  bool sparsify_errors = false;
  int sparsify_base_degree = -1;
  int sparsify_max_degree = -1;
  int sparsify_reactivate_limit = -1;

  std::string str();
};

// Flat open-addressed Zobrist hash set storing 64-bit syndrome hashes paired with 16-bit
// generation tags, supporting sentinel-free lookup (including hash == 0) and O(1) resets between
// detector-ordering trials via generation increment. Note: under no_revisit_dets=true, a 64-bit
// hash collision (~choose(N, 2) / 2^64 <= 1e-9 for N <= 2e5 visited states) acts as a false
// visited hit for that trial.
struct VisitedTable {
  std::vector<uint64_t> hashes;
  std::vector<uint16_t> gens;
  uint64_t mask = 0;
  uint16_t gen = 1;
  size_t count = 0;

  void init(size_t min_cap = 1024) {
    size_t cap = 16;
    while (cap < min_cap) cap <<= 1;
    hashes.assign(cap, 0ULL);
    gens.assign(cap, 0);
    mask = cap - 1;
    gen = 1;
    count = 0;
  }

  bool empty() const {
    return hashes.empty();
  }

  void clear() {
    count = 0;
    if (++gen == 0) {
      std::fill(gens.begin(), gens.end(), 0);
      gen = 1;
    }
  }

  bool contains(uint64_t hash) const {
    if (hashes.empty()) return false;
    size_t idx = static_cast<size_t>(hash) & mask;
    while (gens[idx] == gen) {
      if (hashes[idx] == hash) return true;
      idx = (idx + 1) & mask;
    }
    return false;
  }

  bool insert(uint64_t hash) {
    if (hashes.empty()) init(1024);
    if ((count + 1) * 10 >= hashes.size() * 7) grow();
    size_t idx = static_cast<size_t>(hash) & mask;
    while (gens[idx] == gen) {
      if (hashes[idx] == hash) return false;
      idx = (idx + 1) & mask;
    }
    hashes[idx] = hash;
    gens[idx] = gen;
    ++count;
    return true;
  }

  void grow() {
    const size_t new_cap = hashes.size() << 1;
    std::vector<uint64_t> new_hashes(new_cap, 0ULL);
    std::vector<uint16_t> new_gens(new_cap, 0);
    const uint64_t new_mask = new_cap - 1;
    for (size_t i = 0; i < hashes.size(); ++i) {
      if (gens[i] == gen) {
        const uint64_t h = hashes[i];
        size_t idx = static_cast<size_t>(h) & new_mask;
        while (new_gens[idx] == gen) idx = (idx + 1) & new_mask;
        new_hashes[idx] = h;
        new_gens[idx] = gen;
      }
    }
    hashes = std::move(new_hashes);
    gens = std::move(new_gens);
    mask = new_mask;
  }
};

class Node {
 public:
  double cost;
  // The number of activated detectors (dets for short) at this node
  size_t num_dets;
  size_t depth;
  int64_t error_chain_idx = -1;

  bool operator>(const Node& other) const;
  std::string str();
};

struct DetectorCostTuple {
  uint32_t error_blocked;
  uint32_t detectors_count;
};

struct ErrorCost {
  double likelihood_cost;
  double min_cost;
};

struct ErrorProbabilityUpdate {
  // Index into the flattened DEM supplied to the decoder.
  size_t dem_error_index;
  double probability;
};

struct TesseractDecoder : public Decoder {
  TesseractConfig config;
  Visualizer visualizer;

  explicit TesseractDecoder(TesseractConfig config);
  ~TesseractDecoder() override;

  // Clears the predicted_errors_buffer and fills it with the decoded errors for
  // these detection events.
  void decode_to_errors(const std::vector<uint64_t>& detections);

  // Clears the predicted_errors_buffer and fills it with the decoded errors for
  // these detection events, using a specified detector ordering index.
  void decode_to_errors(const std::vector<uint64_t>& detections, size_t detector_order_index,
                        size_t detector_beam);

  // Returns the bitwise XOR of the observables flipped by the errors in the given array, indexed by
  // the original flattened DEM error indices.
  std::vector<int> get_flipped_observables(const std::vector<size_t>& predicted_errors) const;

  // Returns the sum of likelihood costs of the errors in the given array, indexed by the original
  // flattened DEM error indices.
  double cost_from_errors(const std::vector<size_t>& predicted_errors) const;

  DecodeResult decode_result(const std::vector<uint64_t>& detections) override;

  // Decodes with temporary error probabilities. The original probabilities and
  // internal cost ordering are restored before this method returns or throws.
  DecodeResult decode_result(const std::vector<uint64_t>& detections,
                             const std::vector<ErrorProbabilityUpdate>& probability_updates);

  // Accessors for the retained (merged and nonzero) decoder errors. These are
  // used when preparing multipass correlations.
  size_t error_count() const;
  const common::Symptom& error_symptom(size_t error_index) const;
  double error_probability(size_t error_index) const;
  size_t dem_error_index(size_t error_index) const;

  std::vector<int> decode(const std::vector<uint64_t>& detections);
  void decode_shots(std::vector<stim::SparseShot>& shots,
                    std::vector<std::vector<int>>& obs_predicted);

  bool low_confidence_flag = false;
  std::vector<size_t> predicted_errors_buffer;
  std::vector<size_t> dem_error_to_error;
  std::vector<size_t> error_to_dem_error;
  std::vector<common::Error> errors;
  size_t num_observables;
  size_t num_detectors;

  std::vector<std::vector<int>>& get_eneighbors() {
    for (size_t ei = 0; ei < num_errors; ++ei) {
      ensure_eneighbors(ei);
    }
    return eneighbors;
  }
  void ensure_eneighbors(size_t ei);

  std::vector<std::vector<int>> d2e;
  std::vector<std::vector<int>> sparse_d2e;
  std::vector<uint8_t> sparse_error_active;
  std::vector<int> sparsify_mandatory_errors;
  std::vector<int> sparsify_optional_errors;

  std::vector<std::vector<int>> eneighbors;
  std::vector<std::vector<int>> edets;
  size_t num_errors;
  std::vector<ErrorCost> error_costs;
  std::vector<common::ErrorChainNode> error_chain_arena;

  void initialize_structures(size_t num_detectors);
  double get_detcost(size_t d, const std::vector<DetectorCostTuple>& detector_cost_tuples) const;
  double get_detcost(size_t d, const std::vector<DetectorCostTuple>& detector_cost_tuples,
                     const std::vector<std::vector<int>>& active_d2e) const;
  void flip_detectors_and_block_errors(size_t detector_order_index, int64_t error_chain_idx,
                                       boost::dynamic_bitset<>& detectors,
                                       std::vector<DetectorCostTuple>& detector_cost_tuples,
                                       const std::vector<std::vector<int>>& active_d2e) const;

 private:
  struct ErrorProbabilityRollback {
    std::vector<std::pair<size_t, double>> previous_likelihood_costs;
    std::vector<size_t> modified_error_indices;
    std::vector<size_t> affected_detectors;
  };

  ErrorProbabilityRollback apply_error_probability_updates(
      const std::vector<ErrorProbabilityUpdate>& probability_updates);
  void restore_error_probabilities(const ErrorProbabilityRollback& rollback);
  size_t retained_error_index(size_t dem_error_index) const;
  std::vector<size_t> collect_affected_detectors(
      const std::vector<size_t>& modified_error_indices) const;
  void update_internal_costs(const std::vector<size_t>& modified_error_indices,
                             const std::vector<size_t>& affected_detectors);
  void build_sparse_d2e(const std::vector<uint64_t>& detections);
  const std::vector<size_t>& ensure_det_order_ranks(size_t detector_order_index);
  void decode_to_errors_with_graph(const std::vector<uint64_t>& detections,
                                   size_t detector_order_index, size_t detector_beam,
                                   const std::vector<std::vector<int>>& active_d2e);

  std::vector<std::vector<size_t>> cached_detector_orders_;
  std::vector<std::vector<size_t>> det_order_ranks_;
  std::vector<uint64_t> zobrist_;
  VisitedTable visited_;
  std::vector<uint8_t> eneighbors_ready_;
  std::vector<uint32_t> eneighbor_seen_;
  uint32_t eneighbor_gen_ = 1;

  boost::dynamic_bitset<> initial_detectors_;
  boost::dynamic_bitset<> scratch_detectors_;
  std::vector<DetectorCostTuple> initial_dct_;
  std::vector<DetectorCostTuple> scratch_dct_;
  std::vector<DetectorCostTuple> scratch_next_dct_;
  std::vector<double> scratch_det_cost_cache_;
  std::vector<uint32_t> scratch_det_cost_stamp_;
  uint32_t scratch_det_cost_gen_ = 1;
  std::vector<Node> pq_container_;
};

}  // namespace tesseract_decoder

#endif  // TESSERACT_DECODER_H
