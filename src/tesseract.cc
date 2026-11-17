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

#include "tesseract.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <utility>

namespace {

template <typename Callback>
class ScopeExit {
 public:
  explicit ScopeExit(Callback callback) : callback(std::move(callback)) {}
  ScopeExit(const ScopeExit&) = delete;
  ScopeExit& operator=(const ScopeExit&) = delete;
  ~ScopeExit() noexcept {
    callback();
  }

 private:
  Callback callback;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
  os << "[";
  bool is_first = true;
  for (const auto& x : vec) {
    if (!is_first) {
      os << ", ";
    }
    is_first = false;
    os << x;
  }
  return os << "]";
}

int suggest_sparsify_reactivate_limit_capped(size_t num_detectors, int sparsify_base_degree,
                                             int max_limit) {
  if (sparsify_base_degree < 0) {
    throw std::invalid_argument("sparsify_base_degree must be >= 0.");
  }
  if (num_detectors == 0 || max_limit <= 0) {
    return 0;
  }
  double exponent = static_cast<double>(sparsify_base_degree) - 2.0;
  double max_result = static_cast<double>(max_limit);
  double log_result =
      exponent * std::log(4.5) - std::log(3.0) + std::log(static_cast<double>(num_detectors));
  if (!std::isfinite(log_result) || log_result >= std::log(max_result)) {
    return max_limit;
  }
  double result = std::exp(log_result);
  if (!std::isfinite(result)) {
    return max_limit;
  }
  double rounded = std::round(result);
  if (rounded >= max_result) {
    return max_limit;
  }
  return static_cast<int>(rounded);
}

uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

}  // namespace

namespace tesseract_decoder {

std::string TesseractConfig::str() {
  auto& config = *this;
  std::stringstream ss;
  ss << "TesseractConfig(";
  ss << "dem=DetectorErrorModel_Object" << ", ";
  ss << "det_beam=" << config.det_beam << ", ";
  ss << "no_revisit_dets=" << config.no_revisit_dets << ", ";

  ss << "verbose=" << config.verbose << ", ";
  ss << "merge_errors=" << config.merge_errors << ", ";
  ss << "pqlimit=" << config.pqlimit << ", ";
  ss << "num_detector_orders=" << config.detector_orders.size() << ", ";
  ss << "det_penalty=" << config.det_penalty << ", ";
  ss << "create_visualization=" << config.create_visualization;
  ss << ")";
  return ss.str();
}

int suggest_sparsify_reactivate_limit(size_t num_detectors, int sparsify_base_degree) {
  return suggest_sparsify_reactivate_limit_capped(num_detectors, sparsify_base_degree,
                                                  std::numeric_limits<int>::max());
}

std::string Node::str() {
  std::stringstream ss;
  auto& self = *this;
  ss << "Node(";
  ss << "error_chain_idx=" << self.error_chain_idx << ", ";
  ss << "cost=" << self.cost << ", ";
  ss << "num_dets=" << self.num_dets << ", ";
  return ss.str();
}

bool Node::operator>(const Node& other) const {
  return cost > other.cost || (cost == other.cost && num_dets < other.num_dets);
}

double TesseractDecoder::get_detcost(
    size_t d, const std::vector<DetectorCostTuple>& detector_cost_tuples) const {
  return get_detcost(d, detector_cost_tuples, d2e);
}

double TesseractDecoder::get_detcost(size_t d,
                                     const std::vector<DetectorCostTuple>& detector_cost_tuples,
                                     const std::vector<std::vector<int>>& active_d2e) const {
  double min_cost = INF;
  uint32_t min_det_cost_det_count = std::numeric_limits<uint32_t>::max();
  double error_cost;
  ErrorCost ec;
  DetectorCostTuple dct;

  for (int ei : active_d2e[d]) {
    ec = error_costs[ei];
    if (ec.likelihood_cost * min_det_cost_det_count >=
        min_cost * errors[ei].symptom.detectors.size())
      break;

    dct = detector_cost_tuples[ei];
    if (!dct.error_blocked) {
      error_cost = ec.likelihood_cost;
      if (error_cost * min_det_cost_det_count < min_cost * dct.detectors_count) {
        min_cost = error_cost;
        min_det_cost_det_count = dct.detectors_count;
      }
    }
  }

  return (min_cost / min_det_cost_det_count) + config.det_penalty;
}

TesseractDecoder::TesseractDecoder(TesseractConfig config_) : config(std::move(config_)) {
  if (config.detector_orders.empty()) {
    throw std::invalid_argument("At least one detector order is required.");
  }
  config.dem = common::flatten(config.dem);

  std::vector<size_t> dem_error_map(config.dem.count_errors());
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

  resolve_detector_orders(config.detector_orders, config.dem);
  errors = get_errors_from_dem(config.dem);
  if (config.verbose) {
    for (auto& error : errors) {
      std::cout << error.str() << "\n";
    }
    std::cout << std::flush;
  }
  num_detectors = config.dem.count_detectors();
  num_errors = errors.size();
  num_observables = config.dem.count_observables();
  initialize_structures(num_detectors);
  if (config.create_visualization) {
    auto detectors = get_detector_coords(config.dem);
    visualizer.add_detector_coords(detectors);
    visualizer.add_errors(errors);
  }
}

TesseractDecoder::~TesseractDecoder() = default;

size_t TesseractDecoder::error_count() const {
  return errors.size();
}

const common::Symptom& TesseractDecoder::error_symptom(size_t error_index) const {
  return errors.at(error_index).symptom;
}

double TesseractDecoder::error_probability(size_t error_index) const {
  return errors.at(error_index).get_probability();
}

size_t TesseractDecoder::dem_error_index(size_t error_index) const {
  return error_to_dem_error.at(error_index);
}

size_t TesseractDecoder::retained_error_index(size_t dem_error_index) const {
  if (dem_error_index >= dem_error_to_error.size()) {
    throw std::out_of_range("DEM error index " + std::to_string(dem_error_index) +
                            " is out of range for " + std::to_string(dem_error_to_error.size()) +
                            " DEM errors.");
  }
  size_t error_index = dem_error_to_error[dem_error_index];
  if (error_index == std::numeric_limits<size_t>::max()) {
    throw std::invalid_argument("DEM error index " + std::to_string(dem_error_index) +
                                " was removed from the decoder.");
  }
  return error_index;
}

std::vector<size_t> TesseractDecoder::collect_affected_detectors(
    const std::vector<size_t>& modified_error_indices) const {
  for (size_t error_index : modified_error_indices) {
    if (error_index >= errors.size()) {
      throw std::out_of_range("Modified error index " + std::to_string(error_index) +
                              " is out of range for " + std::to_string(errors.size()) +
                              " decoder errors.");
    }
  }

  std::vector<uint8_t> affected(d2e.size());
  std::vector<size_t> affected_detectors;
  for (size_t error_index : modified_error_indices) {
    for (int detector : edets[error_index]) {
      if (!affected[detector]) {
        affected[detector] = true;
        affected_detectors.push_back(detector);
      }
    }
  }
  return affected_detectors;
}

void TesseractDecoder::update_internal_costs(const std::vector<size_t>& modified_error_indices,
                                             const std::vector<size_t>& affected_detectors) {
  for (size_t ei : modified_error_indices) {
    double min_cost = errors[ei].symptom.detectors.empty()
                          ? errors[ei].likelihood_cost
                          : errors[ei].likelihood_cost / errors[ei].symptom.detectors.size();
    error_costs[ei] = {errors[ei].likelihood_cost, min_cost};
  }

  for (size_t detector : affected_detectors) {
    std::sort(d2e[detector].begin(), d2e[detector].end(), [this](size_t idx_a, size_t idx_b) {
      return error_costs[idx_a].min_cost < error_costs[idx_b].min_cost ||
             (error_costs[idx_a].min_cost == error_costs[idx_b].min_cost && idx_a < idx_b);
    });
  }
}

TesseractDecoder::ErrorProbabilityRollback TesseractDecoder::apply_error_probability_updates(
    const std::vector<ErrorProbabilityUpdate>& probability_updates) {
  std::vector<size_t> error_indices;
  error_indices.reserve(probability_updates.size());
  for (const auto& update : probability_updates) {
    if (!std::isfinite(update.probability) || update.probability <= 0 || update.probability >= 1) {
      throw std::invalid_argument("Error probability must be finite and between 0 and 1.");
    }
    error_indices.push_back(retained_error_index(update.dem_error_index));
  }

  ErrorProbabilityRollback rollback;
  rollback.previous_likelihood_costs.reserve(probability_updates.size());
  rollback.modified_error_indices.reserve(probability_updates.size());
  std::vector<uint8_t> modified(errors.size());
  for (size_t k = 0; k < probability_updates.size(); ++k) {
    size_t error_index = error_indices[k];
    if (!modified[error_index]) {
      rollback.previous_likelihood_costs.emplace_back(error_index,
                                                      errors[error_index].likelihood_cost);
      rollback.modified_error_indices.push_back(error_index);
      modified[error_index] = true;
    }
  }
  rollback.affected_detectors = collect_affected_detectors(rollback.modified_error_indices);

  for (size_t k = 0; k < probability_updates.size(); ++k) {
    errors[error_indices[k]].set_with_probability(probability_updates[k].probability);
  }
  update_internal_costs(rollback.modified_error_indices, rollback.affected_detectors);
  return rollback;
}

void TesseractDecoder::restore_error_probabilities(const ErrorProbabilityRollback& rollback) {
  for (const auto& [error_index, likelihood_cost] : rollback.previous_likelihood_costs) {
    errors[error_index].likelihood_cost = likelihood_cost;
  }
  update_internal_costs(rollback.modified_error_indices, rollback.affected_detectors);
}

void TesseractDecoder::ensure_eneighbors(size_t ei) {
  if (eneighbors_ready_[ei]) return;
  if (++eneighbor_gen_ == 0) {
    std::fill(eneighbor_seen_.begin(), eneighbor_seen_.end(), 0);
    eneighbor_gen_ = 1;
  }
  const uint32_t mark_self = eneighbor_gen_;
  if (++eneighbor_gen_ == 0) {
    std::fill(eneighbor_seen_.begin(), eneighbor_seen_.end(), 0);
    eneighbor_gen_ = 2;
  }
  const uint32_t mark_neighbor = eneighbor_gen_;
  for (int d : edets[ei]) {
    eneighbor_seen_[d] = mark_self;
  }
  auto& out = eneighbors[ei];
  for (int d : edets[ei]) {
    for (int oei : d2e[d]) {
      for (int od : edets[oei]) {
        if (eneighbor_seen_[od] != mark_self && eneighbor_seen_[od] != mark_neighbor) {
          eneighbor_seen_[od] = mark_neighbor;
          out.push_back(od);
        }
      }
    }
  }
  std::sort(out.begin(), out.end());
  eneighbors_ready_[ei] = 1;
}

void TesseractDecoder::initialize_structures(size_t num_detectors) {
  d2e.assign(num_detectors, {});
  edets.assign(num_errors, {});

  for (size_t ei = 0; ei < num_errors; ++ei) {
    edets[ei] = errors[ei].symptom.detectors;
    for (int d : edets[ei]) {
      d2e[d].push_back(ei);
    }
  }

  // Initial fill of error_costs and sorting of d2e for all errors
  error_costs.clear();
  error_costs.reserve(errors.size());
  for (size_t i = 0; i < errors.size(); ++i) {
    double min_cost = errors[i].symptom.detectors.empty()
                          ? errors[i].likelihood_cost
                          : errors[i].likelihood_cost / errors[i].symptom.detectors.size();
    error_costs.push_back({errors[i].likelihood_cost, min_cost});
  }

  for (size_t d = 0; d < num_detectors; ++d) {
    std::sort(d2e[d].begin(), d2e[d].end(), [this](size_t idx_a, size_t idx_b) {
      return error_costs[idx_a].min_cost < error_costs[idx_b].min_cost ||
             (error_costs[idx_a].min_cost == error_costs[idx_b].min_cost && idx_a < idx_b);
    });
  }

  // Lazy eneighbors state: computed on first expansion of each error ei.
  eneighbors.assign(num_errors, {});
  eneighbors_ready_.assign(num_errors, 0);
  eneighbor_seen_.assign(num_detectors, 0);
  eneighbor_gen_ = 1;

  // Precompute inverse detector order ranks: det_order_ranks_[order_idx][d] = position.
  cached_detector_orders_.clear();
  det_order_ranks_.clear();
  cached_detector_orders_.resize(config.detector_orders.size());
  det_order_ranks_.resize(config.detector_orders.size());
  for (size_t oi = 0; oi < config.detector_orders.size(); ++oi) {
    ensure_det_order_ranks(oi);
  }

  // Zobrist hash constants for residual detector vectors.
  zobrist_.resize(num_detectors);
  for (size_t d = 0; d < num_detectors; ++d) {
    zobrist_[d] = splitmix64(d + 1);
  }
  if (config.no_revisit_dets) {
    visited_.init(1024);
  }

  // Reusable per-search scratch buffers to eliminate per-pop heap allocations.
  initial_detectors_.resize(num_detectors, false);
  scratch_detectors_.resize(num_detectors, false);
  initial_dct_.assign(num_errors, {0, 0});
  scratch_dct_.assign(num_errors, {0, 0});
  scratch_next_dct_.assign(num_errors, {0, 0});
  scratch_det_cost_cache_.assign(num_detectors, 0.0);
  scratch_det_cost_stamp_.assign(num_detectors, 0);
  scratch_det_cost_gen_ = 1;

  if (config.sparsify_errors) {
    if (config.sparsify_base_degree <= 0) {
      throw std::invalid_argument(
          "sparsify_base_degree must be > 0 when sparsify_errors is enabled.");
    }
    if (config.sparsify_max_degree < -1) {
      throw std::invalid_argument("sparsify_max_degree must be >= -1.");
    }
    if (config.sparsify_reactivate_limit < -1) {
      throw std::invalid_argument("sparsify_reactivate_limit must be >= -1.");
    }
    if (config.sparsify_max_degree >= 0 &&
        config.sparsify_max_degree < config.sparsify_base_degree) {
      throw std::invalid_argument("sparsify_max_degree must be >= sparsify_base_degree.");
    }

    if (config.sparsify_reactivate_limit == -1) {
      int error_count_limit = static_cast<int>(
          std::min(num_errors, static_cast<size_t>(std::numeric_limits<int>::max())));
      config.sparsify_reactivate_limit = suggest_sparsify_reactivate_limit_capped(
          num_detectors, config.sparsify_base_degree, error_count_limit);
    }

    sparsify_mandatory_errors.clear();
    sparsify_optional_errors.clear();
    for (size_t ei = 0; ei < num_errors; ++ei) {
      int degree = errors[ei].symptom.detectors.size();
      if (degree <= config.sparsify_base_degree) {
        sparsify_mandatory_errors.push_back(ei);
      } else if (degree > config.sparsify_base_degree &&
                 (config.sparsify_max_degree == -1 || degree <= config.sparsify_max_degree)) {
        sparsify_optional_errors.push_back(ei);
      }
    }
    sparse_error_active.assign(num_errors, 0);
    sparse_d2e.resize(num_detectors);
  }
}

void TesseractDecoder::decode_to_errors(const std::vector<uint64_t>& detections) {
  predicted_errors_buffer.clear();
  low_confidence_flag = false;
  if (detections.empty()) {
    return;
  }
  if (config.sparsify_errors) {
    build_sparse_d2e(detections);
  }
  const auto& active_d2e = config.sparsify_errors ? sparse_d2e : d2e;

  std::vector<size_t> best_errors;
  double best_cost = std::numeric_limits<double>::max();
  if (config.detector_orders.empty()) {
    throw std::runtime_error("Detector orders list must not be empty before decoding.");
  }

  if (config.beam_climbing) {
    int beam = 0;
    int order_index = 0;
    for (int trial = 0; trial < std::max(config.det_beam + 1, int(config.detector_orders.size()));
         ++trial) {
      decode_to_errors_with_graph(detections, order_index, beam, active_d2e);
      double local_cost = cost_from_errors(predicted_errors_buffer);
      if (!low_confidence_flag && local_cost < best_cost) {
        best_errors = predicted_errors_buffer;
        best_cost = local_cost;
      }
      if (config.verbose) {
        std::cout << "for detector_order " << order_index << " beam " << beam
                  << " got low confidence " << low_confidence_flag << " and cost " << local_cost
                  << " and obs_mask " << get_flipped_observables(predicted_errors_buffer)
                  << ". Best cost so far: " << best_cost << std::endl;
      }
      beam += 1;
      order_index += 1;
      beam %= (config.det_beam + 1);
      order_index %= config.detector_orders.size();
    }
  } else {
    for (size_t order_index = 0; order_index < config.detector_orders.size(); ++order_index) {
      decode_to_errors_with_graph(detections, order_index, config.det_beam, active_d2e);
      double local_cost = cost_from_errors(predicted_errors_buffer);
      if (!low_confidence_flag && local_cost < best_cost) {
        best_errors = predicted_errors_buffer;
        best_cost = local_cost;
      }
      if (config.verbose) {
        std::cout << "for detector_order " << order_index << " beam " << config.det_beam
                  << " got low confidence " << low_confidence_flag << " and cost " << local_cost
                  << " and obs_mask " << get_flipped_observables(predicted_errors_buffer)
                  << ". Best cost so far: " << best_cost << std::endl;
      }
    }
  }
  predicted_errors_buffer = best_errors;
  low_confidence_flag = best_cost == std::numeric_limits<double>::max();
}

void TesseractDecoder::flip_detectors_and_block_errors(
    size_t detector_order_index, int64_t error_chain_idx, boost::dynamic_bitset<>& detectors,
    std::vector<DetectorCostTuple>& detector_cost_tuples,
    const std::vector<std::vector<int>>& active_d2e) const {
  int64_t walker_idx = error_chain_idx;
  while (walker_idx != -1) {
    const auto& node = error_chain_arena[walker_idx];
    size_t ei = node.error_index;
    size_t min_detector = node.min_detector;

    for (int oei : active_d2e[min_detector]) {
      detector_cost_tuples[oei].error_blocked = 1;
      if (oei == ei) break;
    }

    for (int d : edets[ei]) {
      detectors[d] = !detectors[d];
    }
    walker_idx = node.parent_idx;
  }
}

void TesseractDecoder::decode_to_errors(const std::vector<uint64_t>& detections,
                                        size_t detector_order_index, size_t detector_beam) {
  if (config.sparsify_errors) {
    build_sparse_d2e(detections);
  }
  const auto& active_d2e = config.sparsify_errors ? sparse_d2e : d2e;
  decode_to_errors_with_graph(detections, detector_order_index, detector_beam, active_d2e);
}

const std::vector<size_t>& TesseractDecoder::ensure_det_order_ranks(size_t detector_order_index) {
  const std::vector<size_t>& order = config.detector_orders.at(detector_order_index).get_order();
  if (detector_order_index < det_order_ranks_.size() &&
      det_order_ranks_[detector_order_index].size() == num_detectors &&
      cached_detector_orders_[detector_order_index] == order) {
    return det_order_ranks_[detector_order_index];
  }
  if (order.size() != num_detectors) {
    throw std::invalid_argument("Detector order has size " + std::to_string(order.size()) +
                                ", but the decoder has " + std::to_string(num_detectors) +
                                " detectors.");
  }
  std::vector<size_t> ranks(num_detectors, std::numeric_limits<size_t>::max());
  for (size_t pos = 0; pos < order.size(); ++pos) {
    const size_t d = order[pos];
    if (d >= num_detectors) {
      throw std::invalid_argument("Detector order contains out-of-range detector ID " +
                                  std::to_string(d) + " at position " + std::to_string(pos) + ".");
    }
    if (ranks[d] != std::numeric_limits<size_t>::max()) {
      throw std::invalid_argument("Detector order contains detector ID " + std::to_string(d) +
                                  " more than once.");
    }
    ranks[d] = pos;
  }
  if (det_order_ranks_.size() != config.detector_orders.size()) {
    cached_detector_orders_.resize(config.detector_orders.size());
    det_order_ranks_.resize(config.detector_orders.size());
  }
  cached_detector_orders_[detector_order_index] = order;
  det_order_ranks_[detector_order_index] = std::move(ranks);
  return det_order_ranks_[detector_order_index];
}

void TesseractDecoder::decode_to_errors_with_graph(
    const std::vector<uint64_t>& detections, size_t detector_order_index, size_t detector_beam,
    const std::vector<std::vector<int>>& active_d2e) {
  const std::vector<size_t>& rank_of_detector = ensure_det_order_ranks(detector_order_index);

  predicted_errors_buffer.clear();
  low_confidence_flag = false;
  error_chain_arena.clear();
  if (error_chain_arena.capacity() == 0 && config.pqlimit > 0) {
    error_chain_arena.reserve(std::min<size_t>(config.pqlimit, 1024));
  }

  pq_container_.clear();
  auto pq_push = [&](const Node& n) {
    pq_container_.push_back(n);
    std::push_heap(pq_container_.begin(), pq_container_.end(), std::greater<Node>{});
  };
  auto pq_pop = [&]() -> Node {
    std::pop_heap(pq_container_.begin(), pq_container_.end(), std::greater<Node>{});
    Node top = pq_container_.back();
    pq_container_.pop_back();
    return top;
  };

  if (config.no_revisit_dets) {
    if (visited_.empty()) visited_.init(1024);
    visited_.clear();
  }

  initial_detectors_.reset();
  std::fill(initial_dct_.begin(), initial_dct_.end(), DetectorCostTuple{0, 0});

  uint64_t initial_det_hash = 0ULL;
  size_t active_initial_dets = 0;
  for (size_t d : detections) {
    if (d >= num_detectors) {
      throw std::runtime_error(
          "Symptom " + std::to_string(d) +
          " references a detector >= num_detectors (= " + std::to_string(num_detectors) + ").");
    }
    const bool now_set = !initial_detectors_[d];
    initial_detectors_[d] = now_set;
    initial_det_hash ^= zobrist_[d];
    const int delta = now_set ? 1 : -1;
    active_initial_dets += delta;
    for (int ei : active_d2e[d]) {
      initial_dct_[ei].detectors_count += delta;
    }
  }
  if (active_initial_dets == 0) {
    return;
  }

  double initial_cost = 0;
  for (size_t d = initial_detectors_.find_first(); d != boost::dynamic_bitset<>::npos;
       d = initial_detectors_.find_next(d)) {
    initial_cost += get_detcost(d, initial_dct_, active_d2e);
  }

  if (initial_cost == INF) {
    low_confidence_flag = true;
    return;
  }

  size_t min_num_dets = active_initial_dets;
  size_t max_num_dets = min_num_dets + detector_beam;

  pq_push({initial_cost, min_num_dets, 0, -1});
  size_t num_pq_pushed = 1;

  auto cached_detcost = [&](size_t d, uint32_t stamp) -> double {
    if (scratch_det_cost_stamp_[d] != stamp) {
      scratch_det_cost_stamp_[d] = stamp;
      scratch_det_cost_cache_[d] = get_detcost(d, scratch_dct_, active_d2e);
    }
    return scratch_det_cost_cache_[d];
  };

  while (!pq_container_.empty()) {
    const Node node = pq_pop();

    if (node.num_dets > max_num_dets) continue;

    if (node.num_dets == 0) {
      if (config.create_visualization || config.verbose) {
        scratch_detectors_ = initial_detectors_;
        std::fill(scratch_dct_.begin(), scratch_dct_.end(), DetectorCostTuple{0, 0});
        flip_detectors_and_block_errors(detector_order_index, node.error_chain_idx,
                                        scratch_detectors_, scratch_dct_, active_d2e);
      }
      if (config.create_visualization) {
        visualizer.add_activated_errors(node.error_chain_idx, error_chain_arena);
        visualizer.add_activated_detectors(scratch_detectors_, num_detectors);
      }
      if (config.verbose) {
        std::cout << "activated_errors = ";
        int64_t walker_idx = node.error_chain_idx;
        while (walker_idx != -1) {
          std::cout << error_chain_arena[walker_idx].error_index << ", ";
          walker_idx = error_chain_arena[walker_idx].parent_idx;
        }
        std::cout << std::endl;
        std::cout << "activated_detectors = ";
        for (size_t d = 0; d < num_detectors; ++d) {
          if (scratch_detectors_[d]) {
            std::cout << d << ", ";
          }
        }
        std::cout << std::endl;
        std::cout.precision(13);
        std::cout << "Decoding complete. Cost: " << node.cost
                  << " num_pq_pushed = " << num_pq_pushed << std::endl;
      }
      predicted_errors_buffer.resize(node.depth);
      int64_t walker_idx = node.error_chain_idx;
      for (size_t i = 0; i < node.depth; ++i) {
        predicted_errors_buffer[node.depth - 1 - i] =
            error_to_dem_error[error_chain_arena[walker_idx].error_index];
        walker_idx = error_chain_arena[walker_idx].parent_idx;
      }
      return;
    }

    const uint64_t det_hash = (node.error_chain_idx == -1)
                                  ? initial_det_hash
                                  : error_chain_arena[node.error_chain_idx].det_hash;
    if (config.no_revisit_dets && !visited_.insert(det_hash)) continue;

    scratch_detectors_ = initial_detectors_;
    std::fill(scratch_dct_.begin(), scratch_dct_.end(), DetectorCostTuple{0, 0});
    flip_detectors_and_block_errors(detector_order_index, node.error_chain_idx, scratch_detectors_,
                                    scratch_dct_, active_d2e);

    if (config.create_visualization) {
      visualizer.add_activated_errors(node.error_chain_idx, error_chain_arena);
      visualizer.add_activated_detectors(scratch_detectors_, num_detectors);
    }
    if (config.verbose) {
      std::cout.precision(13);
      std::cout << "len(pq) = " << pq_container_.size() << " num_pq_pushed = " << num_pq_pushed
                << std::endl;
      std::cout << "num_dets = " << node.num_dets << " max_num_dets = " << max_num_dets
                << " cost = " << node.cost << std::endl;
      std::cout << "activated_errors = ";
      int64_t walker_idx = node.error_chain_idx;
      while (walker_idx != -1) {
        std::cout << error_chain_arena[walker_idx].error_index << ", ";
        walker_idx = error_chain_arena[walker_idx].parent_idx;
      }
      std::cout << std::endl;
      std::cout << "activated_detectors = ";
      for (size_t d = 0; d < num_detectors; ++d) {
        if (scratch_detectors_[d]) {
          std::cout << d << ", ";
        }
      }
      std::cout << std::endl;
    }

    if (node.num_dets < min_num_dets) {
      min_num_dets = node.num_dets;
      max_num_dets = std::min(max_num_dets, min_num_dets + detector_beam);
    }

    size_t min_detector = std::numeric_limits<size_t>::max();
    size_t best_rank = std::numeric_limits<size_t>::max();
    for (size_t d = scratch_detectors_.find_first(); d != boost::dynamic_bitset<>::npos;
         d = scratch_detectors_.find_next(d)) {
      if (rank_of_detector[d] < best_rank) {
        best_rank = rank_of_detector[d];
        min_detector = d;
      }
      for (int ei : active_d2e[d]) {
        ++scratch_dct_[ei].detectors_count;
      }
    }

    scratch_next_dct_ = scratch_dct_;
    if (++scratch_det_cost_gen_ == 0) {
      std::fill(scratch_det_cost_stamp_.begin(), scratch_det_cost_stamp_.end(), 0);
      scratch_det_cost_gen_ = 1;
    }
    const uint32_t pop_stamp = scratch_det_cost_gen_;
    size_t prev_ei = std::numeric_limits<size_t>::max();

    for (int ei : active_d2e[min_detector]) {
      if (scratch_dct_[ei].error_blocked) continue;

      if (prev_ei != std::numeric_limits<size_t>::max()) {
        for (int d : edets[prev_ei]) {
          int fired = scratch_detectors_[d] ? 1 : -1;
          for (int oei : active_d2e[d]) {
            scratch_next_dct_[oei].detectors_count += fired;
          }
        }
      }
      prev_ei = ei;
      scratch_next_dct_[ei].error_blocked = 1;

      uint64_t next_det_hash = det_hash;
      double next_cost = node.cost + errors[ei].likelihood_cost;
      size_t next_num_dets = node.num_dets;

      for (int d : edets[ei]) {
        int fired = scratch_detectors_[d] ? -1 : 1;
        next_num_dets += fired;
        next_det_hash ^= zobrist_[d];
        for (int oei : active_d2e[d]) {
          scratch_next_dct_[oei].detectors_count += fired;
        }
      }

      if (next_num_dets > max_num_dets) continue;

      if (config.no_revisit_dets && next_num_dets > 0 && visited_.contains(next_det_hash)) continue;

      for (int d : edets[ei]) {
        if (scratch_detectors_[d]) {
          next_cost -= cached_detcost(d, pop_stamp);
        } else {
          next_cost += get_detcost(d, scratch_next_dct_, active_d2e);
        }
      }

      ensure_eneighbors(ei);
      for (int od : eneighbors[ei]) {
        if (!scratch_detectors_[od]) continue;
        next_cost -= cached_detcost(od, pop_stamp);
        next_cost += get_detcost(od, scratch_next_dct_, active_d2e);
      }

      if (next_cost == INF) continue;

      error_chain_arena.push_back(
          {static_cast<size_t>(ei), min_detector, node.error_chain_idx, next_det_hash});
      const int64_t child_idx = static_cast<int64_t>(error_chain_arena.size() - 1);

      pq_push({next_cost, next_num_dets, node.depth + 1, child_idx});
      ++num_pq_pushed;

      if (num_pq_pushed > config.pqlimit) {
        if (config.verbose) {
          std::cout << "setting low confidence flag" << std::endl;
        }
        low_confidence_flag = true;
        return;
      }
    }
  }

  if (!pq_container_.empty()) {
    throw std::runtime_error("Priority queue should be empty after decoding failure.");
  }
  if (config.verbose) {
    std::cout << "Decoding failed to converge within beam limit." << std::endl;
  }
  low_confidence_flag = true;
}

double TesseractDecoder::cost_from_errors(const std::vector<size_t>& predicted_errors) const {
  double total_cost = 0;
  for (size_t dem_error_index : predicted_errors) {
    size_t error_index = dem_error_to_error.at(dem_error_index);
    if (error_index == std::numeric_limits<size_t>::max()) {
      throw std::invalid_argument("error index does not map to a retained decoder error");
    }
    total_cost += errors[error_index].likelihood_cost;
  }
  return total_cost;
}

std::vector<int> TesseractDecoder::get_flipped_observables(
    const std::vector<size_t>& predicted_errors) const {
  if (predicted_errors.empty()) return {};
  std::vector<uint8_t> parity(num_observables, 0);
  std::vector<int> extra;
  for (size_t dem_error_index : predicted_errors) {
    size_t error_index = dem_error_to_error.at(dem_error_index);
    if (error_index == std::numeric_limits<size_t>::max()) {
      throw std::invalid_argument("error index does not map to a retained decoder error");
    }
    for (int obs_index : errors[error_index].symptom.observables) {
      if (obs_index >= 0 && static_cast<size_t>(obs_index) < parity.size()) {
        parity[obs_index] ^= 1;
      } else {
        extra.push_back(obs_index);
      }
    }
  }
  std::vector<int> flipped_observables;
  for (size_t o = 0; o < parity.size(); ++o) {
    if (parity[o]) flipped_observables.push_back(static_cast<int>(o));
  }
  if (!extra.empty()) {
    std::sort(extra.begin(), extra.end());
    for (size_t i = 0; i < extra.size();) {
      size_t j = i + 1;
      while (j < extra.size() && extra[j] == extra[i]) ++j;
      if ((j - i) & 1) flipped_observables.push_back(extra[i]);
      i = j;
    }
    std::sort(flipped_observables.begin(), flipped_observables.end());
  }
  return flipped_observables;
}

std::vector<int> TesseractDecoder::decode(const std::vector<uint64_t>& detections) {
  decode_to_errors(detections);
  return get_flipped_observables(predicted_errors_buffer);
}

DecodeResult TesseractDecoder::decode_result(const std::vector<uint64_t>& detections) {
  decode_to_errors(detections);

  DecodeResult result;
  result.predictions = get_flipped_observables(predicted_errors_buffer);
  result.predicted_errors = predicted_errors_buffer;
  result.predicted_errors_populated = true;
  result.low_confidence = low_confidence_flag;
  result.total_cost = cost_from_errors(predicted_errors_buffer);
  return result;
}

DecodeResult TesseractDecoder::decode_result(
    const std::vector<uint64_t>& detections,
    const std::vector<ErrorProbabilityUpdate>& probability_updates) {
  if (probability_updates.empty()) {
    return decode_result(detections);
  }
  ErrorProbabilityRollback rollback = apply_error_probability_updates(probability_updates);
  ScopeExit restore_probabilities([&] { restore_error_probabilities(rollback); });
  return decode_result(detections);
}

void TesseractDecoder::decode_shots(std::vector<stim::SparseShot>& shots,
                                    std::vector<std::vector<int>>& obs_predicted) {
  obs_predicted.resize(shots.size());
  for (size_t i = 0; i < shots.size(); ++i) {
    obs_predicted[i] = decode(shots[i].hits);
  }
}

void TesseractDecoder::build_sparse_d2e(const std::vector<uint64_t>& detections) {
  std::vector<uint8_t> shot_dets(num_detectors, 0);
  for (uint64_t d : detections) {
    if (d < num_detectors) {
      shot_dets[d] ^= 1;
    }
  }

  std::fill(sparse_error_active.begin(), sparse_error_active.end(), 0);

  for (int ei : sparsify_mandatory_errors) {
    sparse_error_active[ei] = 1;
  }

  struct OptionalErrorCandidate {
    int error_index;
    int overlap;
    int degree;
    double likelihood_cost;
  };

  std::vector<OptionalErrorCandidate> candidates;
  candidates.reserve(sparsify_optional_errors.size());

  for (int ei : sparsify_optional_errors) {
    int overlap = 0;
    for (int d : errors[ei].symptom.detectors) {
      if (shot_dets[d]) {
        overlap++;
      }
    }
    if (overlap > 0) {
      candidates.push_back({ei, overlap, static_cast<int>(errors[ei].symptom.detectors.size()),
                            errors[ei].likelihood_cost});
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const OptionalErrorCandidate& a, const OptionalErrorCandidate& b) {
              if (a.overlap != b.overlap) {
                return a.overlap > b.overlap;
              }
              if (a.degree != b.degree) {
                return a.degree < b.degree;
              }
              if (a.likelihood_cost != b.likelihood_cost) {
                return a.likelihood_cost < b.likelihood_cost;
              }
              return a.error_index < b.error_index;
            });

  size_t limit = std::min(static_cast<size_t>(config.sparsify_reactivate_limit), candidates.size());
  for (size_t i = 0; i < limit; ++i) {
    sparse_error_active[candidates[i].error_index] = 1;
  }

  for (size_t d = 0; d < num_detectors; ++d) {
    sparse_d2e[d].clear();
    for (int ei : d2e[d]) {
      if (sparse_error_active[ei]) {
        sparse_d2e[d].push_back(ei);
      }
    }
  }
}

}  // namespace tesseract_decoder
