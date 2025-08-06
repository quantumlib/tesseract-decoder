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
#include <boost/functional/hash.hpp>  // For boost::hash_range
#include <cassert>
#include <functional>  // For std::hash (though not strictly necessary here, but good practice)
#include <iostream>

namespace {

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
  os << "[";
  bool is_first = true;
  for (auto& x : vec) {
    if (!is_first) {
      os << ", ";
    }
    is_first = false;
    os << x;
  }
  os << "]";
  return os;
}

};  // namespace

namespace std {
template <>
struct hash<boost::dynamic_bitset<>> {
  size_t operator()(const boost::dynamic_bitset<>& bs) const {
    // Delegate to Boost's internal hash_value for dynamic_bitset
    // This is the correct and most efficient way.
    return boost::hash_value(bs);
  }
};
}  // namespace std

std::string TesseractConfig::str() {
  auto& config = *this;
  std::stringstream ss;
  ss << "TesseractConfig(";
  ss << "dem=DetectorErrorModel_Object" << ", ";
  ss << "det_beam=" << config.det_beam << ", ";
  ss << "no_revisit_dets=" << config.no_revisit_dets << ", ";
  ss << "at_most_two_errors_per_detector=" << config.at_most_two_errors_per_detector << ", ";
  ss << "verbose=" << config.verbose << ", ";
  ss << "pqlimit=" << config.pqlimit << ", ";
  ss << "det_orders=" << config.det_orders << ", ";
  ss << "det_penalty=" << config.det_penalty << ")";
  return ss.str();
}

std::string Node::str() {
  std::stringstream ss;
  auto& self = *this;
  ss << "Node(";
  ss << "errors=" << self.errors << ", ";
  ss << "cost=" << self.cost << ", ";
  ss << "num_detectors=" << self.num_detectors << ", ";
  return ss.str();
}

bool Node::operator>(const Node& other) const {
  return cost > other.cost || (cost == other.cost && num_detectors < other.num_detectors);
}

double TesseractDecoder::get_detcost(
    size_t d, const std::vector<DetectorCostTuple>& detector_cost_tuples) const {
  double min_cost = INF;
  double error_cost;
  ErrorCost ec;
  DetectorCostTuple dct;

  for (int ei : d2e[d]) {
    ec = error_costs[ei];
    if (ec.min_cost >= min_cost) break;

    dct = detector_cost_tuples[ei];
    if (!dct.error_blocked) {
      error_cost = ec.likelihood_cost / dct.detectors_count;
      if (error_cost < min_cost) {
        min_cost = error_cost;
      }
    }
  }

  return min_cost + config.det_penalty;
}

TesseractDecoder::TesseractDecoder(TesseractConfig config_) : config(config_) {
  config.dem = common::remove_zero_probability_errors(config.dem);
  if (config.det_orders.empty()) {
    config.det_orders.emplace_back(config.dem.count_detectors());
    std::iota(config.det_orders[0].begin(), config.det_orders[0].end(), 0);
  } else {
    for (size_t i = 0; i < config.det_orders.size(); ++i) {
      assert(config.det_orders[i].size() == config.dem.count_detectors());
    }
  }
  assert(config.det_orders.size());
  errors = get_errors_from_dem(config.dem.flattened());
  if (config.verbose) {
    for (auto& error : errors) {
      std::cout << error.str() << std::endl;
    }
  }
  num_detectors = config.dem.count_detectors();
  num_errors = config.dem.count_errors();
  num_observables = config.dem.count_observables();
  initialize_structures(config.dem.count_detectors());
}

void TesseractDecoder::initialize_structures(size_t num_detectors) {
  d2e.resize(num_detectors);
  edets.resize(num_errors);

  for (size_t ei = 0; ei < num_errors; ++ei) {
    edets[ei] = errors[ei].symptom.detectors;
    for (int d : edets[ei]) {
      d2e[d].push_back(ei);
    }
  }

  for (size_t i = 0; i < errors.size(); ++i) {
    error_costs.push_back({errors[i].likelihood_cost,
                           errors[i].likelihood_cost / errors[i].symptom.detectors.size()});
  }

  for (size_t d = 0; d < num_detectors; ++d) {
    std::sort(d2e[d].begin(), d2e[d].end(), [this](size_t idx_a, size_t idx_b) {
      return error_costs[idx_a].min_cost < error_costs[idx_b].min_cost;
    });
  }

  eneighbors.resize(num_errors);

  std::vector<boost::dynamic_bitset<>> edets_bitsets(num_errors,
                                                     boost::dynamic_bitset<>(num_detectors));
  for (size_t ei = 0; ei < num_errors; ++ei) {
    for (int d : edets[ei]) {
      edets_bitsets[ei][d] = 1;
    }
  }

  for (size_t ei = 0; ei < num_errors; ++ei) {
    boost::dynamic_bitset<> neighbor_set(num_detectors, false);
    for (int d : edets[ei]) {
      for (int oei : d2e[d]) {
        // Unify detectors from neighboring errors
        neighbor_set |= edets_bitsets[oei];
      }
    }
    // Remove detectors from error's own set
    neighbor_set &= ~edets_bitsets[ei];

    for (size_t d = neighbor_set.find_first(); d != boost::dynamic_bitset<>::npos;
         d = neighbor_set.find_next(d)) {
      eneighbors[ei].push_back(d);
    }
  }
}

void TesseractDecoder::decode_to_errors(const std::vector<uint64_t>& detections) {
  std::vector<size_t> best_errors;
  double best_cost = std::numeric_limits<double>::max();
  assert(config.det_orders.size());

  if (config.beam_climbing) {
    for (int beam = config.det_beam; beam >= 0; --beam) {
      size_t detector_order = beam % config.det_orders.size();
      decode_to_errors(detections, detector_order, beam);
      double local_cost = cost_from_errors(predicted_errors_buffer);
      if (!low_confidence_flag && local_cost < best_cost) {
        best_errors = predicted_errors_buffer;
        best_cost = local_cost;
      }
      if (config.verbose) {
        std::cout << "for detector_order " << detector_order << " beam " << beam
                  << " got low confidence " << low_confidence_flag << " and cost " << local_cost
                  << " and obs_mask " << get_flipped_observables(predicted_errors_buffer)
                  << ". Best cost so far: " << best_cost << std::endl;
      }
    }
  } else {
    for (size_t detector_order = 0; detector_order < config.det_orders.size(); ++detector_order) {
      decode_to_errors(detections, detector_order, config.det_beam);
      double local_cost = cost_from_errors(predicted_errors_buffer);
      if (!low_confidence_flag && local_cost < best_cost) {
        best_errors = predicted_errors_buffer;
        best_cost = local_cost;
      }
      if (config.verbose) {
        std::cout << "for detector_order " << detector_order << " beam " << config.det_beam
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
    size_t detector_order, const std::vector<size_t>& errors, boost::dynamic_bitset<>& detectors,
    std::vector<DetectorCostTuple>& detector_cost_tuples) const {
  for (size_t ei : errors) {
    size_t min_detector = std::numeric_limits<size_t>::max();
    for (size_t d = 0; d < num_detectors; ++d) {
      if (detectors[config.det_orders[detector_order][d]]) {
        min_detector = config.det_orders[detector_order][d];
        break;
      }
    }

    for (int oei : d2e[min_detector]) {
      detector_cost_tuples[oei].error_blocked = 1;
      if (!config.at_most_two_errors_per_detector && oei == ei) break;
    }

    for (int d : edets[ei]) {
      detectors[d] = !detectors[d];
      if (!detectors[d] && config.at_most_two_errors_per_detector) {
        for (int oei : d2e[d]) {
          detector_cost_tuples[oei].error_blocked = 1;
        }
      }
    }
  }
}

void TesseractDecoder::decode_to_errors(const std::vector<uint64_t>& detections,
                                        size_t detector_order, size_t detector_beam) {
  predicted_errors_buffer.clear();
  low_confidence_flag = false;

  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
  std::unordered_map<size_t, std::unordered_set<boost::dynamic_bitset<>>> visited_detectors;

  boost::dynamic_bitset<> initial_detectors(num_detectors, false);
  std::vector<DetectorCostTuple> initial_detector_cost_tuples(num_errors);

  for (size_t d : detections) {
    initial_detectors[d] = true;
    for (int ei : d2e[d]) {
      ++initial_detector_cost_tuples[ei].detectors_count;
    }
  }

  double initial_cost = 0;
  for (size_t d : detections) {
    initial_cost += get_detcost(d, initial_detector_cost_tuples);
  }

  if (initial_cost == INF) {
    low_confidence_flag = true;
    return;
  }

  size_t min_num_detectors = detections.size();
  size_t max_num_detectors = min_num_detectors + detector_beam;

  std::vector<size_t> next_errors;
  boost::dynamic_bitset<> next_detectors;
  std::vector<DetectorCostTuple> next_detector_cost_tuples;

  pq.push({initial_cost, min_num_detectors, std::vector<size_t>()});
  size_t num_pq_pushed = 1;

  while (!pq.empty()) {
    const Node node = pq.top();
    pq.pop();

    if (node.num_detectors > max_num_detectors) continue;

    boost::dynamic_bitset<> detectors = initial_detectors;
    std::vector<DetectorCostTuple> detector_cost_tuples(num_errors);
    flip_detectors_and_block_errors(detector_order, node.errors, detectors, detector_cost_tuples);

    if (node.num_detectors == 0) {
      if (config.verbose) {
        std::cout << "activated_errors = ";
        for (size_t oei : node.errors) {
          std::cout << oei << ", ";
        }
        std::cout << std::endl;
        std::cout << "activated_detectors = ";
        for (size_t d = 0; d < num_detectors; ++d) {
          if (detectors[d]) {
            std::cout << d << ", ";
          }
        }
        std::cout << std::endl;
        std::cout.precision(13);
        std::cout << "Decoding complete. Cost: " << node.cost
                  << " num_pq_pushed = " << num_pq_pushed << std::endl;
      }
      predicted_errors_buffer = node.errors;
      return;
    }

    if (config.no_revisit_dets && !visited_detectors[node.num_detectors].insert(detectors).second)
      continue;

    if (config.verbose) {
      std::cout.precision(13);
      std::cout << "len(pq) = " << pq.size() << " num_pq_pushed = " << num_pq_pushed << std::endl;
      std::cout << "num_detectors = " << node.num_detectors
                << " max_num_detectors = " << max_num_detectors << " cost = " << node.cost
                << std::endl;
      std::cout << "activated_errors = ";
      for (size_t oei : node.errors) {
        std::cout << oei << ", ";
      }
      std::cout << std::endl;
      std::cout << "activated_detectors = ";
      for (size_t d = 0; d < num_detectors; ++d) {
        if (detectors[d]) {
          std::cout << d << ", ";
        }
      }
      std::cout << std::endl;
    }

    if (node.num_detectors < min_num_detectors) {
      min_num_detectors = node.num_detectors;
      if (config.no_revisit_dets) {
        for (size_t i = min_num_detectors + detector_beam + 1; i <= max_num_detectors; ++i) {
          visited_detectors[i].clear();
        }
      }
      max_num_detectors = std::min(max_num_detectors, min_num_detectors + detector_beam);
    }

    for (size_t d = 0; d < num_detectors; ++d) {
      if (!detectors[d]) continue;
      for (int ei : d2e[d]) {
        ++detector_cost_tuples[ei].detectors_count;
      }
    }

    next_detector_cost_tuples = detector_cost_tuples;

    size_t min_detector = std::numeric_limits<size_t>::max();
    for (size_t d = 0; d < num_detectors; ++d) {
      if (detectors[config.det_orders[detector_order][d]]) {
        min_detector = config.det_orders[detector_order][d];
        break;
      }
    }

    if (config.at_most_two_errors_per_detector) {
      for (int ei : d2e[min_detector]) {
        next_detector_cost_tuples[ei].error_blocked = 1;
      }
    }

    size_t prev_ei = std::numeric_limits<size_t>::max();
    std::vector<double> detector_cost_cache(num_detectors, -1);

    for (int ei : d2e[min_detector]) {
      if (detector_cost_tuples[ei].error_blocked) continue;

      if (prev_ei != std::numeric_limits<size_t>::max()) {
        for (int d : edets[prev_ei]) {
          int fired = detectors[d] ? 1 : -1;
          for (int oei : d2e[d]) {
            next_detector_cost_tuples[oei].detectors_count += fired;

            if (config.at_most_two_errors_per_detector &&
                next_detector_cost_tuples[oei].error_blocked == 2) {
              next_detector_cost_tuples[oei].error_blocked = 0;
            }
          }
        }
      }
      prev_ei = ei;

      next_errors = node.errors;
      next_errors.push_back(ei);
      next_detectors = detectors;
      next_detector_cost_tuples[ei].error_blocked = 1;

      double next_cost = node.cost + errors[ei].likelihood_cost;
      size_t next_num_detectors = node.num_detectors;

      for (int d : edets[ei]) {
        next_detectors[d] = !next_detectors[d];
        int fired = next_detectors[d] ? 1 : -1;
        next_num_detectors += fired;
        for (int oei : d2e[d]) {
          next_detector_cost_tuples[oei].detectors_count += fired;
        }

        if (!next_detectors[d] && config.at_most_two_errors_per_detector) {
          for (int oei : d2e[d]) {
            next_detector_cost_tuples[oei].error_blocked =
                next_detector_cost_tuples[oei].error_blocked == 1
                    ? 1
                    : 2;  // we store '2' value to indicate an error that was blocked due to the
                          // '--at-most-two-error-per-detector' heuristic, in order to revert it in
                          // the next decoding iteration
          }
        }
      }

      if (next_num_detectors > max_num_detectors) continue;

      if (config.no_revisit_dets && visited_detectors[next_num_detectors].find(next_detectors) !=
                                        visited_detectors[next_num_detectors].end())
        continue;

      for (int d : edets[ei]) {
        if (detectors[d]) {
          if (detector_cost_cache[d] == -1) {
            detector_cost_cache[d] = get_detcost(d, detector_cost_tuples);
          }
          next_cost -= detector_cost_cache[d];
        } else {
          next_cost += get_detcost(d, next_detector_cost_tuples);
        }
      }

      for (int od : eneighbors[ei]) {
        if (!detectors[od] || !next_detectors[od]) continue;
        if (detector_cost_cache[od] == -1) {
          detector_cost_cache[od] = get_detcost(od, detector_cost_tuples);
        }
        next_cost -= detector_cost_cache[od];
        next_cost += get_detcost(od, next_detector_cost_tuples);
      }

      if (next_cost == INF) continue;

      pq.push({next_cost, next_num_detectors, next_errors});
      ++num_pq_pushed;

      if (num_pq_pushed > config.pqlimit) {
        low_confidence_flag = true;
        return;
      }
    }
  }

  assert(pq.empty());
  if (config.verbose) {
    std::cout << "Decoding failed to converge within beam limit." << std::endl;
  }
  low_confidence_flag = true;
}

double TesseractDecoder::cost_from_errors(const std::vector<size_t>& predicted_errors) {
  double total_cost = 0;
  // Iterate over all errors and compute the cost
  for (size_t ei : predicted_errors) {
    total_cost += errors[ei].likelihood_cost;
  }
  return total_cost;
}

std::vector<int> TesseractDecoder::get_flipped_observables(
    const std::vector<size_t>& predicted_errors) {
  std::unordered_set<int> flipped_observables_set;

  // Iterate over all errors and compute the mask.
  // We use a set to perform an XOR-like sum.
  // If an observable is already in the set, we remove it (XORing with itself).
  // If it's not, we add it.
  for (size_t ei : predicted_errors) {
    for (int obs_index : errors[ei].symptom.observables) {
      if (flipped_observables_set.count(obs_index)) {
        flipped_observables_set.erase(obs_index);
      } else {
        flipped_observables_set.insert(obs_index);
      }
    }
  }

  // Convert the set to a vector and return it.
  std::vector<int> flipped_observables(flipped_observables_set.begin(),
                                       flipped_observables_set.end());
  // Sort observables
  std::sort(flipped_observables.begin(), flipped_observables.end());
  return flipped_observables;
}

std::vector<int> TesseractDecoder::decode(const std::vector<uint64_t>& detections) {
  decode_to_errors(detections);
  return get_flipped_observables(predicted_errors_buffer);
}

void TesseractDecoder::decode_shots(std::vector<stim::SparseShot>& shots,
                                    std::vector<std::vector<int>>& obs_predicted) {
  obs_predicted.resize(shots.size());
  for (size_t i = 0; i < shots.size(); ++i) {
    obs_predicted[i] = decode(shots[i].hits);
  }
}
