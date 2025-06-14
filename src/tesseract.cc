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
#include <iostream>

namespace
{

  template <typename T>
  std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec)
  {
    os << "[";
    bool is_first = true;
    for (auto &x : vec)
    {
      if (!is_first)
      {
        os << ", ";
      }
      is_first = false;
      os << x;
    }
    os << "]";
    return os;
  }

};

std::string TesseractConfig::str()
{
  auto &config = *this;
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

bool Node::operator>(const Node& other) const {
  return cost > other.cost || (cost == other.cost && num_dets < other.num_dets);
}

std::string Node::str()
{
  std::stringstream ss;
  auto &self = *this;
  ss << "Node(";
  ss << "errs=" << self.errs << ", ";
  ss << "dets=" << self.dets << ", ";
  ss << "cost=" << self.cost << ", ";
  ss << "num_dets=" << self.num_dets << ", ";
  return ss.str();
}

std::string QNode::str() {
  auto & self = *this;
  std::stringstream ss;
  ss << "QNode(";
  ss << "cost=" << self.cost << ", ";
  ss << "num_dets=" << self.num_dets << ", ";
  ss << "errs=" << self.errs << ")";
  return ss.str();
}

double TesseractDecoder::get_detcost(size_t d, 
                                    const std::vector<DetectorCostTuple>& detector_cost_tuples) const {
  double min_cost = INF;
  ErrorCost ec;
  DetectorCostTuple dct;

  for (size_t ei : d2e[d]) {
    ec = error_costs[ei];
    dct = detector_cost_tuples[ei];
    if (ec.min_cost >= min_cost) break;
    if (!dct.error_blocked) {
      double error_cost = ec.likelihood_cost / dct.detectors_count;
      min_cost = std::min(min_cost, error_cost);
      // assert(dct.detectors_count);
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
    error_costs.push_back({errors[i].likelihood_cost, errors[i].likelihood_cost / errors[i].symptom.detectors.size()});
  }
  
  for (size_t d = 0; d < num_detectors; ++d) {
    std::sort(d2e[d].begin(), d2e[d].end(), [this](size_t idx_a, size_t idx_b) {
      return error_costs[idx_a].min_cost < error_costs[idx_b].min_cost;
    });
  }

  eneighbors.resize(num_errors);
  std::vector<std::unordered_set<size_t>> edets_sets(edets.size());
  for (size_t ei = 0; ei < edets.size(); ++ei) {
    edets_sets[ei] = std::unordered_set<size_t>(edets[ei].begin(), edets[ei].end());
  }
  for (size_t ei = 0; ei < num_errors; ++ei) {
    std::set<int> neighbor_set;
    for (int d : edets[ei]) {
      for (int oei : d2e[d]) {
        for (int od : edets[oei]) {
          if (!edets_sets[ei].contains(od)) {
            neighbor_set.insert(od);
          }
        }
      }
    }
    eneighbors[ei] = std::vector<int>(neighbor_set.begin(), neighbor_set.end());
  }
}

struct VectorCharHash {
  size_t operator()(const std::vector<char>& v) const {
    size_t seed = v.size();

    for (char b_val : v) {
      seed = seed * 31 + static_cast<size_t>(b_val);
    }
    return seed;
  }
};

void TesseractDecoder::decode_to_errors(const std::vector<uint64_t>& detections) {
  std::vector<size_t> best_errors;
  double best_cost = std::numeric_limits<double>::max();
  assert(config.det_orders.size());

  if (config.beam_climbing) {
    for (int beam = config.det_beam; beam >= 0; --beam) {
      size_t det_order = beam % config.det_orders.size();
      decode_to_errors(detections, det_order, beam);
      double this_cost = cost_from_errors(predicted_errors_buffer);
      if (!low_confidence_flag && this_cost < best_cost) {
        best_errors = predicted_errors_buffer;
        best_cost = this_cost;
      }
      if (config.verbose) {
        std::cout << "for det_order " << det_order << " beam " << beam
                  << " got low confidence " << low_confidence_flag
                  << " and cost " << this_cost << " and obs_mask "
                  << mask_from_errors(predicted_errors_buffer)
                  << ". Best cost so far: " << best_cost << std::endl;
      }
    }
  } else {
    for (size_t det_order = 0; det_order < config.det_orders.size(); ++det_order) {
      decode_to_errors(detections, det_order, config.det_beam);
      double this_cost = cost_from_errors(predicted_errors_buffer);
      if (!low_confidence_flag && this_cost < best_cost) {
        best_errors = predicted_errors_buffer;
        best_cost = this_cost;
      }
      if (config.verbose) {
        std::cout << "for det_order " << det_order << " beam "
                  << config.det_beam << " got low confidence "
                  << low_confidence_flag << " and cost " << this_cost
                  << " and obs_mask "
                  << mask_from_errors(predicted_errors_buffer)
                  << ". Best cost so far: " << best_cost << std::endl;
      }
    }
  }
  predicted_errors_buffer = best_errors;
  low_confidence_flag = best_cost == std::numeric_limits<double>::max();
}

bool QNode::operator>(const QNode& other) const {
  return cost > other.cost || (cost == other.cost && num_dets < other.num_dets);
}

void TesseractDecoder::to_node(const QNode& qnode, const std::vector<char>& shot_dets,
                              size_t det_order, Node& node, 
                              std::vector<DetectorCostTuple>& detector_cost_tuples) const {
  node.cost = qnode.cost;
  node.errs = qnode.errs;
  node.num_dets = qnode.num_dets;
  node.dets = shot_dets;

  detector_cost_tuples.resize(num_errors);

  for (size_t ei : node.errs) {
    // Get the min index activated detector before updating the dets
    size_t min_det = std::numeric_limits<size_t>::max();
    for (size_t d = 0; d < num_detectors; ++d) {
      if (node.dets[config.det_orders[det_order][d]]) {
        min_det = config.det_orders[det_order][d];
        break;
      }
    }

    // Reconstruct the blocked_errs
    for (size_t oei : d2e[min_det]) {
      detector_cost_tuples[oei].error_blocked = true;
      if (!config.at_most_two_errors_per_detector && oei == ei) break;
    }

    // Reconstruct the dets
    for (size_t d : edets[ei]) {
      node.dets[d] = !node.dets[d];
      if (!node.dets[d] && config.at_most_two_errors_per_detector) {
        for (size_t oei : d2e[d]) {
          detector_cost_tuples[oei].error_blocked = true;
        }
      }
    }
  }
}

void TesseractDecoder::decode_to_errors(const std::vector<uint64_t>& detections, size_t det_order, size_t det_beam) {
  predicted_errors_buffer.clear();
  low_confidence_flag = false;
  std::vector<char> dets(num_detectors, false);
  for (size_t d : detections) {
    dets[d] = true;
  }

  std::priority_queue<QNode, std::vector<QNode>, std::greater<QNode>> pq;
  std::unordered_map<size_t, std::unordered_set<std::vector<char>, VectorCharHash>> discovered_dets;

  size_t min_num_dets = detections.size();
  std::vector<size_t> errs;
  std::vector<DetectorCostTuple> detector_cost_tuples(num_errors);

  for (size_t d = 0; d < num_detectors; ++d) {
    if (!dets[d]) continue;
    for (int ei : d2e[d]) {
      ++detector_cost_tuples[ei].detectors_count;
    }
  }

  double initial_cost = 0.0;
  for (size_t d = 0; d < num_detectors; ++d) {
    if (!dets[d]) continue;
    initial_cost += get_detcost(d, detector_cost_tuples);
  }

  if (initial_cost == INF) {
    low_confidence_flag = true;
    return;
  }

  pq.push({initial_cost, min_num_dets, errs});

  size_t num_pq_pushed = 1;
  size_t max_num_dets = min_num_dets + det_beam;
  Node node;
  std::vector<DetectorCostTuple> next_detector_cost_tuples;
  std::vector<char> next_dets;
  std::vector<size_t> next_errs;

  while (!pq.empty()) {
    const QNode qnode = pq.top();
    if (qnode.num_dets > max_num_dets) {
      pq.pop();
      continue;
    }
    std::vector<DetectorCostTuple> detector_cost_tuples;
    to_node(qnode, dets, det_order, node, detector_cost_tuples);
    pq.pop();

    if (node.num_dets == 0) {
      if (config.verbose) {
        std::cout << "activated_errors = ";
        for (size_t oei : node.errs) {
          std::cout << oei << ", ";
        }
        std::cout << std::endl;
        std::cout << "activated_dets = ";
        for (size_t d = 0; d < num_detectors; ++d) {
          if (node.dets[d]) {
            std::cout << d << ", ";
          }
        }
        std::cout << std::endl;
        std::cout.precision(13);
        std::cout << "Decoding complete. Cost: " << node.cost
                  << " num_pq_pushed = " << num_pq_pushed << std::endl;
      }
      predicted_errors_buffer = node.errs;
      return;
    }

    if (node.num_dets > max_num_dets) continue;

    if (config.no_revisit_dets && !discovered_dets[node.num_dets].insert(node.dets).second) {
      continue;
    }

    if (config.verbose) {
      std::cout.precision(13);
      std::cout << "len(pq) = " << pq.size()
                << " num_pq_pushed = " << num_pq_pushed << std::endl;
      std::cout << "num_dets = " << node.num_dets
                << " max_num_dets = " << max_num_dets << " cost = " << node.cost
                << std::endl;
      std::cout << "activated_errors = ";
      for (size_t oei : node.errs) {
        std::cout << oei << ", ";
      }
      std::cout << std::endl;
      std::cout << "activated_dets = ";
      for (size_t d = 0; d < num_detectors; ++d) {
        if (node.dets[d]) {
          std::cout << d << ", ";
        }
      }
      std::cout << std::endl;
    }

    if (node.num_dets < min_num_dets) {
      min_num_dets = node.num_dets;
      if (config.no_revisit_dets) {
        for (size_t i = min_num_dets + det_beam + 1; i <= max_num_dets; ++i) {
          discovered_dets[i].clear();
        }
      }
      max_num_dets = std::min(max_num_dets, min_num_dets + det_beam);
    }

    // Choose the min det to be the minimum index activated detector
    size_t min_det = std::numeric_limits<size_t>::max();
    for (size_t d = 0; d < num_detectors; ++d) {
      if (node.dets[config.det_orders[det_order][d]]) {
        min_det = config.det_orders[det_order][d];
        break;
      }
    }

    for (size_t d = 0; d < num_detectors; ++d) {
      if (!node.dets[d]) continue;
      for (int ei : d2e[d]) {
        ++detector_cost_tuples[ei].detectors_count;
      }
    }

    // We cache as we recompute the det costs
    std::vector<double> det_costs(num_detectors, -1);
    next_detector_cost_tuples = detector_cost_tuples;
    if (config.at_most_two_errors_per_detector) {
      for (int ei : d2e[min_det]) {
        // Block all errors of this detector -- note this is an approximation
        // where we insist at most 2 errors are incident to any detector
        next_detector_cost_tuples[ei].error_blocked = true;
      }
    }

    // Consider activating any error of the lowest index activated detector
    size_t last_ei = std::numeric_limits<size_t>::max();
    for (size_t ei : d2e[min_det]) {
      if (detector_cost_tuples[ei].error_blocked) {
        continue;
      }

      // Uncompute the previous edits to the next det counts on the last
      // iteration
      if (last_ei != std::numeric_limits<size_t>::max()) {
        for (int d : edets[last_ei]) {
          int fired = node.dets[d] ? 1 : -1;
          for (int oei : d2e[d]) {
            next_detector_cost_tuples[oei].detectors_count += fired;
          }
        }
      }

      last_ei = ei;
      next_detector_cost_tuples[ei].error_blocked = true;

      next_errs = node.errs;
      next_errs.push_back(ei);

      next_dets = node.dets;
      double next_cost = node.cost + errors[ei].likelihood_cost;

      size_t next_num_dets = node.num_dets;
      if (config.at_most_two_errors_per_detector) {
        // next_next_blocked_errs = next_blocked_errs;
      }

      for (int d : edets[ei]) {
        next_dets[d] = !next_dets[d];
        int fired = next_dets[d] ? 1 : -1;
        next_num_dets += fired;
        for (int oei : d2e[d]) {
          next_detector_cost_tuples[oei].detectors_count += fired;
        }

        if (!next_dets[d] && config.at_most_two_errors_per_detector) {
          for (size_t oei : d2e[d]) {
            // next_next_blocked_errs[oei] = true;
          }
        }
      }

      if (next_num_dets > max_num_dets) {
        continue;
      }

      if (config.no_revisit_dets && discovered_dets[next_num_dets].find(next_dets) != discovered_dets[next_num_dets].end()) {
        continue;
      }

      for (int d : edets[ei]) {
        if (node.dets[d]) {
          if (det_costs[d] == -1) {
            det_costs[d] = get_detcost(d, detector_cost_tuples);
          }
          next_cost -= det_costs[d];
        } else {
          // next_cost += get_detcost(d, config.at_most_two_errors_per_detector ? next_next_blocked_errs : next_blocked_errs, next_det_counts);
          next_cost += get_detcost(d, next_detector_cost_tuples);
        }
      }

      for (size_t od : eneighbors[ei]) {
        if (!node.dets[od] || !next_dets[od]) continue;
        if (det_costs[od] == -1) {
          det_costs[od] = get_detcost(od, detector_cost_tuples);
        }
        next_cost -= det_costs[od];
        // next_cost += get_detcost(od, config.at_most_two_errors_per_detector ? next_next_blocked_errs : next_blocked_errs, next_det_counts);
        next_cost += get_detcost(od, next_detector_cost_tuples);
      }

      if (next_cost == INF) {
        continue;
      }

      pq.push({next_cost, next_num_dets, next_errs});
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

common::ObservablesMask TesseractDecoder::mask_from_errors(const std::vector<size_t>& predicted_errors) {
  common::ObservablesMask mask = 0;
  // Iterate over all errors and compute the mask
  for (size_t ei : predicted_errors) {
    mask ^= errors[ei].symptom.observables;
  }
  return mask;
}

common::ObservablesMask TesseractDecoder::decode(const std::vector<uint64_t>& detections) {
  decode_to_errors(detections);
  return mask_from_errors(predicted_errors_buffer);
}

void TesseractDecoder::decode_shots(
    std::vector<stim::SparseShot>& shots,
    std::vector<common::ObservablesMask>& obs_predicted) {
  obs_predicted.resize(shots.size());
  for (size_t i = 0; i < shots.size(); ++i) {
    obs_predicted[i] = decode(shots[i].hits);
  }
}