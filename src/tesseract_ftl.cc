
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

#include "tesseract_ftl.h"

#include <algorithm>
#include <boost/functional/hash.hpp>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace {

constexpr double INF_D = std::numeric_limits<double>::infinity();
constexpr double HEURISTIC_EPS = 1e-9;
constexpr double SIMPLEX_EPS = 1e-9;
constexpr double SEED_TIGHT_EPS = 1e-9;
constexpr double VIOLATION_EPS = 1e-9;
constexpr size_t VIOLATION_BATCH_SIZE = 4;

struct UnionFind {
  std::vector<int> parent;
  std::vector<int> rank;

  explicit UnionFind(size_t n) : parent(n), rank(n, 0) {
    std::iota(parent.begin(), parent.end(), 0);
  }

  int find(int x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }

  void unite(int a, int b) {
    a = find(a);
    b = find(b);
    if (a == b) return;
    if (rank[a] < rank[b]) {
      parent[a] = b;
    } else if (rank[a] > rank[b]) {
      parent[b] = a;
    } else {
      parent[b] = a;
      rank[a]++;
    }
  }
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
  os << "[";
  bool is_first = true;
  for (const auto& x : vec) {
    if (!is_first) os << ", ";
    is_first = false;
    os << x;
  }
  os << "]";
  return os;
}

struct IntVectorHash {
  size_t operator()(const std::vector<int>& values) const {
    return boost::hash_range(values.begin(), values.end());
  }
};

struct DenseSimplexResult {
  bool success = false;
  bool unbounded = false;
  double objective = 0.0;
  size_t pivots = 0;
  std::vector<double> solution;
};

double dot_on_support(const std::vector<double>& values, const std::vector<int>& support) {
  double total = 0.0;
  for (int idx : support) total += values[(size_t)idx];
  return total;
}

// Solves:
//   maximize   sum_i x_i
//   subject to A x <= b
//              x >= 0
// where A is a 0/1 matrix given by row supports for a selected subset of rows.
DenseSimplexResult solve_dense_primal_packing_lp(
    size_t num_vars,
    const std::vector<TesseractFTLDecoder::SingletonPatternConstraint>& constraints,
    const std::vector<int>& selected_rows,
    const std::vector<double>* entering_priorities = nullptr) {
  DenseSimplexResult result;
  result.solution.assign(num_vars, 0.0);

  const size_t num_rows = selected_rows.size();
  if (num_vars == 0) {
    result.success = true;
    return result;
  }
  if (num_rows == 0) {
    result.unbounded = true;
    return result;
  }

  const size_t width = num_vars + num_rows + 1;
  const size_t height = num_rows + 1;
  std::vector<double> tableau(height * width, 0.0);
  std::vector<size_t> basis(num_rows);

  for (size_t row = 0; row < num_rows; ++row) {
    size_t orig_row = (size_t)selected_rows[row];
    for (int col : constraints[orig_row].local_detectors) {
      tableau[row * width + (size_t)col] = 1.0;
    }
    tableau[row * width + num_vars + row] = 1.0;
    tableau[row * width + width - 1] = constraints[orig_row].rhs;
    basis[row] = num_vars + row;
    if (constraints[orig_row].rhs < -SIMPLEX_EPS) {
      throw std::runtime_error("Dense simplex received a negative RHS.");
    }
  }
  for (size_t col = 0; col < num_vars; ++col) {
    tableau[num_rows * width + col] = -1.0;
  }

  auto pivot = [&](size_t pivot_row, size_t pivot_col) {
    const double pivot_value = tableau[pivot_row * width + pivot_col];
    assert(std::abs(pivot_value) > SIMPLEX_EPS);
    const double inv_pivot = 1.0 / pivot_value;
    for (size_t col = 0; col < width; ++col) {
      tableau[pivot_row * width + col] *= inv_pivot;
    }
    tableau[pivot_row * width + pivot_col] = 1.0;

    for (size_t row = 0; row < height; ++row) {
      if (row == pivot_row) continue;
      const double factor = tableau[row * width + pivot_col];
      if (std::abs(factor) <= SIMPLEX_EPS) {
        tableau[row * width + pivot_col] = 0.0;
        continue;
      }
      for (size_t col = 0; col < width; ++col) {
        tableau[row * width + col] -= factor * tableau[pivot_row * width + col];
      }
      tableau[row * width + pivot_col] = 0.0;
    }
    basis[pivot_row] = pivot_col;
    result.pivots++;
  };

  while (true) {
    size_t entering_col = width;
    double entering_priority = -INF_D;
    for (size_t col = 0; col + 1 < width; ++col) {
      if (tableau[num_rows * width + col] >= -SIMPLEX_EPS) continue;
      const bool current_is_original = entering_col < num_vars;
      const bool candidate_is_original = col < num_vars;
      const double candidate_priority = candidate_is_original && entering_priorities != nullptr
                                            ? (*entering_priorities)[col]
                                            : -INF_D;
      if (entering_col == width) {
        entering_col = col;
        entering_priority = candidate_priority;
        continue;
      }
      if (candidate_is_original != current_is_original) {
        if (candidate_is_original) {
          entering_col = col;
          entering_priority = candidate_priority;
        }
        continue;
      }
      if (candidate_priority > entering_priority + SIMPLEX_EPS ||
          (std::abs(candidate_priority - entering_priority) <= SIMPLEX_EPS && col < entering_col)) {
        entering_col = col;
        entering_priority = candidate_priority;
      }
    }
    if (entering_col == width) {
      break;
    }

    size_t leaving_row = num_rows;
    double best_ratio = INF_D;
    for (size_t row = 0; row < num_rows; ++row) {
      const double coeff = tableau[row * width + entering_col];
      if (coeff <= SIMPLEX_EPS) continue;
      const double ratio = tableau[row * width + width - 1] / coeff;
      if (ratio + SIMPLEX_EPS < best_ratio) {
        best_ratio = ratio;
        leaving_row = row;
      } else if (std::abs(ratio - best_ratio) <= SIMPLEX_EPS && leaving_row != num_rows &&
                 basis[row] < basis[leaving_row]) {
        leaving_row = row;
      }
    }

    if (leaving_row == num_rows) {
      result.unbounded = true;
      return result;
    }
    pivot(leaving_row, entering_col);
  }

  for (size_t row = 0; row < num_rows; ++row) {
    if (basis[row] < num_vars) {
      double value = tableau[row * width + width - 1];
      if (std::abs(value) <= SIMPLEX_EPS) value = 0.0;
      result.solution[basis[row]] = value;
    }
  }
  result.objective = tableau[num_rows * width + width - 1];
  if (std::abs(result.objective) <= SIMPLEX_EPS) result.objective = 0.0;
  result.success = true;
  return result;
}

template <typename Solution>
double lookup_detector_budget(const Solution& solution, int detector) {
  auto it = std::lower_bound(solution.active_detectors.begin(), solution.active_detectors.end(),
                             detector);
  if (it == solution.active_detectors.end() || *it != detector) return 0.0;
  const size_t pos = (size_t)(it - solution.active_detectors.begin());
  return solution.detector_budgets[pos];
}

struct SingletonComponentSolveResult {
  bool success = false;
  bool unbounded = false;
  double objective = 0.0;
  size_t reduced_constraints = 0;
  size_t simplex_solves = 0;
  std::vector<double> detector_budgets;
};

SingletonComponentSolveResult solve_singleton_component_lp(
    size_t num_local_detectors,
    const std::vector<TesseractFTLDecoder::SingletonPatternConstraint>& constraints,
    const std::vector<int>& cheapest_constraint_for_local_detector,
    const std::vector<double>& seed_budgets) {
  SingletonComponentSolveResult result;
  result.detector_budgets.assign(num_local_detectors, 0.0);

  if (num_local_detectors == 0) {
    result.success = true;
    return result;
  }
  if (constraints.empty()) {
    result.unbounded = true;
    return result;
  }

  const double seed_total = std::accumulate(seed_budgets.begin(), seed_budgets.end(), 0.0);

  std::vector<uint8_t> selected(constraints.size(), 0);
  std::vector<int> selected_indices;
  selected_indices.reserve(std::min(constraints.size(), num_local_detectors * 2 + 4));

  auto add_constraint = [&](int idx) {
    if (idx < 0) return;
    if (!selected[(size_t)idx]) {
      selected[(size_t)idx] = 1;
      selected_indices.push_back(idx);
    }
  };

  for (size_t row = 0; row < constraints.size(); ++row) {
    const auto& constraint = constraints[row];
    const double slack = constraint.rhs - dot_on_support(seed_budgets, constraint.local_detectors);
    if (slack <= SEED_TIGHT_EPS * (1.0 + constraint.rhs)) {
      add_constraint((int)row);
    }
  }

  std::vector<uint8_t> covered(num_local_detectors, 0);
  for (int idx : selected_indices) {
    for (int local : constraints[(size_t)idx].local_detectors) covered[(size_t)local] = 1;
  }
  for (size_t local = 0; local < num_local_detectors; ++local) {
    if (!covered[local]) {
      const int idx = cheapest_constraint_for_local_detector[local];
      if (idx < 0) {
        throw std::runtime_error("Missing seed constraint for active detector.");
      }
      add_constraint(idx);
      for (int touched : constraints[(size_t)idx].local_detectors) covered[(size_t)touched] = 1;
    }
  }

  if (selected_indices.empty()) {
    throw std::runtime_error("Singleton LP seed set unexpectedly empty.");
  }

  size_t rounds = 0;
  while (true) {
    if (++rounds > constraints.size() + 1) {
      throw std::runtime_error("Constraint generation exceeded the number of unique constraints.");
    }

    DenseSimplexResult simplex =
        solve_dense_primal_packing_lp(num_local_detectors, constraints, selected_indices,
                                      &seed_budgets);
    result.simplex_solves++;
    if (simplex.unbounded) {
      result.unbounded = true;
      return result;
    }
    if (!simplex.success) {
      return result;
    }
    if (simplex.objective + 1e-7 < seed_total) {
      throw std::runtime_error("Reduced singleton LP optimum fell below the projected seed bound.");
    }

    double max_violation = 0.0;
    std::vector<std::pair<double, int>> top_violated;
    top_violated.reserve(VIOLATION_BATCH_SIZE);

    for (size_t row = 0; row < constraints.size(); ++row) {
      if (selected[row]) continue;
      const auto& constraint = constraints[row];
      const double lhs = dot_on_support(simplex.solution, constraint.local_detectors);
      const double violation = lhs - constraint.rhs;
      if (violation > max_violation) {
        max_violation = violation;
      }
      if (violation <= VIOLATION_EPS * (1.0 + constraint.rhs)) continue;

      top_violated.emplace_back(violation, (int)row);
      std::sort(top_violated.begin(), top_violated.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });
      if (top_violated.size() > VIOLATION_BATCH_SIZE) top_violated.pop_back();
    }

    if (max_violation <= VIOLATION_EPS) {
      result.success = true;
      result.objective = simplex.objective;
      result.reduced_constraints = selected_indices.size();
      result.detector_budgets = std::move(simplex.solution);
      return result;
    }

    bool added_any = false;
    for (const auto& [_, idx] : top_violated) {
      if (!selected[(size_t)idx]) {
        add_constraint(idx);
        added_any = true;
      }
    }
    if (!added_any) {
      throw std::runtime_error("Constraint generation identified violations but added no rows.");
    }
  }
}

std::string heuristic_source_to_string(FTLHeuristicSource source) {
  switch (source) {
    case FTLHeuristicSource::kPlain:
      return "plain";
    case FTLHeuristicSource::kProjected:
      return "projected";
    case FTLHeuristicSource::kExact:
      return "exact";
  }
  return "unknown";
}

}  // namespace

std::string TesseractFTLConfig::str() {
  std::stringstream ss;
  ss << "TesseractFTLConfig(";
  ss << "dem=DetectorErrorModel_Object, ";
  ss << "det_beam=" << det_beam << ", ";
  ss << "no_revisit_dets=" << no_revisit_dets << ", ";
  ss << "verbose=" << verbose << ", ";
  ss << "merge_errors=" << merge_errors << ", ";
  ss << "pqlimit=" << pqlimit << ", ";
  ss << "det_orders=" << det_orders << ", ";
  ss << "det_penalty=" << det_penalty << ", ";
  ss << "create_visualization=" << create_visualization << ", ";
  ss << "subset_detcost_size=" << subset_detcost_size;
  ss << ")";
  return ss.str();
}

void TesseractFTLStats::clear() {
  *this = TesseractFTLStats{};
}

void TesseractFTLStats::accumulate(const TesseractFTLStats& other) {
  num_pq_pushed += other.num_pq_pushed;
  num_nodes_popped += other.num_nodes_popped;
  max_queue_size = std::max(max_queue_size, other.max_queue_size);
  heuristic_calls += other.heuristic_calls;
  plain_heuristic_calls += other.plain_heuristic_calls;
  projection_heuristic_calls += other.projection_heuristic_calls;
  exact_refinement_calls += other.exact_refinement_calls;
  lp_calls += other.lp_calls;
  lp_reinserts += other.lp_reinserts;
  projected_nodes_generated += other.projected_nodes_generated;
  projected_nodes_refined += other.projected_nodes_refined;
  total_lp_refinement_gain += other.total_lp_refinement_gain;
  max_lp_refinement_gain = std::max(max_lp_refinement_gain, other.max_lp_refinement_gain);
  lp_total_seconds += other.lp_total_seconds;
}

bool TesseractFTLDecoder::FTLNode::operator>(const FTLNode& other) const {
  return f_cost > other.f_cost || (f_cost == other.f_cost && num_dets < other.num_dets);
}

size_t TesseractFTLDecoder::DynamicBitsetHash::operator()(const boost::dynamic_bitset<>& bs) const {
  return boost::hash_value(bs);
}

TesseractFTLDecoder::TesseractFTLDecoder(TesseractFTLConfig config_) : config(config_) {
  if (config.subset_detcost_size > 1) {
    throw std::invalid_argument(
        "tesseract_ftl singleton mode supports only subset_detcost_size of 0 or 1");
  }

  if (config.subset_detcost_size == 0) {
    TesseractConfig delegate_config;
    delegate_config.dem = config.dem;
    delegate_config.det_beam = config.det_beam;
    delegate_config.beam_climbing = config.beam_climbing;
    delegate_config.no_revisit_dets = config.no_revisit_dets;
    delegate_config.verbose = config.verbose;
    delegate_config.merge_errors = config.merge_errors;
    delegate_config.pqlimit = config.pqlimit;
    delegate_config.det_orders = config.det_orders;
    delegate_config.det_penalty = config.det_penalty;
    delegate_config.create_visualization = config.create_visualization;
    plain_delegate = std::make_unique<TesseractDecoder>(delegate_config);
    errors = plain_delegate->errors;
    num_detectors = plain_delegate->num_detectors;
    num_observables = plain_delegate->num_observables;
    dem_error_to_error = plain_delegate->dem_error_to_error;
    error_to_dem_error = plain_delegate->error_to_dem_error;
    return;
  }

  std::vector<size_t> dem_error_map(config.dem.flattened().count_errors());
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

  if (config.det_orders.empty()) {
    config.det_orders.emplace_back(config.dem.count_detectors());
    std::iota(config.det_orders[0].begin(), config.det_orders[0].end(), 0);
  } else {
    for (const auto& order : config.det_orders) {
      if (order.size() != config.dem.count_detectors()) {
        throw std::invalid_argument(
            "Each detector order list must have a size equal to the number of detectors.");
      }
    }
  }
  if (config.det_orders.empty()) {
    throw std::runtime_error("Detector order list must not be empty.");
  }

  errors = get_errors_from_dem(config.dem.flattened());
  num_detectors = config.dem.count_detectors();
  num_errors = config.dem.count_errors();
  num_observables = config.dem.count_observables();

  initialize_structures(num_detectors);

  if (config.create_visualization) {
    auto detectors = get_detector_coords(config.dem);
    visualizer.add_detector_coords(detectors);
    visualizer.add_errors(errors);
  }
}

TesseractFTLDecoder::~TesseractFTLDecoder() = default;

void TesseractFTLDecoder::initialize_structures(size_t num_detectors_) {
  d2e.resize(num_detectors_);
  edets.resize(num_errors);
  error_costs.resize(num_errors);

  for (size_t ei = 0; ei < num_errors; ++ei) {
    edets[ei] = errors[ei].symptom.detectors;
    for (int d : edets[ei]) {
      d2e[(size_t)d].push_back((int)ei);
    }
    error_costs[ei] = {errors[ei].likelihood_cost,
                       errors[ei].likelihood_cost / errors[ei].symptom.detectors.size()};
  }

  for (size_t d = 0; d < num_detectors_; ++d) {
    std::sort(d2e[d].begin(), d2e[d].end(), [this](int a, int b) {
      return error_costs[(size_t)a].min_cost < error_costs[(size_t)b].min_cost;
    });
  }
}

void TesseractFTLDecoder::flip_detectors_and_block_errors(
    size_t detector_order, int64_t error_chain_idx, boost::dynamic_bitset<>& detectors,
    std::vector<uint8_t>& blocked_flags) const {
  (void)detector_order;
  int64_t walker_idx = error_chain_idx;
  while (walker_idx != -1) {
    const auto& node = error_chain_arena[(size_t)walker_idx];
    const size_t ei = node.error_index;
    const size_t min_detector = node.min_detector;

    for (int oei : d2e[min_detector]) {
      blocked_flags[(size_t)oei] = 1;
      if ((size_t)oei == ei) break;
    }
    for (int d : edets[ei]) detectors[(size_t)d] = !detectors[(size_t)d];
    walker_idx = node.parent_idx;
  }
}

TesseractFTLDecoder::SingletonBuildResult TesseractFTLDecoder::build_singleton_components(
    const boost::dynamic_bitset<>& detectors, const std::vector<uint8_t>& blocked_flags) const {
  SingletonBuildResult result;

  std::vector<int> active_detectors;
  active_detectors.reserve(detectors.count());
  std::vector<int> detector_to_active_pos(num_detectors, -1);
  for (size_t detector = detectors.find_first(); detector != boost::dynamic_bitset<>::npos;
       detector = detectors.find_next(detector)) {
    detector_to_active_pos[detector] = (int)active_detectors.size();
    active_detectors.push_back((int)detector);
  }
  if (active_detectors.empty()) return result;

  UnionFind uf(active_detectors.size());
  std::vector<uint8_t> has_available(active_detectors.size(), 0);

  for (size_t ei = 0; ei < num_errors; ++ei) {
    if (blocked_flags[ei]) continue;
    int first_active = -1;
    for (int detector : edets[ei]) {
      const int active_pos = detector_to_active_pos[(size_t)detector];
      if (active_pos < 0) continue;
      has_available[(size_t)active_pos] = 1;
      if (first_active < 0) {
        first_active = active_pos;
      } else {
        uf.unite(first_active, active_pos);
      }
    }
  }

  for (size_t active_pos = 0; active_pos < active_detectors.size(); ++active_pos) {
    if (!has_available[active_pos]) {
      result.feasible = false;
      return result;
    }
  }

  std::vector<std::vector<int>> positions_by_root(active_detectors.size());
  for (int active_pos = 0; active_pos < (int)active_detectors.size(); ++active_pos) {
    positions_by_root[(size_t)uf.find(active_pos)].push_back(active_pos);
  }

  std::vector<std::vector<int>> component_positions;
  component_positions.reserve(active_detectors.size());
  for (auto& positions : positions_by_root) {
    if (positions.empty()) continue;
    std::sort(positions.begin(), positions.end(), [&](int a, int b) {
      return active_detectors[(size_t)a] < active_detectors[(size_t)b];
    });
    component_positions.push_back(std::move(positions));
  }
  std::sort(component_positions.begin(), component_positions.end(),
            [&](const auto& a, const auto& b) {
              return active_detectors[(size_t)a[0]] < active_detectors[(size_t)b[0]];
            });

  std::vector<int> active_pos_to_component(active_detectors.size(), -1);
  std::vector<int> active_pos_to_local(active_detectors.size(), -1);

  result.components.reserve(component_positions.size());
  for (size_t component_index = 0; component_index < component_positions.size();
       ++component_index) {
    const auto& positions = component_positions[component_index];
    SingletonLPComponent component;
    component.detectors.reserve(positions.size());
    for (size_t local = 0; local < positions.size(); ++local) {
      const int active_pos = positions[local];
      active_pos_to_component[(size_t)active_pos] = (int)component_index;
      active_pos_to_local[(size_t)active_pos] = (int)local;
      component.detectors.push_back(active_detectors[(size_t)active_pos]);
    }
    result.components.push_back(std::move(component));
  }

  std::vector<std::unordered_map<std::vector<int>, double, IntVectorHash>> min_rhs_by_pattern(
      result.components.size());
  std::vector<int> local_hits;
  local_hits.reserve(16);

  for (size_t ei = 0; ei < num_errors; ++ei) {
    if (blocked_flags[ei]) continue;

    int component_index = -1;
    local_hits.clear();

    for (int detector : edets[ei]) {
      const int active_pos = detector_to_active_pos[(size_t)detector];
      if (active_pos < 0) continue;
      if (component_index < 0) {
        component_index = active_pos_to_component[(size_t)active_pos];
      } else {
        assert(component_index == active_pos_to_component[(size_t)active_pos]);
      }
      local_hits.push_back(active_pos_to_local[(size_t)active_pos]);
    }

    if (component_index < 0) continue;

    auto& rhs_map = min_rhs_by_pattern[(size_t)component_index];
    const double rhs = errors[ei].likelihood_cost;
    auto it = rhs_map.find(local_hits);
    if (it == rhs_map.end() || rhs < it->second) {
      rhs_map[local_hits] = rhs;
    }
  }

  for (size_t component_index = 0; component_index < result.components.size(); ++component_index) {
    auto& component = result.components[component_index];
    const auto& rhs_map = min_rhs_by_pattern[component_index];

    component.constraints.reserve(rhs_map.size());
    for (const auto& [local_hits, rhs] : rhs_map) {
      component.constraints.push_back({local_hits, rhs});
    }
    std::sort(component.constraints.begin(), component.constraints.end(),
              [](const auto& a, const auto& b) {
                if (a.local_detectors.size() != b.local_detectors.size()) {
                  return a.local_detectors.size() < b.local_detectors.size();
                }
                if (a.local_detectors != b.local_detectors) {
                  return a.local_detectors < b.local_detectors;
                }
                return a.rhs < b.rhs;
              });

    component.cheapest_constraint_for_local_detector.assign(component.detectors.size(), -1);
    std::vector<double> cheapest_rhs(component.detectors.size(), INF_D);
    for (size_t constraint_index = 0; constraint_index < component.constraints.size();
         ++constraint_index) {
      const auto& constraint = component.constraints[constraint_index];
      for (int local_detector : constraint.local_detectors) {
        if (constraint.rhs < cheapest_rhs[(size_t)local_detector]) {
          cheapest_rhs[(size_t)local_detector] = constraint.rhs;
          component.cheapest_constraint_for_local_detector[(size_t)local_detector] =
              (int)constraint_index;
        }
      }
    }
    for (size_t local = 0; local < component.detectors.size(); ++local) {
      if (component.cheapest_constraint_for_local_detector[local] < 0) {
        result.feasible = false;
        result.components.clear();
        return result;
      }
    }
  }

  return result;
}

TesseractFTLDecoder::ExactSubsetSolution TesseractFTLDecoder::solve_exact_subset_lp(
    const boost::dynamic_bitset<>& detectors, const std::vector<uint8_t>& blocked_flags,
    int64_t warm_solution_idx) {
  stats.heuristic_calls++;
  stats.exact_refinement_calls++;
  const auto start_time = std::chrono::high_resolution_clock::now();

  ExactSubsetSolution solution;
  const auto build = build_singleton_components(detectors, blocked_flags);
  if (!build.feasible) {
    solution.value = INF_D;
    const auto stop_time = std::chrono::high_resolution_clock::now();
    stats.lp_total_seconds +=
        std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count() / 1e6;
    return solution;
  }
  if (build.components.empty()) {
    solution.value = 0.0;
    const auto stop_time = std::chrono::high_resolution_clock::now();
    stats.lp_total_seconds +=
        std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count() / 1e6;
    return solution;
  }

  const ExactSubsetSolution* warm_solution =
      warm_solution_idx >= 0 ? &exact_solution_arena[(size_t)warm_solution_idx] : nullptr;
  solution.value = 0.0;
  solution.num_components = build.components.size();
  std::vector<std::pair<int, double>> detector_budget_pairs;
  detector_budget_pairs.reserve(detectors.count());
  size_t warm_pos = 0;

  for (const auto& component : build.components) {
    std::vector<double> seed_budgets(component.detectors.size(), 0.0);
    if (warm_solution != nullptr) {
      for (size_t local = 0; local < component.detectors.size(); ++local) {
        int det = component.detectors[local];
        while (warm_pos < warm_solution->active_detectors.size() &&
               warm_solution->active_detectors[warm_pos] < det) {
          ++warm_pos;
        }
        if (warm_pos < warm_solution->active_detectors.size() &&
            warm_solution->active_detectors[warm_pos] == det) {
          seed_budgets[local] = warm_solution->detector_budgets[warm_pos];
        }
      }
    }
    const auto component_result = solve_singleton_component_lp(
        component.detectors.size(), component.constraints,
        component.cheapest_constraint_for_local_detector, seed_budgets);
    stats.lp_calls += component_result.simplex_solves;

    if (component_result.unbounded) {
      throw std::runtime_error("Singleton custom LP became unbounded.");
    }
    if (!component_result.success) {
      throw std::runtime_error("Singleton custom LP failed.");
    }

    solution.value += component_result.objective;
    solution.num_active_subsets += component.detectors.size();
    solution.num_variables += component.detectors.size();
    solution.num_constraints += component_result.reduced_constraints;

    for (size_t local = 0; local < component.detectors.size(); ++local) {
      detector_budget_pairs.emplace_back(component.detectors[local],
                                         component_result.detector_budgets[local]);
    }
  }

  if (detector_budget_pairs.size() > 1) {
    std::sort(detector_budget_pairs.begin(), detector_budget_pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
  }
  solution.active_detectors.reserve(detector_budget_pairs.size());
  solution.detector_budgets.reserve(detector_budget_pairs.size());
  for (const auto& [detector, budget] : detector_budget_pairs) {
    solution.active_detectors.push_back(detector);
    solution.detector_budgets.push_back(budget);
  }

  const auto stop_time = std::chrono::high_resolution_clock::now();
  stats.lp_total_seconds +=
      std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count() / 1e6;
  return solution;
}

double TesseractFTLDecoder::project_from_exact_solution(const ExactSubsetSolution& solution,
                                                        const boost::dynamic_bitset<>& detectors,
                                                        const std::vector<uint8_t>& blocked_flags) {
  stats.heuristic_calls++;
  stats.projection_heuristic_calls++;

  double total = 0.0;
  size_t budget_pos = 0;
  for (size_t detector = detectors.find_first(); detector != boost::dynamic_bitset<>::npos;
       detector = detectors.find_next(detector)) {
    bool has_available = false;
    for (int ei : d2e[detector]) {
      if (!blocked_flags[(size_t)ei]) {
        has_available = true;
        break;
      }
    }
    if (!has_available) return INF_D;

    while (budget_pos < solution.active_detectors.size() &&
           solution.active_detectors[budget_pos] < (int)detector) {
      ++budget_pos;
    }
    if (budget_pos < solution.active_detectors.size() &&
        solution.active_detectors[budget_pos] == (int)detector) {
      total += solution.detector_budgets[budget_pos];
    }
  }
  return total;
}

void TesseractFTLDecoder::reset_decode_state() {
  low_confidence_flag = false;
  predicted_errors_buffer.clear();
  error_chain_arena.clear();
  exact_solution_arena.clear();
  stats.clear();
}

void TesseractFTLDecoder::decode_to_errors(const std::vector<uint64_t>& detections) {
  if (config.verbose) {
    std::cout << "shot";
    for (const uint64_t& d : detections) {
      std::cout << " D" << d;
    }
    std::cout << std::endl;
  }
  if (plain_delegate) {
    plain_delegate->decode_to_errors(detections);
    predicted_errors_buffer = plain_delegate->predicted_errors_buffer;
    low_confidence_flag = plain_delegate->low_confidence_flag;
    stats.clear();
    return;
  }

  std::vector<size_t> best_errors;
  double best_cost = std::numeric_limits<double>::max();
  bool any_success = false;
  TesseractFTLStats aggregate_stats;
  stats.clear();

  if (config.beam_climbing) {
    int beam = 0;
    int detector_order = 0;
    for (int trial = 0; trial < std::max(config.det_beam + 1, int(config.det_orders.size()));
         ++trial) {
      decode_to_errors(detections, (size_t)detector_order, (size_t)beam);
      aggregate_stats.accumulate(stats);
      const double local_cost = cost_from_errors(predicted_errors_buffer);
      if (!low_confidence_flag && local_cost < best_cost) {
        best_errors = predicted_errors_buffer;
        best_cost = local_cost;
        any_success = true;
      }
      if (config.verbose) {
        std::cout << "for detector_order " << detector_order << " beam " << beam
                  << " got low confidence " << low_confidence_flag << " and cost " << local_cost
                  << ". Best cost so far: " << best_cost << std::endl;
      }
      beam = (beam + 1) % (config.det_beam + 1);
      detector_order = (detector_order + 1) % config.det_orders.size();
    }
  } else {
    for (size_t detector_order = 0; detector_order < config.det_orders.size(); ++detector_order) {
      decode_to_errors(detections, detector_order, config.det_beam);
      aggregate_stats.accumulate(stats);
      const double local_cost = cost_from_errors(predicted_errors_buffer);
      if (!low_confidence_flag && local_cost < best_cost) {
        best_errors = predicted_errors_buffer;
        best_cost = local_cost;
        any_success = true;
      }
      if (config.verbose) {
        std::cout << "for detector_order " << detector_order << " beam " << config.det_beam
                  << " got low confidence " << low_confidence_flag << " and cost " << local_cost
                  << ". Best cost so far: " << best_cost << std::endl;
      }
    }
  }
  predicted_errors_buffer = best_errors;
  low_confidence_flag = !any_success;
  stats = aggregate_stats;
}

void TesseractFTLDecoder::decode_to_errors(const std::vector<uint64_t>& detections,
                                           size_t detector_order, size_t detector_beam) {
  if (plain_delegate) {
    plain_delegate->decode_to_errors(detections, detector_order, detector_beam);
    predicted_errors_buffer = plain_delegate->predicted_errors_buffer;
    low_confidence_flag = plain_delegate->low_confidence_flag;
    return;
  }

  reset_decode_state();
  if (config.pqlimit != std::numeric_limits<size_t>::max()) {
    const size_t reserve_size = std::min<size_t>(config.pqlimit, 5000000);
    error_chain_arena.reserve(reserve_size);
    exact_solution_arena.reserve(reserve_size / 4 + 1);
  }

  std::priority_queue<FTLNode, std::vector<FTLNode>, std::greater<FTLNode>> pq;
  std::vector<std::unordered_set<boost::dynamic_bitset<>, DynamicBitsetHash>> visited_detectors(
      num_detectors + 1);

  boost::dynamic_bitset<> initial_detectors(num_detectors, false);
  std::vector<uint8_t> initial_blocked_flags(num_errors, 0);
  for (size_t detector : detections) {
    if (detector >= num_detectors) {
      throw std::runtime_error("Symptom references detector >= num_detectors");
    }
    initial_detectors[detector] = true;
  }

  size_t min_num_dets = detections.size();
  size_t max_num_dets =
      detector_beam > num_detectors - min_num_dets ? num_detectors : min_num_dets + detector_beam;

  FTLNode root;
  root.g_cost = 0.0;
  root.num_dets = min_num_dets;
  root.depth = 0;
  root.error_chain_idx = -1;
  root.warm_solution_idx = -1;
  root.exact_solution_idx = -1;

  ExactSubsetSolution root_exact =
      solve_exact_subset_lp(initial_detectors, initial_blocked_flags, -1);
  if (root_exact.value == INF_D) {
    low_confidence_flag = true;
    return;
  }
  exact_solution_arena.push_back(std::move(root_exact));
  root.exact_solution_idx = (int64_t)exact_solution_arena.size() - 1;
  root.f_cost = exact_solution_arena.back().value;
  root.h_cost = exact_solution_arena.back().value;
  root.exact_refined = true;
  root.heuristic_source = FTLHeuristicSource::kExact;
  pq.push(root);
  stats.num_pq_pushed = 1;
  stats.max_queue_size = 1;

  while (!pq.empty()) {
    stats.max_queue_size = std::max(stats.max_queue_size, pq.size());
    FTLNode node = pq.top();
    pq.pop();
    stats.num_nodes_popped++;

    if (node.num_dets > max_num_dets) continue;

    boost::dynamic_bitset<> detectors = initial_detectors;
    std::vector<uint8_t> blocked_flags(num_errors, 0);
    flip_detectors_and_block_errors(detector_order, node.error_chain_idx, detectors, blocked_flags);

    if (config.verbose) {
      const size_t projected_unrefined =
          stats.projected_nodes_generated - stats.projected_nodes_refined;
      std::cout.precision(13);
      std::cout << "nodes_popped=" << stats.num_nodes_popped << " len(pq)=" << pq.size()
                << " nodes_pushed=" << stats.num_pq_pushed << " lp_calls=" << stats.lp_calls
                << " lp_reinserts=" << stats.lp_reinserts
                << " proj_generated=" << stats.projected_nodes_generated
                << " proj_refined=" << stats.projected_nodes_refined
                << " proj_unrefined_so_far=" << projected_unrefined << " num_dets=" << node.num_dets
                << " max_num_dets=" << max_num_dets << " f=" << node.f_cost << " g=" << node.g_cost
                << " h=" << node.h_cost
                << " h_source=" << heuristic_source_to_string(node.heuristic_source)
                << " exact_refined=" << node.exact_refined << std::endl;
    }

    if (node.num_dets == 0) {
      predicted_errors_buffer.resize(node.depth);
      int64_t walker_idx = node.error_chain_idx;
      for (size_t i = 0; i < node.depth; ++i) {
        predicted_errors_buffer[node.depth - 1 - i] =
            error_to_dem_error[error_chain_arena[(size_t)walker_idx].error_index];
        walker_idx = error_chain_arena[(size_t)walker_idx].parent_idx;
      }
      if (config.verbose) {
        std::cout << "Decoding complete. Cost: " << node.g_cost
                  << " num_pq_pushed = " << stats.num_pq_pushed << std::endl;
      }
      return;
    }

    if (node.num_dets < min_num_dets) {
      min_num_dets = node.num_dets;
      const size_t next_max_num_dets = detector_beam > num_detectors - min_num_dets
                                           ? num_detectors
                                           : min_num_dets + detector_beam;
      if (config.no_revisit_dets) {
        for (size_t count = next_max_num_dets + 1; count <= max_num_dets; ++count) {
          visited_detectors[count].clear();
        }
      }
      max_num_dets = std::min(max_num_dets, next_max_num_dets);
    }

    if (!node.exact_refined) {
      const double prev_h = node.h_cost;
      const FTLHeuristicSource prev_source = node.heuristic_source;
      ExactSubsetSolution exact_solution =
          solve_exact_subset_lp(detectors, blocked_flags, node.warm_solution_idx);
      if (prev_source == FTLHeuristicSource::kProjected) stats.projected_nodes_refined++;
      if (exact_solution.value == INF_D) {
        if (config.verbose) {
          std::cout << "  lp_refine exact_h=INF discarded=true" << std::endl;
        }
        continue;
      }
      if (exact_solution.value + 1e-7 < prev_h) {
        throw std::runtime_error("Exact singleton lower bound fell below stored lower bound.");
      }
      const double delta = exact_solution.value - prev_h;
      stats.total_lp_refinement_gain += delta;
      stats.max_lp_refinement_gain = std::max(stats.max_lp_refinement_gain, delta);
      exact_solution_arena.push_back(std::move(exact_solution));
      node.exact_solution_idx = (int64_t)exact_solution_arena.size() - 1;
      node.h_cost = exact_solution_arena.back().value;
      node.f_cost = node.g_cost + node.h_cost;
      node.exact_refined = true;
      node.heuristic_source = FTLHeuristicSource::kExact;
      if (config.verbose) {
        std::cout << "  lp_refine approx_h=" << prev_h << " exact_h=" << node.h_cost
                  << " delta=" << delta << " vars=" << exact_solution_arena.back().num_variables
                  << " constraints=" << exact_solution_arena.back().num_constraints
                  << " reinserted=" << (delta > HEURISTIC_EPS) << std::endl;
      }
      if (delta > HEURISTIC_EPS) {
        stats.lp_reinserts++;
        pq.push(node);
        stats.num_pq_pushed++;
        if (stats.num_pq_pushed > config.pqlimit) {
          low_confidence_flag = true;
          return;
        }
        continue;
      }
    }

    if (config.no_revisit_dets && !visited_detectors[node.num_dets].insert(detectors).second) {
      continue;
    }

    size_t min_detector = std::numeric_limits<size_t>::max();
    for (size_t offset = 0; offset < num_detectors; ++offset) {
      const size_t detector = config.det_orders[detector_order][offset];
      if (detectors[detector]) {
        min_detector = detector;
        break;
      }
    }

    std::vector<uint8_t> prefix_blocked = blocked_flags;

    size_t children_generated = 0;
    size_t children_projected = 0;
    size_t children_beam_pruned = 0;
    size_t children_infeasible = 0;

    for (int ei : d2e[min_detector]) {
      prefix_blocked[(size_t)ei] = 1;
      if (blocked_flags[(size_t)ei]) continue;

      boost::dynamic_bitset<> child_detectors = detectors;
      size_t child_num_dets = node.num_dets;
      for (int detector : edets[(size_t)ei]) {
        if (detectors[(size_t)detector]) {
          --child_num_dets;
        } else {
          ++child_num_dets;
        }
        child_detectors.flip((size_t)detector);
      }
      if (child_num_dets > max_num_dets) {
        children_beam_pruned++;
        continue;
      }

      const double child_h = project_from_exact_solution(
          exact_solution_arena[(size_t)node.exact_solution_idx], child_detectors, prefix_blocked);
      stats.projected_nodes_generated++;
      children_projected++;
      if (child_h == INF_D) {
        children_infeasible++;
        continue;
      }

      error_chain_arena.emplace_back();
      auto& chain_node = error_chain_arena.back();
      chain_node.error_index = (size_t)ei;
      chain_node.min_detector = min_detector;
      chain_node.parent_idx = node.error_chain_idx;

      FTLNode child;
      child.g_cost = node.g_cost + errors[(size_t)ei].likelihood_cost;
      child.h_cost = child_h;
      child.f_cost = child.g_cost + child.h_cost;
      child.num_dets = child_num_dets;
      child.depth = node.depth + 1;
      child.error_chain_idx = (int64_t)error_chain_arena.size() - 1;
      child.warm_solution_idx = node.exact_solution_idx;
      child.exact_solution_idx = -1;
      child.exact_refined = false;
      child.heuristic_source = FTLHeuristicSource::kProjected;
      pq.push(child);
      stats.num_pq_pushed++;
      children_generated++;
      if (stats.num_pq_pushed > config.pqlimit) {
        low_confidence_flag = true;
        return;
      }
    }

    if (config.verbose) {
      const size_t projected_unrefined =
          stats.projected_nodes_generated - stats.projected_nodes_refined;
      std::cout << "  expanded children_generated=" << children_generated
                << " children_projected=" << children_projected
                << " beam_pruned=" << children_beam_pruned << " infeasible=" << children_infeasible
                << " lp_calls=" << stats.lp_calls
                << " proj_unrefined_so_far=" << projected_unrefined << std::endl;
    }
  }

  if (config.verbose) {
    std::cout << "Decoding failed to converge within beam limit." << std::endl;
  }
  low_confidence_flag = true;
}

double TesseractFTLDecoder::cost_from_errors(const std::vector<size_t>& predicted_errors) const {
  if (plain_delegate) return plain_delegate->cost_from_errors(predicted_errors);
  double total_cost = 0.0;
  for (size_t dem_error_index : predicted_errors) {
    const size_t error_index = dem_error_to_error[dem_error_index];
    if (error_index == std::numeric_limits<size_t>::max()) {
      throw std::invalid_argument("error index does not map to a retained decoder error");
    }
    total_cost += errors[error_index].likelihood_cost;
  }
  return total_cost;
}

std::vector<int> TesseractFTLDecoder::get_flipped_observables(
    const std::vector<size_t>& predicted_errors) const {
  if (plain_delegate) return plain_delegate->get_flipped_observables(predicted_errors);
  std::vector<uint8_t> toggled(num_observables, 0);
  for (size_t dem_error_index : predicted_errors) {
    const size_t error_index = dem_error_to_error[dem_error_index];
    if (error_index == std::numeric_limits<size_t>::max()) {
      throw std::invalid_argument("error index does not map to a retained decoder error");
    }
    for (int obs_index : errors[error_index].symptom.observables) {
      toggled[(size_t)obs_index] ^= 1;
    }
  }
  std::vector<int> flipped_observables;
  flipped_observables.reserve(num_observables);
  for (size_t obs_index = 0; obs_index < num_observables; ++obs_index) {
    if (toggled[obs_index]) flipped_observables.push_back((int)obs_index);
  }
  return flipped_observables;
}

std::vector<int> TesseractFTLDecoder::decode(const std::vector<uint64_t>& detections) {
  decode_to_errors(detections);
  return get_flipped_observables(predicted_errors_buffer);
}

void TesseractFTLDecoder::decode_shots(std::vector<stim::SparseShot>& shots,
                                       std::vector<std::vector<int>>& obs_predicted) {
  obs_predicted.resize(shots.size());
  for (size_t i = 0; i < shots.size(); ++i) {
    obs_predicted[i] = decode(shots[i].hits);
  }
}
