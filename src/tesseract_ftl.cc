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
constexpr size_t MAX_PARITY_COMPONENT_COMPRESSED_VARS = 2048;

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

template <typename T>
struct IntVectorHash {
  size_t operator()(const T& values) const {
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

struct DenseGeneralSimplexResult {
  bool success = false;
  bool infeasible = false;
  bool unbounded = false;
  double objective = 0.0;
  size_t pivots = 0;
  std::vector<double> solution;
};

template <typename T>
double dot_on_support(const std::vector<double>& values, const T& support) {
  double total = 0.0;
  for (int idx : support) total += values[(size_t)idx];
  return total;
}

class DenseGeneralSimplexSolver {
 public:
  DenseGeneralSimplexSolver(const std::vector<std::vector<double>>& A,
                            const std::vector<double>& b, const std::vector<double>& c)
      : m_((int)b.size()), n_((int)c.size()), basis_(b.size()), nonbasis_(c.size() + 1) {
    tableau_.assign((size_t)m_ + 2, std::vector<double>((size_t)n_ + 2, 0.0));
    for (int i = 0; i < m_; ++i) {
      for (int j = 0; j < n_; ++j) {
        tableau_[(size_t)i][(size_t)j] = A[(size_t)i][(size_t)j];
      }
      basis_[(size_t)i] = n_ + i;
      tableau_[(size_t)i][(size_t)n_] = -1.0;
      tableau_[(size_t)i][(size_t)n_ + 1] = b[(size_t)i];
    }
    for (int j = 0; j < n_; ++j) {
      nonbasis_[(size_t)j] = j;
      tableau_[(size_t)m_][(size_t)j] = -c[(size_t)j];
    }
    nonbasis_[(size_t)n_] = -1;
    tableau_[(size_t)m_ + 1][(size_t)n_] = 1.0;
  }

  DenseGeneralSimplexResult solve() {
    DenseGeneralSimplexResult result;
    result.solution.assign((size_t)n_, 0.0);

    if (n_ == 0) {
      for (int i = 0; i < m_; ++i) {
        if (tableau_[(size_t)i][1] < -SIMPLEX_EPS) {
          result.infeasible = true;
          return result;
        }
      }
      result.success = true;
      return result;
    }

    if (m_ == 0) {
      for (int j = 0; j < n_; ++j) {
        if (-tableau_[0][(size_t)j] > SIMPLEX_EPS) {
          result.unbounded = true;
          return result;
        }
      }
      result.success = true;
      return result;
    }

    int r = 0;
    for (int i = 1; i < m_; ++i) {
      if (tableau_[(size_t)i][(size_t)n_ + 1] < tableau_[(size_t)r][(size_t)n_ + 1]) {
        r = i;
      }
    }

    if (tableau_[(size_t)r][(size_t)n_ + 1] < -SIMPLEX_EPS) {
      pivot(r, n_);
      if (!simplex(1) || tableau_[(size_t)m_ + 1][(size_t)n_ + 1] < -SIMPLEX_EPS) {
        result.infeasible = true;
        result.pivots = pivots_;
        return result;
      }
      if (tableau_[(size_t)m_ + 1][(size_t)n_ + 1] > SIMPLEX_EPS) {
        result.infeasible = true;
        result.pivots = pivots_;
        return result;
      }
      for (int i = 0; i < m_; ++i) {
        if (basis_[(size_t)i] != -1) continue;
        int s = -1;
        for (int j = 0; j < n_; ++j) {
          if (std::abs(tableau_[(size_t)i][(size_t)j]) <= SIMPLEX_EPS) continue;
          if (s == -1 || nonbasis_[(size_t)j] < nonbasis_[(size_t)s]) s = j;
        }
        if (s != -1) pivot(i, s);
      }
    }

    if (!simplex(2)) {
      result.unbounded = true;
      result.pivots = pivots_;
      return result;
    }

    for (int i = 0; i < m_; ++i) {
      if (basis_[(size_t)i] >= 0 && basis_[(size_t)i] < n_) {
        double value = tableau_[(size_t)i][(size_t)n_ + 1];
        if (std::abs(value) <= SIMPLEX_EPS) value = 0.0;
        result.solution[(size_t)basis_[(size_t)i]] = value;
      }
    }
    result.objective = tableau_[(size_t)m_][(size_t)n_ + 1];
    if (std::abs(result.objective) <= SIMPLEX_EPS) result.objective = 0.0;
    result.pivots = pivots_;
    result.success = true;
    return result;
  }

 private:
  int m_;
  int n_;
  std::vector<int> basis_;
  std::vector<int> nonbasis_;
  std::vector<std::vector<double>> tableau_;
  size_t pivots_ = 0;

  void pivot(int r, int s) {
    const double inv = 1.0 / tableau_[(size_t)r][(size_t)s];
    for (int i = 0; i < m_ + 2; ++i) {
      if (i == r) continue;
      for (int j = 0; j < n_ + 2; ++j) {
        if (j == s) continue;
        tableau_[(size_t)i][(size_t)j] -=
            tableau_[(size_t)r][(size_t)j] * tableau_[(size_t)i][(size_t)s] * inv;
      }
    }
    for (int j = 0; j < n_ + 2; ++j) {
      if (j != s) tableau_[(size_t)r][(size_t)j] *= inv;
    }
    for (int i = 0; i < m_ + 2; ++i) {
      if (i != r) tableau_[(size_t)i][(size_t)s] *= -inv;
    }
    tableau_[(size_t)r][(size_t)s] = inv;
    std::swap(basis_[(size_t)r], nonbasis_[(size_t)s]);
    pivots_++;
  }

  bool simplex(int phase) {
    const int objective_row = phase == 1 ? m_ + 1 : m_;
    while (true) {
      int s = -1;
      for (int j = 0; j <= n_; ++j) {
        if (phase == 2 && nonbasis_[(size_t)j] == -1) continue;
        if (s == -1 || tableau_[(size_t)objective_row][(size_t)j] <
                             tableau_[(size_t)objective_row][(size_t)s] - SIMPLEX_EPS ||
            (std::abs(tableau_[(size_t)objective_row][(size_t)j] -
                      tableau_[(size_t)objective_row][(size_t)s]) <= SIMPLEX_EPS &&
             nonbasis_[(size_t)j] < nonbasis_[(size_t)s])) {
          s = j;
        }
      }
      if (s == -1 || tableau_[(size_t)objective_row][(size_t)s] >= -SIMPLEX_EPS) {
        return true;
      }

      int r = -1;
      for (int i = 0; i < m_; ++i) {
        if (tableau_[(size_t)i][(size_t)s] <= SIMPLEX_EPS) continue;
        const double ratio_i = tableau_[(size_t)i][(size_t)n_ + 1] / tableau_[(size_t)i][(size_t)s];
        if (r == -1) {
          r = i;
          continue;
        }
        const double ratio_r = tableau_[(size_t)r][(size_t)n_ + 1] /
                               tableau_[(size_t)r][(size_t)s];
        if (ratio_i < ratio_r - SIMPLEX_EPS ||
            (std::abs(ratio_i - ratio_r) <= SIMPLEX_EPS &&
             basis_[(size_t)i] < basis_[(size_t)r])) {
          r = i;
        }
      }
      if (r == -1) return false;
      pivot(r, s);
    }
  }
};

DenseGeneralSimplexResult solve_dense_general_max_lp(const std::vector<std::vector<double>>& A,
                                                      const std::vector<double>& b,
                                                      const std::vector<double>& c) {
  DenseGeneralSimplexSolver solver(A, b, c);
  return solver.solve();
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

    DenseSimplexResult simplex = solve_dense_primal_packing_lp(num_local_detectors, constraints,
                                                               selected_indices, &seed_budgets);
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

struct ParityForbiddenSetCut {
  int detector_local = -1;
  std::vector<int> forbidden_local_errors;
};

std::vector<int> parity_cut_key(const ParityForbiddenSetCut& cut) {
  std::vector<int> key;
  key.reserve(cut.forbidden_local_errors.size() + 1);
  key.push_back(cut.detector_local);
  key.insert(key.end(), cut.forbidden_local_errors.begin(), cut.forbidden_local_errors.end());
  return key;
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

std::string detector_choice_policy_to_string(FTLDetectorChoicePolicy policy) {
  switch (policy) {
    case FTLDetectorChoicePolicy::kOrder:
      return "order";
    case FTLDetectorChoicePolicy::kFewestIncidentErrors:
      return "fewest_incident_errors";
    case FTLDetectorChoicePolicy::kLargestBudget:
      return "largest_budget";
    case FTLDetectorChoicePolicy::kLargestBudgetPerIncident:
      return "largest_budget_per_incident";
  }
  return "unknown";
}

std::string error_order_policy_to_string(FTLErrorOrderPolicy policy) {
  switch (policy) {
    case FTLErrorOrderPolicy::kStatic:
      return "static";
    case FTLErrorOrderPolicy::kReducedCost:
      return "reduced_cost";
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
  ss << "lb_level=" << lb_level << ", ";
  ss << "ignore_blocked_errors_in_heuristic=" << ignore_blocked_errors_in_heuristic << ", ";
  ss << "num_min_dets_to_consider=" << num_min_dets_to_consider << ", ";
  ss << "detector_choice_policy=" << detector_choice_policy_to_string(detector_choice_policy)
     << ", ";
  ss << "error_order_policy=" << error_order_policy_to_string(error_order_policy) << ", ";
  ss << "root_det_order_count=" << root_det_order_count << ", ";
  ss << "root_det_order_depth=" << root_det_order_depth << ", ";
  ss << "exact_child_refine_count=" << exact_child_refine_count;
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
  chain_replay_total_seconds += other.chain_replay_total_seconds;
  component_build_total_seconds += other.component_build_total_seconds;
  component_candidate_total_seconds += other.component_candidate_total_seconds;
  component_union_total_seconds += other.component_union_total_seconds;
  component_dedup_total_seconds += other.component_dedup_total_seconds;
  component_finalize_total_seconds += other.component_finalize_total_seconds;
  simplex_total_seconds += other.simplex_total_seconds;
  projection_total_seconds += other.projection_total_seconds;
  component_build_calls += other.component_build_calls;
  simplex_calls += other.simplex_calls;
  projection_calls += other.projection_calls;
  detector_choice_calls += other.detector_choice_calls;
  error_ordering_calls += other.error_ordering_calls;
  total_active_detectors_popped += other.total_active_detectors_popped;
  total_root_order_candidates += other.total_root_order_candidates;
  total_min_detector_candidates += other.total_min_detector_candidates;
  total_min_detectors_selected += other.total_min_detectors_selected;
  total_min_detector_available_errors += other.total_min_detector_available_errors;
  total_min_detector_blocked_errors += other.total_min_detector_blocked_errors;
  total_child_candidates_considered += other.total_child_candidates_considered;
  total_children_generated += other.total_children_generated;
  total_children_beam_pruned += other.total_children_beam_pruned;
  total_children_infeasible += other.total_children_infeasible;
  total_selected_min_detector_budget += other.total_selected_min_detector_budget;
  exact_child_pre_refinements += other.exact_child_pre_refinements;
}

bool TesseractFTLDecoder::FTLNode::operator>(const FTLNode& other) const {
  return f_cost > other.f_cost || (f_cost == other.f_cost && num_dets < other.num_dets);
}

size_t TesseractFTLDecoder::DynamicBitsetHash::operator()(const boost::dynamic_bitset<>& bs) const {
  return boost::hash_value(bs);
}

TesseractFTLDecoder::TesseractFTLDecoder(TesseractFTLConfig config_) : config(config_) {
  if (config.lb_level > 2) {
    throw std::invalid_argument("tesseract_ftl supports only lb_level values 0, 1, and 2");
  }

  if (config.lb_level == 0) {
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
  candidate_error_marks.assign(num_errors, 0);
  candidate_error_mark_epoch = 1;

  for (size_t ei = 0; ei < num_errors; ++ei) {
    edets[ei] = errors[ei].symptom.detectors;
    for (int d : edets[ei]) {
      d2e[(size_t)d].push_back((int)ei);
    }
    const size_t det_degree = errors[ei].symptom.detectors.size();
    error_costs[ei] = {errors[ei].likelihood_cost,
                       det_degree == 0 ? errors[ei].likelihood_cost
                                       : errors[ei].likelihood_cost / det_degree};
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

void block_errors_from_chain(const std::vector<common::ErrorChainNode>& error_chain_arena,
                             const std::vector<std::vector<int>>& d2e, int64_t error_chain_idx,
                             std::vector<uint8_t>& blocked_flags) {
  int64_t walker_idx = error_chain_idx;
  while (walker_idx != -1) {
    const auto& node = error_chain_arena[(size_t)walker_idx];
    const size_t ei = node.error_index;
    const size_t min_detector = node.min_detector;
    for (int oei : d2e[min_detector]) {
      blocked_flags[(size_t)oei] = 1;
      if ((size_t)oei == ei) break;
    }
    walker_idx = node.parent_idx;
  }
}

TesseractFTLDecoder::SingletonBuildResult TesseractFTLDecoder::build_singleton_components(
    const boost::dynamic_bitset<>& detectors, const std::vector<uint8_t>& blocked_flags) {
  SingletonBuildResult result;
  const auto candidate_start_time = std::chrono::high_resolution_clock::now();

  std::vector<int> active_detectors;
  active_detectors.reserve(detectors.count());
  std::vector<int> detector_to_active_pos(num_detectors, -1);
  for (size_t detector = detectors.find_first(); detector != boost::dynamic_bitset<>::npos;
       detector = detectors.find_next(detector)) {
    detector_to_active_pos[detector] = (int)active_detectors.size();
    active_detectors.push_back((int)detector);
  }
  if (active_detectors.empty()) return result;

  if (candidate_error_mark_epoch == std::numeric_limits<uint64_t>::max()) {
    std::fill(candidate_error_marks.begin(), candidate_error_marks.end(), 0);
    candidate_error_mark_epoch = 1;
  }
  const uint64_t mark_epoch = candidate_error_mark_epoch++;
  std::vector<int> candidate_errors;
  for (int detector : active_detectors) {
    for (int ei : d2e[(size_t)detector]) {
      if (blocked_flags[(size_t)ei]) continue;
      if (candidate_error_marks[(size_t)ei] == mark_epoch) continue;
      candidate_error_marks[(size_t)ei] = mark_epoch;
      candidate_errors.push_back(ei);
    }
  }
  const auto candidate_stop_time = std::chrono::high_resolution_clock::now();
  stats.component_candidate_total_seconds += std::chrono::duration_cast<std::chrono::microseconds>(
                                                 candidate_stop_time - candidate_start_time)
                                                 .count() /
                                             1e6;

  const auto union_start_time = std::chrono::high_resolution_clock::now();
  UnionFind uf(active_detectors.size());
  std::vector<uint8_t> has_available(active_detectors.size(), 0);

  for (int ei : candidate_errors) {
    int first_active = -1;
    for (int detector : edets[(size_t)ei]) {
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

  std::vector<int> root_to_component_index(active_detectors.size(), -1);
  std::vector<int> active_pos_to_component(active_detectors.size(), -1);
  std::vector<int> active_pos_to_local(active_detectors.size(), -1);
  result.components.reserve(active_detectors.size());
  for (int active_pos = 0; active_pos < (int)active_detectors.size(); ++active_pos) {
    const int root = uf.find(active_pos);
    int& component_index = root_to_component_index[(size_t)root];
    if (component_index < 0) {
      component_index = (int)result.components.size();
      result.components.emplace_back();
    }
    auto& component = result.components[(size_t)component_index];
    active_pos_to_component[(size_t)active_pos] = component_index;
    active_pos_to_local[(size_t)active_pos] = (int)component.detectors.size();
    component.detectors.push_back(active_detectors[(size_t)active_pos]);
  }
  const auto union_stop_time = std::chrono::high_resolution_clock::now();
  stats.component_union_total_seconds +=
      std::chrono::duration_cast<std::chrono::microseconds>(union_stop_time - union_start_time)
          .count() /
      1e6;

  const auto dedup_start_time = std::chrono::high_resolution_clock::now();
  std::vector<std::unordered_map<std::vector<int>, double, IntVectorHash<std::vector<int>>>>
      min_rhs_by_pattern(result.components.size());
  std::vector<int> local_hits;
  local_hits.reserve(16);

  for (int ei : candidate_errors) {
    int component_index = -1;
    local_hits.clear();

    for (int detector : edets[(size_t)ei]) {
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
    const double rhs = errors[(size_t)ei].likelihood_cost;
    auto& rhs_map = min_rhs_by_pattern[(size_t)component_index];
    auto it = rhs_map.find(local_hits);
    if (it == rhs_map.end() || rhs < it->second) {
      rhs_map[local_hits] = rhs;
    }
  }
  const auto dedup_stop_time = std::chrono::high_resolution_clock::now();
  stats.component_dedup_total_seconds +=
      std::chrono::duration_cast<std::chrono::microseconds>(dedup_stop_time - dedup_start_time)
          .count() /
      1e6;

  const auto finalize_start_time = std::chrono::high_resolution_clock::now();
  for (size_t component_index = 0; component_index < result.components.size(); ++component_index) {
    auto& component = result.components[component_index];
    const auto& rhs_map = min_rhs_by_pattern[component_index];
    component.constraints.reserve(rhs_map.size());
    for (const auto& [local_hits_key, rhs] : rhs_map) {
      component.constraints.push_back({local_hits_key, rhs});
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
  const auto finalize_stop_time = std::chrono::high_resolution_clock::now();
  stats.component_finalize_total_seconds += std::chrono::duration_cast<std::chrono::microseconds>(
                                                finalize_stop_time - finalize_start_time)
                                                .count() /
                                            1e6;

  return result;
}

TesseractFTLDecoder::ExactSubsetSolution TesseractFTLDecoder::solve_singleton_lower_bound(
    const boost::dynamic_bitset<>& detectors, const std::vector<uint8_t>& blocked_flags,
    int64_t warm_solution_idx) {
  ExactSubsetSolution solution;

  const auto build_start_time = std::chrono::high_resolution_clock::now();
  const auto build = build_singleton_components(detectors, blocked_flags);
  const auto build_stop_time = std::chrono::high_resolution_clock::now();
  stats.component_build_calls++;
  stats.component_build_total_seconds +=
      std::chrono::duration_cast<std::chrono::microseconds>(build_stop_time - build_start_time)
          .count() /
      1e6;
  if (!build.feasible) {
    solution.value = INF_D;
    return solution;
  }
  if (build.components.empty()) {
    solution.value = 0.0;
    return solution;
  }

  const ExactSubsetSolution* warm_solution =
      warm_solution_idx >= 0 ? &exact_solution_arena[(size_t)warm_solution_idx] : nullptr;
  solution.value = 0.0;
  solution.num_components = build.components.size();
  std::vector<std::pair<int, double>> detector_budget_pairs;
  detector_budget_pairs.reserve(detectors.count());

  for (const auto& component : build.components) {
    std::vector<double> seed_budgets(component.detectors.size(), 0.0);
    if (warm_solution != nullptr) {
      for (size_t local = 0; local < component.detectors.size(); ++local) {
        seed_budgets[local] = lookup_detector_budget(*warm_solution, component.detectors[local]);
      }
    }
    const auto simplex_start_time = std::chrono::high_resolution_clock::now();
    const auto component_result = solve_singleton_component_lp(
        component.detectors.size(), component.constraints,
        component.cheapest_constraint_for_local_detector, seed_budgets);
    const auto simplex_stop_time = std::chrono::high_resolution_clock::now();
    stats.simplex_calls++;
    stats.simplex_total_seconds += std::chrono::duration_cast<std::chrono::microseconds>(
                                       simplex_stop_time - simplex_start_time)
                                       .count() /
                                   1e6;
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

  return solution;
}

TesseractFTLDecoder::ParityBuildResult TesseractFTLDecoder::build_parity_components(
    const boost::dynamic_bitset<>& detectors, const std::vector<uint8_t>& blocked_flags) {
  ParityBuildResult result;
  if (detectors.none()) return result;

  const auto candidate_start_time = std::chrono::high_resolution_clock::now();

  std::vector<int> active_detectors;
  active_detectors.reserve(detectors.count());
  std::vector<int> detector_to_active_local(num_detectors, -1);
  for (size_t detector = detectors.find_first(); detector != boost::dynamic_bitset<>::npos;
       detector = detectors.find_next(detector)) {
    detector_to_active_local[detector] = (int)active_detectors.size();
    active_detectors.push_back((int)detector);
  }
  if (active_detectors.empty()) return result;

  if (candidate_error_mark_epoch == std::numeric_limits<uint64_t>::max()) {
    std::fill(candidate_error_marks.begin(), candidate_error_marks.end(), 0);
    candidate_error_mark_epoch = 1;
  }
  const uint64_t mark_epoch = candidate_error_mark_epoch++;

  struct CompressedParityVariable {
    std::vector<int> active_local_detectors;
    int representative_error = -1;
    double cost = INF_D;
    size_t raw_count = 0;
  };

  std::vector<CompressedParityVariable> compressed_variables;
  std::unordered_map<std::vector<int>, int, IntVectorHash<std::vector<int>>> footprint_to_variable;
  footprint_to_variable.reserve(active_detectors.size() * 8 + 1);

  std::vector<int> footprint;
  footprint.reserve(8);
  for (int active_detector : active_detectors) {
    for (int ei : d2e[(size_t)active_detector]) {
      if (blocked_flags[(size_t)ei]) continue;
      if (candidate_error_marks[(size_t)ei] == mark_epoch) continue;
      candidate_error_marks[(size_t)ei] = mark_epoch;

      footprint.clear();
      for (int detector : edets[(size_t)ei]) {
        const int active_local = detector_to_active_local[(size_t)detector];
        if (active_local >= 0) footprint.push_back(active_local);
      }
      if (footprint.empty()) continue;
      std::sort(footprint.begin(), footprint.end());
      footprint.erase(std::unique(footprint.begin(), footprint.end()), footprint.end());

      auto it = footprint_to_variable.find(footprint);
      if (it == footprint_to_variable.end()) {
        const int variable_index = (int)compressed_variables.size();
        footprint_to_variable.emplace(footprint, variable_index);
        compressed_variables.emplace_back();
        auto& variable = compressed_variables.back();
        variable.active_local_detectors = footprint;
        variable.representative_error = ei;
        variable.cost = errors[(size_t)ei].likelihood_cost;
        variable.raw_count = 1;
      } else {
        auto& variable = compressed_variables[(size_t)it->second];
        variable.raw_count++;
        const double cost = errors[(size_t)ei].likelihood_cost;
        if (cost < variable.cost) {
          variable.cost = cost;
          variable.representative_error = ei;
        }
      }
    }
  }

  const auto candidate_stop_time = std::chrono::high_resolution_clock::now();
  stats.component_candidate_total_seconds += std::chrono::duration_cast<std::chrono::microseconds>(
                                                 candidate_stop_time - candidate_start_time)
                                                 .count() /
                                             1e6;

  const auto union_start_time = std::chrono::high_resolution_clock::now();
  UnionFind uf(active_detectors.size());
  std::vector<uint8_t> has_available(active_detectors.size(), 0);
  for (const auto& variable : compressed_variables) {
    if (variable.active_local_detectors.empty()) continue;
    const int first = variable.active_local_detectors[0];
    for (int active_local : variable.active_local_detectors) {
      has_available[(size_t)active_local] = 1;
      if (active_local != first) uf.unite(first, active_local);
    }
  }
  for (size_t active_local = 0; active_local < active_detectors.size(); ++active_local) {
    if (!has_available[active_local]) {
      result.feasible = false;
      return result;
    }
  }

  std::vector<int> root_to_component_index(active_detectors.size(), -1);
  std::vector<int> active_local_to_component(active_detectors.size(), -1);
  std::vector<int> active_local_to_component_local(active_detectors.size(), -1);
  for (size_t active_local = 0; active_local < active_detectors.size(); ++active_local) {
    const int root = uf.find((int)active_local);
    int& component_index = root_to_component_index[(size_t)root];
    if (component_index < 0) {
      component_index = (int)result.components.size();
      result.components.emplace_back();
    }
    auto& component = result.components[(size_t)component_index];
    active_local_to_component[active_local] = component_index;
    active_local_to_component_local[active_local] = (int)component.detectors.size();
    component.detectors.push_back(active_detectors[active_local]);
    component.incident_local_errors.emplace_back();
    component.detector_parities.push_back(1);
  }
  const auto union_stop_time = std::chrono::high_resolution_clock::now();
  stats.component_union_total_seconds +=
      std::chrono::duration_cast<std::chrono::microseconds>(union_stop_time - union_start_time)
          .count() /
      1e6;

  const auto finalize_start_time = std::chrono::high_resolution_clock::now();
  for (const auto& variable : compressed_variables) {
    if (variable.active_local_detectors.empty()) continue;
    const int component_index = active_local_to_component[(size_t)variable.active_local_detectors[0]];
    auto& component = result.components[(size_t)component_index];
    const int component_local_error = (int)component.compressed_error_costs.size();
    component.representative_errors.push_back(variable.representative_error);
    component.compressed_error_costs.push_back(variable.cost);
    component.raw_error_count += variable.raw_count;

    for (int active_local : variable.active_local_detectors) {
      assert(active_local_to_component[(size_t)active_local] == component_index);
      const int component_local_detector = active_local_to_component_local[(size_t)active_local];
      component.incident_local_errors[(size_t)component_local_detector].push_back(
          component_local_error);
    }
  }

  for (auto& component : result.components) {
    for (auto& incident : component.incident_local_errors) {
      std::sort(incident.begin(), incident.end());
      incident.erase(std::unique(incident.begin(), incident.end()), incident.end());
      if (incident.empty()) {
        result.feasible = false;
        result.components.clear();
        return result;
      }
    }
  }
  const auto finalize_stop_time = std::chrono::high_resolution_clock::now();
  stats.component_finalize_total_seconds += std::chrono::duration_cast<std::chrono::microseconds>(
                                                finalize_stop_time - finalize_start_time)
                                                .count() /
                                            1e6;

  return result;
}

double TesseractFTLDecoder::solve_parity_component_lp(const ParityLPComponent& component,
                                                       size_t& num_constraints,
                                                       size_t& num_simplex_solves) {
  num_constraints = 0;
  num_simplex_solves = 0;

  const size_t num_vars = component.compressed_error_costs.size();
  if (num_vars == 0) {
    for (size_t local_detector = 0; local_detector < component.detectors.size();
         ++local_detector) {
      if (component.detector_parities[local_detector]) return INF_D;
    }
    return 0.0;
  }

  std::vector<ParityForbiddenSetCut> active_cuts;
  std::unordered_set<std::vector<int>, IntVectorHash<std::vector<int>>> active_cut_keys;
  active_cuts.reserve(component.detectors.size());

  auto add_cut_if_new = [&](ParityForbiddenSetCut cut) {
    std::sort(cut.forbidden_local_errors.begin(), cut.forbidden_local_errors.end());
    auto key = parity_cut_key(cut);
    if (active_cut_keys.insert(key).second) {
      active_cuts.push_back(std::move(cut));
      return true;
    }
    return false;
  };

  for (size_t local_detector = 0; local_detector < component.detectors.size();
       ++local_detector) {
    if (!component.detector_parities[local_detector]) continue;
    if (component.incident_local_errors[local_detector].empty()) return INF_D;
    add_cut_if_new({(int)local_detector, {}});
  }

  std::vector<double> objective(num_vars, 0.0);
  for (size_t local_error = 0; local_error < num_vars; ++local_error) {
    objective[local_error] = -component.compressed_error_costs[local_error];
  }

  std::vector<double> solution(num_vars, 0.0);
  double min_objective = 0.0;

  while (true) {
    std::vector<std::vector<double>> A;
    std::vector<double> b;
    A.reserve(num_vars + active_cuts.size());
    b.reserve(num_vars + active_cuts.size());

    for (size_t local_error = 0; local_error < num_vars; ++local_error) {
      std::vector<double> row(num_vars, 0.0);
      row[local_error] = 1.0;
      A.push_back(std::move(row));
      b.push_back(1.0);
    }

    for (const auto& cut : active_cuts) {
      std::vector<double> row(num_vars, 0.0);
      const auto& incident = component.incident_local_errors[(size_t)cut.detector_local];
      for (int local_error : incident) row[(size_t)local_error] = -1.0;
      for (int local_error : cut.forbidden_local_errors) row[(size_t)local_error] = 1.0;
      A.push_back(std::move(row));
      b.push_back((double)cut.forbidden_local_errors.size() - 1.0);
    }

    DenseGeneralSimplexResult simplex = solve_dense_general_max_lp(A, b, objective);
    num_simplex_solves++;
    if (simplex.infeasible) return INF_D;
    if (simplex.unbounded) {
      throw std::runtime_error("Parity-cut LP became unbounded despite box constraints.");
    }
    if (!simplex.success) {
      throw std::runtime_error("Parity-cut LP failed.");
    }

    solution = std::move(simplex.solution);
    for (double& value : solution) {
      if (value < 0.0 && value > -1e-8) value = 0.0;
      if (value > 1.0 && value < 1.0 + 1e-8) value = 1.0;
    }
    min_objective = -simplex.objective;
    if (std::abs(min_objective) <= SIMPLEX_EPS) min_objective = 0.0;

    std::vector<std::pair<double, ParityForbiddenSetCut>> violated_cuts;
    violated_cuts.reserve(component.detectors.size());

    for (size_t local_detector = 0; local_detector < component.detectors.size();
         ++local_detector) {
      const auto& incident = component.incident_local_errors[local_detector];
      const uint8_t target_parity = component.detector_parities[local_detector];
      if (incident.empty()) {
        if (target_parity) return INF_D;
        continue;
      }

      std::vector<int> forbidden_set;
      forbidden_set.reserve(incident.size());
      int parity = 0;
      double distance = 0.0;
      int best_flip = incident[0];
      double best_flip_delta = INF_D;

      for (int local_error : incident) {
        double y = solution[(size_t)local_error];
        if (y < 0.0) y = 0.0;
        if (y > 1.0) y = 1.0;
        const double flip_delta = std::abs(1.0 - 2.0 * y);
        if (flip_delta < best_flip_delta) {
          best_flip_delta = flip_delta;
          best_flip = local_error;
        }
        if (y > 0.5) {
          forbidden_set.push_back(local_error);
          parity ^= 1;
          distance += 1.0 - y;
        } else {
          distance += y;
        }
      }

      if (parity == (int)target_parity) {
        auto it = std::lower_bound(forbidden_set.begin(), forbidden_set.end(), best_flip);
        if (it != forbidden_set.end() && *it == best_flip) {
          forbidden_set.erase(it);
        } else {
          forbidden_set.insert(it, best_flip);
        }
        distance += best_flip_delta;
      }

      const double violation = 1.0 - distance;
      if (violation <= VIOLATION_EPS) continue;
      violated_cuts.push_back({violation, {(int)local_detector, std::move(forbidden_set)}});
    }

    if (violated_cuts.empty()) {
      num_constraints = num_vars + active_cuts.size();
      return min_objective;
    }

    std::sort(violated_cuts.begin(), violated_cuts.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    bool added_any = false;
    for (auto& [_, cut] : violated_cuts) {
      added_any |= add_cut_if_new(std::move(cut));
    }
    if (!added_any) {
      num_constraints = num_vars + active_cuts.size();
      return min_objective;
    }
  }
}

TesseractFTLDecoder::ExactSubsetSolution TesseractFTLDecoder::solve_parity_lower_bound(
    const boost::dynamic_bitset<>& detectors, const std::vector<uint8_t>& blocked_flags,
    ExactSubsetSolution singleton_seed) {
  const double singleton_value = singleton_seed.value;
  ExactSubsetSolution solution = singleton_seed;
  solution.value = 0.0;
  solution.num_active_subsets = 0;
  solution.num_components = 0;
  solution.num_variables = 0;
  solution.num_raw_variables = 0;
  solution.num_constraints = 0;

  const auto build_start_time = std::chrono::high_resolution_clock::now();
  const auto build = build_parity_components(detectors, blocked_flags);
  const auto build_stop_time = std::chrono::high_resolution_clock::now();
  stats.component_build_calls++;
  stats.component_build_total_seconds +=
      std::chrono::duration_cast<std::chrono::microseconds>(build_stop_time - build_start_time)
          .count() /
      1e6;

  if (!build.feasible) {
    solution.value = INF_D;
    return solution;
  }
  if (build.components.empty()) {
    solution.value = 0.0;
    return solution;
  }

  for (const auto& component : build.components) {
    if (component.compressed_error_costs.size() > MAX_PARITY_COMPONENT_COMPRESSED_VARS) {
      return singleton_seed;
    }

    size_t component_constraints = 0;
    size_t component_simplex_solves = 0;
    const auto simplex_start_time = std::chrono::high_resolution_clock::now();
    const double component_value =
        solve_parity_component_lp(component, component_constraints, component_simplex_solves);
    const auto simplex_stop_time = std::chrono::high_resolution_clock::now();
    stats.simplex_calls += component_simplex_solves;
    stats.lp_calls += component_simplex_solves;
    stats.simplex_total_seconds += std::chrono::duration_cast<std::chrono::microseconds>(
                                       simplex_stop_time - simplex_start_time)
                                       .count() /
                                   1e6;

    if (component_value == INF_D) {
      solution.value = INF_D;
      return solution;
    }
    solution.value += component_value;
    solution.num_components++;
    solution.num_variables += component.compressed_error_costs.size();
    solution.num_raw_variables += component.raw_error_count;
    solution.num_constraints += component_constraints;
    solution.num_active_subsets += component_constraints > component.compressed_error_costs.size()
                                       ? component_constraints - component.compressed_error_costs.size()
                                       : 0;
  }

  if (solution.value + 1e-7 < singleton_value) {
    solution.value = singleton_value;
  }
  return solution;
}

TesseractFTLDecoder::ExactSubsetSolution TesseractFTLDecoder::solve_exact_subset_lp(
    const boost::dynamic_bitset<>& detectors, const std::vector<uint8_t>& blocked_flags,
    int64_t warm_solution_idx) {
  stats.heuristic_calls++;
  stats.exact_refinement_calls++;
  const auto start_time = std::chrono::high_resolution_clock::now();

  std::vector<uint8_t> ignored_blocked_flags;
  const std::vector<uint8_t>* effective_blocked_flags = &blocked_flags;
  if (config.ignore_blocked_errors_in_heuristic) {
    ignored_blocked_flags.assign(num_errors, 0);
    effective_blocked_flags = &ignored_blocked_flags;
  }

  ExactSubsetSolution solution;
  if (config.lb_level == 1) {
    solution = solve_singleton_lower_bound(detectors, *effective_blocked_flags, warm_solution_idx);
  } else if (config.lb_level == 2) {
    ExactSubsetSolution singleton_seed =
        solve_singleton_lower_bound(detectors, *effective_blocked_flags, warm_solution_idx);
    if (singleton_seed.value == INF_D) {
      solution = std::move(singleton_seed);
    } else {
      solution = solve_parity_lower_bound(detectors, *effective_blocked_flags,
                                          std::move(singleton_seed));
    }
  } else {
    throw std::runtime_error("solve_exact_subset_lp called with lb_level 0");
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
  const auto start_time = std::chrono::high_resolution_clock::now();
  stats.projection_calls++;

  double total = 0.0;
  size_t budget_pos = 0;
  const std::vector<uint8_t>* effective_blocked_flags = &blocked_flags;
  std::vector<uint8_t> ignored_blocked_flags;
  if (config.ignore_blocked_errors_in_heuristic) {
    ignored_blocked_flags.assign(num_errors, 0);
    effective_blocked_flags = &ignored_blocked_flags;
  }
  for (size_t detector = detectors.find_first(); detector != boost::dynamic_bitset<>::npos;
       detector = detectors.find_next(detector)) {
    bool has_available = false;
    for (int ei : d2e[detector]) {
      if (!(*effective_blocked_flags)[(size_t)ei]) {
        has_available = true;
        break;
      }
    }
    if (!has_available) {
      const auto stop_time = std::chrono::high_resolution_clock::now();
      stats.projection_total_seconds +=
          std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count() /
          1e6;
      return INF_D;
    }

    while (budget_pos < solution.active_detectors.size() &&
           solution.active_detectors[budget_pos] < (int)detector) {
      ++budget_pos;
    }
    if (budget_pos < solution.active_detectors.size() &&
        solution.active_detectors[budget_pos] == (int)detector) {
      total += solution.detector_budgets[budget_pos];
    }
  }
  const auto stop_time = std::chrono::high_resolution_clock::now();
  stats.projection_total_seconds +=
      std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count() / 1e6;
  return total;
}

std::vector<size_t> TesseractFTLDecoder::select_min_detectors(
    const boost::dynamic_bitset<>& detectors, const std::vector<uint8_t>& blocked_flags,
    size_t detector_order, size_t depth, const ExactSubsetSolution& exact_solution) {
  stats.detector_choice_calls++;
  stats.total_active_detectors_popped += detectors.count();

  struct CandidateDetector {
    size_t detector;
    size_t order_rank;
    size_t available_errors;
    double budget;
  };

  const size_t order_count = depth < config.root_det_order_depth
                                 ? std::min(config.root_det_order_count, config.det_orders.size())
                                 : 1;
  std::vector<uint8_t> seen(num_detectors, 0);
  std::vector<CandidateDetector> candidates;
  candidates.reserve(detectors.count());

  size_t discovery_rank = 0;
  for (size_t order_offset = 0; order_offset < order_count; ++order_offset) {
    size_t taken_from_order = 0;
    const size_t order_index = (detector_order + order_offset) % config.det_orders.size();
    for (size_t offset = 0; offset < num_detectors; ++offset) {
      const size_t detector = config.det_orders[order_index][offset];
      if (!detectors[detector]) continue;
      if (!seen[detector]) {
        seen[detector] = 1;
        size_t available_errors = 0;
        for (int ei : d2e[detector]) {
          if (!blocked_flags[(size_t)ei]) {
            available_errors++;
          }
        }
        candidates.push_back({detector, discovery_rank++, available_errors,
                              lookup_detector_budget(exact_solution, (int)detector)});
      }
      taken_from_order++;
      if (config.detector_choice_policy == FTLDetectorChoicePolicy::kOrder &&
          taken_from_order >= config.num_min_dets_to_consider) {
        break;
      }
    }
  }

  stats.total_root_order_candidates += candidates.size();
  stats.total_min_detector_candidates += candidates.size();

  if (config.detector_choice_policy != FTLDetectorChoicePolicy::kOrder) {
    std::stable_sort(candidates.begin(), candidates.end(), [&](const auto& a, const auto& b) {
      switch (config.detector_choice_policy) {
        case FTLDetectorChoicePolicy::kOrder:
          break;
        case FTLDetectorChoicePolicy::kFewestIncidentErrors:
          if (a.available_errors != b.available_errors) {
            return a.available_errors < b.available_errors;
          }
          break;
        case FTLDetectorChoicePolicy::kLargestBudget:
          if (a.budget != b.budget) return a.budget > b.budget;
          break;
        case FTLDetectorChoicePolicy::kLargestBudgetPerIncident: {
          const double a_score =
              a.available_errors == 0 ? INF_D : a.budget / (double)a.available_errors;
          const double b_score =
              b.available_errors == 0 ? INF_D : b.budget / (double)b.available_errors;
          if (a_score != b_score) return a_score > b_score;
          break;
        }
      }
      if (a.order_rank != b.order_rank) return a.order_rank < b.order_rank;
      return a.detector < b.detector;
    });
  }

  std::vector<size_t> selected;
  selected.reserve(std::min(config.num_min_dets_to_consider, candidates.size()));
  for (const auto& candidate : candidates) {
    selected.push_back(candidate.detector);
    stats.total_min_detectors_selected++;
    stats.total_min_detector_available_errors += candidate.available_errors;
    stats.total_selected_min_detector_budget += candidate.budget;
    if (selected.size() >= config.num_min_dets_to_consider) break;
  }
  return selected;
}

std::vector<int> TesseractFTLDecoder::order_candidate_errors(
    size_t min_detector, const boost::dynamic_bitset<>& detectors,
    const std::vector<uint8_t>& blocked_flags, const ExactSubsetSolution& exact_solution) {
  stats.error_ordering_calls++;

  std::vector<int> ordered_errors;
  ordered_errors.reserve(d2e[min_detector].size());

  if (config.error_order_policy == FTLErrorOrderPolicy::kStatic) {
    for (int ei : d2e[min_detector]) {
      if (blocked_flags[(size_t)ei]) {
        stats.total_min_detector_blocked_errors++;
        continue;
      }
      ordered_errors.push_back(ei);
    }
    return ordered_errors;
  }

  struct CandidateError {
    int error_index;
    size_t order_rank;
    double reduced_cost;
    int net_det_delta;
  };
  std::vector<CandidateError> candidates;
  candidates.reserve(d2e[min_detector].size());
  size_t order_rank = 0;
  for (int ei : d2e[min_detector]) {
    if (blocked_flags[(size_t)ei]) {
      stats.total_min_detector_blocked_errors++;
      continue;
    }
    double covered_budget = 0.0;
    int net_det_delta = 0;
    for (int detector : edets[(size_t)ei]) {
      if (detectors[(size_t)detector]) {
        covered_budget += lookup_detector_budget(exact_solution, detector);
        net_det_delta--;
      } else {
        net_det_delta++;
      }
    }
    candidates.push_back(
        {ei, order_rank++, errors[(size_t)ei].likelihood_cost - covered_budget, net_det_delta});
  }
  std::stable_sort(candidates.begin(), candidates.end(), [&](const auto& a, const auto& b) {
    if (a.reduced_cost != b.reduced_cost) return a.reduced_cost < b.reduced_cost;
    if (a.net_det_delta != b.net_det_delta) return a.net_det_delta < b.net_det_delta;
    return a.order_rank < b.order_rank;
  });
  for (const auto& candidate : candidates) ordered_errors.push_back(candidate.error_index);
  return ordered_errors;
}

void TesseractFTLDecoder::reset_decode_state() {
  low_confidence_flag = false;
  predicted_errors_buffer.clear();
  error_chain_arena.clear();
  detector_state_arena.clear();
  exact_solution_arena.clear();
  exact_solution_cache.clear();
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
    detector_state_arena.reserve(reserve_size + 1);
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
  detector_state_arena.push_back(initial_detectors);
  root.detector_state_idx = 0;
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
  if (config.ignore_blocked_errors_in_heuristic) {
    exact_solution_cache.emplace(initial_detectors, root.exact_solution_idx);
  }
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

    boost::dynamic_bitset<> detectors = detector_state_arena[(size_t)node.detector_state_idx];
    std::vector<uint8_t> blocked_flags(num_errors, 0);
    const auto chain_start_time = std::chrono::high_resolution_clock::now();
    block_errors_from_chain(error_chain_arena, d2e, node.error_chain_idx, blocked_flags);
    const auto chain_stop_time = std::chrono::high_resolution_clock::now();
    stats.chain_replay_total_seconds +=
        std::chrono::duration_cast<std::chrono::microseconds>(chain_stop_time - chain_start_time)
            .count() /
        1e6;

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
      bool used_cached_exact_solution = false;
      int64_t cached_exact_solution_idx = -1;
      if (config.ignore_blocked_errors_in_heuristic) {
        auto it = exact_solution_cache.find(detectors);
        if (it != exact_solution_cache.end()) {
          used_cached_exact_solution = true;
          cached_exact_solution_idx = it->second;
        }
      }
      if (prev_source == FTLHeuristicSource::kProjected) stats.projected_nodes_refined++;
      if (used_cached_exact_solution) {
        node.exact_solution_idx = cached_exact_solution_idx;
        node.h_cost = exact_solution_arena[(size_t)cached_exact_solution_idx].value;
        const double delta = node.h_cost - prev_h;
        if (node.h_cost + 1e-7 < prev_h) {
          throw std::runtime_error("Cached lower bound fell below stored lower bound.");
        }
        stats.total_lp_refinement_gain += delta;
        stats.max_lp_refinement_gain = std::max(stats.max_lp_refinement_gain, delta);
        node.f_cost = node.g_cost + node.h_cost;
        node.exact_refined = true;
        node.heuristic_source = FTLHeuristicSource::kExact;
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
      } else {
        ExactSubsetSolution exact_solution =
            solve_exact_subset_lp(detectors, blocked_flags, node.warm_solution_idx);
        if (exact_solution.value == INF_D) {
          if (config.verbose) {
            std::cout << "  lp_refine exact_h=INF discarded=true" << std::endl;
          }
          continue;
        }
        if (exact_solution.value + 1e-7 < prev_h) {
          throw std::runtime_error("Exact lower bound fell below stored lower bound.");
        }
        const double delta = exact_solution.value - prev_h;
        stats.total_lp_refinement_gain += delta;
        stats.max_lp_refinement_gain = std::max(stats.max_lp_refinement_gain, delta);
        exact_solution_arena.push_back(std::move(exact_solution));
        node.exact_solution_idx = (int64_t)exact_solution_arena.size() - 1;
        if (config.ignore_blocked_errors_in_heuristic) {
          exact_solution_cache.emplace(detectors, node.exact_solution_idx);
        }
        node.h_cost = exact_solution_arena.back().value;
        node.f_cost = node.g_cost + node.h_cost;
        node.exact_refined = true;
        node.heuristic_source = FTLHeuristicSource::kExact;
        if (config.verbose) {
          std::cout << "  lp_refine approx_h=" << prev_h << " exact_h=" << node.h_cost
                    << " delta=" << delta << " vars=" << exact_solution_arena.back().num_variables
                    << " raw_vars=" << exact_solution_arena.back().num_raw_variables
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
    }

    if (config.no_revisit_dets && !visited_detectors[node.num_dets].insert(detectors).second) {
      continue;
    }

    const auto& exact_solution = exact_solution_arena[(size_t)node.exact_solution_idx];
    std::vector<size_t> min_detectors =
        select_min_detectors(detectors, blocked_flags, detector_order, node.depth, exact_solution);
    if (min_detectors.empty()) {
      throw std::runtime_error("Failed to select an active min detector for a non-terminal node.");
    }

    size_t children_generated = 0;
    size_t children_projected = 0;
    size_t children_beam_pruned = 0;
    size_t children_infeasible = 0;
    size_t children_exactly_refined = 0;

    for (size_t min_detector : min_detectors) {
      std::vector<uint8_t> prefix_blocked = blocked_flags;
      const std::vector<int> ordered_errors =
          order_candidate_errors(min_detector, detectors, blocked_flags, exact_solution);
      for (int ei : ordered_errors) {
        prefix_blocked[(size_t)ei] = 1;
        stats.total_child_candidates_considered++;

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
          stats.total_children_beam_pruned++;
          continue;
        }

        double child_h =
            project_from_exact_solution(exact_solution, child_detectors, prefix_blocked);
        stats.projected_nodes_generated++;
        children_projected++;
        if (child_h == INF_D) {
          children_infeasible++;
          stats.total_children_infeasible++;
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
        detector_state_arena.push_back(std::move(child_detectors));
        child.detector_state_idx = (int64_t)detector_state_arena.size() - 1;
        child.warm_solution_idx = node.exact_solution_idx;
        child.exact_solution_idx = -1;
        child.exact_refined = false;
        child.heuristic_source = FTLHeuristicSource::kProjected;

        if (config.exact_child_refine_count > 0 &&
            children_exactly_refined < config.exact_child_refine_count) {
          ExactSubsetSolution child_exact =
              solve_exact_subset_lp(detector_state_arena[(size_t)child.detector_state_idx],
                                    prefix_blocked, child.warm_solution_idx);
          if (child_exact.value == INF_D) {
            children_infeasible++;
            stats.total_children_infeasible++;
            continue;
          }
          exact_solution_arena.push_back(std::move(child_exact));
          child.exact_solution_idx = (int64_t)exact_solution_arena.size() - 1;
          child.h_cost = exact_solution_arena.back().value;
          child.f_cost = child.g_cost + child.h_cost;
          child.exact_refined = true;
          child.heuristic_source = FTLHeuristicSource::kExact;
          children_exactly_refined++;
          stats.exact_child_pre_refinements++;
        }

        pq.push(child);
        stats.num_pq_pushed++;
        children_generated++;
        stats.total_children_generated++;
        if (stats.num_pq_pushed > config.pqlimit) {
          low_confidence_flag = true;
          return;
        }
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
