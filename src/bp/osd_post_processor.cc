#include "bp/osd_post_processor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>

#include "bp/bp_params.h"
#include "stim.h"

namespace bp {
namespace {

using ProfileClock = std::chrono::steady_clock;

inline double seconds_since(const ProfileClock::time_point& start) {
  return std::chrono::duration<double>(ProfileClock::now() - start).count();
}

// Check if residual syndrome S_res has zero bits at or above index num_solved.
inline bool is_residual_syndrome_satisfied(const stim::simd_bits<64>& S_res, size_t num_solved,
                                           size_t num_detectors) {
  if (num_solved >= num_detectors) return true;
  size_t start_word = num_solved / 64, end_word = (num_detectors + 63) / 64;
  uint64_t mask = ~((1ULL << (num_solved % 64)) - 1);
  size_t last_bits = num_detectors % 64;
  uint64_t last_mask = (last_bits == 0) ? ~0ULL : ((1ULL << last_bits) - 1);
  if (start_word + 1 == end_word) return (S_res.u64[start_word] & mask & last_mask) == 0;
  if ((S_res.u64[start_word] & mask) != 0) return false;
  for (size_t w = start_word + 1; w + 1 < end_word; ++w) {
    if (S_res.u64[w] != 0) return false;
  }
  return (S_res.u64[end_word - 1] & last_mask) == 0;
}

}  // namespace

OsdPostProcessor::OsdPostProcessor(const TannerGraph& graph, size_t osd_order,
                                   size_t osd_weight, float osd_truncation_factor,
                                   bool osd_early_exit)
    : graph_(graph),
      osd_order_(osd_order),
      osd_weight_(osd_weight),
      osd_truncation_factor_(osd_truncation_factor),
      osd_early_exit_(osd_early_exit) {
  const size_t num_errors = graph_.num_vars;
  base_var_cost_.assign(num_errors, 1.0);
  for (size_t v = 0; v < num_errors; ++v) {
    double c = std::abs(llr_int_to_double(graph_.priors[v]));
    base_var_cost_[v] = (c > 0.0) ? c : 1e-3;
  }
}

std::vector<uint8_t> OsdPostProcessor::process(
    const BPResult& bp_result, const std::vector<LLR_INT>& posteriors,
    const std::vector<uint64_t>& detection_events,
    const std::vector<std::vector<int>>& hyperedge_observables,
    std::vector<size_t>* predicted_errors,
    const std::vector<LLR_INT>* custom_priors) {
  const auto t_call_start = ProfileClock::now();
  profile_.num_calls++;
  const size_t num_errors = graph_.num_vars, num_detectors = graph_.num_checks;

  size_t num_observables = 0;
  for (const auto& obs_list : hyperedge_observables) {
    for (int obs : obs_list) {
      if (obs >= 0) num_observables = std::max(num_observables, static_cast<size_t>(obs) + 1);
    }
  }

  auto record_flipped_var = [&](size_t v, std::vector<uint8_t>& obs_result) {
    if (predicted_errors != nullptr) predicted_errors->push_back(v);
    for (int obs : hyperedge_observables[v]) {
      if (obs >= 0) obs_result[obs] ^= 1;
    }
  };

  if (bp_result.converged) {
    if (predicted_errors != nullptr) predicted_errors->clear();
    std::vector<uint8_t> obs_result(num_observables, 0);
    for (size_t i = 0; i < num_errors; ++i) {
      if (posteriors[i] < 0) record_flipped_var(i, obs_result);
    }
    profile_.num_converged_early_exits++;
    profile_.t_converged_passthrough += seconds_since(t_call_start);
    return obs_result;
  }

  const auto t_osd_start = ProfileClock::now();
  profile_.num_osd_invocations++;
  profile_.sum_num_errors += num_errors;
  profile_.sum_num_detectors += num_detectors;

  const bool has_custom_priors = (custom_priors != nullptr);
  auto var_cost = [&](size_t c) -> double {
    double prior_c = has_custom_priors
                         ? std::max(1e-3, std::abs(llr_int_to_double((*custom_priors)[c])))
                         : base_var_cost_[c];
    return std::max(1e-4, prior_c + 1e-4 * std::clamp(llr_int_to_double(posteriors[c]), -50.0, 50.0));
  };

  auto t_phase = t_osd_start;
  std::vector<bool> e_hard(num_errors, false);
  for (size_t i = 0; i < num_errors; ++i) {
    if (posteriors[i] < 0) e_hard[i] = true;
  }
  profile_.t_hard_decision += seconds_since(t_phase);

  t_phase = ProfileClock::now();
  stim::simd_bits<64> S_res(num_detectors);
  for (uint64_t d : detection_events) {
    if (d < num_detectors) S_res[d] ^= true;
  }
  for (size_t i = 0; i < num_errors; ++i) {
    if (e_hard[i]) {
      for (size_t e = graph_.var_edge_offsets[i]; e < graph_.var_edge_offsets[i + 1]; ++e) {
        S_res[graph_.var_edges[e]] ^= true;
      }
    }
  }
  profile_.t_residual += seconds_since(t_phase);

  size_t max_osd_cols =
      (osd_truncation_factor_ > 0.0f)
          ? std::min(num_errors, static_cast<size_t>(num_detectors * osd_truncation_factor_))
          : num_errors;
  profile_.sum_max_osd_cols += max_osd_cols;

  t_phase = ProfileClock::now();
  std::vector<size_t> sorted_cols(num_errors);
  std::iota(sorted_cols.begin(), sorted_cols.end(), 0);
  auto cmp = [&](size_t a, size_t b) { return std::abs(posteriors[a]) < std::abs(posteriors[b]); };
  if (max_osd_cols < num_errors) {
    std::nth_element(sorted_cols.begin(), sorted_cols.begin() + max_osd_cols, sorted_cols.end(), cmp);
    std::sort(sorted_cols.begin(), sorted_cols.begin() + max_osd_cols, cmp);
  } else {
    std::sort(sorted_cols.begin(), sorted_cols.end(), cmp);
  }
  profile_.t_sort += seconds_since(t_phase);

  t_phase = ProfileClock::now();
  std::vector<stim::simd_bits<64>> H_rows(num_detectors, stim::simd_bits<64>(max_osd_cols));
  for (size_t i = 0; i < max_osd_cols; ++i) {
    size_t c = sorted_cols[i];
    for (size_t e = graph_.var_edge_offsets[c]; e < graph_.var_edge_offsets[c + 1]; ++e) {
      H_rows[graph_.var_edges[e]][i] = true;
    }
  }
  profile_.t_matrix_build += seconds_since(t_phase);

  t_phase = ProfileClock::now();
  size_t num_solved = 0, cols_processed = 0;
  std::vector<size_t> col_to_pivot_row(num_errors, SIZE_MAX);

  for (size_t i = 0; i < max_osd_cols && num_solved < num_detectors; ++i) {
    if (osd_early_exit_ && is_residual_syndrome_satisfied(S_res, num_solved, num_detectors) &&
        (osd_weight_ == 0 || (cols_processed - num_solved) >= osd_order_)) {
      break;
    }
    cols_processed = i + 1;
    size_t pivot_r = SIZE_MAX;
    for (size_t r = num_solved; r < num_detectors; ++r) {
      if (H_rows[r][i]) {
        pivot_r = r;
        break;
      }
    }
    if (pivot_r == SIZE_MAX) continue;

    if (pivot_r != num_solved) {
      std::swap(H_rows[pivot_r], H_rows[num_solved]);
      bool temp_S = S_res[pivot_r];
      S_res[pivot_r] = S_res[num_solved];
      S_res[num_solved] = temp_S;
    }
    pivot_r = num_solved++;
    col_to_pivot_row[sorted_cols[i]] = pivot_r;

    for (size_t r = 0; r < num_detectors; ++r) {
      if (r != pivot_r && H_rows[r][i]) {
        H_rows[r] ^= H_rows[pivot_r];
        S_res[r] ^= S_res[pivot_r];
      }
    }
  }
  profile_.t_elimination += seconds_since(t_phase);
  profile_.sum_pivots_found += num_solved;
  profile_.sum_cols_processed += cols_processed;

  t_phase = ProfileClock::now();
  std::vector<bool> e_corr_best(num_errors, false);
  double best_weight = std::numeric_limits<double>::infinity(), base_hard_cost = 0.0;
  for (size_t c = 0; c < num_errors; ++c) {
    if (e_hard[c]) base_hard_cost += var_cost(c);
  }
  auto flip_delta = [&](size_t c) -> double { return e_hard[c] ? -var_cost(c) : var_cost(c); };

  std::vector<double> basis_costs(num_solved);
  std::vector<size_t> row_to_basis_col(num_solved);
  for (size_t c = 0; c < num_errors; ++c) {
    size_t r = col_to_pivot_row[c];
    if (r != SIZE_MAX) {
      basis_costs[r] = flip_delta(c);
      row_to_basis_col[r] = c;
    }
  }

  auto evaluate_solution = [&](const stim::simd_bits<64>& current_S, double perturbation_cost,
                               const std::vector<size_t>& perturbed_indices) {
    double total_w = base_hard_cost + perturbation_cost;
    for (size_t r = 0; r < num_solved; ++r) {
      if (current_S[r]) total_w += basis_costs[r];
    }
    if (total_w < best_weight) {
      best_weight = total_w;
      std::fill(e_corr_best.begin(), e_corr_best.end(), false);
      for (size_t idx : perturbed_indices) e_corr_best[idx] = true;
      for (size_t r = 0; r < num_solved; ++r) {
        if (current_S[r]) e_corr_best[row_to_basis_col[r]] = true;
      }
    }
  };

  const size_t solved_bits = std::max<size_t>(1, num_solved);
  stim::simd_bits<64> S_solved(solved_bits);
  for (size_t r = 0; r < num_solved; ++r) {
    if (S_res[r]) S_solved[r] = true;
  }
  evaluate_solution(S_solved, 0.0, {});

  if (osd_weight_ > 0 && num_solved > 0) {
    std::vector<size_t> free_cols;
    for (size_t i = 0; i < max_osd_cols && free_cols.size() < osd_order_; ++i) {
      if (col_to_pivot_row[sorted_cols[i]] == SIZE_MAX) {
        bool in_span = (i < cols_processed || num_solved == num_detectors);
        for (size_t r = num_solved; !in_span && r < num_detectors; ++r) {
          if (H_rows[r][i]) break;
          if (r + 1 == num_detectors) in_span = true;
        }
        if (in_span) free_cols.push_back(i);
      }
    }

    size_t N = std::min(osd_order_, free_cols.size());
    std::vector<stim::simd_bits<64>> F_columns(N, stim::simd_bits<64>(solved_bits));
    std::vector<double> F_costs(N);
    for (size_t k = 0; k < N; ++k) {
      size_t i = free_cols[k];
      for (size_t r = 0; r < num_solved; ++r) {
        if (H_rows[r][i]) F_columns[k][r] = true;
      }
      F_costs[k] = flip_delta(sorted_cols[i]);
    }

    for (size_t idx = 0; idx < N; ++idx) {
      stim::simd_bits<64> S_trial = S_solved;
      S_trial ^= F_columns[idx];
      evaluate_solution(S_trial, F_costs[idx], {sorted_cols[free_cols[idx]]});
    }
    if (osd_weight_ >= 2) {
      for (size_t idx = 0; idx < N; ++idx) {
        for (size_t jdx = idx + 1; jdx < N; ++jdx) {
          stim::simd_bits<64> S_trial = S_solved;
          S_trial ^= F_columns[idx];
          S_trial ^= F_columns[jdx];
          evaluate_solution(S_trial, F_costs[idx] + F_costs[jdx],
                            {sorted_cols[free_cols[idx]], sorted_cols[free_cols[jdx]]});
        }
      }
    }
  }

  if (predicted_errors != nullptr) predicted_errors->clear();
  std::vector<uint8_t> obs_result(num_observables, 0);
  for (size_t c = 0; c < num_errors; ++c) {
    if (e_hard[c] ^ e_corr_best[c]) record_flipped_var(c, obs_result);
  }
  profile_.t_solution += seconds_since(t_phase);
  profile_.t_total += seconds_since(t_osd_start);
  return obs_result;
}

}  // namespace bp
