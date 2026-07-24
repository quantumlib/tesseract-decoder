#include "bp/osd_post_processor.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

#include "bp/check_update.h"  // For llr_int_to_double
#include "stim.h"

namespace bp {

std::vector<uint8_t> OsdPostProcessor::process(
    const BPResult& bp_result, const std::vector<LLR_INT>& posteriors,
    const std::vector<uint64_t>& detection_events,
    const std::vector<std::vector<int>>& hyperedge_observables) {
  size_t num_errors = graph_.variable_nodes.size();
  size_t num_detectors = graph_.check_nodes.size();

  // Find max observable index to size return vector
  size_t num_observables = 0;
  for (const auto& obs_list : hyperedge_observables) {
    for (int obs : obs_list) {
      if (obs >= 0) {
        num_observables = std::max(num_observables, (size_t)obs + 1);
      }
    }
  }

  // 1. Initial hard decision based on posteriors
  std::vector<bool> e_hard(num_errors, false);
  for (size_t i = 0; i < num_errors; ++i) {
    if (posteriors[i] < 0) {
      e_hard[i] = true;
    }
  }

  // If BP converged perfectly, we can just return the hard decision observables.
  if (bp_result.converged) {
    std::vector<uint8_t> obs_result(num_observables, 0);
    for (size_t i = 0; i < num_errors; ++i) {
      if (e_hard[i]) {
        for (auto& obs : hyperedge_observables[i]) {
          if (obs >= 0) obs_result[obs] ^= 1;
        }
      }
    }
    return obs_result;
  }

  // 2. Compute residual syndrome
  stim::simd_bits<64> S_res(num_detectors);
  for (auto d : detection_events) {
    S_res[d] ^= true;
  }
  for (size_t i = 0; i < num_errors; ++i) {
    if (e_hard[i]) {
      size_t start = graph_.var_edge_offsets[i];
      size_t end = graph_.var_edge_offsets[i + 1];
      for (size_t e = start; e < end; ++e) {
        uint32_t d = graph_.var_edges[e];
        S_res[d] ^= true;
      }
    }
  }

  // 3. Sort columns (variable nodes/errors) by reliability (absolute value of posterior).
  std::vector<size_t> sorted_cols(num_errors);
  std::iota(sorted_cols.begin(), sorted_cols.end(), 0);
  std::sort(sorted_cols.begin(), sorted_cols.end(),
            [&](size_t a, size_t b) { return std::abs(posteriors[a]) < std::abs(posteriors[b]); });

  // 4. Build parity check matrix H rows using stim::simd_bits.
  std::vector<stim::simd_bits<64>> H_rows(num_detectors, stim::simd_bits<64>(num_errors));
  for (size_t c = 0; c < num_errors; ++c) {
    size_t start = graph_.var_edge_offsets[c];
    size_t end = graph_.var_edge_offsets[c + 1];
    for (size_t e = start; e < end; ++e) {
      uint32_t d = graph_.var_edges[e];
      H_rows[d][c] = true;
    }
  }

  // 5. Gaussian elimination with column pivoting
  size_t num_solved = 0;
  std::vector<size_t> col_to_pivot_row(num_errors, SIZE_MAX);

  for (size_t i = 0; i < num_errors && num_solved < num_detectors; ++i) {
    size_t c = sorted_cols[i];

    size_t pivot_r = SIZE_MAX;
    for (size_t r = num_solved; r < num_detectors; ++r) {
      if (H_rows[r][c]) {
        pivot_r = r;
        break;
      }
    }

    if (pivot_r == SIZE_MAX) {
      continue;
    }

    if (pivot_r != num_solved) {
      std::swap(H_rows[pivot_r], H_rows[num_solved]);
      bool temp_S = S_res[pivot_r];
      S_res[pivot_r] = S_res[num_solved];
      S_res[num_solved] = temp_S;
    }
    pivot_r = num_solved;

    for (size_t r = 0; r < num_detectors; ++r) {
      if (r != pivot_r && H_rows[r][c]) {
        H_rows[r] ^= H_rows[pivot_r];
        S_res[r] ^= S_res[pivot_r];
      }
    }

    col_to_pivot_row[c] = pivot_r;
    num_solved++;
  }

  // 6. Form the final correction
  std::vector<bool> e_corr_best(num_errors, false);
  double best_weight = -1.0;

  std::vector<double> basis_costs(num_solved);
  std::vector<size_t> row_to_basis_col(num_solved);
  for (size_t c = 0; c < num_errors; ++c) {
    size_t r = col_to_pivot_row[c];
    if (r != SIZE_MAX) {
      basis_costs[r] = std::abs(llr_int_to_double(posteriors[c]));
      row_to_basis_col[r] = c;
    }
  }

  auto evaluate_solution = [&](const stim::simd_bits<64>& current_S, double perturbation_cost,
                               const std::vector<size_t>& perturbed_indices) {
    double total_w = perturbation_cost;
    for (size_t r = 0; r < num_solved; ++r) {
      if (current_S[r]) {
        total_w += basis_costs[r];
      }
    }

    if (best_weight < 0 || total_w < best_weight) {
      best_weight = total_w;
      std::fill(e_corr_best.begin(), e_corr_best.end(), false);
      for (auto idx : perturbed_indices) e_corr_best[idx] = true;
      for (size_t r = 0; r < num_solved; ++r) {
        if (current_S[r]) e_corr_best[row_to_basis_col[r]] = true;
      }
    }
  };

  evaluate_solution(S_res, 0.0, {});

  if (osd_weight_ > 0) {
    std::vector<size_t> free_cols;
    for (auto c : sorted_cols) {
      if (col_to_pivot_row[c] == SIZE_MAX) {
        free_cols.push_back(c);
      }
    }

    size_t N = std::min(osd_order_, free_cols.size());
    std::vector<size_t> F_subset(free_cols.begin(), free_cols.begin() + N);

    std::vector<stim::simd_bits<64>> F_columns;
    std::vector<double> F_costs;
    for (auto c : F_subset) {
      stim::simd_bits<64> col(num_solved);
      for (size_t r = 0; r < num_solved; ++r) {
        if (H_rows[r][c]) col[r] = true;
      }
      F_columns.push_back(std::move(col));
      F_costs.push_back(std::abs(llr_int_to_double(posteriors[c])));
    }

    // OSD-1
    for (size_t i = 0; i < F_columns.size(); ++i) {
      stim::simd_bits<64> S_trial = S_res;
      S_trial ^= F_columns[i];
      evaluate_solution(S_trial, F_costs[i], {F_subset[i]});
    }

    // OSD-2
    if (osd_weight_ >= 2) {
      for (size_t i = 0; i < F_columns.size(); ++i) {
        for (size_t j = i + 1; j < F_columns.size(); ++j) {
          stim::simd_bits<64> S_trial = S_res;
          S_trial ^= F_columns[i];
          S_trial ^= F_columns[j];
          evaluate_solution(S_trial, F_costs[i] + F_costs[j], {F_subset[i], F_subset[j]});
        }
      }
    }
  }

  // 7. Combine hard decision with best OSD correction and calculate observables
  std::vector<uint8_t> obs_result(num_observables, 0);
  for (size_t c = 0; c < num_errors; ++c) {
    if (e_hard[c] ^ e_corr_best[c]) {
      for (auto& obs : hyperedge_observables[c]) {
        if (obs >= 0) obs_result[obs] ^= 1;
      }
    }
  }

  return obs_result;
}

}  // namespace bp
