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

// Check if residual syndrome S_res has zero bits at or above index num_solved (unpivoted detectors).
inline bool is_residual_syndrome_satisfied(const stim::simd_bits<64>& S_res,
                                           const stim::simd_bits<64>& unpivoted_mask,
                                           size_t det_words) {
  for (size_t w = 0; w < det_words; ++w) {
    if ((S_res.u64[w] & unpivoted_mask.u64[w]) != 0) return false;
  }
  return true;
}

}  // namespace

OsdPostProcessor::OsdPostProcessor(const TannerGraph& graph, size_t osd_order,
                                   size_t osd_weight, float osd_truncation_factor,
                                   bool osd_early_exit, bool osd_col_truncation,
                                   bool osd_partial_sort)
    : graph_(graph),
      osd_order_(osd_order),
      osd_weight_(osd_weight),
      osd_truncation_factor_(osd_truncation_factor),
      osd_early_exit_(osd_early_exit),
      osd_col_truncation_(osd_col_truncation),
      osd_partial_sort_(osd_partial_sort) {
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
  std::vector<uint8_t> e_hard(num_errors, 0);
  for (size_t i = 0; i < num_errors; ++i) {
    if (posteriors[i] < 0) e_hard[i] = 1;
  }
  profile_.t_hard_decision += seconds_since(t_phase);

  t_phase = ProfileClock::now();
  const size_t det_words = (std::max<size_t>(1, num_detectors) + 63) / 64;
  const size_t det_bits = det_words * 64;
  stim::simd_bits<64> S_res(det_bits);
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

  // Construct the Interior Local Subgraph (D_local, V_interior) around the residual syndrome:
  // 1. D_0 = {d : S_res[d] == 1}
  // 2. D_1 = {d : touched by any v in N(D_0)}
  // 3. Close D_local = D_0 U D_1 under connected e_hard[v] == 1 variables
  // 4. V_interior = {v : N(v) != empty and N(v) subset of D_local}
  std::vector<uint8_t> in_D_local(num_detectors, 0);
  std::vector<uint8_t> in_V_interior(num_errors, 0);
  std::vector<size_t> unsat_dets;
  std::vector<size_t> d_queue;
  for (size_t d = 0; d < num_detectors; ++d) {
    if (S_res[d]) {
      unsat_dets.push_back(d);
      in_D_local[d] = 1;
      d_queue.push_back(d);
    }
  }

  const bool use_interior_pullin = (osd_weight_ > 0) && (osd_order_ > 0);
  size_t v_interior_size = 0;
  if (use_interior_pullin && !unsat_dets.empty()) {
    for (size_t d0 : unsat_dets) {
      for (size_t e = graph_.check_edge_offsets[d0]; e < graph_.check_edge_offsets[d0 + 1]; ++e) {
        size_t v = graph_.check_edges[e];
        for (size_t ve = graph_.var_edge_offsets[v]; ve < graph_.var_edge_offsets[v + 1]; ++ve) {
          size_t d1 = graph_.var_edges[ve];
          if (!in_D_local[d1]) {
            in_D_local[d1] = 1;
            d_queue.push_back(d1);
          }
        }
      }
    }
    std::vector<uint8_t> eh_visited(num_errors, 0);
    size_t q_head = 0;
    while (q_head < d_queue.size()) {
      size_t d = d_queue[q_head++];
      for (size_t e = graph_.check_edge_offsets[d]; e < graph_.check_edge_offsets[d + 1]; ++e) {
        size_t v = graph_.check_edges[e];
        if (e_hard[v] && !eh_visited[v]) {
          eh_visited[v] = 1;
          for (size_t ve = graph_.var_edge_offsets[v]; ve < graph_.var_edge_offsets[v + 1]; ++ve) {
            size_t d2 = graph_.var_edges[ve];
            if (!in_D_local[d2]) {
              in_D_local[d2] = 1;
              d_queue.push_back(d2);
            }
          }
        }
      }
    }

    std::vector<size_t> cand_vars;
    std::vector<uint8_t> seen_var(num_errors, 0);
    for (size_t d : d_queue) {
      for (size_t e = graph_.check_edge_offsets[d]; e < graph_.check_edge_offsets[d + 1]; ++e) {
        size_t v = graph_.check_edges[e];
        if (!seen_var[v]) {
          seen_var[v] = 1;
          cand_vars.push_back(v);
        }
      }
    }
    for (size_t v : cand_vars) {
      if (graph_.var_edge_offsets[v] == graph_.var_edge_offsets[v + 1]) continue;
      bool all_in = true;
      for (size_t ve = graph_.var_edge_offsets[v]; ve < graph_.var_edge_offsets[v + 1]; ++ve) {
        if (!in_D_local[graph_.var_edges[ve]]) {
          all_in = false;
          break;
        }
      }
      if (all_in) {
        in_V_interior[v] = 1;
        v_interior_size++;
      }
    }
  }
  profile_.t_residual += seconds_since(t_phase);

  const size_t trunc_cols =
      (osd_truncation_factor_ > 0.0f)
          ? std::min(num_errors, static_cast<size_t>(num_detectors * osd_truncation_factor_))
          : num_errors;
  const size_t max_osd_cols = osd_col_truncation_ ? trunc_cols : num_errors;
  profile_.sum_max_osd_cols += max_osd_cols;

  t_phase = ProfileClock::now();
  std::vector<size_t> sorted_cols(num_errors);
  std::iota(sorted_cols.begin(), sorted_cols.end(), 0);
  auto cmp = [&](size_t a, size_t b) {
    const LLR_UINT ma = static_cast<LLR_UINT>(std::abs(posteriors[a]));
    const LLR_UINT mb = static_cast<LLR_UINT>(std::abs(posteriors[b]));
    return (ma != mb) ? (ma < mb) : (a < b);
  };
  if (osd_partial_sort_ && max_osd_cols < num_errors) {
    std::nth_element(sorted_cols.begin(), sorted_cols.begin() + max_osd_cols, sorted_cols.end(), cmp);
    std::sort(sorted_cols.begin(), sorted_cols.begin() + max_osd_cols, cmp);
  } else {
    std::sort(sorted_cols.begin(), sorted_cols.end(), cmp);
  }

  std::vector<size_t> working_cols;
  working_cols.reserve(max_osd_cols + v_interior_size);
  if (use_interior_pullin && trunc_cols < num_errors) {
    std::vector<uint8_t> in_working(num_errors, 0);
    for (size_t i = 0; i < trunc_cols; ++i) {
      size_t v = sorted_cols[i];
      working_cols.push_back(v);
      in_working[v] = 1;
    }
    std::vector<size_t> extra_int;
    for (size_t v = 0; v < num_errors; ++v) {
      if (in_V_interior[v] && !in_working[v]) {
        extra_int.push_back(v);
        in_working[v] = 1;
      }
    }
    std::sort(extra_int.begin(), extra_int.end(), cmp);
    for (size_t v : extra_int) working_cols.push_back(v);
    if (!osd_col_truncation_) {
      for (size_t i = trunc_cols; i < num_errors; ++i) {
        size_t v = sorted_cols[i];
        if (!in_working[v]) {
          working_cols.push_back(v);
        }
      }
    }
  } else {
    working_cols.assign(sorted_cols.begin(), sorted_cols.begin() + max_osd_cols);
  }
  const size_t W_active =
      (use_interior_pullin && trunc_cols < num_errors)
          ? std::min(working_cols.size(), trunc_cols + v_interior_size)
          : std::min(working_cols.size(), trunc_cols);
  profile_.t_sort += seconds_since(t_phase);

  t_phase = ProfileClock::now();
  const size_t W = working_cols.size();
  const size_t eff_order =
      (osd_weight_ >= 2 && osd_order_ >= 30) ? (osd_order_ * 2) : osd_order_;
  const size_t max_wide_order = std::max<size_t>(eff_order, 120);

  stim::simd_bits<64> unpivoted_mask(det_bits);
  for (size_t d = 0; d < num_detectors; ++d) unpivoted_mask[d] = true;
  std::vector<stim::simd_bits<64>> prebuilt_cols;
  if (!osd_col_truncation_) {
    prebuilt_cols.resize(W, stim::simd_bits<64>(det_bits));
    for (size_t i = 0; i < W; ++i) {
      size_t c = working_cols[i];
      for (size_t e = graph_.var_edge_offsets[c]; e < graph_.var_edge_offsets[c + 1]; ++e) {
        prebuilt_cols[i][graph_.var_edges[e]] = true;
      }
    }
  }
  profile_.t_matrix_build += seconds_since(t_phase);

  t_phase = ProfileClock::now();
  size_t num_solved = 0, cols_processed = 0;
  std::vector<size_t> row_to_basis_col;
  row_to_basis_col.reserve(std::min<size_t>(num_detectors, 512));
  std::vector<size_t> pivot_det;
  pivot_det.reserve(std::min<size_t>(num_detectors, 512));
  std::vector<stim::simd_bits<64>> pivot_elim_vec;
  pivot_elim_vec.reserve(std::min<size_t>(num_detectors, 512));

  std::vector<uint8_t> col_eliminated(W, 0);
  struct FreeCandidate {
    size_t working_idx;
    std::vector<size_t> piv_rows;
  };
  std::vector<FreeCandidate> free_candidates;
  free_candidates.reserve(max_wide_order + 64);

  stim::simd_bits<64> v(det_bits);

  auto eliminate_col = [&](size_t i, bool record_free = true) {
    col_eliminated[i] = 1;
    cols_processed++;
    size_t c = working_cols[i];
    if (!osd_col_truncation_) {
      v = prebuilt_cols[i];
    } else {
      v.clear();
      for (size_t e = graph_.var_edge_offsets[c]; e < graph_.var_edge_offsets[c + 1]; ++e) {
        v[graph_.var_edges[e]] = true;
      }
    }
    for (size_t q = 0; q < num_solved; ++q) {
      if (v[pivot_det[q]]) {
        v ^= pivot_elim_vec[q];
      }
    }
    size_t d_piv = SIZE_MAX;
    for (size_t w = 0; w < det_words; ++w) {
      uint64_t m = v.u64[w] & unpivoted_mask.u64[w];
      if (m != 0) {
        d_piv = (w << 6) + __builtin_ctzll(m);
        break;
      }
    }
    if (d_piv != SIZE_MAX) {
      num_solved++;
      unpivoted_mask[d_piv] = false;
      pivot_det.push_back(d_piv);
      row_to_basis_col.push_back(c);
      v[d_piv] = false;
      if (S_res[d_piv]) {
        S_res ^= v;
      }
      pivot_elim_vec.push_back(v);
    } else if (record_free && osd_weight_ > 0) {
      std::vector<size_t> rows;
      for (size_t q = 0; q < num_solved; ++q) {
        if (v[pivot_det[q]]) rows.push_back(q);
      }
      free_candidates.push_back({i, std::move(rows)});
    }
  };

  size_t active_free_cap = eff_order;
  size_t i_sat = 0;
  for (size_t i = 0; i < W_active && num_solved < num_detectors &&
                     !is_residual_syndrome_satisfied(S_res, unpivoted_mask, det_words);
       ++i) {
    eliminate_col(i, true);
    i_sat = i + 1;
  }
  if (osd_weight_ > 0) {
    if (use_interior_pullin) {
      for (size_t i = i_sat; i < W_active; ++i) {
        if (in_V_interior[working_cols[i]]) {
          eliminate_col(i, true);
        }
      }
    }
    size_t local_osd0_faults = 0;
    double local_osd0_weight = 0.0;
    for (size_t c = 0; c < num_errors; ++c) {
      if (e_hard[c] && (!use_interior_pullin || in_V_interior[c])) {
        local_osd0_faults++;
        local_osd0_weight += var_cost(c);
      }
    }
    for (size_t r = 0; r < num_solved; ++r) {
      if (S_res[pivot_det[r]]) {
        size_t bc = row_to_basis_col[r];
        if (e_hard[bc] && (!use_interior_pullin || in_V_interior[bc])) {
          if (local_osd0_faults > 0) local_osd0_faults--;
          local_osd0_weight -= var_cost(bc);
        } else if (!e_hard[bc]) {
          local_osd0_faults++;
          local_osd0_weight += var_cost(bc);
        }
      }
    }
    if (local_osd0_faults >= 3 || local_osd0_weight >= 15.0) {
      active_free_cap = std::max<size_t>(eff_order, std::min<size_t>(eff_order * 2, 120));
    }
    const size_t elim_free_limit =
        (active_free_cap > eff_order) ? std::min<size_t>(active_free_cap * 2, max_wide_order)
                                      : eff_order;
    for (size_t i = i_sat; i < W_active; ++i) {
      if (col_eliminated[i]) continue;
      if ((cols_processed - num_solved) >= elim_free_limit) break;
      eliminate_col(i, true);
    }
  }
  const size_t num_solved_active = num_solved;
  if (!osd_early_exit_) {
    for (size_t i = 0; i < W; ++i) {
      if (!col_eliminated[i]) {
        eliminate_col(i, false);
      }
    }
  }

  profile_.t_elimination += seconds_since(t_phase);
  profile_.sum_pivots_found += num_solved;
  profile_.sum_cols_processed += cols_processed;
  num_solved = num_solved_active;

  t_phase = ProfileClock::now();
  double base_hard_cost = 0.0;
  for (size_t c = 0; c < num_errors; ++c) {
    if (e_hard[c]) base_hard_cost += var_cost(c);
  }
  auto flip_delta = [&](size_t c) -> double { return e_hard[c] ? -var_cost(c) : var_cost(c); };

  const size_t solved_words = (std::max<size_t>(1, num_solved) + 63) / 64;
  const size_t solved_bits = solved_words * 64;
  std::vector<double> basis_costs(solved_bits, 0.0);
  for (size_t r = 0; r < num_solved; ++r) {
    basis_costs[r] = flip_delta(row_to_basis_col[r]);
  }

  stim::simd_bits<64> S_solved(solved_bits);
  double osd0_piv_cost = 0.0;
  for (size_t r = 0; r < num_solved; ++r) {
    if (S_res[pivot_det[r]]) {
      S_solved[r] = true;
      osd0_piv_cost += basis_costs[r];
    }
  }
  const double osd0_weight = base_hard_cost + osd0_piv_cost;
  double best_weight = osd0_weight;

  std::vector<size_t> free_cols;
  std::vector<stim::simd_bits<64>> F_columns;
  std::vector<double> F_costs;
  std::vector<size_t> best_free_indices;

  if (osd_weight_ > 0 && num_solved > 0 && eff_order > 0) {
    std::vector<uint8_t> active_pivot(num_solved, 0);
    for (size_t r = 0; r < num_solved; ++r) {
      size_t bc = row_to_basis_col[r];
      if (S_solved[r] || (basis_costs[r] < 0.0 && (!use_interior_pullin || in_V_interior[bc]))) {
        active_pivot[r] = 1;
      }
    }

    free_cols.reserve(active_free_cap);
    F_columns.reserve(active_free_cap);
    F_costs.reserve(active_free_cap);

    for (const auto& fc : free_candidates) {
      size_t i = fc.working_idx;
      size_t c = working_cols[i];
      const bool is_int_ehard = e_hard[c] && (!use_interior_pullin || in_V_interior[c]);
      if (free_cols.size() >= active_free_cap && !is_int_ehard) continue;
      bool touches = is_int_ehard;
      for (size_t r : fc.piv_rows) {
        if (r < num_solved && active_pivot[r]) {
          touches = true;
          break;
        }
      }
      if (!touches) continue;
      free_cols.push_back(i);
      stim::simd_bits<64> f_col(solved_bits);
      for (size_t r : fc.piv_rows) {
        if (r < num_solved) f_col[r] = true;
      }
      F_columns.push_back(std::move(f_col));
      F_costs.push_back(flip_delta(c));
    }

    const size_t N = free_cols.size();
    struct ImprovingMove {
      double weight;
      uint32_t count;
      size_t f_idx[4];
    };
    std::vector<ImprovingMove> imp_moves;
    imp_moves.reserve(256);

    auto eval_piv_words = [&](const uint64_t* words) -> double {
      double piv_w = 0.0;
      for (size_t w_idx = 0; w_idx < solved_words; ++w_idx) {
        uint64_t w = words[w_idx];
        size_t base_r = w_idx << 6;
        while (w) {
          size_t b = __builtin_ctzll(w);
          piv_w += basis_costs[base_r + b];
          w &= w - 1;
        }
      }
      return piv_w;
    };

    std::vector<uint64_t> tmp_words(solved_words);
    std::vector<uint64_t> s_i_words(solved_words);
    std::vector<uint64_t> s_ij_words(solved_words);
    std::vector<double> w1_weight(N, osd0_weight);

    for (size_t idx = 0; idx < N; ++idx) {
      for (size_t w = 0; w < solved_words; ++w) {
        tmp_words[w] = S_solved.u64[w] ^ F_columns[idx].u64[w];
      }
      double total_w = base_hard_cost + F_costs[idx] + eval_piv_words(tmp_words.data());
      w1_weight[idx] = total_w;
      if (total_w < osd0_weight - 1e-6) {
        imp_moves.push_back({total_w, 1, {idx, 0, 0, 0}});
      }
      if (total_w < best_weight) {
        best_weight = total_w;
        best_free_indices = {idx};
      }
    }

    if (osd_weight_ >= 2) {
      for (size_t idx = 0; idx < N; ++idx) {
        for (size_t w = 0; w < solved_words; ++w) {
          s_i_words[w] = S_solved.u64[w] ^ F_columns[idx].u64[w];
        }
        double base_i = base_hard_cost + F_costs[idx];
        double bound_i = std::min(osd0_weight, w1_weight[idx]);
        for (size_t jdx = idx + 1; jdx < N; ++jdx) {
          for (size_t w = 0; w < solved_words; ++w) {
            tmp_words[w] = s_i_words[w] ^ F_columns[jdx].u64[w];
          }
          double total_w = base_i + F_costs[jdx] + eval_piv_words(tmp_words.data());
          if (total_w < std::min(bound_i, w1_weight[jdx]) - 1e-6) {
            imp_moves.push_back({total_w, 2, {idx, jdx, 0, 0}});
          }
          if (total_w < best_weight) {
            best_weight = total_w;
            best_free_indices = {idx, jdx};
          }
        }
      }

      if (N >= 3) {
        std::vector<size_t> w3_cols;
        w3_cols.reserve(40);
        for (size_t idx = 0; idx < N && w3_cols.size() < 40; ++idx) {
          if (F_costs[idx] < 0.0 ||
              std::abs(llr_int_to_double(posteriors[working_cols[free_cols[idx]]])) < 12.0) {
            w3_cols.push_back(idx);
          }
        }
        const size_t N3 = w3_cols.size();
        for (size_t a3 = 0; a3 < N3; ++a3) {
          size_t idx = w3_cols[a3];
          for (size_t w = 0; w < solved_words; ++w) {
            s_i_words[w] = S_solved.u64[w] ^ F_columns[idx].u64[w];
          }
          double base_i = base_hard_cost + F_costs[idx];
          double bound_i = std::min(osd0_weight, w1_weight[idx]);
          for (size_t b3 = a3 + 1; b3 < N3; ++b3) {
            size_t jdx = w3_cols[b3];
            for (size_t w = 0; w < solved_words; ++w) {
              s_ij_words[w] = s_i_words[w] ^ F_columns[jdx].u64[w];
            }
            double base_ij = base_i + F_costs[jdx];
            double w2_ij = base_ij + eval_piv_words(s_ij_words.data());
            double bound_ij = std::min({bound_i, w1_weight[jdx], w2_ij});
            for (size_t c3 = b3 + 1; c3 < N3; ++c3) {
              size_t kdx = w3_cols[c3];
              for (size_t w = 0; w < solved_words; ++w) {
                tmp_words[w] = s_ij_words[w] ^ F_columns[kdx].u64[w];
              }
              double total_w = base_ij + F_costs[kdx] + eval_piv_words(tmp_words.data());
              if (total_w < std::min(bound_ij, w1_weight[kdx]) - 1e-6) {
                imp_moves.push_back({total_w, 3, {idx, jdx, kdx, 0}});
              }
              if (total_w < best_weight) {
                best_weight = total_w;
                best_free_indices = {idx, jdx, kdx};
              }
            }
          }
        }
      }
    }

    if (!imp_moves.empty()) {
      std::sort(imp_moves.begin(), imp_moves.end(),
                [](const ImprovingMove& a, const ImprovingMove& b) { return a.weight < b.weight; });
      size_t K = std::min<size_t>(imp_moves.size(), 64);
      for (size_t a = 0; a < K; ++a) {
        std::vector<size_t> cur_set(imp_moves[a].f_idx, imp_moves[a].f_idx + imp_moves[a].count);
        double cur_best_w = imp_moves[a].weight;
        for (size_t b = 0; b < K; ++b) {
          if (a == b) continue;
          bool disjoint = true;
          for (uint32_t m = 0; m < imp_moves[b].count; ++m) {
            size_t k = imp_moves[b].f_idx[m];
            if (std::find(cur_set.begin(), cur_set.end(), k) != cur_set.end()) {
              disjoint = false;
              break;
            }
          }
          if (!disjoint) continue;
          std::vector<size_t> u_set = cur_set;
          for (uint32_t m = 0; m < imp_moves[b].count; ++m) {
            u_set.push_back(imp_moves[b].f_idx[m]);
          }
          for (size_t w = 0; w < solved_words; ++w) tmp_words[w] = S_solved.u64[w];
          double pert_cost = 0.0;
          for (size_t k : u_set) {
            pert_cost += F_costs[k];
            for (size_t w = 0; w < solved_words; ++w) tmp_words[w] ^= F_columns[k].u64[w];
          }
          double total_w = base_hard_cost + pert_cost + eval_piv_words(tmp_words.data());
          if (total_w < cur_best_w - 1e-6) {
            cur_best_w = total_w;
            cur_set = std::move(u_set);
            if (total_w < best_weight - 1e-6) {
              best_weight = total_w;
              best_free_indices = cur_set;
            }
          }
        }
      }
    }

    // 2-Pass Greedy Coordinate-Descent Refinement over all N free columns.
    if (N > 0) {
      std::vector<uint8_t> in_best_free(N, 0);
      for (size_t w = 0; w < solved_words; ++w) s_i_words[w] = S_solved.u64[w];
      double cur_free_cost = 0.0;
      for (size_t k : best_free_indices) {
        in_best_free[k] = 1;
        cur_free_cost += F_costs[k];
        for (size_t w = 0; w < solved_words; ++w) s_i_words[w] ^= F_columns[k].u64[w];
      }
      bool any_refined = false;
      for (size_t pass = 0; pass < 2; ++pass) {
        bool improved = false;
        for (size_t idx = 0; idx < N; ++idx) {
          for (size_t w = 0; w < solved_words; ++w) {
            tmp_words[w] = s_i_words[w] ^ F_columns[idx].u64[w];
          }
          double cand_fc =
              in_best_free[idx] ? (cur_free_cost - F_costs[idx]) : (cur_free_cost + F_costs[idx]);
          double cand_total = base_hard_cost + cand_fc + eval_piv_words(tmp_words.data());
          if (cand_total < best_weight - 1e-6) {
            best_weight = cand_total;
            cur_free_cost = cand_fc;
            in_best_free[idx] ^= 1;
            for (size_t w = 0; w < solved_words; ++w) s_i_words[w] = tmp_words[w];
            improved = true;
            any_refined = true;
          }
        }
        if (!improved) break;
      }
      if (any_refined) {
        best_free_indices.clear();
        for (size_t idx = 0; idx < N; ++idx) {
          if (in_best_free[idx]) best_free_indices.push_back(idx);
        }
      }
    }
  }

  stim::simd_bits<64> S_best = S_solved;
  std::vector<uint8_t> e_corr_best(num_errors, 0);
  for (size_t k : best_free_indices) {
    S_best ^= F_columns[k];
    e_corr_best[working_cols[free_cols[k]]] = 1;
  }
  for (size_t r = 0; r < num_solved; ++r) {
    if (S_best[r]) e_corr_best[row_to_basis_col[r]] = 1;
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
