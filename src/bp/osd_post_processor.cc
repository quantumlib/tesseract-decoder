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

// 128-bit column bitset for fast L1-cache per-cluster Gauss-Jordan RREF.
struct Col128 {
  uint64_t w[2] = {0, 0};
  inline bool is_zero() const { return (w[0] | w[1]) == 0; }
  inline bool test(size_t b) const { return (w[b >> 6] & (1ULL << (b & 63))) != 0; }
  inline void set(size_t b) { w[b >> 6] |= (1ULL << (b & 63)); }
  inline void xor_with(const Col128& o) {
    w[0] ^= o.w[0];
    w[1] ^= o.w[1];
  }
  inline size_t first_bit() const {
    return (w[0] != 0) ? __builtin_ctzll(w[0]) : (64 + __builtin_ctzll(w[1]));
  }
};

}  // namespace

OsdPostProcessor::OsdPostProcessor(const TannerGraph& graph, size_t osd_order,
                                   size_t osd_weight, float osd_truncation_factor,
                                   bool osd_early_exit)
    : graph_(graph),
      osd_order_(osd_order),
      osd_weight_(osd_weight),
      osd_truncation_factor_(osd_truncation_factor),
      osd_early_exit_(osd_early_exit) {
  const size_t num_errors = graph_.num_vars, num_detectors = graph_.num_checks;
  num_det_words_ = std::max<size_t>(1, (num_detectors + 63) / 64);

  base_var_cost_.assign(num_errors, 1.0);
  if (num_detectors <= 2048) var_mask_words_.assign(num_errors * num_det_words_, 0ULL);

  for (size_t v = 0; v < num_errors; ++v) {
    double c = std::abs(llr_int_to_double(graph_.priors[v]));
    base_var_cost_[v] = (c > 0.0) ? c : 1e-3;
    if (!var_mask_words_.empty()) {
      for (size_t e = graph_.var_edge_offsets[v]; e < graph_.var_edge_offsets[v + 1]; ++e) {
        uint32_t d = graph_.var_edges[e];
        if (d < num_detectors) {
          var_mask_words_[v * num_det_words_ + (d / 64)] |= (1ULL << (d % 64));
        }
      }
    }
  }

  det_to_bit_.assign(num_detectors, -1);
  var_in_cand_.assign(num_errors, 0);

  if (num_detectors <= 2048 && !var_mask_words_.empty()) {
    std::vector<uint64_t> reach1(num_detectors * num_det_words_, 0ULL);
    for (size_t d = 0; d < num_detectors; ++d) {
      uint64_t* r1 = &reach1[d * num_det_words_];
      r1[d / 64] |= (1ULL << (d % 64));
      for (size_t e = graph_.check_edge_offsets[d]; e < graph_.check_edge_offsets[d + 1]; ++e) {
        uint32_t v = graph_.check_edges[e];
        if (v < num_errors) {
          const uint64_t* vm = &var_mask_words_[v * num_det_words_];
          for (size_t w = 0; w < num_det_words_; ++w) r1[w] |= vm[w];
        }
      }
    }
    det_reach2_words_.assign(num_detectors * num_det_words_, 0ULL);
    for (size_t d = 0; d < num_detectors; ++d) {
      uint64_t* r2 = &det_reach2_words_[d * num_det_words_];
      const uint64_t* r1 = &reach1[d * num_det_words_];
      for (size_t w = 0; w < num_det_words_; ++w) {
        uint64_t word = r1[w];
        r2[w] |= word;
        for (; word != 0; word &= (word - 1)) {
          uint32_t d1 = static_cast<uint32_t>(w * 64 + __builtin_ctzll(word));
          if (d1 < num_detectors) {
            const uint64_t* r1_d1 = &reach1[d1 * num_det_words_];
            for (size_t w2 = 0; w2 < num_det_words_; ++w2) r2[w2] |= r1_d1[w2];
          }
        }
      }
    }
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
  auto var_order_cmp = [&](uint32_t a, uint32_t b) {
    return (posteriors[a] != posteriors[b]) ? (posteriors[a] < posteriors[b])
                                            : (var_cost(a) < var_cost(b));
  };

  // Per-cluster 128-bit bitwise RREF OSD-2 solver ordered by raw signed posterior LLR ascending.
  auto solve_cluster_osd = [&](const std::vector<uint32_t>& seed_dets,
                               std::vector<size_t>& out_vars) -> bool {
    out_vars.clear();
    if (seed_dets.empty()) return true;
    if (seed_dets.size() > 128) return false;

    std::vector<uint32_t> act_dets = seed_dets, cand_vars;
    act_dets.reserve(128);
    cand_vars.reserve(160);
    const size_t seed_dets_count = act_dets.size();
    for (size_t i = 0; i < seed_dets_count; ++i) det_to_bit_[act_dets[i]] = static_cast<int32_t>(i);

    for (uint32_t d : act_dets) {
      for (size_t e = graph_.check_edge_offsets[d]; e < graph_.check_edge_offsets[d + 1]; ++e) {
        uint32_t v = graph_.check_edges[e];
        if (v < num_errors && !var_in_cand_[v]) {
          var_in_cand_[v] = 1;
          cand_vars.push_back(v);
        }
      }
    }
    std::sort(cand_vars.begin(), cand_vars.end(), var_order_cmp);

    for (uint32_t v : cand_vars) {
      for (size_t e = graph_.var_edge_offsets[v]; e < graph_.var_edge_offsets[v + 1]; ++e) {
        uint32_t d = graph_.var_edges[e];
        if (d < num_detectors && det_to_bit_[d] == -1) {
          if (act_dets.size() >= 128) {
            for (uint32_t cv : cand_vars) var_in_cand_[cv] = 0;
            for (uint32_t ad : act_dets) det_to_bit_[ad] = -1;
            return false;
          }
          det_to_bit_[d] = static_cast<int32_t>(act_dets.size());
          act_dets.push_back(d);
        }
      }
    }

    for (size_t i = seed_dets_count; i < act_dets.size() && cand_vars.size() < 160; ++i) {
      uint32_t d = act_dets[i];
      for (size_t e = graph_.check_edge_offsets[d]; e < graph_.check_edge_offsets[d + 1]; ++e) {
        uint32_t v = graph_.check_edges[e];
        if (v < num_errors && !var_in_cand_[v]) {
          bool all_in = true;
          for (size_t e2 = graph_.var_edge_offsets[v]; e2 < graph_.var_edge_offsets[v + 1]; ++e2) {
            uint32_t ed = graph_.var_edges[e2];
            if (ed < num_detectors && det_to_bit_[ed] == -1) {
              all_in = false;
              break;
            }
          }
          if (all_in) {
            var_in_cand_[v] = 1;
            cand_vars.push_back(v);
            if (cand_vars.size() >= 160) break;
          }
        }
      }
    }
    for (uint32_t v : cand_vars) var_in_cand_[v] = 0;
    std::sort(cand_vars.begin(), cand_vars.end(), var_order_cmp);

    const size_t K = act_dets.size();
    Col128 target_s, pivot_col[128], pivot_comb[128];
    for (size_t i = 0; i < seed_dets_count; ++i) target_s.set(i);
    size_t pivot_bit[128], r_count = 0, f_count = 0;
    uint32_t pivot_var_id[128];
    double pivot_cost[128];
    struct FreeVar {
      uint32_t var_id;
      double cost;
      Col128 comb;
    } free_vars[24];

    for (uint32_t v : cand_vars) {
      Col128 col;
      bool outside_det = false;
      for (size_t e = graph_.var_edge_offsets[v]; e < graph_.var_edge_offsets[v + 1]; ++e) {
        uint32_t d = graph_.var_edges[e];
        if (d < num_detectors) {
          int32_t bit = det_to_bit_[d];
          if (bit >= 0 && static_cast<size_t>(bit) < K) col.set(static_cast<size_t>(bit));
          else {
            outside_det = true;
            break;
          }
        }
      }
      if (outside_det || col.is_zero()) continue;

      Col128 rem = col, comb;
      for (size_t r = 0; r < r_count; ++r) {
        if (col.test(pivot_bit[r])) {
          rem.xor_with(pivot_col[r]);
          comb.xor_with(pivot_comb[r]);
        }
      }
      if (!rem.is_zero()) {
        if (r_count < 128) {
          size_t b = rem.first_bit(), r_new = r_count++;
          pivot_bit[r_new] = b;
          pivot_col[r_new] = rem;
          comb.set(r_new);
          pivot_comb[r_new] = comb;
          pivot_var_id[r_new] = v;
          pivot_cost[r_new] = var_cost(v);
          for (size_t r = 0; r < r_new; ++r) {
            if (pivot_col[r].test(b)) {
              pivot_col[r].xor_with(rem);
              pivot_comb[r].xor_with(comb);
            }
          }
        }
      } else if (f_count < 24) {
        free_vars[f_count++] = {v, var_cost(v), comb};
      }
    }
    for (uint32_t d : act_dets) det_to_bit_[d] = -1;

    Col128 rem_s = target_s, base_pivots;
    for (size_t r = 0; r < r_count; ++r) {
      if (target_s.test(pivot_bit[r])) {
        rem_s.xor_with(pivot_col[r]);
        base_pivots.xor_with(pivot_comb[r]);
      }
    }
    if (!rem_s.is_zero()) return false;

    auto eval_pivots_cost = [&](const Col128& p_mask) -> double {
      double c = 0.0;
      for (size_t k = 0; k < 2; ++k) {
        for (uint64_t wk = p_mask.w[k]; wk != 0; wk &= (wk - 1)) {
          c += pivot_cost[k * 64 + __builtin_ctzll(wk)];
        }
      }
      return c;
    };

    Col128 best_pivots = base_pivots;
    uint32_t best_f1 = UINT32_MAX, best_f2 = UINT32_MAX;
    double best_cost = eval_pivots_cost(base_pivots);

    if (osd_weight_ >= 1 && osd_order_ > 0) {
      const size_t f1_limit = std::min(f_count, osd_order_);
      for (size_t i = 0; i < f1_limit; ++i) {
        Col128 p_mask = base_pivots;
        p_mask.xor_with(free_vars[i].comb);
        double c = free_vars[i].cost + eval_pivots_cost(p_mask);
        if (c + 1e-6 < best_cost) {
          best_cost = c;
          best_pivots = p_mask;
          best_f1 = free_vars[i].var_id;
          best_f2 = UINT32_MAX;
        }
      }
      if (osd_weight_ >= 2) {
        const size_t f2_limit = std::min({f_count, osd_order_, static_cast<size_t>(16)});
        for (size_t i = 0; i < f2_limit; ++i) {
          if (free_vars[i].cost >= best_cost) continue;
          Col128 p_i = base_pivots;
          p_i.xor_with(free_vars[i].comb);
          for (size_t j = i + 1; j < f2_limit; ++j) {
            double c_ij = free_vars[i].cost + free_vars[j].cost;
            if (c_ij >= best_cost) continue;
            Col128 p_ij = p_i;
            p_ij.xor_with(free_vars[j].comb);
            double c = c_ij + eval_pivots_cost(p_ij);
            if (c + 1e-6 < best_cost) {
              best_cost = c;
              best_pivots = p_ij;
              best_f1 = free_vars[i].var_id;
              best_f2 = free_vars[j].var_id;
            }
          }
        }
      }
    }

    for (size_t k = 0; k < 2; ++k) {
      for (uint64_t wk = best_pivots.w[k]; wk != 0; wk &= (wk - 1)) {
        out_vars.push_back(pivot_var_id[k * 64 + __builtin_ctzll(wk)]);
      }
    }
    if (best_f1 != UINT32_MAX) out_vars.push_back(best_f1);
    if (best_f2 != UINT32_MAX) out_vars.push_back(best_f2);
    return true;
  };

  auto solve_via_clusters = [&](const std::vector<uint64_t>& syn_words,
                                std::vector<uint32_t>& io_vars) -> bool {
    if (det_reach2_words_.empty()) return false;
    std::vector<uint64_t> remaining_words = syn_words, verify_words = syn_words;
    std::vector<uint64_t> frontier_words(num_det_words_, 0ULL);
    std::vector<uint32_t> cluster_dets;
    std::vector<size_t> osd_vars;
    cluster_dets.reserve(128);

    while (true) {
      uint32_t d0 = UINT32_MAX;
      for (size_t w = 0; w < num_det_words_; ++w) {
        if (remaining_words[w] != 0) {
          d0 = static_cast<uint32_t>(w * 64 + __builtin_ctzll(remaining_words[w]));
          break;
        }
      }
      if (d0 == UINT32_MAX) break;

      std::fill(frontier_words.begin(), frontier_words.end(), 0ULL);
      cluster_dets = {d0};
      frontier_words[d0 / 64] |= (1ULL << (d0 % 64));
      remaining_words[d0 / 64] &= ~(1ULL << (d0 % 64));

      while (true) {
        uint32_t u = UINT32_MAX;
        for (size_t w = 0; w < num_det_words_; ++w) {
          if (frontier_words[w] != 0) {
            uint32_t bit = __builtin_ctzll(frontier_words[w]);
            frontier_words[w] &= ~(1ULL << bit);
            u = static_cast<uint32_t>(w * 64 + bit);
            break;
          }
        }
        if (u == UINT32_MAX) break;

        const uint64_t* reach_u = &det_reach2_words_[u * num_det_words_];
        for (size_t w = 0; w < num_det_words_; ++w) {
          uint64_t newly_reached = remaining_words[w] & reach_u[w];
          if (newly_reached != 0) {
            frontier_words[w] |= newly_reached;
            remaining_words[w] &= ~newly_reached;
            for (uint64_t tmp = newly_reached; tmp != 0; tmp &= (tmp - 1)) {
              cluster_dets.push_back(static_cast<uint32_t>(w * 64 + __builtin_ctzll(tmp)));
            }
          }
        }
      }

      std::sort(cluster_dets.begin(), cluster_dets.end());
      if (!solve_cluster_osd(cluster_dets, osd_vars)) return false;
      for (size_t v : osd_vars) {
        if (v < num_errors) {
          io_vars.push_back(static_cast<uint32_t>(v));
          const uint64_t* vm = &var_mask_words_[v * num_det_words_];
          for (size_t w = 0; w < num_det_words_; ++w) verify_words[w] ^= vm[w];
        }
      }
    }
    for (size_t w = 0; w < num_det_words_; ++w) {
      if (verify_words[w] != 0) return false;
    }
    return true;
  };

  std::vector<uint32_t> best_sparse_vars;
  double best_full_weight = std::numeric_limits<double>::infinity();

  auto try_cluster_candidate = [&](const std::vector<uint64_t>& syn_words,
                                   std::vector<uint32_t> cand_vars) {
    if (!solve_via_clusters(syn_words, cand_vars)) return;
    std::sort(cand_vars.begin(), cand_vars.end());
    size_t w = 0;
    double cost = 0.0;
    for (size_t r = 0; r < cand_vars.size();) {
      size_t r2 = r + 1;
      while (r2 < cand_vars.size() && cand_vars[r2] == cand_vars[r]) ++r2;
      if ((r2 - r) & 1) {
        cand_vars[w++] = cand_vars[r];
        cost += var_cost(cand_vars[r]);
      }
      r = r2;
    }
    cand_vars.resize(w);
    if (cost < best_full_weight) {
      best_full_weight = cost;
      best_sparse_vars = std::move(cand_vars);
    }
  };

  auto t_phase = t_osd_start;
  if (osd_weight_ > 0 && !det_reach2_words_.empty()) {
    std::vector<uint64_t> full_syn_words(num_det_words_, 0ULL);
    for (uint64_t d : detection_events) {
      if (d < num_detectors) full_syn_words[d / 64] ^= (1ULL << (d % 64));
    }

    // Candidate A: Unpeeled full-syndrome per-cluster Col128 OSD-2.
    try_cluster_candidate(full_syn_words, {});

    // Candidate B: Peeled-BP + Iterative Majority-Active Un-Peeling + Per-Cluster Solver.
    std::vector<uint64_t> s_res_words = full_syn_words;
    std::vector<uint32_t> tentative_bp_vars;
    for (size_t v = 0; v < num_errors; ++v) {
      if (posteriors[v] < 0) {
        tentative_bp_vars.push_back(static_cast<uint32_t>(v));
        const uint64_t* vm = &var_mask_words_[v * num_det_words_];
        for (size_t w = 0; w < num_det_words_; ++w) s_res_words[w] ^= vm[w];
      }
    }
    if (!tentative_bp_vars.empty()) {
      std::sort(tentative_bp_vars.begin(), tentative_bp_vars.end(),
                [&](uint32_t a, uint32_t b) { return posteriors[a] > posteriors[b]; });
      for (bool changed = true; changed;) {
        changed = false;
        for (auto it = tentative_bp_vars.begin(); it != tentative_bp_vars.end();) {
          uint32_t v = *it;
          size_t deg = graph_.var_edge_offsets[v + 1] - graph_.var_edge_offsets[v], active_cnt = 0;
          const uint64_t* vm = &var_mask_words_[v * num_det_words_];
          for (size_t w = 0; w < num_det_words_; ++w) {
            active_cnt += __builtin_popcountll(s_res_words[w] & vm[w]);
          }
          if (active_cnt * 2 > deg) {
            for (size_t w = 0; w < num_det_words_; ++w) s_res_words[w] ^= vm[w];
            it = tentative_bp_vars.erase(it);
            changed = true;
          } else {
            ++it;
          }
        }
      }
    }
    if (s_res_words != full_syn_words) {
      try_cluster_candidate(s_res_words, std::move(tentative_bp_vars));
    }
  }

  if (best_full_weight < std::numeric_limits<double>::infinity()) {
    profile_.t_elimination += seconds_since(t_phase);
    t_phase = ProfileClock::now();
    if (predicted_errors != nullptr) predicted_errors->clear();
    std::vector<uint8_t> obs_result(num_observables, 0);
    for (uint32_t c : best_sparse_vars) record_flipped_var(c, obs_result);
    profile_.t_solution += seconds_since(t_phase);
    profile_.t_total += seconds_since(t_osd_start);
    return obs_result;
  }

  // Fallback: Global stim::simd_bits<64> OSD solver (OSD-0 or when a cluster exceeds 128 detectors).
  t_phase = ProfileClock::now();
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
