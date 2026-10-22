#include "bp/osd_post_processor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>

#include "bp/check_update.h"  // For llr_int_to_double
#include "stim.h"

namespace bp {
namespace {

using ProfileClock = std::chrono::steady_clock;

inline double seconds_since(const ProfileClock::time_point& start) {
  return std::chrono::duration<double>(ProfileClock::now() - start).count();
}

// Check if residual syndrome S_res has zero bits at or below index num_solved.
// When true, the pivots found so far completely span the residual syndrome.
inline bool is_residual_syndrome_satisfied(const stim::simd_bits<64>& S_res, size_t num_solved,
                                           size_t num_detectors) {
  if (num_solved >= num_detectors) return true;
  size_t start_word = num_solved / 64;
  size_t end_word = (num_detectors + 63) / 64;
  size_t bit_offset = num_solved % 64;
  uint64_t mask = ~((1ULL << bit_offset) - 1);
  if (start_word + 1 == end_word) {
    size_t last_bits = num_detectors % 64;
    if (last_bits != 0) {
      mask &= (1ULL << last_bits) - 1;
    }
    return (S_res.u64[start_word] & mask) == 0;
  }
  if ((S_res.u64[start_word] & mask) != 0) {
    return false;
  }
  for (size_t w = start_word + 1; w + 1 < end_word; ++w) {
    if (S_res.u64[w] != 0) return false;
  }
  size_t last_word = end_word - 1;
  size_t last_bits = num_detectors % 64;
  uint64_t last_mask = (last_bits == 0) ? ~0ULL : ((1ULL << last_bits) - 1);
  return (S_res.u64[last_word] & last_mask) == 0;
}

// 128-bit column bitset for fast L1-cache per-cluster Gauss-Jordan RREF
struct Col128 {
  uint64_t lo = 0;
  uint64_t hi = 0;

  inline bool is_zero() const { return lo == 0 && hi == 0; }
  inline bool test(size_t b) const {
    return (b < 64) ? ((lo & (1ULL << b)) != 0) : ((hi & (1ULL << (b - 64))) != 0);
  }
  inline void set(size_t b) {
    if (b < 64) {
      lo |= (1ULL << b);
    } else {
      hi |= (1ULL << (b - 64));
    }
  }
  inline void xor_with(const Col128& o) {
    lo ^= o.lo;
    hi ^= o.hi;
  }
  inline size_t first_bit() const {
    if (lo != 0) return __builtin_ctzll(lo);
    return 64 + __builtin_ctzll(hi);
  }
};

}  // namespace

OsdPostProcessor::OsdPostProcessor(const TannerGraph<LLR_INT>& graph, size_t osd_order,
                                   size_t osd_weight, float osd_truncation_factor,
                                   bool osd_early_exit)
    : graph_(graph),
      osd_order_(osd_order),
      osd_weight_(osd_weight),
      osd_truncation_factor_(osd_truncation_factor),
      osd_early_exit_(osd_early_exit) {
  const size_t num_errors = graph_.variable_nodes.size();
  const size_t num_detectors = graph_.check_nodes.size();
  num_det_words_ = std::max<size_t>(1, (num_detectors + 63) / 64);

  auto splitmix64 = [](uint64_t x) -> uint64_t {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
  };

  det_sig_.resize(num_detectors);
  for (size_t d = 0; d < num_detectors; ++d) {
    uint64_t s = splitmix64(d + 1);
    det_sig_[d] = (s == 0) ? 1ULL : s;
  }

  var_sig_.assign(num_errors, 0ULL);
  base_var_cost_.assign(num_errors, 1.0);
  min_base_var_cost_ = std::numeric_limits<double>::infinity();
  if (num_detectors <= 16384) {
    var_mask_words_.assign(num_errors * num_det_words_, 0ULL);
  }

  for (size_t v = 0; v < num_errors; ++v) {
    double c = std::abs(llr_int_to_double(graph_.variable_nodes[v].prior));
    if (c <= 0.0) c = 1e-3;
    base_var_cost_[v] = c;
    min_base_var_cost_ = std::min(min_base_var_cost_, c);
    uint64_t sig = 0;
    for (size_t e = graph_.var_edge_offsets[v]; e < graph_.var_edge_offsets[v + 1]; ++e) {
      uint32_t d = graph_.var_edges[e];
      if (d < num_detectors) {
        sig ^= det_sig_[d];
        if (!var_mask_words_.empty()) {
          var_mask_words_[v * num_det_words_ + (d / 64)] |= (1ULL << (d % 64));
        }
      }
    }
    var_sig_[v] = sig;
  }
  if (!std::isfinite(min_base_var_cost_)) min_base_var_cost_ = 0.0;

  size_t lut1_cap = 16;
  while (lut1_cap < 4 * std::max<size_t>(1, num_errors)) {
    lut1_cap <<= 1;
  }
  lut1_.assign(lut1_cap, Lut1Entry{});
  lut1_mask_ = lut1_cap - 1;
  for (size_t v = 0; v < num_errors; ++v) {
    uint64_t sig = var_sig_[v];
    if (sig == 0) continue;
    size_t idx = static_cast<size_t>(sig) & lut1_mask_;
    while (lut1_[idx].var_id != UINT32_MAX) {
      if (lut1_[idx].sig == sig) {
        if (base_var_cost_[v] < base_var_cost_[lut1_[idx].var_id]) {
          lut1_[idx].var_id = static_cast<uint32_t>(v);
        }
        break;
      }
      idx = (idx + 1) & lut1_mask_;
    }
    if (lut1_[idx].var_id == UINT32_MAX) {
      lut1_[idx].sig = sig;
      lut1_[idx].var_id = static_cast<uint32_t>(v);
    }
  }

  check_sorted_offsets_ = graph_.check_edge_offsets;
  check_sorted_vars_.assign(graph_.check_edges.begin(), graph_.check_edges.end());
  for (size_t d = 0; d < num_detectors; ++d) {
    size_t start = check_sorted_offsets_[d];
    size_t end = check_sorted_offsets_[d + 1];
    std::sort(check_sorted_vars_.begin() + start, check_sorted_vars_.begin() + end,
              [&](uint32_t a, uint32_t b) {
                if (base_var_cost_[a] != base_var_cost_[b]) {
                  return base_var_cost_[a] < base_var_cost_[b];
                }
                return a < b;
              });
  }

  det_to_bit_.assign(num_detectors, -1);
  var_in_cand_.assign(num_errors, 0);

  if (num_detectors <= 16384 && !var_mask_words_.empty()) {
    det_reach1_words_.assign(num_detectors * num_det_words_, 0ULL);
    for (size_t d = 0; d < num_detectors; ++d) {
      uint64_t* r1 = &det_reach1_words_[d * num_det_words_];
      r1[d / 64] |= (1ULL << (d % 64));
      for (size_t e = graph_.check_edge_offsets[d]; e < graph_.check_edge_offsets[d + 1]; ++e) {
        uint32_t v = graph_.check_edges[e];
        if (v < num_errors) {
          const uint64_t* vm = &var_mask_words_[v * num_det_words_];
          for (size_t w = 0; w < num_det_words_; ++w) {
            r1[w] |= vm[w];
          }
        }
      }
    }
    det_reach2_words_.assign(num_detectors * num_det_words_, 0ULL);
    for (size_t d = 0; d < num_detectors; ++d) {
      uint64_t* r2 = &det_reach2_words_[d * num_det_words_];
      const uint64_t* r1 = &det_reach1_words_[d * num_det_words_];
      for (size_t w = 0; w < num_det_words_; ++w) {
        uint64_t word = r1[w];
        r2[w] |= word;
        while (word != 0) {
          uint32_t bit = __builtin_ctzll(word);
          word &= (word - 1);
          uint32_t d1 = static_cast<uint32_t>(w * 64 + bit);
          if (d1 < num_detectors) {
            const uint64_t* r1_d1 = &det_reach1_words_[d1 * num_det_words_];
            for (size_t w2 = 0; w2 < num_det_words_; ++w2) {
              r2[w2] |= r1_d1[w2];
            }
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

  size_t num_errors = graph_.variable_nodes.size();
  size_t num_detectors = graph_.check_nodes.size();

  size_t num_observables = 0;
  for (const auto& obs_list : hyperedge_observables) {
    for (int obs : obs_list) {
      if (obs >= 0) {
        num_observables = std::max(num_observables, (size_t)obs + 1);
      }
    }
  }

  // If BP converged perfectly, we can just return the hard decision observables
  // without running OSD or polluting per-OSD-invocation phase timers.
  if (bp_result.converged) {
    if (predicted_errors != nullptr) {
      predicted_errors->clear();
    }
    std::vector<uint8_t> obs_result(num_observables, 0);
    for (size_t i = 0; i < num_errors; ++i) {
      if (posteriors[i] < 0) {
        if (predicted_errors != nullptr) {
          predicted_errors->push_back(i);
        }
        for (auto& obs : hyperedge_observables[i]) {
          if (obs >= 0) obs_result[obs] ^= 1;
        }
      }
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
    // Small posterior tie-breaker when prior costs are equal
    double tie_break = 1e-4 * std::clamp(llr_int_to_double(posteriors[c]), -50.0, 50.0);
    return std::max(1e-4, prior_c + tie_break);
  };

  // Exact 1-, 2-, and 3-variable minimum-prior-cost cluster solver via SplitMix64 LUT1 + Pivot Meet-in-the-Middle
  auto try_solve_exact_123 = [&](const uint64_t* target_words, uint64_t target_sig,
                                 size_t target_popcount, std::vector<size_t>& out_vars,
                                 double& out_cost) -> bool {
    out_vars.clear();
    out_cost = std::numeric_limits<double>::infinity();
    if (target_popcount == 0 || var_mask_words_.empty()) return false;

    auto words_equal_1 = [&](uint32_t v) -> bool {
      const uint64_t* vm = &var_mask_words_[v * num_det_words_];
      for (size_t w = 0; w < num_det_words_; ++w) {
        if (vm[w] != target_words[w]) return false;
      }
      return true;
    };
    auto words_equal_2 = [&](uint32_t v1, uint32_t v2) -> bool {
      const uint64_t* vm1 = &var_mask_words_[v1 * num_det_words_];
      const uint64_t* vm2 = &var_mask_words_[v2 * num_det_words_];
      for (size_t w = 0; w < num_det_words_; ++w) {
        if ((vm1[w] ^ vm2[w]) != target_words[w]) return false;
      }
      return true;
    };
    auto words_equal_3 = [&](uint32_t v1, uint32_t v2, uint32_t v3) -> bool {
      const uint64_t* vm1 = &var_mask_words_[v1 * num_det_words_];
      const uint64_t* vm2 = &var_mask_words_[v2 * num_det_words_];
      const uint64_t* vm3 = &var_mask_words_[v3 * num_det_words_];
      for (size_t w = 0; w < num_det_words_; ++w) {
        if ((vm1[w] ^ vm2[w] ^ vm3[w]) != target_words[w]) return false;
      }
      return true;
    };

    // 1. O(1) 1-variable exact match
    if (target_popcount <= 6) {
      uint32_t v = lookup_lut1(target_sig);
      if (v != UINT32_MAX && words_equal_1(v)) {
        out_vars = {v};
        out_cost = var_cost(v);
        return true;
      }
    }

    if (target_popcount > 12) return false;

    // Find minimum-degree active detector d0 in target_words
    uint32_t d0 = UINT32_MAX;
    size_t min_deg0 = SIZE_MAX;
    for (size_t w = 0; w < num_det_words_; ++w) {
      uint64_t word = target_words[w];
      while (word != 0) {
        uint32_t bit = __builtin_ctzll(word);
        word &= (word - 1);
        uint32_t d = static_cast<uint32_t>(w * 64 + bit);
        if (d < num_detectors) {
          size_t deg = check_sorted_offsets_[d + 1] - check_sorted_offsets_[d];
          if (deg < min_deg0) {
            min_deg0 = deg;
            d0 = d;
          }
        }
      }
    }
    if (d0 == UINT32_MAX) return false;

    size_t start0 = check_sorted_offsets_[d0];
    size_t end0 = check_sorted_offsets_[d0 + 1];

    double best_cost = std::numeric_limits<double>::infinity();
    size_t best_arity = 0;
    uint32_t best_v1 = UINT32_MAX, best_v2 = UINT32_MAX, best_v3 = UINT32_MAX;

    // Also check if any variable incident to d0 is a 1-variable exact match
    for (size_t k1 = start0; k1 < end0; ++k1) {
      uint32_t v1 = check_sorted_vars_[k1];
      if (var_sig_[v1] == target_sig && words_equal_1(v1)) {
        double c1 = var_cost(v1);
        if (c1 < best_cost) {
          best_cost = c1;
          best_arity = 1;
          best_v1 = v1;
        }
      }
    }
    if (best_arity == 1) {
      out_vars = {best_v1};
      out_cost = best_cost;
      return true;
    }

    // 2. Exact 2-variable match via d0-pivot + lut1_
    if (target_popcount <= 10) {
      for (size_t k1 = start0; k1 < end0; ++k1) {
        uint32_t v1 = check_sorted_vars_[k1];
        double c1 = var_cost(v1);
        if (!has_custom_priors && c1 + min_base_var_cost_ >= best_cost) break;
        uint32_t v2 = lookup_lut1(target_sig ^ var_sig_[v1]);
        if (v2 != UINT32_MAX && v2 != v1) {
          double total_c = c1 + var_cost(v2);
          if (total_c < best_cost && words_equal_2(v1, v2)) {
            best_cost = total_c;
            best_arity = 2;
            best_v1 = v1;
            best_v2 = v2;
          }
        }
      }
    }

    // 3. Exact 3-variable match via (d0, d1)-pivot + lut1_ meet-in-the-middle
    for (size_t k1 = start0; k1 < end0; ++k1) {
      uint32_t v1 = check_sorted_vars_[k1];
      double c1 = var_cost(v1);
      if (!has_custom_priors && c1 + 2.0 * min_base_var_cost_ >= best_cost) break;
      uint64_t rem_sig1 = target_sig ^ var_sig_[v1];

      uint32_t d1 = UINT32_MAX;
      size_t min_deg1 = SIZE_MAX;
      const uint64_t* v1_mask = &var_mask_words_[v1 * num_det_words_];
      for (size_t w = 0; w < num_det_words_; ++w) {
        uint64_t rem = target_words[w] ^ v1_mask[w];
        while (rem != 0) {
          uint32_t bit = __builtin_ctzll(rem);
          rem &= (rem - 1);
          uint32_t d = static_cast<uint32_t>(w * 64 + bit);
          if (d < num_detectors) {
            size_t deg = check_sorted_offsets_[d + 1] - check_sorted_offsets_[d];
            if (deg < min_deg1) {
              min_deg1 = deg;
              d1 = d;
            }
          }
        }
      }
      if (d1 == UINT32_MAX) continue;

      size_t start1 = check_sorted_offsets_[d1];
      size_t end1 = check_sorted_offsets_[d1 + 1];
      for (size_t k2 = start1; k2 < end1; ++k2) {
        uint32_t v2 = check_sorted_vars_[k2];
        if (v2 == v1) continue;
        double c12 = c1 + var_cost(v2);
        if (!has_custom_priors && c12 + min_base_var_cost_ >= best_cost) break;
        uint32_t v3 = lookup_lut1(rem_sig1 ^ var_sig_[v2]);
        if (v3 != UINT32_MAX && v3 != v1 && v3 != v2) {
          double total_c = c12 + var_cost(v3);
          if (total_c < best_cost && words_equal_3(v1, v2, v3)) {
            best_cost = total_c;
            best_arity = 3;
            best_v1 = v1;
            best_v2 = v2;
            best_v3 = v3;
          }
        }
      }
    }

    if (best_arity == 2) {
      out_vars = {best_v1, best_v2};
      out_cost = best_cost;
      return true;
    }
    if (best_arity == 3) {
      out_vars = {best_v1, best_v2, best_v3};
      out_cost = best_cost;
      return true;
    }
    return false;
  };

  // Per-cluster 128-bit bitwise RREF OSD-2 solver ordered by raw signed posterior LLR ascending.
  auto solve_cluster_osd = [&](const std::vector<uint32_t>& seed_dets,
                               std::vector<size_t>& out_vars,
                               double& out_cost) -> bool {
    out_vars.clear();
    out_cost = std::numeric_limits<double>::infinity();
    if (seed_dets.empty()) {
      out_cost = 0.0;
      return true;
    }
    if (seed_dets.size() > 128) return false;

    std::vector<uint32_t> act_dets = seed_dets;
    act_dets.reserve(128);
    const size_t seed_dets_count = act_dets.size();
    for (size_t i = 0; i < seed_dets_count; ++i) {
      det_to_bit_[act_dets[i]] = static_cast<int32_t>(i);
    }

    std::vector<uint32_t> cand_vars;
    cand_vars.reserve(160);
    for (size_t i = 0; i < seed_dets_count; ++i) {
      uint32_t d = act_dets[i];
      for (size_t e = graph_.check_edge_offsets[d]; e < graph_.check_edge_offsets[d + 1]; ++e) {
        uint32_t v = graph_.check_edges[e];
        if (v < num_errors && !var_in_cand_[v]) {
          var_in_cand_[v] = 1;
          cand_vars.push_back(v);
        }
      }
    }

    // Sort 1-hop candidate variables by raw signed posterior LLR ascending so the most likely
    // 1-hop variables get their check neighbors into act_dets first if act_dets approaches 128.
    std::sort(cand_vars.begin(), cand_vars.end(), [&](uint32_t a, uint32_t b) {
      if (posteriors[a] != posteriors[b]) return posteriors[a] < posteriors[b];
      return var_cost(a) < var_cost(b);
    });

    // Expand act_dets up to 128 with detectors touched by 1-hop candidate variables
    for (uint32_t v : cand_vars) {
      if (act_dets.size() >= 128) break;
      for (size_t e = graph_.var_edge_offsets[v]; e < graph_.var_edge_offsets[v + 1]; ++e) {
        uint32_t d = graph_.var_edges[e];
        if (d < num_detectors && det_to_bit_[d] == -1) {
          if (act_dets.size() < 128) {
            det_to_bit_[d] = static_cast<int32_t>(act_dets.size());
            act_dets.push_back(d);
          }
        }
      }
    }

    // Include 2-hop bridge variables whose incident detectors ALL lie inside act_dets
    if (cand_vars.size() < 160) {
      for (size_t i = seed_dets_count; i < act_dets.size() && cand_vars.size() < 160; ++i) {
        uint32_t d = act_dets[i];
        for (size_t e = graph_.check_edge_offsets[d]; e < graph_.check_edge_offsets[d + 1]; ++e) {
          uint32_t v = graph_.check_edges[e];
          if (v < num_errors && !var_in_cand_[v]) {
            bool all_in = true;
            for (size_t e2 = graph_.var_edge_offsets[v]; e2 < graph_.var_edge_offsets[v + 1];
                 ++e2) {
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
    }

    for (uint32_t v : cand_vars) {
      var_in_cand_[v] = 0;
    }

    std::sort(cand_vars.begin(), cand_vars.end(), [&](uint32_t a, uint32_t b) {
      if (posteriors[a] != posteriors[b]) return posteriors[a] < posteriors[b];
      return var_cost(a) < var_cost(b);
    });

    const size_t K = act_dets.size();
    Col128 target_s;
    for (size_t i = 0; i < seed_dets_count; ++i) {
      target_s.set(i);
    }

    // Full Gauss-Jordan Elimination (RREF) in L1 registers/stack
    Col128 pivot_col[128];
    Col128 pivot_comb[128];
    size_t pivot_bit[128];
    uint32_t pivot_var_id[128];
    double pivot_cost[128];
    size_t r_count = 0;

    struct FreeVar {
      uint32_t var_id;
      double cost;
      Col128 comb;
    };
    FreeVar free_vars[24];
    size_t f_count = 0;

    for (uint32_t v : cand_vars) {
      Col128 col;
      bool outside_det = false;
      for (size_t e = graph_.var_edge_offsets[v]; e < graph_.var_edge_offsets[v + 1]; ++e) {
        uint32_t d = graph_.var_edges[e];
        if (d < num_detectors) {
          int32_t bit = det_to_bit_[d];
          if (bit >= 0 && static_cast<size_t>(bit) < K) {
            col.set(static_cast<size_t>(bit));
          } else {
            outside_det = true;
            break;
          }
        }
      }
      if (outside_det || col.is_zero()) continue;

      Col128 rem = col;
      Col128 comb;
      for (size_t r = 0; r < r_count; ++r) {
        if (col.test(pivot_bit[r])) {
          rem.xor_with(pivot_col[r]);
          comb.xor_with(pivot_comb[r]);
        }
      }

      if (!rem.is_zero()) {
        if (r_count < 128) {
          size_t b = rem.first_bit();
          size_t r_new = r_count++;
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

    for (uint32_t d : act_dets) {
      det_to_bit_[d] = -1;
    }

    Col128 rem_s = target_s;
    Col128 base_pivots;
    for (size_t r = 0; r < r_count; ++r) {
      if (target_s.test(pivot_bit[r])) {
        rem_s.xor_with(pivot_col[r]);
        base_pivots.xor_with(pivot_comb[r]);
      }
    }

    if (!rem_s.is_zero()) {
      return false;
    }

    auto eval_pivots_cost = [&](const Col128& p_mask) -> double {
      double c = 0.0;
      uint64_t w0 = p_mask.lo;
      while (w0 != 0) {
        uint32_t bit = __builtin_ctzll(w0);
        w0 &= (w0 - 1);
        c += pivot_cost[bit];
      }
      uint64_t w1 = p_mask.hi;
      while (w1 != 0) {
        uint32_t bit = __builtin_ctzll(w1);
        w1 &= (w1 - 1);
        c += pivot_cost[64 + bit];
      }
      return c;
    };

    // OSD-0
    Col128 best_pivots = base_pivots;
    uint32_t best_f1 = UINT32_MAX;
    uint32_t best_f2 = UINT32_MAX;
    double best_cost = eval_pivots_cost(base_pivots);

    // OSD-1
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

      // OSD-2 (over top 16 free variables)
      if (osd_weight_ >= 2) {
        const size_t f2_limit = std::min({f_count, osd_order_, static_cast<size_t>(16)});
        for (size_t i = 0; i < f2_limit; ++i) {
          Col128 p_i = base_pivots;
          p_i.xor_with(free_vars[i].comb);
          double c_i = free_vars[i].cost;
          if (c_i >= best_cost) continue;
          for (size_t j = i + 1; j < f2_limit; ++j) {
            double c_ij = c_i + free_vars[j].cost;
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

    uint64_t w0 = best_pivots.lo;
    while (w0 != 0) {
      uint32_t bit = __builtin_ctzll(w0);
      w0 &= (w0 - 1);
      out_vars.push_back(pivot_var_id[bit]);
    }
    uint64_t w1 = best_pivots.hi;
    while (w1 != 0) {
      uint32_t bit = __builtin_ctzll(w1);
      w1 &= (w1 - 1);
      out_vars.push_back(pivot_var_id[64 + bit]);
    }
    if (best_f1 != UINT32_MAX) out_vars.push_back(best_f1);
    if (best_f2 != UINT32_MAX) out_vars.push_back(best_f2);
    out_cost = best_cost;
    return true;
  };

  auto canonicalize_sparse_vars = [&](std::vector<uint32_t>& vars, double& out_cost) {
    std::sort(vars.begin(), vars.end());
    size_t w = 0;
    double c = 0.0;
    for (size_t r = 0; r < vars.size();) {
      size_t r2 = r + 1;
      while (r2 < vars.size() && vars[r2] == vars[r]) ++r2;
      if ((r2 - r) & 1) {
        vars[w++] = vars[r];
        c += var_cost(vars[r]);
      }
      r = r2;
    }
    vars.resize(w);
    out_cost = c;
  };

  // Decompose active syndrome words into active-defect clusters (using reach_table) and solve each
  // cluster by competing exact 1/2/3-variable matching against 128-bit Col128 RREF OSD-2.
  auto solve_via_clusters_with_reach = [&](const std::vector<uint64_t>& syn_words,
                                           const std::vector<uint64_t>& reach_table,
                                           std::vector<uint32_t>& io_vars) -> bool {
    if (reach_table.empty()) return false;
    std::vector<uint64_t> remaining_words = syn_words;
    std::vector<uint64_t> verify_words = syn_words;
    std::vector<uint64_t> cluster_words(num_det_words_, 0ULL);
    std::vector<uint64_t> frontier_words(num_det_words_, 0ULL);
    std::vector<uint32_t> cluster_dets;
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

      std::fill(cluster_words.begin(), cluster_words.end(), 0ULL);
      std::fill(frontier_words.begin(), frontier_words.end(), 0ULL);
      cluster_dets.clear();

      cluster_words[d0 / 64] |= (1ULL << (d0 % 64));
      frontier_words[d0 / 64] |= (1ULL << (d0 % 64));
      remaining_words[d0 / 64] &= ~(1ULL << (d0 % 64));
      uint64_t cluster_sig = det_sig_[d0];
      cluster_dets.push_back(d0);

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

        const uint64_t* reach_u = &reach_table[u * num_det_words_];
        for (size_t w = 0; w < num_det_words_; ++w) {
          uint64_t newly_reached = remaining_words[w] & reach_u[w];
          if (newly_reached != 0) {
            cluster_words[w] |= newly_reached;
            frontier_words[w] |= newly_reached;
            remaining_words[w] &= ~newly_reached;
            uint64_t tmp = newly_reached;
            while (tmp != 0) {
              uint32_t bit = __builtin_ctzll(tmp);
              tmp &= (tmp - 1);
              uint32_t d_new = static_cast<uint32_t>(w * 64 + bit);
              cluster_sig ^= det_sig_[d_new];
              cluster_dets.push_back(d_new);
            }
          }
        }
      }

      std::sort(cluster_dets.begin(), cluster_dets.end());
      std::vector<size_t> exact_vars;
      double exact_cost = std::numeric_limits<double>::infinity();
      bool has_exact = try_solve_exact_123(cluster_words.data(), cluster_sig,
                                           cluster_dets.size(), exact_vars, exact_cost);

      std::vector<size_t> osd_vars;
      double osd_cost = std::numeric_limits<double>::infinity();
      bool has_osd = solve_cluster_osd(cluster_dets, osd_vars, osd_cost);

      if (!has_exact && !has_osd) {
        return false;
      }
      const std::vector<size_t>& chosen =
          (has_osd && (!has_exact || osd_cost + 1e-5 < exact_cost)) ? osd_vars : exact_vars;
      for (size_t v : chosen) {
        if (v < num_errors) {
          io_vars.push_back(static_cast<uint32_t>(v));
          const uint64_t* vm = &var_mask_words_[v * num_det_words_];
          for (size_t w = 0; w < num_det_words_; ++w) {
            verify_words[w] ^= vm[w];
          }
        }
      }
    }
    for (size_t w = 0; w < num_det_words_; ++w) {
      if (verify_words[w] != 0) return false;
    }
    return true;
  };

  auto solve_via_clusters = [&](const std::vector<uint64_t>& syn_words,
                                std::vector<uint32_t>& io_vars) -> bool {
    std::vector<uint32_t> saved_vars = io_vars;
    if (solve_via_clusters_with_reach(syn_words, det_reach2_words_, io_vars)) {
      return true;
    }
    io_vars = std::move(saved_vars);
    return solve_via_clusters_with_reach(syn_words, det_reach1_words_, io_vars);
  };

  std::vector<uint32_t> best_sparse_vars;
  double best_full_weight = std::numeric_limits<double>::infinity();

  auto t_phase = t_osd_start;
  if (osd_weight_ > 0 && !det_reach2_words_.empty()) {
    std::vector<uint64_t> full_syn_words(num_det_words_, 0ULL);
    for (uint64_t d : detection_events) {
      if (d < num_detectors) {
        full_syn_words[d / 64] ^= (1ULL << (d % 64));
      }
    }

    // Candidate A: Unpeeled full-syndrome per-cluster Exact 1/2/3 + Col128 OSD-2
    {
      std::vector<uint32_t> unpeeled_vars;
      if (solve_via_clusters(full_syn_words, unpeeled_vars)) {
        double w = 0.0;
        canonicalize_sparse_vars(unpeeled_vars, w);
        if (w < best_full_weight) {
          best_full_weight = w;
          best_sparse_vars = std::move(unpeeled_vars);
        }
      }
    }

    // Candidate B: Peeled-BP + Iterative Majority-Active & Trapping-Set Un-Peeling + Per-Cluster Solver
    {
      std::vector<uint64_t> s_res_words = full_syn_words;
      std::vector<uint32_t> tentative_bp_vars;
      for (size_t v = 0; v < num_errors; ++v) {
        if (posteriors[v] < 0) {
          tentative_bp_vars.push_back(static_cast<uint32_t>(v));
          const uint64_t* vm = &var_mask_words_[v * num_det_words_];
          for (size_t w = 0; w < num_det_words_; ++w) {
            s_res_words[w] ^= vm[w];
          }
        }
      }

      if (!tentative_bp_vars.empty()) {
        // Sort weakest negative posteriors first (closest to 0)
        std::sort(tentative_bp_vars.begin(), tentative_bp_vars.end(),
                  [&](uint32_t a, uint32_t b) { return posteriors[a] > posteriors[b]; });

        // Iterative majority-active un-peeling
        bool changed = true;
        while (changed) {
          changed = false;
          for (auto it = tentative_bp_vars.begin(); it != tentative_bp_vars.end();) {
            uint32_t v = *it;
            size_t deg = graph_.var_edge_offsets[v + 1] - graph_.var_edge_offsets[v];
            size_t active_cnt = 0;
            const uint64_t* vm = &var_mask_words_[v * num_det_words_];
            for (size_t w = 0; w < num_det_words_; ++w) {
              active_cnt += __builtin_popcountll(s_res_words[w] & vm[w]);
            }
            if (active_cnt * 2 > deg) {
              for (size_t w = 0; w < num_det_words_; ++w) {
                s_res_words[w] ^= vm[w];
              }
              it = tentative_bp_vars.erase(it);
              changed = true;
            } else {
              ++it;
            }
          }
        }

        // Connected-component trapping-set un-peeling for small trapping sets (<= 8 variables)
        std::vector<uint64_t> sim_s_res = s_res_words;
        std::vector<uint32_t> rem_vars = tentative_bp_vars;
        std::vector<uint32_t> cc_unpeel;
        bool cc_changed = true;
        while (cc_changed && cc_unpeel.size() <= 8) {
          cc_changed = false;
          for (auto it = rem_vars.begin(); it != rem_vars.end();) {
            uint32_t v = *it;
            const uint64_t* vm = &var_mask_words_[v * num_det_words_];
            bool touches_active = false;
            for (size_t w = 0; w < num_det_words_; ++w) {
              if ((sim_s_res[w] & vm[w]) != 0) {
                touches_active = true;
                break;
              }
            }
            if (touches_active) {
              for (size_t w = 0; w < num_det_words_; ++w) {
                sim_s_res[w] ^= vm[w];
              }
              cc_unpeel.push_back(v);
              it = rem_vars.erase(it);
              cc_changed = true;
              if (cc_unpeel.size() > 8) break;
            } else {
              ++it;
            }
          }
        }
        if (!cc_unpeel.empty() && cc_unpeel.size() <= 8) {
          s_res_words = std::move(sim_s_res);
          tentative_bp_vars = std::move(rem_vars);
        }
      }

      if (s_res_words != full_syn_words && solve_via_clusters(s_res_words, tentative_bp_vars)) {
        double w = 0.0;
        canonicalize_sparse_vars(tentative_bp_vars, w);
        if (w < best_full_weight) {
          best_full_weight = w;
          best_sparse_vars = std::move(tentative_bp_vars);
        }
      }
    }
  }

  if (best_full_weight < std::numeric_limits<double>::infinity()) {
    profile_.t_elimination += seconds_since(t_phase);
    t_phase = ProfileClock::now();
    if (predicted_errors != nullptr) {
      predicted_errors->clear();
    }
    std::vector<uint8_t> obs_result(num_observables, 0);
    for (uint32_t c : best_sparse_vars) {
      if (predicted_errors != nullptr) {
        predicted_errors->push_back(c);
      }
      for (auto& obs : hyperedge_observables[c]) {
        if (obs >= 0) obs_result[obs] ^= 1;
      }
    }
    profile_.t_solution += seconds_since(t_phase);
    profile_.t_total += seconds_since(t_osd_start);
    return obs_result;
  }

  // Fallback: Global stim::simd_bits<64> OSD solver when a cluster exceeds 128 detectors
  // 1. Initial hard decision based on posteriors
  t_phase = ProfileClock::now();
  std::vector<bool> e_hard(num_errors, false);
  for (size_t i = 0; i < num_errors; ++i) {
    if (posteriors[i] < 0) {
      e_hard[i] = true;
    }
  }
  profile_.t_hard_decision += seconds_since(t_phase);

  // 2. Compute residual syndrome
  t_phase = ProfileClock::now();
  stim::simd_bits<64> S_res(num_detectors);
  for (auto d : detection_events) {
    if (d < num_detectors) {
      S_res[d] ^= true;
    }
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
  profile_.t_residual += seconds_since(t_phase);

  // 3. Sort columns (variable nodes/errors) by reliability (absolute value of posterior).
  // When truncation is enabled, we only need the top max_osd_cols sorted. Using
  // std::nth_element partitions the top-K columns in O(N) expected time, followed
  // by sorting only the K-element prefix in O(K log K) time (up to 7-10X faster).
  size_t max_osd_cols = num_errors;
  if (osd_truncation_factor_ > 0.0f) {
    max_osd_cols = std::min(num_errors, (size_t)(num_detectors * osd_truncation_factor_));
  }
  profile_.sum_max_osd_cols += max_osd_cols;

  t_phase = ProfileClock::now();
  std::vector<size_t> sorted_cols(num_errors);
  std::iota(sorted_cols.begin(), sorted_cols.end(), 0);
  auto cmp = [&](size_t a, size_t b) {
    return std::abs(posteriors[a]) < std::abs(posteriors[b]);
  };
  if (max_osd_cols < num_errors) {
    std::nth_element(sorted_cols.begin(), sorted_cols.begin() + max_osd_cols, sorted_cols.end(),
                     cmp);
    std::sort(sorted_cols.begin(), sorted_cols.begin() + max_osd_cols, cmp);
  } else {
    std::sort(sorted_cols.begin(), sorted_cols.end(), cmp);
  }
  profile_.t_sort += seconds_since(t_phase);

  // 4. Build parity check matrix H rows using stim::simd_bits.
  // We index the columns of H_rows by 'i' (the sorted rank index) to save memory and speed up XORs.
  t_phase = ProfileClock::now();
  std::vector<stim::simd_bits<64>> H_rows(num_detectors, stim::simd_bits<64>(max_osd_cols));
  for (size_t i = 0; i < max_osd_cols; ++i) {
    size_t c = sorted_cols[i];
    size_t start = graph_.var_edge_offsets[c];
    size_t end = graph_.var_edge_offsets[c + 1];
    for (size_t e = start; e < end; ++e) {
      uint32_t d = graph_.var_edges[e];
      H_rows[d][i] = true;
    }
  }
  profile_.t_matrix_build += seconds_since(t_phase);

  // 5. Gaussian elimination with column pivoting
  t_phase = ProfileClock::now();
  size_t num_solved = 0;
  size_t cols_processed = 0;
  std::vector<size_t> col_to_pivot_row(num_errors, SIZE_MAX);

  for (size_t i = 0; i < max_osd_cols && num_solved < num_detectors; ++i) {
    const size_t num_free_so_far = cols_processed - num_solved;
    if (osd_early_exit_ && is_residual_syndrome_satisfied(S_res, num_solved, num_detectors) &&
        (osd_weight_ == 0 || num_free_so_far >= osd_order_)) {
      break;
    }
    size_t c = sorted_cols[i];
    cols_processed = i + 1;

    size_t pivot_r = SIZE_MAX;
    for (size_t r = num_solved; r < num_detectors; ++r) {
      if (H_rows[r][i]) {
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
      if (r != pivot_r && H_rows[r][i]) {
        H_rows[r] ^= H_rows[pivot_r];
        S_res[r] ^= S_res[pivot_r];
      }
    }

    col_to_pivot_row[c] = pivot_r;
    num_solved++;
  }
  profile_.t_elimination += seconds_since(t_phase);
  profile_.sum_pivots_found += num_solved;
  profile_.sum_cols_processed += cols_processed;

  // 6. Form the final correction
  t_phase = ProfileClock::now();
  std::vector<bool> e_corr_best(num_errors, false);
  double best_weight = std::numeric_limits<double>::infinity();

  double base_hard_cost = 0.0;
  for (size_t c = 0; c < num_errors; ++c) {
    if (e_hard[c]) {
      base_hard_cost += var_cost(c);
    }
  }

  auto flip_delta = [&](size_t c) -> double {
    double w = var_cost(c);
    return e_hard[c] ? -w : w;
  };

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
      if (current_S[r]) {
        total_w += basis_costs[r];
      }
    }

    if (total_w < best_weight) {
      best_weight = total_w;
      std::fill(e_corr_best.begin(), e_corr_best.end(), false);
      for (auto idx : perturbed_indices) e_corr_best[idx] = true;
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
      size_t c = sorted_cols[i];
      if (col_to_pivot_row[c] == SIZE_MAX) {
        bool in_pivot_span = (i < cols_processed || num_solved == num_detectors);
        if (!in_pivot_span) {
          in_pivot_span = true;
          for (size_t r = num_solved; r < num_detectors; ++r) {
            if (H_rows[r][i]) {
              in_pivot_span = false;
              break;
            }
          }
        }
        if (in_pivot_span) {
          free_cols.push_back(i);
        }
      }
    }

    size_t N = std::min(osd_order_, free_cols.size());
    std::vector<size_t> F_subset(free_cols.begin(), free_cols.begin() + N);

    std::vector<stim::simd_bits<64>> F_columns;
    std::vector<double> F_costs;
    for (auto i : F_subset) {
      size_t c = sorted_cols[i];
      stim::simd_bits<64> col(solved_bits);
      for (size_t r = 0; r < num_solved; ++r) {
        if (H_rows[r][i]) col[r] = true;
      }
      F_columns.push_back(std::move(col));
      F_costs.push_back(flip_delta(c));
    }

    // OSD-1
    for (size_t idx = 0; idx < F_columns.size(); ++idx) {
      stim::simd_bits<64> S_trial = S_solved;
      S_trial ^= F_columns[idx];
      evaluate_solution(S_trial, F_costs[idx], {sorted_cols[F_subset[idx]]});
    }

    // OSD-2
    if (osd_weight_ >= 2) {
      for (size_t idx = 0; idx < F_columns.size(); ++idx) {
        for (size_t jdx = idx + 1; jdx < F_columns.size(); ++jdx) {
          stim::simd_bits<64> S_trial = S_solved;
          S_trial ^= F_columns[idx];
          S_trial ^= F_columns[jdx];
          evaluate_solution(S_trial, F_costs[idx] + F_costs[jdx],
                            {sorted_cols[F_subset[idx]], sorted_cols[F_subset[jdx]]});
        }
      }
    }
  }

  // 7. Combine hard decision with best OSD correction and calculate observables
  if (predicted_errors != nullptr) {
    predicted_errors->clear();
  }
  std::vector<uint8_t> obs_result(num_observables, 0);
  for (size_t c = 0; c < num_errors; ++c) {
    if (e_hard[c] ^ e_corr_best[c]) {
      if (predicted_errors != nullptr) {
        predicted_errors->push_back(c);
      }
      for (auto& obs : hyperedge_observables[c]) {
        if (obs >= 0) obs_result[obs] ^= 1;
      }
    }
  }

  profile_.t_solution += seconds_since(t_phase);
  profile_.t_total += seconds_since(t_osd_start);
  return obs_result;
}

}  // namespace bp
