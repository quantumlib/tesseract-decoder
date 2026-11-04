#include "bp/bp_serial_min_sum.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace bp {
namespace {

template <size_t kLanes, bool kSerialSchedule>
std::vector<BPResult> bp_min_sum_impl(
    TannerGraph& graph, const std::vector<std::vector<size_t>>& detection_events_batch,
    std::vector<LLR_INT>& posteriors_flat, size_t max_iters, float normalization_factor,
    bool stop_at_convergence, bool random_schedule, uint64_t random_seed,
    const std::vector<LLR_INT>* custom_priors_flat, int damping_num,
    std::vector<LLR_INT>* tstar_posteriors_flat, size_t start_iter, bool allow_early_stop) {
  size_t actual_batch_size = detection_events_batch.size();
  if (actual_batch_size > kLanes) {
    throw std::invalid_argument("Provided batch size exceeds kLanes");
  }

  if (graph.syndromes.size() < graph.num_checks * kLanes) {
    graph.syndromes.resize(graph.num_checks * kLanes, 0);
  } else {
    for (size_t c = 0; c < graph.num_checks; ++c) {
      size_t syn_idx = c * kLanes;
      for (size_t b = 0; b < kLanes; ++b) {
        graph.syndromes[syn_idx + b] = 0;
      }
    }
  }
  uint8_t* __restrict__ batched_syndromes = graph.syndromes.data();
  for (size_t b = 0; b < actual_batch_size; ++b) {
    for (size_t d : detection_events_batch[b]) {
      if (d < graph.num_checks) {
        batched_syndromes[d * kLanes + b] = 1;
      }
    }
  }

  alignas(64) uint8_t active_shots[kLanes];
  size_t num_active = actual_batch_size;
  for (size_t b = 0; b < kLanes; ++b) {
    active_shots[b] = (b < actual_batch_size) ? 1 : 0;
  }

  std::vector<BPResult> results(actual_batch_size, {false, 0, 0});

  const LLR_UINT max_mag = std::numeric_limits<LLR_UINT>::max();
  const LLR_UINT alpha_fp = static_cast<LLR_UINT>(std::lround(normalization_factor * 256.0f));

  const size_t total_lanes = graph.num_vars * kLanes;
  if (posteriors_flat.size() < total_lanes) {
    posteriors_flat.resize(total_lanes);
  }

  if (start_iter == 0) {
    for (size_t e = 0; e < graph.num_edges; ++e) {
      size_t msg_idx = e * kLanes;
#pragma GCC ivdep
      for (size_t b = 0; b < kLanes; ++b) {
        graph.check_to_var_messages[msg_idx + b] = 0;
      }
    }

    if (custom_priors_flat != nullptr) {
      for (size_t i = 0; i < graph.num_vars; ++i) {
        size_t var_post_idx = i * kLanes;
#pragma GCC ivdep
        for (size_t b = 0; b < kLanes; ++b) {
          posteriors_flat[var_post_idx + b] = (*custom_priors_flat)[var_post_idx + b];
        }
      }
    } else {
      for (size_t i = 0; i < graph.num_vars; ++i) {
        LLR_INT prior_val = graph.priors[i];
        size_t var_post_idx = i * kLanes;
#pragma GCC ivdep
        for (size_t b = 0; b < kLanes; ++b) {
          posteriors_flat[var_post_idx + b] = prior_val;
        }
      }
    }
  }

  if (graph.converged_posteriors_flat.size() < total_lanes) {
    graph.converged_posteriors_flat.resize(total_lanes);
  }
  LLR_INT* __restrict__ converged_posteriors_flat = graph.converged_posteriors_flat.data();

  const bool track_tstar = (tstar_posteriors_flat != nullptr);
  if (track_tstar) {
    if (tstar_posteriors_flat->size() < total_lanes) {
      tstar_posteriors_flat->resize(total_lanes);
    }
    if (start_iter == 0) {
      for (size_t i = 0; i < graph.num_vars; ++i) {
        size_t base = i * kLanes;
#pragma GCC ivdep
        for (size_t b = 0; b < kLanes; ++b) {
          (*tstar_posteriors_flat)[base + b] = posteriors_flat[base + b];
        }
      }
    }
  }
  alignas(64) uint32_t min_unsat[kLanes];
  for (size_t b = 0; b < kLanes; ++b) {
    min_unsat[b] = std::numeric_limits<uint32_t>::max();
  }
  if (start_iter > 0 && track_tstar) {
    for (size_t b = 0; b < kLanes; ++b) {
      min_unsat[b] = 0;
    }
    for (size_t i = 0; i < graph.num_checks; ++i) {
      size_t start = graph.check_edge_offsets[i];
      size_t end = graph.check_edge_offsets[i + 1];
      size_t syn_idx = i * kLanes;
      alignas(64) uint8_t parity[kLanes] = {0};
      for (size_t e = start; e < end; ++e) {
        size_t v_base = graph.check_edges[e] * kLanes;
#pragma GCC ivdep
        for (size_t b = 0; b < kLanes; ++b) {
          parity[b] ^= static_cast<uint8_t>((*tstar_posteriors_flat)[v_base + b] < 0);
        }
      }
#pragma GCC ivdep
      for (size_t b = 0; b < kLanes; ++b) {
        min_unsat[b] += (parity[b] != batched_syndromes[syn_idx + b]) ? 1u : 0u;
      }
    }
  }

  std::vector<size_t> check_order(graph.num_checks);
  for (size_t i = 0; i < graph.num_checks; ++i) {
    check_order[i] = i;
  }
  uint64_t rng_state = random_seed ? random_seed : 123456789ULL;
  auto fast_rand = [&rng_state]() -> uint32_t {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return static_cast<uint32_t>((rng_state * 0x2545F4914F6CDD1DULL) >> 32);
  };

  if (start_iter > 0 && random_schedule && graph.num_checks > 1) {
    for (size_t pre = 0; pre < start_iter; ++pre) {
      for (size_t i = graph.num_checks - 1; i > 0; --i) {
        size_t j = fast_rand() % (i + 1);
        std::swap(check_order[i], check_order[j]);
      }
    }
  }

  constexpr size_t kMaxStackDeg = 64;
  alignas(64) LLR_INT q_stack[kMaxStackDeg * kLanes];
  std::vector<LLR_INT> q_heap;

  LLR_INT* __restrict__ c2v_base = graph.check_to_var_messages.data();
  LLR_INT* __restrict__ post_base = posteriors_flat.data();

  size_t iter = start_iter;
  for (; iter < max_iters && (!allow_early_stop || num_active > 0); ++iter) {
    if (random_schedule && graph.num_checks > 1) {
      for (size_t i = graph.num_checks - 1; i > 0; --i) {
        size_t j = fast_rand() % (i + 1);
        std::swap(check_order[i], check_order[j]);
      }
    }

    // Iterate through check nodes (Layered Serial if kSerialSchedule, Parallel Flooding otherwise)
    for (size_t c_idx = 0; c_idx < graph.num_checks; ++c_idx) {
      size_t c = check_order[c_idx];
      size_t start = graph.check_edge_offsets[c];
      size_t end = graph.check_edge_offsets[c + 1];
      size_t deg = end - start;
      if (deg == 0) continue;

      size_t syn_idx = c * kLanes;

      alignas(64) LLR_UINT min1[kLanes];
      alignas(64) LLR_UINT min2[kLanes];
      alignas(64) uint32_t min1_idx[kLanes];
      alignas(64) uint32_t total_sign_prod[kLanes];

      LLR_INT* __restrict__ q_cache = q_stack;
      if (deg > kMaxStackDeg) [[unlikely]] {
        if (q_heap.size() < deg * kLanes) {
          q_heap.resize(deg * kLanes);
        }
        q_cache = q_heap.data();
      }

#pragma GCC ivdep
      for (size_t b = 0; b < kLanes; ++b) {
        min1[b] = max_mag;
        min2[b] = max_mag;
        min1_idx[b] = UINT32_MAX;
        total_sign_prod[b] = 0;
      }

      // Pass 1: Compute variable-to-check extrinsic messages Q_{c,v} = L_v - R_{c,v}
      // and find min1, min2, and total sign product.
      for (size_t e = start; e < end; ++e) {
        const uint32_t local_e = static_cast<uint32_t>(e - start);
        const LLR_INT* __restrict__ post_row = post_base + graph.check_edges[e] * kLanes;
        const LLR_INT* __restrict__ c2v_row = c2v_base + e * kLanes;
        LLR_INT* __restrict__ q_row = q_cache + local_e * kLanes;

#pragma GCC ivdep
        for (size_t b = 0; b < kLanes; ++b) {
          LLR_INT old_r = c2v_row[b];
          LLR_INT q_msg = post_row[b] - old_r;
          q_row[b] = q_msg;

          LLR_UINT mag = static_cast<LLR_UINT>(std::abs(q_msg));
          uint32_t sign = (q_msg < 0) ? 1u : 0u;
          total_sign_prod[b] ^= sign;

          bool lt1 = (mag < min1[b]);
          min2[b] = lt1 ? min1[b] : ((mag < min2[b]) ? mag : min2[b]);
          min1[b] = lt1 ? mag : min1[b];
          min1_idx[b] = lt1 ? local_e : min1_idx[b];
        }
      }

      alignas(64) LLR_INT scaled_min1[kLanes];
      alignas(64) LLR_INT scaled_min2[kLanes];
      alignas(64) uint32_t syn_xor_total[kLanes];
#pragma GCC ivdep
      for (size_t b = 0; b < kLanes; ++b) {
        LLR_UINT c1 = std::min(min1[b], MAX_SAFE_LLR_INT);
        LLR_UINT c2 = std::min(min2[b], MAX_SAFE_LLR_INT);
        scaled_min1[b] = static_cast<LLR_INT>((c1 * alpha_fp) >> LLR_FP_BITS);
        scaled_min2[b] = static_cast<LLR_INT>((c2 * alpha_fp) >> LLR_FP_BITS);
        syn_xor_total[b] =
            static_cast<uint32_t>(batched_syndromes[syn_idx + b]) ^ total_sign_prod[b];
      }

      // Pass 2: Compute new check-to-variable message R'_{c,v}
      // and (if kSerialSchedule) immediately update posterior L'_v = Q_{c,v} + R'_{c,v}.
      if (damping_num <= 0) {
        for (size_t e = start; e < end; ++e) {
          const uint32_t local_e = static_cast<uint32_t>(e - start);
          LLR_INT* __restrict__ post_row = post_base + graph.check_edges[e] * kLanes;
          LLR_INT* __restrict__ c2v_row = c2v_base + e * kLanes;
          const LLR_INT* __restrict__ q_row = q_cache + local_e * kLanes;

#pragma GCC ivdep
          for (size_t b = 0; b < kLanes; ++b) {
            LLR_INT q_msg = q_row[b];
            uint32_t q_sign = (q_msg < 0) ? 1u : 0u;
            uint32_t final_sign = syn_xor_total[b] ^ q_sign;

            LLR_INT final_mag = (local_e == min1_idx[b]) ? scaled_min2[b] : scaled_min1[b];
            LLR_INT new_r = (final_sign != 0) ? -final_mag : final_mag;
            c2v_row[b] = new_r;
            if constexpr (kSerialSchedule) {
              post_row[b] = q_msg + new_r;
            }
          }
        }
      } else {
        const LLR_INT w_new = 8 - damping_num;
        const LLR_INT w_old = damping_num;
        for (size_t e = start; e < end; ++e) {
          const uint32_t local_e = static_cast<uint32_t>(e - start);
          LLR_INT* __restrict__ post_row = post_base + graph.check_edges[e] * kLanes;
          LLR_INT* __restrict__ c2v_row = c2v_base + e * kLanes;
          const LLR_INT* __restrict__ q_row = q_cache + local_e * kLanes;

#pragma GCC ivdep
          for (size_t b = 0; b < kLanes; ++b) {
            LLR_INT q_msg = q_row[b];
            uint32_t q_sign = (q_msg < 0) ? 1u : 0u;
            uint32_t final_sign = syn_xor_total[b] ^ q_sign;

            LLR_INT final_mag = (local_e == min1_idx[b]) ? scaled_min2[b] : scaled_min1[b];
            LLR_INT raw_r = (final_sign != 0) ? -final_mag : final_mag;
            LLR_INT old_r = c2v_row[b];
            LLR_INT new_r = (w_new * raw_r + w_old * old_r) / 8;
            c2v_row[b] = new_r;
            if constexpr (kSerialSchedule) {
              post_row[b] = q_msg + new_r;
            }
          }
        }
      }
    }  // End of check loop

    if constexpr (!kSerialSchedule) {
      // Parallel flooding: update all variable posteriors at the end of the iteration.
      if (custom_priors_flat != nullptr) {
        for (size_t i = 0; i < graph.num_vars; ++i) {
          size_t var_post_idx = i * kLanes;
#pragma GCC ivdep
          for (size_t b = 0; b < kLanes; ++b) {
            posteriors_flat[var_post_idx + b] = (*custom_priors_flat)[var_post_idx + b];
          }
        }
      } else {
        for (size_t i = 0; i < graph.num_vars; ++i) {
          LLR_INT prior_val = graph.priors[i];
          size_t var_post_idx = i * kLanes;
#pragma GCC ivdep
          for (size_t b = 0; b < kLanes; ++b) {
            posteriors_flat[var_post_idx + b] = prior_val;
          }
        }
      }
      for (size_t e = 0; e < graph.num_edges; ++e) {
        const size_t v_idx = graph.check_edges[e] * kLanes;
        const size_t msg_idx = e * kLanes;
#pragma GCC ivdep
        for (size_t b = 0; b < kLanes; ++b) {
          posteriors_flat[v_idx + b] += graph.check_to_var_messages[msg_idx + b];
        }
      }
    }

    // --- Convergence & Minimum-Residual-Syndrome Check (End of iteration) ---
    if (stop_at_convergence || track_tstar || (iter == max_iters - 1)) {
      alignas(64) uint8_t shot_converged[kLanes];
#pragma GCC ivdep
      for (size_t b = 0; b < kLanes; ++b) {
        shot_converged[b] = active_shots[b];
      }

      const bool is_final_iter = (iter + 1 == max_iters);
      const bool do_tstar_this_iter = track_tstar;
      const bool need_counts = is_final_iter || do_tstar_this_iter;
      alignas(64) uint32_t unsat_count[kLanes] = {0};

      for (size_t i = 0; i < graph.num_checks; ++i) {
        if (!is_final_iter) {
          if (!do_tstar_this_iter) {
            uint8_t any_candidate = 0;
#pragma GCC ivdep
            for (size_t b = 0; b < kLanes; ++b) {
              any_candidate |= shot_converged[b];
            }
            if (!any_candidate) break;
          } else if ((i & 15) == 0 && i > 0) {
            uint8_t any_candidate = 0;
#pragma GCC ivdep
            for (size_t b = 0; b < kLanes; ++b) {
              any_candidate |= (active_shots[b] & (unsat_count[b] <= min_unsat[b]));
            }
            if (!any_candidate) break;
          }
        }

        size_t start = graph.check_edge_offsets[i];
        size_t end = graph.check_edge_offsets[i + 1];
        size_t syn_idx = i * kLanes;

        alignas(64) uint8_t parity[kLanes] = {0};
        for (size_t e = start; e < end; ++e) {
          size_t v_base = graph.check_edges[e] * kLanes;
#pragma GCC ivdep
          for (size_t b = 0; b < kLanes; ++b) {
            parity[b] ^= static_cast<uint8_t>(posteriors_flat[v_base + b] < 0);
          }
        }
#pragma GCC ivdep
        for (size_t b = 0; b < kLanes; ++b) {
          uint8_t mismatch = (parity[b] != batched_syndromes[syn_idx + b]) ? 1 : 0;
          if (mismatch) {
            shot_converged[b] = 0;
          }
          if (need_counts) {
            unsat_count[b] += mismatch;
          }
        }
      }

      alignas(64) uint8_t update_tstar[kLanes] = {0};
      bool any_update_tstar = false;

      for (size_t b = 0; b < actual_batch_size; ++b) {
        if (active_shots[b] && shot_converged[b]) {
          active_shots[b] = 0;
          num_active--;
          results[b].converged = true;
          results[b].num_iters = iter + 1;
          results[b].unsat_checks = 0;
          min_unsat[b] = 0;
          for (size_t v = 0; v < graph.num_vars; ++v) {
            converged_posteriors_flat[v * kLanes + b] =
                posteriors_flat[v * kLanes + b];
          }
        } else if (do_tstar_this_iter && active_shots[b] && unsat_count[b] < min_unsat[b]) {
          min_unsat[b] = unsat_count[b];
          update_tstar[b] = 1;
          any_update_tstar = true;
        }
      }

      if (do_tstar_this_iter && any_update_tstar) {
        LLR_INT* __restrict__ ts_ptr = tstar_posteriors_flat->data();
        const LLR_INT* __restrict__ post_ptr = posteriors_flat.data();
        for (size_t v = 0; v < graph.num_vars; ++v) {
          size_t base = v * kLanes;
#pragma GCC ivdep
          for (size_t b = 0; b < kLanes; ++b) {
            if (update_tstar[b]) {
              ts_ptr[base + b] = post_ptr[base + b];
            }
          }
        }
      }

      if (is_final_iter) {
        for (size_t b = 0; b < actual_batch_size; ++b) {
          if (active_shots[b]) {
            results[b].unsat_checks = track_tstar ? min_unsat[b] : unsat_count[b];
          }
        }
      }
    }
  }

  for (size_t b = 0; b < actual_batch_size; ++b) {
    if (active_shots[b]) {
      results[b].converged = false;
      results[b].num_iters = iter;
    } else if (results[b].converged && results[b].num_iters < iter) {
      for (size_t v = 0; v < graph.num_vars; ++v) {
        posteriors_flat[v * kLanes + b] =
            converged_posteriors_flat[v * kLanes + b];
      }
    }
    if (track_tstar && results[b].converged) {
      for (size_t v = 0; v < graph.num_vars; ++v) {
        (*tstar_posteriors_flat)[v * kLanes + b] =
            posteriors_flat[v * kLanes + b];
      }
    }
  }

  return results;
}

}  // namespace

std::vector<BPResult> bp_serial_min_sum(
    TannerGraph& graph, const std::vector<std::vector<size_t>>& detection_events_batch,
    std::vector<LLR_INT>& posteriors_flat, size_t max_iters, float normalization_factor,
    bool stop_at_convergence, bool random_schedule, uint64_t random_seed,
    const std::vector<LLR_INT>* custom_priors_flat, int damping_num,
    std::vector<LLR_INT>* tstar_posteriors_flat, bool serial_schedule, bool simd_batch,
    size_t start_iter, bool allow_early_stop) {
  (void)simd_batch;
  if (serial_schedule) {
    return bp_min_sum_impl<BP_BATCH_SIZE, true>(
        graph, detection_events_batch, posteriors_flat, max_iters, normalization_factor,
        stop_at_convergence, random_schedule, random_seed, custom_priors_flat, damping_num,
        tstar_posteriors_flat, start_iter, allow_early_stop);
  }
  return bp_min_sum_impl<BP_BATCH_SIZE, false>(
      graph, detection_events_batch, posteriors_flat, max_iters, normalization_factor,
      stop_at_convergence, random_schedule, random_seed, custom_priors_flat, damping_num,
      tstar_posteriors_flat, start_iter, allow_early_stop);
}

namespace {

inline void touch_var(TannerGraph& graph, size_t v) {
  if (graph.var_dirty_stamp[v] != graph.dirty_gen) {
    graph.var_dirty_stamp[v] = graph.dirty_gen;
    graph.dirty_vars.push_back(v);
  }
}

inline void touch_check(TannerGraph& graph, size_t c) {
  if (graph.check_dirty_stamp[c] != graph.check_dirty_gen) {
    graph.check_dirty_stamp[c] = graph.check_dirty_gen;
    graph.dirty_checks.push_back(c);
  }
}

void apply_sparse_custom_priors(
    TannerGraph& graph, const std::vector<std::pair<size_t, LLR_INT>>* sparse_custom_priors,
    LLR_INT alpha_fp) {
  if (!sparse_custom_priors || sparse_custom_priors->empty()) return;

  for (const auto& [v, new_prior] : *sparse_custom_priors) {
    if (v >= graph.num_vars) continue;
    touch_var(graph, v);
    graph.work_posteriors[v] += (new_prior - graph.work_priors[v]);
    graph.work_priors[v] = new_prior;
    const size_t v_start = graph.var_edge_offsets[v];
    const size_t v_end = graph.var_edge_offsets[v + 1];
    for (size_t ve = v_start; ve < v_end; ++ve) {
      touch_check(graph, graph.var_edges[ve]);
    }
  }

  const size_t num_reweighted_checks = graph.dirty_checks.size();
  for (size_t idx = 0; idx < num_reweighted_checks; ++idx) {
    const size_t c = graph.dirty_checks[idx];
    LLR_UINT m1 = MAX_SAFE_LLR_INT;
    LLR_UINT m2 = MAX_SAFE_LLR_INT;
    size_t argmin = 0;
    uint8_t sp = 0;
    const size_t start = graph.check_edge_offsets[c];
    const size_t end = graph.check_edge_offsets[c + 1];
    for (size_t e = start; e < end; ++e) {
      const size_t v = graph.check_edges[e];
      const LLR_INT p = graph.work_priors[v];
      if (p < 0) sp ^= 1;
      LLR_UINT mag = static_cast<LLR_UINT>(std::abs(p));
      if (mag > MAX_SAFE_LLR_INT) mag = MAX_SAFE_LLR_INT;
      if (mag < m1) {
        m2 = m1;
        m1 = mag;
        argmin = e - start;
      } else if (mag < m2) {
        m2 = mag;
      }
    }
    graph.work_min2_vac[c] = m2;
    for (size_t e = start; e < end; ++e) {
      const size_t v = graph.check_edges[e];
      const size_t local_idx = e - start;
      const LLR_UINT ext_min = (local_idx == argmin) ? m2 : m1;
      const LLR_INT scaled_mag =
          static_cast<LLR_INT>((static_cast<int64_t>(ext_min) * alpha_fp) >> LLR_FP_BITS);
      const uint8_t v_sign = (graph.work_priors[v] < 0) ? 1u : 0u;
      const uint8_t r_sign = sp ^ v_sign;
      const LLR_INT r_val = r_sign ? -scaled_mag : scaled_mag;
      const LLR_INT old_m = graph.work_m_vac[e];
      if (r_val != old_m) {
        touch_var(graph, v);
        graph.work_posteriors[v] += (r_val - old_m);
        graph.work_m_vac[e] = r_val;
        graph.work_check_to_var[e] = r_val;
      }
    }
  }
}

size_t compute_global_unsat_and_flipped(
    TannerGraph& graph, const std::vector<size_t>& unique_fired_checks,
    std::vector<size_t>& out_flipped_vars) {
  out_flipped_vars.clear();
  size_t unsat = unique_fired_checks.size();
  for (size_t c : unique_fired_checks) {
    graph.check_parity[c] = 1;
  }
  for (size_t v : graph.dirty_vars) {
    if (graph.work_posteriors[v] < 0) {
      out_flipped_vars.push_back(v);
      const size_t v_start = graph.var_edge_offsets[v];
      const size_t v_end = graph.var_edge_offsets[v + 1];
      for (size_t ve = v_start; ve < v_end; ++ve) {
        const size_t c = graph.var_edges[ve];
        if (graph.check_parity[c]) {
          graph.check_parity[c] = 0;
          unsat--;
        } else {
          graph.check_parity[c] = 1;
          unsat++;
        }
      }
    }
  }
  for (size_t c : unique_fired_checks) {
    graph.check_parity[c] = 0;
  }
  for (size_t v : out_flipped_vars) {
    const size_t v_start = graph.var_edge_offsets[v];
    const size_t v_end = graph.var_edge_offsets[v + 1];
    for (size_t ve = v_start; ve < v_end; ++ve) {
      graph.check_parity[graph.var_edges[ve]] = 0;
    }
  }
  return unsat;
}

void restore_sparse_workspace(
    TannerGraph& graph, const std::vector<size_t>& unique_fired_checks) {
  for (size_t c : unique_fired_checks) {
    graph.work_syndromes[c] = 0;
  }
  for (size_t c : graph.dirty_checks) {
    graph.work_min2_vac[c] = graph.min2_vac[c];
    const size_t start = graph.check_edge_offsets[c];
    const size_t end = graph.check_edge_offsets[c + 1];
    for (size_t e = start; e < end; ++e) {
      graph.work_m_vac[e] = graph.m_vac[e];
      graph.work_check_to_var[e] = graph.m_vac[e];
    }
  }
  for (size_t v : graph.dirty_vars) {
    graph.work_priors[v] = graph.priors[v];
    graph.work_posteriors[v] = graph.v_posterior_vac[v];
  }
  graph.dirty_vars.clear();
  graph.dirty_checks.clear();
  if (++graph.dirty_gen == 0) {
    std::fill(graph.var_dirty_stamp.begin(), graph.var_dirty_stamp.end(), 0);
    graph.dirty_gen = 1;
  }
  if (++graph.check_dirty_gen == 0) {
    std::fill(graph.check_dirty_stamp.begin(), graph.check_dirty_stamp.end(), 0);
    graph.check_dirty_gen = 1;
  }
}

}  // namespace

SparseShotBpResult bp_wavefront_min_sum_shot(
    TannerGraph& graph, const std::vector<size_t>& detection_events, size_t max_iters,
    float normalization_factor, bool stop_at_convergence,
    const std::vector<std::pair<size_t, LLR_INT>>* sparse_custom_priors,
    std::vector<LLR_INT>* out_unconverged_posteriors,
    std::vector<LLR_INT>* out_tstar_posteriors) {
  graph.ensure_vacuum_state(normalization_factor);
  const LLR_INT alpha_fp = graph.cached_alpha_fp;

  SparseShotBpResult result;
  if (detection_events.empty() && (!sparse_custom_priors || sparse_custom_priors->empty())) {
    result.bp_res = {true, 1, 0};
    return result;
  }

  apply_sparse_custom_priors(graph, sparse_custom_priors, alpha_fp);

  std::vector<size_t> unique_fired;
  unique_fired.reserve(detection_events.size());
  for (size_t c : detection_events) {
    if (c < graph.num_checks) {
      if (graph.work_syndromes[c] == 0) {
        unique_fired.push_back(c);
      }
      graph.work_syndromes[c] ^= 1;
    }
  }
  size_t write_idx = 0;
  for (size_t read_idx = 0; read_idx < unique_fired.size(); ++read_idx) {
    const size_t c = unique_fired[read_idx];
    if (graph.work_syndromes[c] != 0) {
      unique_fired[write_idx++] = c;
    }
  }
  unique_fired.resize(write_idx);

  if (unique_fired.empty()) {
    size_t unsat = compute_global_unsat_and_flipped(graph, unique_fired, result.flipped_vars);
    result.bp_res = {unsat == 0, 1, unsat};
    restore_sparse_workspace(graph, unique_fired);
    return result;
  }

  result.active_checks_visited += unique_fired.size();
  for (size_t c : unique_fired) {
    touch_check(graph, c);
    const size_t start = graph.check_edge_offsets[c];
    const size_t end = graph.check_edge_offsets[c + 1];
    for (size_t e = start; e < end; ++e) {
      const size_t v = graph.check_edges[e];
      touch_var(graph, v);
      graph.work_posteriors[v] -= graph.work_check_to_var[e];
      graph.work_check_to_var[e] = 0;
    }
  }
  for (size_t c : unique_fired) {
    const size_t start = graph.check_edge_offsets[c];
    const size_t end = graph.check_edge_offsets[c + 1];
    LLR_UINT m1 = MAX_SAFE_LLR_INT;
    LLR_UINT m2 = MAX_SAFE_LLR_INT;
    size_t min1_idx = 0;
    uint8_t sign_prod = 1u;

    for (size_t e = start; e < end; ++e) {
      const size_t v = graph.check_edges[e];
      const LLR_INT raw_q = graph.work_posteriors[v];
      const LLR_INT q_val = std::min(raw_q, graph.v_posterior_vac[v]);
      if (q_val < 0) sign_prod ^= 1;
      LLR_UINT mag = static_cast<LLR_UINT>(std::abs(q_val));
      if (mag > MAX_SAFE_LLR_INT) mag = MAX_SAFE_LLR_INT;
      if (mag < m1) {
        m2 = m1;
        m1 = mag;
        min1_idx = e - start;
      } else if (mag < m2) {
        m2 = mag;
      }
    }

    for (size_t e = start; e < end; ++e) {
      const size_t v = graph.check_edges[e];
      const size_t local_idx = e - start;
      const LLR_INT raw_q = graph.work_posteriors[v];
      const uint8_t q_sign = (raw_q < 0) ? 1u : 0u;
      const uint8_t r_sign = sign_prod ^ q_sign;
      const LLR_UINT ext_min = (local_idx == min1_idx) ? m2 : m1;
      const LLR_INT scaled_mag =
          static_cast<LLR_INT>((static_cast<int64_t>(ext_min) * alpha_fp) >> LLR_FP_BITS);
      const LLR_INT r_val = r_sign ? -scaled_mag : scaled_mag;
      graph.work_check_to_var[e] = r_val;
      graph.work_posteriors[v] = raw_q + r_val;
    }
  }

  size_t unsat = compute_global_unsat_and_flipped(graph, unique_fired, result.flipped_vars);
  if (stop_at_convergence && unsat == 0) {
    result.bp_res = {true, 1, 0};
    restore_sparse_workspace(graph, unique_fired);
    return result;
  }

  size_t min_unsat = std::numeric_limits<size_t>::max();
  std::vector<size_t> best_flipped = result.flipped_vars;
  std::vector<std::pair<size_t, LLR_INT>> best_dirty_posteriors;
  best_dirty_posteriors.reserve(graph.dirty_vars.size() * 2);

  std::fill(graph.active_checks_words.begin(), graph.active_checks_words.end(), 0ULL);
  for (size_t v : graph.dirty_vars) {
    const size_t v_start = graph.var_edge_offsets[v];
    const size_t v_end = graph.var_edge_offsets[v + 1];
    const LLR_INT vac_drop = graph.v_posterior_vac[v] - graph.work_posteriors[v];
    for (size_t ve = v_start; ve < v_end; ++ve) {
      const size_t c = graph.var_edges[ve];
      const size_t ce = graph.var_edge_to_check_edge[ve];
      const LLR_INT q_vc = graph.work_posteriors[v] - graph.work_check_to_var[ce];
      const LLR_INT thresh = static_cast<LLR_INT>(graph.work_min2_vac[c]);
      const LLR_INT abs_mv = std::abs(graph.work_m_vac[ce]);
      if (q_vc < 0 || q_vc <= thresh || vac_drop > 2 * abs_mv) {
        graph.active_checks_words[c >> 6] |= (1ULL << (c & 63));
      }
    }
  }

  size_t iter = 1;
  for (; iter < max_iters; ++iter) {
    bool any_active = false;
    for (size_t w = 0; w < graph.num_check_words; ++w) {
      if (graph.active_checks_words[w] != 0ULL) {
        any_active = true;
        break;
      }
    }
    if (!any_active) break;

    std::fill(graph.next_active_checks_words.begin(), graph.next_active_checks_words.end(), 0ULL);

    for (size_t w = 0; w < graph.num_check_words; ++w) {
      uint64_t word = graph.active_checks_words[w];
      if (word == 0ULL) continue;
      graph.active_checks_words[w] = 0ULL;

      while (word != 0ULL) {
        const int bit = __builtin_ctzll(word);
        word &= (word - 1ULL);
        const size_t c = (w << 6) + static_cast<size_t>(bit);
        if (c >= graph.num_checks) break;

        touch_check(graph, c);
        result.active_checks_visited++;

        const size_t start = graph.check_edge_offsets[c];
        const size_t end = graph.check_edge_offsets[c + 1];
        LLR_UINT m1 = MAX_SAFE_LLR_INT;
        LLR_UINT m2 = MAX_SAFE_LLR_INT;
        size_t min1_idx = 0;
        uint8_t sign_prod = graph.work_syndromes[c];

        for (size_t e = start; e < end; ++e) {
          const size_t v = graph.check_edges[e];
          const LLR_INT raw_q = graph.work_posteriors[v] - graph.work_check_to_var[e];
          const LLR_INT q_val = std::min(raw_q, graph.v_posterior_vac[v]);
          if (q_val < 0) sign_prod ^= 1;
          LLR_UINT mag = static_cast<LLR_UINT>(std::abs(q_val));
          if (mag > MAX_SAFE_LLR_INT) mag = MAX_SAFE_LLR_INT;
          if (mag < m1) {
            m2 = m1;
            m1 = mag;
            min1_idx = e - start;
          } else if (mag < m2) {
            m2 = mag;
          }
        }

        for (size_t e = start; e < end; ++e) {
          const size_t v = graph.check_edges[e];
          const size_t local_idx = e - start;
          const LLR_INT old_r = graph.work_check_to_var[e];
          const LLR_INT old_post = graph.work_posteriors[v];
          const LLR_INT raw_q = old_post - old_r;
          const uint8_t q_sign = (raw_q < 0) ? 1u : 0u;
          const uint8_t r_sign = sign_prod ^ q_sign;
          const LLR_UINT ext_min = (local_idx == min1_idx) ? m2 : m1;
          const LLR_INT scaled_mag =
              static_cast<LLR_INT>((static_cast<int64_t>(ext_min) * alpha_fp) >> LLR_FP_BITS);
          const LLR_INT r_val = r_sign ? -scaled_mag : scaled_mag;
          const LLR_INT diff = r_val - old_r;
          if (diff != 0) {
            touch_var(graph, v);
            const LLR_INT new_post = raw_q + r_val;
            graph.work_check_to_var[e] = r_val;
            graph.work_posteriors[v] = new_post;

            const size_t v_start = graph.var_edge_offsets[v];
            const size_t v_end = graph.var_edge_offsets[v + 1];
            for (size_t ve = v_start; ve < v_end; ++ve) {
              const size_t c2 = graph.var_edges[ve];
              if (c2 == c) continue;
              const size_t ce2 = graph.var_edge_to_check_edge[ve];
              const LLR_INT r_c2 = graph.work_check_to_var[ce2];
              const LLR_INT old_q_vc2 = old_post - r_c2;
              const LLR_INT new_q_vc2 = new_post - r_c2;
              const LLR_INT thresh2 = static_cast<LLR_INT>(graph.work_min2_vac[c2]);
              if (new_q_vc2 < 0 || new_q_vc2 <= thresh2 || old_q_vc2 <= thresh2) {
                graph.next_active_checks_words[c2 >> 6] |= (1ULL << (c2 & 63));
              }
            }
          }
        }
      }
    }

    unsat = compute_global_unsat_and_flipped(graph, unique_fired, result.flipped_vars);
    if ((iter >= 4 || iter + 1 == max_iters || best_dirty_posteriors.empty()) &&
        unsat <= min_unsat) {
      min_unsat = unsat;
      best_flipped = result.flipped_vars;
      best_dirty_posteriors.clear();
      for (size_t v : graph.dirty_vars) {
        best_dirty_posteriors.push_back({v, graph.work_posteriors[v]});
      }
    }

    if (stop_at_convergence && unsat == 0) {
      ++iter;
      break;
    }

    std::swap(graph.active_checks_words, graph.next_active_checks_words);
  }

  result.bp_res.converged = (min_unsat == 0 && stop_at_convergence) ? true : (unsat == 0);
  result.bp_res.num_iters = iter;
  result.bp_res.unsat_checks = min_unsat;
  if (!result.bp_res.converged) {
    result.flipped_vars = std::move(best_flipped);
    if (out_unconverged_posteriors) {
      *out_unconverged_posteriors = graph.v_posterior_vac;
      for (size_t v : graph.dirty_vars) {
        (*out_unconverged_posteriors)[v] = graph.work_posteriors[v];
      }
    }
    if (out_tstar_posteriors) {
      *out_tstar_posteriors = graph.v_posterior_vac;
      for (const auto& [v, post] : best_dirty_posteriors) {
        (*out_tstar_posteriors)[v] = post;
      }
    }
  }

  restore_sparse_workspace(graph, unique_fired);
  return result;
}

}  // namespace bp
