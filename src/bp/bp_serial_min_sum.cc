#include "bp/bp_serial_min_sum.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace bp {

std::vector<BPResult> bp_serial_min_sum(
    TannerGraph& graph, const std::vector<std::vector<size_t>>& detection_events_batch,
    std::vector<LLR_INT>& posteriors_flat, size_t max_iters, float normalization_factor,
    bool stop_at_convergence, bool random_schedule, uint64_t random_seed,
    const std::vector<LLR_INT>* custom_priors_flat) {
  size_t actual_batch_size = detection_events_batch.size();
  if (actual_batch_size > BP_BATCH_SIZE) {
    throw std::invalid_argument("Provided batch size exceeds BP_BATCH_SIZE");
  }

  if (graph.syndromes.size() < graph.num_checks * BP_BATCH_SIZE) {
    graph.syndromes.resize(graph.num_checks * BP_BATCH_SIZE, 0);
  } else {
    std::fill(graph.syndromes.begin(), graph.syndromes.end(), 0);
  }
  uint8_t* __restrict__ batched_syndromes = graph.syndromes.data();
  for (size_t b = 0; b < actual_batch_size; ++b) {
    for (size_t d : detection_events_batch[b]) {
      if (d < graph.num_checks) {
        batched_syndromes[d * BP_BATCH_SIZE + b] = 1;
      }
    }
  }

  if (graph.converged_mask.size() < BP_BATCH_SIZE) {
    graph.converged_mask.resize(BP_BATCH_SIZE, 0);
  }
  uint8_t* __restrict__ active_shots = graph.converged_mask.data();
  size_t num_active = actual_batch_size;
  for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
    active_shots[b] = (b < actual_batch_size) ? 1 : 0;
  }

  std::vector<BPResult> results(actual_batch_size, {false, 0, 0});

  const LLR_UINT max_mag = std::numeric_limits<LLR_UINT>::max();
  const LLR_UINT alpha_fp = static_cast<LLR_UINT>(std::lround(normalization_factor * 256.0f));

  std::fill(graph.check_to_var_messages.begin(), graph.check_to_var_messages.end(), 0);

  if (custom_priors_flat != nullptr) {
    posteriors_flat = *custom_priors_flat;
  } else {
    for (size_t i = 0; i < graph.num_vars; ++i) {
      LLR_INT prior_val = graph.priors[i];
      size_t var_post_idx = i * BP_BATCH_SIZE;
#pragma GCC ivdep
      for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
        posteriors_flat[var_post_idx + b] = prior_val;
      }
    }
  }

  const size_t total_lanes = graph.num_vars * BP_BATCH_SIZE;
  if (graph.converged_posteriors_flat.size() < total_lanes) {
    graph.converged_posteriors_flat.resize(total_lanes);
  }
  LLR_INT* __restrict__ converged_posteriors_flat = graph.converged_posteriors_flat.data();

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

  constexpr size_t kMaxStackDeg = 64;
  alignas(64) LLR_INT q_stack[kMaxStackDeg * BP_BATCH_SIZE];
  std::vector<LLR_INT> q_heap;

  size_t iter = 0;
  for (iter = 0; iter < max_iters && num_active > 0; ++iter) {
    if (random_schedule && graph.num_checks > 1) {
      for (size_t i = graph.num_checks - 1; i > 0; --i) {
        size_t j = fast_rand() % (i + 1);
        std::swap(check_order[i], check_order[j]);
      }
    }

    // Horizontal / Layered Schedule: Iterate through check nodes
    for (size_t c_idx = 0; c_idx < graph.num_checks; ++c_idx) {
      size_t c = check_order[c_idx];
      size_t start = graph.check_edge_offsets[c];
      size_t end = graph.check_edge_offsets[c + 1];
      size_t deg = end - start;
      if (deg == 0) continue;

      size_t syn_idx = c * BP_BATCH_SIZE;

      alignas(64) LLR_UINT min1[BP_BATCH_SIZE];
      alignas(64) LLR_UINT min2[BP_BATCH_SIZE];
      alignas(64) uint32_t min1_idx[BP_BATCH_SIZE];
      alignas(64) uint8_t total_sign_prod[BP_BATCH_SIZE];

      LLR_INT* __restrict__ q_cache = q_stack;
      if (deg > kMaxStackDeg) [[unlikely]] {
        if (q_heap.size() < deg * BP_BATCH_SIZE) {
          q_heap.resize(deg * BP_BATCH_SIZE);
        }
        q_cache = q_heap.data();
      }

#pragma GCC ivdep
      for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
        min1[b] = max_mag;
        min2[b] = max_mag;
        min1_idx[b] = UINT32_MAX;
        total_sign_prod[b] = 0;
      }

      // Pass 1: Compute variable-to-check extrinsic messages Q_{c,v} = L_v - R_{c,v}
      // and find min1, min2, and total sign product (100% unmasked 32-bit AVX-512).
      for (size_t e = start; e < end; ++e) {
        const uint32_t local_e = static_cast<uint32_t>(e - start);
        const size_t v_idx = graph.check_edges[e] * BP_BATCH_SIZE;
        const size_t msg_idx = e * BP_BATCH_SIZE;
        LLR_INT* __restrict__ q_row = q_cache + local_e * BP_BATCH_SIZE;

#pragma GCC ivdep
        for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
          LLR_INT old_r = graph.check_to_var_messages[msg_idx + b];
          LLR_INT q_msg = posteriors_flat[v_idx + b] - old_r;
          q_row[b] = q_msg;

          LLR_UINT mag = static_cast<LLR_UINT>(std::abs(q_msg));
          uint8_t sign = (q_msg < 0) ? 1 : 0;
          total_sign_prod[b] ^= sign;

          if (mag < min1[b]) {
            min2[b] = min1[b];
            min1[b] = mag;
            min1_idx[b] = local_e;
          } else if (mag < min2[b]) {
            min2[b] = mag;
          }
        }
      }

      // Pass 2: Compute new check-to-variable message R'_{c,v}
      // and immediately update posterior L'_v = Q_{c,v} + R'_{c,v} (100% unmasked 32-bit AVX-512).
      for (size_t e = start; e < end; ++e) {
        const uint32_t local_e = static_cast<uint32_t>(e - start);
        const size_t v_idx = graph.check_edges[e] * BP_BATCH_SIZE;
        const size_t msg_idx = e * BP_BATCH_SIZE;
        const LLR_INT* __restrict__ q_row = q_cache + local_e * BP_BATCH_SIZE;

#pragma GCC ivdep
        for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
          LLR_INT q_msg = q_row[b];

          uint8_t q_sign = (q_msg < 0) ? 1 : 0;
          uint8_t extrinsic_sign = total_sign_prod[b] ^ q_sign;
          uint8_t final_sign = batched_syndromes[syn_idx + b] ^ extrinsic_sign;

          LLR_UINT min_mag = (local_e == min1_idx[b]) ? min2[b] : min1[b];
          LLR_UINT clamped_mag = std::min(min_mag, MAX_SAFE_LLR_INT);
          LLR_INT final_mag =
              static_cast<LLR_INT>((static_cast<uint64_t>(clamped_mag) * alpha_fp) >> LLR_FP_BITS);

          LLR_INT new_r = final_sign ? -final_mag : final_mag;
          graph.check_to_var_messages[msg_idx + b] = new_r;
          posteriors_flat[v_idx + b] = q_msg + new_r;
        }
      }
    }  // End of check loop

    // --- Convergence Check (End of iteration) ---
    if (stop_at_convergence || (iter == max_iters - 1)) {
      alignas(64) uint8_t shot_converged[BP_BATCH_SIZE];
#pragma GCC ivdep
      for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
        shot_converged[b] = active_shots[b];
      }

      const bool is_final_iter = (iter + 1 == max_iters);
      alignas(64) uint32_t unsat_count[BP_BATCH_SIZE] = {0};

      for (size_t i = 0; i < graph.num_checks; ++i) {
        if (!is_final_iter) {
          uint8_t any_candidate = 0;
#pragma GCC ivdep
          for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
            any_candidate |= shot_converged[b];
          }
          if (!any_candidate) break;
        }

        size_t start = graph.check_edge_offsets[i];
        size_t end = graph.check_edge_offsets[i + 1];
        size_t syn_idx = i * BP_BATCH_SIZE;

        alignas(64) uint8_t parity[BP_BATCH_SIZE] = {0};
        for (size_t e = start; e < end; ++e) {
          size_t v_base = graph.check_edges[e] * BP_BATCH_SIZE;
#pragma GCC ivdep
          for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
            parity[b] ^= static_cast<uint8_t>(posteriors_flat[v_base + b] < 0);
          }
        }
#pragma GCC ivdep
        for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
          uint8_t mismatch = (parity[b] != batched_syndromes[syn_idx + b]) ? 1 : 0;
          if (mismatch) {
            shot_converged[b] = 0;
          }
          if (is_final_iter) {
            unsat_count[b] += mismatch;
          }
        }
      }

      for (size_t b = 0; b < actual_batch_size; ++b) {
        if (active_shots[b] && shot_converged[b]) {
          active_shots[b] = 0;
          num_active--;
          results[b].converged = true;
          results[b].num_iters = iter + 1;
          results[b].unsat_checks = 0;
          for (size_t v = 0; v < graph.num_vars; ++v) {
            converged_posteriors_flat[v * BP_BATCH_SIZE + b] =
                posteriors_flat[v * BP_BATCH_SIZE + b];
          }
        }
      }

      if (is_final_iter) {
        for (size_t b = 0; b < actual_batch_size; ++b) {
          if (active_shots[b]) {
            results[b].unsat_checks = unsat_count[b];
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
        posteriors_flat[v * BP_BATCH_SIZE + b] =
            converged_posteriors_flat[v * BP_BATCH_SIZE + b];
      }
    }
  }

  return results;
}

}  // namespace bp
