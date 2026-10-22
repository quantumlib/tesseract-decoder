#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "bp/batched_bp_parallel_min_sum.h"

namespace bp {

template <typename T>
std::vector<BPResult> batched_bp_parallel_min_sum(
    BatchedTannerGraph<T>& graph, const std::vector<std::vector<size_t>>& detection_events_batch,
    std::vector<T>& posteriors_flat, size_t max_iters, float normalization_factor,
    bool stop_at_convergence) {
  size_t actual_batch_size = detection_events_batch.size();
  if (actual_batch_size > BP_BATCH_SIZE) {
    throw std::invalid_argument("Provided batch size exceeds BP_BATCH_SIZE");
  }

  // Set up batched syndromes.
  // For each check node, we have an array of length BP_BATCH_SIZE indicating if
  // it's a detection event. Using uint8_t instead of bool array for SIMD
  // boolean logic mapping.
  std::vector<uint8_t> batched_syndromes(graph.num_checks * BP_BATCH_SIZE, 0);
  for (size_t b = 0; b < actual_batch_size; ++b) {
    for (size_t d : detection_events_batch[b]) {
      if (d < graph.num_checks) {
        batched_syndromes[d * BP_BATCH_SIZE + b] = 1;
      }
    }
  }

  // Track which shots in the batch are still active
  std::vector<uint8_t> active_shots(BP_BATCH_SIZE, 0);
  for (size_t b = 0; b < actual_batch_size; ++b) active_shots[b] = 1;
  size_t num_active = actual_batch_size;

  std::vector<BPResult> results(actual_batch_size, {false, 0});

  using T_MAG = typename llr_traits<T>::magnitude_type;
  const T_MAG max_mag = std::numeric_limits<T_MAG>::max();
  const T_MAG alpha_fp = static_cast<T_MAG>(std::lround(normalization_factor * 256.0f));

  // Initialize all check-to-variable messages to 0.
  std::fill(graph.check_to_var_messages.begin(), graph.check_to_var_messages.end(), 0);

  size_t iter = 0;
  for (iter = 0; iter < max_iters && num_active > 0; ++iter) {
    // --- Phase 1: Variable-to-Check messages (V->C) ---
    for (size_t i = 0; i < graph.num_variables; ++i) {
      size_t start = graph.var_edge_offsets[i];
      size_t end = graph.var_edge_offsets[i + 1];
      T prior_val = graph.priors[i];

      alignas(64) T total_c2v[BP_BATCH_SIZE];
#pragma GCC ivdep
      for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
        total_c2v[b] = prior_val;
      }

      for (size_t e = start; e < end; ++e) {
        size_t msg_idx = e * BP_BATCH_SIZE;
#pragma GCC ivdep
        for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
          total_c2v[b] += graph.check_to_var_messages[msg_idx + b];
        }
      }

      for (size_t e = start; e < end; ++e) {
        size_t msg_idx = e * BP_BATCH_SIZE;
        size_t c_e = graph.var_edge_rev_indices[e];
        size_t out_idx = c_e * BP_BATCH_SIZE;
#pragma GCC ivdep
        for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
          T msg = total_c2v[b] - graph.check_to_var_messages[msg_idx + b];
          if (active_shots[b]) {
            graph.var_to_check_messages[out_idx + b] = msg;
          }
        }
      }
    }

    // --- Phase 2: Check-to-Variable messages (C->V) (Normalized Min-Sum) ---
    for (size_t i = 0; i < graph.num_checks; ++i) {
      size_t start = graph.check_edge_offsets[i];
      size_t end = graph.check_edge_offsets[i + 1];
      size_t deg = end - start;
      if (deg == 0) continue;

      size_t syn_idx = i * BP_BATCH_SIZE;

      alignas(64) T_MAG min1[BP_BATCH_SIZE];
      alignas(64) T_MAG min2[BP_BATCH_SIZE];
      alignas(64) uint32_t min1_idx[BP_BATCH_SIZE];
      alignas(64) uint8_t total_sign_prod[BP_BATCH_SIZE];

#pragma GCC ivdep
      for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
        min1[b] = max_mag;
        min2[b] = max_mag;
        min1_idx[b] = UINT32_MAX;
        total_sign_prod[b] = 0;
      }

      for (size_t e = start; e < end; ++e) {
        const uint32_t local_e = static_cast<uint32_t>(e - start);
        size_t msg_idx = e * BP_BATCH_SIZE;
#pragma GCC ivdep
        for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
          T v2c_msg = graph.var_to_check_messages[msg_idx + b];
          T_MAG mag = static_cast<T_MAG>(std::abs(v2c_msg));
          uint8_t sign = (v2c_msg < 0) ? 1 : 0;
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

      for (size_t e = start; e < end; ++e) {
        const uint32_t local_e = static_cast<uint32_t>(e - start);
        size_t msg_idx = e * BP_BATCH_SIZE;
        size_t v_e = graph.check_edge_rev_indices[e];
        size_t out_idx = v_e * BP_BATCH_SIZE;

#pragma GCC ivdep
        for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
          uint8_t v2c_sign = (graph.var_to_check_messages[msg_idx + b] < 0) ? 1 : 0;
          uint8_t extrinsic_sign = total_sign_prod[b] ^ v2c_sign;
          uint8_t final_sign = batched_syndromes[syn_idx + b] ^ extrinsic_sign;

          T_MAG mag_to_send = (local_e == min1_idx[b]) ? min2[b] : min1[b];
          T_MAG normalized_mag;
          if constexpr (std::is_integral_v<T>) {
            normalized_mag =
                static_cast<T_MAG>((static_cast<uint64_t>(mag_to_send) * alpha_fp) >> 8);
          } else {
            normalized_mag = mag_to_send * normalization_factor;
          }

          T final_mag;
          if constexpr (std::is_integral_v<T>) {
            final_mag =
                std::min(normalized_mag, static_cast<T_MAG>(std::numeric_limits<T>::max()));
          } else {
            final_mag = normalized_mag;
          }

          T new_msg = final_sign ? -final_mag : final_mag;
          if (active_shots[b]) {
            graph.check_to_var_messages[out_idx + b] = new_msg;
          }
        }
      }
    }

    // --- Posterior and Convergence Check ---
    if (stop_at_convergence || (iter == max_iters - 1)) {
      // Calculate current posteriors for active shots in flat layout
      for (size_t i = 0; i < graph.num_variables; ++i) {
        size_t start = graph.var_edge_offsets[i];
        size_t end = graph.var_edge_offsets[i + 1];
        T prior_val = graph.priors[i];
        size_t var_post_idx = i * BP_BATCH_SIZE;

        alignas(64) T post[BP_BATCH_SIZE];
#pragma GCC ivdep
        for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
          post[b] = prior_val;
        }

        for (size_t e = start; e < end; ++e) {
          size_t msg_idx = e * BP_BATCH_SIZE;
#pragma GCC ivdep
          for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
            post[b] += graph.check_to_var_messages[msg_idx + b];
          }
        }

#pragma GCC ivdep
        for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
          if (active_shots[b]) {
            posteriors_flat[var_post_idx + b] = post[b];
          }
        }
      }

      // Check convergence
      // A shot is converged if ALL checks are satisfied.
      alignas(64) uint8_t shot_converged[BP_BATCH_SIZE];
#pragma GCC ivdep
      for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
        shot_converged[b] = active_shots[b];
      }

      for (size_t i = 0; i < graph.num_checks; ++i) {
        uint8_t any_candidate = 0;
#pragma GCC ivdep
        for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
          any_candidate |= shot_converged[b];
        }
        if (!any_candidate) break;

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
          if (parity[b] != batched_syndromes[syn_idx + b]) {
            shot_converged[b] = 0;
          }
        }
      }

      for (size_t b = 0; b < actual_batch_size; ++b) {
        if (active_shots[b] && shot_converged[b]) {
          active_shots[b] = 0;
          num_active--;
          results[b].converged = true;
          results[b].num_iters = iter + 1;
        }
      }
    }
  }

  // Update results for shots that didn't converge early
  for (size_t b = 0; b < actual_batch_size; ++b) {
    if (active_shots[b]) {
      results[b].converged = false;
      results[b].num_iters = iter;  // `iter` will be `max_iters` here
      // Posteriors were already computed in the last iteration block
    }
  }

  return results;
}

}  // namespace bp