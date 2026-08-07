#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "bp/batched_bp_serial_min_sum.h"

namespace bp {

template <typename T>
std::vector<BPResult> batched_bp_serial_min_sum(
    BatchedTannerGraph<T>& graph, const std::vector<std::vector<size_t>>& detection_events_batch,
    std::vector<T>& posteriors_flat, size_t max_iters, float normalization_factor,
    bool stop_at_convergence) {
  size_t actual_batch_size = detection_events_batch.size();
  if (actual_batch_size > BP_BATCH_SIZE) {
    throw std::invalid_argument("Provided batch size exceeds BP_BATCH_SIZE");
  }

  // Set up batched syndromes.
  std::vector<uint8_t> batched_syndromes(graph.num_checks * BP_BATCH_SIZE, 0);
  for (size_t b = 0; b < actual_batch_size; ++b) {
    for (size_t d : detection_events_batch[b]) {
      batched_syndromes[d * BP_BATCH_SIZE + b] = 1;
    }
  }

  // Track which shots in the batch are still active
  std::vector<uint8_t> active_shots(BP_BATCH_SIZE, 0);
  for (size_t b = 0; b < actual_batch_size; ++b) active_shots[b] = 1;
  size_t num_active = actual_batch_size;

  std::vector<BPResult> results(actual_batch_size, {false, 0});

  using T_MAG = typename llr_traits<T>::magnitude_type;
  const T_MAG max_mag = std::numeric_limits<T_MAG>::max();

  // Initialize check_to_var messages to 0
  std::fill(graph.check_to_var_messages.begin(), graph.check_to_var_messages.end(), 0);

  // Initialize posteriors_flat to priors
  for (size_t i = 0; i < graph.num_variables; ++i) {
    T prior_val = graph.priors[i];
    size_t var_post_idx = i * BP_BATCH_SIZE;
#pragma GCC ivdep
    for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
      posteriors_flat[var_post_idx + b] = prior_val;
    }
  }

  size_t iter = 0;
  for (iter = 0; iter < max_iters && num_active > 0; ++iter) {
    // Horizontal / Layered Schedule: Iterate through check nodes
    for (size_t c = 0; c < graph.num_checks; ++c) {
      size_t start = graph.check_edge_offsets[c];
      size_t end = graph.check_edge_offsets[c + 1];
      size_t deg = end - start;
      if (deg == 0) continue;

      size_t syn_idx = c * BP_BATCH_SIZE;

      T_MAG min1[BP_BATCH_SIZE];
      T_MAG min2[BP_BATCH_SIZE];
      size_t min1_idx[BP_BATCH_SIZE];
      uint8_t total_sign_prod[BP_BATCH_SIZE];

#pragma GCC ivdep
      for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
        min1[b] = max_mag;
        min2[b] = max_mag;
        min1_idx[b] = SIZE_MAX;
        total_sign_prod[b] = 0;
      }

      // Pass 1: Compute variable-to-check extrinsic messages Q_{c,v} = L_v - R_{c,v}
      // and find min1, min2, and total sign product.
      for (size_t e = start; e < end; ++e) {
        size_t v = graph.check_edges[e];
        size_t v_idx = v * BP_BATCH_SIZE;
        size_t msg_idx = e * BP_BATCH_SIZE;

#pragma GCC ivdep
        for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
          if (!active_shots[b]) continue;

          T old_r = graph.check_to_var_messages[msg_idx + b];
          T q_msg = posteriors_flat[v_idx + b] - old_r;

          T_MAG mag = (T_MAG)std::abs(q_msg);
          uint8_t sign = (q_msg < 0) ? 1 : 0;
          total_sign_prod[b] ^= sign;

          if (mag < min1[b]) {
            min2[b] = min1[b];
            min1[b] = mag;
            min1_idx[b] = e;
          } else if (mag < min2[b]) {
            min2[b] = mag;
          }
        }
      }

      // Pass 2: Compute new check-to-variable message R'_{c,v}
      // and immediately update posterior L'_v = Q_{c,v} + R'_{c,v} = L_v + (R'_{c,v} - R_{c,v})
      for (size_t e = start; e < end; ++e) {
        size_t v = graph.check_edges[e];
        size_t v_idx = v * BP_BATCH_SIZE;
        size_t msg_idx = e * BP_BATCH_SIZE;

#pragma GCC ivdep
        for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
          if (!active_shots[b]) continue;

          T old_r = graph.check_to_var_messages[msg_idx + b];
          T q_msg = posteriors_flat[v_idx + b] - old_r;

          uint8_t q_sign = (q_msg < 0) ? 1 : 0;
          uint8_t extrinsic_sign = total_sign_prod[b] ^ q_sign;
          uint8_t final_sign = batched_syndromes[syn_idx + b] ^ extrinsic_sign;

          T_MAG min_mag = (e == min1_idx[b]) ? min2[b] : min1[b];
          T_MAG normalized_mag = min_mag * normalization_factor;
          T final_mag;
          if constexpr (std::is_integral_v<T>) {
            final_mag =
                std::min((T_MAG)normalized_mag, static_cast<T_MAG>(std::numeric_limits<T>::max()));
          } else {
            final_mag = normalized_mag;
          }

          T new_r = final_sign ? -final_mag : final_mag;
          graph.check_to_var_messages[msg_idx + b] = new_r;
          posteriors_flat[v_idx + b] = q_msg + new_r;
        }
      }
    }  // End of check loop

    // --- Convergence Check (End of iteration) ---
    if (stop_at_convergence || (iter == max_iters - 1)) {
      std::vector<uint8_t> shot_converged(actual_batch_size, 1);

      for (size_t i = 0; i < graph.num_checks; ++i) {
        size_t start = graph.check_edge_offsets[i];
        size_t end = graph.check_edge_offsets[i + 1];
        size_t syn_idx = i * BP_BATCH_SIZE;

        for (size_t b = 0; b < actual_batch_size; ++b) {
          if (!active_shots[b]) continue;
          if (!shot_converged[b]) continue;

          uint8_t posterior_parity = 0;
          for (size_t e = start; e < end; ++e) {
            size_t v_idx = graph.check_edges[e];
            if (posteriors_flat[v_idx * BP_BATCH_SIZE + b] < 0) {
              posterior_parity ^= 1;
            }
          }

          if (posterior_parity != batched_syndromes[syn_idx + b]) {
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
      results[b].num_iters = iter;
    }
  }

  return results;
}

}  // namespace bp