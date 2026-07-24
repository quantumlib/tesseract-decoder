#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "bp/batched_bp_serial_min_sum.h"

namespace bp {

template <typename T>
std::vector<BPResult> batched_bp_serial_min_sum(
    BatchedTannerGraph<T>& graph, const std::vector<std::vector<size_t>>& detection_events_batch,
    std::vector<std::vector<T>>& posteriors_batch, size_t max_iters, float normalization_factor,
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

  // --- State Tracking Arrays ---
  // For each check node c, and batch b, we track min1, min2, min1_idx, and sign_prod.
  std::vector<T_MAG> check_min1(graph.num_checks * BP_BATCH_SIZE, max_mag);
  std::vector<T_MAG> check_min2(graph.num_checks * BP_BATCH_SIZE, max_mag);
  std::vector<size_t> check_min1_idx(graph.num_checks * BP_BATCH_SIZE, SIZE_MAX);
  std::vector<uint8_t> check_sign_prod(graph.num_checks * BP_BATCH_SIZE, 0);

  // Initialization: var_to_check messages are set to prior values
  for (size_t i = 0; i < graph.num_variables; ++i) {
    size_t start = graph.var_edge_offsets[i];
    size_t end = graph.var_edge_offsets[i + 1];
    T prior_val = graph.priors[i];

    for (size_t e = start; e < end; ++e) {
      size_t c_e = graph.var_edge_rev_indices[e];
      size_t out_idx = c_e * BP_BATCH_SIZE;
#pragma GCC ivdep
      for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
        graph.var_to_check_messages[out_idx + b] = prior_val;
      }
    }
  }

  // Initial full scan to populate check node states
  for (size_t i = 0; i < graph.num_checks; ++i) {
    size_t start = graph.check_edge_offsets[i];
    size_t end = graph.check_edge_offsets[i + 1];
    size_t state_idx = i * BP_BATCH_SIZE;

    for (size_t c_e = start; c_e < end; ++c_e) {
      size_t msg_idx = c_e * BP_BATCH_SIZE;
#pragma GCC ivdep
      for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
        T msg = graph.var_to_check_messages[msg_idx + b];
        T_MAG mag = (T_MAG)std::abs(msg);
        uint8_t sign = (msg < 0) ? 1 : 0;

        check_sign_prod[state_idx + b] ^= sign;

        if (mag < check_min1[state_idx + b]) {
          check_min2[state_idx + b] = check_min1[state_idx + b];
          check_min1[state_idx + b] = mag;
          check_min1_idx[state_idx + b] = c_e;
        } else if (mag < check_min2[state_idx + b]) {
          check_min2[state_idx + b] = mag;
        }
      }
    }
  }

  // Helper lambda to do a full rescan of a specific check node for the entire batch.
  auto rescan_check_node_batched = [&](size_t c_idx, const uint8_t* needs_rescan) {
    size_t start = graph.check_edge_offsets[c_idx];
    size_t end = graph.check_edge_offsets[c_idx + 1];
    size_t state_idx = c_idx * BP_BATCH_SIZE;

#pragma GCC ivdep
    for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
      if (needs_rescan[b]) {
        check_min1[state_idx + b] = max_mag;
        check_min2[state_idx + b] = max_mag;
        check_min1_idx[state_idx + b] = SIZE_MAX;
      }
    }

    for (size_t c_e = start; c_e < end; ++c_e) {
      size_t msg_idx = c_e * BP_BATCH_SIZE;
#pragma GCC ivdep
      for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
        if (needs_rescan[b]) {
          T msg = graph.var_to_check_messages[msg_idx + b];
          T_MAG mag = (T_MAG)std::abs(msg);
          if (mag < check_min1[state_idx + b]) {
            check_min2[state_idx + b] = check_min1[state_idx + b];
            check_min1[state_idx + b] = mag;
            check_min1_idx[state_idx + b] = c_e;
          } else if (mag < check_min2[state_idx + b]) {
            check_min2[state_idx + b] = mag;
          }
        }
      }
    }
  };

  size_t iter = 0;
  for (iter = 0; iter < max_iters && num_active > 0; ++iter) {
    // Serial Schedule: Iterate through variable nodes
    for (size_t i = 0; i < graph.num_variables; ++i) {
      size_t start = graph.var_edge_offsets[i];
      size_t end = graph.var_edge_offsets[i + 1];
      T prior_val = graph.priors[i];

// 1. Fetch C->V messages and compute new posterior in O(1) time per edge
#pragma GCC ivdep
      for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
        if (active_shots[b]) {
          posteriors_batch[b][i] = prior_val;
        }
      }

      for (size_t e = start; e < end; ++e) {
        size_t c_idx = graph.var_edges[e];
        size_t c_e =
            graph.var_edge_rev_indices[e];  // The index of this edge in the check node's list
        size_t state_idx = c_idx * BP_BATCH_SIZE;
        size_t syn_idx = c_idx * BP_BATCH_SIZE;
        size_t msg_idx = c_e * BP_BATCH_SIZE;

#pragma GCC ivdep
        for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
          if (!active_shots[b]) continue;

          T_MAG min_mag = (check_min1_idx[state_idx + b] == c_e) ? check_min2[state_idx + b]
                                                                 : check_min1[state_idx + b];

          T my_msg = graph.var_to_check_messages[msg_idx + b];
          uint8_t my_sign = (my_msg < 0) ? 1 : 0;
          uint8_t extrinsic_sign = check_sign_prod[state_idx + b] ^ my_sign;
          uint8_t final_sign = batched_syndromes[syn_idx + b] ^ extrinsic_sign;

          T_MAG normalized_mag = min_mag * normalization_factor;
          T final_mag;
          if constexpr (std::is_integral_v<T>) {
            final_mag =
                std::min((T_MAG)normalized_mag, static_cast<T_MAG>(std::numeric_limits<T>::max()));
          } else {
            final_mag = normalized_mag;
          }

          T c2v_msg = final_sign ? -final_mag : final_mag;
          posteriors_batch[b][i] += c2v_msg;

          // Store this for step 2 so we don't recompute it
          graph.check_to_var_messages[e * BP_BATCH_SIZE + b] = c2v_msg;
        }
      }

      // 2. Compute new V->C messages and update Check Node states
      for (size_t e = start; e < end; ++e) {
        size_t c_idx = graph.var_edges[e];
        size_t c_e = graph.var_edge_rev_indices[e];
        size_t state_idx = c_idx * BP_BATCH_SIZE;
        size_t msg_idx = c_e * BP_BATCH_SIZE;

        uint8_t needs_rescan[BP_BATCH_SIZE] = {0};
        bool any_needs_rescan = false;

#pragma GCC ivdep
        for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
          if (!active_shots[b]) continue;

          T old_msg = graph.var_to_check_messages[msg_idx + b];
          T new_msg = posteriors_batch[b][i] - graph.check_to_var_messages[e * BP_BATCH_SIZE + b];

          if (old_msg == new_msg) continue;

          graph.var_to_check_messages[msg_idx + b] = new_msg;

          T_MAG old_mag = (T_MAG)std::abs(old_msg);
          T_MAG new_mag = (T_MAG)std::abs(new_msg);
          uint8_t old_sign = (old_msg < 0) ? 1 : 0;
          uint8_t new_sign = (new_msg < 0) ? 1 : 0;

          // Incremental sign update
          check_sign_prod[state_idx + b] ^= (old_sign ^ new_sign);

          // Incremental magnitude update
          if (new_mag < check_min1[state_idx + b]) {
            // New absolute minimum found!
            if (check_min1_idx[state_idx + b] != c_e) {
              check_min2[state_idx + b] = check_min1[state_idx + b];
            }
            check_min1[state_idx + b] = new_mag;
            check_min1_idx[state_idx + b] = c_e;
          } else if (new_mag < check_min2[state_idx + b] && check_min1_idx[state_idx + b] != c_e) {
            // New second minimum found!
            check_min2[state_idx + b] = new_mag;
          } else if (check_min1_idx[state_idx + b] == c_e && new_mag > old_mag) {
            // The minimum grew. We must rescan to find the new true minimums.
            needs_rescan[b] = 1;
            any_needs_rescan = true;
          } else if (old_mag == check_min2[state_idx + b] && new_mag > old_mag) {
            // The second minimum grew. We must rescan.
            needs_rescan[b] = 1;
            any_needs_rescan = true;
          }
        }

        if (any_needs_rescan) {
          rescan_check_node_batched(c_idx, needs_rescan);
        }
      }
    }  // End of variable loop

    // --- Convergence Check (End of iteration) ---
    if (stop_at_convergence || (iter == max_iters - 1)) {
      std::vector<uint8_t> shot_converged(actual_batch_size, 1);

      for (size_t i = 0; i < graph.num_checks; ++i) {
        size_t start = graph.check_edge_offsets[i];
        size_t end = graph.check_edge_offsets[i + 1];
        size_t syn_idx = i * BP_BATCH_SIZE;

        for (size_t b = 0; b < actual_batch_size; ++b) {
          if (!active_shots[b]) continue;
          if (!shot_converged[b]) continue;  // Already failed a check

          uint8_t posterior_parity = 0;
          for (size_t e = start; e < end; ++e) {
            size_t v_idx = graph.check_edges[e];
            if (posteriors_batch[b][v_idx] < 0) {
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
      results[b].num_iters = iter;  // `iter` will be `max_iters` here
    }
  }

  return results;
}

}  // namespace bp