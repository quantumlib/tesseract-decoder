#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
#include <immintrin.h>
#endif

#include "bp/batched_bp_serial_min_sum.h"

namespace bp {

template <typename T>
std::vector<BPResult> batched_bp_serial_min_sum(
    BatchedTannerGraph<T>& graph, const std::vector<std::vector<size_t>>& detection_events_batch,
    std::vector<T>& posteriors_flat, size_t max_iters, float normalization_factor,
    bool stop_at_convergence, bool random_schedule, uint64_t random_seed) {
  size_t actual_batch_size = detection_events_batch.size();
  if (actual_batch_size > BP_BATCH_SIZE) {
    throw std::invalid_argument("Provided batch size exceeds BP_BATCH_SIZE");
  }

  // Set up batched syndromes as 16-bit masks per check node.
  std::vector<uint16_t> check_syndrome_masks(graph.num_checks, 0);
  for (size_t b = 0; b < actual_batch_size; ++b) {
    for (size_t d : detection_events_batch[b]) {
      check_syndrome_masks[d] |= (static_cast<uint16_t>(1) << b);
    }
  }

  uint16_t active_mask = (actual_batch_size == 16)
                             ? 0xFFFF
                             : static_cast<uint16_t>((1U << actual_batch_size) - 1);
  size_t num_active = actual_batch_size;

  std::vector<BPResult> results(actual_batch_size, {false, 0});

  // Initialize check_to_var messages to 0
  std::fill(graph.check_to_var_messages.begin(), graph.check_to_var_messages.end(), 0);

  // Initialize posteriors_flat to priors
  for (size_t i = 0; i < graph.num_variables; ++i) {
    T prior_val = graph.priors[i];
    size_t var_post_idx = i * BP_BATCH_SIZE;
#if defined(__AVX512F__)
    if constexpr (std::is_same_v<T, int32_t>) {
      __m512i v_prior = _mm512_set1_epi32(prior_val);
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(&posteriors_flat[var_post_idx]), v_prior);
    } else {
      for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
        posteriors_flat[var_post_idx + b] = prior_val;
      }
    }
#else
    for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
      posteriors_flat[var_post_idx + b] = prior_val;
    }
#endif
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

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
  if constexpr (std::is_same_v<T, int32_t>) {
    const __m512 v_norm = _mm512_set1_ps(normalization_factor);
    const __m512i v_zero = _mm512_setzero_si512();

    size_t iter = 0;
    for (iter = 0; iter < max_iters && active_mask != 0; ++iter) {
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

        uint16_t syn_mask = check_syndrome_masks[c];

        __m512i min1 = _mm512_set1_epi32(std::numeric_limits<int32_t>::max());
        __m512i min2 = _mm512_set1_epi32(std::numeric_limits<int32_t>::max());
        __m512i min1_idx = _mm512_set1_epi32(-1);
        uint16_t total_sign_mask = 0;

        // Pass 1: Compute variable-to-check extrinsic messages Q_{c,v} = L_v - R_{c,v}
        // and find min1, min2, and total sign product across 16 SIMD lanes.
        for (size_t e = start; e < end; ++e) {
          size_t v = graph.check_edges[e];
          size_t v_idx = v * BP_BATCH_SIZE;
          size_t msg_idx = e * BP_BATCH_SIZE;

          __m512i* v_ptr = reinterpret_cast<__m512i*>(&posteriors_flat[v_idx]);
          __m512i* msg_ptr = reinterpret_cast<__m512i*>(&graph.check_to_var_messages[msg_idx]);

          __m512i v_post = _mm512_loadu_si512(v_ptr);
          __m512i old_r = _mm512_loadu_si512(msg_ptr);
          __m512i q_msg = _mm512_sub_epi32(v_post, old_r);

          __m512i mag = _mm512_abs_epi32(q_msg);
          __mmask16 q_sign = _mm512_movepi32_mask(q_msg);
          total_sign_mask ^= static_cast<uint16_t>(q_sign);

          __mmask16 is_less_min1 = _mm512_cmplt_epi32_mask(mag, min1);
          __mmask16 is_less_min2 = _mm512_cmplt_epi32_mask(mag, min2);

          // If mag < min1: min2 = min1, min1 = mag, min1_idx = e
          min2 = _mm512_mask_blend_epi32(is_less_min1, min2, min1);
          min1 = _mm512_mask_blend_epi32(is_less_min1, min1, mag);
          min1_idx = _mm512_mask_blend_epi32(is_less_min1, min1_idx, _mm512_set1_epi32(static_cast<int32_t>(e)));

          // Else if mag < min2: min2 = mag
          __mmask16 update_min2 = is_less_min2 & ~is_less_min1;
          min2 = _mm512_mask_blend_epi32(update_min2, min2, mag);
        }

        // Pass 2: Compute new check-to-variable message R'_{c,v}
        // and immediately update posterior L'_v = Q_{c,v} + R'_{c,v}
        for (size_t e = start; e < end; ++e) {
          size_t v = graph.check_edges[e];
          size_t v_idx = v * BP_BATCH_SIZE;
          size_t msg_idx = e * BP_BATCH_SIZE;

          __m512i* v_ptr = reinterpret_cast<__m512i*>(&posteriors_flat[v_idx]);
          __m512i* msg_ptr = reinterpret_cast<__m512i*>(&graph.check_to_var_messages[msg_idx]);

          __m512i v_post = _mm512_loadu_si512(v_ptr);
          __m512i old_r = _mm512_loadu_si512(msg_ptr);
          __m512i q_msg = _mm512_sub_epi32(v_post, old_r);

          __mmask16 is_min1 = _mm512_cmpeq_epi32_mask(min1_idx, _mm512_set1_epi32(static_cast<int32_t>(e)));
          __m512i min_mag = _mm512_mask_blend_epi32(is_min1, min1, min2);

          // Normalized Min-Sum scaling
          __m512 mag_f = _mm512_cvtepi32_ps(min_mag);
          __m512 norm_f = _mm512_mul_ps(mag_f, v_norm);
          __m512i final_mag = _mm512_cvtps_epi32(norm_f);

          __mmask16 q_sign = _mm512_movepi32_mask(q_msg);
          uint16_t ext_sign = total_sign_mask ^ static_cast<uint16_t>(q_sign);
          __mmask16 final_sign = static_cast<__mmask16>(syn_mask ^ ext_sign);

          __m512i neg_final_mag = _mm512_sub_epi32(v_zero, final_mag);
          __m512i new_r = _mm512_mask_blend_epi32(final_sign, final_mag, neg_final_mag);

          _mm512_storeu_si512(msg_ptr, new_r);

          // Immediate Layered Update: L'_v = Q_{c,v} + R'_{c,v}
          __m512i new_post = _mm512_add_epi32(q_msg, new_r);
          _mm512_storeu_si512(v_ptr, new_post);
        }
      }  // End of check loop

      // --- Convergence Check (End of iteration) ---
      if (stop_at_convergence || (iter == max_iters - 1)) {
        uint16_t converged_mask = active_mask;

        for (size_t c = 0; c < graph.num_checks; ++c) {
          size_t start = graph.check_edge_offsets[c];
          size_t end = graph.check_edge_offsets[c + 1];
          uint16_t syn_mask = check_syndrome_masks[c];

          uint16_t posterior_parity_mask = 0;
          for (size_t e = start; e < end; ++e) {
            size_t v = graph.check_edges[e];
            __m512i v_post = _mm512_loadu_si512(reinterpret_cast<__m512i*>(&posteriors_flat[v * BP_BATCH_SIZE]));
            posterior_parity_mask ^= static_cast<uint16_t>(_mm512_movepi32_mask(v_post));
          }

          uint16_t failed_checks = (posterior_parity_mask ^ syn_mask);
          converged_mask &= ~failed_checks;
        }

        // Deactivate newly converged shots
        for (size_t b = 0; b < actual_batch_size; ++b) {
          uint16_t bit = static_cast<uint16_t>(1U << b);
          if ((active_mask & bit) && (converged_mask & bit)) {
            active_mask &= ~bit;
            results[b].converged = true;
            results[b].num_iters = iter + 1;
          }
        }
      }
    }

    // Update results for shots that didn't converge early
    for (size_t b = 0; b < actual_batch_size; ++b) {
      uint16_t bit = static_cast<uint16_t>(1U << b);
      if (active_mask & bit) {
        results[b].converged = false;
        results[b].num_iters = iter;
      }
    }

    return results;
  }
#endif

  // Fallback portable path (for non-int32 types or non-AVX-512 targets)
  using T_MAG = typename llr_traits<T>::magnitude_type;
  const T_MAG max_mag = std::numeric_limits<T_MAG>::max();

  std::vector<uint8_t> active_shots(BP_BATCH_SIZE, 0);
  for (size_t b = 0; b < actual_batch_size; ++b) active_shots[b] = 1;

  std::vector<uint8_t> batched_syndromes(graph.num_checks * BP_BATCH_SIZE, 0);
  for (size_t b = 0; b < actual_batch_size; ++b) {
    for (size_t d : detection_events_batch[b]) {
      batched_syndromes[d * BP_BATCH_SIZE + b] = 1;
    }
  }

  size_t iter = 0;
  for (iter = 0; iter < max_iters && num_active > 0; ++iter) {
    if (random_schedule && graph.num_checks > 1) {
      for (size_t i = graph.num_checks - 1; i > 0; --i) {
        size_t j = fast_rand() % (i + 1);
        std::swap(check_order[i], check_order[j]);
      }
    }

    for (size_t c_idx = 0; c_idx < graph.num_checks; ++c_idx) {
      size_t c = check_order[c_idx];
      size_t start = graph.check_edge_offsets[c];
      size_t end = graph.check_edge_offsets[c + 1];
      size_t deg = end - start;
      if (deg == 0) continue;

      size_t syn_idx = c * BP_BATCH_SIZE;

      T_MAG min1[BP_BATCH_SIZE];
      T_MAG min2[BP_BATCH_SIZE];
      size_t min1_idx[BP_BATCH_SIZE];
      uint8_t total_sign_prod[BP_BATCH_SIZE];

      for (size_t b = 0; b < BP_BATCH_SIZE; ++b) {
        min1[b] = max_mag;
        min2[b] = max_mag;
        min1_idx[b] = SIZE_MAX;
        total_sign_prod[b] = 0;
      }

      for (size_t e = start; e < end; ++e) {
        size_t v = graph.check_edges[e];
        size_t v_idx = v * BP_BATCH_SIZE;
        size_t msg_idx = e * BP_BATCH_SIZE;

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

      for (size_t e = start; e < end; ++e) {
        size_t v = graph.check_edges[e];
        size_t v_idx = v * BP_BATCH_SIZE;
        size_t msg_idx = e * BP_BATCH_SIZE;

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
    }

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

  for (size_t b = 0; b < actual_batch_size; ++b) {
    if (active_shots[b]) {
      results[b].converged = false;
      results[b].num_iters = iter;
    }
  }

  return results;
}

}  // namespace bp