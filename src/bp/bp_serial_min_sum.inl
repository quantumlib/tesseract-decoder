#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "bp/bp_serial_min_sum.h"

namespace bp {

template <typename T>
BPResult bp_serial_min_sum(TannerGraph<T>& graph, const std::vector<size_t>& detection_events,
                           std::vector<T>& posteriors, size_t max_iters, float normalization_factor,
                           bool stop_at_convergence, bool random_schedule, uint64_t random_seed) {
  graph.add_detection_events(detection_events);

  // Initialize posteriors to priors
  for (size_t i = 0; i < graph.variable_nodes.size(); ++i) {
    graph.variable_nodes[i].posterior = graph.variable_nodes[i].prior;
  }
  std::fill(graph.check_to_var_messages.begin(), graph.check_to_var_messages.end(), 0);

  using T_MAG = typename llr_traits<T>::magnitude_type;
  const T_MAG max_mag = std::numeric_limits<T_MAG>::max();

  std::vector<size_t> check_order(graph.check_nodes.size());
  for (size_t i = 0; i < graph.check_nodes.size(); ++i) {
    check_order[i] = i;
  }
  uint64_t rng_state = random_seed ? random_seed : 123456789ULL;
  auto fast_rand = [&rng_state]() -> uint32_t {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return static_cast<uint32_t>((rng_state * 0x2545F4914F6CDD1DULL) >> 32);
  };

  bool has_converged = false;
  size_t iter = 0;
  for (iter = 0; iter < max_iters; ++iter) {
    if (random_schedule && graph.check_nodes.size() > 1) {
      for (size_t i = graph.check_nodes.size() - 1; i > 0; --i) {
        size_t j = fast_rand() % (i + 1);
        std::swap(check_order[i], check_order[j]);
      }
    }

    // Horizontal / Layered Schedule: Iterate through check nodes
    for (size_t c_idx = 0; c_idx < graph.check_nodes.size(); ++c_idx) {
      size_t c = check_order[c_idx];
      auto& check = graph.check_nodes[c];
      size_t start = graph.check_edge_offsets[c];
      size_t end = graph.check_edge_offsets[c + 1];
      size_t deg = end - start;
      if (deg == 0) continue;

      T_MAG min1 = max_mag;
      T_MAG min2 = max_mag;
      size_t min1_idx = SIZE_MAX;
      uint8_t total_sign_prod = 0;

      // Pass 1: Compute Q_{c,v} = L_v - R_{c,v} and find min1, min2, and sign product
      for (size_t e = start; e < end; ++e) {
        size_t v = graph.check_edges[e];
        T old_r = graph.check_to_var_messages[e];
        T q_msg = graph.variable_nodes[v].posterior - old_r;

        T_MAG mag = (T_MAG)std::abs(q_msg);
        uint8_t sign = (q_msg < 0) ? 1 : 0;
        total_sign_prod ^= sign;

        if (mag < min1) {
          min2 = min1;
          min1 = mag;
          min1_idx = e;
        } else if (mag < min2) {
          min2 = mag;
        }
      }

      // Pass 2: Compute R'_{c,v} and immediately update L_v = Q_{c,v} + R'_{c,v}
      for (size_t e = start; e < end; ++e) {
        size_t v = graph.check_edges[e];
        T old_r = graph.check_to_var_messages[e];
        T q_msg = graph.variable_nodes[v].posterior - old_r;

        uint8_t q_sign = (q_msg < 0) ? 1 : 0;
        uint8_t extrinsic_sign = total_sign_prod ^ q_sign;
        uint8_t final_sign = check.syndrome ^ extrinsic_sign;

        T_MAG min_mag = (e == min1_idx) ? min2 : min1;
        T_MAG normalized_mag = min_mag * normalization_factor;
        T final_mag;
        if constexpr (std::is_integral_v<T>) {
          final_mag =
              std::min((T_MAG)normalized_mag, static_cast<T_MAG>(std::numeric_limits<T>::max()));
        } else {
          final_mag = normalized_mag;
        }

        T new_r = final_sign ? -final_mag : final_mag;
        graph.check_to_var_messages[e] = new_r;
        graph.variable_nodes[v].posterior = sat_add(q_msg, new_r);
      }
    }

    // --- Convergence Check (End of iteration) ---
    if (stop_at_convergence || (iter == max_iters - 1)) {
      has_converged = bp::check_convergence(graph);
      if (has_converged) {
        iter++;
        break;
      }
    }
  }

  // Extract posteriors
  for (size_t i = 0; i < graph.variable_nodes.size(); i++) {
    posteriors[i] = graph.variable_nodes[i].posterior;
  }

  graph.remove_detection_events(detection_events);
  return {has_converged, iter};
}

}  // namespace bp