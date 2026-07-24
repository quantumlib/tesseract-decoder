#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "bp/bp_serial_min_sum.h"

namespace bp {

template <typename T>
BPResult bp_serial_min_sum(TannerGraph<T>& graph, const std::vector<size_t>& detection_events,
                           std::vector<T>& posteriors, size_t max_iters, float normalization_factor,
                           bool stop_at_convergence) {
  graph.add_detection_events(detection_events);

  // Initialization: var_to_check messages are set to prior values
  for (size_t i = 0; i < graph.variable_nodes.size(); ++i) {
    size_t start = graph.var_edge_offsets[i];
    size_t end = graph.var_edge_offsets[i + 1];
    T prior_val = graph.variable_nodes[i].prior;

    for (size_t e = start; e < end; ++e) {
      size_t c_e = graph.var_edge_rev_indices[e];
      graph.var_to_check_messages[c_e] = prior_val;
    }
  }

  using T_MAG = typename llr_traits<T>::magnitude_type;
  const T_MAG max_mag = std::numeric_limits<T_MAG>::max();

  bool has_converged = false;
  size_t iter = 0;
  for (iter = 0; iter < max_iters; ++iter) {
    // Serial Schedule: Iterate through variable nodes
    for (size_t i = 0; i < graph.variable_nodes.size(); ++i) {
      auto& variable = graph.variable_nodes[i];
      size_t start = graph.var_edge_offsets[i];
      size_t end = graph.var_edge_offsets[i + 1];

      // 1. Fetch check-to-var messages dynamically for this variable
      for (size_t e = start; e < end; ++e) {
        size_t c_idx = graph.var_edges[e];
        auto& check = graph.check_nodes[c_idx];
        size_t c_start = graph.check_edge_offsets[c_idx];
        size_t c_end = graph.check_edge_offsets[c_idx + 1];

        T_MAG min_mag = max_mag;
        uint8_t sign_prod = 0;

        // Compute min1 and product of signs for all neighbors of check c EXCEPT variable i
        for (size_t c_e = c_start; c_e < c_end; ++c_e) {
          if (graph.check_edges[c_e] == i) continue;  // Skip self

          T msg = graph.var_to_check_messages[c_e];
          T_MAG mag = (T_MAG)std::abs(msg);
          if (mag < min_mag) {
            min_mag = mag;
          }
          uint8_t sign = (msg < 0) ? 1 : 0;
          sign_prod ^= sign;
        }

        // Compute final check-to-var message
        uint8_t final_sign = check.syndrome ^ sign_prod;

        T_MAG normalized_mag = min_mag * normalization_factor;
        T final_mag;
        if constexpr (std::is_integral_v<T>) {
          final_mag =
              std::min((T_MAG)normalized_mag, static_cast<T_MAG>(std::numeric_limits<T>::max()));
        } else {
          final_mag = normalized_mag;
        }

        graph.check_to_var_messages[e] = final_sign ? -final_mag : final_mag;
      }

      // 2. Compute updated posterior for the current variable node
      variable.posterior = variable.prior;
      for (size_t e = start; e < end; ++e) {
        variable.posterior = sat_add(variable.posterior, graph.check_to_var_messages[e]);
      }

      // 3. Update var-to-check messages for the current variable node
      for (size_t e = start; e < end; ++e) {
        size_t c_e = graph.var_edge_rev_indices[e];
        graph.var_to_check_messages[c_e] = variable.posterior - graph.check_to_var_messages[e];
      }
    }  // End of variable loop

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