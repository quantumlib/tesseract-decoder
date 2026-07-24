#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "bp/bp_parallel_min_sum.h"
#include "bp/tanner_graph_util.h"

namespace bp {

template <typename T>
BPResult bp_parallel_min_sum(TannerGraph<T>& graph, const std::vector<size_t>& detection_events,
                             std::vector<T>& posteriors, size_t max_iters,
                             float normalization_factor, bool stop_at_convergence) {
  graph.add_detection_events(detection_events);

  // Initialize all check-to-variable messages to 0 for the first iteration.
  std::fill(graph.check_to_var_messages.begin(), graph.check_to_var_messages.end(), 0);

  bool has_converged = false;
  size_t iter = 0;
  for (iter = 0; iter < max_iters; ++iter) {
    // --- Phase 1: Variable-to-Check messages (V->C) ---
    // Each variable node tells its neighboring check nodes what it believes,
    // based on its prior and the messages from all *other* check nodes.
    for (size_t i = 0; i < graph.variable_nodes.size(); ++i) {
      auto& variable = graph.variable_nodes[i];
      size_t start = graph.var_edge_offsets[i];
      size_t end = graph.var_edge_offsets[i + 1];

      // Calculate the variable's current total belief (posterior LLR) by summing
      // its prior and all incoming C->V messages from the previous iteration.
      T total_c2v = variable.prior;
      for (size_t e = start; e < end; ++e) {
        total_c2v = sat_add(total_c2v, graph.check_to_var_messages[e]);
      }

      // For each neighboring check, compute the extrinsic V->C message.
      for (size_t e = start; e < end; ++e) {
        T v2c_msg = total_c2v - graph.check_to_var_messages[e];
        size_t c_e = graph.var_edge_rev_indices[e];
        graph.var_to_check_messages[c_e] = v2c_msg;
      }
    }

    // --- Phase 2: Check-to-Variable messages (C->V) (Normalized Min-Sum) ---
    // Each check node tells its neighboring variable nodes whether their parity
    // constraint is satisfied, based on messages from all *other* variables.
    for (size_t i = 0; i < graph.check_nodes.size(); ++i) {
      auto& check = graph.check_nodes[i];
      size_t start = graph.check_edge_offsets[i];
      size_t end = graph.check_edge_offsets[i + 1];
      size_t deg = end - start;
      if (deg == 0) {
        continue;
      }

      // --- First Pass: Find the two minimum magnitudes and total sign product ---
      using T_MAG = typename llr_traits<T>::magnitude_type;
      T_MAG min1 = std::numeric_limits<T_MAG>::max();
      T_MAG min2 = std::numeric_limits<T_MAG>::max();
      size_t min1_idx = -1;
      bool total_sign_prod = false;

      for (size_t e = start; e < end; ++e) {
        T v2c_msg = graph.var_to_check_messages[e];
        T_MAG mag = (T_MAG)std::abs(v2c_msg);
        total_sign_prod ^= (v2c_msg < 0);

        if (mag < min1) {
          min2 = min1;
          min1 = mag;
          min1_idx = e;
        } else if (mag < min2) {
          min2 = mag;
        }
      }

      // --- Second Pass: Compute and store outgoing messages ---
      for (size_t e = start; e < end; ++e) {
        bool extrinsic_sign = total_sign_prod ^ (graph.var_to_check_messages[e] < 0);
        bool final_sign = check.syndrome ^ extrinsic_sign;

        T_MAG mag_to_send = (e == min1_idx) ? min2 : min1;

        T_MAG normalized_mag = mag_to_send * normalization_factor;
        T final_mag;

        if constexpr (std::is_integral_v<T>) {
          final_mag =
              std::min((T_MAG)normalized_mag, static_cast<T_MAG>(std::numeric_limits<T>::max()));
        } else {
          final_mag = normalized_mag;
        }

        size_t v_e = graph.check_edge_rev_indices[e];
        graph.check_to_var_messages[v_e] = final_sign ? -final_mag : final_mag;
      }
    }

    // --- Posterior and Convergence Check ---
    if (stop_at_convergence || (iter == max_iters - 1)) {
      for (size_t i = 0; i < graph.variable_nodes.size(); ++i) {
        auto& variable = graph.variable_nodes[i];
        variable.posterior = variable.prior;
        size_t start = graph.var_edge_offsets[i];
        size_t end = graph.var_edge_offsets[i + 1];
        for (size_t e = start; e < end; ++e) {
          variable.posterior = sat_add(variable.posterior, graph.check_to_var_messages[e]);
        }
        posteriors[i] = variable.posterior;
      }

      has_converged = bp::check_convergence(graph);
      if (has_converged) {
        iter++;
        break;
      }
    }
  }

  // --- Final Posterior Calculation (if not already done) ---
  if (!has_converged && iter == max_iters) {
    for (size_t i = 0; i < graph.variable_nodes.size(); ++i) {
      auto& variable = graph.variable_nodes[i];
      variable.posterior = variable.prior;
      size_t start = graph.var_edge_offsets[i];
      size_t end = graph.var_edge_offsets[i + 1];
      for (size_t e = start; e < end; ++e) {
        variable.posterior = sat_add(variable.posterior, graph.check_to_var_messages[e]);
      }
      posteriors[i] = variable.posterior;
    }
  }

  graph.remove_detection_events(detection_events);
  return {has_converged, iter};
}

}  // namespace bp