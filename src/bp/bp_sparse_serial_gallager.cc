#include "bp/bp_sparse_serial_gallager.h"

#include <algorithm>
#include <limits>

#include "bp/tanner_graph_util.h"

namespace bp {

void print_graph_2(TannerGraph<LLR_INT>& graph) {}

void prepare_tanner_graph_for_bp_sparse_serial_gallager(TannerGraph<LLR_INT>& graph,
                                                        GallagerLookupTable& lut) {
  // Reset parities and totals at check nodes
  for (auto& check : graph.check_nodes) {
    check.message_parity = false;
    check.message_total = 0;
  }

  for (size_t i = 0; i < graph.variable_nodes.size(); ++i) {
    auto& variable = graph.variable_nodes[i];
    size_t start = graph.var_edge_offsets[i];
    size_t end = graph.var_edge_offsets[i + 1];

    bool prior_is_negative = variable.prior < 0;
    LLR_INT sign = prior_is_negative ? -1 : 1;
    // We initially set the variable to check messages equal to the priors
    // with the Gallager involution transform applied (and sign preserved)
    LLR_UINT f_mag = lut.f(std::abs(variable.prior));
    LLR_INT abs_f_message =
        std::min(f_mag, static_cast<LLR_UINT>(std::numeric_limits<LLR_INT>::max()));
    LLR_INT f_message = sign * abs_f_message;

    for (size_t e = start; e < end; ++e) {
      size_t c_idx = graph.var_edges[e];
      auto& check = graph.check_nodes[c_idx];
      size_t c_e = graph.var_edge_rev_indices[e];

      graph.var_to_check_messages[c_e] = f_message;

      check.message_parity ^= prior_is_negative;
      check.message_total += abs_f_message;
    }
    variable.message_prior_cached = f_message;
  }
  // Cache the initial message parity and message total to enable fast reset
  for (auto& check : graph.check_nodes) {
    check.message_parity_cached = check.message_parity;
    check.message_total_cached = check.message_total;
  }
}

void reset_modified_tanner_graph_nodes(TannerGraph<LLR_INT>& graph) {
  // Reset the totals on the checks
  for (auto c_idx : graph.syndrome_vicinity) {
    auto& c = graph.check_nodes[c_idx];
    c.message_parity = c.message_parity_cached;
    c.message_total = c.message_total_cached;
  }

  // Reset the intitial variable-to-check messages
  for (auto v_idx : graph.syndrome_boundary) {
    auto& v = graph.variable_nodes[v_idx];
    size_t start = graph.var_edge_offsets[v_idx];
    size_t end = graph.var_edge_offsets[v_idx + 1];
    for (size_t e = start; e < end; ++e) {
      size_t c_e = graph.var_edge_rev_indices[e];
      graph.var_to_check_messages[c_e] = v.message_prior_cached;
    }
  }
}

BPResult bp_sparse_serial_gallager(TannerGraph<LLR_INT>& graph,
                                   const std::vector<size_t>& detection_events,
                                   std::vector<LLR_INT>& posteriors, size_t max_iters,
                                   GallagerLookupTable& lut, bool stop_at_convergence) {
  graph.syndrome_boundary.clear();
  for (auto d : detection_events) {
    auto& check = graph.check_nodes[d];
    check.syndrome = true;
    size_t start = graph.check_edge_offsets[d];
    size_t end = graph.check_edge_offsets[d + 1];
    for (size_t e = start; e < end; ++e) {
      size_t v_idx = graph.check_edges[e];
      if (graph.variable_nodes[v_idx].posterior != std::numeric_limits<LLR_INT>::max()) {
        graph.syndrome_boundary.push_back(v_idx);
        graph.variable_nodes[v_idx].posterior = std::numeric_limits<LLR_INT>::max();
      }
    }
  }

  graph.syndrome_vicinity.clear();
  for (auto v_idx : graph.syndrome_boundary) {
    size_t start = graph.var_edge_offsets[v_idx];
    size_t end = graph.var_edge_offsets[v_idx + 1];
    for (size_t e = start; e < end; ++e) {
      size_t c_idx = graph.var_edges[e];
      if (graph.check_nodes[c_idx].message_total != std::numeric_limits<LLR_INT>::max()) {
        graph.syndrome_vicinity.push_back(c_idx);
        graph.check_nodes[c_idx].message_total = std::numeric_limits<LLR_INT>::max();
      }
    }
  }
  for (auto c_idx : graph.syndrome_vicinity) {
    graph.check_nodes[c_idx].message_total = graph.check_nodes[c_idx].message_total_cached;
  }

  bool has_converged = false;
  size_t iter = 0;
  for (iter = 0; iter < max_iters; iter++) {
    for (auto v_idx : graph.syndrome_boundary) {
      auto& variable = graph.variable_nodes[v_idx];
      size_t start = graph.var_edge_offsets[v_idx];
      size_t end = graph.var_edge_offsets[v_idx + 1];

      // Compute check-to-variable messages
      for (size_t e = start; e < end; ++e) {
        size_t c_idx = graph.var_edges[e];
        auto& check = graph.check_nodes[c_idx];
        size_t c_e = graph.var_edge_rev_indices[e];

        size_t c_start = graph.check_edge_offsets[c_idx];
        size_t c_end = graph.check_edge_offsets[c_idx + 1];

        LLR_INT sign =
            (check.message_parity ^ (graph.var_to_check_messages[c_e] < 0) ^ check.syndrome) ? -1
                                                                                             : 1;
        if (c_end - c_start == 1) {
          graph.check_to_var_messages[e] = sign * bp::LLR_FOR_PROB_ZERO;
          continue;
        }

        // Send the check-to-variable message
        LLR_UINT f_mag =
            lut.f(std::abs(check.message_total - std::abs(graph.var_to_check_messages[c_e])));
        LLR_INT mag_clamped =
            std::min(f_mag, static_cast<LLR_UINT>(std::numeric_limits<LLR_INT>::max()));
        graph.check_to_var_messages[e] = sign * mag_clamped;
      }

      // Do sum update
      variable.posterior = variable.prior;
      for (size_t e = start; e < end; ++e) {
        variable.posterior += graph.check_to_var_messages[e];
      }

      // Send variable to check messages by subtracting from total
      for (size_t e = start; e < end; ++e) {
        size_t c_idx = graph.var_edges[e];
        auto& check = graph.check_nodes[c_idx];
        size_t c_e = graph.var_edge_rev_indices[e];

        LLR_INT var_to_check = variable.posterior - graph.check_to_var_messages[e];
        bool is_negative = var_to_check < 0;
        LLR_INT sign = is_negative ? -1 : 1;
        LLR_UINT f_mag = lut.f(std::abs(var_to_check));
        LLR_INT abs_message =
            std::min(f_mag, static_cast<LLR_UINT>(std::numeric_limits<LLR_INT>::max()));

        check.message_total += abs_message;
        check.message_total -= std::abs(graph.var_to_check_messages[c_e]);
        check.message_parity ^= is_negative ^ (graph.var_to_check_messages[c_e] < 0);

        graph.var_to_check_messages[c_e] = sign * abs_message;
      }
    }

    if (stop_at_convergence || (iter == max_iters - 1)) {
      has_converged = bp::check_convergence(graph.syndrome_vicinity, graph);
      if (has_converged) {
        iter++;
        break;
      }
    }
  }

  // Extract posteriors
  posteriors.resize(graph.variable_nodes.size());
  for (size_t i = 0; i < graph.variable_nodes.size(); i++) {
    posteriors[i] = graph.variable_nodes[i].prior;
  }
  for (auto v_idx : graph.syndrome_boundary) {
    posteriors[v_idx] = graph.variable_nodes[v_idx].posterior;
  }

  graph.remove_detection_events(detection_events);

  // Sparse reset
  reset_modified_tanner_graph_nodes(graph);

  return {has_converged, iter};
}

}  // namespace bp