#ifndef BELIEFPROPAGATION_BP_TANNER_GRAPH_H_
#define BELIEFPROPAGATION_BP_TANNER_GRAPH_H_

#include <limits>
#include <stdexcept>
#include <vector>

#include "bp/bp_types.h"
#include "bp/check_update.h"

namespace bp {

const bp::LLR_INT LLR_FOR_PROB_ZERO = 2 * bp::NUM_LUT_BINS;

template <typename T>
struct VariableNode {
  T prior;
  T message_prior_cached;
  T posterior;
};

template <typename T>
struct CheckNode {
  bool syndrome;
  bool message_parity;
  bool message_parity_cached;
  T message_total;
  T message_total_cached;
  size_t next_idx;
};

template <typename T>
struct TannerGraph {
  using T_MAG = typename llr_traits<T>::magnitude_type;

  std::vector<VariableNode<T>> variable_nodes;
  std::vector<CheckNode<T>> check_nodes;

  // CSR for Variable -> Check edges
  std::vector<size_t> var_edge_offsets;      // size: num_vars + 1
  std::vector<size_t> var_edges;             // check indices
  std::vector<size_t> var_edge_rev_indices;  // reverse index in check_edges
  std::vector<T> check_to_var_messages;      // check->var message (indexed by var_edge)

  // CSR for Check -> Variable edges
  std::vector<size_t> check_edge_offsets;      // size: num_checks + 1
  std::vector<size_t> check_edges;             // var indices
  std::vector<size_t> check_edge_rev_indices;  // reverse index in var_edges
  std::vector<T> var_to_check_messages;        // var->check message (indexed by check_edge)

  // Forward/Back states for Box-Plus (indexed by check node)
  std::vector<size_t> check_fb_offsets;  // size: num_checks + 1
  std::vector<T_MAG> check_forward_back;

  // Sparse BP state
  std::vector<size_t> syndrome_boundary;  // variable node indices
  std::vector<size_t> syndrome_vicinity;  // check node indices

  struct PendingEdge {
    size_t var;
    size_t check;
  };
  std::vector<PendingEdge> pending_edges;

  TannerGraph() = default;
  TannerGraph(size_t num_variables, size_t num_checks, const std::vector<T>& priors);

  void add_edge(size_t variable, size_t check);
  void build();  // Must be called after all add_edge calls

  void add_detection_events(const std::vector<size_t>& detection_events);
  void remove_detection_events(const std::vector<size_t>& detection_events);
  size_t count_edges() const;
};

}  // namespace bp

#include "bp/tanner_graph.inl"

#endif /* BELIEFPROPAGATION_BP_TANNER_GRAPH_H_ */