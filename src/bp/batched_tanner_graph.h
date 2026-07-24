#ifndef BELIEFPROPAGATION_BP_BATCHED_TANNER_GRAPH_H_
#define BELIEFPROPAGATION_BP_BATCHED_TANNER_GRAPH_H_

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "bp/bp_types.h"
#include "bp/tanner_graph.h"

namespace bp {

// Configurable batch size. 16 means we process 16 shots concurrently.
// 16 * 32-bit (LLR_INT) = 512 bits, perfectly matching AVX-512.
constexpr size_t BP_BATCH_SIZE = 16;

template <typename T>
struct BatchedTannerGraph {
  using T_MAG = typename llr_traits<T>::magnitude_type;

  size_t num_variables;
  size_t num_checks;

  // Priors: Each variable node has a prior.
  // We don't replicate it across batches in memory, we just read the single scalar
  // and apply it to all batch elements during computation.
  std::vector<T> priors;

  // CSR for Variable -> Check edges
  std::vector<size_t> var_edge_offsets;      // size: num_variables + 1
  std::vector<size_t> var_edges;             // check indices
  std::vector<size_t> var_edge_rev_indices;  // reverse index in check_edges

  // For batched messages, we store arrays of size (num_edges * BATCH_SIZE).
  // The layout is interleaved: [e0_b0, e0_b1, ..., e0_b15, e1_b0, e1_b1, ...]
  // This contiguous layout ensures perfect auto-vectorization across the BATCH_SIZE loop.
  std::vector<T> check_to_var_messages;

  // CSR for Check -> Variable edges
  std::vector<size_t> check_edge_offsets;      // size: num_checks + 1
  std::vector<size_t> check_edges;             // var indices
  std::vector<size_t> check_edge_rev_indices;  // reverse index in var_edges
  std::vector<T> var_to_check_messages;

  BatchedTannerGraph() = default;
  BatchedTannerGraph(size_t num_variables, size_t num_checks, const std::vector<T>& priors);

  void build_from_unbatched(const TannerGraph<T>& unbatched_graph);
};

template <typename T>
BatchedTannerGraph<T>::BatchedTannerGraph(size_t num_variables, size_t num_checks,
                                          const std::vector<T>& priors)
    : num_variables(num_variables), num_checks(num_checks), priors(priors) {}

template <typename T>
void BatchedTannerGraph<T>::build_from_unbatched(const TannerGraph<T>& unbatched_graph) {
  num_variables = unbatched_graph.variable_nodes.size();
  num_checks = unbatched_graph.check_nodes.size();

  priors.resize(num_variables);
  for (size_t i = 0; i < num_variables; ++i) {
    priors[i] = unbatched_graph.variable_nodes[i].prior;
  }

  var_edge_offsets = unbatched_graph.var_edge_offsets;
  var_edges = unbatched_graph.var_edges;
  var_edge_rev_indices = unbatched_graph.var_edge_rev_indices;
  check_to_var_messages.assign(var_edges.size() * BP_BATCH_SIZE, 0);

  check_edge_offsets = unbatched_graph.check_edge_offsets;
  check_edges = unbatched_graph.check_edges;
  check_edge_rev_indices = unbatched_graph.check_edge_rev_indices;
  var_to_check_messages.assign(check_edges.size() * BP_BATCH_SIZE, 0);
}

}  // namespace bp

#endif  // BELIEFPROPAGATION_BP_BATCHED_TANNER_GRAPH_H_