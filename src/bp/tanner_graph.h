#ifndef BELIEFPROPAGATION_BP_TANNER_GRAPH_H_
#define BELIEFPROPAGATION_BP_TANNER_GRAPH_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "bp/bp_params.h"

namespace bp {

// 16 shots processed concurrently per SIMD batch (16 * 32-bit LLR_INT = 512 bits for AVX-512).
constexpr size_t BP_BATCH_SIZE = 16;

struct TannerGraph {
  size_t num_vars = 0;
  size_t num_checks = 0;
  size_t num_edges = 0;
  size_t batch_size = BP_BATCH_SIZE;

  // Per-variable prior LLRs (size: num_vars).
  std::vector<LLR_INT> priors;

  // CSR for Variable -> Check adjacency.
  std::vector<size_t> var_edge_offsets;  // size: num_vars + 1
  std::vector<size_t> var_edges;         // size: num_edges (check indices)

  // CSR for Check -> Variable adjacency.
  std::vector<size_t> check_edge_offsets;  // size: num_checks + 1
  std::vector<size_t> check_edges;         // size: num_edges (variable indices)

  // Optional observable bitmask per variable.
  std::vector<uint64_t> var_obs_masks;

  // Pre-allocated batched BP runtime buffers (interleaved by batch_size).
  std::vector<uint8_t> syndromes;                 // size: num_checks * batch_size
  std::vector<LLR_INT> var_posteriors;            // size: num_vars * batch_size
  std::vector<LLR_INT> check_to_var_messages;     // size: num_edges * batch_size
  std::vector<uint8_t> converged_mask;            // size: batch_size
  std::vector<LLR_INT> converged_posteriors_flat; // size: num_vars * batch_size

  struct PendingEdge {
    size_t var;
    size_t check;
  };
  std::vector<PendingEdge> pending_edges;

  TannerGraph() = default;
  TannerGraph(size_t num_variables, size_t num_checks, const std::vector<LLR_INT>& priors);

  void add_edge(size_t variable, size_t check);
  void build();
  size_t count_edges() const;
};

}  // namespace bp

#endif  // BELIEFPROPAGATION_BP_TANNER_GRAPH_H_
