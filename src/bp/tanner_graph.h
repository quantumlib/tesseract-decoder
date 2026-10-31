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
  std::vector<size_t> var_edge_offsets;        // size: num_vars + 1
  std::vector<size_t> var_edges;               // size: num_edges (check indices)
  std::vector<size_t> var_edge_to_check_edge;  // size: num_edges (index in check_edges)

  // CSR for Check -> Variable adjacency.
  std::vector<size_t> check_edge_offsets;  // size: num_checks + 1
  std::vector<size_t> check_edges;         // size: num_edges (variable indices)

  // Precomputed Vacuum Resting State (s = 0) for Vacuum-Differential Wavefront BP.
  std::vector<LLR_UINT> min1_vac;       // size: num_checks
  std::vector<LLR_UINT> min2_vac;       // size: num_checks
  std::vector<size_t> argmin_vac;       // size: num_checks (local index in check_edges[c])
  std::vector<uint8_t> sign_prod_vac;   // size: num_checks
  std::vector<LLR_INT> m_vac;           // size: num_edges (indexed by check edge e)
  std::vector<LLR_INT> v_posterior_vac; // size: num_vars
  LLR_INT cached_alpha_fp = -1;
  bool vacuum_initialized = false;

  // Persistent single-shot sparse workspace buffers (restored in O(|dirty|) per shot).
  size_t num_check_words = 0;
  std::vector<LLR_INT> work_priors;                // size: num_vars
  std::vector<LLR_INT> work_m_vac;                 // size: num_edges
  std::vector<LLR_INT> work_check_to_var;          // size: num_edges
  std::vector<LLR_INT> work_posteriors;            // size: num_vars
  std::vector<LLR_UINT> work_min2_vac;             // size: num_checks
  std::vector<uint8_t> work_syndromes;             // size: num_checks
  std::vector<uint8_t> check_parity;               // size: num_checks
  std::vector<uint64_t> active_checks_words;       // size: num_check_words
  std::vector<uint64_t> next_active_checks_words;  // size: num_check_words

  std::vector<uint32_t> var_dirty_stamp;
  uint32_t dirty_gen = 1;
  std::vector<size_t> dirty_vars;

  std::vector<uint32_t> check_dirty_stamp;
  uint32_t check_dirty_gen = 1;
  std::vector<size_t> dirty_checks;

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
  void ensure_vacuum_state(float normalization_factor);
  size_t count_edges() const;
};

}  // namespace bp

#endif  // BELIEFPROPAGATION_BP_TANNER_GRAPH_H_
