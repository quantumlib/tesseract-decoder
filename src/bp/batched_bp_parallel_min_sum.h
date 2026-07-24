#ifndef BELIEFPROPAGATION_BP_BATCHED_BP_PARALLEL_MIN_SUM_H_
#define BELIEFPROPAGATION_BP_BATCHED_BP_PARALLEL_MIN_SUM_H_

#include <vector>

#include "bp/batched_tanner_graph.h"
#include "bp/tanner_graph_util.h"  // For BPResult

namespace bp {

// Runs a batched parallel min-sum belief propagation decoder.
//
// Args:
//   graph: The BatchedTannerGraph to run the decoder on.
//   detection_events_batch: A 2D vector [BATCH_SIZE][num_events]. The syndromes
//     for all shots in the current batch.
//   posteriors_batch: A 2D vector [BATCH_SIZE][num_variables] to store the
//     final posterior LLRs of the variable nodes for all shots.
//   max_iters: The maximum number of iterations to run the decoder for.
//   normalization_factor: The alpha factor for Normalized Min-Sum.
//   stop_at_convergence: If true, the decoder will stop early for specific
//     shots if their syndrome is satisfied.
//
// Returns:
//   A vector of BPResult structs (size BATCH_SIZE) containing whether each shot
//   converged and the number of iterations performed.
template <typename T>
std::vector<BPResult> batched_bp_parallel_min_sum(
    BatchedTannerGraph<T>& graph, const std::vector<std::vector<size_t>>& detection_events_batch,
    std::vector<std::vector<T>>& posteriors_batch, size_t max_iters, float normalization_factor,
    bool stop_at_convergence = true);

}  // namespace bp

#include "bp/batched_bp_parallel_min_sum.inl"

#endif  // BELIEFPROPAGATION_BP_BATCHED_BP_PARALLEL_MIN_SUM_H_