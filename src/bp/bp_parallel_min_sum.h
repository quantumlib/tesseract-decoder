#ifndef BELIEFPROPAGATION_BP_BP_PARALLEL_MIN_SUM_H_
#define BELIEFPROPAGATION_BP_BP_PARALLEL_MIN_SUM_H_

#include <vector>

#include "bp/tanner_graph.h"
#include "bp/tanner_graph_util.h"

namespace bp {

// Runs a parallel min-sum belief propagation decoder.
//
// Args:
//   graph: The TannerGraph to run the decoder on.
//   detection_events: A vector of check node indices that have detected a
//     syndrome.
//   posteriors: A vector to store the final posterior LLRs of the variable
//     nodes.
//   max_iters: The maximum number of iterations to run the decoder for.ƒllog

//   normalization_factor: The alpha factor for Normalized Min-Sum.
//   stop_at_convergence: If true, the decoder will stop early if the syndrome
//     is satisfied.
//
// Returns:
//   A BPResult struct containing whether the syndrome was satisfied and the
//   number of iterations performed.
template <typename T>
BPResult bp_parallel_min_sum(TannerGraph<T>& graph, const std::vector<size_t>& detection_events,
                             std::vector<T>& posteriors, size_t max_iters,
                             float normalization_factor, bool stop_at_convergence = true);

}  // namespace bp

#include "bp/bp_parallel_min_sum.inl"

#endif  // BELIEFPROPAGATION_BP_BP_PARALLEL_MIN_SUM_H_
