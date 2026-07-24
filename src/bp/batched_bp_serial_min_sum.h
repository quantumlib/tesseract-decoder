#ifndef BELIEFPROPAGATION_BP_BATCHED_BP_SERIAL_MIN_SUM_H_
#define BELIEFPROPAGATION_BP_BATCHED_BP_SERIAL_MIN_SUM_H_

#include <vector>

#include "bp/batched_tanner_graph.h"
#include "bp/tanner_graph_util.h"

namespace bp {

template <typename T>
std::vector<BPResult> batched_bp_serial_min_sum(
    BatchedTannerGraph<T>& graph, const std::vector<std::vector<size_t>>& detection_events_batch,
    std::vector<std::vector<T>>& posteriors_batch, size_t max_iters, float normalization_factor,
    bool stop_at_convergence = true);

}  // namespace bp

#include "bp/batched_bp_serial_min_sum.inl"

#endif  // BELIEFPROPAGATION_BP_BATCHED_BP_SERIAL_MIN_SUM_H_