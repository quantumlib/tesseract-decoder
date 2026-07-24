#ifndef BELIEFPROPAGATION_BP_BP_SERIAL_MIN_SUM_H_
#define BELIEFPROPAGATION_BP_BP_SERIAL_MIN_SUM_H_

#include <vector>

#include "bp/tanner_graph.h"
#include "bp/tanner_graph_util.h"

namespace bp {

template <typename T>
BPResult bp_serial_min_sum(TannerGraph<T>& graph, const std::vector<size_t>& detection_events,
                           std::vector<T>& posteriors, size_t max_iters, float normalization_factor,
                           bool stop_at_convergence = true);

}  // namespace bp

#include "bp/bp_serial_min_sum.inl"

#endif  // BELIEFPROPAGATION_BP_BP_SERIAL_MIN_SUM_H_
