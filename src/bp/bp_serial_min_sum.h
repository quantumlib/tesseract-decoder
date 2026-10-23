#ifndef BELIEFPROPAGATION_BP_BP_SERIAL_MIN_SUM_H_
#define BELIEFPROPAGATION_BP_BP_SERIAL_MIN_SUM_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "bp/bp_params.h"
#include "bp/tanner_graph.h"
#include "bp/tanner_graph_util.h"

namespace bp {

std::vector<BPResult> bp_serial_min_sum(
    TannerGraph& graph, const std::vector<std::vector<size_t>>& detection_events_batch,
    std::vector<LLR_INT>& posteriors_flat, size_t max_iters, float normalization_factor,
    bool stop_at_convergence = true, bool random_schedule = false, uint64_t random_seed = 0,
    const std::vector<LLR_INT>* custom_priors_flat = nullptr);

}  // namespace bp

#endif  // BELIEFPROPAGATION_BP_BP_SERIAL_MIN_SUM_H_
