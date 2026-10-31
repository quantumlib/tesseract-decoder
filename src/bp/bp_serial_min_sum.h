#ifndef BELIEFPROPAGATION_BP_BP_SERIAL_MIN_SUM_H_
#define BELIEFPROPAGATION_BP_BP_SERIAL_MIN_SUM_H_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "bp/bp_params.h"
#include "bp/tanner_graph.h"
#include "bp/tanner_graph_util.h"

namespace bp {

struct SparseShotBpResult {
  BPResult bp_res;
  std::vector<size_t> flipped_vars;
  size_t active_checks_visited = 0;
};

std::vector<BPResult> bp_serial_min_sum(
    TannerGraph& graph, const std::vector<std::vector<size_t>>& detection_events_batch,
    std::vector<LLR_INT>& posteriors_flat, size_t max_iters, float normalization_factor,
    bool stop_at_convergence = true, bool random_schedule = false, uint64_t random_seed = 0,
    const std::vector<LLR_INT>* custom_priors_flat = nullptr, int damping_num = 0,
    std::vector<LLR_INT>* tstar_posteriors_flat = nullptr);

// Dynamic Vacuum-Differential Wavefront Min-Sum BP.
SparseShotBpResult bp_wavefront_min_sum_shot(
    TannerGraph& graph, const std::vector<size_t>& detection_events, size_t max_iters,
    float normalization_factor, bool stop_at_convergence = true,
    const std::vector<std::pair<size_t, LLR_INT>>* sparse_custom_priors = nullptr,
    std::vector<LLR_INT>* out_unconverged_posteriors = nullptr,
    std::vector<LLR_INT>* out_tstar_posteriors = nullptr);

}  // namespace bp

#endif  // BELIEFPROPAGATION_BP_BP_SERIAL_MIN_SUM_H_
