#ifndef BELIEFPROPAGATION_BP_SPARSE_SERIAL_GALLAGER_H_
#define BELIEFPROPAGATION_BP_SPARSE_SERIAL_GALLAGER_H_

#include <limits>

#include "bp/tanner_graph.h"
#include "bp/tanner_graph_util.h"

namespace bp {

void prepare_tanner_graph_for_bp_sparse_serial_gallager(TannerGraph<LLR_INT>& graph,
                                                        GallagerLookupTable& lut);

// Run belief propagation using a sparse serial schedule using the Gallager
// update rule implemented using a lookup table.
// The marginals computed by BP are output as log-likelihood ratios (LLR)
// log((1-p_i)/p_i) for each error mechanism i in the vector `posteriors`.
// Returns the BPResult {converged, num_iters} where `converged` is true if
// all checks are satisfied by the hard decisions (where a bit is flipped
// if its LLR < 0). `num_iters` is the number of completed iterations of BP.
// If stop_at_convergence is set to true, then BP will stop as soon as the
// checks are satisfied by the hard decisions.
BPResult bp_sparse_serial_gallager(TannerGraph<LLR_INT>& graph,
                                   const std::vector<size_t>& detection_events,
                                   std::vector<LLR_INT>& posteriors, size_t max_iters,
                                   GallagerLookupTable& lut, bool stop_at_convergence = true);

}  // namespace bp

#endif /* BELIEFPROPAGATION_BP_SPARSE_SERIAL_GALLAGER_H_ */