#ifndef TESSERACT_BP_BP_TRACE_H_
#define TESSERACT_BP_BP_TRACE_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace bp {

// Static bipartite Tanner graph structure for inspection and visualization.
struct TannerGraphTopology {
  size_t num_vars = 0;
  size_t num_checks = 0;
  size_t num_edges = 0;
  size_t num_observables = 0;
  std::vector<double> priors;
  std::vector<uint64_t> obs_masks;
  std::vector<std::vector<int>> hyperedge_observables;
  // Canonical edges indexed 0 .. num_edges - 1 in variable-major CSR order:
  // edges[e] = (var_idx, check_idx).
  std::vector<std::pair<size_t, size_t>> edges;
};

// Snapshot of a single Belief Propagation iteration (t = 1 .. T).
struct BpIterationTrace {
  size_t iter = 0;
  // Indexed by canonical edge index e in [0, num_edges):
  std::vector<double> var_to_check_messages;
  std::vector<double> check_to_var_messages;
  // Indexed by variable index v in [0, num_vars):
  std::vector<double> posteriors;
  std::vector<bool> hard_decisions;
  // Indexed by check index c in [0, num_checks):
  std::vector<bool> check_satisfied;
  std::vector<bool> residual_syndrome;
  bool converged = false;
};

// Post-processing trace for Ordered Statistics Decoding (OSD).
struct OsdTrace {
  bool triggered = false;
  std::vector<bool> residual_syndrome;
  std::vector<size_t> col_to_var;
  std::vector<size_t> pivot_vars;
  std::vector<size_t> free_vars;
  std::vector<size_t> best_error;
};

// Post-processing trace for Tesseract (Residual Peeling + 2-Hop Clustering + A*).
struct TesseractTrace {
  bool triggered = false;
  std::vector<size_t> peeled_vars;
  std::vector<size_t> active_vars;
  std::vector<size_t> unsolved_dets;
  std::vector<std::vector<size_t>> comp_dets;
  std::vector<std::vector<size_t>> comp_vars;
  std::vector<double> residual_llr_costs;
  std::vector<size_t> astar_flipped_vars;
  std::vector<size_t> final_flipped_vars;
  bool used_osd_fallback = false;
};

// Complete single-component BP + post-processor decode trace.
struct BpDecodeTrace {
  TannerGraphTopology topology;
  std::vector<uint64_t> detection_events;
  std::vector<BpIterationTrace> iterations;
  bool bp_converged = false;
  size_t num_iters = 0;
  std::string post_processor_type;
  OsdTrace osd_trace;
  TesseractTrace tesseract_trace;
  std::vector<size_t> predicted_errors;
  std::vector<uint8_t> predicted_observables;
};

// Cross-component prior reweighting event between passes in MultiPassBpDecoder.
struct ReweightEventTrace {
  size_t pass = 0;
  size_t source_comp_idx = 0;
  size_t source_var_idx = 0;
  size_t target_comp_idx = 0;
  size_t target_var_idx = 0;
  double old_prior = 0.0;
  double new_prior = 0.0;
};

// Per-pass component trace inside MultiPassBpDecoder.
struct ComponentPassTrace {
  size_t pass = 0;
  size_t comp_idx = 0;
  BpDecodeTrace trace;
};

// Complete Multi-Pass / DEM-Shattering decode trace.
struct MultiPassBpDecodeTrace {
  std::vector<TannerGraphTopology> component_topologies;
  std::vector<int> detector_components;
  std::vector<std::vector<size_t>> pass_schedule;
  std::vector<ComponentPassTrace> component_passes;
  std::vector<ReweightEventTrace> reweight_events;
  std::vector<uint8_t> predicted_observables;
};

}  // namespace bp

#endif  // TESSERACT_BP_BP_TRACE_H_
