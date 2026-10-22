#ifndef TESSERACT_BP_OSD_POST_PROCESSOR_H_
#define TESSERACT_BP_OSD_POST_PROCESSOR_H_

#include <cstddef>
#include <cstdint>

#include "bp/bp_trace.h"
#include "bp/post_processor.h"
#include "bp/tanner_graph.h"

namespace bp {

// Per-phase timings and counters accumulated across all calls to
// OsdPostProcessor::process on a single instance.
//
// OSD is the slow path (it only runs when BP fails to converge), so the cost of
// the handful of clock reads per call is negligible relative to the work being
// measured. Counters are plain (non-atomic) members: each decoding thread in
// bp_main owns its own OsdPostProcessor instance, so there is no sharing.
// Use OsdProfile::merge to aggregate across threads.
struct OsdProfile {
  // Number of times process() was called.
  uint64_t num_calls = 0;
  // Calls that returned early because BP had already converged (no OSD work).
  uint64_t num_converged_early_exits = 0;
  // Calls that ran the full OSD path.
  uint64_t num_osd_invocations = 0;

  // Wall time in seconds for calls that exited early because BP converged.
  double t_converged_passthrough = 0.0;

  // Wall time in seconds, broken down by phase. Only accumulated for calls that
  // actually run OSD (i.e. excluding converged early exits).
  double t_hard_decision = 0.0;  // initial hard decision from posteriors
  double t_residual = 0.0;       // residual syndrome computation
  double t_sort = 0.0;           // ordering columns by posterior reliability
  double t_matrix_build = 0.0;   // materializing the dense truncated H
  double t_elimination = 0.0;    // Gauss-Jordan elimination with column pivoting
  double t_solution = 0.0;       // correction extraction + OSD-w perturbation search
  double t_total = 0.0;          // whole OSD invocation (excluding converged early exits)

  // Problem-size and progress counters, summed over OSD invocations. Divide by
  // num_osd_invocations for per-call averages.
  uint64_t sum_num_errors = 0;      // columns available (N)
  uint64_t sum_num_detectors = 0;   // rows (M)
  uint64_t sum_max_osd_cols = 0;    // columns retained after truncation
  uint64_t sum_pivots_found = 0;    // rank reached by elimination
  uint64_t sum_cols_processed = 0;  // columns actually visited by the pivot loop

  void merge(const OsdProfile& other) {
    num_calls += other.num_calls;
    num_converged_early_exits += other.num_converged_early_exits;
    num_osd_invocations += other.num_osd_invocations;
    t_converged_passthrough += other.t_converged_passthrough;
    t_hard_decision += other.t_hard_decision;
    t_residual += other.t_residual;
    t_sort += other.t_sort;
    t_matrix_build += other.t_matrix_build;
    t_elimination += other.t_elimination;
    t_solution += other.t_solution;
    t_total += other.t_total;
    sum_num_errors += other.sum_num_errors;
    sum_num_detectors += other.sum_num_detectors;
    sum_max_osd_cols += other.sum_max_osd_cols;
    sum_pivots_found += other.sum_pivots_found;
    sum_cols_processed += other.sum_cols_processed;
  }
};

class OsdPostProcessor : public PostProcessor {
 public:
  // osd_order specifies the size of the information set subset to perturb.
  // osd_weight specifies the maximum Hamming weight of perturbations to explore.
  OsdPostProcessor(const TannerGraph<LLR_INT>& graph, size_t osd_order = 0, size_t osd_weight = 0,
                   float osd_truncation_factor = 0.0f, bool osd_early_exit = true);
  virtual ~OsdPostProcessor() = default;

  std::vector<uint8_t> process(
      const BPResult& bp_result, const std::vector<LLR_INT>& posteriors,
      const std::vector<uint64_t>& detection_events,
      const std::vector<std::vector<int>>& hyperedge_observables,
      std::vector<size_t>* predicted_errors = nullptr,
      const std::vector<LLR_INT>* custom_priors = nullptr) override;

  // Trace-enabled post-processing (only invoked when tracing is explicitly requested).
  std::vector<uint8_t> process_with_trace(
      const BPResult& bp_result, const std::vector<LLR_INT>& posteriors,
      const std::vector<uint64_t>& detection_events,
      const std::vector<std::vector<int>>& hyperedge_observables,
      OsdTrace& trace,
      std::vector<size_t>* predicted_errors = nullptr,
      const std::vector<LLR_INT>* custom_priors = nullptr);

  // Per-phase timings and counters accumulated by this instance.
  const OsdProfile& profile() const {
    return profile_;
  }
  void reset_profile() {
    profile_ = OsdProfile();
  }

 private:
  struct Lut1Entry {
    uint64_t sig = 0;
    uint32_t var_id = UINT32_MAX;
  };
  uint32_t lookup_lut1(uint64_t sig) const {
    if (lut1_.empty() || sig == 0) return UINT32_MAX;
    size_t idx = static_cast<size_t>(sig) & lut1_mask_;
    while (lut1_[idx].var_id != UINT32_MAX) {
      if (lut1_[idx].sig == sig) return lut1_[idx].var_id;
      idx = (idx + 1) & lut1_mask_;
    }
    return UINT32_MAX;
  }

  const TannerGraph<LLR_INT>& graph_;
  size_t osd_order_;
  size_t osd_weight_;
  float osd_truncation_factor_;
  bool osd_early_exit_;
  OsdProfile profile_;

  size_t num_det_words_ = 0;
  std::vector<uint64_t> det_sig_;
  std::vector<uint64_t> var_sig_;
  std::vector<double> base_var_cost_;
  double min_base_var_cost_ = 0.0;
  std::vector<Lut1Entry> lut1_;
  size_t lut1_mask_ = 0;
  std::vector<size_t> check_sorted_offsets_;
  std::vector<uint32_t> check_sorted_vars_;
  std::vector<uint64_t> var_mask_words_;
  std::vector<uint64_t> det_reach1_words_;
  std::vector<uint64_t> det_reach2_words_;
  mutable std::vector<int32_t> det_to_bit_;
  mutable std::vector<uint8_t> var_in_cand_;
};

}  // namespace bp

#endif  // TESSERACT_BP_OSD_POST_PROCESSOR_H_
