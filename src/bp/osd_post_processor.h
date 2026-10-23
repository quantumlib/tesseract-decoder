#ifndef TESSERACT_BP_OSD_POST_PROCESSOR_H_
#define TESSERACT_BP_OSD_POST_PROCESSOR_H_

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "bp/bp_serial_min_sum.h"
#include "bp/tanner_graph.h"

namespace bp {

struct OsdProfile {
  uint64_t num_calls = 0, num_converged_early_exits = 0, num_osd_invocations = 0;
  uint64_t sum_num_errors = 0, sum_num_detectors = 0, sum_max_osd_cols = 0;
  uint64_t sum_cols_processed = 0, sum_pivots_found = 0;
  double t_converged_passthrough = 0.0, t_hard_decision = 0.0, t_residual = 0.0;
  double t_sort = 0.0, t_matrix_build = 0.0, t_elimination = 0.0, t_solution = 0.0, t_total = 0.0;

  void merge(const OsdProfile& o) {
    num_calls += o.num_calls;
    num_converged_early_exits += o.num_converged_early_exits;
    num_osd_invocations += o.num_osd_invocations;
    sum_num_errors += o.sum_num_errors;
    sum_num_detectors += o.sum_num_detectors;
    sum_max_osd_cols += o.sum_max_osd_cols;
    sum_cols_processed += o.sum_cols_processed;
    sum_pivots_found += o.sum_pivots_found;
    t_converged_passthrough += o.t_converged_passthrough;
    t_hard_decision += o.t_hard_decision;
    t_residual += o.t_residual;
    t_sort += o.t_sort;
    t_matrix_build += o.t_matrix_build;
    t_elimination += o.t_elimination;
    t_solution += o.t_solution;
    t_total += o.t_total;
  }

  void report(FILE* out = stdout) const {
    const double inv = num_osd_invocations > 0 ? 1.0 / num_osd_invocations : 0.0;
    const double tot = t_total > 0.0 ? t_total : 1e-30;
    const double t_acc =
        t_hard_decision + t_residual + t_sort + t_matrix_build + t_elimination + t_solution;
    const double t_other = t_total > t_acc ? t_total - t_acc : 0.0;

    std::fprintf(out, "\n=== OSD profile ===\n");
    std::fprintf(out, "  calls                 %llu\n",
                 static_cast<unsigned long long>(num_calls));
    std::fprintf(out, "  converged early exits %llu (%.6f s, %.3f us/exit)\n",
                 static_cast<unsigned long long>(num_converged_early_exits), t_converged_passthrough,
                 num_converged_early_exits > 0
                     ? 1e6 * t_converged_passthrough / num_converged_early_exits
                     : 0.0);
    std::fprintf(out, "  OSD invocations       %llu\n",
                 static_cast<unsigned long long>(num_osd_invocations));
    if (num_osd_invocations == 0) return;

    std::fprintf(out, "  detectors (M)         %.0f\n", sum_num_detectors * inv);
    std::fprintf(out, "  errors (N)            %.0f\n", sum_num_errors * inv);
    std::fprintf(out, "  cols after truncation %.0f\n", sum_max_osd_cols * inv);
    std::fprintf(out, "  cols processed        %.0f\n", sum_cols_processed * inv);
    std::fprintf(out, "  pivots found          %.0f\n", sum_pivots_found * inv);
    std::fprintf(out, "  ---------------------------------------------------------------\n");
    auto row = [&](const char* label, double t) {
      std::fprintf(out, "  %-22s %12.6f s  %6.2f %%  %12.3f us/call\n", label, t,
                   100.0 * t / tot, 1e6 * t * inv);
    };
    row("hard_decision", t_hard_decision);
    row("residual_syndrome", t_residual);
    row("sort", t_sort);
    row("matrix_build", t_matrix_build);
    row("elimination", t_elimination);
    row("solution", t_solution);
    row("other", t_other);
    std::fprintf(out, "  ---------------------------------------------------------------\n");
    row("TOTAL", t_total);
  }
};

class OsdPostProcessor {
 public:
  OsdPostProcessor(const TannerGraph& graph, size_t osd_order = 0, size_t osd_weight = 0,
                   float osd_truncation_factor = 0.0f, bool osd_early_exit = true);

  std::vector<uint8_t> process(
      const BPResult& bp_result, const std::vector<LLR_INT>& posteriors,
      const std::vector<uint64_t>& detection_events,
      const std::vector<std::vector<int>>& hyperedge_observables,
      std::vector<size_t>* predicted_errors = nullptr,
      const std::vector<LLR_INT>* custom_priors = nullptr);

  const OsdProfile& profile() const { return profile_; }
  void reset_profile() { profile_ = OsdProfile(); }

 private:
  const TannerGraph& graph_;
  size_t osd_order_;
  size_t osd_weight_;
  float osd_truncation_factor_;
  bool osd_early_exit_;
  OsdProfile profile_;

  size_t num_det_words_ = 0;
  std::vector<double> base_var_cost_;
  std::vector<uint64_t> var_mask_words_;
  std::vector<uint64_t> det_reach2_words_;
  mutable std::vector<int32_t> det_to_bit_;
  mutable std::vector<uint8_t> var_in_cand_;
};

}  // namespace bp

#endif  // TESSERACT_BP_OSD_POST_PROCESSOR_H_
