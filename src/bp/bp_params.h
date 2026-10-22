#ifndef BELIEFPROPAGATION_BP_BP_PARAMS_H_
#define BELIEFPROPAGATION_BP_BP_PARAMS_H_

#include <cstdint>
#include <string>

namespace bp {

// Used to store discretised signed log-likelihood
// ratios, such as those sent in check-to-variable updates in BP.
typedef std::int32_t LLR_INT;
// Used to store unsigned integers at the check nodes.
// E.g. this could be a discretised unsigned LLR in a min-sum or tanh update,
// or a discretised f(abs(LLR)) in a Gallager update, where f is is the
// involution transform used in the Gallager update.
typedef std::uint32_t LLR_UINT;

// The number of bins used for the lookup tables (both Gallager and box-plus).
// This is also used (along with MAX_BP_LUT_LL) to convert LLRs from double to
// LLR_INT or LLR_UINT and vice versa.
const LLR_INT NUM_LUT_BINS = 1 << 14;

// The maximum allowed value of a log-likelihood ratio log((1-p)/p) representable
// as input to one of the lookup tables, as a floating point number. Used along with
// NUM_LUT_BINS to convert from double to LLR_INT or LLR_UINT and vice versa. Note that
// BP (depending on the implementation) can in general handle much higher LLRs than this.
// This is the case for the box-plus update, where the LUT is only used to compute a
// correction term with a small relevant domain. In this case the max LLR is limited
// by the precision of LLR_INT and LLR_UINT and can therefore be orders of
// magnitude higher (e.g. if these are 32 bits).
const double MAX_BP_LUT_LL = 20.0;

const size_t DEFAULT_MAX_ITER = 20;
const std::string DEFAULT_UPDATE_RULE = "min-sum";
const std::string DEFAULT_SCHEDULE = "serial";
const bool DEFAULT_STOP_AT_CONVERGENCE = true;
const bool DEFAULT_RANDOM_SCHEDULE = false;
const uint64_t DEFAULT_RANDOM_SEED = 0;

struct BPParams {
  size_t max_iter;
  std::string update_rule;
  std::string schedule;
  bool stop_at_convergence;
  float normalization_factor;
  bool random_schedule;
  uint64_t random_seed;
  // Factor used to truncate the number of columns in the dense OSD parity-check matrix.
  // The matrix width is bounded by min(num_errors, num_detectors * factor).
  // A value of 1.2 provides significant speedups without degrading LER.
  // A value of 0.0 disables truncation (falls back to exact OSD width).
  float osd_truncation_factor;
  // If true, enables early stopping in OSD-0 elimination when residual syndrome is satisfied.
  bool osd_early_exit;
  int osd_order;
  int osd_weight;
  std::string post_processor;
  int tesseract_det_beam;
  size_t tesseract_pqlimit;
  bool use_soft_reweighting;

  BPParams(size_t max_iter = DEFAULT_MAX_ITER, std::string update_rule = DEFAULT_UPDATE_RULE,
           std::string schedule = DEFAULT_SCHEDULE,
           bool stop_at_convergence = DEFAULT_STOP_AT_CONVERGENCE,
           float normalization_factor = 0.875f, bool random_schedule = DEFAULT_RANDOM_SCHEDULE,
           uint64_t random_seed = DEFAULT_RANDOM_SEED, float osd_truncation_factor = 1.2f,
           bool osd_early_exit = true,
           int osd_order = -1, int osd_weight = 0,
           std::string post_processor = "osd",
           int tesseract_det_beam = 5, size_t tesseract_pqlimit = 2000,
           bool use_soft_reweighting = true)
      : max_iter(max_iter),
        update_rule(update_rule),
        schedule(schedule),
        stop_at_convergence(stop_at_convergence),
        normalization_factor(normalization_factor),
        random_schedule(random_schedule),
        random_seed(random_seed),
        osd_truncation_factor(osd_truncation_factor),
        osd_early_exit(osd_early_exit),
        osd_order(osd_order),
        osd_weight(osd_weight),
        post_processor(post_processor),
        tesseract_det_beam(tesseract_det_beam),
        tesseract_pqlimit(tesseract_pqlimit),
        use_soft_reweighting(use_soft_reweighting) {}

  bool operator==(const BPParams& other) const {
    return max_iter == other.max_iter && update_rule == other.update_rule &&
           schedule == other.schedule && stop_at_convergence == other.stop_at_convergence &&
           normalization_factor == other.normalization_factor &&
           random_schedule == other.random_schedule && random_seed == other.random_seed &&
           osd_truncation_factor == other.osd_truncation_factor &&
           osd_early_exit == other.osd_early_exit &&
           osd_order == other.osd_order && osd_weight == other.osd_weight &&
           post_processor == other.post_processor &&
           tesseract_det_beam == other.tesseract_det_beam &&
           tesseract_pqlimit == other.tesseract_pqlimit &&
           use_soft_reweighting == other.use_soft_reweighting;
  }
};
}  // namespace bp

#endif /* BELIEFPROPAGATION_BP_BP_PARAMS_H_ */