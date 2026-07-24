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

const size_t DEFAULT_MAX_ITER = 4;
const std::string DEFAULT_UPDATE_RULE = "box-plus";
const std::string DEFAULT_SCHEDULE = "serial";
const size_t DEFAULT_MIN_ITER = 0;
const double DEFAULT_ALPHA = 0.6;  // Coefficient for minsum update (check to variable) messages
const double DEFAULT_BETA = 1.0;   // Coefficient for tanh update (check to variable) messages
const double DEFAULT_DELTA = 1.0;  // Coefficient for sum update (variable to check) messages
const bool DEFAULT_STOP_AT_CONVERGENCE = true;  // If true stop once BP converges in serial-box-plus
// If true, sort the error mechanisms (not edges) by their prior error probability, in descending
// order
const bool DEFAULT_SORT_PRIORS = false;
// The fraction of error mechanisms to use in BP. Determines the maximum variable node index
// included, specified as the fraction of all error mechanisms.
const double DEFAULT_VARIABLE_NODE_TRUNCATION_FRACTION = 1.0;

struct BPParams {
  size_t max_iter;
  std::string update_rule;
  std::string schedule;
  bool stop_at_convergence;
  bool sort_priors;
  double variable_node_truncation_fraction;
  float normalization_factor;
  BPParams(size_t max_iter = DEFAULT_MAX_ITER, std::string update_rule = DEFAULT_UPDATE_RULE,
           std::string schedule = DEFAULT_SCHEDULE,
           bool stop_at_convergence = DEFAULT_STOP_AT_CONVERGENCE,
           bool sort_priors = DEFAULT_SORT_PRIORS,
           double variable_node_truncation_fraction = DEFAULT_VARIABLE_NODE_TRUNCATION_FRACTION,
           float normalization_factor = 0.875f)
      : max_iter(max_iter),
        update_rule(update_rule),
        schedule(schedule),
        stop_at_convergence(stop_at_convergence),
        sort_priors(sort_priors),
        variable_node_truncation_fraction(variable_node_truncation_fraction),
        normalization_factor(normalization_factor) {}

  bool operator==(const BPParams& other) const {
    return max_iter == other.max_iter && update_rule == other.update_rule &&
           schedule == other.schedule && stop_at_convergence == other.stop_at_convergence &&
           sort_priors == other.sort_priors &&
           variable_node_truncation_fraction == other.variable_node_truncation_fraction &&
           normalization_factor == other.normalization_factor;
  }
};
}  // namespace bp

#endif /* BELIEFPROPAGATION_BP_BP_PARAMS_H_ */