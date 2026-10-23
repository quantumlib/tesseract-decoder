#ifndef BELIEFPROPAGATION_BP_BP_PARAMS_H_
#define BELIEFPROPAGATION_BP_BP_PARAMS_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace bp {

using LLR_INT = std::int32_t;
using LLR_UINT = std::uint32_t;

constexpr LLR_INT LLR_FP_BITS = 8;
constexpr double LLR_FP_SCALE = 256.0;
constexpr LLR_UINT MAX_SAFE_LLR_INT = 1u << 23;
constexpr double MIN_PROB = 1e-15;

constexpr LLR_INT NUM_LUT_BINS = 1 << 14;
constexpr double MAX_BP_LUT_LL = 20.0;

inline LLR_INT llr_double_to_int(double llr) {
  LLR_INT discrete_llr = static_cast<LLR_INT>(std::round(llr * NUM_LUT_BINS / MAX_BP_LUT_LL));
  if (std::abs(discrete_llr) > NUM_LUT_BINS) {
    discrete_llr = discrete_llr < 0 ? -NUM_LUT_BINS : NUM_LUT_BINS;
  }
  return discrete_llr;
}

inline LLR_INT prob_double_to_llr_int(double prob) {
  return llr_double_to_int(std::log((1.0 - prob) / prob));
}

inline double llr_int_to_double(LLR_INT discrete_llr) {
  return discrete_llr * MAX_BP_LUT_LL / NUM_LUT_BINS;
}

constexpr size_t DEFAULT_MAX_ITER = 20;
constexpr bool DEFAULT_STOP_AT_CONVERGENCE = true;
constexpr bool DEFAULT_RANDOM_SCHEDULE = false;
constexpr uint64_t DEFAULT_RANDOM_SEED = 0;

struct BPParams {
  size_t max_iter = DEFAULT_MAX_ITER;
  bool stop_at_convergence = DEFAULT_STOP_AT_CONVERGENCE;
  float normalization_factor = 0.875f;
  bool random_schedule = DEFAULT_RANDOM_SCHEDULE;
  uint64_t random_seed = DEFAULT_RANDOM_SEED;
  float osd_truncation_factor = 1.2f;
  bool osd_early_exit = true;
  int osd_order = -1;
  int osd_weight = 0;
  bool use_soft_reweighting = true;

  static BPParams Osd(size_t max_iter = DEFAULT_MAX_ITER, int osd_order = 24, int osd_weight = 2) {
    BPParams p;
    p.max_iter = max_iter;
    p.osd_order = osd_order;
    p.osd_weight = osd_weight;
    return p;
  }

  static BPParams Fast(size_t max_iter = 10, int osd_order = 10, int osd_weight = 0) {
    BPParams p;
    p.max_iter = max_iter;
    p.osd_order = osd_order;
    p.osd_weight = osd_weight;
    p.osd_truncation_factor = 1.2f;
    p.osd_early_exit = true;
    return p;
  }

  bool operator==(const BPParams& other) const = default;
};

}  // namespace bp

#endif  // BELIEFPROPAGATION_BP_BP_PARAMS_H_
