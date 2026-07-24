#ifndef BELIEFPROPAGATION_BP_CHECK_UPDATE_H_
#define BELIEFPROPAGATION_BP_CHECK_UPDATE_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "bp/bp_params.h"

namespace bp {

LLR_INT llr_double_to_int(double llr);

LLR_INT prob_double_to_llr_int(double prob);

double llr_int_to_double(LLR_INT discrete_llr);

double gallager_involution(double x);

inline double gallager_involution(double x) {
  return std::log((std::exp(x) + 1) / (std::exp(x) - 1));
}

class GallagerLookupTable {
 public:
  GallagerLookupTable();
  LLR_UINT f(LLR_UINT x) const;

 private:
  LLR_UINT lut[NUM_LUT_BINS];
};

inline LLR_UINT GallagerLookupTable::f(LLR_UINT x) const {
  return x < NUM_LUT_BINS ? lut[x] : 0;
}

double box_plus_correction(double x);

inline double box_plus_correction(double x) {
  return std::log(1 + std::exp(-x));
}

class BoxPlusCorrectionLUT {
 public:
  BoxPlusCorrectionLUT() {
    for (size_t i = 0; i < NUM_LUT_BINS; i++) {
      lut[i] = llr_double_to_int(box_plus_correction(llr_int_to_double(i)));
      lut[i] = std::min((LLR_UINT)(NUM_LUT_BINS - 1), lut[i]);
    }
  }
  LLR_UINT f(LLR_UINT x) const {
    return x < NUM_LUT_BINS ? lut[x] : 0;
  }

 private:
  LLR_UINT lut[NUM_LUT_BINS];
};

LLR_UINT box_plus(LLR_UINT a, LLR_UINT b, const BoxPlusCorrectionLUT& lut);

inline LLR_UINT box_plus(LLR_UINT a, LLR_UINT b, const BoxPlusCorrectionLUT& lut) {
  return std::min(a, b) + lut.f(a + b) -
         lut.f(std::abs(static_cast<LLR_INT>(a) - static_cast<LLR_INT>(b)));
}

double box_plus_fp(double a, double b);

inline double box_plus_fp(double a, double b) {
  return std::min(a, b) + box_plus_correction(a + b) - box_plus_correction(a - b);
}

double box_plus_correction_linear_approx(double x);

inline double box_plus_correction_linear_approx(double x) {
  // Approximation from: https://doi.org/10.1109/ICC.2005.1494430
  return std::max(0.6 - 0.24 * x, 0.0);
}

double box_plus_linear_approx(double a, double b);

inline double box_plus_linear_approx(double a, double b) {
  return std::min(a, b) + box_plus_correction_linear_approx(a + b) -
         box_plus_correction_linear_approx(a - b);
}

}  // namespace bp

#endif /* BELIEFPROPAGATION_BP_CHECK_UPDATE_H_ */