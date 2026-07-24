#include "bp/check_update.h"

#include <algorithm>
#include <iostream>

namespace bp {

LLR_INT llr_double_to_int(double llr) {
  LLR_INT discrete_llr = round(llr * NUM_LUT_BINS / MAX_BP_LUT_LL);
  if (abs(discrete_llr) > NUM_LUT_BINS) {
    discrete_llr = discrete_llr < 0 ? -NUM_LUT_BINS : NUM_LUT_BINS;
  }
  return discrete_llr;
}

LLR_INT prob_double_to_llr_int(double prob) {
  return llr_double_to_int(std::log((1 - prob) / prob));
}

double llr_int_to_double(LLR_INT discrete_llr) {
  return discrete_llr * MAX_BP_LUT_LL / NUM_LUT_BINS;
}

GallagerLookupTable::GallagerLookupTable() {
  lut[0] = llr_double_to_int(gallager_involution(0.25 * MAX_BP_LUT_LL / NUM_LUT_BINS));
  for (size_t i = 1; i < NUM_LUT_BINS; i++) {
    lut[i] = llr_double_to_int(gallager_involution(llr_int_to_double(i)));
    lut[i] = std::min((bp::LLR_UINT)(NUM_LUT_BINS - 1), lut[i]);
  }
}

}  // namespace bp