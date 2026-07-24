#ifndef BELIEFPROPAGATION_BP_BP_TYPES_H_
#define BELIEFPROPAGATION_BP_BP_TYPES_H_

#include <type_traits>

#include "bp/bp_params.h"

namespace bp {

// A type trait to determine the appropriate type for storing the magnitude
// of a given Log-Likelihood Ratio (LLR) type.
template <typename T>
struct llr_traits;

// Specialization for LLR_INT: magnitude_type is the corresponding unsigned int.
template <>
struct llr_traits<LLR_INT> {
  using magnitude_type = LLR_UINT;
};

// Specialization for float: magnitude_type is float itself.
template <>
struct llr_traits<float> {
  using magnitude_type = float;
};

// Specialization for double: magnitude_type is double itself.
template <>
struct llr_traits<double> {
  using magnitude_type = double;
};

}  // namespace bp

#endif  // BELIEFPROPAGATION_BP_BP_TYPES_H_