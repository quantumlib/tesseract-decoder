#ifndef BELIEFPROPAGATION_BP_BP_TEST_UTIL_H_
#define BELIEFPROPAGATION_BP_BP_TEST_UTIL_H_

#include <cmath>
#include <vector>

#include "bp/check_update.h"
#include "gtest/gtest.h"

namespace bp {
void assert_llrs_close(const std::vector<bp::LLR_INT>& vec_a, const std::vector<bp::LLR_INT>& vec_b,
                       double tol);
}

#endif /* BELIEFPROPAGATION_BP_BP_TEST_UTIL_H_ */