
#include "bp/bp_test_util.h"

namespace bp {
void assert_llrs_close(const std::vector<bp::LLR_INT>& vec_a, const std::vector<bp::LLR_INT>& vec_b,
                       double tol) {
  for (size_t i = 0; i < vec_a.size(); i++) {
    EXPECT_NEAR(bp::llr_int_to_double(vec_a[i]), bp::llr_int_to_double(vec_b[i]), tol);
  }
}
}  // namespace bp
