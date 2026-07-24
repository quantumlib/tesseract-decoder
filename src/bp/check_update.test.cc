#include "bp/check_update.h"

#include <cmath>
#include <iostream>

#include "gtest/gtest.h"

TEST(CheckUpdate, BoxPlusUsingLUT) {
  std::vector<double> llrs = {13.0, 12.4, 12, 13, -5, 5};
  for (size_t i = 0; i < llrs.size() / 2; i++) {
    double llr1 = std::abs(llrs[2 * i]);
    double llr2 = std::abs(llrs[2 * i + 1]);

    double expected = 2 * std::atanh(std::tanh(llr1 / 2) * std::tanh(llr2 / 2));

    bp::BoxPlusCorrectionLUT lut;
    bp::LLR_UINT out = bp::box_plus(bp::llr_double_to_int(llr1), bp::llr_double_to_int(llr2), lut);

    EXPECT_NEAR(bp::llr_int_to_double(out), expected, 0.01);
  }
}

TEST(CheckUpdate, GallagerLUT) {
  bp::GallagerLookupTable lut;
  std::vector<double> llrs = {4.2, 5.5, 6.5, 8.0, 0.1, 0.3};
  for (size_t i = 0; i < llrs.size() / 2; i++) {
    double llr1 = llrs[2 * i];
    double llr2 = llrs[2 * i + 1];
    EXPECT_NEAR(2 * std::atanh(std::tanh(llr1 / 2) * std::tanh(llr2 / 2)),
                bp::llr_int_to_double(
                    lut.f(lut.f(bp::llr_double_to_int(llr1)) + lut.f(bp::llr_double_to_int(llr2)))),
                0.1);
  }
}
