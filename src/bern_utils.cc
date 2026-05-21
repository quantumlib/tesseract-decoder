#include "bern_utils.h"
#include <cmath>
#include <limits>

namespace tesseract {

double bernoulli_xor(double p1, double p2) {
    return p1 * (1 - p2) + p2 * (1 - p1);
}

double to_weight(double probability) {
    if (probability >= 1.0) {
        return -std::numeric_limits<double>::infinity();
    }
    if (probability <= 0) {
        return std::numeric_limits<double>::infinity();
    }
    return std::log((1 - probability) / probability);
}

} // namespace two_pass_decoding
