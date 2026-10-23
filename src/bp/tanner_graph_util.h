#ifndef BP_UTIL_H
#define BP_UTIL_H

#include <cstddef>

namespace bp {

struct BPResult {
  bool converged = false;
  size_t num_iters = 0;
  size_t unsat_checks = 0;
};

}  // namespace bp

#endif  // BP_UTIL_H
