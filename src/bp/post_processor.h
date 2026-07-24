#ifndef TESSERACT_BP_POST_PROCESSOR_H_
#define TESSERACT_BP_POST_PROCESSOR_H_

#include <cstdint>
#include <vector>

#include "bp/bp_types.h"
#include "bp/tanner_graph_util.h"

namespace bp {

class PostProcessor {
 public:
  virtual ~PostProcessor() = default;

  virtual std::vector<uint8_t> process(
      const BPResult& bp_result, const std::vector<LLR_INT>& posteriors,
      const std::vector<uint64_t>& detection_events,
      const std::vector<std::vector<int>>& hyperedge_observables) = 0;
};

}  // namespace bp

#endif  // TESSERACT_BP_POST_PROCESSOR_H_
