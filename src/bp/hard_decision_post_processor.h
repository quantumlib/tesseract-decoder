#ifndef TESSERACT_BP_HARD_DECISION_POST_PROCESSOR_H_
#define TESSERACT_BP_HARD_DECISION_POST_PROCESSOR_H_

#include "bp/post_processor.h"

namespace bp {

class HardDecisionPostProcessor : public PostProcessor {
 public:
  HardDecisionPostProcessor() = default;
  virtual ~HardDecisionPostProcessor() = default;

  std::vector<uint8_t> process(
      const BPResult& bp_result, const std::vector<LLR_INT>& posteriors,
      const std::vector<uint64_t>& detection_events,
      const std::vector<std::vector<int>>& hyperedge_observables,
      std::vector<size_t>* predicted_errors = nullptr,
      const std::vector<LLR_INT>* custom_priors = nullptr) override;
};

}  // namespace bp

#endif  // TESSERACT_BP_HARD_DECISION_POST_PROCESSOR_H_
