#ifndef TESSERACT_BP_TESSERACT_POST_PROCESSOR_H_
#define TESSERACT_BP_TESSERACT_POST_PROCESSOR_H_

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "bp/bp_params.h"
#include "bp/bp_trace.h"
#include "bp/osd_post_processor.h"
#include "bp/post_processor.h"
#include "bp/tanner_graph.h"
#include "stim.h"
#include "tesseract.h"

namespace bp {

class TesseractPostProcessor : public PostProcessor {
 public:
  TesseractPostProcessor(const stim::DetectorErrorModel& merged_dem,
                         const TannerGraph<LLR_INT>& graph,
                         const std::vector<std::vector<int>>& hyperedge_observables,
                         const BPParams& params);
  ~TesseractPostProcessor() override = default;

  std::vector<uint8_t> process(
      const BPResult& bp_result, const std::vector<LLR_INT>& posteriors,
      const std::vector<uint64_t>& detection_events,
      const std::vector<std::vector<int>>& hyperedge_observables,
      std::vector<size_t>* predicted_errors = nullptr,
      const std::vector<LLR_INT>* custom_priors = nullptr) override;

  // Trace-enabled post-processing (only invoked when tracing is explicitly requested).
  std::vector<uint8_t> process_with_trace(
      const BPResult& bp_result, const std::vector<LLR_INT>& posteriors,
      const std::vector<uint64_t>& detection_events,
      const std::vector<std::vector<int>>& hyperedge_observables,
      TesseractTrace& trace,
      OsdTrace* osd_fallback_trace = nullptr,
      std::vector<size_t>* predicted_errors = nullptr,
      const std::vector<LLR_INT>* custom_priors = nullptr);

  const OsdProfile& profile() const {
    return profile_;
  }
  void reset_profile() {
    profile_ = OsdProfile();
    if (osd_fallback_) {
      osd_fallback_->reset_profile();
    }
  }

 private:
  const TannerGraph<LLR_INT>& graph_;
  BPParams params_;
  mutable tesseract_decoder::TesseractDecoder tesseract_decoder_;
  std::unique_ptr<OsdPostProcessor> osd_fallback_;
  mutable OsdProfile profile_;
};

}  // namespace bp

#endif  // TESSERACT_BP_TESSERACT_POST_PROCESSOR_H_
