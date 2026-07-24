#ifndef TESSERACT_BP_OSD_POST_PROCESSOR_H_
#define TESSERACT_BP_OSD_POST_PROCESSOR_H_

#include "bp/post_processor.h"
#include "bp/tanner_graph.h"

namespace bp {

class OsdPostProcessor : public PostProcessor {
 public:
  // osd_order specifies the size of the information set subset to perturb.
  // osd_weight specifies the maximum Hamming weight of perturbations to explore.
  OsdPostProcessor(const TannerGraph<LLR_INT>& graph, size_t osd_order = 0, size_t osd_weight = 0)
      : graph_(graph), osd_order_(osd_order), osd_weight_(osd_weight) {}
  virtual ~OsdPostProcessor() = default;

  std::vector<uint8_t> process(const BPResult& bp_result, const std::vector<LLR_INT>& posteriors,
                               const std::vector<uint64_t>& detection_events,
                               const std::vector<std::vector<int>>& hyperedge_observables) override;

 private:
  const TannerGraph<LLR_INT>& graph_;
  size_t osd_order_;
  size_t osd_weight_;
};

}  // namespace bp

#endif  // TESSERACT_BP_OSD_POST_PROCESSOR_H_
