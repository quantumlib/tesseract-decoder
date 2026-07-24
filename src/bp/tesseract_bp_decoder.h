#ifndef TESSERACT_BP_DECODER_H_
#define TESSERACT_BP_DECODER_H_

#include <memory>
#include <vector>

#include "bp/batched_tanner_graph.h"
#include "bp/bp_params.h"
#include "bp/post_processor.h"
#include "bp/tanner_graph.h"
#include "stim.h"

namespace bp {

class TesseractBpDecoder {
 public:
  TesseractBpDecoder(const stim::DetectorErrorModel& dem, const BPParams& params);

  // Decodes a given syndrome (detection events) and returns the logical observable correction.
  std::vector<uint8_t> decode(const std::vector<uint64_t>& detection_events,
                              const std::shared_ptr<PostProcessor>& post_processor);

  // Decodes a batch of syndromes.
  std::vector<std::vector<uint8_t>> decode_batch(
      const std::vector<std::vector<uint64_t>>& detection_events_batch,
      const std::shared_ptr<PostProcessor>& post_processor);

  size_t num_detectors() const {
    return graph_.check_nodes.size();
  }
  size_t num_observables() const;

  std::shared_ptr<PostProcessor> create_osd_post_processor(size_t osd_order,
                                                           size_t osd_weight) const;

 private:
  BPParams params_;
  TannerGraph<LLR_INT> graph_;
  BatchedTannerGraph<LLR_INT> batched_graph_;
  std::vector<std::vector<int>> hyperedge_observables_;
};

}  // namespace bp

#endif  // TESSERACT_BP_DECODER_H_
