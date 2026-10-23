#ifndef TESSERACT_BP_DECODER_H_
#define TESSERACT_BP_DECODER_H_

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "bp/bp_params.h"
#include "bp/osd_post_processor.h"
#include "bp/tanner_graph.h"
#include "common.h"
#include "stim.h"

namespace bp {

struct BpDecodeProfile {
  double t_construct = 0.0;
  double t_bp = 0.0;
  double t_post = 0.0;
  double t_deinterleave = 0.0;

  uint64_t num_decode_calls = 0;
  uint64_t num_shots = 0;
  uint64_t num_converged = 0;
  uint64_t sum_iters = 0;

  static constexpr size_t kNumIterBuckets = 256;
  std::array<uint64_t, kNumIterBuckets> iter_hist{};
  std::array<uint64_t, kNumIterBuckets> conv_iter_hist{};

  void record_iters(size_t iters, bool converged = false) {
    sum_iters += iters;
    size_t b = iters < kNumIterBuckets - 1 ? iters : kNumIterBuckets - 1;
    iter_hist[b]++;
    if (converged) {
      conv_iter_hist[b]++;
    }
  }

  void merge(const BpDecodeProfile& other) {
    t_construct += other.t_construct;
    t_bp += other.t_bp;
    t_post += other.t_post;
    t_deinterleave += other.t_deinterleave;
    num_decode_calls += other.num_decode_calls;
    num_shots += other.num_shots;
    num_converged += other.num_converged;
    sum_iters += other.sum_iters;
    for (size_t i = 0; i < kNumIterBuckets; ++i) {
      iter_hist[i] += other.iter_hist[i];
      conv_iter_hist[i] += other.conv_iter_hist[i];
    }
  }
};

class TesseractBpDecoder {
 public:
  TesseractBpDecoder(const stim::DetectorErrorModel& dem, const BPParams& params);

  std::vector<uint8_t> decode(
      const std::vector<uint64_t>& detection_events,
      const std::shared_ptr<OsdPostProcessor>& post_processor = nullptr,
      const std::vector<LLR_INT>* custom_priors = nullptr,
      std::vector<size_t>* predicted_errors = nullptr);

  std::vector<std::vector<uint8_t>> decode_batch(
      const std::vector<std::vector<uint64_t>>& detection_events_batch,
      const std::shared_ptr<OsdPostProcessor>& post_processor = nullptr,
      const std::vector<std::vector<LLR_INT>>* custom_priors_per_shot = nullptr,
      std::vector<std::vector<size_t>>* predicted_errors_batch = nullptr);

  size_t num_detectors() const { return graph_.num_checks; }
  size_t num_variables() const { return graph_.num_vars; }
  const TannerGraph& graph() const { return graph_; }
  tesseract_decoder::common::Symptom variable_symptom(size_t var_idx) const;
  size_t num_observables() const;

  std::shared_ptr<OsdPostProcessor> create_osd_post_processor(size_t osd_order,
                                                              size_t osd_weight) const;
  std::shared_ptr<OsdPostProcessor> create_post_processor() const;

  const stim::DetectorErrorModel& flat_dem() const { return flat_dem_; }

  const BpDecodeProfile& profile() const { return profile_; }
  void reset_profile() { profile_ = BpDecodeProfile(); }

 private:
  BPParams params_;
  stim::DetectorErrorModel flat_dem_;
  TannerGraph graph_;
  std::vector<std::vector<int>> hyperedge_observables_;
  BpDecodeProfile profile_;
};

}  // namespace bp

#endif  // TESSERACT_BP_DECODER_H_
