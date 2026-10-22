#ifndef TESSERACT_BP_DECODER_H_
#define TESSERACT_BP_DECODER_H_

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "bp/batched_tanner_graph.h"
#include "bp/bp_params.h"
#include "bp/bp_trace.h"
#include "bp/post_processor.h"
#include "bp/tanner_graph.h"
#include "common.h"
#include "stim.h"

namespace bp {

// Timings and counters for the decode path, complementing OsdProfile (which
// only covers what happens inside the post-processor).
//
// Splitting setup / BP / post-processing matters because the post-processor is
// often a minority of end-to-end runtime, so speedups there are Amdahl-capped.
// As with OsdProfile, counters are plain members: each decode thread owns its
// own TesseractBpDecoder. Use merge() to aggregate across threads.
struct BpDecodeProfile {
  // One-time construction cost: merging indistinguishable errors, building the
  // Tanner graph and its batched counterpart. Amortized over all shots, so it
  // matters most for short runs.
  double t_construct = 0.0;

  // Per-shot costs.
  double t_bp = 0.0;    // message passing only
  double t_post = 0.0;  // post-processor (OSD or hard decision)

  // Batched path only: scattering posteriors_flat (interleaved as
  // [var * BP_BATCH_SIZE + lane]) into a contiguous per-shot vector. This is a
  // strided O(num_vars) copy per shot, so it is timed separately rather than
  // being hidden inside t_bp or t_post.
  double t_deinterleave = 0.0;

  uint64_t num_decode_calls = 0;  // decode() / decode_batch() invocations
  uint64_t num_shots = 0;         // shots decoded
  uint64_t num_converged = 0;     // shots where BP reached a valid syndrome
  uint64_t sum_iters = 0;         // total BP iterations across shots

  // Histogram of BP iterations per shot. The final bucket is a catch-all.
  // Reveals the convergence distribution, which (not max_iter) determines the
  // real cost of BP.
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

  // Decodes a given syndrome (detection events) and returns the logical observable correction.
  std::vector<uint8_t> decode(
      const std::vector<uint64_t>& detection_events,
      const std::shared_ptr<PostProcessor>& post_processor,
      const std::vector<LLR_INT>* custom_priors = nullptr,
      std::vector<size_t>* predicted_errors = nullptr);

  // Decodes a batch of syndromes.
  std::vector<std::vector<uint8_t>> decode_batch(
      const std::vector<std::vector<uint64_t>>& detection_events_batch,
      const std::shared_ptr<PostProcessor>& post_processor,
      const std::vector<std::vector<LLR_INT>>* custom_priors_per_shot = nullptr,
      std::vector<std::vector<size_t>>* predicted_errors_batch = nullptr);

  size_t num_detectors() const {
    return graph_.check_nodes.size();
  }
  size_t num_variables() const {
    return graph_.variable_nodes.size();
  }
  const TannerGraph<LLR_INT>& graph() const {
    return graph_;
  }
  tesseract_decoder::common::Symptom variable_symptom(size_t var_idx) const;
  size_t num_observables() const;

  std::shared_ptr<PostProcessor> create_osd_post_processor(size_t osd_order,
                                                           size_t osd_weight) const;
  std::shared_ptr<PostProcessor> create_tesseract_post_processor() const;
  std::shared_ptr<PostProcessor> create_post_processor() const;

  const stim::DetectorErrorModel& flat_dem() const {
    return flat_dem_;
  }

  // Setup / BP / post-processing timings accumulated by this instance.
  const BpDecodeProfile& profile() const {
    return profile_;
  }
  void reset_profile() {
    profile_ = BpDecodeProfile();
  }

  // Inspection & step-by-step tracing (strictly opt-in; zero overhead on decode() / decode_batch()).
  TannerGraphTopology get_topology() const;
  BpDecodeTrace decode_trace(
      const std::vector<uint64_t>& detection_events,
      const std::shared_ptr<PostProcessor>& post_processor = nullptr,
      const std::vector<LLR_INT>* custom_priors = nullptr);

 private:
  BPParams params_;
  stim::DetectorErrorModel flat_dem_;
  TannerGraph<LLR_INT> graph_;
  BatchedTannerGraph<LLR_INT> batched_graph_;
  std::vector<std::vector<int>> hyperedge_observables_;
  BpDecodeProfile profile_;
};

}  // namespace bp

#endif  // TESSERACT_BP_DECODER_H_
