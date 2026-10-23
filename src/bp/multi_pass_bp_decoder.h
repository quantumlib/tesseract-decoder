#ifndef TESSERACT_BP_MULTI_PASS_BP_DECODER_H_
#define TESSERACT_BP_MULTI_PASS_BP_DECODER_H_

#include <array>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "bp/bp_params.h"
#include "bp/osd_post_processor.h"
#include "bp/tesseract_bp_decoder.h"
#include "common.h"
#include "multi_pass/dem_decomposition.h"
#include "multi_pass/error_correlations.h"
#include "stim.h"

namespace bp {

struct LocalReweightRule {
  size_t target_comp_idx;
  size_t target_var_idx;
  LLR_INT reweight_llr;
};

struct MultiPassBpConfig {
  BPParams params;
  size_t num_passes = 2;
  std::vector<int> detector_components;
  bool soft_reweight = true;
};

class MultiPassBpDecoder {
 public:
  MultiPassBpDecoder(const stim::DetectorErrorModel& dem, const MultiPassBpConfig& config);
  MultiPassBpDecoder(const stim::DetectorErrorModel& dem, const BPParams& params,
                     size_t num_passes = 2, const std::vector<int>& detector_components = {});

  std::vector<uint8_t> decode(const std::vector<uint64_t>& detection_events);
  std::vector<std::vector<uint8_t>> decode_batch(
      const std::vector<std::vector<uint64_t>>& detection_events_batch);

  size_t num_observables() const;
  size_t total_detectors() const;
  size_t total_variables() const;

  BpDecodeProfile profile() const;
  OsdProfile osd_profile() const;
  void reset_profile();

  const std::vector<std::vector<size_t>>& pass_schedule() const { return pass_schedule_; }
  const std::vector<int>& global_det_to_comp_id() const { return global_det_to_comp_id_; }

 private:
  void initialize(const stim::DetectorErrorModel& dem, const std::vector<int>& detector_components,
                  const BPParams& params);
  void build_causal_schedule();

  size_t num_passes_;
  bool soft_reweight_ = true;

  struct ComponentDecoder {
    int assignment_label = 0;
    size_t active_detector_count = 0;
    bool affects_observable = false;
    std::unique_ptr<TesseractBpDecoder> decoder;
    std::shared_ptr<OsdPostProcessor> post_processor;
    std::unordered_map<size_t, std::vector<LocalReweightRule>> reweight_rules;
  };

  std::array<ComponentDecoder, 2> component_decoders_;
  std::vector<int> global_det_to_comp_id_;
  std::vector<std::vector<size_t>> pass_schedule_;
};

std::vector<int> classify_detector_components(const stim::DetectorErrorModel& dem);

}  // namespace bp

#endif  // TESSERACT_BP_MULTI_PASS_BP_DECODER_H_
