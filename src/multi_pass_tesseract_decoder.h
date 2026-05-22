#ifndef MULTI_PASS_TESSERACT_DECODER_H
#define MULTI_PASS_TESSERACT_DECODER_H

#include <map>
#include <memory>
#include <vector>

#include "dem_decomposition.h"
#include "error_correlations.h"
#include "stim.h"
#include "tanner_graph.h"
#include "tesseract.h"
#include "utils.h"

namespace tesseract {

enum class SchedulingStrategy {
  Static,  // Current: All components in all passes
  Causal   // Topological: Causal back-propagation
};

class MultiPassTesseractDecoder {
 public:
  MultiPassTesseractDecoder(
      const stim::DetectorErrorModel& dem, size_t num_passes,
      const DetectorClassifier& classifier,
      const TesseractConfig& base_config = TesseractConfig(),
      size_t num_det_orders = 1, DetOrder det_order_method = DetOrder::DetBFS,
      uint64_t seed = 0,
      SchedulingStrategy strategy = SchedulingStrategy::Static);

  std::vector<int> decode(const std::vector<uint64_t>& detections);

  void decode_shots(std::vector<stim::SparseShot>& shots,
                    std::vector<std::vector<int>>& obs_predicted);

  static void validate_annotations(const stim::DetectorErrorModel& dem,
                                   const DetectorClassifier& classifier);

  size_t get_last_shot_num_reweights() const { return last_shot_num_reweights; }
  size_t num_components() const { return component_decoders.size(); }

 private:
  struct LocalReweightRule {
    size_t target_comp_idx;
    size_t target_error_idx;
    double conditional_prob;
  };

  struct ComponentDecoder {
    std::unique_ptr<TesseractDecoder> decoder;
    std::set<int> component_detectors;  // Global indices
    std::map<int, int> global_to_local_det;
    std::vector<double> original_costs;
    std::map<Hyperedge, std::vector<size_t>> symptom_to_error_index;
    std::vector<std::vector<LocalReweightRule>> error_index_to_rules;
    std::vector<size_t> modified_error_indices;
    std::vector<size_t> shot_all_modified_error_indices;
    bool affects_observable = false;
  };

  size_t num_passes;
  SchedulingStrategy strategy;
  size_t total_global_detectors;
  TesseractConfig base_config;
  size_t num_det_orders;
  ::DetOrder det_order_method;
  uint64_t seed;
  size_t last_shot_num_reweights = 0;
  std::map<size_t, std::vector<size_t>> component_predictions;
  std::vector<size_t> modified_component_indices;
  std::vector<size_t> final_pass_active_components;
  std::vector<ComponentDecoder> component_decoders;
  std::vector<std::vector<size_t>> pass_schedule;
  std::vector<int> global_det_to_comp_id;

  void initialize(const stim::DetectorErrorModel& dem,
                  const DetectorClassifier& classifier);
  void build_static_schedule();
  void build_causal_schedule();

  friend class MultiPassTraceVisualizer;
  friend class MultiPassDebugger;
};

class MultiPassDebugger {
 public:
  static const std::vector<std::vector<size_t>>& get_pass_schedule(
      const MultiPassTesseractDecoder& decoder) {
    return decoder.pass_schedule;
  }
  static size_t num_components(const MultiPassTesseractDecoder& decoder) {
    return decoder.component_decoders.size();
  }
  static const TesseractDecoder& get_component_decoder(
      const MultiPassTesseractDecoder& decoder, size_t i) {
    return *decoder.component_decoders[i].decoder;
  }
  static const std::vector<size_t>& get_modified_component_indices(
      const MultiPassTesseractDecoder& decoder) {
    return decoder.modified_component_indices;
  }
  static const MultiPassTesseractDecoder::ComponentDecoder&
  get_component_decoder_full(const MultiPassTesseractDecoder& decoder,
                             size_t i) {
    return decoder.component_decoders[i];
  }
};

}  // namespace tesseract

#endif  // MULTI_PASS_TESSERACT_DECODER_H
