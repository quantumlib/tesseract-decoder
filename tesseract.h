#ifndef TESSERACT_DECODER_H
#define TESSERACT_DECODER_H

#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "stim.h"
#include "utils.h"

constexpr size_t INF_DET_BEAM = std::numeric_limits<uint16_t>::max();

struct TesseractConfig {
  stim::DetectorErrorModel dem;
  int det_beam = INF_DET_BEAM;
  bool beam_climbing = false;
  bool detcost_prefer_clean = false;
  bool at_most_two_errors_per_detector = false;
  bool verbose;
  size_t pqlimit = std::numeric_limits<size_t>::max();
  std::vector<std::vector<size_t>> det_orders;
};

class Node {
 public:
  std::vector<size_t> errs;
  std::vector<bool> dets;
  double cost;
  size_t num_dets;
  std::vector<bool> blocked_errs;

  bool operator>(const Node& other) const;
};

class QNode {
 public:
  double cost;
  size_t num_dets;
  std::vector<size_t> errs;

  bool operator>(const QNode& other) const;
};

struct TesseractDecoder {
  TesseractConfig config;
  explicit TesseractDecoder(TesseractConfig config);

  // Clears the predicted_errors_buffer and fills it with the decoded errors for these detection
  // events.
  void decode_to_errors(const std::vector<size_t>& detections);

  // Clears the predicted_errors_buffer and fills it with the decoded errors for these detection
  // events, using a specified detector ordering index.
  void decode_to_errors(const std::vector<size_t>& detections, size_t det_order);
  // Returns the bitwise XOR of all the observables bitmasks of all errors in the predicted errors
  // buffer.
  common::ObservablesMask mask_from_errors(const std::vector<size_t>& predicted_errors);
  // Returns the sum of the likelihood costs (minus-log-likelihood-ratios) of all errors in the
  // predicted errors buffer.
  double cost_from_errors(const std::vector<size_t>& predicted_errors);
  common::ObservablesMask decode(const std::vector<size_t>& detections);

  void decode_shots(
      std::vector<stim::SparseShot>& shots, std::vector<common::ObservablesMask>& obs_predicted);

  bool low_confidence_flag = false;
  std::vector<size_t> predicted_errors_buffer;

  int det_beam;
  std::vector<common::Error> errors;

 private:
  std::vector<std::vector<int>> d2e;
  std::vector<std::vector<int>> eneighbors;
  std::vector<std::vector<int>> edets;
  size_t num_detectors;
  size_t num_errors;

  void initialize_structures(size_t num_detectors);
  double get_detcost(
      size_t d, const std::vector<bool>& blocked_errs, const std::vector<size_t>& det_counts,
      const std::vector<bool>& dets) const;
  Node to_node(const QNode& qnode, const std::vector<bool>& shot_dets, size_t det_order) const;
};

#endif  // TESSERACT_DECODER_H
