// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TESSERACT_DECODER_H
#define TESSERACT_DECODER_H

#include <queue>
#include <string>
#include <unordered_map>
#include <vector>
#include <unordered_set>

#include "common.h"
#include "stim.h"
#include "utils.h"

constexpr size_t INF_DET_BEAM = std::numeric_limits<uint16_t>::max();

struct TesseractConfig {
  stim::DetectorErrorModel dem;
  int det_beam = INF_DET_BEAM;
  bool beam_climbing = false;
  bool no_revisit_dets = false;
  bool at_most_two_errors_per_detector = false;
  bool verbose;
  size_t pqlimit = std::numeric_limits<size_t>::max();
  std::vector<std::vector<size_t>> det_orders;
  double det_penalty = 0;
  bool with_openmp = false;
  int beam_climbing_openmp_threads = 0;
  int det_orders_openmp_threads = 0;
};

class Node {
 public:
  std::vector<size_t> errs;
  std::vector<char> dets;
  double cost;
  size_t num_dets;
  std::vector<char> blocked_errs;

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

  // Returns the decoded errors for these detection events.
  std::pair<std::vector<size_t>, bool> decode_to_errors(const std::vector<uint64_t>& detections);

  // Returns the decoded errors for these detection events, using a specified detector ordering index and detection beam.
  std::pair<std::vector<size_t>, bool> decode_to_errors(const std::vector<uint64_t>& detections, size_t det_order, size_t det_beam);

  // Returns the bitwise XOR of all the observables bitmasks of all errors in
  // the predicted errors buffer.
  common::ObservablesMask mask_from_errors(const std::vector<size_t>& predicted_errors);

  // Returns the sum of the likelihood costs (minus-log-likelihood-ratios) of
  // all errors in the predicted errors buffer.
  double cost_from_errors(const std::vector<size_t>& predicted_errors);
  common::ObservablesMask decode(const std::vector<uint64_t>& detections);

  void decode_shots(std::vector<stim::SparseShot>& shots, std::vector<common::ObservablesMask>& obs_predicted);

  std::vector<common::Error> errors;

 private:
  std::vector<std::vector<int>> d2e;
  std::vector<std::vector<int>> eneighbors;
  std::vector<std::vector<int>> edets;
  size_t num_detectors;
  size_t num_errors;

  void initialize_structures(size_t num_detectors);
  double get_detcost(size_t d, const std::vector<char>& blocked_errs, const std::vector<size_t>& det_counts) const;
  void to_node(const QNode& qnode, const std::vector<char>& shot_dets, size_t det_order, Node& node) const;
};

#endif  // TESSERACT_DECODER_H