#ifndef SIMPLEX_HPP
#define SIMPLEX_HPP
#include <vector>

#include "common.h"
#include "stim.h"

struct HighsModel;
struct Highs;
enum class HighsStatus;

struct SimplexConfig {
  stim::DetectorErrorModel dem;
  bool parallelize = false;
  size_t window_length = 0;
  size_t window_slide_length = 0;
  bool verbose = false;
  bool windowing_enabled() { return (window_length != 0); }
};

struct SimplexDecoder {
  SimplexConfig config;
  std::vector<common::Error> errors;
  size_t num_detectors = 0;
  size_t num_observables = 0;
  std::vector<size_t> predicted_errors_buffer;
  std::vector<common::ObservablesMask> error_masks;
  std::vector<std::vector<size_t>> start_time_to_errors;
  std::vector<std::vector<size_t>> end_time_to_errors;

  std::unique_ptr<HighsModel> model;
  std::unique_ptr<Highs> highs;
  std::unique_ptr<HighsStatus> return_status;

  // For consistency with Tesseract, we provide a low confidence flag on Simplex decoder which is
  // always set to false
  const bool low_confidence_flag = false;

  SimplexDecoder(SimplexConfig config);

  void init_ilp();

  // Clears the predicted_errors_buffer and fills it with the decoded errors for these detection
  // events.
  void decode_to_errors(const std::vector<size_t>& detections);
  // Returns the bitwise XOR of all the observables bitmasks of all errors in the predicted errors
  // buffer.
  common::ObservablesMask mask_from_errors(const std::vector<size_t>& predicted_errors);
  // Returns the sum of the likelihood costs (minus-log-likelihood-ratios) of all errors in the
  // predicted errors buffer.
  double cost_from_errors(const std::vector<size_t>& predicted_errors);
  common::ObservablesMask decode(const std::vector<size_t>& detections);

  void decode_shots(
      std::vector<stim::SparseShot>& shots, std::vector<common::ObservablesMask>& obs_predicted);

  ~SimplexDecoder();
};

#endif  // SIMPLEX_HPP
