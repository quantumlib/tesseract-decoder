#ifndef TESSERACT_COMMON_H
#define TESSERACT_COMMON_H
#include <unordered_map>

#include "stim.h"

namespace common {
using ObservablesMask = std::uint64_t;

// Represents the effect of an error
struct Symptom {
  std::vector<int> detectors;
  ObservablesMask observables;

  struct hash {
    size_t operator()(const Symptom& s) const {
      size_t hash = 0;
      for (int i : s.detectors) {
        hash += std::hash<int>{}(i);
      }
      hash ^= s.observables;
      return hash;
    }
  };

  std::vector<stim::DemTarget> as_dem_instruction_targets() const;
  bool operator==(const Symptom& other) const {
    return detectors == other.detectors && observables == other.observables;
  }
  std::string str();
};

// Represents an error / weighted hyperedge
struct Error {
  double likelihood_cost;
  double probability;
  Symptom symptom;
  std::vector<bool> dets_array;
  Error() = default;
  Error(
      double likelihood_cost, std::vector<int>& detectors, ObservablesMask observables,
      std::vector<bool>& dets_array)
      : likelihood_cost(likelihood_cost), symptom{detectors, observables}, dets_array(dets_array) {}
  Error(
      double likelihood_cost, double probability, std::vector<int>& detectors,
      ObservablesMask observables, std::vector<bool>& dets_array)
      : likelihood_cost(likelihood_cost),
        probability(probability),
        symptom{detectors, observables},
        dets_array(dets_array) {}
  Error(const stim::DemInstruction& error);
  std::string str();
};

// Makes a new (flattened) dem where identical error mechanisms have been merged.
stim::DetectorErrorModel merge_identical_errors(const stim::DetectorErrorModel& dem);

// Makes a new dem where the probabilities of errors are estimated from the fraction of shots they
// were used in.
stim::DetectorErrorModel dem_from_counts(
    stim::DetectorErrorModel& orig_dem, const std::vector<size_t>& error_counts, size_t num_shots);

}  // namespace common

#endif
