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

#ifndef TESSERACT_COMMON_H
#define TESSERACT_COMMON_H
#include <unordered_map>

#include "stim.h"

namespace common {

// Represents the effect of an error
struct Symptom {
  std::vector<int> detectors;
  std::vector<int> observables;

  struct hash {
    size_t operator()(const Symptom& s) const {
      size_t hash = 0;
      for (int i : s.detectors) {
        hash += std::hash<int>{}(i);
      }
      for (int i : s.observables) {
        hash += std::hash<int>{}(i);
      }
      return hash;
    }
  };

  std::vector<stim::DemTarget> as_dem_instruction_targets() const;
  bool operator==(const Symptom& other) const {
    return detectors == other.detectors && observables == other.observables;
  }
  std::string str() const;
};

// Represents an error / weighted hyperedge
struct Error {
  double likelihood_cost;
  Symptom symptom;
  Error() = default;
  Error(double likelihood_cost, std::vector<int>& detectors, std::vector<int> observables)
      : likelihood_cost(likelihood_cost), symptom{detectors, observables} {}
  Error(const stim::DemInstruction& error);
  std::string str() const;

  // Get/calculate the probability from the likelihood cost.
  double get_probability() const;

  // Set/calculate the likelihood cost from a probability.
  void set_with_probability(double p);
};

// Makes a new (flattened) dem where identical error mechanisms have been
// merged.
// `error_index_map[old_error_index]` gives the corresponding merged DEM error
// index in the returned DEM.
stim::DetectorErrorModel merge_indistinguishable_errors(const stim::DetectorErrorModel& dem,
                                                        std::vector<size_t>& error_index_map);

// Returns a copy of the given error model with any zero-probability DEM_ERROR
// instructions removed.
// `error_index_map[old_error_index]` gives the corresponding retained DEM error
// index in the returned DEM, or `std::numeric_limits<size_t>::max()` if the
// error was removed.
stim::DetectorErrorModel remove_zero_probability_errors(const stim::DetectorErrorModel& dem,
                                                        std::vector<size_t>& error_index_map);

// Updates the base_map by chaining it with next_map.
// base_map[i] = next_map[base_map[i]]
void chain_error_maps(std::vector<size_t>& base_map, const std::vector<size_t>& next_map);

// Inverts the error_map to create a mapping from output error indices back to
// the first original error index that maps to it.
std::vector<size_t> invert_error_map(const std::vector<size_t>& error_map,
                                     size_t num_output_errors);

// Makes a new dem where the probabilities of errors are estimated from the
// fraction of shots they were used in.
// Throws std::invalid_argument if `orig_dem` contains zero-probability errors;
// call remove_zero_probability_errors first.
stim::DetectorErrorModel dem_from_counts(stim::DetectorErrorModel& orig_dem,
                                         const std::vector<size_t>& error_counts, size_t num_shots);

/// Computes the weight of an edge resulting from merging edges with weight `a' and weight `b',
/// assuming each edge weight is a log-likelihood ratio log((1-p)/p) associated with the probability
/// p of an error occurring on the edge, and that the error mechanisms associated with the two edges
/// being merged are independent.
double merge_weights(double a, double b);

}  // namespace common

#endif
