#ifndef ERROR_CORRELATIONS_H
#define ERROR_CORRELATIONS_H

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "stim.h"

namespace tesseract_decoder {

struct ComponentSymptom {
  std::vector<int> detectors;
  std::vector<int> observables;

  bool operator==(const ComponentSymptom& other) const;
  bool operator<(const ComponentSymptom& other) const;
};

/**
 * Represents a probability adjustment for an affected component symptom.
 */
struct ImpliedProbability {
  ComponentSymptom affected_symptom;
  double probability;  // Represents the conditional probability P(affected | causal)

  std::string str() const;
  bool operator==(const ImpliedProbability& other) const;
  bool operator<(const ImpliedProbability& other) const;
};

using JointProbsMap = std::map<ComponentSymptom, std::map<ComponentSymptom, double>>;
using ImpliedProbsMap = std::map<ComponentSymptom, std::vector<ImpliedProbability>>;

/**
 * Calculates marginal and joint probabilities for component symptoms in a decomposed DEM.
 * Separated groups in one error instruction retain the original physical correlation.
 */
JointProbsMap get_hyperedge_joint_probabilities(const stim::DetectorErrorModel& dem,
                                                const std::vector<int>& global_det_to_comp_id);

/**
 * Calculates conditional probabilities from joint probabilities.
 */
ImpliedProbsMap get_implied_hyperedge_probabilities(const JointProbsMap& joint_probs);

/**
 * Complete workflow for analyzing correlations within a stim::DetectorErrorModel.
 */
ImpliedProbsMap process_dem_correlations(const stim::DetectorErrorModel& dem,
                                         const std::vector<int>& global_det_to_comp_id);

}  // namespace tesseract_decoder

#endif  // ERROR_CORRELATIONS_H
