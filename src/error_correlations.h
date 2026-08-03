#ifndef ERROR_CORRELATIONS_H
#define ERROR_CORRELATIONS_H

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "stim.h"

namespace tesseract {

/**
 * Represents a probability adjustment for an affected hyperedge given a causal hyperedge.
 */
struct ImpliedProbability {
  std::vector<int> affected_hyperedge;
  double probability;  // Represents the conditional probability P(affected | causal)

  std::string str() const;
  bool operator==(const ImpliedProbability& other) const;
  bool operator<(const ImpliedProbability& other) const;
};

// Type alias for hyperedge (sorted detector indices)
using Hyperedge = std::vector<int>;
// Type alias for joint probabilities map: causal_hyperedge -> {affected_hyperedge -> joint_prob}
using JointProbsMap = std::map<Hyperedge, std::map<Hyperedge, double>>;
// Type alias for implied probabilities map: causal_hyperedge -> list of conditional probability
// updates
using ImpliedProbsMap = std::map<Hyperedge, std::vector<ImpliedProbability>>;

/**
 * Calculates marginal and joint probabilities for hyperedges in a DEM.
 * Note: Assumes the input DEM has NOT been decomposed yet, as we need bridging errors
 * to find joint probabilities.
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

}  // namespace tesseract

#endif  // ERROR_CORRELATIONS_H
