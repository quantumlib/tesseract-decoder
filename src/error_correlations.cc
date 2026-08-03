#include "error_correlations.h"

#include <iostream>
#include <sstream>

namespace tesseract {

std::string ImpliedProbability::str() const {
  std::stringstream ss;
  ss << "ImpliedProbability(affected={";
  for (size_t i = 0; i < affected_hyperedge.size(); ++i) {
    ss << affected_hyperedge[i] << (i == affected_hyperedge.size() - 1 ? "" : ",");
  }
  ss << "}, prob=" << probability << ")";
  return ss.str();
}

bool ImpliedProbability::operator==(const ImpliedProbability& other) const {
  return affected_hyperedge == other.affected_hyperedge &&
         std::abs(probability - other.probability) < 1e-12;
}

bool ImpliedProbability::operator<(const ImpliedProbability& other) const {
  if (affected_hyperedge != other.affected_hyperedge) {
    return affected_hyperedge < other.affected_hyperedge;
  }
  return probability < other.probability;
}

JointProbsMap get_hyperedge_joint_probabilities(const stim::DetectorErrorModel& dem,
                                                const std::vector<int>& global_det_to_comp_id) {
  JointProbsMap joint_probs;
  auto flattened = dem.flattened();

  for (const auto& inst : flattened.instructions) {
    if (inst.type != stim::DemInstructionType::DEM_ERROR) continue;

    double p = inst.arg_data[0];

    std::map<int, Hyperedge> comp_targets;
    for (const auto& target : inst.target_data) {
      if (target.is_relative_detector_id()) {
        int d = target.val();
        int cid =
            (d >= 0 && (size_t)d < global_det_to_comp_id.size()) ? global_det_to_comp_id[d] : -1;
        if (cid != -1) comp_targets[cid].push_back(d);
      }
    }

    std::vector<Hyperedge> components;
    for (auto& [cid, h] : comp_targets) {
      std::sort(h.begin(), h.end());
      components.push_back(h);
    }

    // 1. Marginal probabilities (diagonal)
    for (const auto& h : components) {
      if (joint_probs[h].find(h) == joint_probs[h].end()) {
        joint_probs[h][h] = 0.0;
      }
      // P(A) = P(A) XOR p
      joint_probs[h][h] = joint_probs[h][h] * (1 - p) + p * (1 - joint_probs[h][h]);
    }

    // 2. Joint probabilities (off-diagonal)
    // For a bridging error p connecting A and B, P(A and B) += p (approx)
    // Actually, the joint probability is accurately tracked via the same XOR logic
    // if we assume independence of other error mechanisms.
    if (components.size() > 1) {
      for (size_t i = 0; i < components.size(); ++i) {
        for (size_t j = 0; j < components.size(); ++j) {
          if (i == j) continue;
          const auto& hi = components[i];
          const auto& hj = components[j];
          if (joint_probs[hi].find(hj) == joint_probs[hi].end()) {
            joint_probs[hi][hj] = 0.0;
          }
          // For small p, joint probability P(A and B) is roughly the sum of p's of bridging errors
          joint_probs[hi][hj] = joint_probs[hi][hj] * (1 - p) + p * (1 - joint_probs[hi][hj]);
        }
      }
    }
  }

  return joint_probs;
}

ImpliedProbsMap get_implied_hyperedge_probabilities(const JointProbsMap& joint_probs) {
  ImpliedProbsMap implied_probs;

  for (const auto& [causal, affected_map] : joint_probs) {
    double p_causal = 0.0;
    auto it_self = affected_map.find(causal);
    if (it_self != affected_map.end()) {
      p_causal = it_self->second;
    }

    if (p_causal <= 0 || p_causal >= 1.0) continue;

    for (const auto& [affected, p_joint] : affected_map) {
      if (causal == affected) continue;

      // Conditional Probability P(affected | causal) = P(affected and causal) / P(causal)
      double p_conditional = p_joint / p_causal;

      // Cap to 1.0 (numerical precision)
      if (p_conditional > 1.0) p_conditional = 1.0;
      if (p_conditional < 0.0) p_conditional = 0.0;

      implied_probs[causal].push_back({affected, p_conditional});
    }
  }

  return implied_probs;
}

ImpliedProbsMap process_dem_correlations(const stim::DetectorErrorModel& dem,
                                         const std::vector<int>& global_det_to_comp_id) {
  auto joint = get_hyperedge_joint_probabilities(dem, global_det_to_comp_id);
  return get_implied_hyperedge_probabilities(joint);
}

}  // namespace tesseract
