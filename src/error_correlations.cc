#include "error_correlations.h"

#include <sstream>
#include <stdexcept>

namespace tesseract {

bool ComponentSymptom::operator==(const ComponentSymptom& other) const {
  return detectors == other.detectors && observables == other.observables;
}

bool ComponentSymptom::operator<(const ComponentSymptom& other) const {
  if (detectors != other.detectors) return detectors < other.detectors;
  return observables < other.observables;
}

std::string ImpliedProbability::str() const {
  std::stringstream ss;
  ss << "ImpliedProbability(detectors={";
  for (size_t i = 0; i < affected_symptom.detectors.size(); ++i) {
    ss << affected_symptom.detectors[i] << (i == affected_symptom.detectors.size() - 1 ? "" : ",");
  }
  ss << "}, observables={";
  for (size_t i = 0; i < affected_symptom.observables.size(); ++i) {
    ss << affected_symptom.observables[i]
       << (i == affected_symptom.observables.size() - 1 ? "" : ",");
  }
  ss << "}, prob=" << probability << ")";
  return ss.str();
}

bool ImpliedProbability::operator==(const ImpliedProbability& other) const {
  return affected_symptom == other.affected_symptom &&
         std::abs(probability - other.probability) < 1e-12;
}

bool ImpliedProbability::operator<(const ImpliedProbability& other) const {
  if (!(affected_symptom == other.affected_symptom)) {
    return affected_symptom < other.affected_symptom;
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

    std::vector<ComponentSymptom> components;
    inst.for_separated_targets([&](std::span<const stim::DemTarget> group) {
      ComponentSymptom symptom;
      int component_id = -1;
      for (const auto& target : group) {
        if (target.is_relative_detector_id()) {
          int detector = target.val();
          if (detector < 0 || (size_t)detector >= global_det_to_comp_id.size() ||
              global_det_to_comp_id[detector] < 0) {
            throw std::invalid_argument("Invalid component assignment for detector D" +
                                        std::to_string(detector) + ".");
          }
          int detector_component = global_det_to_comp_id[detector];
          if (component_id != -1 && component_id != detector_component) {
            throw std::invalid_argument(
                "A decomposed error group contains detectors from multiple components.");
          }
          component_id = detector_component;
          symptom.detectors.push_back(detector);
        } else if (target.is_observable_id()) {
          symptom.observables.push_back(target.val());
        }
      }

      if (symptom.detectors.empty()) return;
      std::sort(symptom.detectors.begin(), symptom.detectors.end());
      std::sort(symptom.observables.begin(), symptom.observables.end());
      components.push_back(std::move(symptom));
    });

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
