#include "bp/hard_decision_post_processor.h"

#include <algorithm>
#include <iostream>

namespace bp {

std::vector<uint8_t> HardDecisionPostProcessor::process(
    [[maybe_unused]] const BPResult& bp_result, const std::vector<LLR_INT>& posteriors,
    [[maybe_unused]] const std::vector<uint64_t>& detection_events,
    const std::vector<std::vector<int>>& hyperedge_observables) {
  // Dynamically find the maximum observable index to size the output vector.
  size_t max_obs_index = 0;
  bool found_any_obs = false;
  for (const auto& obs_list : hyperedge_observables) {
    for (int obs : obs_list) {
      if (obs >= 0) {
        max_obs_index = std::max(max_obs_index, (size_t)obs);
        found_any_obs = true;
      }
    }
  }

  size_t num_observables = found_any_obs ? max_obs_index + 1 : 0;
  std::vector<uint8_t> obs_result(num_observables, 0);

  for (size_t i = 0; i < posteriors.size(); i++) {
    if (posteriors[i] < 0) {  // Error occurred (negative LLR)
      if (i < hyperedge_observables.size()) {
        for (int obs : hyperedge_observables[i]) {
          if (obs >= 0) {
            obs_result[obs] ^= 1;
          }
        }
      }
    }
  }
  return obs_result;
}

}  // namespace bp
