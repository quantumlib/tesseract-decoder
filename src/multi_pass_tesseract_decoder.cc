#include "multi_pass_tesseract_decoder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <set>
#include <sstream>

#include "dem_decomposition.h"

namespace tesseract {

MultiPassTesseractDecoder::MultiPassTesseractDecoder(
    const stim::DetectorErrorModel& dem, size_t num_passes,
    const DetectorClassifier& classifier, const TesseractConfig& base_config,
    size_t num_det_orders, DetOrder det_order_method, uint64_t seed,
    SchedulingStrategy strategy)
    : num_passes(num_passes),
      strategy(strategy),
      total_global_detectors(dem.count_detectors()),
      base_config(base_config),
      num_det_orders(num_det_orders),
      det_order_method(det_order_method),
      seed(seed) {
  initialize(dem, classifier);
}

void MultiPassTesseractDecoder::validate_annotations(
    const stim::DetectorErrorModel& dem, const DetectorClassifier& classifier) {
  stim::DetectorErrorModel flattened = dem.flattened();
  size_t total_global_detectors = (size_t)flattened.count_detectors();

  std::set<uint64_t> all_ids;
  std::map<uint64_t, std::string> tags;
  for (const auto& inst : flattened.instructions) {
    if (inst.type == stim::DemInstructionType::DEM_DETECTOR) {
      uint64_t d = inst.target_data[0].val();
      all_ids.insert(d);
      tags[d] = inst.tag;
    }
  }
  auto coords_map = flattened.get_detector_coordinates(all_ids);

  std::set<int> unique_classes;
  for (size_t i = 0; i < total_global_detectors; ++i) {
    std::vector<double> c =
        coords_map.count(i) ? coords_map.at(i) : std::vector<double>{};
    std::string t = tags.count(i) ? tags.at(i) : "";
    int cls = classifier((int)i, c, t);
    if (cls != -1) unique_classes.insert(cls);
  }
  if (unique_classes.size() < 2) {
    throw std::invalid_argument(
        "Multi-pass decoding requires an annotated Stim circuit/DEM with at "
        "least "
        "2 stabilizer components.");
  }
}

void MultiPassTesseractDecoder::initialize(
    const stim::DetectorErrorModel& dem, const DetectorClassifier& classifier) {
  stim::DetectorErrorModel flattened = dem.flattened();
  // std::cout << "DEBUG flattened:\n" << flattened << std::endl;
  total_global_detectors = (size_t)flattened.count_detectors();

  std::vector<int> detector_classes(total_global_detectors, -1);
  std::set<uint64_t> all_ids;
  std::map<uint64_t, std::string> tags;
  for (const auto& inst : flattened.instructions) {
    if (inst.type == stim::DemInstructionType::DEM_DETECTOR) {
      uint64_t d = inst.target_data[0].val();
      all_ids.insert(d);
      tags[d] = inst.tag;
    }
  }
  auto coords_map = flattened.get_detector_coordinates(all_ids);
  for (uint64_t i = 0; i < total_global_detectors; ++i) {
    std::vector<double> c =
        coords_map.count(i) ? coords_map.at(i) : std::vector<double>{};
    std::string t = tags.count(i) ? tags.at(i) : "";
    detector_classes[i] = classifier((int)i, c, t);
  }

  stim::DetectorErrorModel decomposed =
      decompose_errors_using_generic_classifier(flattened, classifier, true);
  // std::cout << "DEBUG decomposed:\n" << decomposed << std::endl;
  stim::DetectorErrorModel merged = merge_indistinguishable_errors(decomposed);
  // std::cout << "DEBUG merged:\n" << merged << std::endl;

  std::set<int> unique_classes;
  for (int c : detector_classes)
    if (c != -1) unique_classes.insert(c);

  std::map<int, int> class_to_comp_id;
  int next_comp_id = 0;
  for (int c : unique_classes) class_to_comp_id[c] = next_comp_id++;

  size_t num_components = unique_classes.size();
  component_decoders.resize(num_components);

  global_det_to_comp_id.assign(total_global_detectors, -1);
  for (size_t i = 0; i < total_global_detectors; ++i) {
    int c = detector_classes[i];
    if (c != -1 && class_to_comp_id.count(c)) {
      int cid = class_to_comp_id[c];
      global_det_to_comp_id[i] = cid;
      component_decoders[cid].component_detectors.insert((int)i);
      // std::cout << "DEBUG: Assigned Global Det " << i << " to Component " <<
      // cid << std::endl;
    }
  }

  ImpliedProbsMap raw_correlations =
      process_dem_correlations(flattened, global_det_to_comp_id);

  auto component_dems_raw = split_dem_by_component(merged, [&](int d) {
    return (d >= 0 && (size_t)d < total_global_detectors)
               ? global_det_to_comp_id[d]
               : -1;
  });

  // std::cout << "DEBUG component_dems_raw[0]:\n" << component_dems_raw[0] <<
  // std::endl; std::cout << "DEBUG component_dems_raw[1]:\n" <<
  // component_dems_raw[1] << std::endl;

  for (size_t i = 0; i < component_decoders.size(); ++i) {
    auto& cd = component_decoders[i];

    for (size_t global_d = 0; global_d < total_global_detectors; ++global_d) {
      cd.global_to_local_det[global_d] = (int)global_d;
    }

    stim::DetectorErrorModel local_dem;
    for (size_t global_d = 0; global_d < total_global_detectors; ++global_d) {
      std::vector<double> c = coords_map.count(global_d)
                                  ? coords_map.at(global_d)
                                  : std::vector<double>{};
      std::string t = tags.count(global_d) ? tags.at(global_d) : "";
      local_dem.append_detector_instruction(
          c, stim::DemTarget::relative_detector_id(global_d), t);
    }

    for (const auto& inst : component_dems_raw[i].instructions) {
      if (inst.type == stim::DemInstructionType::DEM_ERROR) {
        bool has_obs = false;
        for (const auto& t : inst.target_data) {
          if (t.is_observable_id()) has_obs = true;
        }
        if (has_obs) cd.affects_observable = true;
        local_dem.append_error_instruction(inst.arg_data[0], inst.target_data,
                                           inst.tag);
      } else if (inst.type ==
                 stim::DemInstructionType::DEM_LOGICAL_OBSERVABLE) {
        local_dem.append_dem_instruction(inst);
      }
    }

    TesseractConfig config = base_config;
    config.dem = local_dem;
    config.merge_errors = true;
    config.det_orders =
        build_det_orders(config.dem, num_det_orders, det_order_method, seed);

    cd.decoder = std::make_unique<TesseractDecoder>(config);
    if (base_config.verbose) {
      std::cout << "DEBUG: Component " << i << " initialized with "
                << cd.decoder->errors.size() << " errors and "
                << config.dem.count_detectors() << " detectors." << std::endl;
    }
    cd.error_index_to_rules.resize(cd.decoder->errors.size());

    for (size_t ei = 0; ei < cd.decoder->errors.size(); ++ei) {
      cd.original_costs.push_back(cd.decoder->errors[ei].likelihood_cost);
      Hyperedge global_symptom = cd.decoder->errors[ei].symptom.detectors;
      std::sort(global_symptom.begin(), global_symptom.end());
      cd.symptom_to_error_index[global_symptom].push_back(ei);
    }
  }

  for (const auto& [global_symptom, implied_probs] : raw_correlations) {
    Hyperedge causal_symptom = global_symptom;
    std::sort(causal_symptom.begin(), causal_symptom.end());
    int causal_comp = -1;
    if (!causal_symptom.empty())
      causal_comp = global_det_to_comp_id[causal_symptom[0]];
    if (causal_comp == -1) continue;

    auto it = component_decoders[causal_comp].symptom_to_error_index.find(
        causal_symptom);
    if (it == component_decoders[causal_comp].symptom_to_error_index.end())
      continue;

    // Loop through all degenerate causal error indices!
    for (size_t causal_err_idx : it->second) {
      for (const auto& imp : implied_probs) {
        Hyperedge target_symptom = imp.affected_hyperedge;
        std::sort(target_symptom.begin(), target_symptom.end());
        int target_comp = -1;
        if (!target_symptom.empty())
          target_comp = global_det_to_comp_id[target_symptom[0]];
        if (target_comp == -1) continue;

        auto t_it = component_decoders[target_comp].symptom_to_error_index.find(
            target_symptom);
        if (t_it !=
            component_decoders[target_comp].symptom_to_error_index.end()) {
          // Loop through all degenerate target error indices and add rules to
          // each!
          for (size_t target_err_idx : t_it->second) {
            component_decoders[causal_comp]
                .error_index_to_rules[causal_err_idx]
                .push_back(
                    {(size_t)target_comp, target_err_idx, imp.probability});
          }
        }
      }
    }
  }

  if (strategy == SchedulingStrategy::Static) {
    build_static_schedule();
  } else if (strategy == SchedulingStrategy::Causal) {
    build_causal_schedule();
  }
}

void MultiPassTesseractDecoder::build_static_schedule() {
  pass_schedule.assign(num_passes, {});
  for (size_t p = 0; p < num_passes; ++p) {
    for (size_t i = 0; i < component_decoders.size(); ++i) {
      pass_schedule[p].push_back(i);
    }
  }
}

void MultiPassTesseractDecoder::build_causal_schedule() {
  size_t num_components = component_decoders.size();
  std::vector<std::set<size_t>> schedule_sets(num_passes);

  // Initial seed: Final pass includes all components that directly affect an
  // observable.
  for (size_t i = 0; i < num_components; ++i) {
    if (component_decoders[i].affects_observable) {
      schedule_sets[num_passes - 1].insert(i);
    }
  }

  // Back-propagate dependencies through passes.
  // A component is needed in pass p if it can reweight a component needed in
  // pass p+1.
  for (int p = (int)num_passes - 2; p >= 0; --p) {
    // Start with everyone needed in the next pass (they might need to re-decode
    // or bias others) Actually, if a component is in pass p+1, it's because it
    // was influenced by pass p.
    for (size_t target_comp_idx : schedule_sets[p + 1]) {
      for (size_t causal_comp_idx = 0; causal_comp_idx < num_components;
           ++causal_comp_idx) {
        for (const auto& rules :
             component_decoders[causal_comp_idx].error_index_to_rules) {
          for (const auto& rule : rules) {
            if (rule.target_comp_idx == target_comp_idx) {
              schedule_sets[p].insert(causal_comp_idx);
            }
          }
        }
      }
    }
  }

  // Convert sets to pass_schedule vectors.
  pass_schedule.assign(num_passes, {});
  for (size_t p = 0; p < num_passes; ++p) {
    for (size_t c_idx : schedule_sets[p]) {
      pass_schedule[p].push_back(c_idx);
    }
  }
}

std::vector<int> MultiPassTesseractDecoder::decode(
    const std::vector<uint64_t>& detections) {
  last_shot_num_reweights = 0;

  // 1. Multi-Pass Loop: Sequentially schedules component passes and propagates
  // priors.
  for (size_t pass = 0; pass < num_passes; ++pass) {
    bool is_final_pass = (pass == num_passes - 1);

    // Decode scheduled components for the current pass layer using persistent
    // local buffers.
    for (size_t comp_idx : pass_schedule[pass]) {
      auto& cd = component_decoders[comp_idx];
      std::vector<uint64_t> local_dets;
      for (uint64_t d : detections) {
        if (cd.component_detectors.count((int)d)) {
          local_dets.push_back(d);
        }
      }

      cd.decoder->decode_to_errors(local_dets);
      component_predictions[comp_idx] = cd.decoder->predicted_errors_buffer;
    }

    if (!is_final_pass) {
      // Step A: Apply Damped Fractional Memory to previously modified priors.
      // Smoothly decay current modifications back toward the baseline to
      // prevent message saturation.
      double gamma = 0.5;  // Tunable decay factor: 1.0 is strict isolation, 0.0
                           // is full accumulation.

      for (size_t m_comp_idx : modified_component_indices) {
        auto& cd = component_decoders[m_comp_idx];
        if (!cd.shot_all_modified_error_indices.empty()) {
          for (size_t idx : cd.shot_all_modified_error_indices) {
            double baseline_cost = cd.original_costs[idx];
            double current_cost = cd.decoder->errors[idx].likelihood_cost;
            cd.decoder->errors[idx].likelihood_cost =
                gamma * baseline_cost + (1.0 - gamma) * current_cost;
          }
          cd.decoder->update_internal_costs(cd.shot_all_modified_error_indices);
          // Retain tracking indices so the final Surgical Reset completely
          // clears cross-shot state.
        }
      }

      // Step B: Broadcast reweighting rules derived strictly from the latest
      // predictions.
      for (size_t comp_idx : pass_schedule[pass]) {
        auto& cd = component_decoders[comp_idx];
        for (size_t dem_err_idx : cd.decoder->predicted_errors_buffer) {
          size_t internal_err_idx =
              cd.decoder->dem_error_to_error.at(dem_err_idx);
          if (internal_err_idx == std::numeric_limits<size_t>::max()) continue;

          for (const auto& rule : cd.error_index_to_rules[internal_err_idx]) {
            auto& target_cd = component_decoders[rule.target_comp_idx];

            modified_component_indices.push_back(rule.target_comp_idx);

            // Apply Max-Prob Rule safely for concurrent rules within this pass
            // layer.
            double current_p = target_cd.decoder->errors[rule.target_error_idx]
                                   .get_probability();
            if (rule.conditional_prob > current_p) {
              target_cd.decoder->errors[rule.target_error_idx]
                  .set_with_probability(std::min(rule.conditional_prob, 0.5));
              target_cd.shot_all_modified_error_indices.push_back(
                  rule.target_error_idx);
              last_shot_num_reweights++;
            }
          }
        }
      }

      // Step C: Deduplicate modified tracking vectors and synchronize internal
      // graph costs.
      std::sort(modified_component_indices.begin(),
                modified_component_indices.end());
      modified_component_indices.erase(
          std::unique(modified_component_indices.begin(),
                      modified_component_indices.end()),
          modified_component_indices.end());

      for (size_t m_comp_idx : modified_component_indices) {
        auto& cd = component_decoders[m_comp_idx];
        if (!cd.shot_all_modified_error_indices.empty()) {
          std::sort(cd.shot_all_modified_error_indices.begin(),
                    cd.shot_all_modified_error_indices.end());
          cd.shot_all_modified_error_indices.erase(
              std::unique(cd.shot_all_modified_error_indices.begin(),
                          cd.shot_all_modified_error_indices.end()),
              cd.shot_all_modified_error_indices.end());
          cd.decoder->update_internal_costs(cd.shot_all_modified_error_indices);
        }
      }
    }
  }

  // 2. Unified Logical Extraction: Collect final predictions from ALL
  // components that ran during the shot.
  std::set<int> flipped_observables;
  for (const auto& [comp_idx, preds] : component_predictions) {
    auto& cd = component_decoders[comp_idx];
    if (preds.empty()) continue;

    std::vector<int> local_flips = cd.decoder->get_flipped_observables(preds);
    for (int obs : local_flips) {
      if (flipped_observables.count(obs))
        flipped_observables.erase(obs);
      else
        flipped_observables.insert(obs);
    }
  }

  // 3. Surgical Reset: Restore modified costs to leave the internal structures
  // pristine for the next shot.
  for (size_t m_comp_idx : modified_component_indices) {
    auto& cd = component_decoders[m_comp_idx];
    if (!cd.shot_all_modified_error_indices.empty()) {
      for (size_t idx : cd.shot_all_modified_error_indices) {
        cd.decoder->errors[idx].likelihood_cost = cd.original_costs[idx];
      }
      cd.decoder->update_internal_costs(cd.shot_all_modified_error_indices);
      cd.shot_all_modified_error_indices.clear();
    }
  }

  modified_component_indices.clear();
  final_pass_active_components.clear();

  return std::vector<int>(flipped_observables.begin(),
                          flipped_observables.end());
}

}  // namespace tesseract
