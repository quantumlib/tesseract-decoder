#include "dem_decomposition.h"
#include "bern_utils.h"

#include <vector>
#include <set>
#include <utility>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <unordered_set>

#include "stim.h"

namespace tesseract {

// Helper function to generate all combinations of observables
void generate_obs_combinations(
    const std::vector<std::set<std::vector<int>>>& obs_options_by_component,
    std::vector<std::vector<int>>& current_combination,
    std::vector<std::vector<std::vector<int>>>& all_combinations,
    int component_index) {
    if (component_index == (int)obs_options_by_component.size()) {
        all_combinations.push_back(current_combination);
        return;
    }

    for (const auto& obs_option : obs_options_by_component[component_index]) {
        current_combination.push_back(obs_option);
        generate_obs_combinations(obs_options_by_component, current_combination, all_combinations, component_index + 1);
        current_combination.pop_back();
    }
}

std::vector<int> reduce_symmetric_difference(const std::vector<int>& items) {
    std::set<int> unpaired_set;
    for (int item : items) {
        if (unpaired_set.count(item)) {
            unpaired_set.erase(item);
        } else {
            unpaired_set.insert(item);
        }
    }
    return std::vector<int>(unpaired_set.begin(), unpaired_set.end());
}

std::vector<int> reduce_set_symmetric_difference(const std::vector<std::vector<int>>& sets) {
    std::vector<int> all_items;
    for (const auto& s : sets) {
        all_items.insert(all_items.end(), s.begin(), s.end());
    }
    return reduce_symmetric_difference(all_items);
}

std::pair<std::vector<int>, std::vector<int>> undecomposed_error_detectors_and_observables(
    const stim::DemInstruction& instruction) {
    if (instruction.type != stim::DemInstructionType::DEM_ERROR) {
        throw std::invalid_argument("DEM instruction must be an error");
    }

    std::vector<int> detectors;
    std::vector<int> observables;
    for (const auto& target : instruction.target_data) {
        if (target.is_relative_detector_id()) {
            detectors.push_back(target.val());
        } else if (target.is_observable_id()) {
            observables.push_back(target.val());
        }
    }

    return {reduce_symmetric_difference(detectors), reduce_symmetric_difference(observables)};
}

std::vector<std::vector<int>> get_component_obs_matching_undecomposed_obs(
    const std::vector<std::set<std::vector<int>>>& obs_options_by_component,
    const std::vector<int>& error_obs,
    int num_missing_components,
    bool allow_remnant_errors) {

    if (!allow_remnant_errors && num_missing_components > 0) {
        return {};
    }

    std::vector<std::vector<std::vector<int>>> all_combinations;
    std::vector<std::vector<int>> current_combination;
    generate_obs_combinations(obs_options_by_component, current_combination, all_combinations, 0);

    std::vector<int> error_obs_reduced = reduce_symmetric_difference(error_obs);
    std::set<int> error_obs_set(error_obs_reduced.begin(), error_obs_reduced.end());
    
    for (const auto& combination : all_combinations) {
        std::vector<int> known_obs_sum = reduce_set_symmetric_difference(combination);
        
        // Residual = error_obs XOR known_obs_sum
        std::vector<int> residual_input = error_obs_reduced;
        residual_input.insert(residual_input.end(), known_obs_sum.begin(), known_obs_sum.end());
        std::vector<int> residual = reduce_symmetric_difference(residual_input);

        if (residual.empty()) {
            // Case A: Residual is empty. All missing components get no observables.
            std::vector<std::vector<int>> result = combination;
            for (int i = 0; i < num_missing_components; ++i) result.push_back({});
            return result;
        }

        if (num_missing_components >= 1 && allow_remnant_errors) {
            // Case B: Residual is non-empty and at least one component is missing.
            // Assign the entire residual to the first missing component.
            std::vector<std::vector<int>> result = combination;
            result.push_back(residual);
            for (int i = 0; i < num_missing_components - 1; ++i) result.push_back({});
            return result;
        }
    }

    // Best effort logic if allow_remnant_errors is true
    if (allow_remnant_errors) {
        if (!obs_options_by_component.empty()) {
            // Use the first combination and force residual into the first component
            std::vector<std::vector<int>> first_combination;
            for (const auto& options : obs_options_by_component) {
                first_combination.push_back(*options.begin());
            }
            std::vector<int> first_obs_sum = reduce_set_symmetric_difference(first_combination);
            
            std::vector<int> residual_input = error_obs_reduced;
            residual_input.insert(residual_input.end(), first_obs_sum.begin(), first_obs_sum.end());
            std::vector<int> residual = reduce_symmetric_difference(residual_input);

            std::vector<int> forced_first_input = first_combination[0];
            forced_first_input.insert(forced_first_input.end(), residual.begin(), residual.end());
            first_combination[0] = reduce_symmetric_difference(forced_first_input);

            for (int i = 0; i < num_missing_components; ++i) first_combination.push_back({});
            return first_combination;
        } else if (num_missing_components > 0) {
            // No known components? Put everything in the first missing one.
            std::vector<std::vector<int>> result;
            result.push_back(error_obs_reduced);
            for (int i = 0; i < num_missing_components - 1; ++i) result.push_back({});
            return result;
        }
    }

    return {};
}

stim::DetectorErrorModel decompose_errors_using_detector_assignment(
    const stim::DetectorErrorModel& dem,
    const std::function<int(int)>& detector_component_func,
    bool allow_remnant_errors) {

    stim::DetectorErrorModel flattened_dem = dem.flattened();
    std::map<std::vector<int>, std::set<std::vector<int>>> single_component_dets_to_obs;

    for (const auto& instruction : flattened_dem.instructions) {
        if (instruction.type != stim::DemInstructionType::DEM_ERROR) continue;

        auto [detectors, observables] = undecomposed_error_detectors_and_observables(instruction);
        
        std::unordered_set<int> components;
        for (int d : detectors) components.insert(detector_component_func(d));
        
        if (components.size() <= 1) {
            single_component_dets_to_obs[detectors].insert(observables);
        }
    }

    stim::DetectorErrorModel output_dem;
    for (const auto& instruction : flattened_dem.instructions) {
        if (instruction.type != stim::DemInstructionType::DEM_ERROR) {
            output_dem.append_dem_instruction(instruction);
            continue;
        }

        auto [detectors, observables] = undecomposed_error_detectors_and_observables(instruction);
        
        std::map<int, std::vector<int>> dets_by_comp_id;
        std::set<int> unique_components;
        for (int d : detectors) {
            int c = detector_component_func(d);
            dets_by_comp_id[c].push_back(d);
            unique_components.insert(c);
        }

        std::vector<std::vector<int>> dets_by_component;
        std::vector<std::set<std::vector<int>>> obs_options_by_known_component;
        std::vector<std::vector<int>> missing_components_dets;

        for (int c : unique_components) {
            std::vector<int> component_dets = dets_by_comp_id[c];
            std::sort(component_dets.begin(), component_dets.end());

            if (single_component_dets_to_obs.count(component_dets)) {
                dets_by_component.push_back(component_dets);
                obs_options_by_known_component.push_back(single_component_dets_to_obs[component_dets]);
            } else {
                if (!allow_remnant_errors) {
                    throw std::invalid_argument("Component not present as its own error and allow_remnant_errors=false");
                }
                missing_components_dets.push_back(component_dets);
            }
        }

        std::vector<std::vector<int>> consistent_obs_by_component = get_component_obs_matching_undecomposed_obs(
            obs_options_by_known_component, observables, (int)missing_components_dets.size(), allow_remnant_errors);

        if (consistent_obs_by_component.empty()) {
            throw std::invalid_argument("Error instruction could not be decomposed consistently.");
        }

        std::vector<stim::DemTarget> targets;
        std::vector<std::vector<int>> all_dets = dets_by_component;
        all_dets.insert(all_dets.end(), missing_components_dets.begin(), missing_components_dets.end());

        for (size_t i = 0; i < all_dets.size(); ++i) {
            for (int d : all_dets[i]) targets.push_back(stim::DemTarget::relative_detector_id(d));
            for (int o : consistent_obs_by_component[i]) targets.push_back(stim::DemTarget::observable_id(o));
            if (i != all_dets.size() - 1) targets.push_back(stim::DemTarget::separator());
        }
        
        output_dem.append_error_instruction(instruction.arg_data[0], targets, instruction.tag);
    }
    return output_dem;
}

stim::DetectorErrorModel decompose_errors_using_generic_classifier(
    const stim::DetectorErrorModel& dem,
    const DetectorClassifier& classifier,
    bool allow_remnant_errors) {
    
    // 1. Collect all detectors and their metadata
    std::set<uint64_t> all_detector_indices;
    std::map<int, std::string> detector_tags;
    for (const auto& inst : dem.flattened().instructions) {
        if (inst.type == stim::DemInstructionType::DEM_DETECTOR) {
            int d = inst.target_data[0].val();
            all_detector_indices.insert(d);
            detector_tags[d] = inst.tag;
        }
    }

    auto detector_coords = dem.get_detector_coordinates(all_detector_indices);

    // 2. Pre-classify detectors using the generic classifier
    std::map<int, int> classification_cache;
    for (uint64_t d : all_detector_indices) {
        std::vector<double> coords = detector_coords.count(d) ? detector_coords.at(d) : std::vector<double>{};
        classification_cache[d] = classifier((int)d, coords, detector_tags[d]);
    }

    // 3. Decompose using the cached classification
    auto component_func = [&](int d) {
        return classification_cache.count(d) ? classification_cache.at(d) : 0;
    };

    return decompose_errors_using_detector_assignment(dem, component_func, allow_remnant_errors);
}

std::map<int, stim::DetectorErrorModel> split_dem_by_component(
    const stim::DetectorErrorModel& dem,
    const std::function<int(int)>& detector_component_func) {
    
    std::map<int, stim::DetectorErrorModel> component_dems;

    for (const auto& instruction : dem.instructions) {
        if (instruction.type == stim::DemInstructionType::DEM_ERROR) {
            double prob = instruction.arg_data[0];
            
            size_t group_start = 0;
            for (size_t k = 0; k <= instruction.target_data.size(); ++k) {
                if (k == instruction.target_data.size() || instruction.target_data[k].is_separator()) {
                    std::vector<stim::DemTarget> component_targets;
                    std::set<int> component_ids;
                    for (size_t j = group_start; j < k; ++j) {
                         const auto& target = instruction.target_data[j];
                         component_targets.push_back(target);
                         if (target.is_relative_detector_id()) {
                             component_ids.insert(detector_component_func(target.val()));
                         }
                    }

                    if (component_ids.empty()) {
                        // If no detectors, we can't assign it to a component based on detectors.
                        // For now, let's skip or handle separately.
                    } else if (component_ids.size() > 1) {
                         throw std::invalid_argument("Mixed component ID in a single error component group.");
                    } else {
                        int comp_id = *component_ids.begin();
                        component_dems[comp_id].append_error_instruction(prob, component_targets, "");
                    }
                    group_start = k + 1;
                }
            }
        } else if (instruction.type == stim::DemInstructionType::DEM_DETECTOR || 
                   instruction.type == stim::DemInstructionType::DEM_LOGICAL_OBSERVABLE) {
            for (auto& pair : component_dems) {
                pair.second.append_dem_instruction(instruction);
            }
        }
    }
    return component_dems;
}

stim::DetectorErrorModel undecompose_errors(const stim::DetectorErrorModel& dem) {
    stim::DetectorErrorModel undecomposed_dem;
    for (const auto& instruction : dem.instructions) {
        if (instruction.type == stim::DemInstructionType::DEM_REPEAT_BLOCK) {
            undecomposed_dem.append_repeat_block(
                instruction.repeat_block_rep_count(),
                undecompose_errors(instruction.repeat_block_body(dem)),
                instruction.tag
            );
            continue;
        }

        if (instruction.type != stim::DemInstructionType::DEM_ERROR) {
            undecomposed_dem.append_dem_instruction(instruction);
            continue;
        }

        auto [detectors, observables] = undecomposed_error_detectors_and_observables(instruction);
        std::vector<stim::DemTarget> targets;
        for (int d : detectors) targets.push_back(stim::DemTarget::relative_detector_id(d));
        for (int o : observables) targets.push_back(stim::DemTarget::observable_id(o));

        undecomposed_dem.append_error_instruction(instruction.arg_data[0], targets, instruction.tag);
    }
    return undecomposed_dem;
}

stim::DetectorErrorModel merge_indistinguishable_errors(const stim::DetectorErrorModel& dem) {
    // Key is a set of (sorted_detectors, sorted_observables) components
    typedef std::pair<std::vector<int>, std::vector<int>> ComponentSymptom;
    std::map<std::set<ComponentSymptom>, double> symptom_to_prob;
    stim::DetectorErrorModel merged_dem;

    for (const auto& instruction : dem.flattened().instructions) {
        if (instruction.type != stim::DemInstructionType::DEM_ERROR) {
            merged_dem.append_dem_instruction(instruction);
            continue;
        }

        double prob = instruction.arg_data[0];
        std::set<ComponentSymptom> decomposed_symptom;
        
        instruction.for_separated_targets([&](std::span<const stim::DemTarget> group) {
            std::vector<int> dets;
            std::vector<int> obs;
            for (const auto& t : group) {
                if (t.is_relative_detector_id()) dets.push_back(t.val());
                else if (t.is_observable_id()) obs.push_back(t.val());
            }
            std::sort(dets.begin(), dets.end());
            std::sort(obs.begin(), obs.end());
            decomposed_symptom.insert({dets, obs});
        });
        
        if (symptom_to_prob.find(decomposed_symptom) == symptom_to_prob.end()) {
            symptom_to_prob[decomposed_symptom] = 0.0;
        }
        symptom_to_prob[decomposed_symptom] = tesseract::bernoulli_xor(symptom_to_prob[decomposed_symptom], prob);
    }

    for (auto const& [decomposed_symptom, prob] : symptom_to_prob) {
        if (prob > 0) {
            std::vector<stim::DemTarget> targets;
            size_t i = 0;
            for (const auto& comp : decomposed_symptom) {
                for (int d : comp.first) targets.push_back(stim::DemTarget::relative_detector_id(d));
                for (int o : comp.second) targets.push_back(stim::DemTarget::observable_id(o));
                if (i < decomposed_symptom.size() - 1) targets.push_back(stim::DemTarget::separator());
                i++;
            }
            merged_dem.append_error_instruction(prob, targets, "");
        }
    }
    return merged_dem;
}

} // namespace tesseract
