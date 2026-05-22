#include "multi_pass_tesseract_decoder.h"
#include "dem_decomposition.h"
#include <iostream>
#include <set>
#include <algorithm>
#include <sstream>
#include <chrono>
#include <cmath>

namespace tesseract {

MultiPassTesseractDecoder::MultiPassTesseractDecoder(
    const stim::DetectorErrorModel& dem,
    size_t num_passes,
    const DetectorClassifier& classifier,
    const TesseractConfig& base_config,
    size_t num_det_orders,
    DetOrder det_order_method,
    uint64_t seed,
    SchedulingStrategy strategy) 
    : num_passes(num_passes), strategy(strategy), 
      total_global_detectors(dem.count_detectors()),
      base_config(base_config), 
      num_det_orders(num_det_orders), det_order_method(det_order_method), seed(seed) {
    initialize(dem, classifier);
}

void MultiPassTesseractDecoder::initialize(
    const stim::DetectorErrorModel& dem, 
    const DetectorClassifier& classifier) {
    
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
        std::vector<double> c = coords_map.count(i) ? coords_map.at(i) : std::vector<double>{};
        std::string t = tags.count(i) ? tags.at(i) : "";
        detector_classes[i] = classifier((int)i, c, t);
    }

    stim::DetectorErrorModel decomposed = decompose_errors_using_generic_classifier(flattened, classifier, true);
    // std::cout << "DEBUG decomposed:\n" << decomposed << std::endl;
    stim::DetectorErrorModel merged = merge_indistinguishable_errors(decomposed);
    // std::cout << "DEBUG merged:\n" << merged << std::endl;
    ImpliedProbsMap raw_correlations = process_dem_correlations(merged);
    
    std::set<int> unique_classes;
    for (int c : detector_classes) if (c != -1) unique_classes.insert(c);
    
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
            // std::cout << "DEBUG: Assigned Global Det " << i << " to Component " << cid << std::endl;
        }
    }

    auto component_dems_raw = split_dem_by_component(merged, [&](int d) { 
        return (d >= 0 && (size_t)d < total_global_detectors) ? global_det_to_comp_id[d] : -1; 
    });

    // std::cout << "DEBUG component_dems_raw[0]:\n" << component_dems_raw[0] << std::endl;
    // std::cout << "DEBUG component_dems_raw[1]:\n" << component_dems_raw[1] << std::endl;

    for (size_t i = 0; i < component_decoders.size(); ++i) {
        auto& cd = component_decoders[i];
        
        std::vector<int> sorted_global_dets(cd.component_detectors.begin(), cd.component_detectors.end());
        std::sort(sorted_global_dets.begin(), sorted_global_dets.end());
        for (size_t local_idx = 0; local_idx < sorted_global_dets.size(); ++local_idx) {
            cd.global_to_local_det[sorted_global_dets[local_idx]] = (int)local_idx;
        }

        stim::DetectorErrorModel local_dem;
        // MUST append detector instructions for ALL local detectors first to set count_detectors() correctly
        for (size_t local_idx = 0; local_idx < sorted_global_dets.size(); ++local_idx) {
            int global_d = sorted_global_dets[local_idx];
            std::vector<double> c = coords_map.count(global_d) ? coords_map.at(global_d) : std::vector<double>{};
            std::string t = tags.count(global_d) ? tags.at(global_d) : "";
            local_dem.append_detector_instruction(c, stim::DemTarget::relative_detector_id(local_idx), t);
        }

        for (const auto& inst : component_dems_raw[i].instructions) {
            if (inst.type == stim::DemInstructionType::DEM_ERROR) {
                std::vector<stim::DemTarget> local_targets;
                bool has_obs = false;
                for (const auto& t : inst.target_data) {
                    if (t.is_relative_detector_id()) {
                        int global_d = t.val();
                        local_targets.push_back(stim::DemTarget::relative_detector_id(cd.global_to_local_det.at(global_d)));
                    } else {
                        local_targets.push_back(t);
                        if (t.is_observable_id()) has_obs = true;
                    }
                }
                if (has_obs) cd.affects_observable = true;
                local_dem.append_error_instruction(inst.arg_data[0], local_targets, inst.tag);
            }
 else if (inst.type == stim::DemInstructionType::DEM_LOGICAL_OBSERVABLE) {
                local_dem.append_dem_instruction(inst);
            }
        }

        // std::cout << "DEBUG: local_dem " << i << " : " << local_dem << std::endl;

        TesseractConfig config = base_config;
        config.dem = local_dem;
        config.merge_errors = true;
        config.det_orders = build_det_orders(config.dem, num_det_orders, det_order_method, seed + i);
        
        cd.decoder = std::make_unique<TesseractDecoder>(config);
        // std::cout << "DEBUG: Component " << i << " initialized with " << cd.decoder->errors.size() << " errors and " << config.dem.count_detectors() << " detectors." << std::endl;
        /*
        for (size_t ei = 0; ei < cd.decoder->errors.size(); ei++) {
            // std::cout << "  Comp " << i << " Err " << ei << ": D";
            for (int d : cd.decoder->errors[ei].symptom.detectors) // std::cout << d << " ";
            // std::cout << std::endl;
        }
        */
        cd.error_index_to_rules.resize(cd.decoder->errors.size());
        
        for (size_t ei = 0; ei < cd.decoder->errors.size(); ++ei) {
            cd.original_costs.push_back(cd.decoder->errors[ei].likelihood_cost);
            Hyperedge local_symptom = cd.decoder->errors[ei].symptom.detectors;
            Hyperedge global_symptom;
            for (int local_d : local_symptom) global_symptom.push_back(sorted_global_dets[local_d]);
            std::sort(global_symptom.begin(), global_symptom.end());
            cd.symptom_to_error_index[global_symptom] = ei;
        }
    }

    for (const auto& [global_symptom, implied_probs] : raw_correlations) {
        Hyperedge causal_symptom = global_symptom;
        std::sort(causal_symptom.begin(), causal_symptom.end());
        int causal_comp = -1;
        if (!causal_symptom.empty()) causal_comp = global_det_to_comp_id[causal_symptom[0]];
        if (causal_comp == -1) continue;
        auto it = component_decoders[causal_comp].symptom_to_error_index.find(causal_symptom);
        if (it == component_decoders[causal_comp].symptom_to_error_index.end()) continue;
        size_t causal_err_idx = it->second;
        for (const auto& imp : implied_probs) {
            Hyperedge target_symptom = imp.affected_hyperedge;
            std::sort(target_symptom.begin(), target_symptom.end());
            int target_comp = -1;
            if (!target_symptom.empty()) target_comp = global_det_to_comp_id[target_symptom[0]];
            if (target_comp == -1) continue;
            auto t_it = component_decoders[target_comp].symptom_to_error_index.find(target_symptom);
            if (t_it != component_decoders[target_comp].symptom_to_error_index.end()) {
                component_decoders[causal_comp].error_index_to_rules[causal_err_idx].push_back({
                    (size_t)target_comp, t_it->second, imp.probability
                });
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

    // Initial seed: Final pass includes all components that directly affect an observable.
    for (size_t i = 0; i < num_components; ++i) {
        if (component_decoders[i].affects_observable) {
            schedule_sets[num_passes - 1].insert(i);
        }
    }

    // Back-propagate dependencies through passes.
    // A component is needed in pass p if it can reweight a component needed in pass p+1.
    for (int p = (int)num_passes - 2; p >= 0; --p) {
        // Start with everyone needed in the next pass (they might need to re-decode or bias others)
        // Actually, if a component is in pass p+1, it's because it was influenced by pass p.
        for (size_t target_comp_idx : schedule_sets[p + 1]) {
            for (size_t causal_comp_idx = 0; causal_comp_idx < num_components; ++causal_comp_idx) {
                for (const auto& rules : component_decoders[causal_comp_idx].error_index_to_rules) {
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

std::vector<int> MultiPassTesseractDecoder::decode(const std::vector<uint64_t>& detections) {
    last_shot_num_reweights = 0;
    // 1. Multi-Pass Loop: Earlier passes only bias the final pass.
    for (size_t pass = 0; pass < num_passes; ++pass) {
        bool is_final_pass = (pass == num_passes - 1);

        for (size_t comp_idx : pass_schedule[pass]) {
            auto& cd = component_decoders[comp_idx];
            std::vector<uint64_t> local_dets;
            for (uint64_t d : detections) {
                if (cd.global_to_local_det.count((int)d)) {
                    local_dets.push_back((uint64_t)cd.global_to_local_det.at((int)d));
                }
            }
            
            // Perform decoding for this component in this pass.
            cd.decoder->decode_to_errors(local_dets);

            if (is_final_pass) {
                // Track components that decode in the final pass for extraction.
                final_pass_active_components.push_back(comp_idx);
            } else {
                // If this is NOT the final pass, use the results for reweighting, then discard them.
                for (size_t dem_err_idx : cd.decoder->predicted_errors_buffer) {
                    size_t internal_err_idx = cd.decoder->dem_error_to_error.at(dem_err_idx);
                    if (internal_err_idx == std::numeric_limits<size_t>::max()) continue;
                    
                    for (const auto& rule : cd.error_index_to_rules[internal_err_idx]) {
                        auto& target_cd = component_decoders[rule.target_comp_idx];
                        
                        // Track modified components only once per shot.
                        if (target_cd.modified_error_indices.empty()) {
                            modified_component_indices.push_back(rule.target_comp_idx);
                        }

                        // Cap probability at 0.499 to prevent negative costs in the engine.
                        target_cd.decoder->errors[rule.target_error_idx].set_with_probability(std::min(rule.conditional_prob, 0.499));
                        target_cd.modified_error_indices.push_back(rule.target_error_idx);
                        last_shot_num_reweights++;
                    }
                }
                // Clear the buffer so these intermediate decisions don't contribute to the final prediction.
                cd.decoder->predicted_errors_buffer.clear();
            }
        }

        // Sync modified costs for the next pass.
        if (!is_final_pass) {
            for (size_t m_comp_idx : modified_component_indices) {
                auto& cd = component_decoders[m_comp_idx];
                if (!cd.modified_error_indices.empty()) {
                    cd.decoder->update_internal_costs(cd.modified_error_indices);
                }
            }
        }
    }

    // 2. Unified Logical Extraction: Collect final-pass predictions from only active components.
    std::set<int> flipped_observables;
    for (size_t comp_idx : final_pass_active_components) {
        auto& cd = component_decoders[comp_idx];
        if (cd.decoder->predicted_errors_buffer.empty()) continue;
        
        std::vector<int> local_flips = cd.decoder->get_flipped_observables(cd.decoder->predicted_errors_buffer);
        for (int obs : local_flips) {
            if (flipped_observables.count(obs)) flipped_observables.erase(obs);
            else flipped_observables.insert(obs);
        }
    }

    // 3. Surgical Reset: Restore modified costs for the next shot.
    for (size_t m_comp_idx : modified_component_indices) {
        auto& cd = component_decoders[m_comp_idx];
        for (size_t idx : cd.modified_error_indices) {
            cd.decoder->errors[idx].likelihood_cost = cd.original_costs[idx];
        }
        cd.decoder->update_internal_costs(cd.modified_error_indices);
        cd.modified_error_indices.clear();
    }
    
    // Clear shot-level tracking vectors.
    modified_component_indices.clear();
    final_pass_active_components.clear();

    return std::vector<int>(flipped_observables.begin(), flipped_observables.end());
}

} // namespace tesseract
