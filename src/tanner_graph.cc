#include "tanner_graph.h"
#include <unordered_map>
#include <algorithm>

namespace tesseract {

std::vector<TannerComponent> TannerGraph::find_components(const stim::DetectorErrorModel& dem) {
    int num_detectors = (int)dem.count_detectors();
    int num_observables = (int)dem.count_observables();
    int total_symptoms = num_detectors + num_observables;

    UnionFind uf(total_symptoms);
    std::vector<bool> symptom_active(total_symptoms, false);

    // 1. Union symptoms connected by errors
    auto flattened = dem.flattened();
    for (size_t i = 0; i < flattened.instructions.size(); ++i) {
        const auto& inst = flattened.instructions[i];
        if (inst.type != stim::DemInstructionType::DEM_ERROR) continue;

        // Manually split by separators to handle decomposed errors
        size_t group_start = 0;
        for (size_t k = 0; k <= inst.target_data.size(); ++k) {
            if (k == inst.target_data.size() || inst.target_data[k].is_separator()) {
                std::vector<int> group_symptoms;
                for (size_t j = group_start; j < k; ++j) {
                    const auto& target = inst.target_data[j];
                    int sym_id = -1;
                    if (target.is_relative_detector_id()) {
                        sym_id = target.val();
                    } else if (target.is_observable_id()) {
                        sym_id = num_detectors + target.val();
                    }
                    
                    if (sym_id != -1) {
                        group_symptoms.push_back(sym_id);
                        symptom_active[sym_id] = true;
                    }
                }

                for (size_t j = 1; j < group_symptoms.size(); ++j) {
                    uf.unite(group_symptoms[0], group_symptoms[j]);
                }
                group_start = k + 1;
            }
        }
    }

    // 2. Group symptoms by root
    std::unordered_map<int, TannerComponent> root_to_component;
    for (int i = 0; i < total_symptoms; ++i) {
        if (!symptom_active[i]) continue;
        
        int root = uf.find(i);
        if (root_to_component.find(root) == root_to_component.end()) {
            root_to_component[root] = TannerComponent();
        }

        if (i < num_detectors) {
            root_to_component[root].detectors.push_back(i);
        } else {
            root_to_component[root].observables.push_back(i - num_detectors);
            root_to_component[root].affects_observable = true;
        }
    }

    // 3. Assign errors to components
    for (size_t i = 0; i < flattened.instructions.size(); ++i) {
        const auto& inst = flattened.instructions[i];
        if (inst.type != stim::DemInstructionType::DEM_ERROR) continue;

        std::set<int> roots_touched;
        for (const auto& target : inst.target_data) {
            int sym_id = -1;
            if (target.is_relative_detector_id()) {
                sym_id = target.val();
            } else if (target.is_observable_id()) {
                sym_id = num_detectors + target.val();
            }
            if (sym_id != -1) {
                roots_touched.insert(uf.find(sym_id));
            }
        }

        for (int root : roots_touched) {
            root_to_component[root].error_indices.push_back(i);
        }
    }

    std::vector<TannerComponent> components;
    for (auto& pair : root_to_component) {
        components.push_back(std::move(pair.second));
    }

    return components;
}

} // namespace tesseract
