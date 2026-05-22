#ifndef TANNER_GRAPH_H
#define TANNER_GRAPH_H

#include <vector>
#include <map>
#include <set>
#include "stim.h"

namespace tesseract {

/**
 * Represents an independent connected component of the Tanner graph.
 */
struct TannerComponent {
    std::vector<int> detectors;
    std::vector<int> observables;
    std::vector<size_t> error_indices; // Indices of instructions in the DEM
    bool affects_observable = false;
};

/**
 * Utility to analyze the Tanner graph of a DetectorErrorModel.
 */
class TannerGraph {
public:
    /**
     * Finds all connected components in the provided DetectorErrorModel.
     * 
     * Assumes the DEM has been decomposed (errors affect only one component's symptoms).
     * If an error bridges symptoms, they will be unioned into the same component.
     */
    static std::vector<TannerComponent> find_components(const stim::DetectorErrorModel& dem);

private:
    struct UnionFind {
        std::vector<int> parent;
        UnionFind(size_t n) {
            parent.resize(n);
            for (size_t i = 0; i < n; ++i) parent[i] = i;
        }
        int find(int i) {
            if (parent[i] == i) return i;
            return parent[i] = find(parent[i]);
        }
        void unite(int i, int j) {
            int root_i = find(i);
            int root_j = find(j);
            if (root_i != root_j) parent[root_i] = root_j;
        }
    };
};

} // namespace tesseract

#endif // TANNER_GRAPH_H
