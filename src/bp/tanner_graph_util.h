#ifndef BP_UTIL_H
#define BP_UTIL_H

#include "bp/tanner_graph.h"

namespace bp {

struct BPResult {
  bool converged;
  size_t num_iters;
};

template <typename T>
T sat_add(T a, T b) {
  if constexpr (std::is_integral_v<T>) {
    if (a > 0 && b > std::numeric_limits<T>::max() - a) return std::numeric_limits<T>::max();
    if (a < 0 && b < std::numeric_limits<T>::min() - a) return std::numeric_limits<T>::min();
  }
  return a + b;
}

template <typename T>
bool check_convergence(const TannerGraph<T>& graph) {
  for (size_t i = 0; i < graph.check_nodes.size(); ++i) {
    bool posterior_parity = false;
    size_t start = graph.check_edge_offsets[i];
    size_t end = graph.check_edge_offsets[i + 1];
    for (size_t e = start; e < end; ++e) {
      size_t v = graph.check_edges[e];
      posterior_parity ^= graph.variable_nodes[v].posterior < 0;
    }
    if (posterior_parity != graph.check_nodes[i].syndrome) {
      return false;
    }
  }
  return true;
}

template <typename T>
bool check_convergence(const std::vector<size_t>& checks, const TannerGraph<T>& graph) {
  for (size_t c_idx : checks) {
    bool posterior_parity = false;
    size_t start = graph.check_edge_offsets[c_idx];
    size_t end = graph.check_edge_offsets[c_idx + 1];
    for (size_t e = start; e < end; ++e) {
      size_t v = graph.check_edges[e];
      posterior_parity ^= graph.variable_nodes[v].posterior < 0;
    }
    if (posterior_parity != graph.check_nodes[c_idx].syndrome) {
      return false;
    }
  }
  return true;
}

}  // namespace bp

#endif /* BP_UTIL_H */