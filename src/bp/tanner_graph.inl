#ifndef BELIEFPROPAGATION_BP_TANNER_GRAPH_INL_
#define BELIEFPROPAGATION_BP_TANNER_GRAPH_INL_

#include <iostream>

namespace bp {

template <typename T>
TannerGraph<T>::TannerGraph(size_t num_variables, size_t num_checks, const std::vector<T>& priors)
    : variable_nodes(num_variables), check_nodes(num_checks) {
  if (priors.size() != num_variables) {
    throw std::invalid_argument("priors.size() != num_variables");
  }
  for (size_t i = 0; i < num_variables; i++) {
    variable_nodes[i].prior = priors[i];
  }
}

template <typename T>
void TannerGraph<T>::add_edge(size_t variable, size_t check) {
  pending_edges.push_back({variable, check});
}

template <typename T>
void TannerGraph<T>::build() {
  size_t num_vars = variable_nodes.size();
  size_t num_checks = check_nodes.size();

  var_edge_offsets.assign(num_vars + 1, 0);
  check_edge_offsets.assign(num_checks + 1, 0);
  check_fb_offsets.assign(num_checks + 1, 0);

  for (const auto& e : pending_edges) {
    var_edge_offsets[e.var + 1]++;
    check_edge_offsets[e.check + 1]++;
  }

  size_t var_edge_total = 0;
  for (size_t i = 1; i <= num_vars; ++i) {
    var_edge_offsets[i] += var_edge_offsets[i - 1];
    var_edge_total = var_edge_offsets[i];
  }

  size_t check_edge_total = 0;
  size_t check_fb_total = 0;
  for (size_t i = 1; i <= num_checks; ++i) {
    size_t deg = check_edge_offsets[i];
    check_edge_offsets[i] += check_edge_offsets[i - 1];
    check_edge_total = check_edge_offsets[i];

    check_fb_offsets[i - 1] = check_fb_total;
    if (deg > 1) {
      check_fb_total += 2 * (deg - 1);
    }
  }
  check_fb_offsets[num_checks] = check_fb_total;

  var_edges.resize(var_edge_total);
  var_edge_rev_indices.resize(var_edge_total);
  check_to_var_messages.assign(var_edge_total, 0);

  check_edges.resize(check_edge_total);
  check_edge_rev_indices.resize(check_edge_total);
  var_to_check_messages.assign(check_edge_total, 0);

  check_forward_back.assign(check_fb_total, 0);

  // Track insertion offsets
  std::vector<size_t> v_cur = var_edge_offsets;
  std::vector<size_t> c_cur = check_edge_offsets;

  for (const auto& e : pending_edges) {
    size_t v_e = v_cur[e.var]++;
    size_t c_e = c_cur[e.check]++;

    var_edges[v_e] = e.check;
    var_edge_rev_indices[v_e] = c_e;

    check_edges[c_e] = e.var;
    check_edge_rev_indices[c_e] = v_e;
  }

  pending_edges.clear();
  pending_edges.shrink_to_fit();
}

template <typename T>
void TannerGraph<T>::add_detection_events(const std::vector<size_t>& detection_events) {
  for (auto& d : detection_events) {
    check_nodes[d].syndrome = true;
  }
}

template <typename T>
void TannerGraph<T>::remove_detection_events(const std::vector<size_t>& detection_events) {
  for (auto& d : detection_events) {
    check_nodes[d].syndrome = false;
  }
}

template <typename T>
size_t TannerGraph<T>::count_edges() const {
  if (!pending_edges.empty()) return pending_edges.size();
  return var_edges.size();
}

}  // namespace bp

#endif  // BELIEFPROPAGATION_BP_TANNER_GRAPH_INL_