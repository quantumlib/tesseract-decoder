#include "bp/tanner_graph.h"

#include <stdexcept>

namespace bp {

TannerGraph::TannerGraph(size_t num_variables, size_t num_checks,
                         const std::vector<LLR_INT>& priors)
    : num_vars(num_variables), num_checks(num_checks), priors(priors) {
  if (priors.size() != num_variables) {
    throw std::invalid_argument("priors.size() != num_variables");
  }
}

void TannerGraph::add_edge(size_t variable, size_t check) {
  pending_edges.push_back({variable, check});
}

void TannerGraph::build() {
  var_edge_offsets.assign(num_vars + 1, 0);
  check_edge_offsets.assign(num_checks + 1, 0);

  for (const auto& e : pending_edges) {
    var_edge_offsets[e.var + 1]++;
    check_edge_offsets[e.check + 1]++;
  }

  for (size_t i = 1; i <= num_vars; ++i) {
    var_edge_offsets[i] += var_edge_offsets[i - 1];
  }
  for (size_t i = 1; i <= num_checks; ++i) {
    check_edge_offsets[i] += check_edge_offsets[i - 1];
  }

  num_edges = var_edge_offsets[num_vars];
  var_edges.resize(num_edges);
  check_edges.resize(num_edges);

  std::vector<size_t> v_cur = var_edge_offsets;
  std::vector<size_t> c_cur = check_edge_offsets;

  for (const auto& e : pending_edges) {
    var_edges[v_cur[e.var]++] = e.check;
    check_edges[c_cur[e.check]++] = e.var;
  }

  pending_edges.clear();
  pending_edges.shrink_to_fit();

  syndromes.assign(num_checks * batch_size, 0);
  var_posteriors.assign(num_vars * batch_size, 0);
  check_to_var_messages.assign(num_edges * batch_size, 0);
  converged_mask.assign(batch_size, 0);
  converged_posteriors_flat.assign(num_vars * batch_size, 0);
}

size_t TannerGraph::count_edges() const {
  if (!pending_edges.empty()) return pending_edges.size();
  return num_edges;
}

}  // namespace bp
