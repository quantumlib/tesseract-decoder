#include "bp/tanner_graph.h"

#include <cmath>
#include <cstdlib>
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
  var_edge_to_check_edge.resize(num_edges);
  check_edges.resize(num_edges);

  std::vector<size_t> v_cur = var_edge_offsets;
  std::vector<size_t> c_cur = check_edge_offsets;

  for (const auto& e : pending_edges) {
    size_t ve = v_cur[e.var]++;
    size_t ce = c_cur[e.check]++;
    var_edges[ve] = e.check;
    check_edges[ce] = e.var;
    var_edge_to_check_edge[ve] = ce;
  }

  pending_edges.clear();
  pending_edges.shrink_to_fit();

  syndromes.assign(num_checks * batch_size, 0);
  var_posteriors.assign(num_vars * batch_size, 0);
  check_to_var_messages.assign(num_edges * batch_size, 0);
  converged_mask.assign(batch_size, 0);
  converged_posteriors_flat.assign(num_vars * batch_size, 0);

  min1_vac.assign(num_checks, MAX_SAFE_LLR_INT);
  min2_vac.assign(num_checks, MAX_SAFE_LLR_INT);
  argmin_vac.assign(num_checks, 0);
  sign_prod_vac.assign(num_checks, 0);

  for (size_t c = 0; c < num_checks; ++c) {
    LLR_UINT m1 = MAX_SAFE_LLR_INT;
    LLR_UINT m2 = MAX_SAFE_LLR_INT;
    size_t argmin = 0;
    uint8_t sp = 0;
    const size_t start = check_edge_offsets[c];
    const size_t end = check_edge_offsets[c + 1];
    for (size_t e = start; e < end; ++e) {
      const size_t v = check_edges[e];
      const LLR_INT p = priors[v];
      if (p < 0) sp ^= 1;
      LLR_UINT mag = static_cast<LLR_UINT>(std::abs(p));
      if (mag > MAX_SAFE_LLR_INT) mag = MAX_SAFE_LLR_INT;
      if (mag < m1) {
        m2 = m1;
        m1 = mag;
        argmin = e - start;
      } else if (mag < m2) {
        m2 = mag;
      }
    }
    min1_vac[c] = m1;
    min2_vac[c] = m2;
    argmin_vac[c] = argmin;
    sign_prod_vac[c] = sp;
  }

  vacuum_initialized = false;
  cached_alpha_fp = -1;
}

void TannerGraph::ensure_vacuum_state(float normalization_factor) {
  const LLR_INT alpha_fp = static_cast<LLR_INT>(std::round(normalization_factor * LLR_FP_SCALE));
  if (vacuum_initialized && cached_alpha_fp == alpha_fp) {
    return;
  }
  cached_alpha_fp = alpha_fp;
  vacuum_initialized = true;

  m_vac.assign(num_edges, 0);
  v_posterior_vac = priors;

  for (size_t c = 0; c < num_checks; ++c) {
    const LLR_UINT m1 = min1_vac[c];
    const LLR_UINT m2 = min2_vac[c];
    const size_t argmin = argmin_vac[c];
    const uint8_t sp = sign_prod_vac[c];
    const size_t start = check_edge_offsets[c];
    const size_t end = check_edge_offsets[c + 1];
    for (size_t e = start; e < end; ++e) {
      const size_t v = check_edges[e];
      const size_t local_idx = e - start;
      const LLR_UINT ext_min = (local_idx == argmin) ? m2 : m1;
      const LLR_INT scaled_mag =
          static_cast<LLR_INT>((static_cast<int64_t>(ext_min) * alpha_fp) >> LLR_FP_BITS);
      const uint8_t v_sign = (priors[v] < 0) ? 1u : 0u;
      const uint8_t r_sign = sp ^ v_sign;
      const LLR_INT r_val = r_sign ? -scaled_mag : scaled_mag;
      m_vac[e] = r_val;
      v_posterior_vac[v] += r_val;
    }
  }

  num_check_words = (num_checks + 63) / 64;
  work_priors = priors;
  work_m_vac = m_vac;
  work_check_to_var = m_vac;
  work_posteriors = v_posterior_vac;
  work_min2_vac = min2_vac;
  work_syndromes.assign(num_checks, 0);
  check_parity.assign(num_checks, 0);
  active_checks_words.assign(num_check_words, 0);
  next_active_checks_words.assign(num_check_words, 0);

  var_dirty_stamp.assign(num_vars, 0);
  dirty_gen = 1;
  dirty_vars.clear();

  check_dirty_stamp.assign(num_checks, 0);
  check_dirty_gen = 1;
  dirty_checks.clear();
}

size_t TannerGraph::count_edges() const {
  if (!pending_edges.empty()) return pending_edges.size();
  return num_edges;
}

}  // namespace bp
