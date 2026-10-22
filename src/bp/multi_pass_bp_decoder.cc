#include "bp/multi_pass_bp_decoder.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <utility>

#include "bp/check_update.h"
#include "bp/tesseract_post_processor.h"

namespace bp {

std::vector<int> classify_detector_components(const stim::DetectorErrorModel& dem) {
  stim::DetectorErrorModel flattened = dem.flattened();
  size_t num_detectors = flattened.count_detectors();
  std::vector<int> detector_components(num_detectors, -1);

  bool all_have_basis_tags = true;
  std::vector<bool> has_detector_instruction(num_detectors, false);
  for (const stim::DemInstruction& instruction : flattened.instructions) {
    if (instruction.type != stim::DemInstructionType::DEM_DETECTOR) continue;
    if (instruction.target_data.size() != 1 || !instruction.target_data[0].is_relative_detector_id()) {
      all_have_basis_tags = false;
      break;
    }
    size_t det = instruction.target_data[0].val();
    if (det >= num_detectors || instruction.tag.empty()) {
      all_have_basis_tags = false;
      break;
    }
    nlohmann::json metadata = nlohmann::json::parse(instruction.tag, nullptr, false);
    if (metadata.is_discarded() || !metadata.is_object()) {
      all_have_basis_tags = false;
      break;
    }
    std::string basis;
    if (metadata.contains("measure_basis") && metadata["measure_basis"].is_string()) {
      basis = metadata["measure_basis"].get<std::string>();
    } else if (metadata.contains("md") && metadata["md"].is_object()) {
      const auto& md = metadata["md"];
      if (md.contains("measure_basis") && md["measure_basis"].is_string()) {
        basis = md["measure_basis"].get<std::string>();
      } else if (md.contains("basis") && md["basis"].is_string()) {
        basis = md["basis"].get<std::string>();
      }
    } else if (metadata.contains("basis") && metadata["basis"].is_string()) {
      basis = metadata["basis"].get<std::string>();
    }
    if (basis == "X") {
      detector_components[det] = 0;
    } else if (basis == "Z") {
      detector_components[det] = 1;
    } else {
      all_have_basis_tags = false;
      break;
    }
    has_detector_instruction[det] = true;
  }

  if (all_have_basis_tags) {
    for (size_t d = 0; d < num_detectors; ++d) {
      if (!has_detector_instruction[d] || detector_components[d] < 0) {
        all_have_basis_tags = false;
        break;
      }
    }
  }

  if (all_have_basis_tags) {
    return detector_components;
  }

  // 2. Check if DEM already has explicit '^' decomposition separators
  std::vector<size_t> parent(num_detectors);
  for (size_t d = 0; d < num_detectors; ++d) parent[d] = d;
  auto find_root = [&](size_t x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  auto unite = [&](size_t a, size_t b) {
    size_t ra = find_root(a);
    size_t rb = find_root(b);
    if (ra != rb) parent[rb] = ra;
  };

  std::vector<bool> det_active(num_detectors, false);
  bool has_caret = false;
  for (const stim::DemInstruction& instruction : flattened.instructions) {
    if (instruction.type != stim::DemInstructionType::DEM_ERROR) continue;
    for (const auto& t : instruction.target_data) {
      if (t.is_separator()) {
        has_caret = true;
      } else if (t.is_relative_detector_id() && t.val() < num_detectors) {
        det_active[t.val()] = true;
      }
    }
  }

  if (has_caret) {
    for (const stim::DemInstruction& instruction : flattened.instructions) {
      if (instruction.type != stim::DemInstructionType::DEM_ERROR) continue;
      instruction.for_separated_targets([&](std::span<const stim::DemTarget> group) {
        size_t first_det = std::numeric_limits<size_t>::max();
        for (const auto& t : group) {
          if (t.is_relative_detector_id() && t.val() < num_detectors) {
            if (first_det == std::numeric_limits<size_t>::max()) {
              first_det = t.val();
            } else {
              unite(first_det, t.val());
            }
          }
        }
      });
    }
  } else {
    // 3. Pure Detector-Set Reducibility Union-Find on irreducible degree-2 and degree-3
    // error mechanisms (ignoring observable targets).
    std::set<std::vector<uint32_t>> unique_det_sets;
    std::set<std::vector<uint32_t>> deg1_sets;
    std::set<std::vector<uint32_t>> deg2_sets;
    std::set<std::vector<uint32_t>> deg3_sets;

    for (const stim::DemInstruction& instruction : flattened.instructions) {
      if (instruction.type != stim::DemInstructionType::DEM_ERROR) continue;
      double p = instruction.arg_data.empty() ? 0.0 : instruction.arg_data[0];
      if (p <= 0.0) continue;
      std::vector<uint32_t> dets;
      for (const auto& t : instruction.target_data) {
        if (t.is_relative_detector_id() && t.val() < num_detectors) {
          dets.push_back(static_cast<uint32_t>(t.val()));
        }
      }
      std::sort(dets.begin(), dets.end());
      size_t w = 0;
      for (size_t r = 0; r < dets.size();) {
        size_t r2 = r + 1;
        while (r2 < dets.size() && dets[r2] == dets[r]) ++r2;
        if ((r2 - r) & 1) dets[w++] = dets[r];
        r = r2;
      }
      dets.resize(w);
      if (dets.empty()) continue;
      unique_det_sets.insert(dets);
      if (dets.size() == 1) {
        deg1_sets.insert(dets);
      } else if (dets.size() == 2) {
        deg2_sets.insert(dets);
      } else if (dets.size() == 3) {
        deg3_sets.insert(dets);
      }
    }

    // Union irreducible degree-2 mechanisms (not {1} + {1})
    for (const auto& s : deg2_sets) {
      bool reducible_deg2 =
          (deg1_sets.count({s[0]}) > 0 && deg1_sets.count({s[1]}) > 0);
      if (!reducible_deg2) {
        unite(s[0], s[1]);
      }
    }
    // Union irreducible degree-3 mechanisms (not {1} + {2})
    for (const auto& s : deg3_sets) {
      bool reducible_deg3 =
          (deg1_sets.count({s[0]}) > 0 && deg2_sets.count({s[1], s[2]}) > 0) ||
          (deg1_sets.count({s[1]}) > 0 && deg2_sets.count({s[0], s[2]}) > 0) ||
          (deg1_sets.count({s[2]}) > 0 && deg2_sets.count({s[0], s[1]}) > 0);
      if (!reducible_deg3) {
        unite(s[0], s[1]);
        unite(s[0], s[2]);
      }
    }
  }

  std::map<size_t, int> comp_map;
  for (size_t d = 0; d < num_detectors; ++d) {
    if (!det_active[d]) continue;
    size_t r = find_root(d);
    if (comp_map.find(r) == comp_map.end()) {
      int next_id = static_cast<int>(comp_map.size());
      comp_map[r] = next_id;
    }
  }

  if (comp_map.size() == 2) {
    for (size_t d = 0; d < num_detectors; ++d) {
      detector_components[d] = det_active[d] ? comp_map[find_root(d)] : 0;
    }
    return detector_components;
  }

  // If all active detectors are connected in a single irreducible component (comp_map.size() <= 1),
  // the circuit is non-CSS (entangled X/Z) and must not be torn apart by coordinate heuristics.
  if (comp_map.size() <= 1) {
    detector_components.assign(num_detectors, 0);
    return detector_components;
  }

  // 4. Otherwise (comp_map.size() > 2), inspect detector coordinates only if every
  // irreducible Union-Find component is monochromatic under the coordinate assignment.
  std::set<uint64_t> all_dets;
  for (size_t d = 0; d < num_detectors; ++d) {
    all_dets.insert(d);
  }
  auto coords_map = dem.get_detector_coordinates(all_dets);
  double max_last_coord = -std::numeric_limits<double>::infinity();
  for (const auto& [det, coord] : coords_map) {
    if (!coord.empty()) {
      max_last_coord = std::max(max_last_coord, coord.back());
    }
  }

  for (size_t d = 0; d < num_detectors; ++d) {
    auto it = coords_map.find(d);
    if (it != coords_map.end() && !it->second.empty()) {
      double last_c = it->second.back();
      if (max_last_coord >= 3.0 && max_last_coord <= 6.0) {
        detector_components[d] = (last_c < 3.0) ? 0 : 1;
      } else {
        detector_components[d] = static_cast<int>(std::lround(last_c)) & 1;
      }
    } else {
      detector_components[d] = 0;
    }
  }

  // Verify no irreducible Union-Find component was split across two different labels
  std::map<size_t, int> root_label;
  for (size_t d = 0; d < num_detectors; ++d) {
    if (!det_active[d]) continue;
    size_t r = find_root(d);
    auto it = root_label.find(r);
    if (it == root_label.end()) {
      root_label[r] = detector_components[d];
    } else if (it->second != detector_components[d]) {
      detector_components.assign(num_detectors, 0);
      return detector_components;
    }
  }
  return detector_components;
}

MultiPassBpDecoder::MultiPassBpDecoder(const stim::DetectorErrorModel& dem,
                                       const MultiPassBpConfig& config)
    : num_passes_(config.num_passes),
      strategy_(config.strategy),
      soft_reweight_(config.soft_reweight || config.params.use_soft_reweighting) {
  if (num_passes_ < 1) {
    throw std::invalid_argument("num_passes must be at least 1.");
  }
  initialize(dem, config.detector_components, config.params);
}

MultiPassBpDecoder::MultiPassBpDecoder(const stim::DetectorErrorModel& dem,
                                       const BPParams& params, size_t num_passes,
                                       const std::vector<int>& detector_components,
                                       tesseract_decoder::SchedulingStrategy strategy)
    : num_passes_(num_passes), strategy_(strategy), soft_reweight_(params.use_soft_reweighting) {
  if (num_passes_ < 1) {
    throw std::invalid_argument("num_passes must be at least 1.");
  }
  initialize(dem, detector_components, params);
}

void MultiPassBpDecoder::initialize(const stim::DetectorErrorModel& dem,
                                    const std::vector<int>& detector_components,
                                    const BPParams& params) {
  std::vector<int> comps = detector_components;
  if (comps.empty()) {
    comps = classify_detector_components(dem);
  }

  auto setup_single_component_fallback = [&]() {
    single_component_fallback_ = true;
    size_t num_dets = dem.flattened().count_detectors();
    global_det_to_comp_id_.assign(num_dets, 0);
    component_decoders_[0].assignment_label = 0;
    component_decoders_[0].active_detector_count = num_dets;
    component_decoders_[0].decoder = std::make_unique<TesseractBpDecoder>(dem, params);
    component_decoders_[0].post_processor = component_decoders_[0].decoder->create_post_processor();
    component_decoders_[0].affects_observable =
        (component_decoders_[0].decoder->num_observables() > 0);

    stim::DetectorErrorModel empty_dem;
    component_decoders_[1].assignment_label = 1;
    component_decoders_[1].active_detector_count = 0;
    component_decoders_[1].decoder = std::make_unique<TesseractBpDecoder>(empty_dem, params);
    component_decoders_[1].post_processor = component_decoders_[1].decoder->create_post_processor();
    component_decoders_[1].affects_observable = false;

    pass_schedule_ = {{0}};
  };

  std::set<int> distinct_labels(comps.begin(), comps.end());
  if (distinct_labels.size() != 2) {
    setup_single_component_fallback();
    return;
  }

  tesseract_decoder::TwoComponentDem prepared;
  try {
    prepared = tesseract_decoder::prepare_two_component_dem(dem, comps);
  } catch (const std::invalid_argument&) {
    setup_single_component_fallback();
    return;
  }
  global_det_to_comp_id_ = prepared.detector_components;

  for (size_t d = 0; d < global_det_to_comp_id_.size(); ++d) {
    int c = global_det_to_comp_id_[d];
    component_decoders_[c].active_detector_count++;
  }

  tesseract_decoder::ReweightProbsMap reweight_probabilities;
  if (num_passes_ > 1) {
    reweight_probabilities = tesseract_decoder::process_dem_correlations(prepared);
  }

  std::array<std::map<tesseract_decoder::common::Symptom, std::pair<size_t, LLR_INT>,
                      tesseract_decoder::common::Symptom::less>,
             2>
      symptom_to_var;
  std::array<std::map<std::vector<int>, std::pair<size_t, LLR_INT>>, 2> dets_to_var;

  for (size_t c = 0; c < 2; ++c) {
    auto& comp_dec = component_decoders_[c];
    comp_dec.assignment_label = prepared.assignment_labels[c];
    comp_dec.decoder = std::make_unique<TesseractBpDecoder>(prepared.component_dems[c], params);
    comp_dec.post_processor = comp_dec.decoder->create_post_processor();

    size_t n_vars = comp_dec.decoder->num_variables();
    comp_dec.affects_observable = (comp_dec.decoder->num_observables() > 0);

    for (size_t v = 0; v < n_vars; ++v) {
      auto sym = comp_dec.decoder->variable_symptom(v);
      LLR_INT prior = comp_dec.decoder->graph().variable_nodes[v].prior;
      if (num_passes_ > 1) {
        symptom_to_var[c][sym] = {v, prior};
        dets_to_var[c][sym.detectors] = {v, prior};
      }
    }
  }

  if (num_passes_ > 1) {
    for (const auto& [causal_symptom, probabilities] : reweight_probabilities) {
      if (causal_symptom.detectors.empty()) continue;
      int causal_comp = global_det_to_comp_id_.at(causal_symptom.detectors.at(0));
      size_t source_var = 0;
      auto it_causal = symptom_to_var[causal_comp].find(causal_symptom);
      if (it_causal != symptom_to_var[causal_comp].end()) {
        source_var = it_causal->second.first;
      } else {
        auto it_d = dets_to_var[causal_comp].find(causal_symptom.detectors);
        if (it_d == dets_to_var[causal_comp].end()) continue;
        source_var = it_d->second.first;
      }

      for (const auto& prob : probabilities) {
        if (prob.affected_symptom.detectors.empty()) continue;
        int target_comp = global_det_to_comp_id_.at(prob.affected_symptom.detectors.at(0));
        size_t target_var = 0;
        LLR_INT orig_llr = 0;
        auto it_target = symptom_to_var[target_comp].find(prob.affected_symptom);
        if (it_target != symptom_to_var[target_comp].end()) {
          target_var = it_target->second.first;
          orig_llr = it_target->second.second;
        } else {
          auto it_td = dets_to_var[target_comp].find(prob.affected_symptom.detectors);
          if (it_td == dets_to_var[target_comp].end()) continue;
          target_var = it_td->second.first;
          orig_llr = it_td->second.second;
        }

        LLR_INT reweight_llr = prob_double_to_llr_int(std::min(prob.probability, 0.499));
        if (reweight_llr < orig_llr) {
          auto& rules_vec = component_decoders_[causal_comp].reweight_rules[source_var];
          bool merged_existing = false;
          for (auto& existing : rules_vec) {
            if (existing.target_comp_idx == static_cast<size_t>(target_comp) &&
                existing.target_var_idx == target_var) {
              existing.reweight_llr = std::min(existing.reweight_llr, reweight_llr);
              merged_existing = true;
              break;
            }
          }
          if (!merged_existing) {
            rules_vec.push_back(
                LocalReweightRule{static_cast<size_t>(target_comp), target_var, reweight_llr});
          }
        }
      }
    }
  }

  if (strategy_ == tesseract_decoder::SchedulingStrategy::Static) {
    build_static_schedule();
  } else {
    build_causal_schedule();
  }
}

void MultiPassBpDecoder::build_static_schedule() {
  pass_schedule_.assign(num_passes_, {0, 1});
}

void MultiPassBpDecoder::build_causal_schedule() {
  std::vector<std::set<size_t>> schedule_sets(num_passes_);
  for (size_t c = 0; c < 2; ++c) {
    if (component_decoders_[c].affects_observable) {
      schedule_sets.back().insert(c);
    }
  }
  if (schedule_sets.back().empty()) {
    schedule_sets.back().insert(0);
  }

  for (int pass = static_cast<int>(num_passes_) - 2; pass >= 0; --pass) {
    for (size_t target_comp : schedule_sets[pass + 1]) {
      for (size_t src_comp = 0; src_comp < 2; ++src_comp) {
        for (const auto& [var_idx, rules] : component_decoders_[src_comp].reweight_rules) {
          for (const auto& rule : rules) {
            if (rule.target_comp_idx == target_comp) {
              schedule_sets[pass].insert(src_comp);
            }
          }
        }
      }
    }
    if (schedule_sets[pass].empty()) {
      schedule_sets[pass] = schedule_sets[pass + 1];
    }
  }

  pass_schedule_.assign(num_passes_, {});
  for (size_t p = 0; p < num_passes_; ++p) {
    pass_schedule_[p].assign(schedule_sets[p].begin(), schedule_sets[p].end());
  }
}

std::vector<std::vector<uint8_t>> MultiPassBpDecoder::decode_batch(
    const std::vector<std::vector<uint64_t>>& detection_events_batch) {
  size_t num_shots = detection_events_batch.size();
  std::vector<std::vector<uint8_t>> results(num_shots);
  if (num_shots == 0) return results;

  std::array<std::vector<std::vector<uint64_t>>, 2> comp_syndromes;
  comp_syndromes[0].resize(num_shots);
  comp_syndromes[1].resize(num_shots);
  for (size_t s = 0; s < num_shots; ++s) {
    for (uint64_t d : detection_events_batch[s]) {
      if (d < global_det_to_comp_id_.size()) {
        int c = global_det_to_comp_id_[d];
        comp_syndromes[c][s].push_back(d);
      }
    }
  }

  std::array<std::vector<LLR_INT>, 2> base_priors;
  std::array<std::vector<std::vector<LLR_INT>>, 2> custom_priors_per_shot;
  for (size_t c = 0; c < 2; ++c) {
    size_t n_vars = component_decoders_[c].decoder->num_variables();
    const auto& var_nodes = component_decoders_[c].decoder->graph().variable_nodes;
    base_priors[c].resize(n_vars);
    for (size_t v = 0; v < n_vars; ++v) {
      base_priors[c][v] = var_nodes[v].prior;
    }
    custom_priors_per_shot[c].assign(num_shots, base_priors[c]);
  }

  auto compute_reweighted_priors = [&](size_t src_comp, const std::vector<size_t>& src_errors,
                                       size_t dst_comp) -> std::vector<LLR_INT> {
    std::vector<LLR_INT> dst_priors = base_priors[dst_comp];
    const auto& reweight_map = component_decoders_[src_comp].reweight_rules;
    if (reweight_map.empty()) return dst_priors;
    for (size_t v_src : src_errors) {
      auto it = reweight_map.find(v_src);
      if (it == reweight_map.end()) continue;
      for (const auto& rule : it->second) {
        if (rule.target_comp_idx != dst_comp) continue;
        LLR_INT& cur_p = dst_priors[rule.target_var_idx];
        if (soft_reweight_) {
          cur_p = std::min(cur_p, static_cast<LLR_INT>((cur_p + 3 * rule.reweight_llr) / 4));
        } else {
          cur_p = std::min(cur_p, rule.reweight_llr);
        }
      }
    }
    return dst_priors;
  };

  if (!single_component_fallback_ && num_passes_ >= 3 &&
      strategy_ == tesseract_decoder::SchedulingStrategy::Causal &&
      (component_decoders_[0].affects_observable != component_decoders_[1].affects_observable)) {
    const size_t obs_comp = component_decoders_[0].affects_observable ? 0 : 1;
    const size_t aux_comp = 1 - obs_comp;

    auto compute_joint_cost = [&](const std::vector<size_t>& aux_errs,
                                  const std::vector<size_t>& obs_errs,
                                  const std::vector<LLR_INT>& obs_reweighted_priors) -> int64_t {
      int64_t cost = 0;
      for (size_t v : aux_errs) {
        if (v < base_priors[aux_comp].size()) {
          cost += std::max<int64_t>(1, base_priors[aux_comp][v]);
        }
      }
      for (size_t v : obs_errs) {
        if (v < obs_reweighted_priors.size()) {
          cost += std::max<int64_t>(1, obs_reweighted_priors[v]);
        }
      }
      return cost;
    };

    // Trajectory A: aux (clean) -> obs_A
    std::vector<std::vector<size_t>> aux_errs_A;
    component_decoders_[aux_comp].decoder->decode_batch(
        comp_syndromes[aux_comp], component_decoders_[aux_comp].post_processor, nullptr,
        &aux_errs_A);

    std::vector<std::vector<LLR_INT>> obs_priors_A(num_shots);
    for (size_t s = 0; s < num_shots; ++s) {
      obs_priors_A[s] = compute_reweighted_priors(aux_comp, aux_errs_A[s], obs_comp);
    }
    std::vector<std::vector<size_t>> obs_errs_A;
    std::vector<std::vector<uint8_t>> obs_res_A =
        component_decoders_[obs_comp].decoder->decode_batch(
            comp_syndromes[obs_comp], component_decoders_[obs_comp].post_processor, &obs_priors_A,
            &obs_errs_A);

    // Trajectory B: obs (clean) -> aux_B -> obs_B (on differing shots)
    std::vector<std::vector<size_t>> obs_errs_0;
    std::vector<std::vector<uint8_t>> obs_res_0 =
        component_decoders_[obs_comp].decoder->decode_batch(
            comp_syndromes[obs_comp], component_decoders_[obs_comp].post_processor, nullptr,
            &obs_errs_0);

    std::vector<std::vector<LLR_INT>> aux_priors_B(num_shots);
    for (size_t s = 0; s < num_shots; ++s) {
      aux_priors_B[s] = compute_reweighted_priors(obs_comp, obs_errs_0[s], aux_comp);
    }
    std::vector<std::vector<size_t>> aux_errs_B;
    component_decoders_[aux_comp].decoder->decode_batch(
        comp_syndromes[aux_comp], component_decoders_[aux_comp].post_processor, &aux_priors_B,
        &aux_errs_B);

    std::vector<size_t> diff_shots;
    std::vector<std::vector<uint64_t>> diff_obs_syndromes;
    std::vector<std::vector<LLR_INT>> diff_obs_priors_B;
    for (size_t s = 0; s < num_shots; ++s) {
      if (aux_errs_B[s] != aux_errs_A[s]) {
        diff_shots.push_back(s);
        diff_obs_syndromes.push_back(comp_syndromes[obs_comp][s]);
        diff_obs_priors_B.push_back(compute_reweighted_priors(aux_comp, aux_errs_B[s], obs_comp));
      }
    }

    std::vector<std::vector<size_t>> diff_obs_errs_B;
    std::vector<std::vector<uint8_t>> diff_obs_res_B;
    if (!diff_shots.empty()) {
      diff_obs_res_B = component_decoders_[obs_comp].decoder->decode_batch(
          diff_obs_syndromes, component_decoders_[obs_comp].post_processor, &diff_obs_priors_B,
          &diff_obs_errs_B);
    }

    size_t num_obs = num_observables();
    for (size_t s = 0; s < num_shots; ++s) {
      results[s].assign(num_obs, 0);
      const auto& c_corr = obs_res_A[s];
      for (size_t o = 0; o < c_corr.size() && o < num_obs; ++o) {
        results[s][o] ^= c_corr[o];
      }
    }

    for (size_t idx = 0; idx < diff_shots.size(); ++idx) {
      size_t s = diff_shots[idx];
      int64_t cost_A = compute_joint_cost(aux_errs_A[s], obs_errs_A[s], obs_priors_A[s]);
      int64_t cost_B = compute_joint_cost(aux_errs_B[s], diff_obs_errs_B[idx], diff_obs_priors_B[idx]);
      int64_t cost_0 = compute_joint_cost(aux_errs_B[s], obs_errs_0[s], diff_obs_priors_B[idx]);

      const std::vector<uint8_t>* best_corr = &obs_res_A[s];
      int64_t best_c = cost_A;
      if (cost_B < best_c) {
        best_c = cost_B;
        best_corr = &diff_obs_res_B[idx];
      }
      if (cost_0 < best_c) {
        best_c = cost_0;
        best_corr = &obs_res_0[s];
      }
      results[s].assign(num_obs, 0);
      for (size_t o = 0; o < best_corr->size() && o < num_obs; ++o) {
        results[s][o] ^= (*best_corr)[o];
      }
    }

    return results;
  }

  std::array<std::vector<std::vector<size_t>>, 2> comp_predicted_errors;
  std::array<std::vector<std::vector<uint8_t>>, 2> comp_results;

  for (size_t pass = 0; pass < pass_schedule_.size(); ++pass) {
    const bool is_final_pass = (pass + 1 == pass_schedule_.size());
    for (size_t comp : pass_schedule_[pass]) {
      comp_results[comp] = component_decoders_[comp].decoder->decode_batch(
          comp_syndromes[comp], component_decoders_[comp].post_processor,
          (pass == 0) ? nullptr : &custom_priors_per_shot[comp],
          is_final_pass ? nullptr : &comp_predicted_errors[comp]);
    }

    if (!is_final_pass) {
      for (size_t source_comp : pass_schedule_[pass]) {
        const auto& reweight_map = component_decoders_[source_comp].reweight_rules;
        if (reweight_map.empty()) continue;

        for (size_t s = 0; s < num_shots; ++s) {
          for (size_t v_src : comp_predicted_errors[source_comp][s]) {
            auto it = reweight_map.find(v_src);
            if (it == reweight_map.end()) continue;
            for (const auto& rule : it->second) {
              LLR_INT& cur_p = custom_priors_per_shot[rule.target_comp_idx][s][rule.target_var_idx];
              if (soft_reweight_) {
                cur_p = std::min(cur_p, static_cast<LLR_INT>((cur_p + 3 * rule.reweight_llr) / 4));
              } else {
                cur_p = std::min(cur_p, rule.reweight_llr);
              }
            }
          }
        }
      }
    }
  }

  size_t num_obs = num_observables();
  for (size_t s = 0; s < num_shots; ++s) {
    results[s].assign(num_obs, 0);
    for (size_t comp : pass_schedule_.back()) {
      const auto& c_corr = comp_results[comp][s];
      for (size_t o = 0; o < c_corr.size() && o < num_obs; ++o) {
        results[s][o] ^= c_corr[o];
      }
    }
  }

  return results;
}

std::vector<uint8_t> MultiPassBpDecoder::decode(const std::vector<uint64_t>& detection_events) {
  auto batch_res = decode_batch({detection_events});
  return batch_res.empty() ? std::vector<uint8_t>{} : batch_res[0];
}

size_t MultiPassBpDecoder::num_observables() const {
  size_t max_obs = 0;
  for (size_t c = 0; c < 2; ++c) {
    max_obs = std::max(max_obs, component_decoders_[c].decoder->num_observables());
  }
  return max_obs;
}

size_t MultiPassBpDecoder::total_detectors() const {
  return component_decoders_[0].active_detector_count + component_decoders_[1].active_detector_count;
}

size_t MultiPassBpDecoder::total_variables() const {
  return component_decoders_[0].decoder->num_variables() + component_decoders_[1].decoder->num_variables();
}

BpDecodeProfile MultiPassBpDecoder::profile() const {
  BpDecodeProfile agg;
  for (size_t c = 0; c < 2; ++c) {
    const auto& cp = component_decoders_[c].decoder->profile();
    agg.t_construct += cp.t_construct;
    agg.t_bp += cp.t_bp;
    agg.t_deinterleave += cp.t_deinterleave;
    agg.t_post += cp.t_post;
    agg.sum_iters += cp.sum_iters;
    agg.num_decode_calls += cp.num_decode_calls;
    for (size_t i = 0; i < BpDecodeProfile::kNumIterBuckets; ++i) {
      agg.iter_hist[i] += cp.iter_hist[i];
    }
  }

  size_t final_comp = pass_schedule_.empty() ? 0 : pass_schedule_.back().front();
  if (!pass_schedule_.empty()) {
    for (size_t comp : pass_schedule_.back()) {
      if (component_decoders_[comp].affects_observable) {
        final_comp = comp;
        break;
      }
    }
  }
  const auto& fp = component_decoders_[final_comp].decoder->profile();
  agg.num_shots = fp.num_shots;
  agg.num_converged = fp.num_converged;
  agg.conv_iter_hist = fp.conv_iter_hist;
  return agg;
}

OsdProfile MultiPassBpDecoder::osd_profile() const {
  OsdProfile agg;
  for (size_t c = 0; c < 2; ++c) {
    auto osd_pp = std::dynamic_pointer_cast<OsdPostProcessor>(component_decoders_[c].post_processor);
    if (osd_pp) {
      agg.merge(osd_pp->profile());
    }
    auto tess_pp = std::dynamic_pointer_cast<TesseractPostProcessor>(component_decoders_[c].post_processor);
    if (tess_pp) {
      agg.merge(tess_pp->profile());
    }
  }
  return agg;
}

void MultiPassBpDecoder::reset_profile() {
  for (size_t c = 0; c < 2; ++c) {
    component_decoders_[c].decoder->reset_profile();
    auto osd_pp = std::dynamic_pointer_cast<OsdPostProcessor>(component_decoders_[c].post_processor);
    if (osd_pp) {
      osd_pp->reset_profile();
    }
    auto tess_pp = std::dynamic_pointer_cast<TesseractPostProcessor>(component_decoders_[c].post_processor);
    if (tess_pp) {
      tess_pp->reset_profile();
    }
  }
}

}  // namespace bp
