#include "bp/multi_pass_bp_decoder.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>

#include "bp/bp_params.h"

namespace bp {

std::vector<int> classify_detector_components(const stim::DetectorErrorModel& dem) {
  stim::DetectorErrorModel flattened = dem.flattened();
  size_t num_detectors = flattened.count_detectors();
  std::vector<int> detector_components(num_detectors, -1);

  // 1. Check explicit detector basis tags ("X" -> 0, "Z" -> 1).
  bool all_have_basis_tags = (num_detectors > 0);
  size_t tagged_count = 0;
  for (const stim::DemInstruction& inst : flattened.instructions) {
    if (inst.type != stim::DemInstructionType::DEM_DETECTOR) continue;
    if (inst.target_data.size() != 1 || !inst.target_data[0].is_relative_detector_id() ||
        inst.target_data[0].val() >= num_detectors || inst.tag.empty()) {
      all_have_basis_tags = false;
      break;
    }
    nlohmann::json meta = nlohmann::json::parse(inst.tag, nullptr, false);
    if (meta.is_discarded() || !meta.is_object()) {
      all_have_basis_tags = false;
      break;
    }
    const auto& obj = (meta.contains("md") && meta["md"].is_object()) ? meta["md"] : meta;
    std::string basis = (obj.contains("measure_basis") && obj["measure_basis"].is_string())
                            ? obj["measure_basis"].get<std::string>()
                            : ((obj.contains("basis") && obj["basis"].is_string())
                                   ? obj["basis"].get<std::string>()
                                   : "");
    size_t det = inst.target_data[0].val();
    if (basis == "X" || basis == "Z") {
      if (detector_components[det] < 0) tagged_count++;
      detector_components[det] = (basis == "X") ? 0 : 1;
    } else {
      all_have_basis_tags = false;
      break;
    }
  }
  if (all_have_basis_tags && tagged_count == num_detectors) return detector_components;

  // 2. Union-Find over explicit '^' separators or irreducible degree-2/3 detector sets.
  std::vector<size_t> parent(num_detectors);
  std::iota(parent.begin(), parent.end(), 0);
  auto find_root = [&](size_t x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  auto unite = [&](size_t a, size_t b) {
    size_t ra = find_root(a), rb = find_root(b);
    if (ra != rb) parent[rb] = ra;
  };

  std::vector<bool> det_active(num_detectors, false);
  bool has_caret = false;
  for (const stim::DemInstruction& inst : flattened.instructions) {
    if (inst.type != stim::DemInstructionType::DEM_ERROR) continue;
    for (const auto& t : inst.target_data) {
      if (t.is_separator()) has_caret = true;
      else if (t.is_relative_detector_id() && t.val() < num_detectors) det_active[t.val()] = true;
    }
  }

  if (has_caret) {
    for (const stim::DemInstruction& inst : flattened.instructions) {
      if (inst.type != stim::DemInstructionType::DEM_ERROR) continue;
      inst.for_separated_targets([&](std::span<const stim::DemTarget> group) {
        size_t first_det = SIZE_MAX;
        for (const auto& t : group) {
          if (t.is_relative_detector_id() && t.val() < num_detectors) {
            if (first_det == SIZE_MAX) first_det = t.val();
            else unite(first_det, t.val());
          }
        }
      });
    }
  } else {
    std::set<std::vector<uint32_t>> deg1_sets, deg2_sets, deg3_sets;
    for (const stim::DemInstruction& inst : flattened.instructions) {
      if (inst.type != stim::DemInstructionType::DEM_ERROR || inst.arg_data.empty() ||
          inst.arg_data[0] <= 0.0) {
        continue;
      }
      std::vector<uint32_t> dets;
      for (const auto& t : inst.target_data) {
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
      if (dets.size() == 1) deg1_sets.insert(dets);
      else if (dets.size() == 2) deg2_sets.insert(dets);
      else if (dets.size() == 3) deg3_sets.insert(dets);
    }
    for (const auto& s : deg2_sets) {
      if (!(deg1_sets.count({s[0]}) && deg1_sets.count({s[1]}))) unite(s[0], s[1]);
    }
    for (const auto& s : deg3_sets) {
      if (!((deg1_sets.count({s[0]}) && deg2_sets.count({s[1], s[2]})) ||
            (deg1_sets.count({s[1]}) && deg2_sets.count({s[0], s[2]})) ||
            (deg1_sets.count({s[2]}) && deg2_sets.count({s[0], s[1]})))) {
        unite(s[0], s[1]);
        unite(s[0], s[2]);
      }
    }
  }

  std::map<size_t, int> comp_map;
  for (size_t d = 0; d < num_detectors; ++d) {
    if (det_active[d]) comp_map.try_emplace(find_root(d), static_cast<int>(comp_map.size()));
  }
  if (comp_map.size() == 2) {
    for (size_t d = 0; d < num_detectors; ++d) {
      detector_components[d] = det_active[d] ? comp_map[find_root(d)] : 0;
    }
    return detector_components;
  }
  if (comp_map.size() <= 1) return std::vector<int>(num_detectors, 0);

  // 3. Coordinate fallback when >2 irreducible components exist, provided no component is split.
  std::set<uint64_t> all_dets;
  for (size_t d = 0; d < num_detectors; ++d) all_dets.insert(d);
  auto coords_map = dem.get_detector_coordinates(all_dets);
  double max_last_coord = -std::numeric_limits<double>::infinity();
  for (const auto& [det, coord] : coords_map) {
    if (!coord.empty()) max_last_coord = std::max(max_last_coord, coord.back());
  }
  for (size_t d = 0; d < num_detectors; ++d) {
    auto it = coords_map.find(d);
    if (it != coords_map.end() && !it->second.empty()) {
      double last_c = it->second.back();
      detector_components[d] = (max_last_coord >= 3.0 && max_last_coord <= 6.0)
                                   ? ((last_c < 3.0) ? 0 : 1)
                                   : (static_cast<int>(std::lround(last_c)) & 1);
    } else {
      detector_components[d] = 0;
    }
  }
  std::map<size_t, int> root_label;
  for (size_t d = 0; d < num_detectors; ++d) {
    if (!det_active[d]) continue;
    auto [it, inserted] = root_label.try_emplace(find_root(d), detector_components[d]);
    if (!inserted && it->second != detector_components[d]) {
      return std::vector<int>(num_detectors, 0);
    }
  }
  return detector_components;
}

MultiPassBpDecoder::MultiPassBpDecoder(const stim::DetectorErrorModel& dem,
                                       const MultiPassBpConfig& config)
    : num_passes_(config.num_passes),
      soft_reweight_(config.soft_reweight && config.params.use_soft_reweighting) {
  if (num_passes_ < 1) throw std::invalid_argument("num_passes must be at least 1.");
  initialize(dem, config.detector_components, config.params);
}

MultiPassBpDecoder::MultiPassBpDecoder(const stim::DetectorErrorModel& dem,
                                       const BPParams& params, size_t num_passes,
                                       const std::vector<int>& detector_components)
    : num_passes_(num_passes), soft_reweight_(params.use_soft_reweighting) {
  if (num_passes_ < 1) throw std::invalid_argument("num_passes must be at least 1.");
  initialize(dem, detector_components, params);
}

void MultiPassBpDecoder::initialize(const stim::DetectorErrorModel& dem,
                                    const std::vector<int>& detector_components,
                                    const BPParams& params) {
  std::vector<int> comps =
      detector_components.empty() ? classify_detector_components(dem) : detector_components;

  auto setup_single_component_fallback = [&]() {
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

  if (std::set<int>(comps.begin(), comps.end()).size() != 2) {
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
  for (int c : global_det_to_comp_id_) component_decoders_[c].active_detector_count++;

  // Compact each component onto its own detectors so that its Tanner graph (and the dense OSD
  // matrix built from it) has no empty rows for the other component's detectors.
  std::array<std::vector<int>, 2> local_to_global;
  global_to_local_det_.assign(global_det_to_comp_id_.size(), 0);
  for (size_t d = 0; d < global_det_to_comp_id_.size(); ++d) {
    auto& l2g = local_to_global[global_det_to_comp_id_[d]];
    global_to_local_det_[d] = static_cast<uint32_t>(l2g.size());
    l2g.push_back(static_cast<int>(d));
  }
  auto compact_component_dem = [&](const stim::DetectorErrorModel& comp_dem, size_t c) {
    stim::DetectorErrorModel out;
    std::vector<stim::DemTarget> targets;
    for (const auto& inst : comp_dem.flattened().instructions) {
      if (inst.type != stim::DemInstructionType::DEM_ERROR) continue;
      targets.clear();
      for (const auto& t : inst.target_data) {
        targets.push_back(t.is_relative_detector_id()
                              ? stim::DemTarget::relative_detector_id(global_to_local_det_[t.val()])
                              : t);
      }
      out.append_error_instruction(inst.arg_data[0], targets, inst.tag);
    }
    if (!local_to_global[c].empty()) {
      // Pin the detector count so detectors without error mechanisms still get a check row.
      const std::vector<double> no_coords;
      out.append_detector_instruction(
          no_coords, stim::DemTarget::relative_detector_id(local_to_global[c].size() - 1), "");
    }
    return out;
  };

  std::array<std::map<tesseract_decoder::common::Symptom, std::pair<size_t, LLR_INT>,
                      tesseract_decoder::common::Symptom::less>,
             2>
      symptom_to_var;
  std::array<std::map<std::vector<int>, std::pair<size_t, LLR_INT>>, 2> dets_to_var;

  for (size_t c = 0; c < 2; ++c) {
    auto& comp_dec = component_decoders_[c];
    comp_dec.assignment_label = prepared.assignment_labels[c];
    comp_dec.decoder = std::make_unique<TesseractBpDecoder>(
        compact_component_dem(prepared.component_dems[c], c), params);
    comp_dec.post_processor = comp_dec.decoder->create_post_processor();
    comp_dec.affects_observable = (comp_dec.decoder->num_observables() > 0);

    if (num_passes_ > 1) {
      for (size_t v = 0; v < comp_dec.decoder->num_variables(); ++v) {
        auto sym = comp_dec.decoder->variable_symptom(v);
        // local_to_global is increasing, so the translated detector list stays sorted.
        for (int& d : sym.detectors) d = local_to_global[c][d];
        LLR_INT prior = comp_dec.decoder->graph().priors[v];
        symptom_to_var[c][sym] = {v, prior};
        dets_to_var[c][sym.detectors] = {v, prior};
      }
    }
  }

  if (num_passes_ > 1) {
    auto lookup_var = [&](int comp, const tesseract_decoder::common::Symptom& sym,
                          size_t& out_var, LLR_INT& out_llr) -> bool {
      if (auto it = symptom_to_var[comp].find(sym); it != symptom_to_var[comp].end()) {
        out_var = it->second.first;
        out_llr = it->second.second;
        return true;
      }
      if (auto it_d = dets_to_var[comp].find(sym.detectors); it_d != dets_to_var[comp].end()) {
        out_var = it_d->second.first;
        out_llr = it_d->second.second;
        return true;
      }
      return false;
    };

    for (const auto& [causal_symptom, probabilities] :
         tesseract_decoder::process_dem_correlations(prepared)) {
      if (causal_symptom.detectors.empty()) continue;
      int causal_comp = global_det_to_comp_id_.at(causal_symptom.detectors.at(0));
      size_t source_var = 0;
      LLR_INT unused_llr = 0;
      if (!lookup_var(causal_comp, causal_symptom, source_var, unused_llr)) continue;

      for (const auto& prob : probabilities) {
        if (prob.affected_symptom.detectors.empty()) continue;
        int target_comp = global_det_to_comp_id_.at(prob.affected_symptom.detectors.at(0));
        size_t target_var = 0;
        LLR_INT orig_llr = 0;
        if (!lookup_var(target_comp, prob.affected_symptom, target_var, orig_llr)) continue;

        LLR_INT reweight_llr = prob_double_to_llr_int(std::min(prob.probability, 0.499));
        if (reweight_llr < orig_llr) {
          auto& rules_vec = component_decoders_[causal_comp].reweight_rules[source_var];
          bool merged = false;
          for (auto& existing : rules_vec) {
            if (existing.target_comp_idx == static_cast<size_t>(target_comp) &&
                existing.target_var_idx == target_var) {
              existing.reweight_llr = std::min(existing.reweight_llr, reweight_llr);
              merged = true;
              break;
            }
          }
          if (!merged) {
            rules_vec.push_back({static_cast<size_t>(target_comp), target_var, reweight_llr});
          }
        }
      }
    }
  }
  build_causal_schedule();
}

void MultiPassBpDecoder::build_causal_schedule() {
  std::vector<std::set<size_t>> schedule_sets(num_passes_);
  for (size_t c = 0; c < 2; ++c) {
    if (component_decoders_[c].affects_observable) schedule_sets.back().insert(c);
  }
  if (schedule_sets.back().empty()) schedule_sets.back().insert(0);

  for (int pass = static_cast<int>(num_passes_) - 2; pass >= 0; --pass) {
    for (size_t target_comp : schedule_sets[pass + 1]) {
      for (size_t src_comp = 0; src_comp < 2; ++src_comp) {
        for (const auto& [var_idx, rules] : component_decoders_[src_comp].reweight_rules) {
          for (const auto& rule : rules) {
            if (rule.target_comp_idx == target_comp) schedule_sets[pass].insert(src_comp);
          }
        }
      }
    }
    if (schedule_sets[pass].empty()) schedule_sets[pass] = schedule_sets[pass + 1];
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
        comp_syndromes[global_det_to_comp_id_[d]][s].push_back(
            global_to_local_det_.empty() ? d : global_to_local_det_[d]);
      }
    }
  }

  std::array<std::vector<std::vector<LLR_INT>>, 2> custom_priors_per_shot;
  if (pass_schedule_.size() > 1) {
    for (size_t c = 0; c < 2; ++c) {
      custom_priors_per_shot[c].assign(num_shots, component_decoders_[c].decoder->graph().priors);
    }
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
              cur_p = soft_reweight_
                          ? std::min(cur_p, static_cast<LLR_INT>((cur_p + 3 * rule.reweight_llr) / 4))
                          : std::min(cur_p, rule.reweight_llr);
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
      for (size_t o = 0; o < c_corr.size() && o < num_obs; ++o) results[s][o] ^= c_corr[o];
    }
  }
  return results;
}

std::vector<uint8_t> MultiPassBpDecoder::decode(const std::vector<uint64_t>& detection_events) {
  auto batch_res = decode_batch({detection_events});
  return batch_res.empty() ? std::vector<uint8_t>{} : batch_res[0];
}

size_t MultiPassBpDecoder::num_observables() const {
  return std::max(component_decoders_[0].decoder->num_observables(),
                  component_decoders_[1].decoder->num_observables());
}

size_t MultiPassBpDecoder::total_detectors() const {
  return component_decoders_[0].active_detector_count + component_decoders_[1].active_detector_count;
}

size_t MultiPassBpDecoder::total_variables() const {
  return component_decoders_[0].decoder->num_variables() +
         component_decoders_[1].decoder->num_variables();
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
    if (component_decoders_[c].post_processor) {
      agg.merge(component_decoders_[c].post_processor->profile());
    }
  }
  return agg;
}

void MultiPassBpDecoder::reset_profile() {
  for (size_t c = 0; c < 2; ++c) {
    component_decoders_[c].decoder->reset_profile();
    if (component_decoders_[c].post_processor) {
      component_decoders_[c].post_processor->reset_profile();
    }
  }
}

}  // namespace bp
