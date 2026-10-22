#ifndef _BP_PYBIND_H
#define _BP_PYBIND_H

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bp/bp_params.h"
#include "bp/bp_trace.h"
#include "bp/hard_decision_post_processor.h"
#include "bp/multi_pass_bp_decoder.h"
#include "bp/osd_post_processor.h"
#include "bp/post_processor.h"
#include "bp/tesseract_bp_decoder.h"
#include "bp/tesseract_post_processor.h"
#include "stim.h"
#include "stim_utils.pybind.h"  // For parse_py_object

namespace py = pybind11;

namespace bp {

inline py::dict topology_to_pydict(const TannerGraphTopology& topo) {
  py::dict d;
  d["num_vars"] = topo.num_vars;
  d["num_checks"] = topo.num_checks;
  d["num_edges"] = topo.num_edges;
  d["num_observables"] = topo.num_observables;
  d["priors"] = topo.priors;
  d["obs_masks"] = topo.obs_masks;
  d["hyperedge_observables"] = topo.hyperedge_observables;
  d["edges"] = topo.edges;
  return d;
}

inline py::dict bp_trace_to_pydict(const BpDecodeTrace& tr) {
  py::dict d;
  d["topology"] = topology_to_pydict(tr.topology);
  d["detection_events"] = tr.detection_events;
  d["bp_converged"] = tr.bp_converged;
  d["num_iters"] = tr.num_iters;
  d["post_processor_type"] = tr.post_processor_type;
  d["predicted_errors"] = tr.predicted_errors;
  d["predicted_observables"] = tr.predicted_observables;

  py::list iters;
  for (const auto& it : tr.iterations) {
    py::dict id;
    id["iter"] = it.iter;
    id["var_to_check_messages"] = it.var_to_check_messages;
    id["check_to_var_messages"] = it.check_to_var_messages;
    id["posteriors"] = it.posteriors;
    id["hard_decisions"] = it.hard_decisions;
    id["check_satisfied"] = it.check_satisfied;
    id["residual_syndrome"] = it.residual_syndrome;
    id["converged"] = it.converged;
    iters.append(std::move(id));
  }
  d["iterations"] = std::move(iters);

  py::dict osd_d;
  osd_d["triggered"] = tr.osd_trace.triggered;
  osd_d["residual_syndrome"] = tr.osd_trace.residual_syndrome;
  osd_d["col_to_var"] = tr.osd_trace.col_to_var;
  osd_d["pivot_vars"] = tr.osd_trace.pivot_vars;
  osd_d["free_vars"] = tr.osd_trace.free_vars;
  osd_d["best_error"] = tr.osd_trace.best_error;
  d["osd_trace"] = std::move(osd_d);

  py::dict tess_d;
  tess_d["triggered"] = tr.tesseract_trace.triggered;
  tess_d["peeled_vars"] = tr.tesseract_trace.peeled_vars;
  tess_d["active_vars"] = tr.tesseract_trace.active_vars;
  tess_d["unsolved_dets"] = tr.tesseract_trace.unsolved_dets;
  tess_d["comp_dets"] = tr.tesseract_trace.comp_dets;
  tess_d["comp_vars"] = tr.tesseract_trace.comp_vars;
  tess_d["residual_llr_costs"] = tr.tesseract_trace.residual_llr_costs;
  tess_d["astar_flipped_vars"] = tr.tesseract_trace.astar_flipped_vars;
  tess_d["final_flipped_vars"] = tr.tesseract_trace.final_flipped_vars;
  tess_d["used_osd_fallback"] = tr.tesseract_trace.used_osd_fallback;
  d["tesseract_trace"] = std::move(tess_d);

  return d;
}

inline py::dict multipass_trace_to_pydict(const MultiPassBpDecodeTrace& mtr) {
  py::dict d;
  py::list comp_topos;
  for (const auto& t : mtr.component_topologies) {
    comp_topos.append(topology_to_pydict(t));
  }
  d["component_topologies"] = std::move(comp_topos);
  d["detector_components"] = mtr.detector_components;
  d["pass_schedule"] = mtr.pass_schedule;
  d["predicted_observables"] = mtr.predicted_observables;

  py::list passes;
  for (const auto& cp : mtr.component_passes) {
    py::dict cpd;
    cpd["pass"] = cp.pass;
    cpd["comp_idx"] = cp.comp_idx;
    cpd["trace"] = bp_trace_to_pydict(cp.trace);
    passes.append(std::move(cpd));
  }
  d["component_passes"] = std::move(passes);

  py::list reweights;
  for (const auto& ev : mtr.reweight_events) {
    py::dict evd;
    evd["pass"] = ev.pass;
    evd["source_comp_idx"] = ev.source_comp_idx;
    evd["source_var_idx"] = ev.source_var_idx;
    evd["target_comp_idx"] = ev.target_comp_idx;
    evd["target_var_idx"] = ev.target_var_idx;
    evd["old_prior"] = ev.old_prior;
    evd["new_prior"] = ev.new_prior;
    reweights.append(std::move(evd));
  }
  d["reweight_events"] = std::move(reweights);

  return d;
}

inline std::vector<uint64_t> parse_syndrome_or_detections(const py::object& syndrome_obj,
                                                          size_t num_detectors) {
  if (py::isinstance<py::array_t<bool>>(syndrome_obj)) {
    auto syndrome = syndrome_obj.cast<py::array_t<bool>>();
    if (static_cast<size_t>(syndrome.size()) != num_detectors) {
      throw std::invalid_argument("Syndrome size does not match decoder detector count.");
    }
    std::vector<uint64_t> detections;
    auto unchecked = syndrome.unchecked<1>();
    for (size_t i = 0; i < static_cast<size_t>(unchecked.size()); ++i) {
      if (unchecked(i)) {
        detections.push_back(i);
      }
    }
    return detections;
  }
  // Otherwise accept a sequence of integer detector indices or booleans
  py::sequence seq = syndrome_obj.cast<py::sequence>();
  if (seq.size() == num_detectors && seq.size() > 0 && py::isinstance<py::bool_>(seq[0])) {
    std::vector<uint64_t> detections;
    for (size_t i = 0; i < seq.size(); ++i) {
      if (seq[i].cast<bool>()) {
        detections.push_back(i);
      }
    }
    return detections;
  }
  std::vector<uint64_t> detections;
  detections.reserve(seq.size());
  for (auto item : seq) {
    detections.push_back(item.cast<uint64_t>());
  }
  return detections;
}

void add_bp_module(py::module& root) {
  auto m = root.def_submodule("bp", "Module containing the Belief Propagation decoder");

  py::class_<BPParams>(m, "BPParams", R"pbdoc(
        Configuration for the Belief Propagation decoder.
    )pbdoc")
      .def(py::init<>())
      .def_readwrite("max_iter", &BPParams::max_iter)
      .def_readwrite("update_rule", &BPParams::update_rule)
      .def_readwrite("schedule", &BPParams::schedule)
      .def_readwrite("normalization_factor", &BPParams::normalization_factor)
      .def_readwrite("random_schedule", &BPParams::random_schedule)
      .def_readwrite("random_seed", &BPParams::random_seed)
      .def_readwrite("osd_truncation_factor", &BPParams::osd_truncation_factor,
                     "Multiplier defining the maximum number of free columns for the dense OSD "
                     "Gaussian elimination matrix (e.g. 1.2 bounds to 1.2x the number of defect "
                     "detectors). Use 0.0 to disable truncation.")
      .def_readwrite("osd_early_exit", &BPParams::osd_early_exit)
      .def_readwrite("osd_order", &BPParams::osd_order)
      .def_readwrite("osd_weight", &BPParams::osd_weight)
      .def_readwrite("use_soft_reweighting", &BPParams::use_soft_reweighting)
      .def_readwrite("post_processor", &BPParams::post_processor)
      .def_readwrite("tesseract_det_beam", &BPParams::tesseract_det_beam)
      .def_readwrite("tesseract_pqlimit", &BPParams::tesseract_pqlimit);

  py::class_<PostProcessor, std::shared_ptr<PostProcessor>>(m, "PostProcessor");

  py::class_<HardDecisionPostProcessor, PostProcessor, std::shared_ptr<HardDecisionPostProcessor>>(
      m, "HardDecisionPostProcessor")
      .def(py::init<>());

  py::class_<OsdPostProcessor, PostProcessor, std::shared_ptr<OsdPostProcessor>>(m,
                                                                                 "OsdPostProcessor")
      .def(py::init<const TannerGraph<LLR_INT>&, size_t, size_t>(), py::arg("graph"),
           py::arg("osd_order") = 0, py::arg("osd_weight") = 0);

  py::class_<TesseractPostProcessor, PostProcessor, std::shared_ptr<TesseractPostProcessor>>(
      m, "TesseractPostProcessor");

  py::class_<TesseractBpDecoder>(m, "TesseractBpDecoder", R"pbdoc(
        A top-level Belief Propagation orchestrator.
    )pbdoc")
      .def(py::init([](py::object dem, const BPParams& config) {
             return std::make_unique<TesseractBpDecoder>(
                 tesseract_decoder::parse_py_object<stim::DetectorErrorModel>(dem), config);
           }),
           py::arg("dem"), py::arg("config"))
      .def("create_osd_post_processor", &TesseractBpDecoder::create_osd_post_processor,
           py::arg("osd_order") = 0, py::arg("osd_weight") = 0)
      .def("create_tesseract_post_processor", &TesseractBpDecoder::create_tesseract_post_processor)
      .def("create_post_processor", &TesseractBpDecoder::create_post_processor)
      .def_property_readonly("num_detectors", &TesseractBpDecoder::num_detectors)
      .def_property_readonly("num_variables", &TesseractBpDecoder::num_variables)
      .def_property_readonly("num_observables", &TesseractBpDecoder::num_observables)
      .def(
          "get_topology",
          [](const TesseractBpDecoder& self) { return topology_to_pydict(self.get_topology()); },
          R"pbdoc(Returns the static bipartite Tanner graph topology as a dict.)pbdoc")
      .def(
          "decode_trace",
          [](TesseractBpDecoder& self, const py::object& syndrome,
             const std::shared_ptr<PostProcessor>& post_processor) {
            std::vector<uint64_t> detections =
                parse_syndrome_or_detections(syndrome, self.num_detectors());
            return bp_trace_to_pydict(self.decode_trace(detections, post_processor));
          },
          py::arg("syndrome"), py::arg("post_processor") = nullptr,
          R"pbdoc(
            Decodes a single shot with full step-by-step BP iteration and post-processor tracing.
            Accepts either a dense boolean syndrome array or a list of active detector indices.
          )pbdoc")
      .def(
          "decode",
          [](TesseractBpDecoder& self, const py::array_t<bool>& syndrome,
             const std::shared_ptr<PostProcessor>& post_processor) {
            if ((size_t)syndrome.size() != self.num_detectors()) {
              throw std::invalid_argument("Syndrome size does not match decoder detector count.");
            }

            std::vector<uint64_t> detections;
            auto syndrome_unchecked = syndrome.unchecked<1>();
            for (size_t i = 0; i < (size_t)syndrome_unchecked.size(); ++i) {
              if (syndrome_unchecked(i)) {
                detections.push_back(i);
              }
            }

            std::vector<uint8_t> predictions = self.decode(detections, post_processor);

            size_t num_obs = self.num_observables();
            std::vector<char> result(num_obs, 0);
            for (size_t i = 0; i < predictions.size(); ++i) {
              if (predictions[i]) {
                result[i] = 1;
              }
            }

            return py::array(py::dtype::of<bool>(), result.size(), result.data());
          },
          py::arg("syndrome"), py::arg("post_processor"),
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(),
          R"pbdoc(
            Decodes a single shot from a dense boolean array using a post-processor.
            Returns predicted observables.
          )pbdoc");

  py::class_<MultiPassBpConfig>(m, "MultiPassBpConfig")
      .def(py::init<BPParams, size_t, std::vector<int>, tesseract_decoder::SchedulingStrategy, bool>(),
           py::arg("params") = BPParams(), py::arg("num_passes") = 2,
           py::arg("detector_components") = std::vector<int>(),
           py::arg("strategy") = tesseract_decoder::SchedulingStrategy::Causal,
           py::arg("soft_reweight") = true)
      .def_readwrite("params", &MultiPassBpConfig::params)
      .def_readwrite("num_passes", &MultiPassBpConfig::num_passes)
      .def_readwrite("detector_components", &MultiPassBpConfig::detector_components)
      .def_readwrite("strategy", &MultiPassBpConfig::strategy)
      .def_readwrite("soft_reweight", &MultiPassBpConfig::soft_reweight);

  py::class_<MultiPassBpDecoder>(m, "MultiPassBpDecoder", R"pbdoc(
        Multi-Pass Belief Propagation decoder.
    )pbdoc")
      .def(py::init([](py::object dem, const MultiPassBpConfig& config) {
             return std::make_unique<MultiPassBpDecoder>(
                 tesseract_decoder::parse_py_object<stim::DetectorErrorModel>(dem), config);
           }),
           py::arg("dem"), py::arg("config"))
      .def(py::init([](py::object dem, const BPParams& params, size_t num_passes) {
             return std::make_unique<MultiPassBpDecoder>(
                 tesseract_decoder::parse_py_object<stim::DetectorErrorModel>(dem), params,
                 num_passes);
           }),
           py::arg("dem"), py::arg("params"), py::arg("num_passes") = 2)
      .def_property_readonly("num_detectors", &MultiPassBpDecoder::total_detectors)
      .def_property_readonly("total_detectors", &MultiPassBpDecoder::total_detectors)
      .def_property_readonly("num_observables", &MultiPassBpDecoder::num_observables)
      .def_property_readonly("total_variables", &MultiPassBpDecoder::total_variables)
      .def(
          "get_component_topology",
          [](const MultiPassBpDecoder& self, size_t comp_idx) {
            return topology_to_pydict(self.get_component_topology(comp_idx));
          },
          py::arg("comp_idx"),
          R"pbdoc(Returns the Tanner graph topology for component comp_idx (0 or 1) as a dict.)pbdoc")
      .def(
          "decode_trace",
          [](MultiPassBpDecoder& self, const py::object& syndrome) {
            std::vector<uint64_t> detections =
                parse_syndrome_or_detections(syndrome, self.total_detectors());
            return multipass_trace_to_pydict(self.decode_trace(detections));
          },
          py::arg("syndrome"),
          R"pbdoc(
            Decodes a single shot with full Multi-Pass / DEM-shattering trace, per-component BP
            iterations, cross-component reweighting events, and post-processor traces.
          )pbdoc")
      .def(
          "decode",
          [](MultiPassBpDecoder& self, const py::array_t<bool>& syndrome) {
            if ((size_t)syndrome.size() != self.total_detectors()) {
              throw std::invalid_argument("Syndrome size does not match decoder detector count.");
            }

            std::vector<uint64_t> detections;
            auto syndrome_unchecked = syndrome.unchecked<1>();
            for (size_t i = 0; i < (size_t)syndrome_unchecked.size(); ++i) {
              if (syndrome_unchecked(i)) {
                detections.push_back(i);
              }
            }

            std::vector<uint8_t> predictions = self.decode(detections);

            size_t num_obs = self.num_observables();
            std::vector<char> result(num_obs, 0);
            for (size_t i = 0; i < predictions.size(); ++i) {
              if (predictions[i]) {
                result[i] = 1;
              }
            }

            return py::array(py::dtype::of<bool>(), result.size(), result.data());
          },
          py::arg("syndrome"),
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(),
          R"pbdoc(
            Decodes a single shot from a dense boolean array using Multi-Pass BP.
            Returns predicted observables.
          )pbdoc");
}

}  // namespace bp

#endif
