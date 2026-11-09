#ifndef _BP_PYBIND_H
#define _BP_PYBIND_H

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bp/bp_params.h"
#include "bp/multi_pass_bp_decoder.h"
#include "bp/osd_post_processor.h"
#include "bp/tesseract_bp_decoder.h"
#include "stim.h"
#include "stim_utils.pybind.h"

namespace py = pybind11;

namespace bp {

void add_bp_module(py::module& root) {
  auto m = root.def_submodule("bp", "Module containing the Belief Propagation decoder");

  py::class_<BPParams>(m, "BPParams", "Configuration for the Belief Propagation decoder.")
      .def(py::init<>())
      .def_static("Osd", &BPParams::Osd, py::arg("max_iter") = DEFAULT_MAX_ITER,
                  py::arg("osd_order") = 24, py::arg("osd_weight") = 2)
      .def_static("Fast", &BPParams::Fast, py::arg("max_iter") = 10,
                  py::arg("osd_order") = 10, py::arg("osd_weight") = 0)
      .def_readwrite("max_iter", &BPParams::max_iter)
      .def_readwrite("stop_at_convergence", &BPParams::stop_at_convergence)
      .def_readwrite("normalization_factor", &BPParams::normalization_factor)
      .def_readwrite("random_schedule", &BPParams::random_schedule)
      .def_readwrite("random_seed", &BPParams::random_seed)
      .def_readwrite("osd_truncation_factor", &BPParams::osd_truncation_factor)
      .def_readwrite("osd_early_exit", &BPParams::osd_early_exit)
      .def_readwrite("osd_order", &BPParams::osd_order)
      .def_readwrite("osd_weight", &BPParams::osd_weight)
      .def_readwrite("use_soft_reweighting", &BPParams::use_soft_reweighting);

  py::class_<OsdPostProcessor, std::shared_ptr<OsdPostProcessor>>(m, "OsdPostProcessor")
      .def(py::init<const TannerGraph&, size_t, size_t>(), py::arg("graph"),
           py::arg("osd_order") = 0, py::arg("osd_weight") = 0);

  py::class_<TesseractBpDecoder>(m, "TesseractBpDecoder",
                                 "A top-level Belief Propagation orchestrator.")
      .def(py::init([](py::object dem, const BPParams& config) {
             return std::make_unique<TesseractBpDecoder>(
                 tesseract_decoder::parse_py_object<stim::DetectorErrorModel>(dem), config);
           }),
           py::arg("dem"), py::arg("config") = BPParams())
      .def("create_osd_post_processor", &TesseractBpDecoder::create_osd_post_processor,
           py::arg("osd_order") = 0, py::arg("osd_weight") = 0)
      .def("create_post_processor", &TesseractBpDecoder::create_post_processor)
      .def_property_readonly("num_detectors", &TesseractBpDecoder::num_detectors)
      .def_property_readonly("num_variables", &TesseractBpDecoder::num_variables)
      .def_property_readonly("num_observables", &TesseractBpDecoder::num_observables)
      .def(
          "decode",
          [](TesseractBpDecoder& self, const py::array_t<bool>& syndrome,
             const std::shared_ptr<OsdPostProcessor>& post_processor) {
            if (static_cast<size_t>(syndrome.size()) != self.num_detectors()) {
              throw std::invalid_argument("Syndrome size does not match decoder detector count.");
            }
            std::vector<uint64_t> detections;
            auto syndrome_unchecked = syndrome.unchecked<1>();
            for (size_t i = 0; i < static_cast<size_t>(syndrome_unchecked.size()); ++i) {
              if (syndrome_unchecked(i)) detections.push_back(i);
            }
            std::vector<uint8_t> predictions = self.decode(detections, post_processor);
            std::vector<char> result(self.num_observables(), 0);
            for (size_t i = 0; i < predictions.size(); ++i) {
              if (predictions[i]) result[i] = 1;
            }
            return py::array(py::dtype::of<bool>(), result.size(), result.data());
          },
          py::arg("syndrome"), py::arg("post_processor") = nullptr,
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(),
          "Decodes a single shot from a dense boolean array and returns predicted observables.")
      .def(
          "decode_to_errors",
          [](TesseractBpDecoder& self, const py::array_t<bool>& syndrome,
             const std::shared_ptr<OsdPostProcessor>& post_processor) {
            if (static_cast<size_t>(syndrome.size()) != self.num_detectors()) {
              throw std::invalid_argument("Syndrome size does not match decoder detector count.");
            }
            std::vector<std::vector<uint64_t>> batch(1);
            auto syndrome_unchecked = syndrome.unchecked<1>();
            for (size_t i = 0; i < static_cast<size_t>(syndrome_unchecked.size()); ++i) {
              if (syndrome_unchecked(i)) batch[0].push_back(i);
            }
            std::vector<std::vector<size_t>> errors_batch;
            std::vector<uint8_t> low_conf;
            std::vector<std::vector<uint8_t>> preds = self.decode_batch(
                batch, post_processor, /*custom_priors_per_shot=*/nullptr, &errors_batch,
                /*sparse_custom_priors_per_shot=*/nullptr, &low_conf);
            std::vector<char> obs(self.num_observables(), 0);
            if (!preds.empty()) {
              for (size_t i = 0; i < preds[0].size() && i < obs.size(); ++i) {
                if (preds[0][i]) obs[i] = 1;
              }
            }
            std::vector<size_t> errs =
                errors_batch.empty() ? std::vector<size_t>() : errors_batch[0];
            int lc = low_conf.empty() ? 0 : static_cast<int>(low_conf[0]);
            return py::make_tuple(py::array(py::dtype::of<bool>(), obs.size(), obs.data()), errs,
                                  lc);
          },
          py::arg("syndrome"), py::arg("post_processor") = nullptr,
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(),
          "Decodes one shot; returns (predicted observables, predicted variable indices, "
          "low-confidence code: 0 = n/a, 1 = low confidence, 2 = confident).")
      .def(
          "variable_symptom",
          [](const TesseractBpDecoder& self, size_t var_idx) {
            if (var_idx >= self.num_variables()) {
              throw std::out_of_range("variable index out of range");
            }
            auto s = self.variable_symptom(var_idx);
            return py::make_tuple(s.detectors, s.observables);
          },
          py::arg("var_idx"),
          "Returns (sorted detector indices, sorted observable indices) of a Tanner variable.")
      .def("cost_from_errors", &TesseractBpDecoder::cost_from_errors, py::arg("predicted_errors"));

  py::class_<MultiPassBpConfig>(m, "MultiPassBpConfig")
      .def(py::init<BPParams, size_t, std::vector<int>, bool>(),
           py::arg("params") = BPParams(), py::arg("num_passes") = 2,
           py::arg("detector_components") = std::vector<int>(),
           py::arg("soft_reweight") = true)
      .def_readwrite("params", &MultiPassBpConfig::params)
      .def_readwrite("num_passes", &MultiPassBpConfig::num_passes)
      .def_readwrite("detector_components", &MultiPassBpConfig::detector_components)
      .def_readwrite("soft_reweight", &MultiPassBpConfig::soft_reweight);

  py::class_<MultiPassBpDecoder>(m, "MultiPassBpDecoder", "Multi-Pass Belief Propagation decoder.")
      .def(py::init([](py::object dem, const MultiPassBpConfig& config) {
             return std::make_unique<MultiPassBpDecoder>(
                 tesseract_decoder::parse_py_object<stim::DetectorErrorModel>(dem), config);
           }),
           py::arg("dem"), py::arg("config") = MultiPassBpConfig())
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
          "decode",
          [](MultiPassBpDecoder& self, const py::array_t<bool>& syndrome) {
            if (static_cast<size_t>(syndrome.size()) != self.total_detectors()) {
              throw std::invalid_argument("Syndrome size does not match decoder detector count.");
            }
            std::vector<uint64_t> detections;
            auto syndrome_unchecked = syndrome.unchecked<1>();
            for (size_t i = 0; i < static_cast<size_t>(syndrome_unchecked.size()); ++i) {
              if (syndrome_unchecked(i)) detections.push_back(i);
            }
            std::vector<uint8_t> predictions = self.decode(detections);
            std::vector<char> result(self.num_observables(), 0);
            for (size_t i = 0; i < predictions.size(); ++i) {
              if (predictions[i]) result[i] = 1;
            }
            return py::array(py::dtype::of<bool>(), result.size(), result.data());
          },
          py::arg("syndrome"),
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(),
          "Decodes a single shot from a dense boolean array using Multi-Pass BP.")
      .def(
          "decode_batch_info",
          [](MultiPassBpDecoder& self, const py::array_t<bool>& syndromes) {
            if (syndromes.ndim() != 2 ||
                static_cast<size_t>(syndromes.shape(1)) != self.total_detectors()) {
              throw std::invalid_argument("syndromes must have shape (num_shots, num_detectors).");
            }
            auto syn = syndromes.unchecked<2>();
            const size_t num_shots = static_cast<size_t>(syn.shape(0));
            std::vector<std::vector<uint64_t>> batch(num_shots);
            for (size_t s = 0; s < num_shots; ++s) {
              for (size_t d = 0; d < static_cast<size_t>(syn.shape(1)); ++d) {
                if (syn(s, d)) batch[s].push_back(d);
              }
            }
            std::vector<uint8_t> low_conf;
            std::vector<double> costs;
            std::vector<std::vector<uint8_t>> preds = self.decode_batch(batch, &low_conf, &costs);
            const size_t num_obs = self.num_observables();
            py::array_t<uint8_t> out_preds({num_shots, num_obs});
            py::array_t<uint8_t> out_lc(num_shots);
            py::array_t<double> out_costs(num_shots);
            auto p = out_preds.mutable_unchecked<2>();
            auto l = out_lc.mutable_unchecked<1>();
            auto c = out_costs.mutable_unchecked<1>();
            for (size_t s = 0; s < num_shots; ++s) {
              for (size_t o = 0; o < num_obs; ++o) {
                p(s, o) = (s < preds.size() && o < preds[s].size() && preds[s][o]) ? 1 : 0;
              }
              l(s) = s < low_conf.size() ? low_conf[s] : 0;
              c(s) = s < costs.size() ? costs[s] : 0.0;
            }
            return py::make_tuple(out_preds, out_lc, out_costs);
          },
          py::arg("syndromes"),
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(),
          "Decodes a (num_shots, num_detectors) boolean batch; returns (predictions uint8[S, O], "
          "low-confidence reason codes uint8[S] (0 = confident, 1..5 = suspect), stage-1 "
          "costs float64[S]) as used by the bp_main Tesseract-fallback cascade.");
}

}  // namespace bp

#endif
