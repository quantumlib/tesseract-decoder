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
          "Decodes a single shot from a dense boolean array and returns predicted observables.");

  py::class_<MultiPassBpConfig>(m, "MultiPassBpConfig")
      .def(py::init<BPParams, size_t, std::vector<int>, bool>(),
           py::arg("params") = BPParams(), py::arg("num_passes") = 2,
           py::arg("detector_components") = std::vector<int>(),
           py::arg("soft_reweight") = true)
      .def_readwrite("params", &MultiPassBpConfig::params)
      .def_readwrite("num_passes", &MultiPassBpConfig::num_passes)
      .def_readwrite("detector_components", &MultiPassBpConfig::detector_components)
      .def_readwrite("soft_reweight", &MultiPassBpConfig::soft_reweight);

  py::class_<MultiPassBpDecoder>(m, "MultiPassBpDecoder",
                                 "Multi-Pass Belief Propagation decoder.")
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
          "Decodes a single shot from a dense boolean array using Multi-Pass BP.");
}

}  // namespace bp

#endif
