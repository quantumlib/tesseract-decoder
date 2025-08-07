// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef _TESSERACT_PYBIND_H
#define _TESSERACT_PYBIND_H

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "stim_utils.pybind.h"
#include "tesseract.h"

namespace py = pybind11;

namespace {
TesseractConfig tesseract_config_maker(
    py::object dem, int det_beam = INF_DET_BEAM, bool beam_climbing = false,
    bool no_revisit_dets = false, bool at_most_two_errors_per_detector = false,
    bool verbose = false, size_t pqlimit = std::numeric_limits<size_t>::max(),
    std::vector<std::vector<size_t>> det_orders = std::vector<std::vector<size_t>>(),
    double det_penalty = 0.0) {
  stim::DetectorErrorModel input_dem = parse_py_object<stim::DetectorErrorModel>(dem);
  return TesseractConfig({input_dem, det_beam, beam_climbing, no_revisit_dets,
                          at_most_two_errors_per_detector, verbose, pqlimit, det_orders,
                          det_penalty});
}
};  // namespace
void add_tesseract_module(py::module& root) {
  auto m = root.def_submodule("tesseract", "Module containing the tesseract algorithm");

  m.attr("INF_DET_BEAM") = INF_DET_BEAM;
  py::class_<TesseractConfig>(m, "TesseractConfig")
      .def(py::init(&tesseract_config_maker), py::arg("dem"), py::arg("det_beam") = INF_DET_BEAM,
           py::arg("beam_climbing") = false, py::arg("no_revisit_dets") = false,
           py::arg("at_most_two_errors_per_detector") = false, py::arg("verbose") = false,
           py::arg("pqlimit") = std::numeric_limits<size_t>::max(),
             py::arg("det_orders") = std::vector<std::vector<size_t>>(), py::arg("det_penalty") = 0.0)
      .def_property("dem", &dem_getter<TesseractConfig>, &dem_setter<TesseractConfig>)
      .def_readwrite("det_beam", &TesseractConfig::det_beam)
      .def_readwrite("no_revisit_dets", &TesseractConfig::no_revisit_dets)
      .def_readwrite("at_most_two_errors_per_detector",
                     &TesseractConfig::at_most_two_errors_per_detector)
      .def_readwrite("verbose", &TesseractConfig::verbose)
      .def_readwrite("pqlimit", &TesseractConfig::pqlimit)
      .def_readwrite("det_orders", &TesseractConfig::det_orders)
      .def_readwrite("det_penalty", &TesseractConfig::det_penalty)
      .def("__str__", &TesseractConfig::str);

  py::class_<Node>(m, "Node")
      .def(py::init<double, size_t, std::vector<size_t>>(), py::arg("cost") = 0.0,
           py::arg("num_detectors") = 0, py::arg("errors") = std::vector<size_t>())
      .def_readwrite("errors", &Node::errors)
      .def_readwrite("cost", &Node::cost)
      .def_readwrite("num_detectors", &Node::num_detectors)
      .def(py::self > py::self)
      .def("__str__", &Node::str);

  py::class_<TesseractDecoder>(m, "TesseractDecoder")
      .def(py::init<TesseractConfig>(), py::arg("config"))
      .def("decode_to_errors",
           py::overload_cast<const std::vector<uint64_t>&>(&TesseractDecoder::decode_to_errors),
           py::arg("detections"))
      .def("decode_to_errors",
           py::overload_cast<const std::vector<uint64_t>&, size_t, size_t>(
               &TesseractDecoder::decode_to_errors),
           py::arg("detections"), py::arg("det_order"), py::arg("det_beam"))
      .def(
          "get_observables_from_errors",
          [](TesseractDecoder& self, const std::vector<size_t>& predicted_errors) {
            std::vector<bool> result(self.num_observables, false);
            const auto& indices = self.get_flipped_observables(predicted_errors);
            for (int index : indices) {
              result[index] = true;
            }
            return result;
          },
          py::arg("predicted_errors"))
      .def("cost_from_errors", &TesseractDecoder::cost_from_errors, py::arg("predicted_errors"))
      .def(
          "decode",
          [](TesseractDecoder& self, const std::vector<uint64_t>& detections) {
            std::vector<bool> result(self.num_observables, false);
            self.decode(detections);
            for (size_t ei : self.predicted_errors_buffer) {
              for (int obs_index : self.errors[ei].symptom.observables) {
                result[obs_index] = result[obs_index] ^ true;
              }
            }
            return result;
          },
          py::arg("detections"))
      .def_readwrite("low_confidence_flag", &TesseractDecoder::low_confidence_flag)
      .def_readwrite("predicted_errors_buffer", &TesseractDecoder::predicted_errors_buffer)
      .def_readwrite("errors", &TesseractDecoder::errors);
}

#endif