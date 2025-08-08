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

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
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
           py::arg("detections"),
           py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
      .def("decode_to_errors",
           py::overload_cast<const std::vector<uint64_t>&, size_t, size_t>(
               &TesseractDecoder::decode_to_errors),
           py::arg("detections"), py::arg("det_order"), py::arg("det_beam"),
           py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
      .def(
          "get_observables_from_errors",
          [](TesseractDecoder& self, const std::vector<size_t>& predicted_errors) {
            std::vector<bool> result(self.num_observables, false);
            for (size_t ei : predicted_errors) {
              for (int obs_index : self.errors[ei].symptom.observables) {
                result[obs_index] = result[obs_index] ^ true;
              }
            }
            return result;
          },
          py::arg("predicted_errors"))
      .def("cost_from_errors", &TesseractDecoder::cost_from_errors, py::arg("predicted_errors"))
      .def(
          "decode_from_detection_events",
          [](TesseractDecoder& self, const std::vector<uint64_t>& detections) {
            std::vector<char> result(self.num_observables, false);
            self.decode(detections);
            for (size_t ei : self.predicted_errors_buffer) {
              for (int obs_index : self.errors[ei].symptom.observables) {
                result[obs_index] = result[obs_index] ^ true;
              }
            }
            return py::array(py::dtype::of<bool>(), result.size(), result.data());
          },
          py::arg("detections"),
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(),
          R"pbdoc(
          Decodes a single shot from a list of detection events.

          Parameters
          ----------
          detections : list[int]
              A list of `uint64_t` indices corresponding to the detectors that were
              fired. This input represents a single measurement shot.

          Returns
          -------
          np.ndarray
              A 1D NumPy array of booleans. Each boolean value indicates whether the
              corresponding logical observable has been flipped by the decoded error.
      )pbdoc")
      .def(
          "decode",
          [](TesseractDecoder& self, const py::array_t<bool>& syndrome) {
            std::vector<uint64_t> detections;
            auto syndrome_unchecked = syndrome.unchecked<1>();
            for (size_t i = 0; i < syndrome_unchecked.size(); ++i) {
              if (syndrome_unchecked(i)) {
                detections.push_back(i);
              }
            }
            self.decode(detections);
            // Note: `std::vector<bool>` is a special C++ template that does not
            // provide a contiguous memory block, which is required by `pybind11`
            // for direct NumPy array creation. Therefore, I use `std::vector<char>`
            // instead to ensure compatibility with `py::array`.
            std::vector<char> result(self.num_observables, 0);
            for (size_t ei : self.predicted_errors_buffer) {
              for (int obs_index : self.errors[ei].symptom.observables) {
                result[obs_index] = result[obs_index] ^ true;
              }
            }
            return py::array(py::dtype::of<bool>(), result.size(), result.data());
          },
          py::arg("syndrome"),
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(),
          R"pbdoc(
        Decodes a single shot.

        Parameters
        ----------
        syndrome : np.ndarray
            A 1D NumPy array of booleans representing the detector outcomes for a single shot.
            The length of the array should match the number of detectors in the DEM.

        Returns
        -------
        np.ndarray
            A 1D NumPy array of booleans indicating which observables are flipped.
            The length of the array matches the number of observables.
    )pbdoc")
      .def(
          "decode_batch",
          [](TesseractDecoder& self, const py::array_t<bool>& syndromes) {
            // Check the dimensions of the `syndromes` argument.
            if (syndromes.ndim() != 2) {
              throw std::runtime_error("Input syndromes must be a 2D NumPy array.");
            }

            // Retrieve the number of shots, detectors and the syndrome patterns.
            auto syndromes_unchecked = syndromes.unchecked<2>();
            size_t num_shots = syndromes_unchecked.shape(0);
            size_t num_detectors = syndromes_unchecked.shape(1);

            // Allocate the result array.
            py::array_t<bool> result({num_shots, self.num_observables});
            auto result_unchecked = result.mutable_unchecked<2>();

            // Process and decode each shot.
            for (size_t i = 0; i < num_shots; ++i) {
              std::vector<uint64_t> detections;
              for (size_t j = 0; j < num_detectors; ++j) {
                if (syndromes_unchecked(i, j)) {
                  detections.push_back(j);
                }
              }
              self.decode(detections);

              // Note: I must do this if I want to modify the results on the 'result_unchecked'
              // itself.
              for (size_t k = 0; k < self.num_observables; ++k) {
                result_unchecked(i, k) = 0;
              }

              // Collect results for the current shot being decoded.
              for (size_t ei : self.predicted_errors_buffer) {
                for (int obs_index : self.errors[ei].symptom.observables) {
                  result_unchecked(i, obs_index) ^= 1;
                }
              }
            }

            return result;
          },
          py::arg("syndromes"),
          R"pbdoc(
        Decodes a batch of shots.

        Parameters
        ----------
        syndromes : np.ndarray
            A 2D NumPy array of booleans where each row corresponds to a shot and
            each column corresponds to a logical observable. Each row is the decoder's prediction of which observables were flipped in the shot. The shape is
            a new array with num_detectors size.

        Returns
        -------
        np.ndarray
            A 2D NumPy array of booleans where each row corresponds to a shot and
            that short specifies which logical observable are flipped. The shape is
            (num_shots, num_observables).
    )pbdoc")
      .def_readwrite("low_confidence_flag", &TesseractDecoder::low_confidence_flag)
      .def_readwrite("predicted_errors_buffer", &TesseractDecoder::predicted_errors_buffer)
      .def_readwrite("errors", &TesseractDecoder::errors);
}

#endif