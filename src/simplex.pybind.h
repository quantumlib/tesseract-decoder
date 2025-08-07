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

#ifndef _SIMPLEX_PYBIND_H
#define _SIMPLEX_PYBIND_H

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "common.h"
#include "simplex.h"
#include "stim_utils.pybind.h"

namespace py = pybind11;

namespace {
SimplexConfig simplex_config_maker(py::object dem, bool parallelize = false,
                                   size_t window_length = 0, size_t window_slide_length = 0,
                                   bool verbose = false) {
  stim::DetectorErrorModel input_dem = parse_py_object<stim::DetectorErrorModel>(dem);
  return SimplexConfig({input_dem, parallelize, window_length, window_slide_length, verbose});
}

};  // namespace

void add_simplex_module(py::module& root) {
  auto m =
      root.def_submodule("simplex", "Module containing the SimplexDecoder and related methods");

  py::class_<SimplexConfig>(m, "SimplexConfig")
      .def(py::init(&simplex_config_maker), py::arg("dem"), py::arg("parallelize") = false,
           py::arg("window_length") = 0, py::arg("window_slide_length") = 0,
           py::arg("verbose") = false)
      .def_property("dem", &dem_getter<SimplexConfig>, &dem_setter<SimplexConfig>)
      .def_readwrite("parallelize", &SimplexConfig::parallelize)
      .def_readwrite("window_length", &SimplexConfig::window_length)
      .def_readwrite("window_slide_length", &SimplexConfig::window_slide_length)
      .def_readwrite("verbose", &SimplexConfig::verbose)
      .def("windowing_enabled", &SimplexConfig::windowing_enabled)
      .def("__str__", &SimplexConfig::str);

  py::class_<SimplexDecoder>(m, "SimplexDecoder")
      .def(py::init<SimplexConfig>(), py::arg("config"))
      .def_readwrite("config", &SimplexDecoder::config)
      .def_readwrite("errors", &SimplexDecoder::errors)
      .def_readwrite("num_detectors", &SimplexDecoder::num_detectors)
      .def_readwrite("num_observables", &SimplexDecoder::num_observables)
      .def_readwrite("predicted_errors_buffer", &SimplexDecoder::predicted_errors_buffer)
      .def_readwrite("error_masks", &SimplexDecoder::error_masks)
      .def_readwrite("start_time_to_errors", &SimplexDecoder::start_time_to_errors)
      .def_readwrite("end_time_to_errors", &SimplexDecoder::end_time_to_errors)
      .def_readonly("low_confidence_flag", &SimplexDecoder::low_confidence_flag)
      .def("init_ilp", &SimplexDecoder::init_ilp)
      .def("decode_to_errors", &SimplexDecoder::decode_to_errors, py::arg("detections"),
           py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
      .def(
          "get_observables_from_errors",
          [](SimplexDecoder& self, const std::vector<size_t>& predicted_errors) {
            std::vector<bool> result(self.num_observables, false);
            for (size_t ei : predicted_errors) {
              for (int obs_index : self.errors[ei].symptom.observables) {
                result[obs_index] = result[obs_index] ^ true;
              }
            }
            return result;
          },
          py::arg("predicted_errors"))
      .def("cost_from_errors", &SimplexDecoder::cost_from_errors, py::arg("predicted_errors"))
      .def(
          "decode_from_detection_events",
          [](SimplexDecoder& self, const std::vector<uint64_t>& detections) {
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
          [](SimplexDecoder& self, const py::array_t<bool>& syndrome) {
            std::vector<uint64_t> detections;
            auto syndrome_unchecked = syndrome.unchecked<1>();
            for (size_t i = 0; i < syndrome_unchecked.size(); i++) {
              if (syndrome_unchecked(i)) {
                detections.push_back(i);
              }
            }
            self.decode(detections);
            // Note: `std::vector<bool>` is a special C++ template that does not
            // provide a contiguous memory block, which is required by `pybind11`
            // for direct NumPy array creation. Therefore, I use `std::vector<char>`
            // to ensure compatibility with `py::array`.
            std::vector<char> result(self.num_observables, 0);
            for (size_t ei : self.predicted_errors_buffer) {
              for (int obs_index : self.errors[ei].symptom.observables) {
                result[obs_index] = result[obs_index] ^ 1;
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
            A 1D NumPy array of booleans representing the detection events for a single shot.
            The length of the array should match the number of detectors in the DEM.

        Returns
        -------
        np.ndarray
            A 1D NumPy array of booleans indicating which observables are flipped.
            The length of the array matches the number of observables.
    )pbdoc")
      .def(
          "decode_batch",
          [](SimplexDecoder& self, const py::array_t<bool>& syndromes) {
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

              // Collect results for the current shot being decoded.
              std::vector<char> shot_result(self.num_observables, 0);
              for (size_t ei : self.predicted_errors_buffer) {
                for (int obs_index : self.errors[ei].symptom.observables) {
                  shot_result[obs_index] ^= 1;
                }
              }

              // Copy the result into the pre-allocated array.
              for (size_t k = 0; k < self.num_observables; ++k) {
                result_unchecked(i, k) = shot_result[k];
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
            A 2D NumPy array of booleans where each row represents a single shot's
            detector outcomes. The shape should be (num_shots, num_detectors): each shot has
            a new array with num_detectors size.

        Returns
        -------
        np.ndarray
            A 2D NumPy array of booleans where each row corresponds to a shot and
            that short specifies which logical observable are flipped. The shape is
            (num_shots, num_observables).
    )pbdoc");
}
#endif
