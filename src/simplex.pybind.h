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

#include <sstream>

#include "common.h"
#include "simplex.h"
#include "stim_utils.pybind.h"

namespace py = pybind11;

namespace {
// Helper function to compile the decoder.
std::unique_ptr<SimplexDecoder> _compile_simplex_decoder_helper(const SimplexConfig& self) {
  return std::make_unique<SimplexDecoder>(self);
}

SimplexConfig simplex_config_maker(py::object dem, bool parallelize = false,
                                   size_t window_length = 0, size_t window_slide_length = 0,
                                   bool verbose = false, bool merge_errors = true) {
  stim::DetectorErrorModel input_dem = parse_py_object<stim::DetectorErrorModel>(dem);
  return SimplexConfig(
      {input_dem, parallelize, window_length, window_slide_length, verbose, merge_errors});
}

};  // namespace

void add_simplex_module(py::module& root) {
  auto m =
      root.def_submodule("simplex", "Module containing the SimplexDecoder and related methods");

  py::class_<SimplexConfig>(m, "SimplexConfig", R"pbdoc(
        Configuration object for the `SimplexDecoder`.

        This class holds all the parameters needed to initialize and configure a
        Simplex decoder instance, including the detector error model and
        decoding options.
    )pbdoc")
      .def(py::init(&simplex_config_maker), py::arg("dem"), py::arg("parallelize") = false,
           py::arg("window_length") = 0, py::arg("window_slide_length") = 0,
           py::arg("verbose") = false, py::arg("merge_errors") = true, R"pbdoc(
            The constructor for the `SimplexConfig` class.

            Parameters
            ----------
            dem : stim.DetectorErrorModel
                The detector error model to be decoded.
            parallelize : bool, default=False
                Whether to use multithreading for decoding.
            window_length : int, default=0
                The length of the time window for decoding. A value of 0 disables windowing.
            window_slide_length : int, default=0
                The number of time steps to slide the window after each decode. A value of 0
                disables windowing.
            verbose : bool, default=False
                If True, enables verbose logging from the decoder.
            merge_errors : bool, default=True
                If True, merges error channels that have identical syndrome patterns.
           )pbdoc")
      .def_property("dem", &dem_getter<SimplexConfig>, &dem_setter<SimplexConfig>,
                    "The `stim.DetectorErrorModel` that defines the error channels and detectors.")
      .def_readwrite("parallelize", &SimplexConfig::parallelize,
                     "If True, enables multithreaded decoding.")
      .def_readwrite("window_length", &SimplexConfig::window_length,
                     "The number of time steps in each decoding window.")
      .def_readwrite("window_slide_length", &SimplexConfig::window_slide_length,
                     "The number of time steps the window slides after each decode.")
      .def_readwrite("verbose", &SimplexConfig::verbose,
                     "If True, the decoder will print verbose output.")
      .def_readwrite("merge_errors", &SimplexConfig::merge_errors,
                     "If True, identical error mechanisms will be merged.")
      .def("windowing_enabled", &SimplexConfig::windowing_enabled,
           "Returns True if windowing is enabled (i.e., `window_length > 0`).")
      .def("__str__", &SimplexConfig::str)
      .def("compile_decoder", &_compile_simplex_decoder_helper,
           py::return_value_policy::take_ownership, R"pbdoc(
            Compiles the configuration into a new SimplexDecoder instance.

            Returns
            -------
            SimplexDecoder
                A new SimplexDecoder instance configured with the current
                settings.
           )pbdoc");

  py::class_<SimplexDecoder>(m, "SimplexDecoder", R"pbdoc(
        A class that implements the Simplex decoding algorithm.

        It can decode syndromes from a `stim.DetectorErrorModel` to predict
        which observables have been flipped.
    )pbdoc")
      .def(py::init<SimplexConfig>(), py::arg("config"), R"pbdoc(
        The constructor for the `SimplexDecoder` class.

        Parameters
        ----------
        config : SimplexConfig
            The configuration object for the decoder.
      )pbdoc")
      .def_readwrite("config", &SimplexDecoder::config,
                     "The configuration used to create this decoder.")
      .def_readwrite("errors", &SimplexDecoder::errors,
                     "The list of all errors in the detector error model.")
      .def_readwrite("num_detectors", &SimplexDecoder::num_detectors,
                     "The total number of detectors in the detector error model.")
      .def_readwrite("num_observables", &SimplexDecoder::num_observables,
                     "The total number of logical observables in the detector error model.")
      .def_readwrite(
          "predicted_errors_buffer", &SimplexDecoder::predicted_errors_buffer,
          "A buffer containing the predicted errors from the most recent decode operation.")
      .def_readwrite("error_masks", &SimplexDecoder::error_masks,
                     "The list of error masks used for decoding.")
      .def_readwrite(
          "start_time_to_errors", &SimplexDecoder::start_time_to_errors,
          "A map from a detector's start time to the errors that are correlated with it.")
      .def_readwrite("end_time_to_errors", &SimplexDecoder::end_time_to_errors,
                     "A map from a detector's end time to the errors that are correlated with it.")
      .def_readonly("low_confidence_flag", &SimplexDecoder::low_confidence_flag,
                    "A flag indicating if the decoder's prediction has low confidence.")
      .def("init_ilp", &SimplexDecoder::init_ilp, R"pbdoc(
        Initializes the Integer Linear Programming (ILP) solver.

        This method must be called before decoding.
      )pbdoc")
      .def(
          "decode_to_errors",
          [](SimplexDecoder& self, const py::array_t<bool>& syndrome) {
            if ((size_t)syndrome.size() != self.num_detectors) {
              std::ostringstream msg;
              msg << "Syndrome array size (" << syndrome.size()
                  << ") does not match the number of detectors in the decoder ("
                  << self.num_detectors << ").";
              throw std::invalid_argument(msg.str());
            }

            std::vector<uint64_t> detections;
            auto syndrome_unchecked = syndrome.unchecked<1>();
            for (size_t i = 0; i < (size_t)syndrome_unchecked.size(); ++i) {
              if (syndrome_unchecked(i)) {
                detections.push_back(i);
              }
            }
            self.decode_to_errors(detections);
            return self.predicted_errors_buffer;
          },
          py::arg("syndrome"),
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(),
          R"pbdoc(
            Decodes a single shot to a list of error indices.

            Parameters
            ----------
            syndrome : np.ndarray
                A 1D NumPy array of booleans representing the detector outcomes for a single shot.
                The length of the array should match the number of detectors in the DEM.

            Returns
            -------
            list[int]
                A list of predicted error indices.
          )pbdoc")
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
          py::arg("predicted_errors"), R"pbdoc(
            Converts a list of predicted error indices into a list of
            flipped logical observables.

            Parameters
            ----------
            predicted_errors : list[int]
                A list of integers representing the predicted error indices.

            Returns
            -------
            list[bool]
                A list of booleans, where each boolean corresponds to a
                logical observable and is `True` if the observable was flipped.
           )pbdoc")
      .def("cost_from_errors", &SimplexDecoder::cost_from_errors, py::arg("predicted_errors"),
           R"pbdoc(
            Calculates the total logarithmic probability cost for a given set of
            predicted errors. The cost is a measure of how likely a set of errors is.

            Parameters
            ----------
            predicted_errors : list[int]
                A list of integers representing the predicted error indices.

            Returns
            -------
            float
                A float representing the total logarithmic probability cost.
           )pbdoc")
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
              A list of indices corresponding to the detectors that were
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
            if ((size_t)syndrome.size() != self.num_detectors) {
              std::ostringstream msg;
              msg << "Syndrome array size (" << syndrome.size()
                  << ") does not match the number of detectors in the decoder ("
                  << self.num_detectors << ").";
              throw std::invalid_argument(msg.str());
            }

            std::vector<uint64_t> detections;
            auto syndrome_unchecked = syndrome.unchecked<1>();
            for (size_t i = 0; i < (size_t)syndrome_unchecked.size(); i++) {
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

            if (num_detectors != self.num_detectors) {
              std::ostringstream msg;
              msg << "The number of detectors in the input array (" << num_detectors
                  << ") does not match the number of detectors in the decoder ("
                  << self.num_detectors << ").";
              throw std::invalid_argument(msg.str());
            }

            // Allocate the result array.
            py::array_t<bool> result({num_shots, self.num_observables});
            result.attr("fill")(0);
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
            A 2D NumPy array of booleans where each row represents a single shot's
            detector outcomes. The shape should be (num_shots, num_detectors): each shot has
            a new array with num_detectors size.

        Returns
        -------
        np.ndarray
            A 2D NumPy array of booleans where each row corresponds to a shot and
            each column corresponds to a logical observable. Each row is the decoder's prediction of which observables were flipped in the shot. The shape is
            (num_shots, num_observables).
    )pbdoc");
}
#endif
