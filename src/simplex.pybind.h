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

#ifndef SIMPLEX_PYBIND_H
#define SIMPLEX_PYBIND_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "simplex.h"
#include "utils.pybind.h"

namespace py = pybind11;

inline void add_simplex_module(py::module& m) {
  py::class_<SimplexDecoder>(m, "SimplexDecoder")
      .def(py::init<SimplexConfig>(), py::arg("config"))
      .def(
          "decode",
          [](SimplexDecoder& self, const py::array_t<bool>& syndrome) {
            if ((size_t)syndrome.size() != self.num_detectors) {
              std::string msg = "Syndrome array size (" + std::to_string(syndrome.size()) +
                                ") does not match the number of detectors in the decoder (" +
                                std::to_string(self.num_detectors) + ").";
              throw std::invalid_argument(msg);
            }

            std::vector<uint64_t> detections;
            auto syndrome_unchecked = syndrome.unchecked<1>();
            for (size_t i = 0; i < (size_t)syndrome_unchecked.size(); ++i) {
              if (syndrome_unchecked(i)) {
                detections.push_back(i);
              }
            }
            return self.decode(detections);
          },
          py::arg("syndrome"), R"pbdoc(
            Decodes the given syndrome and returns the XOR-sum of the flipped observables.
        )pbdoc")
      .def(
          "decode_to_errors",
          [](SimplexDecoder& self, const py::array_t<bool>& syndrome) {
            if ((size_t)syndrome.size() != self.num_detectors) {
              std::string msg = "Syndrome array size (" + std::to_string(syndrome.size()) +
                                ") does not match the number of detectors in the decoder (" +
                                std::to_string(self.num_detectors) + ").";
              throw std::invalid_argument(msg);
            }

            std::vector<uint64_t> detections;
            auto syndrome_unchecked = syndrome.unchecked<1>();
            for (size_t i = 0; i < (size_t)syndrome_unchecked.size(); ++i) {
              if (syndrome_unchecked(i)) {
                detections.push_back(i);
              }
            }
            self.decode_to_errors(detections);
            // Translate internal indices to original for Python return.
            std::vector<size_t> result;
            for (size_t iei : self.predicted_errors_buffer) {
              result.push_back(self.get_original_error_index(iei));
            }
            return result;
          },
          py::arg("syndrome"), R"pbdoc(
            Decodes the given syndrome and returns the predicted errors.
            Indices correspond to the original flattened detector error model.
        )pbdoc")
      .def_property_readonly(
          "predicted_errors_buffer",
          [](const SimplexDecoder& self) {
            std::vector<size_t> result;
            for (size_t iei : self.predicted_errors_buffer) {
              result.push_back(self.get_original_error_index(iei));
            }
            return result;
          },
          "A buffer containing the predicted errors from the most recent decode operation. "
          "The indices correspond to the original flattened DEM.")
      .def_readwrite("errors", &SimplexDecoder::errors,
                     "The list of all errors in the detector error model.")
      .def(
          "cost_from_errors",
          [](const SimplexDecoder& self, const std::vector<size_t>& predicted_errors) {
            std::vector<size_t> internal;
            for (size_t ei : predicted_errors) {
              internal.push_back(ei < self.errors.size() ? ei : self.get_internal_error_index(ei));
            }
            return self.cost_from_errors(internal);
          },
          py::arg("predicted_errors"),
          "Returns the sum of the likelihood costs of the provided errors.")
      .def(
          "get_flipped_observables",
          [](const SimplexDecoder& self, const std::vector<size_t>& predicted_errors) {
            std::vector<size_t> internal;
            for (size_t ei : predicted_errors) {
              internal.push_back(ei < self.errors.size() ? ei : self.get_internal_error_index(ei));
            }
            return self.get_flipped_observables(internal);
          },
          py::arg("predicted_errors"),
          "Returns the bitwise XOR of the observables bitmasks of the provided errors.")
      .def("get_original_error_index", &SimplexDecoder::get_original_error_index,
           py::arg("internal_index"),
           "Returns the index of the provided internal optimized error in the original flattened "
           "DEM.")
      .def("get_internal_error_index", &SimplexDecoder::get_internal_error_index,
           py::arg("original_index"),
           "Returns the internal index of the provided original error if it is the representative, "
           "or size_t max if it was merged or removed.")
      .def_readonly("num_detectors", &SimplexDecoder::num_detectors,
                    "The total number of detectors in the detector error model.")
      .def_readonly("num_observables", &SimplexDecoder::num_observables,
                    "The total number of logical observables in the detector error model.");
}

#endif
