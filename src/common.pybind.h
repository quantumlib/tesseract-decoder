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

#ifndef TESSERACT_COMMON_PY_H
#define TESSERACT_COMMON_PY_H

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

#include "common.h"
#include "stim_utils.pybind.h"

namespace py = pybind11;

void add_common_module(py::module& root) {
  auto m = root.def_submodule("common", "classes commonly used by the decoder");

  py::class_<common::Symptom>(m, "Symptom", R"pbdoc(
        Represents a symptom of an error, which is a list of detectors and a list of observables

        A symptom is defined by a list of detectors that are flipped and a list of
        observables that are flipped.
    )pbdoc")
      .def(py::init<std::vector<int>, std::vector<int>>(),
           py::arg("detectors") = std::vector<int>(), py::arg("observables") = std::vector<int>(),
           R"pbdoc(
            The constructor for the `Symptom` class.

            Parameters
            ----------
            detectors : list[int], default=[]
                The indices of the detectors in this symptom.
            observables : list[int], default=[]
                The indices of the flipped observables.
           )pbdoc")
      .def_readwrite("detectors", &common::Symptom::detectors,
                     "A list of the detector indices that are flipped in this symptom.")
      .def_readwrite("observables", &common::Symptom::observables,
                     "A list of observable indices that are flipped in this symptom.")
      .def("__str__", &common::Symptom::str)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def(
          "as_dem_instruction_targets",
          [](common::Symptom s) {
            std::vector<py::object> ret;
            for (auto& t : s.as_dem_instruction_targets())
              ret.push_back(make_py_object(t, "DemTarget"));
            return ret;
          },
          R"pbdoc(
        Converts the symptom into a list of `stim.DemTarget` objects.
        
        Returns
        -------
        list[stim.DemTarget]
            A list of `stim.DemTarget` objects representing the detectors and observables.
      )pbdoc");

  py::class_<common::Error>(m, "Error", R"pbdoc(
        Represents an error, including its cost, and symptom.

        An error is a physical event (or set of indistinguishable physical events)
        defined by the detectors and observables that it flips in the circuit.
    )pbdoc")
      .def_readwrite("likelihood_cost", &common::Error::likelihood_cost,
                     "The cost of this error (often log((1 - probability) / probability)).")
      .def_readwrite("symptom", &common::Error::symptom, "The symptom associated with this error.")
      .def("__str__", &common::Error::str)
      .def(py::init<>(), R"pbdoc(
        Default constructor for the `Error` class.
      )pbdoc")
      .def(py::init<double, std::vector<int>&, std::vector<int>>(), py::arg("likelihood_cost"),
           py::arg("detectors"), py::arg("observables"), R"pbdoc(
            Constructor for the `Error` class.

            Parameters
            ----------
            likelihood_cost : float
                The cost of this error. 
                This is often `log((1 - probability) / probability)`.
            detectors : list[int]
                A list of indices of the detectors flipped by this error.
            observables : list[int]
                A list of indices of the observables flipped by this error.
           )pbdoc")

      .def(py::init([](py::object edi) {
             std::vector<double> args;
             std::vector<stim::DemTarget> targets;
             auto di = parse_py_dem_instruction(edi, args, targets);
             return new common::Error(di);
           }),
           py::arg("error"), R"pbdoc(
            Constructor that creates an `Error` from a `stim.DemInstruction`.

            Parameters
            ----------
            error : stim.DemInstruction
                The instruction to convert into an `Error` object.
           )pbdoc")
      .def("get_probability", &common::Error::get_probability,
           R"pbdoc(
            Gets the probability associated with the likelihood cost.

            Returns
            -------
            float
                The probability of the error, calculated from the likelihood cost.
           )pbdoc")
      .def("set_with_probability", &common::Error::set_with_probability, py::arg("probability"),
           R"pbdoc(
            Sets the likelihood cost based on a given probability.

            Parameters
            ----------
            probability : float
                The probability to use for setting the likelihood cost.
                Must be between 0 and 1 (exclusive).

            Raises
            ------
            ValueError
                If the provided probability is not between 0 and 1.
           )pbdoc");

  m.def(
      "merge_indistinguishable_errors",
      [](py::object dem) {
        auto input_dem = parse_py_object<stim::DetectorErrorModel>(dem);
        std::vector<size_t> error_index_map;
        auto res = common::merge_indistinguishable_errors(input_dem, error_index_map);
        return make_py_object(res, "DetectorErrorModel");
      },
      py::arg("dem"), R"pbdoc(
        Merges identical errors in a `stim.DetectorErrorModel`.
        
        Errors are identical if they flip the same set of detectors and observables (the same symptom).
        For example, two identical errors with probabilities p1 and p2
        would be merged into a single error with the same symptom,
        but with probability `p1 * (1 - p2) + p2 * (1 - p1)`

        Parameters
        ----------
        dem : stim.DetectorErrorModel
            The detector error model to process.

        Returns
        -------
        stim.DetectorErrorModel
            A new `DetectorErrorModel` with identical errors merged.
      )pbdoc");
  m.def(
      "remove_zero_probability_errors",
      [](py::object dem) {
        return make_py_object(([&]() {
                                std::vector<size_t> error_index_map;
                                return common::remove_zero_probability_errors(
                                    parse_py_object<stim::DetectorErrorModel>(dem),
                                    error_index_map);
                              })(),
                              "DetectorErrorModel");
      },
      py::arg("dem"), R"pbdoc(
        Removes errors with a probability of 0 from a `stim.DetectorErrorModel`.

        Parameters
        ----------
        dem : stim.DetectorErrorModel
            The detector error model to process.

        Returns
        -------
        stim.DetectorErrorModel
            A new `DetectorErrorModel` with zero-probability errors removed.
      )pbdoc");
  m.def(
      "dem_from_counts",
      [](py::object orig_dem, const std::vector<size_t> error_counts, size_t num_shots) {
        auto dem = parse_py_object<stim::DetectorErrorModel>(orig_dem);
        return make_py_object(common::dem_from_counts(dem, error_counts, num_shots),
                              "DetectorErrorModel");
      },
      py::arg("orig_dem"), py::arg("error_counts"), py::arg("num_shots"), R"pbdoc(
        Re-weights errors in a `stim.DetectorErrorModel` based on observed counts.

        This function re-calculates the probability of each error based on a list of
        observed counts and the total number of shots.

        Parameters
        ----------
        orig_dem : stim.DetectorErrorModel
            The original detector error model.
        error_counts : list[int]
            A list of counts for each error in the DEM.
        num_shots : int
            The total number of shots in the experiment.

        Returns
        -------
        stim.DetectorErrorModel
            A new `DetectorErrorModel` with updated error probabilities.
      )pbdoc");
}

#endif
