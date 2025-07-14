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

namespace py = pybind11;

void add_common_module(py::module &root) {
  auto m = root.def_submodule("common", "classes commonly used by the decoder");

  py::class_<common::Symptom>(m, "Symptom")
      .def(py::init<std::vector<int>, common::ObservablesMask>(),
           py::arg("detectors") = std::vector<int>(), py::arg("observables") = 0)
      .def_readwrite("detectors", &common::Symptom::detectors)
      .def_readwrite("observables", &common::Symptom::observables)
      .def("__str__", &common::Symptom::str)
      .def(py::self == py::self)
      .def(py::self != py::self);
     //  .def("as_dem_instruction_targets", [](common::Symptom s) {
     //    std::vector<stim_pybind::ExposedDemTarget> ret;
     //    for (auto &t : s.as_dem_instruction_targets()) ret.emplace_back(t);
     //    return ret;
     //  });

  py::class_<common::Error>(m, "Error")
      .def_readwrite("likelihood_cost", &common::Error::likelihood_cost)
      .def_readwrite("probability", &common::Error::probability)
      .def_readwrite("symptom", &common::Error::symptom)
      .def("__str__", &common::Error::str)
      .def(py::init<>())
      .def(py::init<double, std::vector<int> &, common::ObservablesMask, std::vector<bool> &>(),
           py::arg("likelihood_cost"), py::arg("detectors"), py::arg("observables"),
           py::arg("dets_array"))
      .def(py::init<double, double, std::vector<int> &, common::ObservablesMask,
                    std::vector<bool> &>(),
           py::arg("likelihood_cost"), py::arg("probability"), py::arg("detectors"),
           py::arg("observables"), py::arg("dets_array"));
     //  .def(py::init([](stim_pybind::ExposedDemInstruction edi) {
     //         return new common::Error(edi.as_dem_instruction());
     //       }),
     //       py::arg("error"));

  m.def("merge_identical_errors", &common::merge_identical_errors, py::arg("dem"));
  m.def("remove_zero_probability_errors", &common::remove_zero_probability_errors, py::arg("dem"));
  m.def("dem_from_counts", &common::dem_from_counts, py::arg("orig_dem"), py::arg("error_counts"),
        py::arg("num_shots"));
}

#endif
