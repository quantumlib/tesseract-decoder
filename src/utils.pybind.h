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

#ifndef _UTILS_PYBIND_H
#define _UTILS_PYBIND_H

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utils.h"

namespace py = pybind11;

void add_utils_module(py::module &root) {
  auto m = root.def_submodule("utils", "utility methods");

  m.attr("EPSILON") = EPSILON;
  m.attr("INF") = INF;
  m.def(
      "get_detector_coords",
      [](py::object dem) {
        auto input_dem = parse_py_object<stim::DetectorErrorModel>(dem);
        return get_detector_coords(input_dem);
      },
      py::arg("dem"));
  m.def(
      "build_detector_graph",
      [](py::object dem) {
        auto input_dem = parse_py_object<stim::DetectorErrorModel>(dem);
        return build_detector_graph(input_dem);
      },
      py::arg("dem"));
  m.def(
      "get_errors_from_dem",
      [](py::object dem) {
        auto input_dem = parse_py_object<stim::DetectorErrorModel>(dem);
        return get_errors_from_dem(input_dem);
      },
      py::arg("dem"));

  // Not exposing sampling_from_dem and sample_shots because they depend on
  // stim::SparseShot which stim doesn't expose to python.
}
#endif
