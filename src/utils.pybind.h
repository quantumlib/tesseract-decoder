#pragma once

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utils.h"

namespace py = pybind11;

void add_utils_module(py::module &root)
{
  auto m = root.def_submodule("utils", "utility methods");

  m.attr("EPSILON") = EPSILON;
  m.attr("INF") = INF;
  m.def("get_detector_coords", &get_detector_coords, py::arg("dem"));
  m.def("build_detector_graph", &build_detector_graph, py::arg("dem"));
  m.def("get_errors_from_dem", &get_errors_from_dem, py::arg("dem"));

  // Not exposing sampling_from_dem and sample_shots because they depend on
  // stim::SparseShot which stim doesn't expose to python.
}
