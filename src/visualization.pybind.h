#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "visualization.h"

namespace py = pybind11;

void add_visualization_module(py::module& root) {
  auto m = root.def_submodule("viz", "Module containing the visualization tools");
  py::class_<Visualizer>(m, "Visualizer")
      .def(py::init<>())
      .def("write", &Visualizer::write, py::arg("fpath"));
}
