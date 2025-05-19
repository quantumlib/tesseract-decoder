#ifndef TESSERACT_COMMON_PY_H
#define TESSERACT_COMMON_PY_H

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "common.h"

namespace py = pybind11;

void add_common_module(py::module &root)
{
    auto m = root.def_submodule("common", "classes commonly used by the decoder");

    // TODO: add as_dem_instruction_targets
    py::class_<common::Symptom>(m, "Symptom")
        .def(py::init<std::vector<int>, common::ObservablesMask>(), py::arg("detectors") = std::vector<int>(), py::arg("observables") = 0)
        .def_readwrite("detectors", &common::Symptom::detectors)
        .def_readwrite("observables", &common::Symptom::observables)
        .def("__str__", &common::Symptom::str)
        .def(py::self == py::self)
        .def(py::self != py::self);

    // TODO: add constructor with stim::DemInstruction.
    py::class_<common::Error>(m, "Error")
        .def_readwrite("likelihood_cost", &common::Error::likelihood_cost)
        .def_readwrite("probability", &common::Error::probability)
        .def_readwrite("symptom", &common::Error::symptom)
        .def("__str__", &common::Error::str)
        .def(py::init<>())
        .def(py::init<double, std::vector<int> &, common::ObservablesMask, std::vector<bool> &>())
        .def(py::init<double, double, std::vector<int> &, common::ObservablesMask, std::vector<bool> &>());
}

#endif
