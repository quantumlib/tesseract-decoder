#pragma once

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "common.h"
#include "simplex.h"

namespace py = pybind11;

void add_simplex_module(py::module &root)
{
    auto m = root.def_submodule(
        "simplex", "Module containing the SimplexDecoder and related methods");

    py::class_<SimplexConfig>(m, "SimplexConfig")
        .def(py::init<stim::DetectorErrorModel, bool, size_t, size_t, bool>(),
             py::arg("dem"), py::arg("parallelize") = false,
             py::arg("window_length") = 0, py::arg("window_slide_length") = 0,
             py::arg("verbose") = false)
        .def_readwrite("dem", &SimplexConfig::dem)
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
        .def_readwrite("predicted_errors_buffer",
                       &SimplexDecoder::predicted_errors_buffer)
        .def_readwrite("error_masks", &SimplexDecoder::error_masks)
        .def_readwrite("start_time_to_errors",
                       &SimplexDecoder::start_time_to_errors)
        .def_readwrite("end_time_to_errors", &SimplexDecoder::end_time_to_errors)
        .def_readonly("low_confidence_flag", &SimplexDecoder::low_confidence_flag)
        .def("init_ilp", &SimplexDecoder::init_ilp)
        .def("decode_to_errors", &SimplexDecoder::decode_to_errors, py::arg("detections"))
        .def("mask_from_errors", &SimplexDecoder::mask_from_errors, py::arg("predicted_errors"))
        .def("cost_from_errors", &SimplexDecoder::cost_from_errors, py::arg("predicted_errors"))
        .def("decode", &SimplexDecoder::decode, py::arg("detections"));
}
