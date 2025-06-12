#pragma once

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tesseract.h"

namespace py = pybind11;

void add_tesseract_module(py::module &root)
{
  auto m = root.def_submodule("tesseract",
                              "Module containing the tesseract algorithm");

  m.attr("INF_DET_BEAM") = INF_DET_BEAM;
  py::class_<TesseractConfig>(m, "TesseractConfig")
      .def(py::init<stim::DetectorErrorModel, int, bool, bool, bool, bool,
                    size_t, std::vector<std::vector<size_t>>, double>(),
           py::arg("dem"), py::arg("det_beam") = INF_DET_BEAM,
           py::arg("beam_climbing") = false, py::arg("no_revisit_dets") = false,
           py::arg("at_most_two_errors_per_detector") = false,
           py::arg("verbose") = false,
           py::arg("pqlimit") = std::numeric_limits<size_t>::max(),
           py::arg("det_orders") = std::vector<std::vector<size_t>>(),
           py::arg("det_penalty") = 0.0)
      .def_readwrite("dem", &TesseractConfig::dem)
      .def_readwrite("det_beam", &TesseractConfig::det_beam)
      .def_readwrite("no_revisit_dets", &TesseractConfig::no_revisit_dets)
      .def_readwrite("at_most_two_errors_per_detector",
                     &TesseractConfig::at_most_two_errors_per_detector)
      .def_readwrite("verbose", &TesseractConfig::verbose)
      .def_readwrite("pqlimit", &TesseractConfig::pqlimit)
      .def_readwrite("det_orders", &TesseractConfig::det_orders)
      .def_readwrite("det_penalty", &TesseractConfig::det_penalty)
      .def("__str__", &TesseractConfig::str);

  py::class_<Node>(m, "Node")
      .def(py::init<std::vector<size_t>, std::vector<char>, double, size_t, std::vector<char>>(),
           py::arg("errs") = std::vector<size_t>(),
           py::arg("dets") = std::vector<char>(),
           py::arg("cost") = 0.0,
           py::arg("num_dets") = 0,
           py::arg("blocked_errs") = std::vector<char>())
      .def_readwrite("errs", &Node::errs)
      .def_readwrite("dets", &Node::dets)
      .def_readwrite("cost", &Node::cost)
      .def_readwrite("num_dets", &Node::num_dets)
      .def_readwrite("blocked_errs", &Node::blocked_errs)
      .def(py::self > py::self)
      .def("__str__", &Node::str);

  py::class_<QNode>(m, "QNode")
      .def(py::init<double, size_t, std::vector<size_t>>(), py::arg("cost") = 0.0, py::arg("num_dets") = 0, py::arg("errs") = std::vector<size_t>())
      .def_readwrite("cost", &QNode::cost)
      .def_readwrite("num_dets", &QNode::num_dets)
      .def_readwrite("errs", &QNode::errs)
      .def(py::self > py::self)
      .def("__str__", &QNode::str);

  py::class_<TesseractDecoder>(m, "TesseractDecoder")
        .def(py::init<TesseractConfig>(), py::arg("config"))
        .def("decode_to_errors", py::overload_cast<const std::vector<uint64_t>&>(&TesseractDecoder::decode_to_errors), py::arg("detections"))
        .def("decode_to_errors", py::overload_cast<const std::vector<uint64_t>& , size_t>(&TesseractDecoder::decode_to_errors), py::arg("detections"), py::arg("det_order"))
        .def("mask_from_errors", &TesseractDecoder::mask_from_errors, py::arg("predicted_errors"))
        .def("cost_from_errors", &TesseractDecoder::cost_from_errors, py::arg("predicted_errors"))
        .def("decode", &TesseractDecoder::decode, py::arg("detections"))
        .def_readwrite("low_confidence_flag", &TesseractDecoder::low_confidence_flag)
        .def_readwrite("predicted_errors_buffer", &TesseractDecoder::predicted_errors_buffer)
        .def_readwrite("det_beam", &TesseractDecoder::det_beam)
        .def_readwrite("errors", &TesseractDecoder::errors);
}
