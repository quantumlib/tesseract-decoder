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

namespace tesseract_decoder {

void add_utils_module(py::module& root) {
  auto m = root.def_submodule("utils", "utility methods");

  m.attr("EPSILON") = EPSILON;
  m.doc() = "A small floating point number used for comparisons.";

  m.attr("INF") = INF;
  m.doc() = "A representation of infinity for floating point numbers.";

  py::enum_<DetectorOrderMethod> detector_order_method(m, "DetectorOrderMethod",
                                                       "Detector ordering methods");
  detector_order_method.value("BFS", DetectorOrderMethod::BFS)
      .value("Index", DetectorOrderMethod::Index)
      .value("Coordinate", DetectorOrderMethod::Coordinate)
      .value("Literal", DetectorOrderMethod::Literal)
      .value("DetBFS", DetectorOrderMethod::BFS)
      .value("DetIndex", DetectorOrderMethod::Index)
      .value("DetCoordinate", DetectorOrderMethod::Coordinate)
      .export_values();
  m.attr("DetOrder") = detector_order_method;

  m.def(
      "get_detector_coords",
      [](py::object dem) {
        auto input_dem = parse_py_object<stim::DetectorErrorModel>(dem);
        return get_detector_coords(input_dem);
      },
      py::arg("dem"), R"pbdoc(
        Returns the coordinates for each detector in a DetectorErrorModel.

        Parameters
        ----------
        dem : stim.DetectorErrorModel
            The detector error model to extract coordinates from.

        Returns
        -------
        list[list[float]]
            If any detector coordinates are declared, returns one entry per
            detector, indexed by detector ID. Missing coordinates are empty
            lists and coordinate vectors may have any dimensionality. Returns
            an empty list if the model declares no detector coordinates.
    )pbdoc");
  m.def(
      "build_detector_graph",
      [](py::object dem) {
        auto input_dem = parse_py_object<stim::DetectorErrorModel>(dem);
        return build_detector_graph(input_dem);
      },
      py::arg("dem"), R"pbdoc(
        Builds a graph representing the connections between detectors.

        This graph is used by the decoder to find error paths.

        Parameters
        ----------
        dem : stim.DetectorErrorModel
            The detector error model used to build the graph.

        Returns
        -------
        list[list[int]]
            An adjacency list representation of the detector graph.
            Each inner list contains the indices of detectors connected
            to the detector at the corresponding index.
            Each positive-probability error's parity-reduced detector symptom
            induces a clique in the graph.
    )pbdoc");
  m.def(
      "build_det_orders",
      [](py::object dem, size_t num_det_orders, DetectorOrderMethod method, uint64_t seed) {
        auto input_dem = parse_py_object<stim::DetectorErrorModel>(dem);
        return build_det_orders(input_dem, num_det_orders, method, seed);
      },
      py::arg("dem"), py::arg("num_det_orders"), py::arg("method") = DetectorOrderMethod::Index,
      py::arg("seed") = 0, R"pbdoc(
        Generates various detector orderings for decoding.

        Parameters
        ----------
        dem : stim.DetectorErrorModel
            The detector error model to generate orders for.
        num_det_orders : int
            The number of detector orderings to generate.
        method : tesseract_decoder.utils.DetectorOrderMethod, default=tesseract_decoder.utils.DetectorOrderMethod.Index
            Strategy for ordering detectors. ``Index`` chooses either increasing
            or decreasing detector index order at random, ``BFS`` performs a
            breadth-first traversal, and ``Coordinate`` projects every declared
            coordinate dimension onto randomized orientations and places detectors
            without coordinates last.
        seed : int, default=0
            A seed for the random number generator. Exact randomized orders
            are reproducible only with a fixed C++ standard library and
            toolchain.

        Returns
        -------
        list[list[int]]
            A list of detector traversal permutations. Each inner list gives
            detector IDs in traversal order: ``order[position] = detector_id``.
    )pbdoc");
  m.def(
      "get_errors_from_dem",
      [](py::object dem) {
        auto input_dem = parse_py_object<stim::DetectorErrorModel>(dem);
        return get_errors_from_dem(input_dem);
      },
      py::arg("dem"), R"pbdoc(
        Extracts a list of errors from a DetectorErrorModel.

        Parameters
        ----------
        dem : stim.DetectorErrorModel
            The detector error model to extract errors from.

        Returns
        -------
        list[common.Error]
            A list of `common.Error` objects representing all the
            errors defined in the DEM.
    )pbdoc");

  // Not exposing sampling_from_dem and sample_shots because they depend on
  // stim::SparseShot which stim doesn't expose to python.
}

}  // namespace tesseract_decoder

#endif
