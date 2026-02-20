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
//
// NOTE: After modifying bindings in this file, regenerate the .pyi type stubs:
//   bazel run //src:generate_stubs -- --output-dir $(pwd)/src

#ifndef _UTILS_PYBIND_H
#define _UTILS_PYBIND_H

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utils.h"

namespace py = pybind11;

void add_utils_module(py::module& root) {
  auto m = root.def_submodule("utils", "utility methods");

  m.attr("EPSILON") = EPSILON;
  m.doc() = "A small floating point number used for comparisons.";

  m.attr("INF") = INF;
  m.doc() = "A representation of infinity for floating point numbers.";

  py::enum_<DetOrder>(m, "DetOrder", "Detector ordering methods")
      .value("DetBFS", DetOrder::DetBFS)
      .value("DetIndex", DetOrder::DetIndex)
      .value("DetCoordinate", DetOrder::DetCoordinate)
      .export_values();

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
            A list where each inner list contains the 3D coordinates
            [x, y, z] of a detector.
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
            Here we say that two detectors are connected if there exists at
            least one error in the DEM which flips both detectors.
    )pbdoc");
  m.def(
      "build_det_orders",
      [](py::object dem, size_t num_det_orders, DetOrder method, uint64_t seed) {
        auto input_dem = parse_py_object<stim::DetectorErrorModel>(dem);
        return build_det_orders(input_dem, num_det_orders, method, seed);
      },
      py::arg("dem"), py::arg("num_det_orders"), py::arg("method") = DetOrder::DetBFS,
      py::arg("seed") = 0, R"pbdoc(
        Generates various detector orderings for decoding.

        Parameters
        ----------
        dem : stim.DetectorErrorModel
            The detector error model to generate orders for.
        num_det_orders : int
            The number of detector orderings to generate.
        method : tesseract_decoder.utils.DetOrder, default=tesseract_decoder.utils.DetOrder.DetBFS
            Strategy for ordering detectors. ``DetBFS`` performs a breadth-first
            traversal, ``DetCoordinate`` uses randomized geometric orientations,
            and ``DetIndex`` chooses either increasing or decreasing detector
            index order at random.
        seed : int, default=0
            A seed for the random number generator.

        Returns
        -------
        list[list[int]]
            A list of detector orderings. Each inner list maps a detector index
            to its position in the ordering.
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
#endif
