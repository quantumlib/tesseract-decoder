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

#ifndef _TESSERACT_PYBIND_H
#define _TESSERACT_PYBIND_H

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "stim_utils.pybind.h"
#include "tesseract.h"

namespace py = pybind11;

namespace {
// Helper function to compile the decoder.
std::unique_ptr<TesseractDecoder> _compile_tesseract_decoder_helper(const TesseractConfig& self) {
  return std::make_unique<TesseractDecoder>(self);
}

TesseractConfig tesseract_config_maker_no_dem(
    int det_beam = INF_DET_BEAM, bool beam_climbing = false, bool no_revisit_dets = false,
    bool at_most_two_errors_per_detector = false, bool verbose = false, bool merge_errors = true,
    size_t pqlimit = std::numeric_limits<size_t>::max(),
    std::vector<std::vector<size_t>> det_orders = std::vector<std::vector<size_t>>(),
    double det_penalty = 0.0, bool create_visualization = false) {
  stim::DetectorErrorModel empty_dem;
  return TesseractConfig({empty_dem, det_beam, beam_climbing, no_revisit_dets,
                          at_most_two_errors_per_detector, verbose, merge_errors, pqlimit,
                          det_orders, det_penalty, create_visualization});
}

TesseractConfig tesseract_config_maker(
    py::object dem, int det_beam = INF_DET_BEAM, bool beam_climbing = false,
    bool no_revisit_dets = false, bool at_most_two_errors_per_detector = false,
    bool verbose = false, bool merge_errors = true,
    size_t pqlimit = std::numeric_limits<size_t>::max(),
    std::vector<std::vector<size_t>> det_orders = std::vector<std::vector<size_t>>(),
    double det_penalty = 0.0, bool create_visualization = false) {
  stim::DetectorErrorModel input_dem = parse_py_object<stim::DetectorErrorModel>(dem);
  return TesseractConfig({input_dem, det_beam, beam_climbing, no_revisit_dets,
                          at_most_two_errors_per_detector, verbose, merge_errors, pqlimit,
                          det_orders, det_penalty, create_visualization});
}
};  // namespace
void add_tesseract_module(py::module& root) {
  auto m = root.def_submodule("tesseract", "Module containing the tesseract algorithm");

  m.attr("INF_DET_BEAM") = INF_DET_BEAM;
  m.doc() = "A sentinel value indicating an infinite beam size for the decoder.";

  py::class_<TesseractConfig>(m, "TesseractConfig", R"pbdoc(
        Configuration object for the `TesseractDecoder`.

        This class holds all the parameters needed to initialize and configure a
        Tesseract decoder instance.
    )pbdoc")
      .def(py::init<>(), R"pbdoc(
        Default constructor for TesseractConfig.
        Creates a new instance with default parameter values.
    )pbdoc")
      .def(py::init(&tesseract_config_maker_no_dem), py::arg("det_beam") = INF_DET_BEAM,
           py::arg("beam_climbing") = false, py::arg("no_revisit_dets") = false,
           py::arg("at_most_two_errors_per_detector") = false, py::arg("verbose") = false,
           py::arg("merge_errors") = true, py::arg("pqlimit") = std::numeric_limits<size_t>::max(),
           py::arg("det_orders") = std::vector<std::vector<size_t>>(), py::arg("det_penalty") = 0.0,
           py::arg("create_visualization") = false,
           R"pbdoc(
             The constructor for the `TesseractConfig` class without a `dem` argument.
             This creates an empty `DetectorErrorModel` by default.

             Parameters
             ----------
             det_beam : int, default=INF_DET_BEAM
                 Beam cutoff that specifies the maximum number of detection events a search state can have.
             beam_climbing : bool, default=False
                 If True, enables a beam climbing heuristic.
             no_revisit_dets : bool, default=False
                 If True, prevents the decoder from revisiting a syndrome pattern more than once.
             at_most_two_errors_per_detector : bool, default=False
                 If True, an optimization is enabled that assumes at most two errors
                 are correlated with each detector.
             verbose : bool, default=False
                 If True, enables verbose logging from the decoder.
              merge_errors : bool, default=True
                 If True, merges error channels that have identical syndrome patterns.
              pqlimit : int, default=max_size_t
                 The maximum size of the priority queue.
              det_orders : list[list[int]], default=empty
                 A list of detector orderings to use for decoding. If empty, the decoder
                 will generate its own orderings.
              det_penalty : float, default=0.0
                 A penalty value added to the cost of each detector visited.
              create_visualization: bool, defualt=False
                 Whether to record the information needed to create a visualization or not.
             )pbdoc")
      .def(py::init(&tesseract_config_maker), py::arg("dem"), py::arg("det_beam") = INF_DET_BEAM,
           py::arg("beam_climbing") = false, py::arg("no_revisit_dets") = false,
           py::arg("at_most_two_errors_per_detector") = false, py::arg("verbose") = false,
           py::arg("merge_errors") = true, py::arg("pqlimit") = std::numeric_limits<size_t>::max(),
           py::arg("det_orders") = std::vector<std::vector<size_t>>(), py::arg("det_penalty") = 0.0,
           py::arg("create_visualization") = false,
           R"pbdoc(
            The constructor for the `TesseractConfig` class.

            Parameters
            ----------
            dem : stim.DetectorErrorModel
                The detector error model to be decoded.
            det_beam : int, default=INF_DET_BEAM
                Beam cutoff that specifies the maximum number of detection events a search state can have.
            beam_climbing : bool, default=False
                If True, enables a beam climbing heuristic.
            no_revisit_dets : bool, default=False
                If True, prevents the decoder from revisiting a syndrome pattern more than once.
            at_most_two_errors_per_detector : bool, default=False
                If True, an optimization is enabled that assumes at most two errors
                are correlated with each detector.
            verbose : bool, default=False
                If True, enables verbose logging from the decoder.
             merge_errors : bool, default=True
                If True, merges error channels that have identical syndrome patterns.
            pqlimit : int, default=max_size_t
                The maximum size of the priority queue.
            det_orders : list[list[int]], default=empty
                A list of detector orderings to use for decoding. If empty, the decoder
                will generate its own orderings.
            det_penalty : float, default=0.0
                A penalty value added to the cost of each detector visited.
            create_visualization: bool, defualt=False
                Whether to record the information needed to create a visualization or not.
           )pbdoc")
      .def_property("dem", &dem_getter<TesseractConfig>, &dem_setter<TesseractConfig>,
                    "The `stim.DetectorErrorModel` that defines the error channels and detectors.")
      .def_readwrite("det_beam", &TesseractConfig::det_beam,
                     "Beam cutoff argument for the beam search.")
      .def_readwrite("beam_climbing", &TesseractConfig::beam_climbing,
                     "Whether to use a beam climbing heuristic.")
      .def_readwrite("no_revisit_dets", &TesseractConfig::no_revisit_dets,
                     "Whether to prevent revisiting same syndrome patterns during decoding.")
      .def_readwrite("at_most_two_errors_per_detector",
                     &TesseractConfig::at_most_two_errors_per_detector,
                     "Whether to assume at most two errors per detector for optimization.")
      .def_readwrite("verbose", &TesseractConfig::verbose,
                     "If True, the decoder will print verbose output.")
      .def_readwrite("merge_errors", &TesseractConfig::merge_errors,
                     "If True, merges error channels that have identical syndrome patterns.")
      .def_readwrite("pqlimit", &TesseractConfig::pqlimit,
                     "The maximum size of the priority queue.")
      .def_readwrite("det_orders", &TesseractConfig::det_orders,
                     "A list of pre-specified detector orderings.")
      .def_readwrite("det_penalty", &TesseractConfig::det_penalty,
                     "The penalty cost added for each detector.")
      .def_readwrite("create_visualization", &TesseractConfig::create_visualization,
                     "If True, records necessary information to create visualization.")
      .def("__str__", &TesseractConfig::str)
      .def("compile_decoder", &_compile_tesseract_decoder_helper,
           py::return_value_policy::take_ownership,
           R"pbdoc(
          Compiles the configuration into a new `TesseractDecoder` instance.

          Returns
          -------
          TesseractDecoder
              A new `TesseractDecoder` instance configured with the current
              settings.
      )pbdoc")
      .def(
          "compile_decoder_for_dem",
          [](TesseractConfig& self, py::object dem) {
            self.dem = parse_py_object<stim::DetectorErrorModel>(dem);
            return std::make_unique<TesseractDecoder>(self);
          },
          py::arg("dem"), py::return_value_policy::take_ownership, R"pbdoc(
            Compiles the configuration into a new `TesseractDecoder` instance
            for a given `dem` object.

            Parameters
            ----------
            dem : stim.DetectorErrorModel
                The detector error model to use for the decoder.

            Returns
            -------
            TesseractDecoder
                A new `TesseractDecoder` instance configured with the
                provided `dem` and the other settings from this
                `TesseractConfig` object.
            )pbdoc");

  py::class_<Node>(m, "Node", R"pbdoc(
        A class representing a node in the Tesseract search graph.

        This is used internally by the decoder to track decoding progress.
    )pbdoc")
      .def(py::init<double, size_t, std::vector<size_t>>(), py::arg("cost") = 0.0,
           py::arg("num_detectors") = 0, py::arg("errors") = std::vector<size_t>(), R"pbdoc(
            The constructor for the `Node` class.

            Parameters
            ----------
            cost : float, default=0.0
                The cost of the path to this node.
            num_detectors : int, default=0
                The number of detectors this search node has.
            errors : list[int], default=empty
                The list of error indices this search node has.
           )pbdoc")
      .def_readwrite("cost", &Node::cost, "The cost of the node.")
      .def_readwrite("num_detectors", &Node::num_detectors,
                     "The number of detectors this search node has.")
      .def_readwrite("errors", &Node::errors, "The list of error indices this search node has.")
      .def(py::self > py::self,
           "Comparison operator for nodes based on cost. This is necessary to prioritize "
           "lower-cost nodes during the search.")
      .def("__str__", &Node::str);

  py::class_<TesseractDecoder>(m, "TesseractDecoder", R"pbdoc(
        A class that implements the Tesseract decoding algorithm.

        It can decode syndromes from a `stim.DetectorErrorModel` to predict
        which observables have been flipped.
    )pbdoc")
      .def(py::init<TesseractConfig>(), py::arg("config"), R"pbdoc(
        The constructor for the `TesseractDecoder` class.

        Parameters
        ----------
        config : TesseractConfig
            The configuration object for the decoder.
      )pbdoc")
      .def("decode_to_errors",
           py::overload_cast<const std::vector<uint64_t>&>(&TesseractDecoder::decode_to_errors),
           py::arg("detections"),
           py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(), R"pbdoc(
            Decodes a single shot to a list of error indices.

            Parameters
            ----------
            detections : list[int]
                A list of indices of the detectors that have fired.

            Returns
            -------
            list[int]
                A list of predicted error indices.
           )pbdoc")
      .def("decode_to_errors",
           py::overload_cast<const std::vector<uint64_t>&, size_t, size_t>(
               &TesseractDecoder::decode_to_errors),
           py::arg("detections"), py::arg("det_order"), py::arg("det_beam"),
           py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(), R"pbdoc(
            Decodes a single shot using a specific detector ordering and beam size.

            Parameters
            ----------
            detections : list[int]
                A list of indices of the detectors that have fired.
            det_order : int
                The index of the detector ordering to use.
            det_beam : int
                The beam size to use during the decoding.

            Returns
            -------
            list[int]
                A list of predicted error indices.
           )pbdoc")
      .def(
          "get_observables_from_errors",
          [](TesseractDecoder& self, const std::vector<size_t>& predicted_errors) {
            std::vector<bool> result(self.num_observables, false);
            for (size_t ei : predicted_errors) {
              for (int obs_index : self.errors[ei].symptom.observables) {
                result[obs_index] = result[obs_index] ^ true;
              }
            }
            return result;
          },
          py::arg("predicted_errors"), R"pbdoc(
            Converts a list of predicted error indices into a list of
            flipped logical observables.

            Parameters
            ----------
            predicted_errors : list[int]
                A list of integers representing the predicted error indices.

            Returns
            -------
            list[bool]
                A list of booleans, where each boolean corresponds to a
                logical observable and is `True` if the observable was flipped.
           )pbdoc")
      .def("cost_from_errors", &TesseractDecoder::cost_from_errors, py::arg("predicted_errors"),
           R"pbdoc(
            Calculates the sum of the likelihood costs of the predicted errors.
            The likelihood cost of an error with probability p is log((1 - p) / p).

            Parameters
            ----------
            predicted_errors : list[int]
                A list of integers representing the predicted error indices.

            Returns
            -------
            float
                A float representing the sum of the likelihood costs of the
                predicted errors.
           )pbdoc")
      .def(
          "decode_from_detection_events",
          [](TesseractDecoder& self, const std::vector<uint64_t>& detections) {
            std::vector<char> result(self.num_observables, false);
            self.decode(detections);
            for (size_t ei : self.predicted_errors_buffer) {
              for (int obs_index : self.errors[ei].symptom.observables) {
                result[obs_index] = result[obs_index] ^ true;
              }
            }
            return py::array(py::dtype::of<bool>(), result.size(), result.data());
          },
          py::arg("detections"),
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(),
          R"pbdoc(
          Decodes a single shot from a list of detection events.

          Parameters
          ----------
          detections : list[int]
              A list of indices corresponding to the detectors that were
              fired. This input represents a single measurement shot.

          Returns
          -------
          np.ndarray
              A 1D NumPy array of booleans. Each boolean value indicates whether the
              decoder predicts that the corresponding logical observable has been flipped.
      )pbdoc")
      .def(
          "decode",
          [](TesseractDecoder& self, const py::array_t<bool>& syndrome) {
            if ((size_t)syndrome.size() != self.num_detectors) {
              std::ostringstream msg;
              msg << "Syndrome array size (" << syndrome.size()
                  << ") does not match the number of detectors in the decoder ("
                  << self.num_detectors << ").";
              throw std::invalid_argument(msg.str());
            }

            std::vector<uint64_t> detections;
            auto syndrome_unchecked = syndrome.unchecked<1>();
            for (size_t i = 0; i < (size_t)syndrome_unchecked.size(); ++i) {
              if (syndrome_unchecked(i)) {
                detections.push_back(i);
              }
            }
            self.decode(detections);
            // Note: `std::vector<bool>` is a special C++ template that does not
            // provide a contiguous memory block, which is required by `pybind11`
            // for direct NumPy array creation. Therefore, I use `std::vector<char>`
            // instead to ensure compatibility with `py::array`.
            std::vector<char> result(self.num_observables, 0);
            for (size_t ei : self.predicted_errors_buffer) {
              for (int obs_index : self.errors[ei].symptom.observables) {
                result[obs_index] = result[obs_index] ^ true;
              }
            }
            return py::array(py::dtype::of<bool>(), result.size(), result.data());
          },
          py::arg("syndrome"),
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(),
          R"pbdoc(
        Decodes a single shot.

        Parameters
        ----------
        syndrome : np.ndarray
            A 1D NumPy array of booleans representing the detector outcomes for a single shot.
            The length of the array should match the number of detectors in the DEM.

        Returns
        -------
        np.ndarray
            A 1D NumPy array of booleans indicating which observables are flipped.
            The length of the array matches the number of observables.
    )pbdoc")
      .def(
          "decode_batch",
          [](TesseractDecoder& self, const py::array_t<bool>& syndromes) {
            // Check the dimensions of the `syndromes` argument.
            if (syndromes.ndim() != 2) {
              throw std::runtime_error("Input syndromes must be a 2D NumPy array.");
            }

            // Retrieve the number of shots, detectors and the syndrome patterns.
            auto syndromes_unchecked = syndromes.unchecked<2>();
            size_t num_shots = syndromes_unchecked.shape(0);
            size_t num_detectors = syndromes_unchecked.shape(1);

            if (num_detectors != self.num_detectors) {
              std::ostringstream msg;
              msg << "The number of detectors in the input array (" << num_detectors
                  << ") does not match the number of detectors in the decoder ("
                  << self.num_detectors << ").";
              throw std::invalid_argument(msg.str());
            }

            // Allocate the result array.
            py::array_t<bool> result({num_shots, self.num_observables});
            result.attr("fill")(0);
            auto result_unchecked = result.mutable_unchecked<2>();

            // Process and decode each shot.
            for (size_t i = 0; i < num_shots; ++i) {
              std::vector<uint64_t> detections;
              for (size_t j = 0; j < num_detectors; ++j) {
                if (syndromes_unchecked(i, j)) {
                  detections.push_back(j);
                }
              }
              self.decode(detections);

              // Collect results for the current shot being decoded.
              for (size_t ei : self.predicted_errors_buffer) {
                for (int obs_index : self.errors[ei].symptom.observables) {
                  result_unchecked(i, obs_index) ^= 1;
                }
              }
            }

            return result;
          },
          py::arg("syndromes"),
          R"pbdoc(
        Decodes a batch of shots.

        Parameters
        ----------
        syndromes : np.ndarray
            A 2D NumPy array of booleans where each row corresponds to a shot and
            each column corresponds to a logical observable. Each row is the decoder's prediction of which observables were flipped in the shot. The shape is
            a new array with num_detectors size.

        Returns
        -------
        np.ndarray
            A 2D NumPy array of booleans where each row corresponds to a shot and
            that short specifies which logical observable are flipped. The shape is
            (num_shots, num_observables).
    )pbdoc")
      .def_readwrite("config", &TesseractDecoder::config,
                     "The configuration used to create this decoder.")
      .def_readwrite("low_confidence_flag", &TesseractDecoder::low_confidence_flag,
                     "A flag indicating if the decoder's prediction has low confidence.")
      .def_readwrite(
          "predicted_errors_buffer", &TesseractDecoder::predicted_errors_buffer,
          "A buffer containing the predicted errors from the most recent decode operation.")
      .def_readwrite("errors", &TesseractDecoder::errors,
                     "The list of all errors in the detector error model.")
      .def_readwrite("num_observables", &TesseractDecoder::num_observables,
                     "The total number of logical observables in the detector error model.")
      .def_readwrite("num_detectors", &TesseractDecoder::num_detectors,
                     "The total number of detectors in the detector error model.")
      .def_readonly("visualizer", &TesseractDecoder::visualizer,
                    "An object that can (if config.create_visualization=True) be used to generate "
                    "visualization of the algorithm");
}

#endif
