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

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include <fstream>
#include <iostream>

#include "stim.h"
#include "tesseract.h"

namespace py = pybind11;

// These are the classes that will be exposed to Python.
struct TesseractSinterCompiledDecoder;
struct TesseractSinterDecoder;

//--------------------------------------------------------------------------------------------------
// This struct implements the sinter.CompiledDecoder API. It holds the pre-compiled decoder
// instance and performs the actual decoding on bit-packed NumPy arrays.
//--------------------------------------------------------------------------------------------------
struct TesseractSinterCompiledDecoder {
  // A pointer to the pre-configured TesseractDecoder.
  std::unique_ptr<TesseractDecoder> decoder;
  uint64_t num_detectors;
  uint64_t num_observables;

  // Decode a batch of syndrome shots in a bit-packed NumPy array.
  py::array_t<uint8_t> decode_shots_bit_packed(
      const py::array_t<uint8_t>& bit_packed_detection_event_data) {
    // Validate input.
    if (bit_packed_detection_event_data.ndim() != 2) {
      throw std::invalid_argument("Input `bit_packed_detection_event_data` must be a 2D array.");
    }

    // Calculate number of bytes per shot.
    const uint64_t num_detector_bytes = (num_detectors + 7) / 8;
    if (bit_packed_detection_event_data.shape(1) != (py::ssize_t)num_detector_bytes) {
      throw std::invalid_argument(
          "Input array's second dimension does not match num_detector_bytes.");
    }

    const size_t num_shots = bit_packed_detection_event_data.shape(0);
    const uint64_t num_observable_bytes = (num_observables + 7) / 8;

    // Result buffer to store the predicted observables for all shots.
    auto result_array =
        py::array_t<uint8_t>({(py::ssize_t)num_shots, (py::ssize_t)num_observable_bytes});
    auto result_buffer = result_array.mutable_data();

    const uint8_t* detections_data = bit_packed_detection_event_data.data();
    const size_t detections_stride = bit_packed_detection_event_data.strides(0);

    // Loop through each shot and decode it with TesseractDecoder.
    for (size_t shot = 0; shot < num_shots; ++shot) {
      const uint8_t* single_shot_data = detections_data + shot * detections_stride;

      // Unpack the shot data into a vector of indices of fired detectors.
      std::vector<uint64_t> detections;
      for (uint64_t i = 0; i < num_detectors; ++i) {
        if ((single_shot_data[i / 8] >> (i % 8)) & 1) {
          detections.push_back(i);
        }
      }

      // Decode with TesseractDecoder.
      std::vector<int> predictions = decoder->decode(detections);

      // Store predictions into the output buffer
      uint8_t* single_result_buffer = result_buffer + shot * num_observable_bytes;
      std::fill(single_result_buffer, single_result_buffer + num_observable_bytes, 0);
      for (size_t obs_index : predictions) {
        if (obs_index >= 0 && obs_index < num_observables) {
          single_result_buffer[obs_index / 8] ^= (1 << (obs_index % 8));
        }
      }
    }

    // Return the result.
    return result_array;
  }
};

//--------------------------------------------------------------------------------------------------
// This struct implements the sinter.Decoder API. It is responsible for creating and compiling
// a decoder for a specific Detector Error Model (DEM).
//--------------------------------------------------------------------------------------------------
struct TesseractSinterDecoder {
  // Parameters for TesseractConfig
  int det_beam;
  bool beam_climbing;
  bool no_revisit_dets;
  bool verbose;
  bool merge_errors;
  size_t pqlimit;
  double det_penalty;
  bool create_visualization;

  // Parameters for build_det_orders
  size_t num_det_orders;
  DetOrder det_order_method;
  uint64_t seed;

  // Default constructor
  TesseractSinterDecoder()
      : det_beam(DEFAULT_DET_BEAM),
        beam_climbing(false),
        no_revisit_dets(true),
        verbose(false),
        merge_errors(true),
        pqlimit(DEFAULT_PQLIMIT),
        det_penalty(0.0),
        create_visualization(false),
        num_det_orders(0),
        det_order_method(DetOrder::DetBFS),
        seed(2384753) {}

  // Constructor with parameters
  TesseractSinterDecoder(int det_beam, bool beam_climbing, bool no_revisit_dets, bool verbose,
                         bool merge_errors, size_t pqlimit, double det_penalty,
                         bool create_visualization, size_t num_det_orders,
                         DetOrder det_order_method, uint64_t seed)
      : det_beam(det_beam),
        beam_climbing(beam_climbing),
        no_revisit_dets(no_revisit_dets),
        verbose(verbose),
        merge_errors(merge_errors),
        pqlimit(pqlimit),
        det_penalty(det_penalty),
        create_visualization(create_visualization),
        num_det_orders(num_det_orders),
        det_order_method(det_order_method),
        seed(seed) {}

  bool operator==(const TesseractSinterDecoder& other) const {
    return det_beam == other.det_beam && beam_climbing == other.beam_climbing &&
           no_revisit_dets == other.no_revisit_dets && verbose == other.verbose &&
           merge_errors == other.merge_errors && pqlimit == other.pqlimit &&
           det_penalty == other.det_penalty && create_visualization == other.create_visualization &&
           num_det_orders == other.num_det_orders && det_order_method == other.det_order_method &&
           seed == other.seed;
  }

  bool operator!=(const TesseractSinterDecoder& other) const {
    return !(*this == other);
  }

  // Take a string representation of the DEM, parse the DEM and return a compiled decoder instance.
  TesseractSinterCompiledDecoder compile_decoder_for_dem(const py::object& dem) {
    const stim::DetectorErrorModel stim_dem(py::cast<std::string>(py::str(dem)).c_str());

    std::vector<std::vector<size_t>> det_orders =
        build_det_orders(stim_dem, num_det_orders, det_order_method, seed);

    TesseractConfig local_config = {
        stim_dem,     det_beam, beam_climbing, no_revisit_dets, verbose,
        merge_errors, pqlimit,  det_orders,    det_penalty,     create_visualization};
    auto decoder = std::make_unique<TesseractDecoder>(local_config);

    return TesseractSinterCompiledDecoder{
        .decoder = std::move(decoder),
        .num_detectors = stim_dem.count_detectors(),
        .num_observables = stim_dem.count_observables(),
    };
  }

  // Decode shots while operating on files that store the DEM information.
  void decode_via_files(uint64_t num_shots, uint64_t num_dets, uint64_t num_obs,
                        const py::object& dem_path, const py::object& dets_b8_in_path,
                        const py::object& obs_predictions_b8_out_path, const py::object& tmp_dir) {
    std::string dem_path_str = py::cast<std::string>(py::str(dem_path));
    std::string dets_in_str = py::cast<std::string>(py::str(dets_b8_in_path));
    std::string obs_out_str = py::cast<std::string>(py::str(obs_predictions_b8_out_path));

    // Read the DEM from the file.
    std::ifstream dem_file(dem_path_str);
    std::stringstream dem_content_stream;
    if (!dem_file) {
      throw std::runtime_error("Failed to open DEM file: " + dem_path_str);
    }
    dem_content_stream << dem_file.rdbuf();
    std::string dem_content_str = dem_content_stream.str();
    dem_file.close();

    // Construct TesseractDecoder.
    const stim::DetectorErrorModel stim_dem(dem_content_str.c_str());
    std::vector<std::vector<size_t>> det_orders =
        build_det_orders(stim_dem, num_det_orders, det_order_method, seed);

    TesseractConfig local_config = {
        stim_dem,     det_beam, beam_climbing, no_revisit_dets, verbose,
        merge_errors, pqlimit,  det_orders,    det_penalty,     create_visualization};
    TesseractDecoder decoder(local_config);

    // Calculate expected number of bytes per shot for detectors and observables.
    const uint64_t num_detector_bytes = (num_dets + 7) / 8;
    const uint64_t num_observable_bytes = (num_obs + 7) / 8;

    std::ifstream input_file(dets_in_str, std::ios::binary);
    if (!input_file) {
      throw std::runtime_error("Failed to open input file: " + dets_in_str);
    }
    std::ofstream output_file(obs_out_str, std::ios::binary);
    if (!output_file) {
      throw std::runtime_error("Failed to open output file: " + obs_out_str);
    }

    std::vector<uint8_t> single_shot_data(num_detector_bytes);
    std::vector<uint8_t> single_result_data(num_observable_bytes);

    for (uint64_t shot = 0; shot < num_shots; ++shot) {
      // Read shot's data.
      input_file.read(reinterpret_cast<char*>(single_shot_data.data()), num_detector_bytes);
      if (input_file.gcount() != (std::streamsize)num_detector_bytes) {
        throw std::runtime_error("Failed to read a full shot from the input file.");
      }

      // Extract shot's data and parse into detector indices.
      std::vector<uint64_t> detections;
      for (uint64_t i = 0; i < num_dets; ++i) {
        if ((single_shot_data[i / 8] >> (i % 8)) & 1) {
          detections.push_back(i);
        }
      }

      std::vector<int> predictions = decoder.decode(detections);

      // Pack the predictions back into a bit-packed format.
      std::fill(single_result_data.begin(), single_result_data.end(), 0);
      for (size_t obs_index : predictions) {
        if (obs_index >= 0 && obs_index < num_obs) {
          single_result_data[obs_index / 8] ^= (1 << (obs_index % 8));
        }
      }

      // Write result to the output file.
      output_file.write(reinterpret_cast<char*>(single_result_data.data()), num_observable_bytes);
    }

    input_file.close();
    output_file.close();
  }
};

//--------------------------------------------------------------------------------------------------
// Expose C++ classes to the Python interpreter.
//--------------------------------------------------------------------------------------------------
void pybind_sinter_compat(py::module& root) {
  auto m = root.def_submodule("tesseract_sinter_compat", R"pbdoc(
        This module provides Python bindings for the Tesseract quantum error
        correction decoder, designed for compatibility with the Sinter library.
    )pbdoc");

  // Bind the TesseractSinterCompiledDecoder.
  py::class_<TesseractSinterCompiledDecoder>(m, "TesseractSinterCompiledDecoder", R"pbdoc(
            A Tesseract decoder preconfigured for a specific Detector Error Model.
        )pbdoc")
      .def("decode_shots_bit_packed", &TesseractSinterCompiledDecoder::decode_shots_bit_packed,
           py::kw_only(), py::arg("bit_packed_detection_event_data"),
           R"pbdoc(
                Predicts observable flips from bit-packed detection events.

                This function decodes a batch of `num_shots` syndrome measurements,
                where each shot's detection events are provided in a bit-packed format.

                :param bit_packed_detection_event_data: A 2D numpy array of shape
                    `(num_shots, ceil(num_detectors / 8))`. Each byte contains
                    8 bits of detection event data. A `1` in bit `k` of byte `j`
                    indicates that detector `8j + k` fired.
                :return: A 2D numpy array of shape `(num_shots, ceil(num_observables / 8))`
                    containing the predicted observable flips in a bit-packed format.
            )pbdoc")
      .def_readwrite("num_detectors", &TesseractSinterCompiledDecoder::num_detectors,
                     R"pbdoc(The number of detectors in the decoder's underlying DEM.)pbdoc")
      .def_readwrite(
          "num_observables", &TesseractSinterCompiledDecoder::num_observables,
          R"pbdoc(The number of logical observables in the decoder's underlying DEM.)pbdoc")
      .def_property_readonly(
          "decoder",
          [](const TesseractSinterCompiledDecoder& self) -> const TesseractDecoder& {
            return *self.decoder;
          },
          py::return_value_policy::reference_internal,
          R"pbdoc(The internal TesseractDecoder instance.)pbdoc");

  // Bind the TesseractSinterDecoder.
  py::class_<TesseractSinterDecoder>(m, "TesseractSinterDecoder", R"pbdoc(
            A factory for creating Tesseract decoders compatible with `sinter`.
        )pbdoc")
      .def(py::init<>(), R"pbdoc(
            Initializes a new TesseractSinterDecoder instance with a default TesseractConfig.
          )pbdoc")
      .def(
          py::init<int, bool, bool, bool, bool, size_t, double, bool, size_t, DetOrder, uint64_t>(),
          py::arg("det_beam") = DEFAULT_DET_BEAM, py::arg("beam_climbing") = false,
          py::arg("no_revisit_dets") = true, py::arg("verbose") = false,
          py::arg("merge_errors") = true, py::arg("pqlimit") = DEFAULT_PQLIMIT,
          py::arg("det_penalty") = 0.0, py::arg("create_visualization") = false,
          py::arg("num_det_orders") = 0, py::arg("det_order_method") = DetOrder::DetBFS,
          py::arg("seed") = 2384753,
          R"pbdoc(
            Initializes a new TesseractSinterDecoder instance with custom TesseractConfig parameters.
           )pbdoc")
      .def("compile_decoder_for_dem", &TesseractSinterDecoder::compile_decoder_for_dem,
           py::kw_only(), py::arg("dem"),
           R"pbdoc(
                Creates a Tesseract decoder preconfigured for the given detector error model.

                :param dem: The `stim.DetectorErrorModel` to configure the decoder for.
                :return: A `TesseractSinterCompiledDecoder` instance that can decode
                    bit-packed shots for the given DEM.
            )pbdoc")
      .def("decode_via_files", &TesseractSinterDecoder::decode_via_files, py::kw_only(),
           py::arg("num_shots"), py::arg("num_dets"), py::arg("num_obs"), py::arg("dem_path"),
           py::arg("dets_b8_in_path"), py::arg("obs_predictions_b8_out_path"), py::arg("tmp_dir"),
           R"pbdoc(
                Decodes data from files and writes the result to a file.

                :param num_shots: The number of shots to decode.
                :param num_dets: The number of detectors in the error model.
                :param num_obs: The number of logical observables in the error model.
                :param dem_path: The path to a file containing the `stim.DetectorErrorModel` string.
                :param dets_b8_in_path: The path to a file containing bit-packed detection events.
                :param obs_predictions_b8_out_path: The path to the output file where
                    bit-packed observable predictions will be written.
                :param tmp_dir: A temporary directory path. (Currently unused, but required by API)
            )pbdoc")
      .def_readwrite("det_beam", &TesseractSinterDecoder::det_beam)
      .def_readwrite("beam_climbing", &TesseractSinterDecoder::beam_climbing)
      .def_readwrite("no_revisit_dets", &TesseractSinterDecoder::no_revisit_dets)
      .def_readwrite("verbose", &TesseractSinterDecoder::verbose)
      .def_readwrite("merge_errors", &TesseractSinterDecoder::merge_errors)
      .def_readwrite("pqlimit", &TesseractSinterDecoder::pqlimit)
      .def_readwrite("det_penalty", &TesseractSinterDecoder::det_penalty)
      .def_readwrite("create_visualization", &TesseractSinterDecoder::create_visualization)
      .def_readwrite("num_det_orders", &TesseractSinterDecoder::num_det_orders)
      .def_readwrite("det_order_method", &TesseractSinterDecoder::det_order_method)
      .def_readwrite("seed", &TesseractSinterDecoder::seed)
      .def(py::self == py::self,
           R"pbdoc(Checks if two TesseractSinterDecoder instances are equal.)pbdoc")
      .def(py::self != py::self,
           R"pbdoc(Checks if two TesseractSinterDecoder instances are not equal.)pbdoc")
      .def(py::pickle(
          [](const TesseractSinterDecoder& self) -> py::tuple {  // __getstate__
            return py::make_tuple(self.det_beam, self.beam_climbing, self.no_revisit_dets,
                                  self.verbose, self.merge_errors, self.pqlimit, self.det_penalty,
                                  self.create_visualization, self.num_det_orders,
                                  self.det_order_method, self.seed);
          },
          [](py::tuple t) {  // __setstate__
            if (t.size() != 11) {
              throw std::runtime_error("Invalid state for TesseractSinterDecoder!");
            }
            return TesseractSinterDecoder(
                t[0].cast<int>(), t[1].cast<bool>(), t[2].cast<bool>(), t[3].cast<bool>(),
                t[4].cast<bool>(), t[5].cast<size_t>(), t[6].cast<double>(), t[7].cast<bool>(),
                t[8].cast<size_t>(), t[9].cast<DetOrder>(), t[10].cast<uint64_t>());
          }));

  // Add a function to create a dictionary of custom decoders
  m.def(
      "make_tesseract_sinter_decoders_dict",
      []() -> py::object {
        auto result = py::dict();
        result["tesseract"] = TesseractSinterDecoder{};
        result["tesseract-long-beam"] = TesseractSinterDecoder(
            /*det_beam=*/20, /*beam_climbing=*/true, /*no_revisit_dets=*/true,
            /*verbose=*/false, /*merge_errors=*/true, /*pqlimit=*/1000000,
            /*det_penalty=*/0.0, /*create_visualization=*/false,
            /*num_det_orders=*/21, /*det_order_method=*/DetOrder::DetIndex, /*seed=*/2384753);
        result["tesseract-short-beam"] = TesseractSinterDecoder(
            /*det_beam=*/15, /*beam_climbing=*/true, /*no_revisit_dets=*/true,
            /*verbose=*/false, /*merge_errors=*/true, /*pqlimit=*/200000,
            /*det_penalty=*/0.0, /*create_visualization=*/false,
            /*num_det_orders=*/16, /*det_order_method=*/DetOrder::DetIndex, /*seed=*/2384753);
        return result;
      },
      R"pbdoc(
        Returns a dictionary mapping decoder names to sinter.Decoder-style objects.
        This allows Sinter to easily discover and use Tesseract as a custom decoder.
      )pbdoc");

  // Aliases that are visible from the root module.
  root.attr("TesseractSinterDecoder") = m.attr("TesseractSinterDecoder");
  root.attr("make_tesseract_sinter_decoders_dict") = m.attr("make_tesseract_sinter_decoders_dict");
}
