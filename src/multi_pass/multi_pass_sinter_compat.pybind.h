// Copyright 2026 Google LLC
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

#ifndef MULTI_PASS_SINTER_COMPAT_PYBIND_H
#define MULTI_PASS_SINTER_COMPAT_PYBIND_H

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <vector>

#include "multi_pass_tesseract_decoder.h"

namespace py = pybind11;

namespace tesseract_decoder {

struct MultiPassSinterCompiledDecoder {
  std::unique_ptr<MultiPassTesseractDecoder> decoder;
  uint64_t num_detectors;
  uint64_t num_observables;

  size_t num_components() const {
    return decoder->num_components();
  }

  py::array_t<uint8_t> decode_shots_bit_packed(
      const py::array_t<uint8_t>& bit_packed_detection_event_data) {
    if (bit_packed_detection_event_data.ndim() != 2) {
      throw std::invalid_argument("Input `bit_packed_detection_event_data` must be a 2D array.");
    }

    const uint64_t num_detector_bytes = (num_detectors + 7) / 8;
    if (bit_packed_detection_event_data.shape(1) != static_cast<py::ssize_t>(num_detector_bytes)) {
      throw std::invalid_argument(
          "Input array's second dimension does not match num_detector_bytes.");
    }

    const py::ssize_t num_shots = bit_packed_detection_event_data.shape(0);
    const py::ssize_t num_observable_bytes = (num_observables + 7) / 8;

    // Sinter reserves the final byte for the per-shot discard flag.
    py::array_t<uint8_t> result_array({num_shots, num_observable_bytes + 1});
    auto detections = bit_packed_detection_event_data.unchecked<2>();
    auto results = result_array.mutable_unchecked<2>();

    for (py::ssize_t shot = 0; shot < num_shots; ++shot) {
      std::vector<uint64_t> fired_detectors;
      for (uint64_t detector = 0; detector < num_detectors; ++detector) {
        if ((detections(shot, detector / 8) >> (detector % 8)) & 1) {
          fired_detectors.push_back(detector);
        }
      }

      MultiPassDecodeResult decoded = decoder->decode_result(fired_detectors);
      for (py::ssize_t byte = 0; byte <= num_observable_bytes; ++byte) {
        results(shot, byte) = 0;
      }
      for (int observable : decoded.predictions) {
        if (observable >= 0 && static_cast<uint64_t>(observable) < num_observables) {
          results(shot, observable / 8) ^= static_cast<uint8_t>(1U << (observable % 8));
        }
      }
      results(shot, num_observable_bytes) = decoded.low_confidence;
    }
    return result_array;
  }
};

MultiPassSinterCompiledDecoder compile_multi_pass_decoder_for_dem(
    const py::object& dem, const std::vector<int>& detector_components, size_t num_passes,
    const TesseractConfig& base_config, size_t num_det_orders, DetOrder det_order_method,
    uint64_t seed, SchedulingStrategy strategy) {
  stim::DetectorErrorModel stim_dem(py::cast<std::string>(py::str(dem)).c_str());
  auto decoder = std::make_unique<MultiPassTesseractDecoder>(
      stim_dem, num_passes, detector_components, base_config, num_det_orders, det_order_method,
      seed, strategy);
  return MultiPassSinterCompiledDecoder{
      .decoder = std::move(decoder),
      .num_detectors = stim_dem.count_detectors(),
      .num_observables = stim_dem.count_observables(),
  };
}

void pybind_multi_pass_sinter_compat(py::module& m) {
  py::enum_<SchedulingStrategy>(m, "SchedulingStrategy")
      .value("Static", SchedulingStrategy::Static)
      .value("Causal", SchedulingStrategy::Causal)
      .export_values();

  // This type and its factory are implementation details. The supported Sinter
  // decoder is the pure-Python `multi_pass_sinter_decoders.MultiPassSinterDecoder`.
  py::class_<MultiPassSinterCompiledDecoder>(m, "_MultiPassSinterCompiledDecoder")
      .def_property_readonly("num_components", &MultiPassSinterCompiledDecoder::num_components)
      .def("decode_shots_bit_packed", &MultiPassSinterCompiledDecoder::decode_shots_bit_packed,
           py::kw_only(), py::arg("bit_packed_detection_event_data"),
           py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

  m.def("_compile_multi_pass_decoder_for_dem", &compile_multi_pass_decoder_for_dem, py::kw_only(),
        py::arg("dem"), py::arg("detector_components"), py::arg("num_passes"),
        py::arg("base_config"), py::arg("num_det_orders"), py::arg("det_order_method"),
        py::arg("seed"), py::arg("strategy"));
}

}  // namespace tesseract_decoder

#endif  // MULTI_PASS_SINTER_COMPAT_PYBIND_H
