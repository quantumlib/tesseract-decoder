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

#ifndef TESSERACT_DECODER_SINTER_COMPAT_PYBIND_H
#define TESSERACT_DECODER_SINTER_COMPAT_PYBIND_H

#include <pybind11/numpy.h>

#include <cstdint>
#include <vector>

#include "sinter_compat.h"

namespace py = pybind11;

namespace tesseract_decoder {

inline py::array_t<uint8_t> decode_sinter_shots_bit_packed(
    Decoder& decoder, uint64_t num_detectors, uint64_t num_observables,
    const py::array_t<uint8_t>& bit_packed_detection_event_data, SinterOutputFormat output_format) {
  if (bit_packed_detection_event_data.ndim() != 2) {
    throw std::invalid_argument("Input `bit_packed_detection_event_data` must be a 2D array.");
  }

  const uint64_t num_detector_bytes = (num_detectors + 7) / 8;
  if (bit_packed_detection_event_data.shape(1) != static_cast<py::ssize_t>(num_detector_bytes)) {
    throw std::invalid_argument(
        "Input array's second dimension does not match num_detector_bytes.");
  }

  const py::ssize_t num_shots = bit_packed_detection_event_data.shape(0);
  const py::ssize_t num_result_bytes = sinter_output_bytes(num_observables, output_format);
  py::array_t<uint8_t> result_array({num_shots, num_result_bytes});
  auto detections = bit_packed_detection_event_data.unchecked<2>();
  auto results = result_array.mutable_unchecked<2>();
  std::vector<uint8_t> packed_result(num_result_bytes);

  for (py::ssize_t shot = 0; shot < num_shots; ++shot) {
    std::vector<uint64_t> fired_detectors;
    for (uint64_t detector = 0; detector < num_detectors; ++detector) {
      if ((detections(shot, detector / 8) >> (detector % 8)) & 1) {
        fired_detectors.push_back(detector);
      }
    }

    pack_sinter_decode_result(decoder.decode_result(fired_detectors), num_observables,
                              output_format, packed_result);
    for (py::ssize_t byte = 0; byte < num_result_bytes; ++byte) {
      results(shot, byte) = packed_result[byte];
    }
  }
  return result_array;
}

}  // namespace tesseract_decoder

#endif  // TESSERACT_DECODER_SINTER_COMPAT_PYBIND_H
