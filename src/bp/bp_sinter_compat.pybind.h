#ifndef _BP_SINTER_COMPAT_PYBIND_H
#define _BP_SINTER_COMPAT_PYBIND_H

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>
#include <iostream>

#include "bp/hard_decision_post_processor.h"
#include "bp/post_processor.h"
#include "bp/tesseract_bp_decoder.h"
#include "stim.h"
#include "stim_utils.pybind.h"

namespace py = pybind11;

namespace bp {

struct TesseractBpSinterCompiledDecoder {
  std::unique_ptr<TesseractBpDecoder> decoder;
  std::shared_ptr<PostProcessor> post_processor;
  uint64_t num_detectors;
  uint64_t num_observables;
  bool use_batched_bp;

  py::array_t<uint8_t> decode_shots_bit_packed(
      const py::array_t<uint8_t>& bit_packed_detection_event_data) {
    if (bit_packed_detection_event_data.ndim() != 2) {
      throw std::invalid_argument("Input must be a 2D array.");
    }

    const uint64_t num_detector_bytes = (num_detectors + 7) / 8;
    if (bit_packed_detection_event_data.shape(1) != (py::ssize_t)num_detector_bytes) {
      throw std::invalid_argument("Input array dimension does not match number of detector bytes.");
    }

    const size_t num_shots = bit_packed_detection_event_data.shape(0);
    const uint64_t num_observable_bytes = (num_observables + 7) / 8;

    auto result_array =
        py::array_t<uint8_t>({(py::ssize_t)num_shots, (py::ssize_t)num_observable_bytes});
    auto result_buffer = result_array.mutable_data();

    const uint8_t* detections_data = bit_packed_detection_event_data.data();
    const size_t detections_stride = bit_packed_detection_event_data.strides(0);

    if (use_batched_bp) {
      std::vector<std::vector<uint64_t>> detection_events_batch;
      for (size_t shot = 0; shot < num_shots; ++shot) {
        const uint8_t* single_shot_data = detections_data + shot * detections_stride;
        std::vector<uint64_t> detections;
        for (uint64_t i = 0; i < num_detectors; ++i) {
          if ((single_shot_data[i / 8] >> (i % 8)) & 1) {
            detections.push_back(i);
          }
        }
        detection_events_batch.push_back(detections);
      }

      std::vector<std::vector<uint8_t>> predictions_batch =
          decoder->decode_batch(detection_events_batch, post_processor);

      for (size_t shot = 0; shot < num_shots; ++shot) {
        uint8_t* single_result_buffer = result_buffer + shot * num_observable_bytes;
        std::fill(single_result_buffer, single_result_buffer + num_observable_bytes, 0);

        for (size_t i = 0; i < num_observables; ++i) {
          if (predictions_batch[shot][i]) {
            single_result_buffer[i / 8] ^= (1 << (i % 8));
          }
        }
      }
    } else {
      for (size_t shot = 0; shot < num_shots; ++shot) {
        const uint8_t* single_shot_data = detections_data + shot * detections_stride;

        std::vector<uint64_t> detections;
        for (uint64_t i = 0; i < num_detectors; ++i) {
          if ((single_shot_data[i / 8] >> (i % 8)) & 1) {
            detections.push_back(i);
          }
        }

        std::vector<uint8_t> predictions = decoder->decode(detections, post_processor);

        uint8_t* single_result_buffer = result_buffer + shot * num_observable_bytes;
        std::fill(single_result_buffer, single_result_buffer + num_observable_bytes, 0);

        for (size_t i = 0; i < num_observables; ++i) {
          if (predictions[i]) {
            single_result_buffer[i / 8] ^= (1 << (i % 8));
          }
        }
      }
    }

    return result_array;
  }
};

// Dummy comment to force rebuild
struct TesseractBpSinterDecoder {
  BPParams params;
  size_t osd_order;
  size_t osd_weight;
  bool use_osd;
  bool use_batched_bp;

  TesseractBpSinterDecoder() : osd_order(0), osd_weight(0), use_osd(false), use_batched_bp(false) {}
  TesseractBpSinterDecoder(const BPParams& p, size_t order = 0, size_t weight = 0, bool osd = false,
                           bool batched = false)
      : params(p), osd_order(order), osd_weight(weight), use_osd(osd), use_batched_bp(batched) {}

  TesseractBpSinterCompiledDecoder compile_decoder_for_dem(const py::object& dem) {
    const stim::DetectorErrorModel stim_dem = parse_py_object<stim::DetectorErrorModel>(dem);
    auto decoder = std::make_unique<TesseractBpDecoder>(stim_dem, params);

    std::shared_ptr<PostProcessor> pp;
    if (use_osd) {
      pp = decoder->create_osd_post_processor(osd_order, osd_weight);
    } else {
      pp = std::make_shared<HardDecisionPostProcessor>();
    }

    return TesseractBpSinterCompiledDecoder{
        .decoder = std::move(decoder),
        .post_processor = pp,
        .num_detectors = stim_dem.count_detectors(),
        .num_observables = stim_dem.count_observables(),
        .use_batched_bp = use_batched_bp,
    };
  }
};

void pybind_bp_sinter_compat(py::module& root) {
  auto m = root.def_submodule("bp_sinter_compat", R"pbdoc(
        Sinter compatibility for the Belief Propagation decoder.
    )pbdoc");

  py::class_<TesseractBpSinterCompiledDecoder>(m, "TesseractBpSinterCompiledDecoder")
      .def("decode_shots_bit_packed", &TesseractBpSinterCompiledDecoder::decode_shots_bit_packed,
           py::kw_only(), py::arg("bit_packed_detection_event_data"));

  py::class_<TesseractBpSinterDecoder>(m, "TesseractBpSinterDecoder")
      .def(py::init<BPParams, size_t, size_t, bool, bool>(), py::arg("params"),
           py::arg("osd_order") = 0, py::arg("osd_weight") = 0, py::arg("use_osd") = false,
           py::arg("use_batched_bp") = false)

      .def("compile_decoder_for_dem", &TesseractBpSinterDecoder::compile_decoder_for_dem,
           py::kw_only(), py::arg("dem"));
}

}  // namespace bp

#endif
