#ifndef _BP_SINTER_COMPAT_PYBIND_H
#define _BP_SINTER_COMPAT_PYBIND_H

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bp/multi_pass_bp_decoder.h"
#include "bp/osd_post_processor.h"
#include "bp/tesseract_bp_decoder.h"
#include "stim.h"
#include "stim_utils.pybind.h"

namespace py = pybind11;

namespace bp {

struct TesseractBpSinterCompiledDecoder {
  std::unique_ptr<TesseractBpDecoder> decoder;
  std::shared_ptr<OsdPostProcessor> post_processor;
  std::unique_ptr<MultiPassBpDecoder> multipass_decoder;
  uint64_t num_detectors;
  uint64_t num_observables;
  bool use_batched_bp;
  bool use_multipass = false;
  size_t num_passes = 2;

  py::array_t<uint8_t> decode_shots_bit_packed(
      const py::array_t<uint8_t>& bit_packed_detection_event_data) {
    if (bit_packed_detection_event_data.ndim() != 2) {
      throw std::invalid_argument("Input must be a 2D array.");
    }

    const uint64_t num_detector_bytes = (num_detectors + 7) / 8;
    if (bit_packed_detection_event_data.shape(1) != static_cast<py::ssize_t>(num_detector_bytes)) {
      throw std::invalid_argument("Input array dimension does not match number of detector bytes.");
    }

    const size_t num_shots = bit_packed_detection_event_data.shape(0);
    const uint64_t num_observable_bytes = (num_observables + 7) / 8;

    auto result_array = py::array_t<uint8_t>(
        {static_cast<py::ssize_t>(num_shots), static_cast<py::ssize_t>(num_observable_bytes)});
    auto result_buffer = result_array.mutable_data();

    const uint8_t* detections_data = bit_packed_detection_event_data.data();
    const size_t detections_stride = bit_packed_detection_event_data.strides(0);

    std::vector<std::vector<uint64_t>> detection_events_batch;
    detection_events_batch.reserve(num_shots);
    for (size_t shot = 0; shot < num_shots; ++shot) {
      const uint8_t* single_shot_data = detections_data + shot * detections_stride;
      std::vector<uint64_t> detections;
      for (uint64_t i = 0; i < num_detectors; ++i) {
        if ((single_shot_data[i / 8] >> (i % 8)) & 1) {
          detections.push_back(i);
        }
      }
      detection_events_batch.push_back(std::move(detections));
    }

    std::vector<std::vector<uint8_t>> predictions_batch =
        (use_multipass && multipass_decoder)
            ? multipass_decoder->decode_batch(detection_events_batch)
            : decoder->decode_batch(detection_events_batch, post_processor);

    for (size_t shot = 0; shot < num_shots; ++shot) {
      uint8_t* single_result_buffer = result_buffer + shot * num_observable_bytes;
      std::fill(single_result_buffer, single_result_buffer + num_observable_bytes, 0);
      for (size_t i = 0; i < num_observables; ++i) {
        if (predictions_batch[shot][i]) {
          single_result_buffer[i / 8] ^= (1 << (i % 8));
        }
      }
    }

    return result_array;
  }
};

struct TesseractBpSinterDecoder {
  BPParams params;
  size_t osd_order = 0;
  size_t osd_weight = 0;
  bool use_osd = false;
  bool use_batched_bp = true;
  bool use_multipass = false;
  size_t num_passes = 2;

  TesseractBpSinterDecoder() = default;
  TesseractBpSinterDecoder(const BPParams& p, size_t order = 0, size_t weight = 0,
                           bool osd = false, bool batched = true, bool multipass = false,
                           size_t passes = 2)
      : params(p),
        osd_order(order),
        osd_weight(weight),
        use_osd(osd),
        use_batched_bp(batched),
        use_multipass(multipass),
        num_passes(passes) {}

  static TesseractBpSinterDecoder BpOsd(size_t max_iter = DEFAULT_MAX_ITER) {
    return TesseractBpSinterDecoder(BPParams::Osd(max_iter), 24, 2, true, true, false, 2);
  }
  static TesseractBpSinterDecoder BpOsdFast(size_t max_iter = 10) {
    return TesseractBpSinterDecoder(BPParams::Fast(max_iter), 10, 0, true, true, false, 2);
  }
  static TesseractBpSinterDecoder MultiPassBpOsd(size_t max_iter = DEFAULT_MAX_ITER) {
    return TesseractBpSinterDecoder(BPParams::Osd(max_iter), 24, 2, true, true, true, 2);
  }
  static TesseractBpSinterDecoder MultiPassBpOsdFast(size_t max_iter = 10) {
    return TesseractBpSinterDecoder(BPParams::Fast(max_iter), 10, 0, true, true, true, 2);
  }

  TesseractBpSinterCompiledDecoder compile_decoder_for_dem(const py::object& dem) {
    const stim::DetectorErrorModel stim_dem =
        tesseract_decoder::parse_py_object<stim::DetectorErrorModel>(dem);

    if (use_multipass) {
      BPParams mp_params = params;
      if (use_osd || osd_order > 0 || osd_weight > 0) {
        mp_params.osd_order = static_cast<int>(osd_order);
        mp_params.osd_weight = static_cast<int>(osd_weight);
      }
      MultiPassBpConfig mp_config{mp_params, num_passes, {}, mp_params.use_soft_reweighting};
      auto mp_dec = std::make_unique<MultiPassBpDecoder>(stim_dem, mp_config);
      return TesseractBpSinterCompiledDecoder{
          .decoder = nullptr,
          .post_processor = nullptr,
          .multipass_decoder = std::move(mp_dec),
          .num_detectors = stim_dem.count_detectors(),
          .num_observables = stim_dem.count_observables(),
          .use_batched_bp = use_batched_bp,
          .use_multipass = true,
          .num_passes = num_passes,
      };
    }

    auto decoder = std::make_unique<TesseractBpDecoder>(stim_dem, params);

    std::shared_ptr<OsdPostProcessor> pp;
    if (use_osd || osd_order > 0 || osd_weight > 0) {
      pp = decoder->create_osd_post_processor(osd_order, osd_weight);
    } else if (params.osd_order >= 0) {
      pp = decoder->create_osd_post_processor(static_cast<size_t>(params.osd_order),
                                              static_cast<size_t>(std::max(0, params.osd_weight)));
    }

    return TesseractBpSinterCompiledDecoder{
        .decoder = std::move(decoder),
        .post_processor = pp,
        .multipass_decoder = nullptr,
        .num_detectors = stim_dem.count_detectors(),
        .num_observables = stim_dem.count_observables(),
        .use_batched_bp = use_batched_bp,
        .use_multipass = false,
        .num_passes = num_passes,
    };
  }
};

void pybind_bp_sinter_compat(py::module& root) {
  auto m = root.def_submodule("bp_sinter_compat",
                              "Sinter compatibility for the Belief Propagation decoder.");

  py::class_<TesseractBpSinterCompiledDecoder>(m, "TesseractBpSinterCompiledDecoder")
      .def_readwrite("use_multipass", &TesseractBpSinterCompiledDecoder::use_multipass)
      .def_readwrite("num_passes", &TesseractBpSinterCompiledDecoder::num_passes)
      .def("decode_shots_bit_packed", &TesseractBpSinterCompiledDecoder::decode_shots_bit_packed,
           py::kw_only(), py::arg("bit_packed_detection_event_data"));

  py::class_<TesseractBpSinterDecoder>(m, "TesseractBpSinterDecoder")
      .def(py::init<BPParams, size_t, size_t, bool, bool, bool, size_t>(),
           py::arg("params") = BPParams(), py::arg("osd_order") = 0, py::arg("osd_weight") = 0,
           py::arg("use_osd") = false, py::arg("use_batched_bp") = true,
           py::arg("use_multipass") = false, py::arg("num_passes") = 2)
      .def_static("BpOsd", &TesseractBpSinterDecoder::BpOsd,
                  py::arg("max_iter") = DEFAULT_MAX_ITER)
      .def_static("BpOsdFast", &TesseractBpSinterDecoder::BpOsdFast,
                  py::arg("max_iter") = 10)
      .def_static("MultiPassBpOsd", &TesseractBpSinterDecoder::MultiPassBpOsd,
                  py::arg("max_iter") = DEFAULT_MAX_ITER)
      .def_static("MultiPassBpOsdFast", &TesseractBpSinterDecoder::MultiPassBpOsdFast,
                  py::arg("max_iter") = 10)
      .def_readwrite("params", &TesseractBpSinterDecoder::params)
      .def_readwrite("osd_order", &TesseractBpSinterDecoder::osd_order)
      .def_readwrite("osd_weight", &TesseractBpSinterDecoder::osd_weight)
      .def_readwrite("use_osd", &TesseractBpSinterDecoder::use_osd)
      .def_readwrite("use_batched_bp", &TesseractBpSinterDecoder::use_batched_bp)
      .def_readwrite("use_multipass", &TesseractBpSinterDecoder::use_multipass)
      .def_readwrite("num_passes", &TesseractBpSinterDecoder::num_passes)
      .def("compile_decoder_for_dem", &TesseractBpSinterDecoder::compile_decoder_for_dem,
           py::kw_only(), py::arg("dem"));
}

}  // namespace bp

#endif
