#ifndef MULTI_PASS_SINTER_COMPAT_PYBIND_H
#define MULTI_PASS_SINTER_COMPAT_PYBIND_H

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <iostream>

#include "multi_pass_tesseract_decoder.h"
#include "dem_decomposition.h"
#include "utils.h"

namespace py = pybind11;

namespace tesseract {

struct MultiPassSinterCompiledDecoder {
    std::unique_ptr<tesseract::MultiPassTesseractDecoder> decoder;
    uint64_t num_detectors;
    uint64_t num_observables;

    MultiPassSinterCompiledDecoder(std::unique_ptr<tesseract::MultiPassTesseractDecoder> d, uint64_t nd, uint64_t no)
        : decoder(std::move(d)), num_detectors(nd), num_observables(no) {}

    size_t num_components() const { return decoder->num_components(); }

    py::array_t<uint8_t> decode_shots_bit_packed(const py::array_t<uint8_t>& bit_packed_detection_event_data) {
        if (bit_packed_detection_event_data.ndim() != 2) throw std::invalid_argument("Input must be 2D.");
        const uint64_t num_detector_bytes = (num_detectors + 7) / 8;
        if (bit_packed_detection_event_data.shape(1) != (py::ssize_t)num_detector_bytes) throw std::invalid_argument("Wrong shape.");

        const size_t num_shots = bit_packed_detection_event_data.shape(0);
        const uint64_t num_observable_bytes = (num_observables + 7) / 8;

        auto result_array = py::array_t<uint8_t>({(py::ssize_t)num_shots, (py::ssize_t)num_observable_bytes});
        auto result_buffer = result_array.mutable_data();

        const uint8_t* detections_data = bit_packed_detection_event_data.data();
        const size_t detections_stride = bit_packed_detection_event_data.strides(0);

        for (size_t shot = 0; shot < num_shots; ++shot) {
            const uint8_t* single_shot_data = detections_data + shot * detections_stride;
            std::vector<uint64_t> detections;
            for (uint64_t i = 0; i < num_detectors; ++i) {
                if ((single_shot_data[i / 8] >> (i % 8)) & 1) detections.push_back(i);
            }

            std::vector<int> predictions = decoder->decode(detections);
            uint8_t* single_result_buffer = result_buffer + shot * num_observable_bytes;
            std::fill(single_result_buffer, single_result_buffer + num_observable_bytes, 0);
            for (int obs_index : predictions) {
                if (obs_index >= 0 && (uint64_t)obs_index < num_observables) {
                    single_result_buffer[obs_index / 8] ^= (1 << (obs_index % 8));
                }
            }
        }
        return result_array;
    }
};

struct MultiPassSinterDecoder {
    size_t num_passes;
    py::object full_decomposer;
    py::object detector_classifier;
    TesseractConfig base_config;
    size_t num_det_orders;
    ::DetOrder det_order_method;
    uint64_t seed;
    SchedulingStrategy strategy;

    MultiPassSinterDecoder(size_t n=2) : num_passes(n), full_decomposer(py::none()), detector_classifier(py::none()), num_det_orders(1), det_order_method(::DetOrder::DetBFS), seed(0), strategy(SchedulingStrategy::Static) {}

    MultiPassSinterCompiledDecoder compile_decoder_for_dem(const py::object& dem) {
        stim::DetectorErrorModel stim_dem;
        
        if (!full_decomposer.is_none()) {
            py::gil_scoped_acquire acquire;
            py::object decomposed_py_dem = full_decomposer(dem);
            stim_dem = stim::DetectorErrorModel(py::cast<std::string>(py::str(decomposed_py_dem)).c_str());
        } else {
            stim_dem = stim::DetectorErrorModel(py::cast<std::string>(py::str(dem)).c_str());
        }

        std::vector<int> classification;
        if (py::isinstance<py::function>(detector_classifier)) {
            uint64_t num_dets = stim_dem.count_detectors();
            
            std::set<uint64_t> detector_ids;
            std::map<uint64_t, std::string> tags;
            for (const auto& inst : stim_dem.flattened().instructions) {
                if (inst.type == stim::DemInstructionType::DEM_DETECTOR) {
                    uint64_t d = inst.target_data[0].val();
                    detector_ids.insert(d);
                    tags[d] = inst.tag;
                }
            }
            auto coords_map = stim_dem.get_detector_coordinates(detector_ids);

            for (uint64_t i = 0; i < num_dets; ++i) {
                std::vector<double> c = coords_map.count(i) ? coords_map.at(i) : std::vector<double>{};
                std::string t = tags.count(i) ? tags.at(i) : "";
                py::gil_scoped_acquire acquire;
                classification.push_back(py::cast<int>(detector_classifier((int)i, c, t)));
            }
        }

        tesseract::DetectorClassifier classifier = [classification](int index, const std::vector<double>& coords, const std::string& tag) -> int {
            if (index >= 0 && (size_t)index < classification.size()) return classification[index];
            return 0;
        };

        auto decoder = std::make_unique<tesseract::MultiPassTesseractDecoder>(stim_dem, num_passes, classifier, base_config, num_det_orders, det_order_method, seed, strategy);

        return MultiPassSinterCompiledDecoder(std::move(decoder), stim_dem.count_detectors(), stim_dem.count_observables());
    }
};

void pybind_multi_pass_sinter_compat(py::module& m) {
    py::enum_<SchedulingStrategy>(m, "SchedulingStrategy")
        .value("Static", SchedulingStrategy::Static)
        .value("Causal", SchedulingStrategy::Causal)
        .export_values();

    py::class_<MultiPassSinterCompiledDecoder>(m, "MultiPassSinterCompiledDecoder")
        .def_property_readonly("num_components", &MultiPassSinterCompiledDecoder::num_components)
        .def("decode_shots_bit_packed", &MultiPassSinterCompiledDecoder::decode_shots_bit_packed,
             py::kw_only(), py::arg("bit_packed_detection_event_data"),
             py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

    py::class_<MultiPassSinterDecoder>(m, "MultiPassSinterDecoder")
        .def(py::init<size_t>(), py::arg("num_passes") = 2)
        .def_readwrite("full_decomposer", &MultiPassSinterDecoder::full_decomposer)
        .def_readwrite("detector_classifier", &MultiPassSinterDecoder::detector_classifier)
        .def_readwrite("base_config", &MultiPassSinterDecoder::base_config)
        .def_readwrite("num_det_orders", &MultiPassSinterDecoder::num_det_orders)
        .def_readwrite("det_order_method", &MultiPassSinterDecoder::det_order_method)
        .def_readwrite("seed", &MultiPassSinterDecoder::seed)
        .def_readwrite("strategy", &MultiPassSinterDecoder::strategy)
        .def("compile_decoder_for_dem", &MultiPassSinterDecoder::compile_decoder_for_dem,
             py::kw_only(), py::arg("dem"));
}

} // namespace tesseract

#endif // MULTI_PASS_SINTER_COMPAT_PYBIND_H
