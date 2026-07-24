#ifndef _BP_PYBIND_H
#define _BP_PYBIND_H

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bp/bp_params.h"
#include "bp/hard_decision_post_processor.h"
#include "bp/osd_post_processor.h"
#include "bp/post_processor.h"
#include "bp/tesseract_bp_decoder.h"
#include "stim.h"
#include "stim_utils.pybind.h"  // For parse_py_object

namespace py = pybind11;

namespace bp {

void add_bp_module(py::module& root) {
  auto m = root.def_submodule("bp", "Module containing the Belief Propagation decoder");

  py::class_<BPParams>(m, "BPParams", R"pbdoc(
        Configuration for the Belief Propagation decoder.
    )pbdoc")
      .def(py::init<>())
      .def_readwrite("max_iter", &BPParams::max_iter)
      .def_readwrite("update_rule", &BPParams::update_rule)
      .def_readwrite("schedule", &BPParams::schedule)
      .def_readwrite("normalization_factor", &BPParams::normalization_factor);

  py::class_<PostProcessor, std::shared_ptr<PostProcessor>>(m, "PostProcessor");

  py::class_<HardDecisionPostProcessor, PostProcessor, std::shared_ptr<HardDecisionPostProcessor>>(
      m, "HardDecisionPostProcessor")
      .def(py::init<>());

  py::class_<OsdPostProcessor, PostProcessor, std::shared_ptr<OsdPostProcessor>>(m,
                                                                                 "OsdPostProcessor")
      .def(py::init<const TannerGraph<LLR_INT>&, size_t, size_t>(), py::arg("graph"),
           py::arg("osd_order") = 0, py::arg("osd_weight") = 0);

  py::class_<TesseractBpDecoder>(m, "TesseractBpDecoder", R"pbdoc(
        A top-level Belief Propagation orchestrator.
    )pbdoc")
      .def(py::init([](py::object dem, const BPParams& config) {
             return std::make_unique<TesseractBpDecoder>(
                 parse_py_object<stim::DetectorErrorModel>(dem), config);
           }),
           py::arg("dem"), py::arg("config"))
      .def("create_osd_post_processor", &TesseractBpDecoder::create_osd_post_processor,
           py::arg("osd_order") = 0, py::arg("osd_weight") = 0)
      .def(
          "decode",
          [](TesseractBpDecoder& self, const py::array_t<bool>& syndrome,
             const std::shared_ptr<PostProcessor>& post_processor) {
            if ((size_t)syndrome.size() != self.num_detectors()) {
              throw std::invalid_argument("Syndrome size does not match decoder detector count.");
            }

            std::vector<uint64_t> detections;
            auto syndrome_unchecked = syndrome.unchecked<1>();
            for (size_t i = 0; i < (size_t)syndrome_unchecked.size(); ++i) {
              if (syndrome_unchecked(i)) {
                detections.push_back(i);
              }
            }

            std::vector<uint8_t> predictions = self.decode(detections, post_processor);

            size_t num_obs = self.num_observables();
            std::vector<char> result(num_obs, 0);
            for (size_t i = 0; i < predictions.size(); ++i) {
              if (predictions[i]) {
                result[i] = 1;
              }
            }

            return py::array(py::dtype::of<bool>(), result.size(), result.data());
          },
          py::arg("syndrome"), py::arg("post_processor"),
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(),
          R"pbdoc(
            Decodes a single shot from a dense boolean array using a post-processor.
            Returns predicted observables.
          )pbdoc");
}

}  // namespace bp

#endif
