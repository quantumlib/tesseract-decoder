#ifndef _STIM_UTILS_PYBIND_H
#define _STIM_UTILS_PYBIND_H

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "stim.h"

namespace {
namespace py = pybind11;
}

template <typename T>
py::object make_py_object(const T cpp_obj, const char* py_name) {
  auto stim_lib = py::module::import("stim");
  return stim_lib.attr(py_name)(cpp_obj.str());
}

template <typename T>
T parse_py_object(py::object py_obj) {
  std::string obj_str = py::cast<std::string>(py_obj.attr("__str__")());
  return T(obj_str);
}

stim::DemInstructionType parse_dit(std::string dit_str) {
  if (dit_str == "error") return stim::DemInstructionType::DEM_ERROR;
  if (dit_str == "detector") return stim::DemInstructionType::DEM_DETECTOR;
  if (dit_str == "logical_observable") return stim::DemInstructionType::DEM_LOGICAL_OBSERVABLE;
  if (dit_str == "shift_detectors") return stim::DemInstructionType::DEM_SHIFT_DETECTORS;
  if (dit_str == "repeat") return stim::DemInstructionType::DEM_REPEAT_BLOCK;
  throw std::invalid_argument("unknown dem instruction type: " + dit_str);
  return stim::DemInstructionType::DEM_DETECTOR;
}

stim::DemTarget parse_py_dem_target(py::object py_obj) {
  return stim::DemTarget::from_text(py::cast<std::string>(py_obj.attr("__str__")()));
}

stim::DemInstruction parse_py_dem_instruction(py::object py_obj, std::vector<double>& args,
                                              std::vector<stim::DemTarget>& targets) {
  for (auto t : py_obj.attr("args_copy")()) args.push_back(t.cast<double>());
  stim::SpanRef args_ref(args);

  for (auto t : py_obj.attr("targets_copy")())
    targets.push_back(parse_py_dem_target(t.cast<py::object>()));

  stim::SpanRef targets_ref(targets);
  auto ty = parse_dit(py::cast<std::string>(py_obj.attr("type")));
  std::string tag = py::cast<std::string>(py_obj.attr("tag"));

  auto di = stim::DemInstruction();
  di.arg_data = args_ref;
  di.target_data = targets_ref;
  di.tag = tag;
  di.type = ty;
  return di;
}

template <typename T>
py::object dem_getter(const T& config) {
  return make_py_object(config.dem, "DetectorErrorModel");
}
template <typename T>
void dem_setter(T& config, py::object dem) {
  config.dem = parse_py_object<stim::DetectorErrorModel>(dem);
}

#endif
