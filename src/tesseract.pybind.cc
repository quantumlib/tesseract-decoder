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

#include "tesseract.pybind.h"

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>

#include "common.pybind.h"
#include "pybind11/detail/common.h"
#include "simplex.pybind.h"
#include "tesseract_sinter_compat.pybind.h"
#include "utils.pybind.h"
#include "visualization.pybind.h"

PYBIND11_MODULE(tesseract_decoder, tesseract) {
  py::module::import("stim");

  add_common_module(tesseract);
  add_utils_module(tesseract);
  add_simplex_module(tesseract);
  add_tesseract_module(tesseract);
  add_visualization_module(tesseract);
  pybind_sinter_compat(tesseract);

  // Adds a context manager to the python library that can be used to redirect C++'s stdout/stderr
  // to python's stdout/stderr at run time like
  // with tesseract_decoder.ostream_redirect(stdout=..., stderr=...):
  //    do_work()
  // This is only needed if the C++ function's stdout/stderr is not redirected to python's
  // stdout/stderr using the py::call_guard<py::scoped_ostream_redirect,
  // py::scoped_estream_redirect>() statement.
  py::add_ostream_redirect(tesseract, "ostream_redirect");
}
