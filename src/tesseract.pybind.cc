#include <pybind11/pybind11.h>

#include "common.pybind.h"
#include "pybind11/detail/common.h"

PYBIND11_MODULE(tesseract_decoder, m) { add_common_module(m); }
