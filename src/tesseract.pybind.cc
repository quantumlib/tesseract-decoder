#include <pybind11/pybind11.h>
#include "pybind11/detail/common.h"

#include "common.pybind.h"
#include "utils.pybind.h"
#include "simplex.pybind.h"
#include "tesseract.pybind.h"

PYBIND11_MODULE(tesseract_decoder, tesseract)
{
    py::module::import("stim");
    add_common_module(tesseract);
    add_utils_module(tesseract);
    add_simplex_module(tesseract);
    add_tesseract_module(tesseract);
}
