#include <pybind11/pybind11.h>
#include "pybind11/detail/common.h"

#include "common.pybind.h"

PYBIND11_MODULE(tesseract_py, m)
{
    add_common_module(m);
}
