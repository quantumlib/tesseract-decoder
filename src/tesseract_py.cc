#include <pybind11/pybind11.h>
#include "pybind11/detail/common.h"

#include "common_py.h"

PYBIND11_MODULE(tesseract_py, m)
{
    add_common_module(m);
}
