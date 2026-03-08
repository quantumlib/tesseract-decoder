# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_python//python:py_test.bzl", "py_test")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_binary.bzl", "py_binary")

py_library(
    name = "shared_decoding_tests",
    srcs = ["shared_decoding_tests.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        "@pypi//pytest",
        "@pypi//stim",
        "@pypi//numpy",
        "//src:lib_tesseract_decoder",
    ],
    imports = ["..", "."],
)


py_test(
    name = "common_test",
    srcs = ["common_test.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        "@pypi//pytest",
        "@pypi//stim",
        "//src:lib_tesseract_decoder",
    ],
    imports = ["..", "."],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        "@pypi//pytest",
        "@pypi//stim",
        "//src:lib_tesseract_decoder",
    ],
    imports = ["..", "."],
)

py_test(
    name = "simplex_test",
    srcs = ["simplex_test.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        "@pypi//pytest",
        "@pypi//stim",
        "//src:lib_tesseract_decoder",
        ":shared_decoding_tests",
    ],
    imports = ["..", "."],
)

py_test(
    name = "tesseract_test",
    srcs = ["tesseract_test.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        "@pypi//pytest",
        "@pypi//stim",
        "//src:lib_tesseract_decoder",
        ":shared_decoding_tests",
    ],
    imports = ["..", "."],
)
py_test(
    name = "tesseract_sinter_compat_test",
    srcs = ["tesseract_sinter_compat_test.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        "@pypi//pytest",
        "@pypi//stim",
        "@pypi//sinter",
        "//src:lib_tesseract_decoder",
    ],
    imports = ["..", "."],
)



py_test(
    name = "stub_test",
    srcs = ["stub_test.py"],
    data = [":generated_stubs"],
    visibility = ["//:__subpackages__"],
    deps = [
        "@pypi//pytest",
    ],
    imports = ["..", "."],
)

py_binary(
    name = "generate_stubs",
    srcs = ["generate_stubs.py"],
    deps = [
        "//src:lib_tesseract_decoder",
        "@pypi//pybind11_stubgen",
        "@pypi//stim",
    ],
    imports = ["..", "."],
)

STUB_FILES = [
    "__init__.pyi",
    "common.pyi",
    "simplex.pyi",
    "tesseract.pyi",
    "tesseract_sinter_compat.pyi",
    "utils.pyi",
    "viz.pyi",
]

genrule(
    name = "generated_stubs",
    tools = [":generate_stubs"],
    outs = ["tesseract_decoder-stubs/" + f for f in STUB_FILES],
    cmd = "$(location :generate_stubs) --output-dir $(@D)",
    visibility = ["//visibility:public"],
)

compile_pip_requirements(
    name = "requirements",
    src = "requirements.in",
    requirements_txt = "requirements_lock.txt",
)
