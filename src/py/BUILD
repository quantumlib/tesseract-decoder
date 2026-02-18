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
)


py_library(
    name = "decomposer",
    srcs = ["decomposer.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        "@pypi//stim",
    ],
)

py_test(
    name = "decomposer_test",
    srcs = ["decomposer_test.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        ":decomposer",
        "@pypi//pytest",
        "@pypi//stim",
    ],
)

compile_pip_requirements(
    name = "requirements",
    src = "requirements.in",
    requirements_txt = "requirements_lock.txt",
)
