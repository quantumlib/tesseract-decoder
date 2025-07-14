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

# load("@benchmark//:benchmark.bzl", "cc_benchmark")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")
load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "debug",
    values = {"compilation_mode": "dbg"},
)

OPT_COPTS = select({
    "//conditions:default": [
        "-Ofast",
        "-fno-fast-math",
        "-march=native",
    ],
    ":debug": ["-Og"],
}) + select({
        "//conditions:default": ["-std=c++20"],
}) + select({
    "@platforms//os:macos": ["-mmacosx-version-min=10.15"],
    "//conditions:default": [],
})

OPT_LINKOPTS = select({
    "//conditions:default": [
        "-Ofast",
        "-fno-fast-math",
        "-march=native",
    ],
    ":debug": [],
}) + [
    "-lpthread",
]

cc_library(
    name = "libcommon",
    srcs = [
        "common.cc",
    ],
    hdrs = [
        "common.h",
    ],
    copts = OPT_COPTS,
    linkopts = OPT_LINKOPTS,
    deps = [
        "@stim//:stim_lib",
    ],
)


pybind_library(
    name = "tesseract_decoder_pybind",
    srcs = [
        "common.pybind.h",
        "utils.pybind.h",
        "simplex.pybind.h",
        "tesseract.pybind.h",
    ],
    deps = [
        ":libcommon",
        ":libutils",
        ":libsimplex",
        ":libtesseract",
    ],
)

pybind_extension(
    name = "tesseract_decoder",
    srcs = [
        "tesseract.pybind.cc",
    ],
    deps = [
        ":tesseract_decoder_pybind",
    ],
)


py_library(
    name="lib_tesseract_decoder",
    imports=["src"],
    deps=[
        ":tesseract_decoder",
    ],
)


cc_library(
    name = "libutils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    copts = OPT_COPTS,
    linkopts = OPT_LINKOPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":libcommon",
        "@stim//:stim_lib",
    ],
)

cc_library(
    name = "libtesseract",
    srcs = ["tesseract.cc"],
    hdrs = ["tesseract.h"],
    copts = OPT_COPTS,
    linkopts = OPT_LINKOPTS,
    deps = [
        ":libutils",
    ],
)

cc_binary(
    name = "tesseract",
    srcs = ["tesseract_main.cc"],
    copts = OPT_COPTS,
    linkopts = OPT_LINKOPTS,
    deps = [
        ":libtesseract",
        "@argparse",
        "@nlohmann_json//:json",
        "@stim//:stim_lib",
    ],
)

cc_test(
    name = "tesseract_tests",
    timeout = "eternal",
    srcs = ["tesseract.test.cc"],
    copts = OPT_COPTS,
    linkopts = OPT_LINKOPTS,
    deps = [
        ":libsimplex",
        ":libtesseract",
        ":libutils",
        "@gtest",
        "@gtest//:gtest_main",
        "@stim//:stim_lib",
    ],
)

cc_test(
    name = "common_tests",
    srcs = ["common.test.cc"],
    copts = OPT_COPTS,
    linkopts = OPT_LINKOPTS,
    linkstatic = True,
    deps = [
        ":libcommon",
        "@gtest",
        "@gtest//:gtest_main",
        "@stim//:stim_lib",
    ],
)

PERF_FILES = glob(["**/*.perf.*"])

cc_binary(
    name = "tesseract_perf",
    srcs = PERF_FILES,
    copts = OPT_COPTS,
    linkopts = OPT_LINKOPTS,
    deps = [
        ":libsimplex",
        ":libtesseract",
    ],
)

cc_library(
    name = "libsimplex",
    srcs = [
        "simplex.cc",
    ],
    hdrs = [
        "simplex.h",
    ],
    copts = OPT_COPTS,
    linkopts = OPT_LINKOPTS,
    deps = [
        ":libcommon",
        "@highs",
        "@stim//:stim_lib",
    ],
)

cc_binary(
    name = "simplex",
    srcs = [
        "simplex_main.cc",
    ],
    copts = OPT_COPTS,
    linkopts = OPT_LINKOPTS,
    deps = [
        ":libcommon",
        ":libsimplex",
        "@argparse",
        "@nlohmann_json//:json",
    ],
)

