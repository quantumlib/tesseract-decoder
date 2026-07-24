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
    ],
    ":debug": ["-Og"],
}) + select({
        "//conditions:default": ["-std=c++20"],
}) + select({
    "@platforms//os:macos": ["-mmacosx-version-min=10.15",],
    "//conditions:default": ["-march=native",],
})

OPT_LINKOPTS = select({
    "//conditions:default": [
        "-Ofast",
        "-fno-fast-math",
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
        "stim_utils.pybind.h",
        "common.pybind.h",
        "utils.pybind.h",
        "simplex.pybind.h",
        "visualization.pybind.h",
        "tesseract.pybind.h",
        "tesseract_sinter_compat.pybind.h",
        "bp/bp.pybind.h",
        "bp/bp_sinter_compat.pybind.h",
    ],
    copts = OPT_COPTS,
    deps = [
        ":libcommon",
        ":libutils",
        ":libsimplex",
        ":libtesseract",
        ":libbp",
    ],
)

pybind_extension(
    name = "tesseract_decoder",
    srcs = [
        "tesseract.pybind.cc",
    ],
    copts = OPT_COPTS,
    deps = [
        ":tesseract_decoder_pybind",
    ],
)

py_library(
    name="lib_tesseract_decoder",
    deps=[
        ":tesseract_decoder",
        "//src/py/_tesseract_py_util:_tesseract_py_util",
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
    name = "libviz",
    srcs = ["visualization.cc"],
    hdrs = ["visualization.h"],
    copts = OPT_COPTS,
    linkopts = OPT_LINKOPTS,
    deps = [
        ":libutils",
        "@boost//:dynamic_bitset",
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
        ":libviz",
        "@boost//:dynamic_bitset",
    ],
)

cc_library(
    name = "libtesseract_trellis",
    srcs = ["tesseract_trellis.cc"],
    hdrs = ["tesseract_trellis.h"],
    copts = OPT_COPTS,
    linkopts = OPT_LINKOPTS,
    deps = [
        ":libcommon",
        ":libutils",
        "@stim//:stim_lib",
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

cc_binary(
    name = "tesseract_trellis",
    srcs = ["tesseract_trellis_main.cc"],
    copts = OPT_COPTS,
    linkopts = OPT_LINKOPTS,
    deps = [
        ":libtesseract_trellis",
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
    name = "tesseract_trellis_tests",
    srcs = ["tesseract_trellis.test.cc"],
    copts = OPT_COPTS,
    linkopts = OPT_LINKOPTS,
    deps = [
        ":libtesseract_trellis",
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
        ":libutils",
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

cc_library(
    name = "libbp",
    srcs = [
        "bp/bp_sparse_serial_gallager.cc",
        "bp/check_update.cc",
        "bp/hard_decision_post_processor.cc",
        "bp/tesseract_bp_decoder.cc",
        "bp/osd_post_processor.cc",
    ],
    hdrs = [
        "bp/batched_bp_parallel_min_sum.h",
        "bp/batched_bp_parallel_min_sum.inl",
        "bp/batched_bp_serial_min_sum.h",
        "bp/batched_bp_serial_min_sum.inl",
        "bp/batched_tanner_graph.h",
        "bp/bp_parallel_min_sum.h",
        "bp/bp_parallel_min_sum.inl",
        "bp/bp_params.h",
        "bp/bp_serial_min_sum.h",
        "bp/bp_serial_min_sum.inl",
        "bp/bp_sparse_serial_gallager.h",
        "bp/bp_types.h",
        "bp/check_update.h",
        "bp/tanner_graph.h",
        "bp/tanner_graph.inl",
        "bp/tanner_graph_util.h",
        "bp/post_processor.h",
        "bp/hard_decision_post_processor.h",
        "bp/tesseract_bp_decoder.h",
        "bp/osd_post_processor.h",
    ],
    copts = OPT_COPTS,
    linkopts = OPT_LINKOPTS,
    includes = ["."],
    deps = [
        "@stim//:stim_lib",
        ":libcommon",
    ],
)

cc_test(
    name = "bp_tests",
    srcs = [
        "bp/batched_bp_parallel_min_sum.test.cc",
        "bp/batched_bp_serial_min_sum.test.cc",
        "bp/bp_parallel_min_sum.test.cc",
        "bp/bp_serial_min_sum.test.cc",
        "bp/bp_sparse_serial_gallager.test.cc",
        "bp/check_update.test.cc",
        "bp/tanner_graph.test.cc",
        "bp/hard_decision_post_processor.test.cc",
        "bp/osd_post_processor.test.cc",
        "bp/bp_test_util.h",
        "bp/bp_test_util.cc",
    ],
    copts = OPT_COPTS,
    linkopts = OPT_LINKOPTS,
    deps = [
        ":libbp",
        "@gtest",
        "@gtest//:gtest_main",
        "@stim//:stim_lib",
    ],
)

cc_binary(
    name = "bp",
    srcs = ["bp_main.cc"],
    copts = OPT_COPTS,
    linkopts = OPT_LINKOPTS,
    deps = [
        ":libbp",
        ":libcommon",
        ":libutils",
        "@argparse",
        "@nlohmann_json//:json",
        "@stim//:stim_lib",
    ],
)
