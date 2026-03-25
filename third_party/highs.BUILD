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

load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary", "cc_test")

NDEBUG_COPTS = [
  "-DNDEBUG",
]

copy_file(
    name = "highs-config",
    src = "src/HConfig.h.bazel.in",
    out = "HConfig.h",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "config",
    srcs = ["HConfig.h"],
    copts = NDEBUG_COPTS,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "highs",
    srcs = glob([
        "extern/filereaderlp/*.cpp",
        "src/interfaces/highs_c_api.cpp",
        "src/io/*.cpp",
        "src/ipm/*.cpp",
        "src/ipm/ipx/*.cc",
        "src/ipm/basiclu/*.c",
        "src/lp_data/*.cpp",
        "src/mip/*.cpp",
        "src/model/*.cpp",
        "src/parallel/*.cpp",
        "src/pdlp/*.cpp",
        "src/pdlp/cupdlp/*.c",
        "src/presolve/*.cpp",
        "src/qpsolver/*.cpp",
        "src/simplex/*.cpp",
        "src/test/*.cpp",
        "src/util/*.cpp",
    ]),
    hdrs = glob([
        "**/*.h",
        "src/qpsolver/*.hpp",
        "src/Highs.h",
        "extern/filereaderlp/*.hpp",
        "extern/zstr/*.hpp",
    ]),
    copts = [
        "-Wno-unused-variable",
        "-Wno-unused-but-set-variable",
    ] + NDEBUG_COPTS,
    includes = [
        "extern",
        # "extern/filereaderlp",
        # "extern/zstr",
        "src",
        # "src/ipm",
        # "src/ipm/ipx",
        # "src/ipm/basiclu",
        # "src/lp_data",
        # "src/mip",
        # "src/model",
        # "src/parallel",
        # "src/presolve",
        # "src/qpsolver",
        # "src/simplex",
        # "src/test",
        # "src/util",
        "bazel-bin",
    ],
    linkopts = ["-lpthread"],
    visibility = ["//visibility:public"],
    deps = [
        "//:config",
        "@zlib",
    ],
)

cc_binary(
    name = "call-highs-example",
    srcs = ["examples/call_highs_from_cpp.cpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//:highs",
    ],
)

## Add tests
copy_file(
    name = "highs-check-config",
    src = "check/HCheckConfig.h.bazel.in",
    out = "HCheckConfig.h",
)

cc_library(
    name = "check-config",
    srcs = ["HCheckConfig.h"],
)

cc_library(
    name = "test_lib",
    testonly = True,
    srcs = [
        "HCheckConfig.h",
        "check/Avgas.cpp",
        "check/TestMain.cpp",
    ],
    hdrs = [
        "check/Avgas.h",
        "check/SpecialLps.h",
        "check/matrix_multiplication.hpp",
        "extern/catch.hpp",
    ],
    copts = ["-Iextern"],
    data = glob(["check/instances/*"]),
    includes = ["check"],
    deps = [
        ":highs",
        "//:check-config",
    ],
)

TEST_NAMES = [
    "TestAlienBasis",
    "TestBasis",
    "TestBasisSolves",
    "TestCheckSolution",
    "TestCrossover",
    "TestDualize",
    "TestEkk",
    "TestFactor",
    "TestFilereader",
    "TestFreezeBasis",
    "TestHighsGFkSolve",
    "TestHighsHash",
    "TestHighsHessian",
    "TestHighsIntegers",
    "TestHighsModel",
    "TestHighsParallel",
    "TestHighsRbTree",
    "TestHotStart",
    "TestHSet",
    "TestICrash",
    "TestInfo",
    "TestIO",
    "TestIpx",
    "TestLogging",
    "TestLpModification",
    "TestLpOrientation",
    "TestLpSolvers",
    "TestLpValidation",
    "TestMipSolver",
    "TestPresolve",
    "TestQpSolver",
    "TestRanging",
    "TestRays",
    "TestSemiVariables",
    "TestSetup",
    "TestSort",
    "TestSpecialLps",
    "TestThrow",
]

[cc_test(
    name = name,
    srcs = ["check/%s.cpp" % name],
    copts = [
        "-Iextern",
        "-Wno-unused-variable",
        "-Wno-unused-but-set-variable",
    ],
    deps = [
        ":highs",
        ":test_lib",
    ],
) for name in TEST_NAMES]
