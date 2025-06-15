load("@pybind11_bazel//:build_defs.bzl", "pybind_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

SOURCE_FILES_NO_MAIN = glob(
    [
        "src/**/*.cc",
        "src/**/*.h",
        "src/**/*.inl",
    ],
    exclude = glob([
        "src/**/*.test.cc",
        "src/**/*.test.h",
        "src/**/*.perf.cc",
        "src/**/*.perf.h",
        "src/**/*.pybind.cc",
        "src/**/*.pybind.h",
        "src/**/main.cc",
    ]),
)

PYBIND_MODULES = [
    "src/stim/py/march.pybind.cc",
    "src/stim/py/stim.pybind.cc",
]

PYBIND_FILES_WITHOUT_MODULES = glob(
    [
        "src/**/*.pybind.cc",
        "src/**/*.pybind.h",
    ],
    exclude=PYBIND_MODULES,
)



pybind_library(
    name = "stim_pybind_lib",
    srcs = SOURCE_FILES_NO_MAIN + PYBIND_FILES_WITHOUT_MODULES,
    copts = [
        "-O3",
        "-std=c++20",
        "-fvisibility=hidden",
        "-march=native",
        "-DVERSION_INFO=0.0.dev0",
    ],
    includes = ["src/"],
  visibility = ["//visibility:public"],
)

pybind_extension(
    name = "stim",
    srcs = PYBIND_MODULES,
    copts = [
        "-O3",
        "-std=c++20",
        "-fvisibility=hidden",
        "-march=native",
        "-DSTIM_PYBIND11_MODULE_NAME=stim",
        "-DVERSION_INFO=0.0.dev0",
    ],
    deps=[":stim_pybind_lib"],
    includes = ["src/"],
  visibility = ["//visibility:public"],
)
