load("@rules_python//python:py_test.bzl", "py_test")
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_binary.bzl", "py_binary")

py_library(
    name = "_tesseract_py_util",
    srcs = glob(["*.py"], exclude=["*_test.py"]),
    visibility = ["//:__subpackages__"],
    deps = [
        "@pypi//stim",
        "@pypi//numpy",
        "@pypi//scipy",
    ],
)

py_binary(
    name = "gari_dem_utils",
    srcs = ["gari_dem_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_tesseract_py_util",
        "@pypi//scipy",
        "@pypi//numpy",
        "@pypi//stim",
    ],
)


py_test(
    name = "demutil_test",
    srcs = ["demutil_test.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        "@pypi//pytest",
        "@pypi//stim",
        "//src:lib_tesseract_decoder",
        ":_tesseract_py_util",
    ],
    imports = ["..", ".", "../.."],
)


py_test(
    name = "decompose_errors_test",
    srcs = ["decompose_errors_test.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        ":_tesseract_py_util",
        "@pypi//pytest",
        "@pypi//stim",
    ],
    imports = ["..", "."],
)
