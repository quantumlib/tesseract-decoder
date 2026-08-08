load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_test.bzl", "py_test")
load("@rules_python//python:py_library.bzl", "py_library")

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
    name = "decompose_errors_cli",
    srcs = ["decompose_errors.py"],
    imports = [".."],
    main = "decompose_errors.py",
    visibility = ["//visibility:public"],
    deps = [
        "@pypi//stim",
    ],
)

py_test(
    name = "gari_test",
    srcs = ["gari_test.py"],
    imports = ["..", ".", "../.."],
    visibility = ["//:__subpackages__"],
    deps = [
        ":_tesseract_py_util",
        "@pypi//numpy",
        "@pypi//pytest",
        "@pypi//stim",
        "//src:lib_tesseract_decoder",
    ],
)

py_test(
    name = "decompose_errors_test",
    srcs = ["decompose_errors_test.py"],
    imports = ["..", ".", "../.."],
    visibility = ["//:__subpackages__"],
    deps = [
        ":_tesseract_py_util",
        "@pypi//pytest",
        "@pypi//stim",
        "//src:lib_tesseract_decoder",
    ],
)
