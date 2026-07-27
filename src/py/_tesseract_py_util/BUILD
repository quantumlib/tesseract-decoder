load("@rules_python//python:py_test.bzl", "py_test")
load("@rules_python//python:py_library.bzl", "py_library")

py_library(
    name = "_tesseract_py_util",
    srcs = glob(
        ["*.py"],
        exclude = [
            "*_test.py",
            "gari.py",
        ],
    ),
    visibility = ["//:__subpackages__"],
    deps = [
        "@pypi//stim",
        "@pypi//numpy",
    ],
)

py_library(
    name = "gari",
    srcs = ["gari.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        ":_tesseract_py_util",
        "@pypi//numpy",
        "@pypi//scipy",
        "@pypi//stim",
    ],
)

py_test(
    name = "gari_test",
    srcs = ["gari_test.py"],
    imports = ["..", "."],
    visibility = ["//:__subpackages__"],
    deps = [
        ":gari",
        "@pypi//numpy",
        "@pypi//pytest",
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
