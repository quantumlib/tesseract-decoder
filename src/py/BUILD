load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")

py_binary(
    name = "tmp",
    srcs = ["tmp.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        "@pypi//numpy",
        "@pypi//stim",
        "//src:lib_tesseract_decoder",
    ],
    # imports = ["//src:lib_tesseract_decoder"],
)

compile_pip_requirements(
    name = "requirements",
    src = "requirements.in",
    requirements_txt = "requirements_lock.txt",
)
