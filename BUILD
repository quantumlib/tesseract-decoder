load("@rules_python//python:packaging.bzl", "py_wheel")

filegroup(
    name="package_description",
    srcs=["README.md"],
    visibility = ["//visibility:public"],
)

py_wheel(
    name="tesseract_decoder_wheel",
    distribution = "tesseract_decoder",
    deps=[
        "//src:tesseract_decoder",
    ],
    version = "$(VERSION)",
    requires=[
        "stim",
    ],
    platform= select({
        "@platforms//os:macos": "macosx_11_0_arm64",
        "@platforms//os:windows": "win32",
        "@platforms//os:linux": "manylinux_2_17_x86_64.manylinux2014_x86_64",
    }),
    description_file=":package_description",
    description_content_type="text/markdown",
    summary="A search-based decoder for quantum error correction (QEC).",
    author="The Tesseract Decoder Authors.",
    homepage="https://github.com/quantumlib/tesseract-decoder",
)
