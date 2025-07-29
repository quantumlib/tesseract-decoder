load("@rules_python//python:packaging.bzl", "py_wheel")

filegroup(
    name="package_description",
    srcs=["README.md"],
    visibility = ["//visibility:public"],
)


filegroup(
    name="package_data",
    srcs=["LICENSE"],
    visibility = ["//visibility:public"],
)

py_wheel(
    name="tesseract_decoder_wheel",
    distribution = "tesseract_decoder",
    deps=[
        "//src:tesseract_decoder",
        ":package_data",
    ],
    version = "$(VERSION)",
    requires=[
        "stim",
    ],
    python_tag="$(TARGET_VERSION)",
    platform= select({
        ":macos_arm": "macosx_11_0_arm64",
        ":macos_x86": "macosx_10_13_x86_64",
        "@platforms//os:windows": "win32",
        "@platforms//os:linux": "manylinux_2_17_x86_64.manylinux2014_x86_64",
    }),
    strip_path_prefixes = ["src"],
    description_file=":package_description",
    description_content_type="text/markdown",
    summary="A search-based decoder for quantum error correction (QEC).",
    author="The Tesseract Decoder Authors.",
    homepage="https://github.com/quantumlib/tesseract-decoder",
)

config_setting(
    name = "macos_arm",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:arm",
    ]
)

config_setting(
    name = "macos_x86",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:x86_64",
    ]
)
