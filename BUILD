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

MANYLINUX_VERSION="manylinux_2_42_x86_64"

py_wheel(
    name="tesseract_decoder_wheel",
    distribution = "tesseract_decoder",
    deps=[
        "//src:tesseract_decoder",
        "//src/py:generated_stubs",
        "//src/py/_tesseract_py_util:_tesseract_py_util",
        ":package_data",
    ],
    version = "$(VERSION)",
    requires=[
        "numpy",
        "scipy",
        "stim",
    ],
    python_tag="$(ABI_TAG)",
    abi = select({
        "@rules_python//python/config_settings:py_freethreaded=yes": "$(ABI_TAG)t",
        "//conditions:default": "$(ABI_TAG)",
    }),
    platform= select({
        ":macos_arm": "macosx_11_0_arm64",
        "@platforms//os:windows": "win32",
        "@platforms//os:linux": MANYLINUX_VERSION,
    }),
    strip_path_prefixes = ["src/py", "src"],
    description_file=":package_description",
    description_content_type="text/markdown",
    summary="A search-based decoder for quantum error correction (QEC).",
    author="The Tesseract Decoder Authors.",
    homepage="https://github.com/quantumlib/tesseract-decoder",
    license="Apache 2",
)

config_setting(
    name = "macos_arm",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:arm64",
    ],
)
