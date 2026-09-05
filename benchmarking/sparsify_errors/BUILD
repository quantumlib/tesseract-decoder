# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

py_library(
    name = "benchmark_data",
    srcs = ["benchmark_data.py"],
    imports = [
        "../../src",
        "../../src/py",
    ],
    deps = [
        "//src:lib_tesseract_decoder",
        "@pypi//stim",
    ],
)

py_binary(
    name = "aggregate",
    srcs = ["aggregate.py"],
    deps = [":benchmark_data"],
)

py_library(
    name = "plot_lib",
    srcs = ["plot.py"],
    deps = [
        ":benchmark_data",
        "@pypi//matplotlib",
        "@pypi//numpy",
        "@pypi//scipy",
    ],
)

py_binary(
    name = "enrich",
    srcs = ["enrich.py"],
    deps = [":benchmark_data"],
)

py_test(
    name = "workflow_test",
    srcs = ["workflow_test.py"],
    deps = [
        ":benchmark_data",
        ":plot_lib",
    ],
)

py_binary(
    name = "plot",
    srcs = ["plot.py"],
    data = ["aggregated_results.jsonl"],
    main = "plot.py",
    deps = [
        ":benchmark_data",
        "@pypi//matplotlib",
        "@pypi//numpy",
        "@pypi//scipy",
    ],
)
