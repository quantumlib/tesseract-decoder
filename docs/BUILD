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
load("@rules_python//python:py_test.bzl", "py_test")

TUTORIAL_FILES = glob([
    "tutorial.ipynb",
]) + glob(
    ["tutorial.py"],
    allow_empty = True,
)

py_binary(
    name = "sync_tutorial_notebook",
    srcs = ["sync_tutorial_notebook.py"],
    data = TUTORIAL_FILES,
    deps = ["@pypi//jupytext"],
)

py_test(
    name = "tutorial_jupytext_sync_test",
    srcs = ["sync_tutorial_notebook.py"],
    args = ["--check"],
    data = TUTORIAL_FILES,
    main = "sync_tutorial_notebook.py",
    deps = ["@pypi//jupytext"],
)
