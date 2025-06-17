# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

HIGHS_VERSION = "1.9.0"
HIGHS_SHA_256 = "dff575df08d88583c109702c7c5c75ff6e51611e6eacca8b5b3fdfba8ecc2cb4"

git_repository(
    name = "stim",
    commit = "bd60b73525fd5a9b30839020eb7554ad369e4337",
    remote = "https://github.com/quantumlib/stim.git",
    shallow_since = "1741128853 +0000",
)

http_archive(
    name = "highs",
    sha256 = HIGHS_SHA_256,
    build_file = "//external:highs.BUILD",
    strip_prefix = "HiGHS-" + HIGHS_VERSION,
    urls = ["https://github.com/ERGO-Code/HiGHS/archive/refs/tags/v" + HIGHS_VERSION + ".tar.gz"],
)

GTEST_VERSION = "1.13.0"

GTEST_SHA256 = "ad7fdba11ea011c1d925b3289cf4af2c66a352e18d4c7264392fead75e919363"

http_archive(
    name = "gtest",
    sha256 = GTEST_SHA256,
    strip_prefix = "googletest-%s" % GTEST_VERSION,
    urls = ["https://github.com/google/googletest/archive/refs/tags/v%s.tar.gz" % GTEST_VERSION],
)

ARGPARSE_SHA_256 = "3e5a59ab7688dcd1f918bc92051a10564113d4f36c3bbed3ef596c25e519a062"

http_archive(
    name = "argparse",
    build_file = "//external:argparse.BUILD",
    sha256 = ARGPARSE_SHA_256,
    strip_prefix = "argparse-3.1",
    urls = ["https://github.com/p-ranav/argparse/archive/refs/tags/v3.1.zip"],
)

git_repository(
    name = "nlohmann_json",
    commit = "9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03",
    remote = "https://github.com/nlohmann/json.git",
    shallow_since = "1701207391 +0100",
)

http_archive(
    name = "platforms",
    urls = ["https://github.com/bazelbuild/platforms/archive/refs/tags/0.0.6.zip"],
    strip_prefix = "platforms-0.0.6",
)

http_archive(
    name = "stim_py",
    build_file = "//external:stim_py.BUILD",
    sha256 = "95236006859d6754be99629d4fb44788e742e962ac8c59caad421ca088f7350e",
    strip_prefix = "stim-1.15.0",
    urls = ["https://github.com/quantumlib/Stim/releases/download/v1.15.0/stim-1.15.0.tar.gz"],
)
