#!/usr/bin/env python3
# Copyright 2025 Google LLC
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

"""Check that a macOS wheel honors the deployment target its filename declares.

Every Mach-O extension inside the wheel must carry a macOS minimum-version
load command (LC_BUILD_VERSION or LC_VERSION_MIN_MACOSX) no newer than the
wheel's platform tag; otherwise pip installs a wheel whose extensions refuse
to load on the oldest macOS the tag claims to support.

Usage:

    python devtools/check_macos_wheel.py wheelhouse/*.whl

Requires the `otool` command-line tool and uses only the Python standard
library. Exits nonzero with a diagnostic if the wheel violates its tag.
"""

import argparse
import pathlib
import re
import subprocess
import tempfile
import zipfile

_MACOS_TAG = re.compile(r"-macosx_(\d+)_(\d+)_(arm64|x86_64)\.whl$")


def declared_target(wheel):
    """Returns the deployment target declared by the wheel filename."""
    match = _MACOS_TAG.search(wheel.name)
    if match is None:
        raise SystemExit(f"could not determine macOS tag from {wheel.name}")
    return (int(match.group(1)), int(match.group(2)), 0)


def parse_version(value):
    parts = [int(part) for part in value.split(".")]
    return tuple((parts + [0, 0, 0])[:3])


def minimum_versions(extension):
    """Returns (load command, version) pairs declaring a macOS minimum."""
    output = subprocess.check_output(["otool", "-l", str(extension)], text=True)
    versions = []
    for command in re.split(r"\n\s*Load command \d+\n", output):
        if re.search(r"^\s*cmd LC_BUILD_VERSION$", command, re.MULTILINE):
            platform = re.search(r"^\s*platform\s+(\S+)$", command, re.MULTILINE)
            minos = re.search(r"^\s*minos\s+(\S+)$", command, re.MULTILINE)
            if platform and platform.group(1) in {"macos", "1"} and minos:
                versions.append(("LC_BUILD_VERSION", minos.group(1)))
        if re.search(r"^\s*cmd LC_VERSION_MIN_MACOSX$", command, re.MULTILINE):
            version = re.search(r"^\s*version\s+(\S+)$", command, re.MULTILINE)
            if version:
                versions.append(("LC_VERSION_MIN_MACOSX", version.group(1)))
    return versions


def check_wheel(wheel):
    target = declared_target(wheel)
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(root)
        extensions = sorted(root.rglob("*.so"))
        if not extensions:
            raise SystemExit(f"no Mach-O extension found in {wheel.name}")
        for extension in extensions:
            name = extension.relative_to(root)
            versions = minimum_versions(extension)
            if not versions:
                raise SystemExit(f"no macOS minimum-version load command in {name}")
            for command, value in versions:
                if parse_version(value) > target:
                    raise SystemExit(
                        f"{name}: {command} minimum {value} exceeds "
                        f"wheel tag macosx_{target[0]}_{target[1]}"
                    )
                print(f"{name}: {command} minimum {value}")
    print(f"{wheel.name}: all Mach-O extensions honor macosx_{target[0]}_{target[1]}")


def main():
    parser = argparse.ArgumentParser(
        description="Check a macOS wheel against its deployment-target tag."
    )
    parser.add_argument("wheel", type=pathlib.Path, help="path to one built .whl file")
    args = parser.parse_args()
    check_wheel(args.wheel)


if __name__ == "__main__":
    main()
