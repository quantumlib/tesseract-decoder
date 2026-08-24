#!/usr/bin/env python3
# Copyright 2026 Google LLC
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

"""Validate an installed Tesseract wheel and run a small decode smoke test."""

import argparse
import pathlib
import sys
import sysconfig
import zipfile

import numpy as np
import stim
import tesseract_decoder
from tesseract_decoder import demutil


def check_wheel_tag(wheel: pathlib.Path, expected_python_tag: str) -> None:
    """Checks the wheel filename and metadata against an expected CPython ABI."""
    expected_fragment = f"-{expected_python_tag}-{expected_python_tag}-"
    if expected_fragment not in wheel.name:
        raise SystemExit(
            f"{wheel.name}: expected filename to contain {expected_fragment!r}"
        )

    with zipfile.ZipFile(wheel) as archive:
        metadata_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")
        ]
        if len(metadata_files) != 1:
            raise SystemExit(
                f"{wheel.name}: expected one WHEEL metadata file, "
                f"found {len(metadata_files)}"
            )
        metadata = archive.read(metadata_files[0]).decode()

    tags = [
        line.removeprefix("Tag: ")
        for line in metadata.splitlines()
        if line.startswith("Tag: ")
    ]
    expected_prefix = f"{expected_python_tag}-{expected_python_tag}-"
    if not tags or any(not tag.startswith(expected_prefix) for tag in tags):
        raise SystemExit(
            f"{wheel.name}: expected only {expected_prefix}* metadata tags, found {tags}"
        )


def check_decode() -> None:
    """Exercises Stim, NumPy, and the installed native decoder together."""
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 L0
        detector(0) D0
        """
    )
    decoder = tesseract_decoder.tesseract.TesseractConfig(
        dem=dem, det_orders=[[0]]
    ).compile_decoder()
    prediction = decoder.decode(np.array([True], dtype=np.bool_))
    if prediction.tolist() != [True]:
        raise SystemExit(f"unexpected decoder prediction: {prediction}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("expected_python_tag", help="expected tag, such as cp314")
    parser.add_argument("wheel", type=pathlib.Path)
    args = parser.parse_args()

    running_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if running_tag != args.expected_python_tag:
        raise SystemExit(
            f"running under {running_tag}, expected {args.expected_python_tag}"
        )
    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        raise SystemExit("free-threaded Python requires a separate wheel and safety audit")

    check_wheel_tag(args.wheel, args.expected_python_tag)
    check_decode()
    print(tesseract_decoder.__file__)
    print(f"{args.wheel.name}: wheel validation passed")


if __name__ == "__main__":
    main()
