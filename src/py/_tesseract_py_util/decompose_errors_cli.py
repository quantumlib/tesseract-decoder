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

"""Command-line interface for decomposing Stim detector error models."""

import argparse
from collections.abc import Sequence
import sys

import stim

from _tesseract_py_util.decompose_errors import decompose_errors


_METHODS = ("stim-surfacecode-coords", "last-coordinate-index")


def call_decompose_errors(
    input_path: str,
    output_path: str,
    *,
    method: str,
    strip_undecomposable_errors: bool,
) -> None:
    """Reads, decomposes, and writes one detector error model."""
    if input_path == "-":
        dem = stim.DetectorErrorModel(sys.stdin.read())
    else:
        dem = stim.DetectorErrorModel.from_file(input_path)

    output_dem = decompose_errors(
        dem,
        method=method,
        strip_undecomposable_errors=strip_undecomposable_errors,
    )
    if output_path == "-":
        print(output_dem)
    else:
        output_dem.to_file(output_path)


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decompose errors in a Stim detector error model."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="-",
        help="Input DEM file (default: standard input; use '-' for standard input).",
    )
    parser.add_argument(
        "-o",
        "--out",
        default="-",
        help="Output DEM file (default: standard output; use '-' for standard output).",
    )
    parser.add_argument(
        "--method",
        choices=_METHODS,
        default="stim-surfacecode-coords",
        help="Detector-component convention used for decomposition.",
    )
    parser.add_argument(
        "--strip-undecomposable-errors",
        action="store_true",
        help="Drop errors that cannot be decomposed instead of failing.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _create_argument_parser()
    args = parser.parse_args(argv)
    try:
        call_decompose_errors(
            args.input,
            args.out,
            method=args.method,
            strip_undecomposable_errors=args.strip_undecomposable_errors,
        )
    except (IndexError, KeyError, OSError, ValueError) as ex:
        print(f"{parser.prog}: error: {ex}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
