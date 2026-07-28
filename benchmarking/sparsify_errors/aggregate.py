#!/usr/bin/env python3
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

"""Aggregates raw per-job sparsification statistics into strict JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmarking.sparsify_errors.benchmark_data import (
    BenchmarkDataError,
    aggregate_raw_rows,
    read_run_directories,
    write_jsonl,
)


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="Run directories containing manifest.json and jobs/*.json.",
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "aggregated_results.jsonl"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        raw_rows, run_provenance, circuit_sha256 = read_run_directories(args.runs)
        rows = aggregate_raw_rows(
            raw_rows,
            run_provenance=run_provenance,
            expected_circuit_sha256=circuit_sha256,
        )
        write_jsonl(args.output, rows)
    except (BenchmarkDataError, OSError) as ex:
        print(ex, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
