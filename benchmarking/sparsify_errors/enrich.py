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

"""Reconstructs self-contained circuit metadata for an existing aggregate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmarking.sparsify_errors.benchmark_data import (
    BenchmarkDataError,
    canonical_jsonl,
    enrich_rows,
    read_jsonl,
    write_text_atomic,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "aggregated_results.jsonl")
    parser.add_argument(
        "--output", type=Path, default=HERE / "aggregated_results.jsonl"
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--assume-merge-errors",
        action="store_true",
        help="Explicitly assume merge_errors=true for legacy rows that lack it.",
    )
    parser.add_argument(
        "--assume-det-order-method",
        choices=("bfs", "coordinate", "index"),
        help="Explicit detector-order assumption for legacy rows that lack it.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of writing when the enriched canonical output differs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = read_jsonl(args.input, require_metadata=False)
        output = canonical_jsonl(
            enrich_rows(
                rows,
                args.repo_root,
                legacy_merge_errors=True if args.assume_merge_errors else None,
                legacy_det_order_method=args.assume_det_order_method,
            )
        )
    except (BenchmarkDataError, OSError) as ex:
        print(ex, file=sys.stderr)
        return 1

    if args.check:
        try:
            current = args.output.read_text(encoding="utf-8")
        except OSError as ex:
            print(ex, file=sys.stderr)
            return 1
        if current != output:
            print(
                f"{args.output} is not canonically enriched; run {Path(__file__).name}",
                file=sys.stderr,
            )
            return 1
        return 0

    write_text_atomic(args.output, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
