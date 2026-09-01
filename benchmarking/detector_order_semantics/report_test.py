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

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from benchmarking.detector_order_semantics import report  # noqa: E402


class ReportTest(unittest.TestCase):
    def setUp(self) -> None:
        fixture = HERE / "reference/pr_claims_raw.jsonl"
        self.records = [
            report.validate_raw_record(json.loads(line), f"fixture:{index}")
            for index, line in enumerate(fixture.read_text().splitlines(), start=1)
        ]
        self.rows = report.make_comparisons(self.records)

    def test_claimed_counts_and_statistics(self) -> None:
        self.assertEqual(len(self.rows), 6)
        color_coordinate = next(
            row
            for row in self.rows
            if row["case_slug"] == "color-d7" and row["order"] == "coordinate"
        )
        self.assertEqual(color_coordinate["baseline_failures"], 941)
        self.assertEqual(color_coordinate["candidate_failures"], 803)
        self.assertAlmostEqual(
            color_coordinate["relative_error_reduction"], 1 - 803 / 941, places=12
        )
        self.assertAlmostEqual(
            color_coordinate["fisher_exact_two_sided_p"], 0.0009790, places=7
        )
        self.assertAlmostEqual(
            color_coordinate["summed_decode_time_speedup"],
            38533.0 / 33777.1,
            places=12,
        )

    def test_outputs_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            report.write_outputs(self.rows, Path(first))
            report.write_outputs(self.rows, Path(second))
            for name in ("results.jsonl", "results.md", "comparison.svg"):
                self.assertEqual(
                    (Path(first) / name).read_bytes(),
                    (Path(second) / name).read_bytes(),
                )

    def test_markdown_labels_timing_and_test(self) -> None:
        text = report.markdown(self.rows)
        self.assertIn("summed per-shot decode time", text)
        self.assertIn("Fisher p (independent)", text)
        self.assertNotIn("Baseline CPU", text)


if __name__ == "__main__":
    unittest.main()
