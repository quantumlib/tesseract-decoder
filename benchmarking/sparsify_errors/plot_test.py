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

import math
import os
import tempfile
import unittest
from pathlib import Path


os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="tesseract-mpl-"))

from benchmarking.sparsify_errors.benchmark_data import BenchmarkDataError  # noqa: E402
from benchmarking.sparsify_errors.plot import (  # noqa: E402
    _group_exact_circuits,
    _marker_map,
    _observed_segments,
    _relative_risk_interval,
    _zero_failure_upper_bound,
    compute_metrics,
    extract_fit_data,
    plot_ler_vs_time,
)


def _row(
    basis: str,
    *,
    failures: int = 2,
    shots: int = 1000,
    reactivate_limit: int = 8,
    sparsify_errors: bool = True,
) -> dict:
    return {
        "basis": basis,
        "circuit_path": (
            "testdata/surfacecodes/"
            f"r=5,d=3,p=0.001,noise=test,c=surface_code_{basis},q=17.stim"
        ),
        "code_family": "surfacecodes",
        "circuit_sha256": "0" * 64,
        "dem_path": "",
        "det_beam": 20,
        "det_order_method": "index",
        "det_order_seed": 123,
        "det_penalty": 0.0,
        "distance": 3,
        "merge_errors": True,
        "no_revisit_dets": True,
        "num_compiled_errors": 100,
        "num_detectors": 20,
        "num_det_orders": 21,
        "num_errors": failures,
        "num_low_confidence": 0,
        "num_optional_errors": 60 if sparsify_errors else None,
        "num_qubits": 17,
        "num_shots": shots,
        "physical_error_rate": 0.001,
        "pqlimit": 1_000_000,
        "rounds": 5,
        "sparsify_base_degree": 2 if sparsify_errors else -1,
        "sparsify_errors": sparsify_errors,
        "sparsify_max_degree": -1,
        "sparsify_reactivate_limit": reactivate_limit,
        "total_time_seconds": 10.0,
    }


class PlotTest(unittest.TestCase):
    def test_compute_metrics_keeps_basis_specific_circuits_separate(self) -> None:
        metrics = compute_metrics([_row("X"), _row("Z")])

        self.assertEqual(len(metrics), 2)
        self.assertEqual(len(_group_exact_circuits(metrics)), 2)
        self.assertEqual({metric["basis"] for metric in metrics}, {"X", "Z"})
        self.assertEqual(
            set(extract_fit_data(metrics, 0.001)),
            {("surfacecodes", "X"), ("surfacecodes", "Z")},
        )

    def test_compute_metrics_rejects_mixed_decoder_configuration(self) -> None:
        changed = _row("Z")
        changed["det_beam"] = 21

        with self.assertRaisesRegex(BenchmarkDataError, "det_beam"):
            compute_metrics([_row("X"), changed])

    def test_zero_failures_are_upper_limits_and_excluded_from_fits(self) -> None:
        baseline = _row("X", failures=4, reactivate_limit=0, sparsify_errors=False)
        censored = _row("X", failures=0, reactivate_limit=8)
        metrics = compute_metrics([baseline, censored])

        upper_limit = next(metric for metric in metrics if metric["is_upper_limit"])
        self.assertAlmostEqual(
            upper_limit["ler"],
            _zero_failure_upper_bound(censored["num_shots"]) / censored["rounds"],
        )

        fit_data = extract_fit_data(metrics, 0.001)[("surfacecodes", "X")]
        self.assertEqual(len(fit_data["min_ler"]), 1)
        self.assertAlmostEqual(fit_data["min_ler"][0], 4 / 1000 / 5)
        self.assertEqual(list(_observed_segments(metrics)), [])

    def test_markers_are_unique_within_a_code_family(self) -> None:
        identities = [
            {"type": "surfacecodes", "d": distance, "q": distance**2, "basis": basis}
            for distance in range(3, 8)
            for basis in ("X", "Z")
        ]

        marker_map = _marker_map(identities)
        self.assertEqual(len(marker_map), 10)
        self.assertEqual(len(set(marker_map.values())), 10)

    def test_relative_risk_uses_corrected_totals(self) -> None:
        low, high = _relative_risk_interval(0, 100, 1, 200)
        expected_center = (1.5 / 201) / (0.5 / 101)

        self.assertAlmostEqual(math.sqrt(low * high), expected_center)

    def test_plot_writes_pdf_only(self) -> None:
        metrics = compute_metrics([_row("X"), _row("Z", failures=0)])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failure-rate.pdf"
            plot_ler_vs_time(metrics, 0.001, output, "Fixture")

            self.assertTrue(output.is_file())
            self.assertFalse(output.with_suffix(".png").exists())


if __name__ == "__main__":
    unittest.main()
