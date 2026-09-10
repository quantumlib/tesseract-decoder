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

import math
import unittest
from unittest import mock

from benchmarking.sparsify_errors import benchmark_data
from benchmarking.sparsify_errors import plot


CIRCUIT_PATH = "testdata/surfacecodes/example.stim"


def aggregate_row(
    reactivate_limit: int,
    *,
    sparsify_errors: bool = True,
    num_errors: int = 10,
    total_time_seconds: float = 1.0,
) -> dict:
    if sparsify_errors:
        base_degree = 2
        max_degree = -1
        mandatory_errors = 1
        optional_errors = 1
    else:
        base_degree = -1
        max_degree = -1
        reactivate_limit = -1
        mandatory_errors = None
        optional_errors = None
    return {
        "basis": "X",
        "circuit_path": CIRCUIT_PATH,
        "circuit_sha256": "d" * 64,
        "code_family": "surfacecodes",
        "dem_path": "",
        "det_beam": 20,
        "det_order_method": "index",
        "det_order_seed": 5,
        "det_penalty": 0.0,
        "distance": 3,
        "merge_errors": True,
        "no_revisit_dets": True,
        "num_compiled_errors": 2,
        "num_detectors": 2,
        "num_det_orders": 1,
        "num_errors": num_errors,
        "num_low_confidence": 0,
        "num_mandatory_errors": mandatory_errors,
        "num_optional_errors": optional_errors,
        "num_qubits": 3,
        "num_raw_dem_errors": 2,
        "num_shots": 1000,
        "physical_error_rate": 0.001,
        "pqlimit": 100,
        "rounds": 3,
        "sparsify_base_degree": base_degree,
        "sparsify_errors": sparsify_errors,
        "sparsify_max_degree": max_degree,
        "sparsify_reactivate_limit": reactivate_limit,
        "total_time_seconds": total_time_seconds,
    }


def raw_row(
    reactivate_limit: int, sample_seed: int, *, sparsify_errors: bool = True
) -> dict:
    row = aggregate_row(reactivate_limit, sparsify_errors=sparsify_errors)
    for field in benchmark_data.METADATA_TYPES:
        row.pop(field, None)
    row.update(
        {
            "beam_climbing": True,
            "det_order_method": "index",
            "max_errors": 10,
            "merge_errors": True,
            "num_compiled_errors": 2,
            "num_detectors": 2,
            "num_raw_dem_errors": 2,
            "num_threads": 1,
            "sample_num_shots": 1000,
            "sample_seed": sample_seed,
        }
    )
    return row


def run_manifest(reactivate_limits: list[int]) -> dict:
    expected_job_count = len(reactivate_limits) + 1
    return {
        "schema_version": 1,
        "run_id": "auto-limit-test",
        "created_at_utc": "2026-09-04T00:00:00+00:00",
        "tesseract_commit": "a" * 40,
        "stim_revision": "b" * 40,
        "hardware_description": "test machine",
        "tesseract_binary_sha256": "c" * 64,
        "git_dirty": False,
        "det_order_method": "index",
        "merge_errors": True,
        "circuit_sha256": {CIRCUIT_PATH: "d" * 64},
        "expected_job_count": expected_job_count,
        "sample_seed_namespace": 0,
        "sample_seed_scheme": benchmark_data.SUPPORTED_SAMPLE_SEED_SCHEME,
        "sample_seed_stride": expected_job_count,
        "sweep": {
            "include_baseline": True,
            "repetitions_per_configuration": 1,
            "sparsify_base_degree_by_directory": {"surfacecodes": 2},
            "sparsify_max_degree": -1,
            "sparsify_reactivate_limits": reactivate_limits,
        },
    }


class BenchmarkDataAutoLimitTest(unittest.TestCase):
    def test_rows_accept_only_auto_or_nonnegative_limits(self) -> None:
        benchmark_data.validate_aggregate_row(aggregate_row(-1), "auto")
        with self.assertRaisesRegex(
            benchmark_data.BenchmarkDataError, "must be -1 or non-negative"
        ):
            benchmark_data.validate_aggregate_row(aggregate_row(-2), "invalid")

    def test_manifest_coverage_distinguishes_auto_from_baseline(self) -> None:
        manifest = run_manifest([-1, 0])
        benchmark_data._validate_run_manifest(manifest, "manifest")
        rows = [
            ("auto", raw_row(-1, 0)),
            ("explicit", raw_row(0, 1)),
            ("baseline", raw_row(-1, 2, sparsify_errors=False)),
        ]
        benchmark_data._validate_run_coverage(manifest, rows, "run")

    def test_manifest_rejects_values_below_auto_sentinel(self) -> None:
        with self.assertRaisesRegex(
            benchmark_data.BenchmarkDataError, "only -1 or non-negative"
        ):
            benchmark_data._validate_run_manifest(run_manifest([-2]), "manifest")


class PlotAutoLimitTest(unittest.TestCase):
    def test_auto_limit_becomes_nonnumeric_metric(self) -> None:
        metrics = plot.compute_metrics(
            [
                aggregate_row(-1),
                aggregate_row(4),
                aggregate_row(-1, sparsify_errors=False),
            ]
        )
        self.assertEqual(metrics[0]["M"], 4)
        self.assertIsNone(metrics[1]["M"])
        self.assertEqual(metrics[2]["M"], float("inf"))

    def test_auto_limit_is_not_connected_or_interpolated(self) -> None:
        explicit_two = {"M": 2, "ler": 0.2, "is_upper_limit": False}
        automatic = {"M": None, "ler": 0.001, "is_upper_limit": False}
        explicit_four = {"M": 4, "ler": 0.1, "is_upper_limit": False}
        baseline = {"M": float("inf"), "ler": 0.05, "is_upper_limit": False}

        segments = list(
            plot._observed_segments([explicit_two, explicit_four, automatic, baseline])
        )
        self.assertEqual(
            segments,
            [(explicit_two, explicit_four), (explicit_four, baseline)],
        )
        self.assertTrue(
            math.isnan(plot.interpolate_required_M([automatic], target_ler=0.1))
        )
        self.assertAlmostEqual(
            plot.interpolate_required_M(
                [automatic, explicit_two, explicit_four], target_ler=0.15
            ),
            math.sqrt(8),
        )

    def test_actual_auto_run_is_preferred_for_heuristic_comparisons(self) -> None:
        automatic = {"M": None}
        explicit_four = {"M": 4}
        explicit_sixteen = {"M": 16}
        self.assertIs(
            plot._automatic_or_nearest_explicit_M(
                [explicit_four, automatic, explicit_sixteen], 13
            ),
            automatic,
        )
        self.assertIs(
            plot._automatic_or_nearest_explicit_M(
                [explicit_four, explicit_sixteen], 13
            ),
            explicit_sixteen,
        )

    def test_auto_run_does_not_change_numeric_M_fit(self) -> None:
        metrics = plot.compute_metrics(
            [
                aggregate_row(-1, num_errors=1, total_time_seconds=50),
                aggregate_row(4, num_errors=100, total_time_seconds=200),
                aggregate_row(8, num_errors=50, total_time_seconds=300),
                aggregate_row(
                    -1,
                    sparsify_errors=False,
                    num_errors=25,
                    total_time_seconds=400,
                ),
            ]
        )
        fit_data = plot.extract_fit_data(metrics, 0.001)[("surfacecodes", "X")]
        self.assertEqual(fit_data["min_ler"], [25 / 1000 / 3])

    def test_numeric_M_analyses_omit_auto_only_data(self) -> None:
        automatic = plot.compute_metrics([aggregate_row(-1)])
        self.assertEqual(plot.extract_fit_data(automatic, 0.001), {})
        with mock.patch.object(plot, "_save_figure") as save_figure:
            plot.plot_stacked_ler_vs_M(automatic, 0.001, "unused.pdf", "numeric M")
            plot.plot_mq_scaling_meta_analysis(
                automatic, 0.001, "unused.pdf", "numeric M / E"
            )
        save_figure.assert_not_called()


if __name__ == "__main__":
    unittest.main()
