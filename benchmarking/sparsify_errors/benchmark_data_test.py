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

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from benchmarking.sparsify_errors.benchmark_data import (
    BenchmarkDataError,
    aggregate_raw_rows,
    enrich_rows,
    read_raw_files,
    read_run_directories,
    validate_aggregate_row,
)


TEST_COMMIT = "0123456789abcdef0123456789abcdef01234567"
TEST_STIM_REVISION = "89abcdef0123456789abcdef0123456789abcdef"
TEST_BINARY_CONTENT = b"fixture tesseract binary\n"
TEST_BINARY_SHA256 = hashlib.sha256(TEST_BINARY_CONTENT).hexdigest()
TEST_HARDWARE = "test CPU"
TEST_SEED_NAMESPACE = 17
TEST_SEED_SCHEME = "run_namespace_times_stride_plus_job_index"
TEST_SEED_STRIDE = 1024


def _run_provenance(*run_ids: str) -> dict:
    selected_run_ids = list(run_ids or ("run-1",))
    return {
        "tesseract_commit": TEST_COMMIT,
        "stim_revision": TEST_STIM_REVISION,
        "hardware_description": TEST_HARDWARE,
        "tesseract_binary_sha256": TEST_BINARY_SHA256,
        "run_ids": selected_run_ids,
        "sample_seed_namespace_by_run": {
            run_id: TEST_SEED_NAMESPACE + index
            for index, run_id in enumerate(selected_run_ids)
        },
        "sample_seed_scheme": TEST_SEED_SCHEME,
        "sample_seed_stride": TEST_SEED_STRIDE,
    }


def _circuit_hashes(repo_root: Path, *paths: str) -> dict[str, str]:
    return {
        path: hashlib.sha256((repo_root / path).read_bytes()).hexdigest()
        for path in paths
    }


def _manifest(repo_root: Path, circuit_path: str, repetitions: int = 1) -> dict:
    return {
        "schema_version": 1,
        "run_id": "run-1",
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "tesseract_commit": TEST_COMMIT,
        "stim_revision": TEST_STIM_REVISION,
        "hardware_description": TEST_HARDWARE,
        "tesseract_binary_sha256": TEST_BINARY_SHA256,
        "git_dirty": False,
        "circuit_sha256": _circuit_hashes(repo_root, circuit_path),
        "expected_job_count": repetitions,
        "sample_seed_namespace": TEST_SEED_NAMESPACE,
        "sample_seed_scheme": TEST_SEED_SCHEME,
        "sample_seed_stride": TEST_SEED_STRIDE,
        "sweep": {
            "include_baseline": False,
            "repetitions_per_configuration": repetitions,
            "sparsify_base_degree_by_directory": {"surfacecodes": 2},
            "sparsify_max_degree": -1,
            "sparsify_reactivate_limits": [8],
        },
    }


def _write_run_artifacts(
    repo_root: Path, run_directory: Path, circuit_path: str
) -> None:
    artifact_directory = run_directory / "artifacts"
    binary_path = artifact_directory / "tesseract"
    binary_path.parent.mkdir(parents=True)
    binary_path.write_bytes(TEST_BINARY_CONTENT)
    binary_path.chmod(0o555)

    snapshot_path = artifact_directory / "repo" / circuit_path
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_bytes((repo_root / circuit_path).read_bytes())
    snapshot_path.chmod(0o444)


def _write_circuit(repo_root: Path, basis: str) -> str:
    relative = Path(
        "testdata/surfacecodes/"
        f"r=1,d=1,p=0.001,noise=test,c=surface_code_{basis},q=1.stim"
    )
    path = repo_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "X_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
        encoding="utf-8",
    )
    return str(relative)


def _aggregate_row(circuit_path: str) -> dict:
    return {
        "circuit_path": circuit_path,
        "dem_path": "",
        "det_beam": 20,
        "det_order_method": "index",
        "det_order_seed": 123,
        "det_penalty": 0.0,
        "merge_errors": True,
        "no_revisit_dets": True,
        "num_det_orders": 21,
        "pqlimit": 1_000_000,
        "sparsify_base_degree": 2,
        "sparsify_errors": True,
        "sparsify_max_degree": -1,
        "sparsify_reactivate_limit": 8,
        "total_time_seconds": 2.0,
        "num_shots": 100,
        "num_low_confidence": 1,
        "num_errors": 2,
    }


def _raw_row(circuit_path: str, seed: int) -> dict:
    return {
        **_aggregate_row(circuit_path),
        "beam_climbing": True,
        "max_errors": 10,
        "num_threads": 30,
        "sample_num_shots": 10_000,
        "sample_seed": seed,
        "num_compiled_errors": 1,
        "num_detectors": 1,
        "num_mandatory_errors": 1,
        "num_optional_errors": 0,
        "num_raw_dem_errors": 1,
    }


class BenchmarkDataTest(unittest.TestCase):
    def test_enriches_with_exact_compiled_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            circuit_path = _write_circuit(repo_root, "X")
            row = enrich_rows([_aggregate_row(circuit_path)], repo_root)[0]

        self.assertEqual(row["basis"], "X")
        self.assertEqual(row["num_detectors"], 1)
        self.assertEqual(row["num_raw_dem_errors"], 1)
        self.assertEqual(row["num_compiled_errors"], 1)
        self.assertEqual(row["num_mandatory_errors"], 1)
        self.assertEqual(row["num_optional_errors"], 0)
        self.assertEqual(len(row["circuit_sha256"]), 64)

    def test_missing_metadata_error_names_source_and_circuit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            circuit_path = _write_circuit(repo_root, "X")
            row = enrich_rows([_aggregate_row(circuit_path)], repo_root)[0]
        del row["num_detectors"]

        with self.assertRaisesRegex(
            BenchmarkDataError,
            r"fixture.jsonl:7.*surface_code_X.*num_detectors",
        ):
            validate_aggregate_row(row, "fixture.jsonl:7")

    def test_aggregation_sums_repetitions_without_pooling_basis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            x_path = _write_circuit(repo_root, "X")
            z_path = _write_circuit(repo_root, "Z")
            x1 = _raw_row(x_path, 11)
            x2 = copy.deepcopy(x1)
            x2.update(
                sample_seed=12,
                num_errors=3,
                num_shots=80,
                total_time_seconds=1.5,
            )
            z1 = _raw_row(z_path, 13)

            rows = aggregate_raw_rows(
                [
                    ("x1.json", x1, "run-1"),
                    ("x2.json", x2, "run-1"),
                    ("z1.json", z1, "run-1"),
                ],
                run_provenance=_run_provenance(),
                expected_circuit_sha256=_circuit_hashes(repo_root, x_path, z_path),
            )

        self.assertEqual(len(rows), 2)
        x_aggregate = next(row for row in rows if row["basis"] == "X")
        self.assertEqual(x_aggregate["num_jobs"], 2)
        self.assertEqual(x_aggregate["num_errors"], 5)
        self.assertEqual(x_aggregate["num_shots"], 180)
        self.assertEqual(x_aggregate["total_time_seconds"], 3.5)
        self.assertEqual(x_aggregate["benchmark_tesseract_commit"], TEST_COMMIT)
        self.assertEqual(x_aggregate["benchmark_hardware"], TEST_HARDWARE)
        self.assertEqual(x_aggregate["benchmark_stim_revision"], TEST_STIM_REVISION)
        self.assertEqual(x_aggregate["benchmark_run_ids"], ["run-1"])
        self.assertEqual(
            x_aggregate["model_metadata_source"], "benchmark_runtime_stats"
        )
        self.assertEqual(x_aggregate["num_compiled_errors"], 1)

    def test_aggregation_rejects_changed_fixed_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            circuit_path = _write_circuit(repo_root, "X")
            first = _raw_row(circuit_path, 11)
            second = _raw_row(circuit_path, 12)
            second["num_threads"] = 12

            with self.assertRaisesRegex(BenchmarkDataError, "num_threads"):
                aggregate_raw_rows(
                    [("first", first, "run-1"), ("second", second, "run-1")],
                    run_provenance=_run_provenance(),
                    expected_circuit_sha256=_circuit_hashes(repo_root, circuit_path),
                )

    def test_aggregation_attributes_only_contributing_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            circuit_path = _write_circuit(repo_root, "X")
            row = _raw_row(circuit_path, 11)

            rows = aggregate_raw_rows(
                [("job.json", row, "run-2")],
                run_provenance=_run_provenance("run-1", "run-2"),
                expected_circuit_sha256=_circuit_hashes(repo_root, circuit_path),
            )

        self.assertEqual(rows[0]["benchmark_run_ids"], ["run-2"])
        self.assertEqual(rows[0]["benchmark_run_count"], 1)

    def test_aggregation_rejects_duplicate_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            circuit_path = _write_circuit(repo_root, "X")
            first = _raw_row(circuit_path, 11)
            second = copy.deepcopy(first)

            with self.assertRaisesRegex(BenchmarkDataError, "duplicate sample_seed"):
                aggregate_raw_rows(
                    [("first", first, "run-1"), ("second", second, "run-1")],
                    run_provenance=_run_provenance(),
                    expected_circuit_sha256=_circuit_hashes(repo_root, circuit_path),
                )

    def test_enrichment_does_not_coerce_merge_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            circuit_path = _write_circuit(repo_root, "X")
            row = _aggregate_row(circuit_path)
            row["merge_errors"] = "false"

            with self.assertRaisesRegex(BenchmarkDataError, "merge_errors.*bool"):
                enrich_rows([row], repo_root)

    def test_nonstandard_json_number_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "job.json"
            path.write_text('{"total_time_seconds":NaN}\n', encoding="utf-8")

            with self.assertRaisesRegex(BenchmarkDataError, "non-standard"):
                read_raw_files([path])

    def test_run_directory_supplies_submission_time_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            circuit_path = _write_circuit(repo_root, "X")
            run_directory = repo_root / "run-1"
            jobs_directory = run_directory / "jobs"
            jobs_directory.mkdir(parents=True)
            _write_run_artifacts(repo_root, run_directory, circuit_path)
            (jobs_directory / "0.json").write_text(
                json.dumps(
                    _raw_row(circuit_path, TEST_SEED_NAMESPACE * TEST_SEED_STRIDE)
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = _manifest(repo_root, circuit_path)
            (run_directory / "manifest.json").write_text(
                json.dumps(manifest) + "\n", encoding="utf-8"
            )

            rows, provenance, circuit_hashes = read_run_directories([run_directory])

        self.assertEqual(len(rows), 1)
        self.assertEqual(provenance["run_ids"], ["run-1"])
        self.assertEqual(provenance["tesseract_commit"], TEST_COMMIT)
        self.assertEqual(circuit_hashes, manifest["circuit_sha256"])
        self.assertEqual(rows[0][2], "run-1")

    def test_run_directory_rejects_incomplete_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            circuit_path = _write_circuit(repo_root, "X")
            run_directory = repo_root / "run-1"
            jobs_directory = run_directory / "jobs"
            jobs_directory.mkdir(parents=True)
            _write_run_artifacts(repo_root, run_directory, circuit_path)
            (jobs_directory / "0.json").write_text(
                json.dumps(
                    _raw_row(circuit_path, TEST_SEED_NAMESPACE * TEST_SEED_STRIDE)
                )
                + "\n",
                encoding="utf-8",
            )
            (run_directory / "manifest.json").write_text(
                json.dumps(_manifest(repo_root, circuit_path, repetitions=2)) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                BenchmarkDataError, "complete contiguous range"
            ):
                read_run_directories([run_directory])

    def test_run_directory_rejects_wrong_sweep_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            circuit_path = _write_circuit(repo_root, "X")
            run_directory = repo_root / "run-1"
            jobs_directory = run_directory / "jobs"
            jobs_directory.mkdir(parents=True)
            _write_run_artifacts(repo_root, run_directory, circuit_path)
            row = _raw_row(circuit_path, TEST_SEED_NAMESPACE * TEST_SEED_STRIDE)
            row["sparsify_reactivate_limit"] = 4
            (jobs_directory / "0.json").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )
            (run_directory / "manifest.json").write_text(
                json.dumps(_manifest(repo_root, circuit_path)) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(BenchmarkDataError, "incomplete sweep grid"):
                read_run_directories([run_directory])

    def test_run_directory_rejects_writable_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            circuit_path = _write_circuit(repo_root, "X")
            run_directory = repo_root / "run-1"
            jobs_directory = run_directory / "jobs"
            jobs_directory.mkdir(parents=True)
            _write_run_artifacts(repo_root, run_directory, circuit_path)
            snapshot_path = run_directory / "artifacts" / "repo" / circuit_path
            snapshot_path.chmod(0o644)
            (jobs_directory / "0.json").write_text(
                json.dumps(
                    _raw_row(circuit_path, TEST_SEED_NAMESPACE * TEST_SEED_STRIDE)
                )
                + "\n",
                encoding="utf-8",
            )
            (run_directory / "manifest.json").write_text(
                json.dumps(_manifest(repo_root, circuit_path)) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(BenchmarkDataError, "must be read-only"):
                read_run_directories([run_directory])

    def test_run_directory_rejects_seed_outside_manifest_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            circuit_path = _write_circuit(repo_root, "X")
            run_directory = repo_root / "run-1"
            jobs_directory = run_directory / "jobs"
            jobs_directory.mkdir(parents=True)
            _write_run_artifacts(repo_root, run_directory, circuit_path)
            (jobs_directory / "0.json").write_text(
                json.dumps(_raw_row(circuit_path, 123)) + "\n", encoding="utf-8"
            )
            (run_directory / "manifest.json").write_text(
                json.dumps(_manifest(repo_root, circuit_path)) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(BenchmarkDataError, "manifest seed scheme"):
                read_run_directories([run_directory])


if __name__ == "__main__":
    unittest.main()
