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

import os
import pathlib
import subprocess

import pytest


def _tesseract_binary() -> pathlib.Path:
    binary = (
        pathlib.Path(os.environ["TEST_SRCDIR"])
        / os.environ["TEST_WORKSPACE"]
        / "src"
        / "tesseract"
    )
    if not binary.exists() and binary.with_suffix(".exe").exists():
        return binary.with_suffix(".exe")
    return binary


def _run_dem(
    tmp_path,
    dem_text: str,
    *extra_args: str,
    shot_data: bytes = b"\0",
    multipass: bool = True,
) -> subprocess.CompletedProcess:
    dem_path = tmp_path / "model.dem"
    shots_path = tmp_path / "shots.b8"
    dem_path.write_text(dem_text)
    shots_path.write_bytes(shot_data)
    command = [
        str(_tesseract_binary()),
        "--dem",
        str(dem_path),
        "--in",
        str(shots_path),
        "--in-format",
        "b8",
        "--threads",
        "1",
    ]
    if multipass:
        command.append("--multipass")
    command.extend(extra_args)
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )


CANONICAL_DEM = r"""
    error(0.1) D0
    error(0.2) D1 L0
    detector[{"measure_basis":"invalid","basis":"X","unrelated":5}] D0
    detector[{"basis":"Z"}] D1
"""


def test_cli_accepts_only_canonical_top_level_basis_tags(tmp_path):
    result = _run_dem(tmp_path, CANONICAL_DEM)
    assert result.returncode == 0, result.stderr
    assert "num_shots = 1" in result.stdout


def test_cli_still_requires_exactly_two_components(tmp_path):
    result = _run_dem(
        tmp_path,
        r"""
        error(0.1) D0
        error(0.2) D1 L0
        detector[{"basis":"X"}] D0
        detector[{"basis":"X"}] D1
        """,
    )
    assert result.returncode != 0
    assert "requires exactly 2 detector components" in result.stderr


@pytest.mark.parametrize(
    ("detector_lines", "message"),
    [
        (
            'detector[{"measure_basis":"X"}](0,0,0,0) D0\n'
            'detector[{"measure_basis":"Z"}](0,0,0,3) D1',
            "top-level basis",
        ),
        (
            'detector[{"md":{"basis":"X"}}] D0\ndetector[{"md":{"basis":"Z"}}] D1',
            "top-level basis",
        ),
        (
            "detector(0,0,0,0) D0\ndetector(0,0,0,3) D1",
            "requires a top-level JSON basis",
        ),
        (
            'detector[{"basis":"Y"}] D0\ndetector[{"basis":"Z"}] D1',
            "invalid top-level basis",
        ),
        ('detector[not-json] D0\ndetector[{"basis":"Z"}] D1', "non-JSON tag"),
    ],
)
def test_cli_rejects_legacy_coordinate_and_malformed_annotations(
    tmp_path, detector_lines, message
):
    result = _run_dem(
        tmp_path,
        "error(0.1) D0\nerror(0.2) D1 L0\n" + detector_lines,
    )
    assert result.returncode != 0
    assert message in result.stderr


def test_tagged_stim_circuit_survives_circuit_to_dem_end_to_end(tmp_path):
    circuit_path = tmp_path / "tagged.stim"
    circuit_path.write_text(
        r"""
        R 0 1
        X_ERROR(0.1) 0
        X_ERROR(0.2) 1
        M 0 1
        DETECTOR[{"basis":"X"}] rec[-2]
        DETECTOR[{"basis":"Z"}] rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """
    )
    result = subprocess.run(
        [
            str(_tesseract_binary()),
            "--circuit",
            str(circuit_path),
            "--sample-num-shots",
            "3",
            "--sample-seed",
            "1",
            "--threads",
            "1",
            "--multipass",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "num_shots = 3" in result.stdout


@pytest.mark.parametrize("extra_args", [(), ("--print-multipass-plan",)])
def test_multipass_construction_failure_does_not_depend_on_plan(tmp_path, extra_args):
    result = _run_dem(
        tmp_path,
        r"""
        error[bad decomposition](0.1) D0 D1 ^ D0
        detector[{"basis":"X"}] D0
        detector[{"basis":"Z"}] D1
        """,
        *extra_args,
    )
    assert result.returncode != 0
    assert "detectors from multiple components" in result.stderr
    assert "bad decomposition" in result.stderr
    assert "num_shots" not in result.stdout


def test_multipass_cli_derives_bfs_orders_from_each_component(tmp_path):
    result = _run_dem(
        tmp_path,
        r"""
        error(0.02) D0 D1 L0
        error(0.1) D3 D4 D5 D6 L1
        error(0.02) D3 D4 D6
        error(0.3) D4 D5
        detector[{"basis":"X"}] D0
        detector[{"basis":"X"}] D1
        detector[{"basis":"X"}] D2
        detector[{"basis":"Z"}] D3
        detector[{"basis":"Z"}] D4
        detector[{"basis":"Z"}] D5
        detector[{"basis":"Z"}] D6
        logical_observable L0
        logical_observable L1
        """,
        "--num-passes",
        "1",
        "--multipass-strategy",
        "static",
        "--det-order-bfs",
        "--num-det-orders",
        "1",
        "--det-order-seed",
        "1",
        "--beam",
        "0",
        "--no-revisit-dets",
        shot_data=bytes([0b01101000]),
    )
    assert result.returncode == 0, result.stderr
    assert "num_shots = 1 num_low_confidence = 0" in result.stdout


def test_monolithic_decoder_result_still_populates_dem_error_counts(tmp_path):
    dem_out = tmp_path / "estimated.dem"
    result = _run_dem(
        tmp_path,
        "error(0.4) D0 L0\ndetector D0\nlogical_observable L0\n",
        "--dem-out",
        str(dem_out),
        shot_data=b"\x01",
        multipass=False,
    )
    assert result.returncode == 0, result.stderr
    assert "error(1) D0 L0" in dem_out.read_text()


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (("--num-passes", "0"), "--num-passes must be 1 or 2"),
        (("--num-passes", "3"), "--num-passes must be 1 or 2"),
        (("--multipass-strategy", "invalid"), "Invalid --multipass-strategy"),
        (("--dem-out", "unused.dem"), "--dem-out is not supported"),
    ],
)
def test_multipass_cli_validation(tmp_path, extra_args, message):
    result = _run_dem(tmp_path, CANONICAL_DEM, *extra_args)
    assert result.returncode != 0
    assert message in result.stderr
