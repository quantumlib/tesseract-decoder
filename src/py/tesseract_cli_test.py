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


def _run_dem(tmp_path, dem_text: str, *extra_args: str) -> subprocess.CompletedProcess:
    dem_path = tmp_path / "model.dem"
    shots_path = tmp_path / "shots.b8"
    dem_path.write_text(dem_text)
    shots_path.write_bytes(b"\0")
    return subprocess.run(
        [
            str(_tesseract_binary()),
            "--dem",
            str(dem_path),
            "--in",
            str(shots_path),
            "--in-format",
            "b8",
            "--threads",
            "1",
            "--multipass",
            *extra_args,
        ],
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


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (("--num-passes", "0"), "--num-passes must be 1 or 2"),
        (("--num-passes", "3"), "--num-passes must be 1 or 2"),
        (("--multipass-strategy", "invalid"), "Invalid --multipass-strategy"),
        (("--dem-out", "unused.dem"), "--dem-out is not supported"),
    ],
)
def test_multipass_cli_validation_is_in_normal_argument_path(
    tmp_path, extra_args, message
):
    result = _run_dem(tmp_path, CANONICAL_DEM, *extra_args)
    assert result.returncode != 0
    assert message in result.stderr
    assert "usage:" in result.stderr.lower()
