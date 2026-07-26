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

import json
from pathlib import Path
import subprocess
import sys

import pytest


_DECODERS = tuple(Path(argument) for argument in sys.argv[1:])
_SCHEMA = "tesseract.gari_layout.v1"


@pytest.fixture
def gari_files(tmp_path):
    circuit = tmp_path / "source.stim"
    circuit.write_text(
        """\
X_ERROR(1) 0 3
M 0 1 2 3
DETECTOR rec[-4]
DETECTOR rec[-3]
DETECTOR rec[-2]
DETECTOR rec[-1]
OBSERVABLE_INCLUDE(0) rec[-4]
""",
        encoding="utf-8",
    )
    dem = tmp_path / "target.dem"
    dem.write_text(
        """\
error(0.1) D0
error(0.1) D1 L0
error(0.1) D2
error(0.1) D3
error(0.1) D4
error(0.1) D5
""",
        encoding="utf-8",
    )
    shots = tmp_path / "source.01"
    shots.write_text("1001\n", encoding="utf-8")
    layout = tmp_path / "target-layout.json"
    layout_data = {
        "schema": _SCHEMA,
        "source_detector_count": 4,
        "gari_detector_count": 6,
        "source_to_gari": [2, 0, 3, 1],
    }
    layout.write_text(json.dumps(layout_data), encoding="utf-8")
    return {
        "circuit": circuit,
        "dem": dem,
        "shots": shots,
        "layout": layout,
        "layout_data": layout_data,
        "tmp_path": tmp_path,
    }


@pytest.fixture(params=_DECODERS, ids=lambda decoder: decoder.name)
def decoder(request):
    return request.param


def _run_decoder(decoder, *args):
    decoder_args = [decoder, "--threads", "1"]
    if decoder.name == "tesseract":
        decoder_args.extend(["--num-det-orders", "1", "--det-order-index"])
    return subprocess.run(
        [*decoder_args, *map(str, args)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _decode(decoder, gari_files, *source_args):
    output = gari_files["tmp_path"] / "predictions.01"
    result = _run_decoder(
        decoder,
        "--dem",
        gari_files["dem"],
        "--gari-layout",
        gari_files["layout"],
        *source_args,
        "--out",
        output,
        "--out-format",
        "01",
    )
    assert result.returncode == 0, result.stderr
    return output.read_text(encoding="utf-8")


def _write_layout(gari_files, data):
    path = gari_files["tmp_path"] / "invalid-layout.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _assert_failure(result, expected, path=None):
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert expected in output
    if path is not None:
        assert str(path) in output


def test_decoders_map_sampled_and_01_shots(decoder, gari_files):
    assert _decode(
        decoder,
        gari_files,
        "--circuit",
        gari_files["circuit"],
        "--sample-num-shots",
        "1",
        "--sample-seed",
        "0",
    ) == "1\n"
    assert _decode(
        decoder,
        gari_files,
        "--in",
        gari_files["shots"],
        "--in-format",
        "01",
    ) == "1\n"


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"schema": "wrong.schema"}, "field 'schema'"),
        ({"source_to_gari": [2, 0, 3]}, "must contain 4 entries"),
        ({"source_to_gari": [2, 0, 6, 1]}, "outside the target range"),
        ({"source_to_gari": [2, 0, 2, 1]}, "injective mapping"),
        ({"gari_detector_count": 7}, "but DEM"),
    ],
)
def test_decoders_reject_invalid_layouts(decoder, gari_files, changes, expected):
    data = {**gari_files["layout_data"], **changes}
    layout = _write_layout(gari_files, data)
    result = _run_decoder(
        decoder,
        "--dem",
        gari_files["dem"],
        "--gari-layout",
        layout,
        "--in",
        gari_files["shots"],
        "--in-format",
        "01",
    )
    _assert_failure(result, expected, layout)


def test_decoders_reject_source_and_no_layout_count_mismatches(
    decoder, gari_files
):
    data = {
        **gari_files["layout_data"],
        "source_detector_count": 3,
        "source_to_gari": [2, 0, 3],
    }
    layout = _write_layout(gari_files, data)
    result = _run_decoder(
        decoder,
        "--circuit",
        gari_files["circuit"],
        "--dem",
        gari_files["dem"],
        "--gari-layout",
        layout,
        "--sample-num-shots",
        "1",
    )
    _assert_failure(result, "but circuit", layout)

    result = _run_decoder(
        decoder,
        "--circuit",
        gari_files["circuit"],
        "--dem",
        gari_files["dem"],
        "--sample-num-shots",
        "1",
    )
    _assert_failure(result, "Supply --gari-layout")


def test_decoders_require_dem_with_layout(decoder, gari_files):
    result = _run_decoder(
        decoder,
        "--circuit",
        gari_files["circuit"],
        "--gari-layout",
        gari_files["layout"],
        "--sample-num-shots",
        "1",
    )
    _assert_failure(result, "--gari-layout requires --dem")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
