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
M 0 1 2 3 4 5
DETECTOR rec[-6]
DETECTOR rec[-5]
DETECTOR rec[-4]
DETECTOR rec[-3]
OBSERVABLE_INCLUDE(0) rec[-6]
OBSERVABLE_INCLUDE(1) rec[-2]
OBSERVABLE_INCLUDE(2) rec[-1]
""",
        encoding="utf-8",
    )
    dem = tmp_path / "target.dem"
    dem.write_text(
        """\
error(0.1) D1 D2 L0
error(0.1) D0 D3
error(0.1) D4 L1
error(0.1) D5 L2
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


def _write_layout(gari_files, data, name="invalid-layout.json"):
    path = gari_files["tmp_path"] / name
    contents = data if isinstance(data, str) else json.dumps(data)
    path.write_text(contents, encoding="utf-8")
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
    ) == "100\n"
    assert _decode(
        decoder,
        gari_files,
        "--in",
        gari_files["shots"],
        "--in-format",
        "01",
    ) == "100\n"


def test_decoders_reject_invalid_layouts_and_counts(decoder, gari_files):
    base = gari_files["layout_data"]
    cases = [
        ("{", "could not parse JSON"),
        ([], "top-level JSON value must be an object"),
        (
            {key: value for key, value in base.items() if key != "schema"},
            "missing required field 'schema'",
        ),
        ({**base, "schema": 1}, "field 'schema' must be a string"),
        ({**base, "schema": "wrong.schema"}, "field 'schema' must be"),
        (
            {
                key: value
                for key, value in base.items()
                if key != "source_to_gari"
            },
            "missing required field 'source_to_gari'",
        ),
        ({**base, "source_detector_count": True}, "must be an integer"),
        ({**base, "gari_detector_count": 2**63}, "is too large"),
        ({**base, "source_to_gari": {}}, "must be an array"),
        ({**base, "source_to_gari": [2, 0, 3.5, 1]}, "source_to_gari[2]"),
        ({**base, "source_to_gari": [2, 0, -1, 1]}, "must be nonnegative"),
        ({**base, "source_to_gari": [2, 0, 3]}, "must contain 4 entries"),
        ({**base, "source_to_gari": [2, 0, 3, 1, 4]}, "must contain 4 entries"),
        ({**base, "source_to_gari": [2, 0, 6, 1]}, "outside the target range"),
        ({**base, "source_to_gari": [2, 0, 2, 1]}, "injective mapping"),
        ({**base, "gari_detector_count": 3}, "must be at least 4"),
        ({**base, "gari_detector_count": 7}, "but DEM"),
    ]
    for index, (data, expected) in enumerate(cases):
        layout = _write_layout(gari_files, data, f"invalid-layout-{index}.json")
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

    source_count_layout = _write_layout(
        gari_files,
        {
            **base,
            "source_detector_count": 3,
            "source_to_gari": [2, 0, 3],
        },
        "source-count-layout.json",
    )
    result = _run_decoder(
        decoder,
        "--circuit",
        gari_files["circuit"],
        "--dem",
        gari_files["dem"],
        "--gari-layout",
        source_count_layout,
        "--sample-num-shots",
        "1",
    )
    _assert_failure(result, "but circuit", source_count_layout)

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


def test_decoders_read_multirecord_01_and_b8(decoder, gari_files):
    records = {
        "01": "1001100\n0110000\n0000000\n1111100\n",
        "b8": bytes([0x19, 0x06, 0x00, 0x1F]),
    }
    for shot_format, contents in records.items():
        shots = gari_files["tmp_path"] / f"source-multiple.{shot_format}"
        if isinstance(contents, bytes):
            shots.write_bytes(contents)
        else:
            shots.write_text(contents, encoding="utf-8")
        output = gari_files["tmp_path"] / f"predictions-{shot_format}.01"
        stats = gari_files["tmp_path"] / f"stats-{shot_format}.json"
        result = _run_decoder(
            decoder,
            "--dem",
            gari_files["dem"],
            "--gari-layout",
            gari_files["layout"],
            "--in",
            shots,
            "--in-format",
            shot_format,
            "--in-includes-appended-observables",
            "--out",
            output,
            "--out-format",
            "01",
            "--stats-out",
            stats,
        )
        assert result.returncode == 0, result.stderr
        assert output.read_text(encoding="utf-8") == "100\n000\n000\n100\n"
        metadata = json.loads(stats.read_text(encoding="utf-8"))
        assert metadata["num_shots"] == 4
        assert metadata["num_errors"] == 0
        assert metadata["gari_layout_path"] == str(gari_files["layout"])
        assert metadata["gari_layout_schema"] == _SCHEMA
        assert metadata["source_detector_count"] == 4
        assert metadata["gari_detector_count"] == 6


def test_decoders_preserve_no_layout_paths_and_document_help(decoder, gari_files):
    ordinary_dem = gari_files["tmp_path"] / "ordinary.dem"
    ordinary_dem.write_text(
        """\
error(0.1) D0 D3 L0
error(0.1) D1 L1
error(0.1) D2 L2
""",
        encoding="utf-8",
    )
    output = gari_files["tmp_path"] / "ordinary-predictions.01"
    result = _run_decoder(
        decoder,
        "--circuit",
        gari_files["circuit"],
        "--dem",
        ordinary_dem,
        "--sample-num-shots",
        "1",
        "--sample-seed",
        "0",
        "--out",
        output,
        "--out-format",
        "01",
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8") == "100\n"

    # Without a circuit or GARI layout, file input is already in the target
    # DEM's detector layout. D1 D2 therefore selects the logical-L0 mechanism.
    target_shots = gari_files["tmp_path"] / "target-layout.01"
    target_shots.write_text("011000\n", encoding="utf-8")
    output = gari_files["tmp_path"] / "target-layout-predictions.01"
    result = _run_decoder(
        decoder,
        "--dem",
        gari_files["dem"],
        "--in",
        target_shots,
        "--in-format",
        "01",
        "--out",
        output,
        "--out-format",
        "01",
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8") == "100\n"

    help_result = _run_decoder(decoder, "--help")
    assert help_result.returncode == 0
    assert "--gari-layout FILE" in help_result.stdout
    assert "target detector layout" in help_result.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
