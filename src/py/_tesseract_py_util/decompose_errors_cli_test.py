# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
from pathlib import Path
import sys

import pytest
import stim

from _tesseract_py_util.decompose_errors import main


def _decomposable_dem() -> stim.DetectorErrorModel:
    return stim.DetectorErrorModel("""
        detector(2, 0, 0) D0
        detector(0, 0, 1) D1
        error(0.1) D0
        error(0.2) D1
        error(0.3) D0 D1
    """)


def _expected_decomposed_dem() -> stim.DetectorErrorModel:
    return stim.DetectorErrorModel("""
        detector(2, 0, 0) D0
        detector(0, 0, 1) D1
        error(0.1) D0
        error(0.2) D1
        error(0.3) D0 ^ D1
    """)


def test_main_reads_stdin_and_writes_stdout(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO(str(_decomposable_dem())))

    exit_code = main(["--method", "last-coordinate-index"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert stim.DetectorErrorModel(captured.out) == _expected_decomposed_dem()


def test_main_reads_and_writes_files(tmp_path: Path):
    input_path = tmp_path / "input.dem"
    output_path = tmp_path / "output.dem"
    _decomposable_dem().to_file(input_path)

    exit_code = main([str(input_path), "--out", str(output_path)])

    assert exit_code == 0
    assert stim.DetectorErrorModel.from_file(output_path) == _expected_decomposed_dem()


def test_main_forwards_strip_undecomposable_errors(monkeypatch, capsys):
    dem = stim.DetectorErrorModel("""
        detector(0) D0
        detector(1) D1
        error(0.1) D0 D1
        error(0.1) D0
    """)
    monkeypatch.setattr(sys, "stdin", io.StringIO(str(dem)))

    exit_code = main(
        ["--method", "last-coordinate-index", "--strip-undecomposable-errors"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert stim.DetectorErrorModel(captured.out) == stim.DetectorErrorModel("""
        detector(0) D0
        detector(1) D1
        error(0.1) D0
    """)


def test_main_reports_decomposition_failure_on_stderr(monkeypatch, capsys):
    dem = stim.DetectorErrorModel("""
        detector(0) D0
        detector(1) D1
        error(0.1) D0 D1
        error(0.1) D0
    """)
    monkeypatch.setattr(sys, "stdin", io.StringIO(str(dem)))

    exit_code = main(["--method", "last-coordinate-index"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "needs to be decomposed into components" in captured.err


def test_main_rejects_unknown_method(capsys):
    with pytest.raises(SystemExit) as ex_info:
        main(["--method", "unknown"])

    captured = capsys.readouterr()
    assert ex_info.value.code == 2
    assert captured.out == ""
    assert "invalid choice" in captured.err
