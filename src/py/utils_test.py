# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import pytest
import stim

from src import tesseract_decoder


_DETECTOR_ERROR_MODEL = stim.DetectorErrorModel(
    """
error(0.125) D0
error(0.375) D0 D1
error(0.25) D1
"""
)


def test_module_has_global_constants():
    assert tesseract_decoder.utils.EPSILON <= 1e-7
    assert not math.isfinite(tesseract_decoder.utils.INF)


def test_get_detector_coords():
    assert tesseract_decoder.utils.get_detector_coords(_DETECTOR_ERROR_MODEL) == []


def test_build_detector_graph():
    assert tesseract_decoder.utils.build_detector_graph(_DETECTOR_ERROR_MODEL) == [
        [1],
        [0],
    ]


def test_build_det_orders():
    assert tesseract_decoder.utils.build_det_orders(
        _DETECTOR_ERROR_MODEL, num_det_orders=1, seed=0
    ) == [[0, 1]]


def test_build_det_orders_coordinate():
    assert tesseract_decoder.utils.build_det_orders(
        _DETECTOR_ERROR_MODEL,
        num_det_orders=1,
        method=tesseract_decoder.utils.DetOrder.DetCoordinate,
        seed=0,
    ) == [[0, 1]]


def test_build_det_orders_index():
    res = tesseract_decoder.utils.build_det_orders(
        _DETECTOR_ERROR_MODEL,
        num_det_orders=1,
        method=tesseract_decoder.utils.DetOrder.DetIndex,
        seed=0,
    )
    assert res == [[0, 1]] or res == [[1, 0]]


def test_get_errors_from_dem():
    expected = "Error{cost=1.945910, symptom=Symptom{detectors=[0], observables=[]}}, Error{cost=0.510826, symptom=Symptom{detectors=[0 1], observables=[]}}, Error{cost=1.098612, symptom=Symptom{detectors=[1], observables=[]}}"
    assert (
        ", ".join(
            map(str, tesseract_decoder.utils.get_errors_from_dem(_DETECTOR_ERROR_MODEL))
        )
        == expected
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
