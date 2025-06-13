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


def test_create_simplex_config():
    sc = tesseract_decoder.simplex.SimplexConfig(_DETECTOR_ERROR_MODEL, window_length=5)
    assert sc.dem == _DETECTOR_ERROR_MODEL
    assert sc.window_length == 5
    assert (
        str(sc)
        == "SimplexConfig(dem=DetectorErrorModel_Object, window_length=5, window_slide_length=0, verbose=0)"
    )


def test_create_simplex_decoder():
    decoder = tesseract_decoder.simplex.SimplexDecoder(
        tesseract_decoder.simplex.SimplexConfig(_DETECTOR_ERROR_MODEL, window_length=5)
    )
    decoder.init_ilp()
    decoder.decode_to_errors([1])
    assert decoder.mask_from_errors([1]) == 0
    assert decoder.cost_from_errors([2]) == pytest.approx(1.0986123)
    assert decoder.decode([1, 2]) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
