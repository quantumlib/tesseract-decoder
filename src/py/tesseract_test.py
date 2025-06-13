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


def test_create_config():
    assert (
        str(tesseract_decoder.tesseract.TesseractConfig(_DETECTOR_ERROR_MODEL))
        == "TesseractConfig(dem=DetectorErrorModel_Object, det_beam=65535, no_revisit_dets=0, at_most_two_errors_per_detector=0, verbose=0, pqlimit=18446744073709551615, det_orders=[], det_penalty=0)"
    )


def test_create_node():
    node = tesseract_decoder.tesseract.Node(dets=["a"])
    assert node.dets == ["a"]


def test_create_qnode():
    qnode = tesseract_decoder.tesseract.QNode(num_dets=5, errs=[42])
    assert qnode.num_dets == 5
    assert str(qnode) == "QNode(cost=0, num_dets=5, errs=[42])"


def test_create_decoder():
    config = tesseract_decoder.tesseract.TesseractConfig(_DETECTOR_ERROR_MODEL)
    decoder = tesseract_decoder.tesseract.TesseractDecoder(config)
    decoder.decode_to_errors([0])
    decoder.decode_to_errors([0], 0)
    assert decoder.mask_from_errors([1]) == 0
    assert decoder.cost_from_errors([1]) == pytest.approx(1.609438)
    assert decoder.decode([0]) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
