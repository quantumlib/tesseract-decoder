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

import pytest
import numpy as np
import stim

from src import tesseract_decoder
from src.py.shared_decoding_tests import (
    shared_test_decode,
    shared_test_decode_batch_with_invalid_dimensions,
    shared_test_decode_batch_with_complex_model,
    shared_test_decoder_predicts_various_observable_flips,
    shared_test_decode_complex_dem,
    shared_test_decode_batch,
    shared_test_decode_from_detection_events,
    shared_test_compile_decoder,
    shared_test_cost_from_errors,
    shared_test_get_observables_from_errors,
    shared_test_merge_errors_affects_cost,
    shared_test_decode_with_mismatched_syndrome_size,
    shared_test_decode_batch_with_mismatched_syndrome_size,
)

_DETECTOR_ERROR_MODEL = stim.DetectorErrorModel(
    """
error(0.125) D0
error(0.375) D0 D1
error(0.25) D1
"""
)


def test_create_node():
    node = tesseract_decoder.tesseract.Node(errors=[1, 0])
    assert node.errors == [1, 0]


def test_create_tesseract_config():
    assert (
        str(tesseract_decoder.tesseract.TesseractConfig(_DETECTOR_ERROR_MODEL))
        == "TesseractConfig(dem=DetectorErrorModel_Object, det_beam=65535, no_revisit_dets=0, at_most_two_errors_per_detector=0, verbose=0, merge_errors=1, pqlimit=18446744073709551615, det_orders=[], det_penalty=0, create_visualization=0)"
    )
    assert (
        tesseract_decoder.tesseract.TesseractConfig(_DETECTOR_ERROR_MODEL).dem
        == _DETECTOR_ERROR_MODEL
    )


def test_create_tesseract_decoder():
    config = tesseract_decoder.tesseract.TesseractConfig(_DETECTOR_ERROR_MODEL)
    decoder = tesseract_decoder.tesseract.TesseractDecoder(config)
    decoder.decode_to_errors(np.array([True, False], dtype=bool))
    decoder.decode_to_errors(
        syndrome=np.array([True, False], dtype=bool), det_order=0, det_beam=0
    )
    assert decoder.get_observables_from_errors([1]) == []
    assert decoder.cost_from_errors([1]) == pytest.approx(0.5108256237659907)


def test_tesseract_compile_decoder():
    shared_test_compile_decoder(
        tesseract_decoder.tesseract.TesseractConfig, 
        tesseract_decoder.tesseract.TesseractDecoder)


def test_tesseract_cost_from_errors():
    shared_test_cost_from_errors(
        tesseract_decoder.tesseract.TesseractDecoder, 
        tesseract_decoder.tesseract.TesseractConfig)


def test_tesseract_get_observables_from_errors():
    shared_test_get_observables_from_errors(
        tesseract_decoder.tesseract.TesseractDecoder, 
        tesseract_decoder.tesseract.TesseractConfig)


def test_tesseract_decode_from_detection_events():
    shared_test_decode_from_detection_events(
        tesseract_decoder.tesseract.TesseractDecoder, 
        tesseract_decoder.tesseract.TesseractConfig)


def test_tesseract_decoder_predicts_various_observable_flips():
    shared_test_decoder_predicts_various_observable_flips(
        tesseract_decoder.tesseract.TesseractDecoder, 
        tesseract_decoder.tesseract.TesseractConfig)


def test_tesseract_decode():
    shared_test_decode(
        tesseract_decoder.tesseract.TesseractDecoder, 
        tesseract_decoder.tesseract.TesseractConfig)


def test_tesseract_decode_complex_dem():
    shared_test_decode_complex_dem(
        tesseract_decoder.tesseract.TesseractDecoder, 
        tesseract_decoder.tesseract.TesseractConfig)


def test_tesseract_decode_batch_with_invalid_dimensions():
    shared_test_decode_batch_with_invalid_dimensions(
        tesseract_decoder.tesseract.TesseractDecoder, 
        tesseract_decoder.tesseract.TesseractConfig)


def test_tesseract_decode_batch():
    shared_test_decode_batch(
        tesseract_decoder.tesseract.TesseractDecoder, 
        tesseract_decoder.tesseract.TesseractConfig)


def test_tesseract_decode_batch_with_complex_model():
    shared_test_decode_batch_with_complex_model(
        tesseract_decoder.tesseract.TesseractDecoder, 
        tesseract_decoder.tesseract.TesseractConfig)


def test_tesseract_merge_errors_affects_cost():
    shared_test_merge_errors_affects_cost(
        tesseract_decoder.tesseract.TesseractDecoder, 
        tesseract_decoder.tesseract.TesseractConfig)

def test_simlpex_decode_with_mismatched_syndrome_size():
    shared_test_decode_with_mismatched_syndrome_size(
        tesseract_decoder.tesseract.TesseractDecoder, 
        tesseract_decoder.tesseract.TesseractConfig)

def test_test_simplex_decode_batch_with_mismatched_syndrome_size():
    shared_test_decode_batch_with_mismatched_syndrome_size(
        tesseract_decoder.tesseract.TesseractDecoder, 
        tesseract_decoder.tesseract.TesseractConfig)

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
