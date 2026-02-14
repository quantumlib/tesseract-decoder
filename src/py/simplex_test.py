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


def test_create_simplex_config():
    sc = tesseract_decoder.simplex.SimplexConfig(_DETECTOR_ERROR_MODEL, window_length=5)
    assert sc.dem == _DETECTOR_ERROR_MODEL
    assert sc.window_length == 5
    assert (
        str(sc)
        == "SimplexConfig(dem=DetectorErrorModel_Object, window_length=5, window_slide_length=0, verbose=0, merge_errors=1)"
    )


def test_create_simplex_decoder():
    decoder = tesseract_decoder.simplex.SimplexDecoder(
        tesseract_decoder.simplex.SimplexConfig(_DETECTOR_ERROR_MODEL, window_length=5)
    )
    errors = decoder.decode_to_errors(np.array([False, True], dtype=bool))
    assert decoder.cost_from_errors(errors) == pytest.approx(1.0986123)


def test_simplex_compile_decoder():
    shared_test_compile_decoder(
        tesseract_decoder.simplex.SimplexConfig, 
        tesseract_decoder.simplex.SimplexDecoder)


def test_tesseract_cost_from_errors():
    shared_test_cost_from_errors(
        tesseract_decoder.simplex.SimplexDecoder, 
        tesseract_decoder.simplex.SimplexConfig)


def test_simplex_get_observables_from_errors():
    shared_test_get_observables_from_errors(
        tesseract_decoder.simplex.SimplexDecoder, 
        tesseract_decoder.simplex.SimplexConfig)


def test_simplex_decode_from_detection_events():
    shared_test_decode_from_detection_events(
        tesseract_decoder.simplex.SimplexDecoder, 
        tesseract_decoder.simplex.SimplexConfig)


def test_simplex_decoder_predicts_various_observable_flips():
    shared_test_decoder_predicts_various_observable_flips(
        tesseract_decoder.simplex.SimplexDecoder, 
        tesseract_decoder.simplex.SimplexConfig)


def test_simplex_decode():
    shared_test_decode(
        tesseract_decoder.simplex.SimplexDecoder, 
        tesseract_decoder.simplex.SimplexConfig)


def test_simplex_decode_complex_dem():
    shared_test_decode_complex_dem(
        tesseract_decoder.simplex.SimplexDecoder, 
        tesseract_decoder.simplex.SimplexConfig)


def test_simplex_decode_batch_with_invalid_dimensions():
    shared_test_decode_batch_with_invalid_dimensions(
        tesseract_decoder.simplex.SimplexDecoder, 
        tesseract_decoder.simplex.SimplexConfig)


def test_simplex_decode_batch():
    shared_test_decode_batch(
        tesseract_decoder.simplex.SimplexDecoder, 
        tesseract_decoder.simplex.SimplexConfig)


def test_simplex_decode_batch_with_complex_model():
    shared_test_decode_batch_with_complex_model(
        tesseract_decoder.simplex.SimplexDecoder, 
        tesseract_decoder.simplex.SimplexConfig)


def test_tesseract_merge_errors_affects_cost():
    shared_test_merge_errors_affects_cost(
        tesseract_decoder.simplex.SimplexDecoder, 
        tesseract_decoder.simplex.SimplexConfig)

def test_simlpex_decode_with_mismatched_syndrome_size():
    shared_test_decode_with_mismatched_syndrome_size(
        tesseract_decoder.simplex.SimplexDecoder, 
        tesseract_decoder.simplex.SimplexConfig)

def test_test_simplex_decode_batch_with_mismatched_syndrome_size():
    shared_test_decode_batch_with_mismatched_syndrome_size(
        tesseract_decoder.simplex.SimplexDecoder, 
        tesseract_decoder.simplex.SimplexConfig)

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
