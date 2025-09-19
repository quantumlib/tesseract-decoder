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





def test_create_tesseract_config():
    config = tesseract_decoder.tesseract.TesseractConfig(_DETECTOR_ERROR_MODEL)
    assert config.dem == _DETECTOR_ERROR_MODEL
    assert config.det_beam == 5
    assert config.no_revisit_dets is True

    assert config.verbose is False
    assert config.merge_errors is True
    assert config.pqlimit == 200000
    assert config.det_penalty == 0
    assert config.create_visualization is False
    assert len(config.det_orders) == 20


def test_create_tesseract_config_with_dem():
    """
    Tests the constructor that takes a `dem` argument.
    """

    config = tesseract_decoder.tesseract.TesseractConfig(_DETECTOR_ERROR_MODEL)
    
    assert config.dem == _DETECTOR_ERROR_MODEL
    assert config.det_beam == 5
    assert config.no_revisit_dets is True

    assert config.verbose is False
    assert config.merge_errors is True
    assert config.pqlimit == 200000
    assert config.det_penalty == 0
    assert config.create_visualization is False
    assert len(config.det_orders) == 20

def test_create_tesseract_config_with_dem_and_custom_args():
    """
    Tests the constructor with a `dem` object and custom arguments.
    """
    # Create an instance with a dem and custom arguments.
    config = tesseract_decoder.tesseract.TesseractConfig(
        dem=_DETECTOR_ERROR_MODEL,
        det_beam=100,
        merge_errors=False,
        det_penalty=0.5
    )
    
    assert config.dem == _DETECTOR_ERROR_MODEL
    assert config.det_beam == 100
    assert config.no_revisit_dets is True

    assert config.verbose is False
    assert config.merge_errors is False
    assert config.pqlimit == 200000
    assert config.det_penalty == 0.5
    assert config.create_visualization is False
    assert len(config.det_orders) == 20
    

def test_compile_decoder_for_dem_basic_functionality():
    """
    Verifies that `compile_decoder_for_dem` returns a `TesseractDecoder` instance.
    """
    config = tesseract_decoder.tesseract.TesseractConfig()
    custom_dem = stim.DetectorErrorModel()
    decoder = config.compile_decoder_for_dem(custom_dem)

    assert isinstance(decoder, tesseract_decoder.tesseract.TesseractDecoder)

def test_compile_decoder_for_dem_sets_dem_on_config():
    """
    Ensures that the `dem` property of the TesseractConfig object is updated
    before the decoder is compiled.
    """
    config = tesseract_decoder.tesseract.TesseractConfig()
    custom_dem = stim.DetectorErrorModel()
    decoder = config.compile_decoder_for_dem(custom_dem)

    # Check that the config object itself has been updated.
    assert config.dem == custom_dem
    # Check that the decoder's config also reflects the change.
    assert decoder.config.dem == custom_dem

def test_compile_decoder_for_dem_preserves_other_config_params():
    """
    Tests that other custom parameters are not overwritten when the `dem` is updated.
    """
    # Create a config with custom parameters.
    config = tesseract_decoder.tesseract.TesseractConfig(det_beam=100, verbose=True, merge_errors=False)

    # Define a new DEM to pass to the method.
    new_dem = stim.DetectorErrorModel()
    decoder = config.compile_decoder_for_dem(new_dem)

    # Assert that the new decoder's config has the new dem, but retains all the other custom parameters.
    assert decoder.config.dem == new_dem
    assert decoder.config.det_beam == 100
    assert decoder.config.verbose is True
    assert decoder.config.merge_errors is False


def test_compile_decoder_for_dem_with_empty_dem():
    """
    Ensures the method works correctly with an empty `dem` object.
    """
    config = tesseract_decoder.tesseract.TesseractConfig(verbose=True)

    empty_dem = stim.DetectorErrorModel()
    decoder = config.compile_decoder_for_dem(empty_dem)

    assert decoder.config.dem == empty_dem
    assert decoder.config.verbose is True

def test_create_tesseract_config_no_dem():
    """
    Tests the new constructor that does not require a `dem` argument.
    """
    # Create an instance with no arguments.
    config = tesseract_decoder.tesseract.TesseractConfig()

    assert config.dem == stim.DetectorErrorModel()
    assert config.det_beam == 5
    assert config.no_revisit_dets is True

    assert config.verbose is False
    assert config.merge_errors is True
    assert config.pqlimit == 200000
    assert config.det_penalty == 0.0
    assert config.create_visualization is False

def test_create_tesseract_config_no_dem_with_custom_args():
    """
    Tests the new constructor with custom arguments to ensure they are passed correctly.
    """
    # Create an instance with no dem but a custom det_beam.
    config = tesseract_decoder.tesseract.TesseractConfig(det_beam=15, verbose=True)

    assert config.dem == stim.DetectorErrorModel()
    assert config.det_beam == 15
    assert config.no_revisit_dets is True

    assert config.verbose is True
    assert config.merge_errors is True
    assert config.pqlimit == 200000
    assert config.det_penalty == 0.0
    assert config.create_visualization is False


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
