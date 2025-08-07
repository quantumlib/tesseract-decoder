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
import numpy as np

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
    assert decoder.get_observables_from_errors([1]) == []
    assert decoder.cost_from_errors([2]) == pytest.approx(1.0986123)

def test_simplex_decoder_predicts_various_observable_flips():
    """
    Tests that the Simplex decoder correctly predicts a logical observable
    flip when a specific detector is triggered by an error that explicitly
    flips that logical observable.
    """

    # Iterate through observable IDs from 0 to 63
    for observable_id in range(64):
        # Create a simple DetectorErrorModel where an error on D0 also flips L{observable_id}
        dem_string = f'''
            error(0.01) D0 L{observable_id}
        '''
        dem = stim.DetectorErrorModel(dem_string)

        # Initialize SimplexConfig and SimplexDecoder with the generated DEM
        config = tesseract_decoder.simplex.SimplexConfig(dem, window_length=1) # window_length must be set
        decoder = tesseract_decoder.simplex.SimplexDecoder(config)
        decoder.init_ilp() # Initialize the ILP solver

        # Simulate a detection event on D0.
        # The decoder should identify the most likely error causing D0,
        # which in this DEM is the error that also flips L{observable_id}.
        syndrome = np.zeros(dem.num_detectors, dtype=bool)
        syndrome[0] = True
        predicted_logical_flips_array = decoder.decode_from_detection_events(syndrome)

        # Convert the boolean array/list to a list of flipped observable IDs
        actual_flipped_observables = [
            idx for idx, is_flipped in enumerate(predicted_logical_flips_array) if is_flipped
        ]

        # Assert that the list of actual flipped observables matches the single expected observable_id.
        assert actual_flipped_observables == [observable_id], \
            (f"For observable L{observable_id}: "
             f"Expected predicted logical flips: [{observable_id}], "
             f"but got: {actual_flipped_observables} (from raw: {predicted_logical_flips_array})")

def test_decode_from_detection_events():
    """
    Tests that the 'test_decode_from_detection_events' method inside SimplexDecoder
    computes a correct decoding output.
    """
    # Create a simple DEM with two detectors and one observable.
    dem_string = f'''
        error(0.1) D0 D1 L0
        error(0.2) D0
    '''
    dem = stim.DetectorErrorModel(dem_string)

    # Configure the decoder.
    config = tesseract_decoder.simplex.SimplexConfig(dem, window_length=1)
    decoder = tesseract_decoder.simplex.SimplexDecoder(config)
    decoder.init_ilp()

    # Case 1: Detectors 0 and 1 fire.
    # The most likely error is D0 D1 L0. Expected logical flip: L0.
    syndrome1 = np.array([True, True], dtype=bool)
    predicted1 = decoder.decode_from_detection_events(syndrome1)
    
    # Verify the type of the output.
    assert isinstance(predicted1, np.ndarray)
    assert predicted1.dtype.type == np.bool_
    
    # Verify the content of the output.
    # The result should have one True value at the index corresponding to the observable.
    assert np.array_equal(predicted1, np.array([True], dtype=bool))

    # Case 2: Only detector 0 fires.
    # The most likely error is D0. Expected logical flip: none.
    syndrome2 = np.array([True, False], dtype=bool)
    predicted2 = decoder.decode_from_detection_events(syndrome2)
    
    # Verify the type of the output.
    assert isinstance(predicted2, np.ndarray)
    assert predicted2.dtype.type == np.bool_

    # Verify the content of the output. All values should be False.
    assert np.array_equal(predicted2, np.array([False], dtype=bool))

def test_decode_from_detection_events_complex_dem():
    """
    Tests the 'decode_from_detection_events' method inside SimplexDecoder with a more complex DEM.
    """
    dem_string = f'''
        error(0.5) D0 D1 L0 L2
        error(0.01) D0
        error(0.01) D1
        error(0.05) D2
    '''

    # Create a complex DEM.
    dem = stim.DetectorErrorModel(dem_string)

    # Configure the decoder.
    config = tesseract_decoder.simplex.SimplexConfig(dem, window_length=1)
    decoder = tesseract_decoder.simplex.SimplexDecoder(config)
    decoder.init_ilp()

    # Create the syndrome and decode it.
    syndrome = np.array([True, True, False], dtype=bool)
    predicted = decoder.decode_from_detection_events(syndrome)

    # Verify the type of the output.
    assert isinstance(predicted, np.ndarray)
    assert predicted.dtype.type == np.bool_

    # Verify the content of the output.
    expected = np.array([True, False, True], dtype=bool)
    assert np.array_equal(predicted, expected)

def test_decode_batch_with_invalid_dimensions():
    """
    Tests that decode_batch raises a RuntimeError when given a 1D array.
    """
    dem_string = f'''
        error(0.1) D0 D1 L0
    '''

    # Create the DEM.
    dem = stim.DetectorErrorModel(dem_string)

    # Configure the decoder.
    config = tesseract_decoder.simplex.SimplexConfig(dem, window_length=1)
    decoder = tesseract_decoder.simplex.SimplexDecoder(config)
    decoder.init_ilp()

    # Try decoding an invalid syndrome.
    invalid_syndrome = np.array([True, True], dtype=bool)
    with pytest.raises(RuntimeError, match="Input syndromes must be a 2D NumPy array."):
        decoder.decode_batch(invalid_syndrome)

def test_decode_batch():
    """
    Tests the decode_batch method with a 2D NumPy array input and output
    for the Simplex decoder. This verifies that the SimplexDecoder can decode
    multiple shots from a batch of shots.
    """
    # Create a simple DEM with two detectors and two observables.
    dem_string = f'''
        error(0.1) D0 D1 L0
        error(0.2) D0 L1
        error(0.05) D1
    '''
    dem = stim.DetectorErrorModel(dem_string)
    config = tesseract_decoder.simplex.SimplexConfig(dem, window_length=1)
    decoder = tesseract_decoder.simplex.SimplexDecoder(config)
    decoder.init_ilp()

    # Shot 1: D0 and D1 fired -> most likely error is D0 D1 L0 -> should predict L0 flip
    # Shot 2: D0 fired -> most likely error is D0 L1 -> should predict L1 flip
    # Shot 3: D1 fired -> most likely error is D1 -> should predict no flips
    syndromes = np.array([
        [True, True],
        [True, False],
        [False, True],
    ], dtype=bool)

    # Decode the batch.
    predictions = decoder.decode_batch(syndromes)

    # Verify the type and dimensions of the output.
    assert isinstance(predictions, np.ndarray)
    assert predictions.dtype.type == np.bool_
    assert predictions.shape == (3, 2)

    # Verify the content of the output.
    expected_predictions = np.array([
        [True, False],   # Shot 1: L0 should flip, L1 should not
        [False, True],   # Shot 2: L0 should not flip, L1 should
        [False, False],  # Shot 3: No flips
    ], dtype=bool)
    assert np.array_equal(predictions, expected_predictions)

def test_decode_batch_more_complex_dem():
    """
    Tests the decode_batch method inside SimplexDecoder with a more complex DEM.
    """
    dem_string = f'''
        error(0.1) D0 D1 L0
        error(0.2) D0 D2 L1
        error(0.05) D1 D3 L0 L2
        error(0.15) D2
        error(0.25) D3
    '''

    # Create a complex DEM.
    dem = stim.DetectorErrorModel(dem_string)

    # Configure the decoder.
    config = tesseract_decoder.simplex.SimplexConfig(dem, window_length=1)
    decoder = tesseract_decoder.simplex.SimplexDecoder(config)
    decoder.init_ilp()

    # Create a batch of syndromes for each shot inside the batch.
    batch_syndromes = np.array([
        [True, True, False, False],
        [True, False, True, False],
        [False, True, False, True],
        [False, False, True, True],
    ], dtype=bool)

    # Decode batch of syndromes.
    predictions = decoder.decode_batch(batch_syndromes)

    # Verify the type and dimensions of the output.
    assert isinstance(predictions, np.ndarray)
    assert predictions.dtype.type == np.bool_
    assert predictions.shape == (4, 3)

    # Verify the content of the output.
    expected_predictions = np.array([
        [True, False, False],
        [False, True, False],
        [True, False, True],
        [False, False, False],
    ], dtype=bool)
    assert np.array_equal(predictions, expected_predictions)

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
