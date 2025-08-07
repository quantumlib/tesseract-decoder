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


def test_create_config():
    assert (
        str(tesseract_decoder.tesseract.TesseractConfig(_DETECTOR_ERROR_MODEL))
        == "TesseractConfig(dem=DetectorErrorModel_Object, det_beam=65535, no_revisit_dets=0, at_most_two_errors_per_detector=0, verbose=0, pqlimit=18446744073709551615, det_orders=[], det_penalty=0)"
    )
    assert (
        tesseract_decoder.tesseract.TesseractConfig(_DETECTOR_ERROR_MODEL).dem
        == _DETECTOR_ERROR_MODEL
    )


def test_create_node():
    node = tesseract_decoder.tesseract.Node(errors=[1, 0])
    assert node.errors == [1, 0]


def test_create_decoder():
    config = tesseract_decoder.tesseract.TesseractConfig(_DETECTOR_ERROR_MODEL)
    decoder = tesseract_decoder.tesseract.TesseractDecoder(config)
    decoder.decode_to_errors([0])
    decoder.decode_to_errors(detections=[0], det_order=0, det_beam=0)
    assert decoder.get_observables_from_errors([1]) == []
    assert decoder.cost_from_errors([1]) == pytest.approx(0.5108256237659907)

def test_tesseract_decoder_predicts_various_observable_flips():
    """
    Tests that the Tesseract decoder correctly predicts a logical observable
    flip when a specific detector is triggered by an error that explicitly
    flips that logical observable.

    This test iterates through various observable IDs to ensure the logic
    correctly handles different positions.
    """
    # Iterate through observable IDs from 0 to 63
    for observable_id in range(64):
        # Create a simple DetectorErrorModel where an error on D0 also flips L{observable_id}
        dem_string = f'''
            error(0.01) D0 L{observable_id}
        '''
        dem = stim.DetectorErrorModel(dem_string)

        # Initialize TesseractConfig and TesseractDecoder with the generated DEM
        config = tesseract_decoder.tesseract.TesseractConfig(dem)
        decoder = tesseract_decoder.tesseract.TesseractDecoder(config)

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

        # Assert that the list of actual flipped observables matches the single expected observable_id
        assert actual_flipped_observables == [observable_id], \
            (f"For observable L{observable_id}: "
             f"Expected predicted logical flips: [{observable_id}], "
             f"but got: {actual_flipped_observables} (from raw: {predicted_logical_flips_array})")

def test_decode_from_detection_events():
    """
    Tests the 'decode_from_detection_events' method inside the Tesseract decoder.
    This test verifies that the method provides appropriate output for a given measurement syndrome.
    """
    # Create a DEM with two detectors and one observable
    dem_string = f'''
        error(0.1) D0 D1 L0
        error(0.2) D0
    '''
    dem = stim.DetectorErrorModel(dem_string)
    config = tesseract_decoder.tesseract.TesseractConfig(dem)
    decoder = tesseract_decoder.tesseract.TesseractDecoder(config)

    # Case 1: Detectors 0 and 1 fire.
    # The most likely error is D0 D1 L0. Expected logical flip is L0.
    syndrome1 = np.array([True, True], dtype=bool)
    predicted1 = decoder.decode_from_detection_events(syndrome1)

    # Verity the type of the output.
    assert isinstance(predicted1, np.ndarray)
    assert predicted1.dtype.type == np.bool_

    # Verity the values of the output.
    # The result should have one True value at the index corresponding to the observable.
    assert np.array_equal(predicted1, np.array([True], dtype=bool))

    # Case 2: Only detector 0 fires.
    # The most likely error is D0. No expected logical flips.
    syndrome2 = np.array([True, False], dtype=bool)
    predicted2 = decoder.decode_from_detection_events(syndrome2)

    # Verity the type of the output.
    assert isinstance(predicted2, np.ndarray)
    assert predicted2.dtype.type == np.bool_

    # Verity the values of the output. All should be False.
    assert np.array_equal(predicted2, np.array([False], dtype=bool))


def test_decode_from_detection_events_more_complex_dem():
    """
    Tests the 'decode_from_detection_events' method with a more complex DEM.
    """
    dem_string = f'''
        error(0.5) D0 D1 L0 L2
        error(0.01) D0
        error(0.01) D1
        error(0.05) D2
    '''
    dem = stim.DetectorErrorModel(dem_string)
    config = tesseract_decoder.tesseract.TesseractConfig(dem)
    decoder = tesseract_decoder.tesseract.TesseractDecoder(config)

    # Simulate detections on D0 and D1.
    # The most likely explanation is the combined D0 D1 L0 L2 error
    # because it has a high probability of 0.5, compared to other errors.
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
    dem = stim.DetectorErrorModel(dem_string)
    config = tesseract_decoder.tesseract.TesseractConfig(dem)
    decoder = tesseract_decoder.tesseract.TesseractDecoder(config)
    
    # Provide a 1D NumPy array instead of the required 2D array.
    invalid_syndrome = np.array([True, True], dtype=bool)

    with pytest.raises(RuntimeError, match="Input syndromes must be a 2D NumPy array."):
        decoder.decode_batch(invalid_syndrome)


def test_decode_batch():
    """
    Tests the decode_batch method with a 2D NumPy array input and output.
    This verifies that multiple shots are decoded correctly and that the output
    format is a 2D NumPy array.
    """
    # Create a simple DEM with two detectors and two observables.
    dem_string = f'''
        error(0.1) D0 D1 L0
        error(0.2) D0 L1
        error(0.05) D1
    '''
    dem = stim.DetectorErrorModel(dem_string)
    config = tesseract_decoder.tesseract.TesseractConfig(dem)
    decoder = tesseract_decoder.tesseract.TesseractDecoder(config)

    # Shot 1: D0 and D1 fired -> most likely error is D0 D1 L0 -> should predict L0 flip
    # Shot 2: D0 fired -> most likely error is D0 L1 -> should predict L1 flip
    # Shot 3: D1 fired -> most likely error is D1 -> should predict no flips
    batch_syndromes = np.array([
        [True, True],
        [True, False],
        [False, True],
    ], dtype=bool)

    # Decode the batch.
    actual_predictions = decoder.decode_batch(batch_syndromes)

    # Assert types and dimensions of the output.
    assert isinstance(actual_predictions, np.ndarray)
    assert actual_predictions.dtype.type == np.bool_
    assert actual_predictions.shape == (3, 2)

    # Expected batch predictions.
    expected_predictions = np.array([
        [True, False],   # Shot 1: should predict L0 flip and no flip for L1
        [False, True],   # Shot 2: should predict no flip for L0, and a flip for L1
        [False, False],  # Shot 3: should predict no flips
    ], dtype=bool)
    assert np.array_equal(actual_predictions, expected_predictions)

def test_decode_batch_more_complex_dem():
    """
    Tests the decode_batch method with a and more complex DEM.
    """
    dem_string = f'''
        error(0.1) D0 D1 L0
        error(0.2) D0 D2 L1
        error(0.05) D1 D3 L0 L2
        error(0.15) D2
        error(0.25) D3
    '''
    dem = stim.DetectorErrorModel(dem_string)
    config = tesseract_decoder.tesseract.TesseractConfig(dem)
    decoder = tesseract_decoder.tesseract.TesseractDecoder(config)

    # Define a complex batch that has 4 syndromes (shots/simulations),
    # and each of them has 4 detectors and 3 observables.
    batch_syndromes = np.array([
        # Detections: D0, D1
        # Most likely error: error(0.1) D0 D1 L0
        # Expected prediction: L0 flip
        [True, True, False, False],

        # Detections: D0, D2
        # Most likely error: error(0.2) D0 D2 L1
        # Expected prediction: L1 flip
        [True, False, True, False],

        # Detections: D1, D3
        # Most likely error: error(0.05) D1 D3 L0 L2
        # Expected prediction: L0, L2 flips
        [False, True, False, True],

        # Detections: D2, D3
        # Most likely error: error(0.15) D2 and error(0.25) D3 -> no combined error
        # Expected prediction: No flips
        [False, False, True, True],
    ], dtype=bool)

    # Decode the batch of shots/simulations.
    actual_predictions = decoder.decode_batch(batch_syndromes)

    # Assert types and dimensions for the given output.
    assert isinstance(actual_predictions, np.ndarray)
    assert actual_predictions.dtype.type == np.bool_
    assert actual_predictions.shape == (4, 3)

    # Expected batch predictions based on the DEM.
    expected_predictions = np.array([
        [True, False, False],   # Shot 1: L0 flip
        [False, True, False],   # Shot 2: L1 flip
        [True, False, True],    # Shot 3: L0 and L2 flips
        [False, False, False],  # Shot 4: No flips
    ], dtype=bool)

    assert np.array_equal(actual_predictions, expected_predictions)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
