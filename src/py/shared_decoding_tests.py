import pytest
import numpy as np
import stim



def shared_test_decode(decoder_class, config_class):
    """
    Tests that the 'decode' method computes a correct decoding output.
    """
    dem_string = f'''
        error(0.1) D0 D1 L0
        error(0.2) D0
    '''
    dem = stim.DetectorErrorModel(dem_string)
    config = config_class(dem)
    decoder = decoder_class(config)
    if hasattr(decoder, 'init_ilp'):
        decoder.init_ilp()

    syndrome1 = np.array([True, True], dtype=bool)
    predicted1 = decoder.decode(syndrome1)
    assert isinstance(predicted1, np.ndarray)
    assert predicted1.dtype.type == np.bool_
    assert np.array_equal(predicted1, np.array([True], dtype=bool))

    syndrome2 = np.array([True, False], dtype=bool)
    predicted2 = decoder.decode(syndrome2)
    assert isinstance(predicted2, np.ndarray)
    assert predicted2.dtype.type == np.bool_
    assert np.array_equal(predicted2, np.array([False], dtype=bool))

def shared_test_decode_complex_dem(decoder_class, config_class):
    """
    Tests the 'decode' method with a more complex DEM.
    """
    dem_string = f'''
        error(0.5) D0 D1 L0 L2
        error(0.01) D0
        error(0.01) D1
        error(0.05) D2
    '''
    dem = stim.DetectorErrorModel(dem_string)
    config = config_class(dem)
    decoder = decoder_class(config)
    if hasattr(decoder, 'init_ilp'):
        decoder.init_ilp()

    syndrome = np.array([True, True, False], dtype=bool)
    predicted = decoder.decode(syndrome)
    assert isinstance(predicted, np.ndarray)
    assert predicted.dtype.type == np.bool_
    expected = np.array([True, False, True], dtype=bool)
    assert np.array_equal(predicted, expected)


def shared_test_decode_batch_with_invalid_dimensions(decoder_class, config_class):
    """
    Tests that decode_batch raises a RuntimeError when given a 1D array.
    """
    dem_string = f'''
        error(0.1) D0 D1 L0
    '''
    dem = stim.DetectorErrorModel(dem_string)
    config = config_class(dem)
    decoder = decoder_class(config)
    if hasattr(decoder, 'init_ilp'):
        decoder.init_ilp()
    invalid_syndrome = np.array([True, True], dtype=bool)
    with pytest.raises(RuntimeError, match="Input syndromes must be a 2D NumPy array."):
        decoder.decode_batch(invalid_syndrome)

def shared_test_decode_batch(decoder_class, config_class):
    """
    Tests the decode_batch method with a 2D NumPy array input and output.
    """
    dem_string = f'''
        error(0.1) D0 D1 L0
        error(0.2) D0 L1
        error(0.05) D1
    '''
    dem = stim.DetectorErrorModel(dem_string)
    config = config_class(dem)
    decoder = decoder_class(config)
    if hasattr(decoder, 'init_ilp'):
        decoder.init_ilp()

    syndromes = np.array([
        [True, True],
        [True, False],
        [False, True],
    ], dtype=bool)
    predictions = decoder.decode_batch(syndromes)
    assert isinstance(predictions, np.ndarray)
    assert predictions.dtype.type == np.bool_
    assert predictions.shape == (3, 2)
    expected_predictions = np.array([
        [True, False],
        [False, True],
        [False, False],
    ], dtype=bool)
    assert np.array_equal(predictions, expected_predictions)

def shared_test_decode_batch_with_complex_model(decoder_class, config_class):
    """
    Tests the decode_batch method with a larger and more complex DEM.
    """
    dem_string = f'''
        error(0.1) D0 D1 L0
        error(0.2) D0 D2 L1
        error(0.05) D1 D3 L0 L2
        error(0.15) D2
        error(0.25) D3
    '''
    dem = stim.DetectorErrorModel(dem_string)
    config = config_class(dem)
    decoder = decoder_class(config)
    if hasattr(decoder, 'init_ilp'):
        decoder.init_ilp()

    batch_syndromes = np.array([
        [True, True, False, False],
        [True, False, True, False],
        [False, True, False, True],
        [False, False, True, True],
    ], dtype=bool)

    actual_predictions = decoder.decode_batch(batch_syndromes)
    assert isinstance(actual_predictions, np.ndarray)
    assert actual_predictions.dtype.type == np.bool_
    assert actual_predictions.shape == (4, 3)

    expected_predictions = np.array([
        [True, False, False],
        [False, True, False],
        [True, False, True],
        [False, False, False],
    ], dtype=bool)
    assert np.array_equal(actual_predictions, expected_predictions)


def shared_test_decoder_predicts_various_observable_flips(decoder_class, config_class):
    """
    Tests that the decoder correctly predicts a logical observable flip when
    a specific detector is triggered by an error that explicitly flips that
    logical observable.
    """
    for observable_id in range(64):
        dem_string = f'''
            error(0.01) D0 L{observable_id}
        '''
        dem = stim.DetectorErrorModel(dem_string)
        config = config_class(dem)
        decoder = decoder_class(config)

        if hasattr(decoder, 'init_ilp'):
            decoder.init_ilp()
        syndrome = np.zeros(dem.num_detectors, dtype=bool)
        syndrome[0] = True
        predicted_logical_flips_array = decoder.decode(syndrome)
        actual_flipped_observables = [
            idx for idx, is_flipped in enumerate(predicted_logical_flips_array) if is_flipped
        ]
        assert actual_flipped_observables == [observable_id], \
            (f"For observable L{observable_id}: "
             f"Expected predicted logical flips: [{observable_id}], "
             f"but got: {actual_flipped_observables} (from raw: {predicted_logical_flips_array})")



def shared_test_decode_from_detection_events(decoder_class, config_class):
    """
    Tests the `decode_from_detection_events` method with a list of detection indices.
    """
    dem_string = f'''
        error(0.1) D0 D1 L0
        error(0.2) D0
    '''
    dem = stim.DetectorErrorModel(dem_string)
    config = config_class(dem)
    decoder = decoder_class(config)
    if hasattr(decoder, 'init_ilp'):
        decoder.init_ilp()

    # Case 1: Detections corresponding to the D0 D1 L0 error
    detections1 = [0, 1]
    predicted1 = decoder.decode_from_detection_events(detections1)
    assert isinstance(predicted1, np.ndarray)
    assert predicted1.dtype.type == np.bool_
    assert np.array_equal(predicted1, np.array([True], dtype=bool))

    # Case 2: Detections corresponding to the D0 error
    detections2 = [0]
    predicted2 = decoder.decode_from_detection_events(detections2)
    assert isinstance(predicted2, np.ndarray)
    assert predicted2.dtype.type == np.bool_
    assert np.array_equal(predicted2, np.array([False], dtype=bool))

    # Case 3: No detections
    detections3 = []
    predicted3 = decoder.decode_from_detection_events(detections3)
    assert isinstance(predicted3, np.ndarray)
    assert predicted3.dtype.type == np.bool_
    assert np.array_equal(predicted3, np.array([False], dtype=bool))
