import pytest
import numpy as np
import stim
import math

def shared_test_compile_decoder(config_class, decoder_class):
    """
    Tests the `compile_decoder` method on a config class.
    """
    dem_string = "error(0.1) D0 D1 L0"
    dem = stim.DetectorErrorModel(dem_string)
    config = config_class(dem)
    config.merge_errors = False
    decoder = config.compile_decoder()
    
    assert isinstance(decoder, decoder_class)
    assert decoder.config.dem == config.dem
    assert decoder.num_observables == dem.num_observables


def shared_test_cost_from_errors(decoder_class, config_class):
    """
    Tests the `cost_from_errors` method, which returns the total cost of a
    predicted error chain. The cost is calculated as the sum of the log-odds
    ratio for each error mechanism.
    """

    dem_string = f'''
        error(0.1) D0 L0
        error(0.2) D1 L1
        error(0.3) D2
        error(0.4) D3 L0 L1
    '''
    dem = stim.DetectorErrorModel(dem_string)
    config = config_class(dem)
    config.merge_errors = False
    decoder = decoder_class(config)

    # Case 1: A single error mechanism that flips one observable (L0).
    errors1 = [0]
    decoder.predicted_errors_buffer = errors1
    cost1 = decoder.cost_from_errors(errors1)
    assert cost1 == pytest.approx(math.log((1 - 0.1) / 0.1))

    # Case 2: A single error mechanism that flips multiple observables (L0, L1).
    errors2 = [3]
    decoder.predicted_errors_buffer = errors2
    cost2 = decoder.cost_from_errors(errors2)
    assert cost2 == pytest.approx(math.log((1 - 0.4) / 0.4))

    # Case 3: Multiple error mechanisms whose effects cancel out (L0 from error 0, L0 from error 3).
    errors3 = [0, 3]
    decoder.predicted_errors_buffer = errors3
    cost3 = decoder.cost_from_errors(errors3)
    assert cost3 == pytest.approx(math.log((1 - 0.1) / 0.1) + math.log((1 - 0.4) / 0.4))

    # Case 4: No errors.
    errors4 = []
    decoder.predicted_errors_buffer = errors4
    cost4 = decoder.cost_from_errors(errors4)
    assert cost4 == pytest.approx(0)

def shared_test_get_observables_from_errors(decoder_class, config_class):
    """
    Tests the `get_observables_from_errors` method, which converts a list of
    error mechanism indices into a list of logical observable flips.
    """
    dem_string = f'''
        error(0.1) D0 L0
        error(0.2) D1 L1
        error(0.3) D2
        error(0.4) D3 L0 L1
    '''
    dem = stim.DetectorErrorModel(dem_string)
    config = config_class(dem)
    config.merge_errors = False
    decoder = decoder_class(config)
    num_observables = dem.num_observables

    # Case 1: A single error mechanism that flips one observable (L0).
    errors1 = [0]
    predicted_list1 = decoder.get_observables_from_errors(errors1)
    assert isinstance(predicted_list1, list)
    predicted_array1 = np.array(predicted_list1, dtype=bool)
    assert predicted_array1.shape[0] == num_observables
    assert np.array_equal(predicted_array1, np.array([True, False], dtype=bool))

    # Case 2: A single error mechanism that flips multiple observables (L0, L1).
    errors2 = [3]
    predicted_list2 = decoder.get_observables_from_errors(errors2)
    assert isinstance(predicted_list2, list)
    predicted_array2 = np.array(predicted_list2, dtype=bool)
    assert predicted_array2.shape[0] == num_observables
    assert np.array_equal(predicted_array2, np.array([True, True], dtype=bool))

    # Case 3: Multiple error mechanisms whose effects cancel out (L0 from error 0, L0 from error 3).
    errors3 = [0, 3]
    predicted_list3 = decoder.get_observables_from_errors(errors3)
    assert isinstance(predicted_list3, list)
    predicted_array3 = np.array(predicted_list3, dtype=bool)
    assert predicted_array3.shape[0] == num_observables
    assert np.array_equal(predicted_array3, np.array([False, True], dtype=bool))
    
    # Case 4: No errors.
    errors4 = []
    predicted_list4 = decoder.get_observables_from_errors(errors4)
    assert isinstance(predicted_list4, list)
    predicted_array4 = np.array(predicted_list4, dtype=bool)
    assert predicted_array4.shape[0] == num_observables
    assert np.array_equal(predicted_array4, np.zeros(num_observables, dtype=bool))


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

    syndrome = np.array([True, True, False], dtype=bool)
    predicted = decoder.decode(syndrome)
    assert isinstance(predicted, np.ndarray)
    assert predicted.dtype.type == np.bool_
    expected = np.array([True, False, True], dtype=bool)
    assert np.array_equal(predicted, expected)

def shared_test_decode_batch_with_invalid_dimensions(decoder_class, config_class):
    """
    Tests that the 'decode_batch' raises a RuntimeError when given a 1D array.
    """
    dem_string = f'''
        error(0.1) D0 D1 L0
    '''
    dem = stim.DetectorErrorModel(dem_string)
    config = config_class(dem)
    decoder = decoder_class(config)
    invalid_syndrome = np.array([True, True], dtype=bool)
    with pytest.raises(RuntimeError, match="Input syndromes must be a 2D NumPy array."):
        decoder.decode_batch(invalid_syndrome)

def shared_test_decode_batch(decoder_class, config_class):
    """
    Tests the 'decode_batch' method with a 2D NumPy array input and output.
    """
    dem_string = f'''
        error(0.1) D0 D1 L0
        error(0.2) D0 L1
        error(0.05) D1
    '''
    dem = stim.DetectorErrorModel(dem_string)
    config = config_class(dem)
    decoder = decoder_class(config)
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
    Tests the 'decode_batch' method with a larger and more complex DEM.
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
    batch_syndromes = np.array([
        [True, True, False, False],
        [True, False, True, False],
        [False, True, False, True],
        [False, False, True, True],
    ], dtype=bool)

    predictions = decoder.decode_batch(batch_syndromes)
    assert isinstance(predictions, np.ndarray)
    assert predictions.dtype.type == np.bool_
    assert predictions.shape == (4, 3)

    expected_predictions = np.array([
        [True, False, False],
        [False, True, False],
        [True, False, True],
        [False, False, False],
    ], dtype=bool)
    assert np.array_equal(predictions, expected_predictions)



def shared_test_merge_errors_affects_cost(decoder_class, config_class):
    """
    Test that the error's cost changes based on the 'merge_errors' setting.
    """
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0
        error(0.01) D0
        """
    )
    syndrome = np.array([True], dtype=bool)
    
    config_no_merge = config_class(dem, merge_errors=False)
    decoder_no_merge = decoder_class(config_no_merge)
    predicted_errors_no_merge = decoder_no_merge.decode_to_errors(syndrome)
    cost_no_merge = decoder_no_merge.cost_from_errors(decoder_no_merge.predicted_errors_buffer)
        
    config_merge = config_class(dem, merge_errors=True)
    decoder_merge = decoder_class(config_merge)
    predicted_errors_merge = decoder_merge.decode_to_errors(syndrome)
    cost_merge = decoder_merge.cost_from_errors(decoder_merge.predicted_errors_buffer)
    
    p_merged = 0.1 * (1 - 0.01) + 0.01 * (1 - 0.1)
    expected_cost_no_merge = math.log((1 - 0.1) / 0.1)
    expected_cost_merge = math.log((1 - p_merged) / p_merged)

    assert predicted_errors_no_merge == predicted_errors_merge
    assert cost_no_merge == pytest.approx(expected_cost_no_merge)
    assert cost_merge == pytest.approx(expected_cost_merge)
    assert cost_no_merge != cost_merge

def shared_test_decode_with_mismatched_syndrome_size(decoder_class, config_class):
    """
    Tests that `decode` raises an error when the input syndrome's length does
    not match the number of detectors in the DEM.
    """
    dem_string = f'''
        error(0.1) D0 D1
    '''
    dem = stim.DetectorErrorModel(dem_string)
    config = config_class(dem)
    decoder = decoder_class(config)

    # Syndrome has 1 detector, but DEM has 2
    invalid_syndrome = np.array([True], dtype=bool)
    with pytest.raises(ValueError, match=r"Syndrome array size \(1\) does not match the number of detectors in the decoder \(2\)\."):
        decoder.decode(invalid_syndrome)

def shared_test_decode_batch_with_mismatched_syndrome_size(decoder_class, config_class):
    """
    Tests that `decode_batch` raises an error when the input syndromes' width
    does not match the number of detectors in the DEM.
    """
    dem_string = f'''
        error(0.1) D0 D1
    '''
    dem = stim.DetectorErrorModel(dem_string)
    config = config_class(dem)
    decoder = decoder_class(config)

    # Syndrome batch has 1 column, but DEM has 2
    invalid_syndromes = np.array([[True], [False]], dtype=bool)
    with pytest.raises(ValueError, match=r"The number of detectors in the input array \(1\) does not match the number of detectors in the decoder \(2\)."):
        decoder.decode_batch(invalid_syndromes)
