import pathlib
import pytest
import numpy as np
import stim
import shutil

from src import tesseract_sinter_compat as tesseract_module


def test_tesseract_sinter_obj_exists():
    """
    Sanity check to ensure the decoder object exists and has the required methods.
    """

    decoder = tesseract_module.TesseractSinterDecoder()
    assert hasattr(decoder, 'compile_decoder_for_dem')
    assert hasattr(decoder, 'decode_via_files')

def test_compile_decoder_for_dem():
    """
    Test the 'compile_decoder_for_dem' method with a specific DEM.
    """
    
    dem = stim.DetectorErrorModel("""
        detector(0, 0, 0) D0
        detector(0, 0, 1) D1
        detector(0, 0, 2) D2
        detector(0, 0, 3) D3
        error(0.1) D0 D1 L0
        error(0.1) D1 D2 L1
        error(0.1) D2 D3 L0
    """)
    
    decoder = tesseract_module.TesseractSinterDecoder()
    compiled_decoder = decoder.compile_decoder_for_dem(dem=dem)
    
    assert compiled_decoder is not None
    assert hasattr(compiled_decoder, 'decode_shots_bit_packed')
    
    # Verify the detector and observable counts are correct
    assert compiled_decoder.num_detectors == dem.num_detectors
    assert compiled_decoder.num_observables == dem.num_observables
    
def test_decode_shots_bit_packed():
    """
    Tests the 'decode_shots_bit_packed' method with a specific DEM and detection event.
    """

    dem = stim.DetectorErrorModel("""
        detector(0, 0, 0) D0
        detector(0, 0, 1) D1
        detector(0, 0, 2) D2
        error(0.1) D0 D1 L0
        error(0.1) D1 D2 L1
    """)
    
    decoder = tesseract_module.TesseractSinterDecoder()
    compiled_decoder = decoder.compile_decoder_for_dem(dem=dem)
    
    num_shots = 1    
    detections_array = np.zeros((num_shots, (dem.num_detectors + 7) // 8), dtype=np.uint8)
    
    # Set bits for detectors D0 and D1
    # This should cause a logical flip on L0.
    detections_array[0][0] |= (1 << 0) # D0
    detections_array[0][0] |= (1 << 1) # D1

    predictions = compiled_decoder.decode_shots_bit_packed(bit_packed_detection_event_data=detections_array)
    
    # Extract the expected predictions from the DEM
    expected_predictions = np.zeros((num_shots, (dem.num_observables + 7) // 8), dtype=np.uint8)
    expected_predictions[0][0] |= (1 << 0) # Logical observable L0 is flipped
    
    # Compare the results
    assert np.array_equal(predictions, expected_predictions)
    


def test_decode_shots_bit_packed_multi_shot():
    """
    Tests the 'decode_shots_bit_packed' method with multiple shots.
    """
    dem = stim.DetectorErrorModel("""
        detector(0, 0, 0) D0
        detector(0, 0, 1) D1
        detector(0, 0, 2) D2
        error(0.1) D0 D1 L0
        error(0.1) D1 D2 L1
    """)
    
    decoder = tesseract_module.TesseractSinterDecoder()
    compiled_decoder = decoder.compile_decoder_for_dem(dem=dem)
    
    num_shots = 3
    detections_array = np.zeros((num_shots, (dem.num_detectors + 7) // 8), dtype=np.uint8)
    
    # Shot 0: D0 and D1 fire. Expect L0 to flip.
    detections_array[0][0] |= (1 << 0) # D0
    detections_array[0][0] |= (1 << 1) # D1

    # Shot 1: D1 and D2 fire. Expect L1 to flip.
    detections_array[1][0] |= (1 << 1) # D1
    detections_array[1][0] |= (1 << 2) # D2
    
    # Shot 2: D0 and D2 fire. Expect L0 and L1 to flip.
    detections_array[2][0] |= (1 << 0) # D0
    detections_array[2][0] |= (1 << 2) # D2

    predictions = compiled_decoder.decode_shots_bit_packed(bit_packed_detection_event_data=detections_array)
    
    expected_predictions = np.zeros((num_shots, (dem.num_observables + 7) // 8), dtype=np.uint8)
    # Expected flip for shot 0 is L0
    expected_predictions[0][0] |= (1 << 0)
    # Expected flip for shot 1 is L1
    expected_predictions[1][0] |= (1 << 1)
    # Expected flip for shot 2 is L0 and L1
    expected_predictions[2][0] |= (1 << 0)
    expected_predictions[2][0] |= (1 << 1)
    
    assert np.array_equal(predictions, expected_predictions)


def test_decode_via_files_sanity_check():
    """
    Tests the 'decode_via_files' method by simulating a small circuit and
    checking for output files.
    """
    
    # Create a temporary directory for test files
    temp_dir = pathlib.Path("./temp_test_files")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    dem_path = temp_dir / "test.dem"
    dets_in_path = temp_dir / "test.b8"
    obs_out_path = temp_dir / "test.out.b8"

    # Create a small circuit and DEM file
    circuit = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=2)
    dem = circuit.detector_error_model()
    with open(dem_path, 'w') as f:
        f.write(str(dem))

    # Generate dummy detection events and save to file
    num_shots = 10
    sampler = circuit.compile_detector_sampler()
    detection_events = sampler.sample(num_shots, bit_packed=True)
    with open(dets_in_path, 'wb') as f:
        f.write(detection_events.tobytes())

    tesseract_module.TesseractSinterDecoder().decode_via_files(
        num_shots=num_shots,
        num_dets=dem.num_detectors,
        num_obs=dem.num_observables,
        dem_path=str(dem_path),
        dets_b8_in_path=str(dets_in_path),
        obs_predictions_b8_out_path=str(obs_out_path),
        tmp_dir=str(temp_dir)
    )

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

def test_decode_via_files():
    """
    Tests the 'decode_via_files' method with a specific DEM and detection event.
    """
    
    # Create a temporary directory for test files
    temp_dir = pathlib.Path("./temp_test_files")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    dem_path = temp_dir / "test.dem"
    dets_in_path = temp_dir / "test.b8"
    obs_out_path = temp_dir / "test.out.b8"
    
    # Create a specific DEM
    dem_string = """
        detector(0, 0, 0) D0
        detector(0, 0, 1) D1
        detector(0, 0, 2) D2
        detector(0, 0, 3) D3
        error(0.1) D0 D1 L0
        error(0.1) D1 D2 L1
        error(0.1) D2 D3 L0
    """
    dem = stim.DetectorErrorModel(dem_string)
    
    # Write the DEM string to a file
    with open(dem_path, 'w') as f:
        f.write(dem_string)

    detections = [0, 1]
    expected_predictions = np.zeros(dem.num_observables, dtype=np.uint8)
    expected_predictions[0] = 1 # Flip on L0
    
    # Pack the detection events into a bit-packed NumPy array
    num_shots = 1
    num_detectors = dem.num_detectors
    detection_events_np = np.zeros(num_shots * ((num_detectors + 7) // 8), dtype=np.uint8)
    for d_idx in detections:
        detection_events_np[d_idx // 8] ^= (1 << (d_idx % 8))

    # Write the packed detection events to the input file
    with open(dets_in_path, 'wb') as f:
        f.write(detection_events_np.tobytes())

    tesseract_module.TesseractSinterDecoder().decode_via_files(
        num_shots=num_shots,
        num_dets=num_detectors,
        num_obs=dem.num_observables,
        dem_path=str(dem_path),
        dets_b8_in_path=str(dets_in_path),
        obs_predictions_b8_out_path=str(obs_out_path),
        tmp_dir=str(temp_dir)
    )
    
    # Read the output file and unpack the results
    with open(obs_out_path, 'rb') as f:
        predictions_bytes = f.read()
    
    # Convert bytes to a numpy array for easy comparison
    predictions_np = np.frombuffer(predictions_bytes, dtype=np.uint8)
    unpacked_predictions = np.zeros(dem.num_observables, dtype=np.uint8)
    for i in range(dem.num_observables):
        if (predictions_np[i // 8] >> (i % 8)) & 1:
            unpacked_predictions[i] = 1

    assert np.array_equal(unpacked_predictions, expected_predictions)
            
    # Clean up temporary files
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
            
def test_decode_via_files_multi_shot():
    """
    Tests the 'decode_via_files' method with multiple shots and a specific DEM.
    """
    # Create a temporary directory for test files
    temp_dir = pathlib.Path("./temp_test_files")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    dem_path = temp_dir / "test.dem"
    dets_in_path = temp_dir / "test.b8"
    obs_out_path = temp_dir / "test.out.b8"
    
    # Create a specific DEM
    dem_string = """
        detector(0, 0, 0) D0
        detector(0, 0, 1) D1
        detector(0, 0, 2) D2
        error(0.1) D0 D1 L0
        error(0.1) D1 D2 L1
    """
    dem = stim.DetectorErrorModel(dem_string)
    
    # Write the DEM string to a file
    with open(dem_path, 'w') as f:
        f.write(dem_string)

    num_shots = 3
    num_detectors = dem.num_detectors
    detection_events_np = np.zeros(num_shots * ((num_detectors + 7) // 8), dtype=np.uint8)

    # Shot 0: D0 and D1 fire. Expected L0 flip.
    detection_events_np[0] |= (1 << 0)
    detection_events_np[0] |= (1 << 1)

    # Shot 1: D1 and D2 fire. Expected L1 flip.
    detection_events_np[1] |= (1 << 1)
    detection_events_np[1] |= (1 << 2)

    # Shot 2: D0 and D2 fire. Expected L0 and L1 flips.
    detection_events_np[2] |= (1 << 0)
    detection_events_np[2] |= (1 << 2)

    # Write the packed detection events to the input file
    with open(dets_in_path, 'wb') as f:
        f.write(detection_events_np.tobytes())

    tesseract_module.TesseractSinterDecoder().decode_via_files(
        num_shots=num_shots,
        num_dets=num_detectors,
        num_obs=dem.num_observables,
        dem_path=str(dem_path),
        dets_b8_in_path=str(dets_in_path),
        obs_predictions_b8_out_path=str(obs_out_path),
        tmp_dir=str(temp_dir)
    )
    
    # Read the output file and unpack the results
    with open(obs_out_path, 'rb') as f:
        predictions_bytes = f.read()

    predictions_np = np.frombuffer(predictions_bytes, dtype=np.uint8)
    
    expected_predictions_np = np.zeros(num_shots * ((dem.num_observables + 7) // 8), dtype=np.uint8)
    expected_predictions_np[0] |= (1 << 0)
    expected_predictions_np[1] |= (1 << 1)
    expected_predictions_np[2] |= (1 << 0)
    expected_predictions_np[2] |= (1 << 1)

    assert np.array_equal(predictions_np, expected_predictions_np)

    # Clean up temporary files
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))