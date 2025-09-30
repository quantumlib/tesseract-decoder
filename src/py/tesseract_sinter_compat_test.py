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

import pathlib
import pytest
import numpy as np
import stim
import shutil
from sinter._decoding._decoding import sample_decode

from src.tesseract_decoder import tesseract_sinter_compat as tesseract_module
from src import tesseract_decoder
import sinter


def test_tesseract_sinter_obj_exists():
    """
    Sanity check to ensure the decoder object exists and has the required methods.
    """

    decoder = tesseract_module.TesseractSinterDecoder()
    assert hasattr(decoder, 'compile_decoder_for_dem')
    assert hasattr(decoder, 'decode_via_files')


@pytest.mark.parametrize("use_custom_config", [False, True])
def test_compile_decoder_for_dem(use_custom_config):
    """
    Test the 'compile_decoder_for_dem' method with and without a custom config.
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

    if use_custom_config:
        decoder = tesseract_module.TesseractSinterDecoder(
            verbose=True,
        )
    else:
        decoder = tesseract_module.TesseractSinterDecoder()

    compiled_decoder = decoder.compile_decoder_for_dem(dem=dem)

    assert compiled_decoder is not None
    assert hasattr(compiled_decoder, 'decode_shots_bit_packed')

    # Verify the detector and observable counts are correct
    assert compiled_decoder.num_detectors == dem.num_detectors
    assert compiled_decoder.num_observables == dem.num_observables

    # Verify the config was correctly applied
    assert compiled_decoder.decoder.config.verbose == use_custom_config


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
    detections_array = np.zeros(
        (num_shots, (dem.num_detectors + 7) // 8), dtype=np.uint8)

    # Set bits for detectors D0 and D1
    # This should cause a logical flip on L0.
    detections_array[0][0] |= (1 << 0)  # D0
    detections_array[0][0] |= (1 << 1)  # D1

    predictions = compiled_decoder.decode_shots_bit_packed(
        bit_packed_detection_event_data=detections_array)

    # Extract the expected predictions from the DEM
    expected_predictions = np.zeros(
        (num_shots, (dem.num_observables + 7) // 8), dtype=np.uint8)
    expected_predictions[0][0] |= (1 << 0)  # Logical observable L0 is flipped

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
    detections_array = np.zeros(
        (num_shots, (dem.num_detectors + 7) // 8), dtype=np.uint8)

    # Shot 0: D0 and D1 fire. Expect L0 to flip.
    detections_array[0][0] |= (1 << 0)  # D0
    detections_array[0][0] |= (1 << 1)  # D1

    # Shot 1: D1 and D2 fire. Expect L1 to flip.
    detections_array[1][0] |= (1 << 1)  # D1
    detections_array[1][0] |= (1 << 2)  # D2

    # Shot 2: D0 and D2 fire. Expect L0 and L1 to flip.
    detections_array[2][0] |= (1 << 0)  # D0
    detections_array[2][0] |= (1 << 2)  # D2

    predictions = compiled_decoder.decode_shots_bit_packed(
        bit_packed_detection_event_data=detections_array)

    expected_predictions = np.zeros(
        (num_shots, (dem.num_observables + 7) // 8), dtype=np.uint8)
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
    circuit = stim.Circuit.generated(
        "repetition_code:memory", distance=3, rounds=2)
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


@pytest.mark.parametrize("use_custom_config", [False, True])
def test_decode_via_files(use_custom_config):
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
    expected_predictions[0] = 1  # Flip on L0

    # Pack the detection events into a bit-packed NumPy array
    num_shots = 1
    num_detectors = dem.num_detectors
    detection_events_np = np.zeros(
        num_shots * ((num_detectors + 7) // 8), dtype=np.uint8)
    for d_idx in detections:
        detection_events_np[d_idx // 8] ^= (1 << (d_idx % 8))

    # Write the packed detection events to the input file
    with open(dets_in_path, 'wb') as f:
        f.write(detection_events_np.tobytes())

    if use_custom_config:
        decoder = tesseract_module.TesseractSinterDecoder(
            verbose=True,
        )
    else:
        decoder = tesseract_module.TesseractSinterDecoder()

    decoder.decode_via_files(
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

    assert decoder.verbose == use_custom_config


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
    detection_events_np = np.zeros(
        num_shots * ((num_detectors + 7) // 8), dtype=np.uint8)

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

    expected_predictions_np = np.zeros(
        num_shots * ((dem.num_observables + 7) // 8), dtype=np.uint8)
    expected_predictions_np[0] |= (1 << 0)
    expected_predictions_np[1] |= (1 << 1)
    expected_predictions_np[2] |= (1 << 0)
    expected_predictions_np[2] |= (1 << 1)

    assert np.array_equal(predictions_np, expected_predictions_np)

    # Clean up temporary files
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def test_sinter_decode_repetition_code():
    """
    Tests the 'tesseract' decoder on a repetition code circuit.
    """
    circuit = stim.Circuit.generated('repetition_code:memory',
                                     rounds=3,
                                     distance=3,
                                     after_clifford_depolarization=0.05)

    result = sample_decode(
        circuit_obj=circuit,
        circuit_path=None,
        dem_obj=circuit.detector_error_model(decompose_errors=True),
        dem_path=None,
        num_shots=1000,
        decoder="tesseract",
        custom_decoders=tesseract_module.make_tesseract_sinter_decoders_dict(),
    )
    assert result.discards == 0
    assert 0 <= result.errors <= 100
    assert result.shots == 1000


def test_sinter_decode_surface_code():
    """
    Tests the 'tesseract' decoder on a more complex surface code circuit.
    """
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=3,
        rounds=15,
        after_clifford_depolarization=0.001,
    )
    result = sample_decode(
        num_shots=1000,
        circuit_obj=circuit,
        circuit_path=None,
        dem_obj=circuit.detector_error_model(decompose_errors=True),
        dem_path=None,
        decoder="tesseract",
        custom_decoders=tesseract_module.make_tesseract_sinter_decoders_dict(),
    )
    assert result.discards == 0
    assert 0 <= result.errors <= 50
    assert result.shots == 1000


def test_sinter_empty():
    """
    Tests the 'tesseract' decoder on an empty circuit.
    """
    circuit = stim.Circuit()
    result = sample_decode(
        circuit_obj=circuit,
        circuit_path=None,
        dem_obj=circuit.detector_error_model(decompose_errors=True),
        dem_path=None,
        num_shots=1000,
        decoder="tesseract",
        custom_decoders=tesseract_module.make_tesseract_sinter_decoders_dict(),
    )
    assert result.discards == 0
    assert result.shots == 1000
    assert result.errors == 0


def test_sinter_no_observables():
    """
    Tests the decoder on a circuit with detectors but no logical observables.
    """
    circuit = stim.Circuit("""
        X_ERROR(0.1) 0
        M 0
        DETECTOR rec[-1]
    """)
    result = sample_decode(
        circuit_obj=circuit,
        circuit_path=None,
        dem_obj=circuit.detector_error_model(decompose_errors=True),
        dem_path=None,
        num_shots=1000,
        decoder="tesseract",
        custom_decoders=tesseract_module.make_tesseract_sinter_decoders_dict(),
    )
    assert result.discards == 0
    assert result.shots == 1000
    assert result.errors == 0


def test_sinter_invincible_observables():
    """
    Tests the decoder on a circuit where an observable is not affected by errors.
    """
    circuit = stim.Circuit("""
        X_ERROR(0.1) 0
        M 0 1
        DETECTOR rec[-2]
        OBSERVABLE_INCLUDE(1) rec[-1]
    """)
    result = sample_decode(
        circuit_obj=circuit,
        circuit_path=None,
        dem_obj=circuit.detector_error_model(decompose_errors=True),
        dem_path=None,
        num_shots=1000,
        decoder="tesseract",
        custom_decoders=tesseract_module.make_tesseract_sinter_decoders_dict(),
    )
    assert result.discards == 0
    assert result.shots == 1000
    assert result.errors == 0


def test_sinter_detector_counting():
    """
    Tests 'that the decoder's detector count is correctly reported via Sinter'.
    """
    circuit = stim.Circuit("""
        X_ERROR(0.1) 0
        X_ERROR(0.2) 1
        M 0 1
        DETECTOR rec[-1]
        DETECTOR rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
        OBSERVABLE_INCLUDE(1) rec[-1] rec[-2]
    """)
    result = sample_decode(
        circuit_obj=circuit,
        circuit_path=None,
        dem_obj=circuit.detector_error_model(decompose_errors=True),
        dem_path=None,
        post_mask=None,
        num_shots=10000,
        decoder="tesseract",
        count_detection_events=True,
        custom_decoders=tesseract_module.make_tesseract_sinter_decoders_dict(),
    )
    assert result.discards == 0
    assert result.custom_counts['detectors_checked'] == 20000
    assert 0.3 * 10000 * \
        0.5 <= result.custom_counts['detection_events'] <= 0.3 * 10000 * 2.0
    assert set(result.custom_counts.keys()) == {
        'detectors_checked', 'detection_events'}


def test_full_scale():
    result, = sinter.collect(
        num_workers=2,
        tasks=[sinter.Task(circuit=stim.Circuit())],
        decoders=["tesseract"],
        max_shots=1000,
        custom_decoders=tesseract_module.make_tesseract_sinter_decoders_dict(),
    )
    assert result.discards == 0
    assert result.shots == 1000
    assert result.errors == 0


def test_full_scale_one_worker():
    # Create a repetition code circuit to test the decoder.
    circuit = stim.Circuit.generated(
        'repetition_code:memory',
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01
    )

    # Use sinter.collect to run the decoding task.
    result, = sinter.collect(
        num_workers=1,
        tasks=[sinter.Task(circuit=circuit)],
        decoders=["tesseract"],
        max_shots=1000,
        custom_decoders=tesseract_module.make_tesseract_sinter_decoders_dict(),
    )

    assert result.discards == 0
    assert result.shots == 1000


def relabel_logical_observables(
    circuit: stim.Circuit,
    relabel_dict: dict[int, int]
) -> stim.Circuit:
    new_circuit = stim.Circuit()
    for inst in circuit:
        if inst.name == "OBSERVABLE_INCLUDE":
            args = inst.gate_args_copy()
            new_args = [relabel_dict[args[0]]]
            new_inst = stim.CircuitInstruction(
                name=inst.name,
                targets=inst.targets_copy(
                ),
                gate_args=new_args,
                tag=inst.tag
            )
            inst = new_inst
        new_circuit.append(inst)
    return new_circuit


@pytest.mark.parametrize(
    "det_beam, beam_climbing, no_revisit_dets, merge_errors",
    [
        # Some standard values
        (20, False, False, True),
        # Beam climbing enabled
        (20, True, False, True),
        # No revisit detectors enabled
        (20, False, True, True),
        # Merge errors disabled
        (20, False, False, False),
    ]
)
def test_decode_shots_bit_packed_vs_decode_batch(det_beam, beam_climbing, no_revisit_dets, merge_errors):
    """
    Compares the output of the Sinter decoder interface against the raw Tesseract decoder
    to ensure they produce identical results across different configurations.
    """

    # 1. Set up the quantum circuit and detector error model.
    p = 0.02
    circuit = stim.Circuit.generated(
        "color_code:memory_xyz",
        distance=3,
        rounds=3,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
        after_reset_flip_probability=p
    )
    circuit = relabel_logical_observables(circuit=circuit, relabel_dict={0: 3})
    dem = circuit.detector_error_model()

    # 2. Compile the Sinter-compatible decoder with the parameterized values for the DEM.
    sinter_decoder = tesseract_module.TesseractSinterDecoder(
        det_beam=det_beam,
        beam_climbing=beam_climbing,
        no_revisit_dets=no_revisit_dets,
        merge_errors=merge_errors,
    )

    # 3. Compile the Sinter-compatible decoder.
    compiled_sinter_decoder = sinter_decoder.compile_decoder_for_dem(dem=dem)

    # 4. Obtain the compiled decoder from the config.
    config = tesseract_decoder.tesseract.TesseractConfig(
        dem=dem,
        det_beam=det_beam,
        beam_climbing=beam_climbing,
        no_revisit_dets=no_revisit_dets,
        merge_errors=merge_errors,
    )
    decoder = config.compile_decoder()

    # 5. Generate a batch of shots and unpack them for comparison.
    sampler = circuit.compile_detector_sampler()
    bitpacked_shots, _ = sampler.sample(
        shots=1000, separate_observables=True, bit_packed=True)
    unpacked_shots = np.unpackbits(bitpacked_shots, bitorder='little', axis=1)

    # 6. Decode the shots using both methods.
    predictions_sinter_bitpacked = compiled_sinter_decoder.decode_shots_bit_packed(
        bit_packed_detection_event_data=bitpacked_shots)
    predictions_sinter = np.unpackbits(
        predictions_sinter_bitpacked, bitorder='little', axis=1)[:, :dem.num_observables]

    predictions_decode_batch = decoder.decode_batch(
        unpacked_shots[:, :dem.num_detectors])
    # 7. Assert that the predictions from both decoders are identical.
    assert np.array_equal(predictions_sinter, predictions_decode_batch)


def test_sinter_collect_different_dems():
    """
    Ensures that Sinter tasks compile with different DEMs before collection.
    """
    # Create a repetition code circuit to test the decoder.
    min_distance = 3
    max_distance = 7
    tasks = [
        sinter.Task(
            circuit=stim.Circuit.generated(
                "repetition_code:memory",
                distance=d,
                rounds=3,
                after_clifford_depolarization=0.1,
            ),
            json_metadata={"d": d},
        )
        for d in range(min_distance, max_distance + 1, 2)
    ]

    # Use sinter.collect to run the decoding task.
    all_results = sinter.collect(
        num_workers=1,
        tasks=tasks,
        decoders=["tesseract-long-beam"],
        max_shots=100,  # Reduced max_shots for testing
        custom_decoders=tesseract_module.make_tesseract_sinter_decoders_dict()
    )

    assert len(all_results) == len(tasks)
    expected_distances = [3,5,7]
    for i, results in enumerate(all_results):
        assert results.json_metadata['d'] == expected_distances[i]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
