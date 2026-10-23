import pytest
import numpy as np
import stim
from tesseract_decoder import bp, bp_sinter_compat

def test_bp_standalone():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0
        error(0.2) D1 L0
    """)
    params = bp.BPParams()
    params.max_iter = 10

    decoder = bp.TesseractBpDecoder(dem, params)

    syndrome = np.array([True, False], dtype=bool)
    predictions = decoder.decode(syndrome)
    assert len(predictions) == 1
    assert not predictions[0]

def test_bp_sinter_compat():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0
        error(0.2) D1 L0
    """)
    params = bp.BPParams()
    params.max_iter = 10

    factory = bp_sinter_compat.TesseractBpSinterDecoder(params)
    compiled = factory.compile_decoder_for_dem(dem=dem)

    shots = np.array([[0x01], [0x02]], dtype=np.uint8)
    predictions = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=shots)

    assert predictions.shape == (2, 1)
    assert predictions[0, 0] == 0x00
    assert predictions[1, 0] == 0x01

def test_bp_osd_standalone():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0
        error(0.2) D1 L0
    """)
    params = bp.BPParams()
    params.max_iter = 10

    decoder = bp.TesseractBpDecoder(dem, params)
    osd = decoder.create_osd_post_processor(osd_order=10, osd_weight=0)

    syndrome = np.array([True, False], dtype=bool)
    predictions = decoder.decode(syndrome, osd)
    assert len(predictions) == 1
    assert not predictions[0]

def test_bp_osd_sinter_compat():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0
        error(0.2) D1 L0
    """)
    params = bp.BPParams()
    params.max_iter = 10

    factory = bp_sinter_compat.TesseractBpSinterDecoder(params, osd_order=10, osd_weight=0)
    compiled = factory.compile_decoder_for_dem(dem=dem)

    shots = np.array([[0x01], [0x02]], dtype=np.uint8)
    predictions = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=shots)

    assert predictions[0, 0] == 0x00
    assert predictions[1, 0] == 0x01

def test_bp_more_than_64_observables():
    dem_str = "error(0.1)"
    for i in range(70):
        dem_str += f" D{i} L{i}"
    dem_str += "\n"
    dem = stim.DetectorErrorModel(dem_str)

    params = bp.BPParams()
    params.max_iter = 10

    factory = bp_sinter_compat.TesseractBpSinterDecoder(params)
    compiled = factory.compile_decoder_for_dem(dem=dem)

    num_det_bytes = (70 + 7) // 8
    shots = np.full((1, num_det_bytes), 0xFF, dtype=np.uint8)

    predictions = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=shots)

    num_obs_bytes = (70 + 7) // 8
    assert predictions.shape == (1, num_obs_bytes)

def test_bp_osd_presets_and_multipass():
    dem = stim.DetectorErrorModel("""
        detector[{"measure_basis": "X"}] D0
        detector[{"measure_basis": "Z"}] D1
        error(0.1) D0 D1 L0
        error(0.01) D0
        error(0.02) D1 L0
        logical_observable L0
    """)
    shots = np.array([[0x03], [0x01]], dtype=np.uint8)

    for factory in [
        bp_sinter_compat.TesseractBpSinterDecoder.BpOsd(max_iter=10),
        bp_sinter_compat.TesseractBpSinterDecoder.BpOsdFast(max_iter=5),
        bp_sinter_compat.TesseractBpSinterDecoder.MultiPassBpOsd(max_iter=10),
        bp_sinter_compat.TesseractBpSinterDecoder.MultiPassBpOsdFast(max_iter=5),
    ]:
        compiled = factory.compile_decoder_for_dem(dem=dem)
        preds = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=shots)
        assert preds.shape == (2, 1)
        assert preds[0, 0] == 0x01
        assert preds[1, 0] == 0x00

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
