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
    params.update_rule = "min-sum"
    params.schedule = "parallel"
    
    decoder = bp.TesseractBpDecoder(dem, params)
    
    # Syndrome: D0 fired, D1 didn't. (Expect L0 to NOT flip!)
    syndrome = np.array([True, False], dtype=bool)
    
    post_processor = bp.HardDecisionPostProcessor()
    
    print("Syndrome:", syndrome)
    predictions = decoder.decode(syndrome, post_processor)
    print("Predictions:", predictions)
    # With D0 fired (0.1 error), we expect L0 to NOT flip.
    assert len(predictions) == 1
    assert not predictions[0]

def test_bp_sinter_compat():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0
        error(0.2) D1 L0
    """)
    params = bp.BPParams()
    params.max_iter = 10
    params.update_rule = "min-sum"
    params.schedule = "parallel"
    
    factory = bp_sinter_compat.TesseractBpSinterDecoder(params) # Default to HardDecision
    compiled = factory.compile_decoder_for_dem(dem=dem)
    
    # 2 shots, 1 byte of detectors each (we have 2 detectors)
    # Shot 0: D0 fired (0x01)
    # Shot 1: D1 fired (0x02)
    shots = np.array([[0x01], [0x02]], dtype=np.uint8)
    
    predictions = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=shots)
    
    # 2 shots, 1 byte of observables each (we have 1 observable)
    assert predictions.shape == (2, 1)
    
    # Shot 0 (D0 fired): L0 shouldn't flip (0x00)
    # Shot 1 (D1 fired): L0 should flip (0x01)
    assert predictions[0, 0] == 0x00
    assert predictions[1, 0] == 0x01

def test_bp_serial_standalone():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0
        error(0.2) D1 L0
    """)
    params = bp.BPParams()
    params.max_iter = 10
    params.update_rule = "min-sum"
    params.schedule = "serial" # Using serial
    
    decoder = bp.TesseractBpDecoder(dem, params)
    
    syndrome = np.array([True, False], dtype=bool) # D0 fired
    post_processor = bp.HardDecisionPostProcessor()
    
    predictions = decoder.decode(syndrome, post_processor)
    assert len(predictions) == 1
    assert not predictions[0]

def test_bp_serial_sinter_compat():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0
        error(0.2) D1 L0
    """)
    params = bp.BPParams()
    params.max_iter = 10
    params.update_rule = "min-sum"
    params.schedule = "serial" # Using serial
    
    factory = bp_sinter_compat.TesseractBpSinterDecoder(params)
    compiled = factory.compile_decoder_for_dem(dem=dem)
    
    shots = np.array([[0x01], [0x02]], dtype=np.uint8)
    predictions = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=shots)
    
    assert predictions[0, 0] == 0x00
    assert predictions[1, 0] == 0x01

def test_bp_osd_standalone():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0
        error(0.2) D1 L0
    """)
    params = bp.BPParams()
    params.max_iter = 10
    params.update_rule = "min-sum"
    
    decoder = bp.TesseractBpDecoder(dem, params)
    
    # Create OSD post processor using factory method!
    osd = decoder.create_osd_post_processor(osd_order=10, osd_weight=0) # OSD-0
    
    syndrome = np.array([True, False], dtype=bool) # D0 fired
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
    params.update_rule = "min-sum"
    
    # Enable OSD-0 in SinterCompat
    factory = bp_sinter_compat.TesseractBpSinterDecoder(params, osd_order=10, osd_weight=0)
    compiled = factory.compile_decoder_for_dem(dem=dem)
    
    shots = np.array([[0x01], [0x02]], dtype=np.uint8)
    predictions = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=shots)
    
    # D0 fired -> L0 shouldn't flip
    # D1 fired -> L0 should flip
    assert predictions[0, 0] == 0x00
    assert predictions[1, 0] == 0x01

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
