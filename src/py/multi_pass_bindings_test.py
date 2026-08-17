import tesseract_decoder
import stim
import numpy as np
import sys
from multi_pass_sinter_decoders import MultiPassSinterDecoder as PythonMultiPassSinterDecoder

def test_multi_pass_sinter_bindings():
    print(f"Loaded tesseract_decoder from: {tesseract_decoder.__file__}", flush=True)
    
    dem = stim.DetectorErrorModel(R"""
        error(0.1) D0 ^ D1 L0
        error(0.01) D0
        error(0.2) D1 L0
        detector D0
        detector D1
        logical_observable L0
    """)

    # 1. Test with Detector Classifier Lambda
    print("Testing MultiPassSinterDecoder with lambda...", flush=True)
    decoder = tesseract_decoder.MultiPassSinterDecoder(num_passes=2)
    assert decoder.strategy == tesseract_decoder.Causal
    decoder.detector_classifier = lambda index, coords, tag: index

    assert PythonMultiPassSinterDecoder().strategy == tesseract_decoder.Causal
    python_static_decoder = PythonMultiPassSinterDecoder(strategy=tesseract_decoder.Static)
    assert python_static_decoder.strategy == tesseract_decoder.Static

    fallback_dem = stim.DetectorErrorModel(R"""
        error(0.1) D0
        error(0.2) D1 L0
        detector[{"measure_basis": 0, "basis": "X"}] D0
        detector[{"measure_basis": "Y"}](0, 0, 0, 3) D1
        logical_observable L0
    """)
    PythonMultiPassSinterDecoder().compile_decoder_for_dem(dem=fallback_dem)
    
    compiled = decoder.compile_decoder_for_dem(dem=dem)
    
    # D0 and D1 both fire. Bit-packed: 0b11 = 3
    dets = np.array([[3]], dtype=np.uint8)
    predictions = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=dets)
    
    print(f"Predictions: {predictions}", flush=True)
    assert (predictions[0, 0] & 1) == 1

    # 2. A decomposer does not replace the required detector classification.
    print("Testing missing classifier rejection...", flush=True)
    def my_decomposer(input_dem):
        print("Full decomposer called!", flush=True)
        return input_dem
        
    decoder.detector_classifier = None
    decoder.full_decomposer = my_decomposer
    try:
        decoder.compile_decoder_for_dem(dem=dem)
        raise AssertionError("Expected detector_classifier to be required")
    except ValueError as error:
        assert "detector_classifier" in str(error)

if __name__ == "__main__":
    try:
        test_multi_pass_sinter_bindings()
        print("Python bindings test PASSED", flush=True)
    except Exception as e:
        print(f"Python bindings test FAILED: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
