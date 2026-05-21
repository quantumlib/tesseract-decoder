import tesseract_decoder
import stim
import numpy as np
import sys

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
    decoder.detector_classifier = lambda index, coords, tag: index
    
    compiled = decoder.compile_decoder_for_dem(dem=dem)
    
    # D0 and D1 both fire. Bit-packed: 0b11 = 3
    dets = np.array([[3]], dtype=np.uint8)
    predictions = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=dets)
    
    print(f"Predictions: {predictions}", flush=True)
    assert (predictions[0, 0] & 1) == 1

    # 2. Test with Full Decomposer
    print("Testing with full decomposer...", flush=True)
    def my_decomposer(input_dem):
        print("Full decomposer called!", flush=True)
        return input_dem
        
    decoder.detector_classifier = None
    decoder.full_decomposer = my_decomposer
    compiled = decoder.compile_decoder_for_dem(dem=dem)
    predictions = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=dets)
    print(f"Predictions: {predictions}", flush=True)
    assert (predictions[0, 0] & 1) == 1

if __name__ == "__main__":
    try:
        test_multi_pass_sinter_bindings()
        print("Python bindings test PASSED", flush=True)
    except Exception as e:
        print(f"Python bindings test FAILED: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
