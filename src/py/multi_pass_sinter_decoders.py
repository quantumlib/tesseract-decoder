import sinter
import stim
import tesseract_decoder as _core

class MultiPassSinterDecoder(sinter.Decoder):
    """
    A sinter-compatible Multi-Pass Tesseract Decoder.
    Wraps the native C++ MultiPassTesseractDecoder.
    """
    def __init__(self, num_passes: int = 2, detector_classifier=None, **base_config_kwargs):
        if num_passes not in (1, 2):
            raise ValueError("num_passes must be 1 or 2.")
        self.num_passes = num_passes
        self.detector_classifier = detector_classifier
        self.base_config_kwargs = base_config_kwargs

    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        # 1. Access the native C++ class
        cpp_decoder = _core.MultiPassSinterDecoder(num_passes=self.num_passes)

        # 2. Attach the classifier if provided
        if self.detector_classifier is not None:
            cpp_decoder.detector_classifier = self.detector_classifier
        else:
            def default_classifier(index: int, coords: list[float], tag: str) -> int:
                if '"basis": "X"' in tag:
                    return 0
                if '"basis": "Z"' in tag:
                    return 1
                if len(coords) >= 4:
                    c3 = int(coords[3])
                    if 0 <= c3 <= 2:
                        return 0
                    if 3 <= c3 <= 5:
                        return 1
                return -1
            cpp_decoder.detector_classifier = default_classifier

        # 3. Apply base configuration (pqlimit, det_beam, etc.)
        for key, value in self.base_config_kwargs.items():
            if hasattr(cpp_decoder.base_config, key):
                setattr(cpp_decoder.base_config, key, value)
            elif hasattr(cpp_decoder, key):
                setattr(cpp_decoder, key, value)

        # 4. Compile and return the native CompiledDecoder
        return cpp_decoder.compile_decoder_for_dem(dem=dem)

def get_sinter_decoders():
    TesseractSinterDecoder = _core.TesseractSinterDecoder
    return {
        "tesseract_mono": TesseractSinterDecoder(
            det_beam=20,
            beam_climbing=True,
            no_revisit_dets=True,
            merge_errors=True,
            pqlimit=1000000,
            num_det_orders=21,
            seed=2384753
        ),
        "tesseract_multipass_1pass": MultiPassSinterDecoder(
            num_passes=1,
            strategy=_core.Causal,
            det_beam=20,
            beam_climbing=True,
            no_revisit_dets=True,
            merge_errors=True,
            pqlimit=1000000,
            num_det_orders=21,
            seed=2384753
        ),
        "tesseract_multipass_2pass": MultiPassSinterDecoder(
            num_passes=2,
            strategy=_core.Causal,
            det_beam=20,
            beam_climbing=True,
            no_revisit_dets=True,
            merge_errors=True,
            pqlimit=1000000,
            num_det_orders=21,
            seed=2384753
        ),
    }
