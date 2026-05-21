import sinter
import stim
from . import _core

class MultiPassSinterDecoder(sinter.Decoder):
    """
    A sinter-compatible Multi-Pass Tesseract Decoder.
    Wraps the native C++ MultiPassTesseractDecoder.
    """
    def __init__(self, num_passes: int = 2, detector_classifier=None, **base_config_kwargs):
        self.num_passes = num_passes
        self.detector_classifier = detector_classifier
        self.base_config_kwargs = base_config_kwargs

    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        # 1. Access the native C++ class
        cpp_decoder = _core.MultiPassSinterDecoder(num_passes=self.num_passes)
        
        # 2. Attach the classifier if provided
        if self.detector_classifier is not None:
            cpp_decoder.detector_classifier = self.detector_classifier
        
        # 3. Apply base configuration (pqlimit, det_beam, etc.)
        for key, value in self.base_config_kwargs.items():
            if hasattr(cpp_decoder.base_config, key):
                setattr(cpp_decoder.base_config, key, value)
            elif hasattr(cpp_decoder, key):
                setattr(cpp_decoder, key, value)

        # 4. Compile and return the native CompiledDecoder
        return cpp_decoder.compile_decoder_for_dem(dem=dem)
