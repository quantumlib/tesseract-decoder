import sinter
import stim
import tesseract_decoder as _core

class MultiPassSinterDecoder(sinter.Decoder):
    """A Sinter-compatible wrapper around the native multi-pass Tesseract decoder.

    Standard Tesseract configuration arguments can be passed through
    ``**base_config_kwargs``.
    """
    def __init__(self, num_passes: int = 2, detector_classifier=None,
                 strategy=_core.Causal, **base_config_kwargs):
        if num_passes not in (1, 2):
            raise ValueError("num_passes must be 1 or 2.")
        self.num_passes = num_passes
        self.detector_classifier = detector_classifier
        self.strategy = strategy
        self.base_config_kwargs = base_config_kwargs

    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        # 1. Access the native C++ class
        cpp_decoder = _core.MultiPassSinterDecoder(num_passes=self.num_passes)
        cpp_decoder.strategy = self.strategy

        # 2. Attach the classifier if provided
        if self.detector_classifier is not None:
            cpp_decoder.detector_classifier = self.detector_classifier
        else:
            def default_classifier(index: int, coords: list[float], tag: str) -> int:
                import json
                # Priority 1: Parse JSON tag for "measure_basis" then "basis".
                # Supports both top-level keys and keys nested under "md".
                if tag:
                    try:
                        tag_data = json.loads(tag)
                        if isinstance(tag_data, dict):
                            md = tag_data.get("md", {})
                            if not isinstance(md, dict):
                                md = {}

                            basis_fields = (
                                (tag_data, "measure_basis"),
                                (md, "measure_basis"),
                                (tag_data, "basis"),
                                (md, "basis"),
                            )
                            for metadata, key in basis_fields:
                                if key not in metadata:
                                    continue
                                if metadata[key] == "X":
                                    return 0
                                if metadata[key] == "Z":
                                    return 1
                                return -1
                    except json.JSONDecodeError:
                        pass
                # Priority 3: Chromobius-style coordinate convention.
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
        "tesseract-long-beam-mono": TesseractSinterDecoder(
            det_beam=20,
            beam_climbing=True,
            no_revisit_dets=True,
            merge_errors=True,
            pqlimit=1000000,
            num_det_orders=21,
            det_order_method=_core.utils.DetOrder.DetIndex,
            seed=2384753
        ),
        "tesseract-long-beam-multipass-1pass": MultiPassSinterDecoder(
            num_passes=1,
            strategy=_core.Causal,
            det_beam=20,
            beam_climbing=True,
            no_revisit_dets=True,
            merge_errors=True,
            pqlimit=1000000,
            num_det_orders=21,
            det_order_method=_core.utils.DetOrder.DetIndex,
            seed=2384753
        ),
        "tesseract-long-beam-multipass-2pass": MultiPassSinterDecoder(
            num_passes=2,
            strategy=_core.Causal,
            det_beam=20,
            beam_climbing=True,
            no_revisit_dets=True,
            merge_errors=True,
            pqlimit=1000000,
            num_det_orders=21,
            det_order_method=_core.utils.DetOrder.DetIndex,
            seed=2384753
        ),
    }
