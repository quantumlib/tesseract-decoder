from __future__ import annotations
from tesseract_decoder.tesseract_sinter_compat import TesseractSinterDecoder
from tesseract_decoder.tesseract_sinter_compat import make_tesseract_sinter_decoders_dict
from . import common
from . import simplex
from . import tesseract
from . import tesseract_sinter_compat
from . import utils
from . import viz
__all__: list[str] = ['TesseractSinterDecoder', 'common', 'make_tesseract_sinter_decoders_dict', 'ostream_redirect', 'simplex', 'tesseract', 'tesseract_sinter_compat', 'utils', 'viz']
class ostream_redirect:
    def __enter__(self) -> None:
        ...
    def __exit__(self, *args) -> None:
        ...
    def __init__(self, stdout: bool = True, stderr: bool = True) -> None:
        ...
