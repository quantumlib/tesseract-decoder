"""

        This module provides Python bindings for the Tesseract quantum error
        correction decoder, designed for compatibility with the Sinter library.
    
"""
from __future__ import annotations
import numpy
import tesseract_decoder.tesseract
import tesseract_decoder.utils
import typing
__all__: list[str] = ['TesseractSinterCompiledDecoder', 'TesseractSinterDecoder', 'make_tesseract_sinter_decoders_dict']
class TesseractSinterCompiledDecoder:
    """
    
                A Tesseract decoder preconfigured for a specific Detector Error Model.
            
    """
    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data: numpy.ndarray[numpy.uint8]) -> numpy.ndarray[numpy.uint8]:
        """
                        Predicts observable flips from bit-packed detection events.
        
                        This function decodes a batch of `num_shots` syndrome measurements,
                        where each shot's detection events are provided in a bit-packed format.
        
                        :param bit_packed_detection_event_data: A 2D numpy array of shape
                            `(num_shots, ceil(num_detectors / 8))`. Each byte contains
                            8 bits of detection event data. A `1` in bit `k` of byte `j`
                            indicates that detector `8j + k` fired.
                        :return: A 2D numpy array of shape `(num_shots, ceil(num_observables / 8))`
                            containing the predicted observable flips in a bit-packed format.
        """
    @property
    def decoder(self) -> tesseract_decoder.tesseract.TesseractDecoder:
        """
        The internal TesseractDecoder instance.
        """
    @property
    def num_detectors(self) -> int:
        """
        The number of detectors in the decoder's underlying DEM.
        """
    @num_detectors.setter
    def num_detectors(self, arg0: int) -> None:
        ...
    @property
    def num_observables(self) -> int:
        """
        The number of logical observables in the decoder's underlying DEM.
        """
    @num_observables.setter
    def num_observables(self, arg0: int) -> None:
        ...
class TesseractSinterDecoder:
    """
    
                A factory for creating Tesseract decoders compatible with `sinter`.
            
    """
    __hash__: typing.ClassVar[None] = None
    beam_climbing: bool
    create_visualization: bool
    det_beam: int
    det_order_method: tesseract_decoder.utils.DetOrder
    det_penalty: float
    merge_errors: bool
    no_revisit_dets: bool
    num_det_orders: int
    pqlimit: int
    seed: int
    verbose: bool
    def __eq__(self, arg0: TesseractSinterDecoder) -> bool:
        """
        Checks if two TesseractSinterDecoder instances are equal.
        """
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __init__(self) -> None:
        """
                    Initializes a new TesseractSinterDecoder instance with a default TesseractConfig.
        """
    @typing.overload
    def __init__(self, det_beam: int = 5, beam_climbing: bool = False, no_revisit_dets: bool = True, verbose: bool = False, merge_errors: bool = True, pqlimit: int = 200000, det_penalty: float = 0.0, create_visualization: bool = False, num_det_orders: int = 0, det_order_method: tesseract_decoder.utils.DetOrder = tesseract_decoder.utils.DetOrder.DetBFS, seed: int = 2384753) -> None:
        """
                    Initializes a new TesseractSinterDecoder instance with custom TesseractConfig parameters.
        """
    def __ne__(self, arg0: TesseractSinterDecoder) -> bool:
        """
        Checks if two TesseractSinterDecoder instances are not equal.
        """
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def compile_decoder_for_dem(self, *, dem: typing.Any) -> TesseractSinterCompiledDecoder:
        """
                        Creates a Tesseract decoder preconfigured for the given detector error model.
        
                        :param dem: The `stim.DetectorErrorModel` to configure the decoder for.
                        :return: A `TesseractSinterCompiledDecoder` instance that can decode
                            bit-packed shots for the given DEM.
        """
    def decode_via_files(self, *, num_shots: int, num_dets: int, num_obs: int, dem_path: typing.Any, dets_b8_in_path: typing.Any, obs_predictions_b8_out_path: typing.Any, tmp_dir: typing.Any) -> None:
        """
                        Decodes data from files and writes the result to a file.
        
                        :param num_shots: The number of shots to decode.
                        :param num_dets: The number of detectors in the error model.
                        :param num_obs: The number of logical observables in the error model.
                        :param dem_path: The path to a file containing the `stim.DetectorErrorModel` string.
                        :param dets_b8_in_path: The path to a file containing bit-packed detection events.
                        :param obs_predictions_b8_out_path: The path to the output file where
                            bit-packed observable predictions will be written.
                        :param tmp_dir: A temporary directory path. (Currently unused, but required by API)
        """
def make_tesseract_sinter_decoders_dict() -> typing.Any:
    """
            Returns a dictionary mapping decoder names to sinter.Decoder-style objects.
            This allows Sinter to easily discover and use Tesseract as a custom decoder.
    """
