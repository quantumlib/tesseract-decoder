"""
A sentinel value indicating an infinite beam size for the decoder.
"""
from __future__ import annotations
import numpy
import tesseract_decoder.common
import typing
__all__: list[str] = ['INF_DET_BEAM', 'TesseractConfig', 'TesseractDecoder']
class TesseractConfig:
    """
    
            Configuration object for the `TesseractDecoder`.
    
            This class holds all the parameters needed to initialize and configure a
            Tesseract decoder instance.
        
    """
    @typing.overload
    def __init__(self) -> None:
        """
                Default constructor for TesseractConfig.
                Creates a new instance with default parameter values.
        """
    @typing.overload
    def __init__(self, det_beam: int = 5, beam_climbing: bool = False, no_revisit_dets: bool = True, verbose: bool = False, merge_errors: bool = True, pqlimit: int = 200000, det_orders: list[list[int]] = [], det_penalty: float = 0.0, create_visualization: bool = False) -> None:
        """
                     The constructor for the `TesseractConfig` class without a `dem` argument.
                     This creates an empty `DetectorErrorModel` by default.
        
                     Parameters
                     ----------
                     det_beam : int, default=INF_DET_BEAM
                         Beam cutoff that specifies the maximum number of detection events a search state can have.
                     beam_climbing : bool, default=False
                         If True, enables a beam climbing heuristic.
                     no_revisit_dets : bool, default=False
                         If True, prevents the decoder from revisiting a syndrome pattern more than once.
                     
                     verbose : bool, default=False
                         If True, enables verbose logging from the decoder.
                      merge_errors : bool, default=True
                         If True, merges error channels that have identical syndrome patterns.
                      pqlimit : int, default=max_size_t
                         The maximum size of the priority queue.
                      det_orders : list[list[int]], default=empty
                         A list of detector orderings to use for decoding. If empty, the decoder
                         will generate its own orderings.
                      det_penalty : float, default=0.0
                         A penalty value added to the cost of each detector visited.
                      create_visualization: bool, defualt=False
                         Whether to record the information needed to create a visualization or not.
        """
    @typing.overload
    def __init__(self, dem: typing.Any, det_beam: int = 5, beam_climbing: bool = False, no_revisit_dets: bool = True, verbose: bool = False, merge_errors: bool = True, pqlimit: int = 200000, det_orders: list[list[int]] = [], det_penalty: float = 0.0, create_visualization: bool = False) -> None:
        """
                    The constructor for the `TesseractConfig` class.
        
                    Parameters
                    ----------
                    dem : stim.DetectorErrorModel
                        The detector error model to be decoded.
                    det_beam : int, default=INF_DET_BEAM
                        Beam cutoff that specifies the maximum number of detection events a search state can have.
                    beam_climbing : bool, default=False
                        If True, enables a beam climbing heuristic.
                    no_revisit_dets : bool, default=False
                        If True, prevents the decoder from revisiting a syndrome pattern more than once.
                    
                    verbose : bool, default=False
                        If True, enables verbose logging from the decoder.
                     merge_errors : bool, default=True
                        If True, merges error channels that have identical syndrome patterns.
                    pqlimit : int, default=max_size_t
                        The maximum size of the priority queue.
                    det_orders : list[list[int]], default=empty
                        A list of detector orderings to use for decoding. If empty, the decoder
                        will generate its own orderings.
                    det_penalty : float, default=0.0
                        A penalty value added to the cost of each detector visited.
                    create_visualization: bool, defualt=False
                        Whether to record the information needed to create a visualization or not.
        """
    def __str__(self) -> str:
        ...
    def compile_decoder(self) -> TesseractDecoder:
        """
                  Compiles the configuration into a new `TesseractDecoder` instance.
        
                  Returns
                  -------
                  TesseractDecoder
                      A new `TesseractDecoder` instance configured with the current
                      settings.
        """
    def compile_decoder_for_dem(self, dem: typing.Any) -> TesseractDecoder:
        """
                    Compiles the configuration into a new `TesseractDecoder` instance
                    for a given `dem` object.
        
                    Parameters
                    ----------
                    dem : stim.DetectorErrorModel
                        The detector error model to use for the decoder.
        
                    Returns
                    -------
                    TesseractDecoder
                        A new `TesseractDecoder` instance configured with the
                        provided `dem` and the other settings from this
                        `TesseractConfig` object.
        """
    @property
    def beam_climbing(self) -> bool:
        """
        Whether to use a beam climbing heuristic.
        """
    @beam_climbing.setter
    def beam_climbing(self, arg0: bool) -> None:
        ...
    @property
    def create_visualization(self) -> bool:
        """
        If True, records necessary information to create visualization.
        """
    @create_visualization.setter
    def create_visualization(self, arg0: bool) -> None:
        ...
    @property
    def dem(self) -> typing.Any:
        """
        The `stim.DetectorErrorModel` that defines the error channels and detectors.
        """
    @dem.setter
    def dem(self, arg1: typing.Any) -> None:
        ...
    @property
    def det_beam(self) -> int:
        """
        Beam cutoff argument for the beam search.
        """
    @det_beam.setter
    def det_beam(self, arg0: int) -> None:
        ...
    @property
    def det_orders(self) -> list[list[int]]:
        """
        A list of pre-specified detector orderings.
        """
    @det_orders.setter
    def det_orders(self, arg0: list[list[int]]) -> None:
        ...
    @property
    def det_penalty(self) -> float:
        """
        The penalty cost added for each detector.
        """
    @det_penalty.setter
    def det_penalty(self, arg0: float) -> None:
        ...
    @property
    def merge_errors(self) -> bool:
        """
        If True, merges error channels that have identical syndrome patterns.
        """
    @merge_errors.setter
    def merge_errors(self, arg0: bool) -> None:
        ...
    @property
    def no_revisit_dets(self) -> bool:
        """
        Whether to prevent revisiting same syndrome patterns during decoding.
        """
    @no_revisit_dets.setter
    def no_revisit_dets(self, arg0: bool) -> None:
        ...
    @property
    def pqlimit(self) -> int:
        """
        The maximum size of the priority queue.
        """
    @pqlimit.setter
    def pqlimit(self, arg0: int) -> None:
        ...
    @property
    def verbose(self) -> bool:
        """
        If True, the decoder will print verbose output.
        """
    @verbose.setter
    def verbose(self, arg0: bool) -> None:
        ...
class TesseractDecoder:
    """
    
            A class that implements the Tesseract decoding algorithm.
    
            It can decode syndromes from a `stim.DetectorErrorModel` to predict
            which observables have been flipped.
        
    """
    def __init__(self, config: TesseractConfig) -> None:
        """
                The constructor for the `TesseractDecoder` class.
        
                Parameters
                ----------
                config : TesseractConfig
                    The configuration object for the decoder.
        """
    def cost_from_errors(self, predicted_errors: list[int]) -> float:
        """
                    Calculates the sum of the likelihood costs of the predicted errors.
                    The likelihood cost of an error with probability p is log((1 - p) / p).
        
                    Parameters
                    ----------
                    predicted_errors : list[int]
                        A list of integers representing error indices from the original flattened DEM.
        
                    Returns
                    -------
                    float
                        A float representing the sum of the likelihood costs of the
                        predicted errors.
        """
    def decode(self, syndrome: numpy.ndarray[bool]) -> numpy.ndarray:
        """
                Decodes a single shot.
        
                Parameters
                ----------
                syndrome : np.ndarray
                    A 1D NumPy array of booleans representing the detector outcomes for a single shot.
                    The length of the array should match the number of detectors in the DEM.
        
                Returns
                -------
                np.ndarray
                    A 1D NumPy array of booleans indicating which observables are flipped.
                    The length of the array matches the number of observables.
        """
    def decode_batch(self, syndromes: numpy.ndarray[bool]) -> numpy.ndarray[bool]:
        """
                Decodes a batch of shots.
        
                Parameters
                ----------
                syndromes : np.ndarray
                    A 2D NumPy array of booleans where each row corresponds to a shot and
                    each column corresponds to a logical observable. Each row is the decoder's prediction of which observables were flipped in the shot. The shape is
                    a new array with num_detectors size.
        
                Returns
                -------
                np.ndarray
                    A 2D NumPy array of booleans where each row corresponds to a shot and
                    that short specifies which logical observable are flipped. The shape is
                    (num_shots, num_observables).
        """
    def decode_from_detection_events(self, detections: list[int]) -> numpy.ndarray:
        """
                  Decodes a single shot from a list of detection events.
        
                  Parameters
                  ----------
                  detections : list[int]
                      A list of indices corresponding to the detectors that were
                      fired. This input represents a single measurement shot.
        
                  Returns
                  -------
                  np.ndarray
                      A 1D NumPy array of booleans. Each boolean value indicates whether the
                      decoder predicts that the corresponding logical observable has been flipped.
        """
    @typing.overload
    def decode_to_errors(self, syndrome: numpy.ndarray[bool]) -> list[int]:
        """
                    Decodes a single shot to a list of error indices.
        
                    Parameters
                    ----------
                    syndrome : np.ndarray
                        A 1D NumPy array of booleans representing the detector outcomes for a single shot.
                        The length of the array should match the number of detectors in the DEM.
        
                    Returns
                    -------
                    list[int]
                        A list of predicted error indices from the original flattened DEM.
        """
    @typing.overload
    def decode_to_errors(self, syndrome: numpy.ndarray[bool], det_order: int, det_beam: int) -> list[int]:
        """
                    Decodes a single shot using a specific detector ordering and beam size.
        
                    Parameters
                    ----------
                    syndrome : np.ndarray
                        A 1D NumPy array of booleans representing the detector outcomes for a single shot.
                        The length of the array should match the number of detectors in the DEM.
                    det_order : int
                        The index of the detector ordering to use.
                    det_beam : int
                        The beam size to use during the decoding.
        
                    Returns
                    -------
                    list[int]
                        A list of predicted error indices from the original flattened DEM.
        """
    def get_observables_from_errors(self, predicted_errors: list[int]) -> list[bool]:
        """
                    Converts a list of predicted error indices into a list of
                    flipped logical observables.
        
                    Parameters
                    ----------
                    predicted_errors : list[int]
                        A list of integers representing error indices from the original flattened DEM.
        
                    Returns
                    -------
                    list[bool]
                        A list of booleans, where each boolean corresponds to a
                        logical observable and is `True` if the observable was flipped.
        """
    @property
    def config(self) -> TesseractConfig:
        """
        The configuration used to create this decoder.
        """
    @config.setter
    def config(self, arg0: TesseractConfig) -> None:
        ...
    @property
    def errors(self) -> list[tesseract_decoder.common.Error]:
        """
        The list of all errors in the detector error model.
        """
    @errors.setter
    def errors(self, arg0: list[tesseract_decoder.common.Error]) -> None:
        ...
    @property
    def low_confidence_flag(self) -> bool:
        """
        A flag indicating if the decoder's prediction has low confidence.
        """
    @low_confidence_flag.setter
    def low_confidence_flag(self, arg0: bool) -> None:
        ...
    @property
    def num_detectors(self) -> int:
        """
        The total number of detectors in the detector error model.
        """
    @num_detectors.setter
    def num_detectors(self, arg0: int) -> None:
        ...
    @property
    def num_observables(self) -> int:
        """
        The total number of logical observables in the detector error model.
        """
    @num_observables.setter
    def num_observables(self, arg0: int) -> None:
        ...
    @property
    def predicted_errors_buffer(self) -> list[int]:
        """
        A buffer containing the predicted errors from the most recent decode operation.
        """
    @predicted_errors_buffer.setter
    def predicted_errors_buffer(self, arg0: list[int]) -> None:
        ...
    @property
    def visualizer(self) -> Visualizer:
        """
        An object that can (if config.create_visualization=True) be used to generate visualization of the algorithm
        """
INF_DET_BEAM: int = 65535
