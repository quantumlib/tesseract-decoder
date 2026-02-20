"""
Module containing the SimplexDecoder and related methods
"""
from __future__ import annotations
import numpy
import tesseract_decoder.common
import typing
__all__: list[str] = ['SimplexConfig', 'SimplexDecoder']
class SimplexConfig:
    """
    
            Configuration object for the `SimplexDecoder`.
    
            This class holds all the parameters needed to initialize and configure a
            Simplex decoder instance, including the detector error model and
            decoding options.
        
    """
    def __init__(self, dem: typing.Any, parallelize: bool = False, window_length: int = 0, window_slide_length: int = 0, verbose: bool = False, merge_errors: bool = True) -> None:
        """
                    The constructor for the `SimplexConfig` class.
        
                    Parameters
                    ----------
                    dem : stim.DetectorErrorModel
                        The detector error model to be decoded.
                    parallelize : bool, default=False
                        Whether to use multithreading for decoding.
                    window_length : int, default=0
                        The length of the time window for decoding. A value of 0 disables windowing.
                    window_slide_length : int, default=0
                        The number of time steps to slide the window after each decode. A value of 0
                        disables windowing.
                    verbose : bool, default=False
                        If True, enables verbose logging from the decoder.
                    merge_errors : bool, default=True
                        If True, merges error channels that have identical syndrome patterns.
        """
    def __str__(self) -> str:
        ...
    def compile_decoder(self) -> SimplexDecoder:
        """
                    Compiles the configuration into a new SimplexDecoder instance.
        
                    Returns
                    -------
                    SimplexDecoder
                        A new SimplexDecoder instance configured with the current
                        settings.
        """
    def windowing_enabled(self) -> bool:
        """
        Returns True if windowing is enabled (i.e., `window_length > 0`).
        """
    @property
    def dem(self) -> typing.Any:
        """
        The `stim.DetectorErrorModel` that defines the error channels and detectors.
        """
    @dem.setter
    def dem(self, arg1: typing.Any) -> None:
        ...
    @property
    def merge_errors(self) -> bool:
        """
        If True, identical error mechanisms will be merged.
        """
    @merge_errors.setter
    def merge_errors(self, arg0: bool) -> None:
        ...
    @property
    def parallelize(self) -> bool:
        """
        If True, enables multithreaded decoding.
        """
    @parallelize.setter
    def parallelize(self, arg0: bool) -> None:
        ...
    @property
    def verbose(self) -> bool:
        """
        If True, the decoder will print verbose output.
        """
    @verbose.setter
    def verbose(self, arg0: bool) -> None:
        ...
    @property
    def window_length(self) -> int:
        """
        The number of time steps in each decoding window.
        """
    @window_length.setter
    def window_length(self, arg0: int) -> None:
        ...
    @property
    def window_slide_length(self) -> int:
        """
        The number of time steps the window slides after each decode.
        """
    @window_slide_length.setter
    def window_slide_length(self, arg0: int) -> None:
        ...
class SimplexDecoder:
    """
    
            A class that implements the Simplex decoding algorithm.
    
            It can decode syndromes from a `stim.DetectorErrorModel` to predict
            which observables have been flipped.
        
    """
    def __init__(self, config: SimplexConfig) -> None:
        """
                The constructor for the `SimplexDecoder` class.
        
                Parameters
                ----------
                config : SimplexConfig
                    The configuration object for the decoder.
        """
    def cost_from_errors(self, predicted_errors: list[int]) -> float:
        """
                    Calculates the total logarithmic probability cost for a given set of
                    predicted errors. The cost is a measure of how likely a set of errors is.
        
                    Parameters
                    ----------
                    predicted_errors : list[int]
                        A list of integers representing error indices from the original flattened DEM.
        
                    Returns
                    -------
                    float
                        A float representing the total logarithmic probability cost.
        """
    def decode(self, syndrome: numpy.ndarray[bool]) -> numpy.ndarray:
        """
                Decodes a single shot.
        
                Parameters
                ----------
                syndrome : np.ndarray
                    A 1D NumPy array of booleans representing the detection events for a single shot.
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
                    A 2D NumPy array of booleans where each row represents a single shot's
                    detector outcomes. The shape should be (num_shots, num_detectors): each shot has
                    a new array with num_detectors size.
        
                Returns
                -------
                np.ndarray
                    A 2D NumPy array of booleans where each row corresponds to a shot and
                    each column corresponds to a logical observable. Each row is the decoder's prediction of which observables were flipped in the shot. The shape is
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
                      corresponding logical observable has been flipped by the decoded error.
        """
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
    def init_ilp(self) -> None:
        """
                Initializes the Integer Linear Programming (ILP) solver.
        
                This method must be called before decoding.
        """
    @property
    def config(self) -> SimplexConfig:
        """
        The configuration used to create this decoder.
        """
    @config.setter
    def config(self, arg0: SimplexConfig) -> None:
        ...
    @property
    def end_time_to_errors(self) -> list[list[int]]:
        """
        A map from a detector's end time to the errors that are correlated with it.
        """
    @end_time_to_errors.setter
    def end_time_to_errors(self, arg0: list[list[int]]) -> None:
        ...
    @property
    def error_masks(self) -> list[list[int]]:
        """
        The list of error masks used for decoding.
        """
    @error_masks.setter
    def error_masks(self, arg0: list[list[int]]) -> None:
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
    def start_time_to_errors(self) -> list[list[int]]:
        """
        A map from a detector's start time to the errors that are correlated with it.
        """
    @start_time_to_errors.setter
    def start_time_to_errors(self, arg0: list[list[int]]) -> None:
        ...
