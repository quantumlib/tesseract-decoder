"""
classes commonly used by the decoder
"""
from __future__ import annotations
import typing
__all__: list[str] = ['Error', 'Symptom', 'dem_from_counts', 'merge_indistinguishable_errors', 'remove_zero_probability_errors']
class Error:
    """
    
            Represents an error, including its cost, and symptom.
    
            An error is a physical event (or set of indistinguishable physical events)
            defined by the detectors and observables that it flips in the circuit.
        
    """
    @typing.overload
    def __init__(self) -> None:
        """
                Default constructor for the `Error` class.
        """
    @typing.overload
    def __init__(self, likelihood_cost: float, detectors: list[int], observables: list[int]) -> None:
        """
                    Constructor for the `Error` class.
        
                    Parameters
                    ----------
                    likelihood_cost : float
                        The cost of this error. 
                        This is often `log((1 - probability) / probability)`.
                    detectors : list[int]
                        A list of indices of the detectors flipped by this error.
                    observables : list[int]
                        A list of indices of the observables flipped by this error.
        """
    @typing.overload
    def __init__(self, error: typing.Any) -> None:
        """
                    Constructor that creates an `Error` from a `stim.DemInstruction`.
        
                    Parameters
                    ----------
                    error : stim.DemInstruction
                        The instruction to convert into an `Error` object.
        """
    def __str__(self) -> str:
        ...
    def get_probability(self) -> float:
        """
                    Gets the probability associated with the likelihood cost.
        
                    Returns
                    -------
                    float
                        The probability of the error, calculated from the likelihood cost.
        """
    def set_with_probability(self, probability: float) -> None:
        """
                    Sets the likelihood cost based on a given probability.
        
                    Parameters
                    ----------
                    probability : float
                        The probability to use for setting the likelihood cost.
                        Must be between 0 and 1 (exclusive).
        
                    Raises
                    ------
                    ValueError
                        If the provided probability is not between 0 and 1.
        """
    @property
    def likelihood_cost(self) -> float:
        """
        The cost of this error (often log((1 - probability) / probability)).
        """
    @likelihood_cost.setter
    def likelihood_cost(self, arg0: float) -> None:
        ...
    @property
    def symptom(self) -> Symptom:
        """
        The symptom associated with this error.
        """
    @symptom.setter
    def symptom(self, arg0: Symptom) -> None:
        ...
class Symptom:
    """
    
            Represents a symptom of an error, which is a list of detectors and a list of observables
    
            A symptom is defined by a list of detectors that are flipped and a list of
            observables that are flipped.
        
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: Symptom) -> bool:
        ...
    def __init__(self, detectors: list[int] = [], observables: list[int] = []) -> None:
        """
                    The constructor for the `Symptom` class.
        
                    Parameters
                    ----------
                    detectors : list[int], default=[]
                        The indices of the detectors in this symptom.
                    observables : list[int], default=[]
                        The indices of the flipped observables.
        """
    def __ne__(self, arg0: Symptom) -> bool:
        ...
    def __str__(self) -> str:
        ...
    def as_dem_instruction_targets(self) -> list[typing.Any]:
        """
                Converts the symptom into a list of `stim.DemTarget` objects.
                
                Returns
                -------
                list[stim.DemTarget]
                    A list of `stim.DemTarget` objects representing the detectors and observables.
        """
    @property
    def detectors(self) -> list[int]:
        """
        A list of the detector indices that are flipped in this symptom.
        """
    @detectors.setter
    def detectors(self, arg0: list[int]) -> None:
        ...
    @property
    def observables(self) -> list[int]:
        """
        A list of observable indices that are flipped in this symptom.
        """
    @observables.setter
    def observables(self, arg0: list[int]) -> None:
        ...
def dem_from_counts(orig_dem: typing.Any, error_counts: list[int], num_shots: int) -> typing.Any:
    """
            Re-weights errors in a `stim.DetectorErrorModel` based on observed counts.
    
            This function re-calculates the probability of each error based on a list of
            observed counts and the total number of shots.
    
            Parameters
            ----------
            orig_dem : stim.DetectorErrorModel
                The original detector error model.
            error_counts : list[int]
                A list of counts for each error in the DEM.
            num_shots : int
                The total number of shots in the experiment.
    
            Returns
            -------
            stim.DetectorErrorModel
                A new `DetectorErrorModel` with updated error probabilities.
    """
def merge_indistinguishable_errors(dem: typing.Any) -> typing.Any:
    """
            Merges identical errors in a `stim.DetectorErrorModel`.
            
            Errors are identical if they flip the same set of detectors and observables (the same symptom).
            For example, two identical errors with probabilities p1 and p2
            would be merged into a single error with the same symptom,
            but with probability `p1 * (1 - p2) + p2 * (1 - p1)`
    
            Parameters
            ----------
            dem : stim.DetectorErrorModel
                The detector error model to process.
    
            Returns
            -------
            stim.DetectorErrorModel
                A new `DetectorErrorModel` with identical errors merged.
    """
def remove_zero_probability_errors(dem: typing.Any) -> typing.Any:
    """
            Removes errors with a probability of 0 from a `stim.DetectorErrorModel`.
    
            Parameters
            ----------
            dem : stim.DetectorErrorModel
                The detector error model to process.
    
            Returns
            -------
            stim.DetectorErrorModel
                A new `DetectorErrorModel` with zero-probability errors removed.
    """
