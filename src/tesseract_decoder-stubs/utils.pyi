"""
A representation of infinity for floating point numbers.
"""
from __future__ import annotations
import tesseract_decoder.common
import typing
__all__: list[str] = ['DetBFS', 'DetCoordinate', 'DetIndex', 'DetOrder', 'EPSILON', 'INF', 'build_det_orders', 'build_detector_graph', 'get_detector_coords', 'get_errors_from_dem']
class DetOrder:
    """
    Detector ordering methods
    
    Members:
    
      DetBFS
    
      DetIndex
    
      DetCoordinate
    """
    DetBFS: typing.ClassVar[DetOrder]  # value = <DetOrder.DetBFS: 0>
    DetCoordinate: typing.ClassVar[DetOrder]  # value = <DetOrder.DetCoordinate: 2>
    DetIndex: typing.ClassVar[DetOrder]  # value = <DetOrder.DetIndex: 1>
    __members__: typing.ClassVar[dict[str, DetOrder]]  # value = {'DetBFS': <DetOrder.DetBFS: 0>, 'DetIndex': <DetOrder.DetIndex: 1>, 'DetCoordinate': <DetOrder.DetCoordinate: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
def build_det_orders(dem: typing.Any, num_det_orders: int, method: DetOrder = DetOrder.DetBFS, seed: int = 0) -> list[list[int]]:
    """
            Generates various detector orderings for decoding.
    
            Parameters
            ----------
            dem : stim.DetectorErrorModel
                The detector error model to generate orders for.
            num_det_orders : int
                The number of detector orderings to generate.
            method : tesseract_decoder.utils.DetOrder, default=tesseract_decoder.utils.DetOrder.DetBFS
                Strategy for ordering detectors. ``DetBFS`` performs a breadth-first
                traversal, ``DetCoordinate`` uses randomized geometric orientations,
                and ``DetIndex`` chooses either increasing or decreasing detector
                index order at random.
            seed : int, default=0
                A seed for the random number generator.
    
            Returns
            -------
            list[list[int]]
                A list of detector orderings. Each inner list maps a detector index
                to its position in the ordering.
    """
def build_detector_graph(dem: typing.Any) -> list[list[int]]:
    """
            Builds a graph representing the connections between detectors.
    
            This graph is used by the decoder to find error paths.
    
            Parameters
            ----------
            dem : stim.DetectorErrorModel
                The detector error model used to build the graph.
    
            Returns
            -------
            list[list[int]]
                An adjacency list representation of the detector graph.
                Each inner list contains the indices of detectors connected
                to the detector at the corresponding index.
                Here we say that two detectors are connected if there exists at
                least one error in the DEM which flips both detectors.
    """
def get_detector_coords(dem: typing.Any) -> list[list[float]]:
    """
            Returns the coordinates for each detector in a DetectorErrorModel.
    
            Parameters
            ----------
            dem : stim.DetectorErrorModel
                The detector error model to extract coordinates from.
    
            Returns
            -------
            list[list[float]]
                A list where each inner list contains the 3D coordinates
                [x, y, z] of a detector.
    """
def get_errors_from_dem(dem: typing.Any) -> list[tesseract_decoder.common.Error]:
    """
            Extracts a list of errors from a DetectorErrorModel.
    
            Parameters
            ----------
            dem : stim.DetectorErrorModel
                The detector error model to extract errors from.
    
            Returns
            -------
            list[common.Error]
                A list of `common.Error` objects representing all the
                errors defined in the DEM.
    """
DetBFS: DetOrder  # value = <DetOrder.DetBFS: 0>
DetCoordinate: DetOrder  # value = <DetOrder.DetCoordinate: 2>
DetIndex: DetOrder  # value = <DetOrder.DetIndex: 1>
EPSILON: float = 1e-07
INF: float  # value = inf
