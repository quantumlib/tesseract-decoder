"""Utilities for detector error model decomposition and re-generalization.

This module is a facade for functionality in `decompose_errors.py` and `generalize_dem.py`.
"""

from typing import List

import stim

from .decompose_errors import (
    reduce_symmetric_difference,
    reduce_set_symmetric_difference,
    undecomposed_error_detectors_and_observables,
    get_component_obs_matching_undecomposed_obs,
    decompose_errors_using_detector_assignment,
    decompose_errors_using_detector_coordinate_assignment,
    detector_coord_to_basis_for_stim_surface_code_convention,
    decompose_errors_using_last_coordinate_index,
    decompose_errors_for_stim_surface_code_coords,
    undecompose_errors,
)
from .generalize_dem import generalize as regeneralize_spatial_dem


def decompose_errors(
    dem: stim.DetectorErrorModel, method: str = "stim-surfacecode-coords"
) -> stim.DetectorErrorModel:
    """Dispatch decomposition strategy by method name."""
    if method == "stim-surfacecode-coords":
        return decompose_errors_for_stim_surface_code_coords(dem)
    if method == "last-coordinate-index":
        return decompose_errors_using_last_coordinate_index(dem)
    raise ValueError(
        "Unknown decomposition method "
        f"{method!r}. Expected 'stim-surfacecode-coords' or 'last-coordinate-index'."
    )
