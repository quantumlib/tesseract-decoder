# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module is a dispatcher for DEMfunctionality such as decomposition and re-generalization,
and related utilities, in `decompose_errors.py` and `generalize_dem.py`.
"""



__all__ = [
    "decompose_errors",
    "regeneralize_spatial_dem",
    "reduce_symmetric_difference",
    "reduce_set_symmetric_difference",
    "undecomposed_error_detectors_and_observables",
    "get_component_obs_matching_undecomposed_obs",
    "decompose_errors_using_detector_assignment",
    "decompose_errors_using_detector_coordinate_assignment",
    "detector_coord_to_basis_for_stim_surface_code_convention",
    "decompose_errors_using_last_coordinate_index",
    "decompose_errors_for_stim_surface_code_coords",
    "undecompose_errors",
]

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
