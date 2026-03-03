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



from _tesseract_py_util.decompose_errors import (
    reduce_symmetric_difference as reduce_symmetric_difference,
    reduce_set_symmetric_difference as reduce_set_symmetric_difference,
    undecomposed_error_detectors_and_observables as undecomposed_error_detectors_and_observables,
    get_component_obs_matching_undecomposed_obs as get_component_obs_matching_undecomposed_obs,
    decompose_errors_using_detector_assignment as decompose_errors_using_detector_assignment,
    decompose_errors_using_detector_coordinate_assignment as decompose_errors_using_detector_coordinate_assignment,
    detector_coord_to_basis_for_stim_surface_code_convention as detector_coord_to_basis_for_stim_surface_code_convention,
    decompose_errors_using_last_coordinate_index as decompose_errors_using_last_coordinate_index,
    decompose_errors_for_stim_surface_code_coords as decompose_errors_for_stim_surface_code_coords,
    undecompose_errors as undecompose_errors,
)
from _tesseract_py_util.generalize_dem import generalize as regeneralize_spatial_dem
from _tesseract_py_util.demutil import decompose_errors

