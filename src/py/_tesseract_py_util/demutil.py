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


import stim
from _tesseract_py_util.decompose_errors import \
    decompose_errors_for_stim_surface_code_coords as \
    decompose_errors_for_stim_surface_code_coords
from _tesseract_py_util.decompose_errors import \
    decompose_errors_using_last_coordinate_index as \
    decompose_errors_using_last_coordinate_index


def decompose_errors(
    dem: stim.DetectorErrorModel,
    method: str = "stim-surfacecode-coords",
    disable_extra_checks: bool = False,
) -> stim.DetectorErrorModel:
    """Dispatch decomposition strategy by method name."""
    if method == "stim-surfacecode-coords":
        return decompose_errors_for_stim_surface_code_coords(
            dem, disable_extra_checks=disable_extra_checks
        )
    if method == "last-coordinate-index":
        return decompose_errors_using_last_coordinate_index(
            dem, disable_extra_checks=disable_extra_checks
        )
    raise ValueError(
        "Unknown decomposition method "
        f"{method!r}. Expected 'stim-surfacecode-coords' or 'last-coordinate-index'."
    )
