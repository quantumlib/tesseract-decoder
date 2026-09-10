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

"""Utilities exported through the public ``tesseract_decoder.demutil`` facade."""

from _tesseract_py_util import gari as gari
from _tesseract_py_util.decompose_errors import (
    decompose_errors as decompose_errors,
    decompose_errors_using_detector_basis_classifier,
)
from _tesseract_py_util.detector_basis import (
    annotate_detector_bases,
    automatic_detector_basis_classifier,
    chromobius_detector_basis_classifier,
    classify_detector_bases,
    last_coordinate_component_classifier,
    stim_surface_code_detector_basis_classifier,
)
from _tesseract_py_util.generalize_dem import generalize as regeneralize_spatial_dem

__all__ = [
    "annotate_detector_bases",
    "automatic_detector_basis_classifier",
    "chromobius_detector_basis_classifier",
    "classify_detector_bases",
    "decompose_errors",
    "decompose_errors_using_detector_basis_classifier",
    "gari",
    "last_coordinate_component_classifier",
    "regeneralize_spatial_dem",
    "stim_surface_code_detector_basis_classifier",
]
