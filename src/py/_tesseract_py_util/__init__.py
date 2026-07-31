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

from _tesseract_py_util.demutil import decompose_errors
from _tesseract_py_util.generalize_dem import \
    generalize as regeneralize_spatial_dem
