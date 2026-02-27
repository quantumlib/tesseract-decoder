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

import pytest
import stim

from src import tesseract_decoder


def _demo_dem() -> stim.DetectorErrorModel:
    return stim.DetectorErrorModel(
        """
        detector(0, 0, 0) D0
        detector(2, 0, 1) D1
        error(0.1) D0
        error(0.2) D1
        error(0.3) D0 D1
        """
    )


def test_import_exposes_demutil_submodule():
    assert hasattr(tesseract_decoder, "demutil")
    assert hasattr(tesseract_decoder.demutil, "regeneralize_spatial_dem")
    assert hasattr(tesseract_decoder.demutil, "decompose_errors")


def test_decompose_errors_dispatch_methods():
    dem = _demo_dem()

    expected_surface = tesseract_decoder.demutil.decompose_errors_for_stim_surface_code_coords(
        dem
    )
    actual_surface = tesseract_decoder.demutil.decompose_errors(
        dem, method="stim-surfacecode-coords"
    )
    assert str(actual_surface) == str(expected_surface)

    expected_last = tesseract_decoder.demutil.decompose_errors_using_last_coordinate_index(
        dem
    )
    actual_last = tesseract_decoder.demutil.decompose_errors(
        dem, method="last-coordinate-index"
    )
    assert str(actual_last) == str(expected_last)


def test_decompose_errors_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown decomposition method"):
        tesseract_decoder.demutil.decompose_errors(_demo_dem(), method="bad-method")


def test_regeneralize_spatial_dem_averages_template_probabilities():
    template_1 = stim.DetectorErrorModel(
        """
        detector(0, 0, 0) D0
        detector(2, 0, 0) D1
        error(0.1) D0
        error(0.2) D1
        """
    )
    template_2 = stim.DetectorErrorModel(
        """
        detector(0, 0, 0) D0
        detector(2, 0, 0) D1
        error(0.3) D0
        error(0.4) D1
        """
    )
    scaffold = stim.DetectorErrorModel(
        """
        detector(0, 0, 0) D0
        detector(2, 0, 0) D1
        error(0.9) D0
        error(0.9) D1
        """
    )

    out = tesseract_decoder.demutil.regeneralize_spatial_dem(
        templates=[template_1, template_2], scaffold=scaffold
    )

    probs = [inst.args_copy()[0] for inst in out if inst.type == "error"]
    assert probs == pytest.approx([0.2, 0.3])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
