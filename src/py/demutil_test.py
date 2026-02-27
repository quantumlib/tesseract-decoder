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


def test_reduce_symmetric_difference_exposed():
    assert tesseract_decoder.demutil.reduce_symmetric_difference([1, 2, 2, 3]) == (1, 3)


def test_reduce_set_symmetric_difference_exposed():
    assert tesseract_decoder.demutil.reduce_set_symmetric_difference(
        [{1, 2}, {2, 3}]
    ) == (1, 3)


def test_undecomposed_error_detectors_and_observables_exposed():
    err = stim.DemInstruction("error", [0.1], [stim.target_relative_detector_id(0)])
    dets, obs = tesseract_decoder.demutil.undecomposed_error_detectors_and_observables(
        err
    )
    assert dets == (0,)
    assert obs == ()


def test_get_component_obs_matching_undecomposed_obs_exposed():
    # Simple case: 1 component, 1 option matching target
    obs_options = [{(0,)}]
    target_obs = (0,)
    result = tesseract_decoder.demutil.get_component_obs_matching_undecomposed_obs(
        obs_options, target_obs
    )
    assert result == [(0,)]


def test_decompose_errors_using_detector_assignment_exposed():
    dem = _demo_dem()
    # Assign D0 (0) -> comp 0, D1 (1) -> comp 1
    # Error D0 D1 (0.3) should split if allowed, but here we just check it runs
    # This function is complex, we just check it returns a DEM
    out = tesseract_decoder.demutil.decompose_errors_using_detector_assignment(
        dem, lambda d: d
    )
    assert isinstance(out, stim.DetectorErrorModel)


def test_decompose_errors_using_detector_coordinate_assignment_exposed():
    dem = _demo_dem()
    # D0 at (0,0,0), D1 at (2,0,1)
    # Assign based on Z coord: D0->0, D1->1
    out = tesseract_decoder.demutil.decompose_errors_using_detector_coordinate_assignment(
        dem, lambda c: int(c[2])
    )
    assert isinstance(out, stim.DetectorErrorModel)


def test_detector_coord_to_basis_exposed():
    # (0,0) -> 0 (X), (1,0) -> 1 (Z) ? check impl
    # Impl: 1 - ((x//2 + y//2) % 2)
    # (0,0) -> 1 - (0%2) = 1
    # (2,0) -> 1 - (1%2) = 0
    assert (
        tesseract_decoder.demutil.detector_coord_to_basis_for_stim_surface_code_convention(
            (0, 0)
        )
        == 1
    )


def test_undecompose_errors_exposed():
    dem = _demo_dem()
    # Undecomposing a flat DEM should be idempotent or similar
    out = tesseract_decoder.demutil.undecompose_errors(dem)
    assert isinstance(out, stim.DetectorErrorModel)
    assert out.num_errors > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
