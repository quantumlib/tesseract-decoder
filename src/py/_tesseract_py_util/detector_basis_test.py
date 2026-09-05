# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import pytest
import stim

from _tesseract_py_util.detector_basis import (
    annotate_detector_bases,
    automatic_detector_basis_classifier,
    chromobius_detector_basis_classifier,
    classify_detector_bases,
    last_coordinate_component_classifier,
    stim_surface_code_detector_basis_classifier,
)


@pytest.mark.parametrize(
    "tag,expected",
    [
        (
            '{"measure_basis":"X","md":{"measure_basis":"Z"},'
            '"basis":"Z"}',
            "X",
        ),
        ('{"md":{"measure_basis":"Z"},"basis":"X"}', "Z"),
        ('{"basis":"X","md":{"basis":"Z"}}', "X"),
        ('{"md":{"basis":"Z"}}', "Z"),
    ],
)
def test_automatic_classifier_metadata_precedence(tag, expected):
    assert automatic_detector_basis_classifier(7, [0, 0, 0, 5], tag) == expected


@pytest.mark.parametrize(
    "tag",
    [
        '{"measure_basis":0,"basis":"X"}',
        '{"md":{"measure_basis":"Y"},"basis":"X"}',
        '{"basis":"x","md":{"basis":"Z"}}',
        '{"md":{"basis":null}}',
    ],
)
def test_automatic_classifier_rejects_invalid_recognized_metadata(tag):
    with pytest.raises(ValueError, match=r"Detector D2 has invalid"):
        automatic_detector_basis_classifier(2, [0, 0, 0, 0], tag)


@pytest.mark.parametrize(
    "coordinate,expected",
    [(0, "X"), (1, "X"), (2, "X"), (3, "Z"), (4, "Z"), (5, "Z")],
)
def test_chromobius_classifier_exact_coordinates(coordinate, expected):
    assert (
        chromobius_detector_basis_classifier(0, [8, 9, 10, coordinate], "")
        == expected
    )


@pytest.mark.parametrize("coordinate", [-1, 6, 100])
def test_chromobius_classifier_leaves_other_integral_coordinates_unclassified(
    coordinate,
):
    assert (
        chromobius_detector_basis_classifier(0, [8, 9, 10, coordinate], "")
        is None
    )


def test_automatic_classifier_malformed_json_still_tries_coordinates():
    assert automatic_detector_basis_classifier(0, [0, 0, 0, 3], "not-json") == "Z"
    assert automatic_detector_basis_classifier(0, [0, 0, 0, 1], "[]") == "X"


@pytest.mark.parametrize("coordinate", [0.5, 2.5, 4.25, float("nan")])
def test_chromobius_classifier_rejects_nonintegral_coordinates(coordinate):
    with pytest.raises(ValueError, match="nonintegral fourth coordinate"):
        chromobius_detector_basis_classifier(4, [0, 0, 0, coordinate], "")


def test_named_legacy_adapters():
    assert stim_surface_code_detector_basis_classifier(0, [2, 0], "") == "X"
    assert stim_surface_code_detector_basis_classifier(0, [0, 0], "") == "Z"
    assert last_coordinate_component_classifier(0, [3, 7, 11], "") == 11
    with pytest.raises(ValueError, match="at least two coordinates"):
        stim_surface_code_detector_basis_classifier(3, [0], "")
    with pytest.raises(ValueError, match="at least one coordinate"):
        last_coordinate_component_classifier(5, [], "")


def test_classify_detector_bases_calls_classifier_once_per_detector():
    dem = stim.DetectorErrorModel("detector D0\ndetector D1")
    calls = []

    def classifier(index, coordinates, tag):
        calls.append((index, coordinates, tag))
        return "X" if index == 0 else "Z"

    assert classify_detector_bases(dem, detector_basis_classifier=classifier) == [
        "X",
        "Z",
    ]
    assert [call[0] for call in calls] == [0, 1]


def test_classify_detector_bases_requires_complete_x_z_classification():
    dem = stim.DetectorErrorModel("detector D0")
    with pytest.raises(ValueError, match=r"Detector D0 could not be classified"):
        classify_detector_bases(dem)
    with pytest.raises(ValueError, match=r"classifier returned 0"):
        classify_detector_bases(
            dem, detector_basis_classifier=lambda _i, _c, _t: 0
        )


def test_tagged_circuit_detector_metadata_survives_dem_conversion():
    circuit = stim.Circuit(r"""
        R 0 1
        X_ERROR(0.1) 0
        M 0 1
        DETECTOR[{"basis":"X"}] rec[-2]
        DETECTOR[{"basis":"Z"}] rec[-1]
    """)
    dem = circuit.detector_error_model()
    assert classify_detector_bases(dem) == ["X", "Z"]


def test_annotate_detector_bases_preserves_dem_structure_and_metadata():
    dem = stim.DetectorErrorModel(r"""
        error[outer-error](0.1) D0 L0
        detector[{"measure_basis":"X","keep":{"a":1}}](1, 2, 3, 0) D0
        shift_detectors[shift-tag](10, 20, 30, 0) 1
        repeat[repeat-tag] 2 {
            detector[{"md":{"basis":"Z"},"keep":[1,2\C}](0, 0, 0, 3) D0
            error[inner-error](0.2) D0
            shift_detectors(0, 0, 0, 0) 1
        }
        logical_observable[logical-tag] L0
    """)

    annotated = annotate_detector_bases(dem)

    original_non_detector_lines = [
        line for line in str(dem).splitlines() if not line.lstrip().startswith("detector")
    ]
    annotated_non_detector_lines = [
        line
        for line in str(annotated).splitlines()
        if not line.lstrip().startswith("detector")
    ]
    assert annotated_non_detector_lines == original_non_detector_lines
    assert annotated.get_detector_coordinates() == dem.get_detector_coordinates()
    assert "repeat[repeat-tag] 2 {" in str(annotated)
    assert "error[outer-error](0.1) D0 L0" in str(annotated)
    assert "error[inner-error](0.2) D0" in str(annotated)
    assert "shift_detectors[shift-tag](10, 20, 30, 0) 1" in str(annotated)
    assert "logical_observable[logical-tag] L0" in str(annotated)

    top_metadata = json.loads(annotated[1].tag)
    repeat_metadata = json.loads(annotated[3].body_copy()[0].tag)
    assert top_metadata == {
        "measure_basis": "X",
        "keep": {"a": 1},
        "basis": "X",
    }
    assert repeat_metadata == {
        "md": {"basis": "Z"},
        "keep": [1, 2],
        "basis": "Z",
    }
    assert classify_detector_bases(annotated) == ["X", "Z", "Z"]


def test_annotate_detector_bases_emits_canonical_top_level_tag():
    dem = stim.DetectorErrorModel("detector(1, 2, 3, 0) D0")
    annotated = annotate_detector_bases(dem)
    assert str(annotated) == 'detector[{"basis":"X"}](1, 2, 3, 0) D0'


@pytest.mark.parametrize(
    "tag,match",
    [
        ("plain-text", "non-JSON tag"),
        ("[]", "not an object"),
        ('{"measure_basis":0}', "invalid top-level measure_basis"),
    ],
)
def test_annotate_detector_bases_rejects_tags_it_cannot_merge(tag, match):
    dem = stim.DetectorErrorModel()
    dem.append(
        stim.DemInstruction(
            "detector",
            [0, 0, 0, 0],
            [stim.target_relative_detector_id(0)],
            tag=tag,
        )
    )
    with pytest.raises(ValueError, match=match):
        annotate_detector_bases(
            dem, detector_basis_classifier=lambda _i, _c, _t: "X"
        )


@pytest.mark.parametrize(
    "tag,path",
    [
        ('{"measure_basis":"Z"}', "top-level measure_basis"),
        ('{"md":{"measure_basis":"Z"}}', "md.measure_basis"),
        ('{"basis":"Z"}', "top-level basis"),
        ('{"md":{"basis":"Z"}}', "md.basis"),
    ],
)
def test_annotate_detector_bases_rejects_conflicting_metadata(tag, path):
    dem = stim.DetectorErrorModel(f"detector[{tag}] D0")
    with pytest.raises(ValueError, match=f"conflicting {path}"):
        annotate_detector_bases(
            dem, detector_basis_classifier=lambda _i, _c, _t: "X"
        )


def test_annotate_detector_bases_rejects_basis_changes_across_repeat():
    dem = stim.DetectorErrorModel("""
        repeat 2 {
            detector D0
            shift_detectors 1
        }
    """)
    with pytest.raises(ValueError, match="differently across repetitions"):
        annotate_detector_bases(
            dem,
            detector_basis_classifier=lambda index, _coords, _tag: (
                "X" if index == 0 else "Z"
            ),
        )


def test_annotate_detector_bases_rejects_undeclared_detectors():
    dem = stim.DetectorErrorModel("error(0.1) D0")
    with pytest.raises(ValueError, match="without detector instructions: D0"):
        annotate_detector_bases(
            dem, detector_basis_classifier=lambda _i, _c, _t: "X"
        )
