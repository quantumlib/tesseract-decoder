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

"""Shared detector-basis classification and DEM annotation helpers."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterator, Sequence
from typing import Literal

import stim


DetectorBasis = Literal["X", "Z"]
DetectorBasisClassifier = Callable[
    [int, Sequence[float], str], DetectorBasis | None
]

_BASIS_VALUES = ("X", "Z")
_DETECTOR_LINE = re.compile(r"^(?P<indent>\s*)detector(?:\[|\(|\s)")


def _metadata_basis(
    *, detector_index: int, metadata: dict[str, object], key: str, path: str
) -> DetectorBasis | None:
    if key not in metadata:
        return None
    value = metadata[key]
    if value not in _BASIS_VALUES:
        raise ValueError(
            f"Detector D{detector_index} has invalid {path}: expected 'X' or "
            f"'Z', got {value!r}."
        )
    return value


def chromobius_detector_basis_classifier(
    detector_index: int, coordinates: Sequence[float], tag: str
) -> DetectorBasis | None:
    """Classifies the Chromobius fourth-coordinate X/Z convention.

    Fourth-coordinate values exactly equal to 0, 1, or 2 are X; values exactly
    equal to 3, 4, or 5 are Z. A missing fourth coordinate is unclassified.
    Nonintegral fourth coordinates are invalid instead of being truncated.
    """
    del tag
    if len(coordinates) < 4:
        return None
    value = coordinates[3]
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(
            f"Detector D{detector_index} has nonintegral fourth coordinate "
            f"{value!r}."
        )
    if not float(value).is_integer():
        raise ValueError(
            f"Detector D{detector_index} has nonintegral fourth coordinate "
            f"{value!r}."
        )
    if value in (0, 1, 2):
        return "X"
    if value in (3, 4, 5):
        return "Z"
    return None


def automatic_detector_basis_classifier(
    detector_index: int, coordinates: Sequence[float], tag: str
) -> DetectorBasis | None:
    """Classifies detector metadata, then the Chromobius coordinate convention.

    Recognized metadata is checked in this strict precedence order:

    1. top-level ``measure_basis``
    2. ``md.measure_basis``
    3. top-level ``basis``
    4. ``md.basis``

    A recognized field with an invalid value is an error and does not fall
    through to lower-precedence metadata or coordinates. Malformed JSON and
    JSON values that are not objects contain no recognized metadata, matching
    the legacy multipass behavior, so coordinate classification is still tried.
    """
    tag_data: object = None
    if tag:
        try:
            tag_data = json.loads(tag)
        except json.JSONDecodeError:
            pass
    if isinstance(tag_data, dict):
        md = tag_data.get("md")
        md_dict = md if isinstance(md, dict) else {}
        fields = (
            (tag_data, "measure_basis", "top-level measure_basis"),
            (md_dict, "measure_basis", "md.measure_basis"),
            (tag_data, "basis", "top-level basis"),
            (md_dict, "basis", "md.basis"),
        )
        for metadata, key, path in fields:
            basis = _metadata_basis(
                detector_index=detector_index,
                metadata=metadata,
                key=key,
                path=path,
            )
            if basis is not None:
                return basis
    return chromobius_detector_basis_classifier(detector_index, coordinates, tag)


def stim_surface_code_detector_basis_classifier(
    detector_index: int, coordinates: Sequence[float], tag: str
) -> DetectorBasis:
    """Classifies Stim generated surface-code detectors from coordinate parity."""
    del tag
    if len(coordinates) < 2:
        raise ValueError(
            f"Detector D{detector_index} needs at least two coordinates for the "
            "Stim surface-code convention."
        )
    x, y = coordinates[:2]
    basis_index = 1 - ((x // 2 + y // 2) % 2)
    return "X" if basis_index == 0 else "Z"


def last_coordinate_component_classifier(
    detector_index: int, coordinates: Sequence[float], tag: str
) -> float:
    """Returns the last coordinate as a generic component label.

    This compatibility adapter intentionally does not claim that the component
    label is an X/Z basis.
    """
    del tag
    if not coordinates:
        raise ValueError(
            f"Detector D{detector_index} needs at least one coordinate for the "
            "last-coordinate component convention."
        )
    return coordinates[-1]


def _flattened_detector_tags(
    dem: stim.DetectorErrorModel,
) -> tuple[stim.DetectorErrorModel, dict[int, str]]:
    flattened = dem.flattened()
    detector_tags: dict[int, str] = {}
    for instruction in flattened:
        if instruction.type != "detector":
            continue
        targets = instruction.targets_copy()
        if len(targets) != 1 or not targets[0].is_relative_detector_id():
            raise ValueError(f"Malformed detector instruction: {instruction}")
        detector_tags[targets[0].val] = instruction.tag
    return flattened, detector_tags


def classify_detector_bases(
    dem: stim.DetectorErrorModel,
    *,
    detector_basis_classifier: DetectorBasisClassifier = (
        automatic_detector_basis_classifier
    ),
) -> list[DetectorBasis]:
    """Classifies every detector in ``dem`` exactly once as X or Z."""
    flattened, detector_tags = _flattened_detector_tags(dem)
    coordinates = flattened.get_detector_coordinates()
    result: list[DetectorBasis] = []
    for detector_index in range(flattened.num_detectors):
        basis = detector_basis_classifier(
            detector_index,
            coordinates.get(detector_index, []),
            detector_tags.get(detector_index, ""),
        )
        if basis not in _BASIS_VALUES:
            raise ValueError(
                f"Detector D{detector_index} could not be classified as X or Z; "
                f"classifier returned {basis!r}."
            )
        result.append(basis)
    return result


def _expanded_detector_paths(
    dem: stim.DetectorErrorModel, prefix: tuple[int, ...] = ()
) -> Iterator[tuple[int, ...]]:
    for instruction_index, instruction in enumerate(dem):
        path = prefix + (instruction_index,)
        if instruction.type == "repeat":
            body = instruction.body_copy()
            for _ in range(instruction.repeat_count):
                yield from _expanded_detector_paths(body, path)
        elif instruction.type == "detector":
            yield path


def _structural_detector_instructions(
    dem: stim.DetectorErrorModel, prefix: tuple[int, ...] = ()
) -> Iterator[tuple[tuple[int, ...], stim.DemInstruction]]:
    for instruction_index, instruction in enumerate(dem):
        path = prefix + (instruction_index,)
        if instruction.type == "repeat":
            yield from _structural_detector_instructions(
                instruction.body_copy(), path
            )
        elif instruction.type == "detector":
            yield path, instruction


def _tag_with_canonical_basis(
    *, detector_index: int, tag: str, basis: DetectorBasis
) -> str:
    if tag:
        try:
            metadata = json.loads(tag)
        except json.JSONDecodeError as ex:
            raise ValueError(
                f"Detector D{detector_index} has a non-JSON tag that cannot be "
                "annotated without clobbering it."
            ) from ex
        if not isinstance(metadata, dict):
            raise ValueError(
                f"Detector D{detector_index} has a JSON tag that is not an object "
                "and cannot be annotated without clobbering it."
            )
    else:
        metadata = {}

    md = metadata.get("md")
    md_dict = md if isinstance(md, dict) else {}
    fields = (
        (metadata, "measure_basis", "top-level measure_basis"),
        (md_dict, "measure_basis", "md.measure_basis"),
        (metadata, "basis", "top-level basis"),
        (md_dict, "basis", "md.basis"),
    )
    for field_metadata, key, path in fields:
        existing = _metadata_basis(
            detector_index=detector_index,
            metadata=field_metadata,
            key=key,
            path=path,
        )
        if existing is not None and existing != basis:
            raise ValueError(
                f"Detector D{detector_index} has conflicting {path} "
                f"{existing!r}; classifier assigned {basis!r}."
            )
    metadata["basis"] = basis
    return json.dumps(metadata, separators=(",", ":"), ensure_ascii=True)


def annotate_detector_bases(
    dem: stim.DetectorErrorModel,
    *,
    detector_basis_classifier: DetectorBasisClassifier = (
        automatic_detector_basis_classifier
    ),
) -> stim.DetectorErrorModel:
    """Returns ``dem`` with canonical top-level X/Z detector basis tags.

    The model's instruction order, repeat blocks, shifts, coordinates, error
    instructions, and non-detector tags are retained. Existing JSON-object
    detector tags are augmented without dropping unrelated metadata. Conflicts,
    non-JSON tags, and JSON tags that are not objects are rejected.
    """
    bases = classify_detector_bases(
        dem, detector_basis_classifier=detector_basis_classifier
    )
    flattened = dem.flattened()
    flattened_detectors = [
        instruction
        for instruction in flattened
        if instruction.type == "detector"
    ]
    expanded_paths = list(_expanded_detector_paths(dem))
    if len(expanded_paths) != len(flattened_detectors):
        raise ValueError("Could not align structured and flattened detector instructions.")

    bases_by_path: dict[tuple[int, ...], set[DetectorBasis]] = {}
    detector_ids_by_path: dict[tuple[int, ...], set[int]] = {}
    declared_detectors: set[int] = set()
    for path, instruction in zip(expanded_paths, flattened_detectors):
        targets = instruction.targets_copy()
        detector_index = targets[0].val
        declared_detectors.add(detector_index)
        bases_by_path.setdefault(path, set()).add(bases[detector_index])
        detector_ids_by_path.setdefault(path, set()).add(detector_index)

    missing_declarations = sorted(set(range(dem.num_detectors)) - declared_detectors)
    if missing_declarations:
        raise ValueError(
            "Cannot annotate detectors without detector instructions: "
            + ", ".join(f"D{d}" for d in missing_declarations)
            + "."
        )

    replacements: list[str] = []
    for path, instruction in _structural_detector_instructions(dem):
        path_bases = bases_by_path.get(path, set())
        if len(path_bases) != 1:
            raise ValueError(
                "Cannot preserve repeat structure because one detector instruction "
                "classifies differently across repetitions."
            )
        basis = next(iter(path_bases))
        detector_ids = detector_ids_by_path[path]
        detector_index = min(detector_ids)
        canonical_tag = _tag_with_canonical_basis(
            detector_index=detector_index,
            tag=instruction.tag,
            basis=basis,
        )
        replacements.append(
            str(
                stim.DemInstruction(
                    type=instruction.type,
                    args=instruction.args_copy(),
                    targets=instruction.targets_copy(),
                    tag=canonical_tag,
                )
            )
        )

    replacement_iterator = iter(replacements)
    output_lines: list[str] = []
    replaced_count = 0
    for line in str(dem).splitlines(keepends=True):
        match = _DETECTOR_LINE.match(line)
        if match is None:
            output_lines.append(line)
            continue
        newline = "\n" if line.endswith("\n") else ""
        output_lines.append(
            match.group("indent") + next(replacement_iterator) + newline
        )
        replaced_count += 1
    if replaced_count != len(replacements):
        raise ValueError("Could not rewrite every structured detector instruction.")
    return stim.DetectorErrorModel("".join(output_lines))
