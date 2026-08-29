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

import itertools
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable
from functools import reduce

import stim

if __package__:
    from .detector_basis import (
        DetectorBasisClassifier,
        automatic_detector_basis_classifier,
        classify_detector_bases,
        last_coordinate_component_classifier,
        stim_surface_code_detector_basis_classifier,
    )
else:
    from detector_basis import (
        DetectorBasisClassifier,
        automatic_detector_basis_classifier,
        classify_detector_bases,
        last_coordinate_component_classifier,
        stim_surface_code_detector_basis_classifier,
    )


def reduce_symmetric_difference(items: Iterable[int]) -> tuple[int]:
    """
    Calculates the symmetric difference of a multiset of items.

    Returns items that appear an odd number of times in the input.
    """
    unpaired_set = reduce(lambda acc, i: acc ^ {i}, items, set())
    return tuple(sorted(unpaired_set))


def reduce_set_symmetric_difference(sets: Iterable[Iterable[int]]) -> tuple[int]:
    return reduce_symmetric_difference(itertools.chain.from_iterable(sets))


def undecomposed_error_detectors_and_observables(
    instruction: stim.DemInstruction,
) -> tuple[tuple[int], tuple[int]]:
    """Outputs the indices of the detectors and observables in a stim error,
    undecomposing the error if necessary."""
    if instruction.type != "error":
        raise ValueError(f"DEM instruction must be an error, not {instruction.type}")
    detectors = reduce_symmetric_difference(
        d.val for d in instruction.targets_copy() if d.is_relative_detector_id()
    )
    observables = reduce_symmetric_difference(
        o.val for o in instruction.targets_copy() if o.is_logical_observable_id()
    )
    return detectors, observables


def _error_has_separator(instruction: stim.DemInstruction) -> bool:
    return any(target.is_separator() for target in instruction.targets_copy())


def _validated_error_groups(
    instruction: stim.DemInstruction,
    detector_component_func: Callable[[int], int],
    *,
    allow_mixed_group: bool,
) -> list[tuple[tuple[int], tuple[int], int | None]]:
    """Parses Stim ``^`` groups and validates their component assignments."""
    if instruction.type != "error":
        raise ValueError(f"DEM instruction must be an error, not {instruction.type}")

    raw_groups: list[list[stim.DemTarget]] = [[]]
    for target in instruction.targets_copy():
        if target.is_separator():
            raw_groups.append([])
        else:
            raw_groups[-1].append(target)

    result = []
    for raw_group in raw_groups:
        detectors = reduce_symmetric_difference(
            target.val
            for target in raw_group
            if target.is_relative_detector_id()
        )
        observables = reduce_symmetric_difference(
            target.val
            for target in raw_group
            if target.is_logical_observable_id()
        )
        if not detectors:
            raise ValueError(
                f"Error instruction `{instruction}` contains a detectorless "
                "decomposition group, which cannot be assigned to a component."
            )
        components = {detector_component_func(d) for d in detectors}
        if len(components) != 1 and not allow_mixed_group:
            raise ValueError(
                f"Error instruction `{instruction}` contains a decomposition "
                "group with detectors from multiple components."
            )
        component = next(iter(components)) if len(components) == 1 else None
        result.append((detectors, observables, component))
    return result


def get_component_obs_matching_undecomposed_obs(
    obs_options_by_component: list[set[tuple[int]]], error_obs: tuple[int]
) -> list[tuple[int]] | None:
    """Given the possible observables that could be a symptom of each component
    of a dem error, find the assignment of observables to components that is
    consistent with the observables associated with the undecomposed error.
    Returns None if there is no assignment that is consistent with the observables
    of the undecomposed error.

    Parameters
    ----------
    obs_options_by_component : list[set[tuple[int]]]
        The possible observables consistent with each component. Here
        `obs_options_by_component[i]` is a set of tuples, where each tuple
        contains the indices of observables that could have been flipped by
        component i. For example, these could be observables flipped by
        an undecomposable error elsewhere in the dem that has the same detectors
        as the component. Note that if there is more than one choice for a given
        component (i.e. if `len(obs_options_by_component[i]) > 1`) then the dem
        must have distance at most 2. If the distance is more than 2, then this
        function makes the trivial assignment of assigning the only possble
        observables to each component.
    error_obs : tuple[int]
        The observables flipped by the undecomposed error.

    Returns
    -------
    list[tuple[int]]
        Assignment of observables to each component.
    """
    error_obs_set = set(reduce_symmetric_difference(error_obs))
    for obs_combinations in itertools.product(*obs_options_by_component):
        obs_from_combination = reduce_set_symmetric_difference(obs_combinations)
        if set(obs_from_combination) == error_obs_set:
            return list(obs_combinations)
    return None


def decompose_errors_using_detector_assignment(
    dem: stim.DetectorErrorModel,
    detector_component_func: Callable[[int], int],
    strip_undecomposable_errors: bool = False,
) -> stim.DetectorErrorModel:
    """Decomposes errors in the detector error model `dem` based on an assignment of
    detectors to components by the function `detector_component_func`.

    An undecomposed error is an error that flips detectors that are all in the same
    component. A decomposed error is an error that flips detectors from more than one
    component, but is decomposed into components where each component corresponds
    to an undecomposed error elsewhere in the dem. The symmetric difference of the
    detectors and observables in the components of a decomposed error will equal
    the detectors and observables of the original error in the dem.
    See https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md#error-instruction
    for more details on the Stim ERROR instruction format, including decomposition.
    Existing Stim ``^`` decomposition groups are preserved after validation. Each
    existing group must contain at least one detector, and all detectors in a group
    must belong to the same component.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model to decompose.
    detector_component_func : Callable[[int], int]
        A function that maps a detector id to its component. i.e. This could map
        a detector index to 0 if it is X-type or to 1 if it is Z-type.
    strip_undecomposable_errors : bool
        If True, errors that cannot be decomposed due to a missing component error
        will be stripped from the output DEM instead of raising a ValueError.

    Returns
    -------
    stim.DetectorErrorModel
        The decomposed detector error model
    """
    dem = dem.flattened()

    single_component_dets_to_obs: dict[tuple[int], set[tuple[int]]] = defaultdict(set)

    for instruction in dem:
        if instruction.type != "error":
            continue

        is_decomposed = _error_has_separator(instruction)
        groups = _validated_error_groups(
            instruction,
            detector_component_func,
            allow_mixed_group=not is_decomposed,
        )
        if is_decomposed:
            for detectors, observables, _component in groups:
                single_component_dets_to_obs[detectors].add(observables)
        else:
            detectors, observables, component = groups[0]
            if component is not None:
                single_component_dets_to_obs[detectors].add(observables)

    output_dem = stim.DetectorErrorModel()

    for instruction in dem:
        if instruction.type != "error":
            output_dem.append(instruction)
            continue

        is_decomposed = _error_has_separator(instruction)
        groups = _validated_error_groups(
            instruction,
            detector_component_func,
            allow_mixed_group=not is_decomposed,
        )
        if is_decomposed:
            output_dem.append(instruction)
            continue

        detectors, observables, _component = groups[0]
        det_components = {d: detector_component_func(d) for d in detectors}
        unique_components = sorted(set(det_components.values()))
        num_components = len(unique_components)

        dets_by_component = []
        obs_options_by_component = []

        is_undecomposable = False
        for c in unique_components:
            component_dets = tuple(
                sorted(d for d in detectors if det_components[d] == c)
            )
            if component_dets not in single_component_dets_to_obs:
                if strip_undecomposable_errors:
                    is_undecomposable = True
                    break
                else:
                    raise ValueError(
                        f"The dem error `{instruction}` needs to be decomposed into components, however "
                        f"the component with detectors {component_dets} is not present as its own error "
                        "in the dem."
                    )
            dets_by_component.append(component_dets)
            obs_options_by_component.append(
                single_component_dets_to_obs[component_dets]
            )

        if is_undecomposable:
            continue

        # Assign observables to each component, such that they are consistent with the
        # observables of the undecomposed error
        consistent_obs_by_component = get_component_obs_matching_undecomposed_obs(
            obs_options_by_component=obs_options_by_component, error_obs=observables
        )

        if consistent_obs_by_component is None:
            if strip_undecomposable_errors:
                continue
            raise ValueError(
                f"The error instruction `{instruction}` could not be decomposed, due to its "
                "observables not being consistent with the observables of any available "
                f"choices of components."
            )

        targets = []
        for i in range(num_components):
            targets.extend(
                stim.target_relative_detector_id(d) for d in dets_by_component[i]
            )
            targets.extend(
                stim.target_logical_observable_id(o)
                for o in consistent_obs_by_component[i]
            )
            if i != num_components - 1:
                targets.append(stim.target_separator())

        decomposed_instruction = stim.DemInstruction(
            type=instruction.type,
            args=instruction.args_copy(),
            targets=targets,
            tag=instruction.tag,
        )
        output_dem.append(decomposed_instruction)

    return output_dem


def decompose_errors_using_detector_coordinate_assignment(
    dem: stim.DetectorErrorModel,
    coord_to_component_func: Callable[[list[float]], int],
    strip_undecomposable_errors: bool = False,
) -> stim.DetectorErrorModel:
    """Decomposes errors in the detector error model `dem` based on an assignment of
    detectors to components using a function of the detector coordinates.

    A detector with coordinates `coords` is assigned to component
    `coord_to_component_func(coords)`. If an error flips detectors that are all
    in component `i` then this error itself is assigned as an error in component `i`.
    This error is said to be undecomposable. If an error flips a set of detectors that
    belong to more than one component, then this function attempts to decompose the
    error into undecomposable errors (i.e. errors with detectors in a single component).
    For a definition of errors and decompositions see:
    https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md#error-instruction.


    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model to decompose
    coord_to_component_func : Callable[[list[float]], int]
        A function that coordinates of a detector to an integer corresponding to
        the index of a component, to be used for the decomposition. The coordinates
        are provided as a list of floats.
    strip_undecomposable_errors : bool
        If True, errors that cannot be decomposed due to a missing component error
        will be stripped from the output DEM instead of raising a ValueError.

    Returns
    -------
    stim.DetectorErrorModel
        The decomposed detector error model. Note that the DEM will also be flattened.
    """
    detector_coords = dem.get_detector_coordinates()
    detector_components = [
        coord_to_component_func(detector_coords.get(detector_id, []))
        for detector_id in range(dem.num_detectors)
    ]

    def component_using_coords(detector_id: int) -> int:
        return detector_components[detector_id]

    return decompose_errors_using_detector_assignment(
        dem=dem,
        detector_component_func=component_using_coords,
        strip_undecomposable_errors=strip_undecomposable_errors,
    )


def detector_coord_to_basis_for_stim_surface_code_convention(coord: tuple[int]) -> int:
    """For detector coordinates consistent with the stim.Circuit.generated
    surface code circuits, return the basis from the detector coordinate.
    Returns 0 for X basis and 1 for Z basis detector."""
    basis = stim_surface_code_detector_basis_classifier(0, coord, "")
    return 0 if basis == "X" else 1


def decompose_errors_using_detector_basis_classifier(
    dem: stim.DetectorErrorModel,
    detector_basis_classifier: DetectorBasisClassifier = (
        automatic_detector_basis_classifier
    ),
    strip_undecomposable_errors: bool = False,
) -> stim.DetectorErrorModel:
    """Decomposes errors using a shared detector X/Z basis classifier."""
    detector_bases = classify_detector_bases(
        dem, detector_basis_classifier=detector_basis_classifier
    )
    detector_components = [0 if basis == "X" else 1 for basis in detector_bases]
    return decompose_errors_using_detector_assignment(
        dem=dem,
        detector_component_func=detector_components.__getitem__,
        strip_undecomposable_errors=strip_undecomposable_errors,
    )


def decompose_errors_using_last_coordinate_index(
    dem: stim.DetectorErrorModel,
    strip_undecomposable_errors: bool = False,
) -> stim.DetectorErrorModel:
    """Decomposes errors in the detector error model `dem` based on an assignment of
    detectors to components by the last element of each detector coordinate.

    An undecomposed error is an error that flips detectors that are all in the same
    component. A decomposed error is an error that flips detectors from more than one
    component, but is decomposed into components where each component corresponds
    to an undecomposed error elsewhere in the dem. The symmetric difference of the
    detectors and observables in the components of a decomposed error will equal
    the detectors and observables of the original error in the dem.
    Existing Stim ``^`` decomposition groups are preserved after validation. Each
    existing group must contain at least one detector, and all detectors in a group
    must belong to the same component.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model to decompose.
    strip_undecomposable_errors : bool
        If True, errors that cannot be decomposed due to a missing component error
        will be stripped from the output DEM instead of raising a ValueError.

    Returns
    -------
    stim.DetectorErrorModel
        The decomposed detector error model
    """
    detector_coords = dem.get_detector_coordinates()
    detector_components = [
        last_coordinate_component_classifier(
            detector_id, detector_coords.get(detector_id, []), ""
        )
        for detector_id in range(dem.num_detectors)
    ]

    return decompose_errors_using_detector_assignment(
        dem=dem,
        detector_component_func=detector_components.__getitem__,
        strip_undecomposable_errors=strip_undecomposable_errors,
    )


def decompose_errors_for_stim_surface_code_coords(
    dem: stim.DetectorErrorModel,
    strip_undecomposable_errors: bool = False,
) -> stim.DetectorErrorModel:
    """Decomposes the errors in the dem, such that each component
    of a decomposed error only triggers detectors of one basis (X or Z)
    based on an assignment of detector coordinates to X or Z basis
    consistent with the convention used in stim.Circuit.generated
    surface code circuits.

    A detector is assumed to be X-type if `(x // 2 + y // 2) % 2 == 1`
    and is assumed to be Z-type if `(x // 2 + y // 2) % 2 == 0` where
    the detector has coordinates (x, y, ...).

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model to decompose
    strip_undecomposable_errors : bool
        If True, errors that cannot be decomposed due to a missing component error
        will be stripped from the output DEM instead of raising a ValueError.

    Returns
    -------
    stim.DetectorErrorModel
        The decomposed detector error model
    """
    return decompose_errors_using_detector_basis_classifier(
        dem=dem,
        detector_basis_classifier=stim_surface_code_detector_basis_classifier,
        strip_undecomposable_errors=strip_undecomposable_errors,
    )


def decompose_errors(
    dem: stim.DetectorErrorModel,
    method: str = "stim-surfacecode-coords",
    strip_undecomposable_errors: bool = False,
) -> stim.DetectorErrorModel:
    """Dispatches to a decomposition strategy selected by name."""
    if method == "stim-surfacecode-coords":
        return decompose_errors_for_stim_surface_code_coords(
            dem, strip_undecomposable_errors=strip_undecomposable_errors
        )
    if method == "last-coordinate-index":
        return decompose_errors_using_last_coordinate_index(
            dem, strip_undecomposable_errors=strip_undecomposable_errors
        )
    raise ValueError(
        "Unknown decomposition method "
        f"{method!r}. Expected 'stim-surfacecode-coords' or 'last-coordinate-index'."
    )


def undecompose_errors(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    """Returns a detector error model with any error decompositions removed.

    If an error is decomposed into components in the dem, it will be replaced with a
    single undecomposed error instruction (of the same probability) with detectors
    equal to the symmetric difference of the detectors of the components, and
    likewise for the observables. Repeat blocks are preserved, rather than flattened.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model to undecompose

    Returns
    -------
    stim.DetectorErrorModel
        The undecomposed detector error model
    """
    undecomposed_dem = stim.DetectorErrorModel()
    for instruction in dem:
        if instruction.type == "repeat":
            undecomposed_dem.append(
                instruction=stim.DemRepeatBlock(
                    repeat_count=instruction.repeat_count,
                    block=undecompose_errors(instruction.body_copy()),
                )
            )
            continue

        if instruction.type != "error":
            undecomposed_dem.append(instruction=instruction)
            continue

        detectors, observables = undecomposed_error_detectors_and_observables(
            instruction=instruction
        )

        targets = [stim.target_relative_detector_id(d) for d in detectors] + [
            stim.target_logical_observable_id(o) for o in observables
        ]

        undecomposed_dem.append(
            stim.DemInstruction(
                type=instruction.type,
                args=instruction.args_copy(),
                targets=targets,
                tag=instruction.tag,
            )
        )
    return undecomposed_dem


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Decompose errors in a Stim detector error model."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="-",
        help="Input DEM file (default: standard input; use '-' for standard input).",
    )
    parser.add_argument(
        "-o",
        "--out",
        default="-",
        help="Output DEM file (default: standard output; use '-' for standard output).",
    )
    parser.add_argument(
        "--method",
        choices=("stim-surfacecode-coords", "last-coordinate-index"),
        default="stim-surfacecode-coords",
        help="Detector-component convention used for decomposition.",
    )
    parser.add_argument(
        "--strip-undecomposable-errors",
        action="store_true",
        help="Drop errors that cannot be decomposed instead of failing.",
    )
    args = parser.parse_args()

    if args.input == "-":
        dem = stim.DetectorErrorModel(sys.stdin.read())
    else:
        dem = stim.DetectorErrorModel.from_file(args.input)

    output_dem = decompose_errors(
        dem,
        method=args.method,
        strip_undecomposable_errors=args.strip_undecomposable_errors,
    )
    if args.out == "-":
        print(output_dem)
    else:
        output_dem.to_file(args.out)
