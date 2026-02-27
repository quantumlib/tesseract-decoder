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

"""Utilities for detector error model decomposition and re-generalization."""

from collections import defaultdict
from collections.abc import Callable, Iterable
from functools import reduce
import itertools
from typing import List

import stim


def reduce_symmetric_difference(items: Iterable[int]) -> tuple[int]:
    """Calculates the symmetric difference of a multiset of items."""
    unpaired_set = reduce(lambda acc, i: acc ^ {i}, items, set())
    return tuple(sorted(unpaired_set))


def reduce_set_symmetric_difference(sets: Iterable[Iterable[int]]) -> tuple[int]:
    return reduce_symmetric_difference(itertools.chain.from_iterable(sets))


def undecomposed_error_detectors_and_observables(
    instruction: stim.DemInstruction,
) -> tuple[tuple[int], tuple[int]]:
    """Outputs detector/observable indices in a stim error, undecomposed if needed."""
    if instruction.type != "error":
        raise ValueError(f"DEM instruction must be an error, not {instruction.type}")
    detectors = reduce_symmetric_difference(
        d.val for d in instruction.targets_copy() if d.is_relative_detector_id()
    )
    observables = reduce_symmetric_difference(
        o.val for o in instruction.targets_copy() if o.is_logical_observable_id()
    )
    return detectors, observables


def get_component_obs_matching_undecomposed_obs(
    obs_options_by_component: list[set[tuple[int]]], error_obs: tuple[int]
) -> list[tuple[int]] | None:
    """Assign component observables consistent with the undecomposed error observables."""
    error_obs_set = set(reduce_symmetric_difference(error_obs))
    for obs_combinations in itertools.product(*obs_options_by_component):
        obs_from_combination = reduce_set_symmetric_difference(obs_combinations)
        if set(obs_from_combination) == error_obs_set:
            return list(obs_combinations)
    return None


def decompose_errors_using_detector_assignment(
    dem: stim.DetectorErrorModel, detector_component_func: Callable[[int], int]
) -> stim.DetectorErrorModel:
    """Decomposes DEM errors from a detector->component assignment function."""
    dem = dem.flattened()

    single_component_dets_to_obs: dict[tuple[int], set[tuple[int]]] = defaultdict(set)

    for instruction in dem:
        if instruction.type != "error":
            continue

        detectors, observables = undecomposed_error_detectors_and_observables(
            instruction=instruction
        )

        if len(set(detector_component_func(d) for d in detectors)) == 1:
            single_component_dets_to_obs[detectors].add(observables)

    output_dem = stim.DetectorErrorModel()

    for instruction in dem:
        if instruction.type != "error":
            output_dem.append(instruction)
            continue

        detectors, observables = undecomposed_error_detectors_and_observables(
            instruction=instruction
        )
        det_components = {d: detector_component_func(d) for d in detectors}
        unique_components = sorted(set(det_components.values()))
        num_components = len(unique_components)

        dets_by_component = []
        obs_options_by_component = []

        for c in unique_components:
            component_dets = tuple(
                sorted(d for d in detectors if det_components[d] == c)
            )
            if component_dets not in single_component_dets_to_obs:
                raise ValueError(
                    f"The dem error `{instruction}` needs to be decomposed into components, however "
                    f"the component with detectors {component_dets} is not present as its own error "
                    "in the dem."
                )
            dets_by_component.append(component_dets)
            obs_options_by_component.append(
                single_component_dets_to_obs[component_dets]
            )

        consistent_obs_by_component = get_component_obs_matching_undecomposed_obs(
            obs_options_by_component=obs_options_by_component, error_obs=observables
        )

        if consistent_obs_by_component is None:
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

        output_dem.append(
            stim.DemInstruction(
                type=instruction.type,
                args=instruction.args_copy(),
                targets=targets,
                tag=instruction.tag,
            )
        )

    return output_dem


def decompose_errors_using_detector_coordinate_assignment(
    dem: stim.DetectorErrorModel, coord_to_component_func: Callable[[list[float]], int]
) -> stim.DetectorErrorModel:
    """Decomposes DEM errors from a detector-coordinate->component assignment function."""
    detector_coords = dem.get_detector_coordinates()

    def component_using_coords(detector_id: int) -> int:
        return coord_to_component_func(detector_coords[detector_id])

    return decompose_errors_using_detector_assignment(
        dem=dem, detector_component_func=component_using_coords
    )


def detector_coord_to_basis_for_stim_surface_code_convention(coord: tuple[int]) -> int:
    """Returns 0 for X and 1 for Z from stim generated surface-code coordinates."""
    x = coord[0]
    y = coord[1]
    return 1 - ((x // 2 + y // 2) % 2)


def decompose_errors_using_last_coordinate_index(
    dem: stim.DetectorErrorModel,
) -> stim.DetectorErrorModel:
    """Decomposes DEM errors using the last detector coordinate as component id."""
    detector_coords = dem.get_detector_coordinates()

    def last_coordinate_component(detector_id: int) -> int:
        return detector_coords[detector_id][-1]

    return decompose_errors_using_detector_assignment(
        dem=dem, detector_component_func=last_coordinate_component
    )


def decompose_errors_for_stim_surface_code_coords(
    dem: stim.DetectorErrorModel,
) -> stim.DetectorErrorModel:
    """Decomposes DEM errors by inferred X/Z detector basis from coordinates."""
    detector_coords = dem.get_detector_coordinates()

    def stim_surface_code_det_component(detector_id: int) -> int:
        return detector_coord_to_basis_for_stim_surface_code_convention(
            detector_coords[detector_id]
        )

    return decompose_errors_using_detector_assignment(
        dem=dem, detector_component_func=stim_surface_code_det_component
    )


def undecompose_errors(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    """Returns a detector error model with decomposition separators removed."""
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


def _get_dets_logicals(error: stim.DemInstruction) -> tuple[set[int], set[int]]:
    dets: set[int] = set()
    logicals: set[int] = set()
    for t in error.targets_copy():
        if t.is_logical_observable_id():
            logicals = logicals.symmetric_difference({t.val})
        elif t.is_relative_detector_id():
            dets = dets.symmetric_difference({t.val})
    return dets, logicals


def regeneralize_spatial_dem(
    templates: List[stim.DetectorErrorModel],
    scaffold: stim.DetectorErrorModel,
    verbose: bool = False,
) -> stim.DetectorErrorModel:
    """Recomputes scaffold error probabilities by averaging matching template errors."""

    def detector_coords(dem: stim.DetectorErrorModel) -> dict[int, tuple[float, ...]]:
        coords = dem.get_detector_coordinates()
        return {k: tuple(v[:3]) for k, v in coords.items()}

    def spatial_key(
        coords: dict[int, tuple[float, ...]],
        dets: set[int],
        logicals: set[int],
    ) -> tuple:
        d_coords = sorted(coords[d] for d in dets)
        min_d = d_coords[0]
        rel = tuple(
            sorted(
                tuple(c[i] - min_d[i] for i in range(len(min_d)))
                for c in d_coords
            )
        )
        return ((min_d[0], min_d[1]), rel, tuple(sorted(logicals)))

    def merged_errors(dem: stim.DetectorErrorModel) -> list[dict]:
        errors_by_symptom = {}
        for error in dem.flattened():
            if error.type != "error":
                continue
            probability = error.args_copy()[0]
            dets, obs = _get_dets_logicals(error)
            key = (tuple(sorted(dets)), tuple(sorted(obs)))
            if key in errors_by_symptom:
                p0 = errors_by_symptom[key]["probability"]
                probability = p0 * (1 - probability) + (1 - p0) * probability
            errors_by_symptom[key] = {
                "probability": probability,
                "detectors": list(dets),
                "observables": list(obs),
            }
        return list(errors_by_symptom.values())

    template_probabilities: dict[tuple, list[float]] = defaultdict(list)
    for template in templates:
        coords = detector_coords(template)
        for error in merged_errors(template):
            key = spatial_key(coords, set(error["detectors"]), set(error["observables"]))
            template_probabilities[key].append(error["probability"])

    if verbose:
        print(f"identified {len(template_probabilities)} distinct template error keys")

    mean_probability = {
        k: sum(v) / len(v)
        for k, v in template_probabilities.items()
    }

    output_dem = stim.DetectorErrorModel()
    for instruction in scaffold.flattened():
        if instruction.type != "error":
            output_dem.append(instruction)

    scaffold_coords = detector_coords(scaffold)
    for error in merged_errors(scaffold):
        key = spatial_key(
            scaffold_coords,
            set(error["detectors"]),
            set(error["observables"]),
        )
        if key not in mean_probability:
            raise ValueError(f"Missing template probability for scaffold error key {key}.")
        output_dem.append(
            stim.DemInstruction(
                type="error",
                args=[mean_probability[key]],
                targets=[
                    *[
                        stim.target_relative_detector_id(d)
                        for d in error["detectors"]
                    ],
                    *[
                        stim.target_logical_observable_id(o)
                        for o in error["observables"]
                    ],
                ],
            )
        )

    return output_dem


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
