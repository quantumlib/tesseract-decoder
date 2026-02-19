import stim
from functools import reduce
import itertools
from collections import defaultdict
from collections.abc import Callable, Iterable


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
    dem: stim.DetectorErrorModel, detector_component_func: Callable[[int], int]
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
    If the dem provided was already decomposed, this decomposition will be ignored
    (each error will be undecomposed before the new decomposition is applied).

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model to decompose.
    detector_component_func : Callable[[int], int]
        A function that maps a detector id to its component. i.e. This could map
        a detector index to 0 if it is X-type or to 1 if it is Z-type.

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

        # Assign observables to each component, such that they are consistent with the
        # observables of the undecomposed error
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

        decomposed_instruction = stim.DemInstruction(
            type=instruction.type,
            args=instruction.args_copy(),
            targets=targets,
            tag=instruction.tag,
        )
        output_dem.append(decomposed_instruction)

    return output_dem


def decompose_errors_using_detector_coordinate_assignment(
    dem: stim.DetectorErrorModel, coord_to_component_func: Callable[[list[float]], int]
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

    Returns
    -------
    stim.DetectorErrorModel
        The decomposed detector error model. Note that the DEM will also be flattened.
    """
    detector_coords = dem.get_detector_coordinates()

    def component_using_coords(detector_id: int) -> int:
        return coord_to_component_func(detector_coords[detector_id])

    return decompose_errors_using_detector_assignment(
        dem=dem, detector_component_func=component_using_coords
    )


def detector_coord_to_basis_for_stim_surface_code_convention(coord: tuple[int]) -> int:
    """For detector coordinates consistent with the stim.Circuit.generated
    surface code circuits, return the basis from the detector coordinate.
    Returns 0 for X basis and 1 for Z basis detector."""
    x = coord[0]
    y = coord[1]
    return 1 - ((x // 2 + y // 2) % 2)


def decompose_errors_using_last_coordinate_index(
    dem: stim.DetectorErrorModel,
) -> stim.DetectorErrorModel:
    """Decomposes errors in the detector error model `dem` based on an assignment of
    detectors to components by the last element of each detector coordinate.

    An undecomposed error is an error that flips detectors that are all in the same
    component. A decomposed error is an error that flips detectors from more than one
    component, but is decomposed into components where each component corresponds
    to an undecomposed error elsewhere in the dem. The symmetric difference of the
    detectors and observables in the components of a decomposed error will equal
    the detectors and observables of the original error in the dem.
    If the dem provided was already decomposed, this decomposition will be ignored
    (each error will be undecomposed before the new decomposition is applied).

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model to decompose.

    Returns
    -------
    stim.DetectorErrorModel
        The decomposed detector error model
    """
    detector_coords = dem.get_detector_coordinates()

    def last_coordinate_component(detector_id: int) -> int:
        return detector_coords[detector_id][-1]

    return decompose_errors_using_detector_assignment(
        dem=dem, detector_component_func=last_coordinate_component
    )


def decompose_errors_for_stim_surface_code_coords(
    dem: stim.DetectorErrorModel,
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

    Returns
    -------
    stim.DetectorErrorModel
        The decomposed detector error model
    """
    detector_coords = dem.get_detector_coordinates()

    def stim_surface_code_det_component(detector_id: int) -> int:
        return detector_coord_to_basis_for_stim_surface_code_convention(
            detector_coords[detector_id]
        )

    return decompose_errors_using_detector_assignment(
        dem=dem, detector_component_func=stim_surface_code_det_component
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
