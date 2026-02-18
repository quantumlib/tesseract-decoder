import pytest
from collections.abc import Iterable
import stim

from decompose_errors import (
    reduce_symmetric_difference,
    reduce_set_symmetric_difference,
    get_component_obs_matching_undecomposed_obs,
    decompose_errors_using_last_coordinate_index,
    detector_coord_to_basis_for_stim_surface_code_convention,
    decompose_errors_for_stim_surface_code_coords,
    undecompose_errors,
    decompose_errors_using_detector_coordinate_assignment,
)


@pytest.mark.parametrize(
    "items,expected_output",
    [([1, 2, 3], (1, 2, 3)), ([1, 1], tuple()), ([3, 0, 1, 4, 1, 2, 4], (0, 2, 3))],
)
def test_reduce_symmetric_difference(items: Iterable[int], expected_output):
    assert reduce_symmetric_difference(items) == expected_output


@pytest.mark.parametrize(
    "sets,expected_output",
    [([{1, 2, 3}, {2, 4, 0}], (0, 1, 3, 4)), ([{}], tuple())],
)
def test_reduce_set_symmetric_difference(sets: Iterable[set], expected_output):
    assert reduce_set_symmetric_difference(sets) == expected_output


@pytest.mark.parametrize(
    "component_obs,error_obs,expected_output",
    [
        ([{(0, 1), (2, 1)}, {(3, 4), (10, 0)}], (1, 10), [(0, 1), (10, 0)]),
        ([{tuple()}, {tuple()}], tuple(), [tuple(), tuple()]),
        ([{tuple()}, {tuple()}], (0,), None),
    ],
)
def test_get_component_obs_matching_undecomposed_obs(
    component_obs, error_obs, expected_output
):
    assert (
        get_component_obs_matching_undecomposed_obs(component_obs, error_obs)
        == expected_output
    )


def test_do_decomposition_last_coordinate_index_two_components():
    dem = stim.DetectorErrorModel("""error(0.1) D0 ^ D1 L1
error(0.01) D0 D3 D3 D1 L5 L4 L4
error(0.3) D0 D1 D3 D3 D2 D3 L0 L5
error(0.2) D3 D2 D0 D0 L0
detector(0) D0
detector(0) D1
detector(1) D2
detector(1) D3""")
    assert str(decompose_errors_using_last_coordinate_index(dem)) == str(
        stim.DetectorErrorModel("""error(0.1) D0 D1 L1
error(0.01) D0 D1 L5
error(0.3) D0 D1 L5 ^ D2 D3 L0
error(0.2) D2 D3 L0
detector(0) D0
detector(0) D1
detector(1) D2
detector(1) D3""")
    )


def test_do_decomposition_last_coordinate_index_three_components():
    dem = stim.DetectorErrorModel("""error(0.1) D0 ^ D1 L1
error(0.01) D0 D1 L5
error(0.3) D0 D1 D2 D3 L0 L5
error(0.2) D2 D3 L0
error(0.35) D0 D1 D2 D3 D5 L5 L10
error(0.6) D5 L0 L10
detector(2,0) D0
detector(2,0) D1
detector(2,1) D2
detector(2,1) D3
detector(2,2) D5""")
    assert str(decompose_errors_using_last_coordinate_index(dem)) == str(
        stim.DetectorErrorModel("""error(0.1) D0 D1 L1
error(0.01) D0 D1 L5
error(0.3) D0 D1 L5 ^ D2 D3 L0
error(0.2) D2 D3 L0
error(0.35) D0 D1 L5 ^ D2 D3 L0 ^ D5 L0 L10
error(0.6) D5 L0 L10
detector(2,0) D0
detector(2,0) D1
detector(2,1) D2
detector(2,1) D3
detector(2,2) D5""")
    )


def test_decompose_undecomposable_error():
    dem = stim.DetectorErrorModel("""error(0.01) D0 D1 L5
error(0.3) D0 D1 D2 D3 L5
detector(0) D0
detector(0) D1
detector(1) D2
detector(1) D3""")
    with pytest.raises(ValueError):
        decompose_errors_using_last_coordinate_index(dem)


def test_decompose_error_without_consistent_obs_decomposition():
    dem = stim.DetectorErrorModel("""error(0.01) D0 D1 L5
error(0.2) D2 D3 L5
error(0.3) D0 D1 D2 D3 L5
detector(0) D0
detector(0) D1
detector(1) D2
detector(1) D3""")
    with pytest.raises(ValueError):
        decompose_errors_using_last_coordinate_index(dem)


def add_basis_coord_to_detector_coords(circuit: stim.Circuit) -> stim.Circuit:
    new_circuit = stim.Circuit()

    for inst in circuit:
        if inst.name == "REPEAT":
            new_circuit.append(
                stim.CircuitRepeatBlock(
                    repeat_count=inst.repeat_count,
                    body=add_basis_coord_to_detector_coords(inst.body_copy()),
                    tag=inst.tag,
                )
            )
            continue

        if inst.name != "DETECTOR":
            new_circuit.append(inst)
            continue
        coords = inst.gate_args_copy()
        coords.append(detector_coord_to_basis_for_stim_surface_code_convention(coords))

        new_circuit.append(
            stim.CircuitInstruction(
                name=inst.name,
                targets=inst.targets_copy(),
                gate_args=coords,
                tag=inst.tag,
            )
        )
    return new_circuit


def test_undecompose_errors_surface_code():
    circuit = stim.Circuit.generated(
        code_task="surface_code:rotated_memory_x",
        distance=5,
        rounds=15,
        after_clifford_depolarization=0.001,
    )

    dem_undecomposed_original_flattened = circuit.detector_error_model().flattened()
    dem_decomposed_using_coords = decompose_errors_for_stim_surface_code_coords(
        dem_undecomposed_original_flattened
    )
    dem_decomposed_using_coords_undecomposed = undecompose_errors(
        dem_decomposed_using_coords
    )
    assert str(dem_undecomposed_original_flattened) == str(
        dem_decomposed_using_coords_undecomposed
    )

    dem_undecomposed_original = circuit.detector_error_model()
    dem_decomposed_original = circuit.detector_error_model(decompose_errors=True)
    dem_undecomposed_from_original = undecompose_errors(dem_decomposed_original)
    assert (
        dem_undecomposed_original.num_detectors
        == dem_undecomposed_from_original.num_detectors
    )
    assert (
        dem_undecomposed_original.num_observables
        == dem_undecomposed_from_original.num_observables
    )

    dem_decomposed_using_coords_func = decompose_errors_using_detector_coordinate_assignment(
        dem=circuit.detector_error_model(),
        coord_to_component_func=detector_coord_to_basis_for_stim_surface_code_convention,
    )
    assert dem_decomposed_using_coords_func == dem_decomposed_using_coords


def test_undecompose_errors_with_repeat_block():
    dem = stim.DetectorErrorModel("""error(0.1) D2 D5 ^ D10 L1
repeat 10 {
    error(0.4) D1 L2 L3 ^ D2 ^ D2 L2
    repeat 3 {
        error(0.3) D10 D11 ^ D12
    }
}
error(0.5) D0 D100""")
    dem_undecomposed = undecompose_errors(dem=dem)
    expected_dem_undecomposed = stim.DetectorErrorModel("""error(0.1) D2 D5 D10 L1
repeat 10 {
    error(0.4) D1 L3
    repeat 3 {
        error(0.3) D10 D11 D12
    }
}
error(0.5) D0 D100""")
    assert str(dem_undecomposed) == str(expected_dem_undecomposed)
