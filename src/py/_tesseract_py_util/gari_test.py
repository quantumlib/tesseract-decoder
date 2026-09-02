# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
import stim

from _tesseract_py_util import gari
from tesseract_decoder import demutil, utils


def _tiny_circuit():
    return stim.Circuit("""
        R 0 1 2 3 4 5
        CORRELATED_ERROR(0.01) X0 X2 X4
        CORRELATED_ERROR(0.02) X1 X3 X5
        CORRELATED_ERROR(0.04) X0 X1 X2 X3 X4
        M 0 1 2 3 4 5
        DETECTOR(0, 0, 0, 3) rec[-5]
        DETECTOR(0, 0, 0, 0) rec[-6]
        DETECTOR(0, 0, 0, 4) rec[-3]
        DETECTOR(0, 0, 0, 2) rec[-4]
        OBSERVABLE_INCLUDE(0) rec[-2]
        OBSERVABLE_INCLUDE(1) rec[-1]
    """)


def _tiny_model():
    source_dem = gari._circuit_to_gari_source_dem(_tiny_circuit())
    checks, logicals, probabilities = gari.dem_to_matrices(source_dem)
    x_detectors, z_detectors = gari._detector_partition_from_fourth_coordinate(
        source_dem
    )
    transform = gari._gari_transform(
        checks,
        logicals,
        x_detectors=x_detectors,
        z_detectors=z_detectors,
    )
    return probabilities, transform


def test_tiny_transform():
    folded_dem = stim.DetectorErrorModel("""
        repeat 2 {
            error(0.1) D0 L0
            shift_detectors 1
        }
    """)
    checks, logicals, probabilities = gari.dem_to_matrices(folded_dem)
    np.testing.assert_array_equal(checks.toarray(), np.eye(2, dtype=np.uint8))
    np.testing.assert_array_equal(logicals.toarray(), [[1, 1]])
    np.testing.assert_allclose(probabilities, [0.1, 0.1])

    with pytest.raises(ValueError, match="integer from 0 to 2"):
        gari._detector_partition_from_fourth_coordinate(
            stim.DetectorErrorModel("detector(0, 0, 0, 2.5) D0")
        )

    with pytest.raises(ValueError, match="decompose_errors=False"):
        gari.dem_to_matrices(stim.DetectorErrorModel("error(0.1) D0 ^ D1"))

    checks, logicals, probabilities = gari.dem_to_matrices(
        stim.DetectorErrorModel("""
            error(0.1) D0 D0
            error(0.2) D0 D0 D1 L0 L1 L1
        """)
    )
    np.testing.assert_array_equal(checks.toarray(), [[0], [1]])
    np.testing.assert_array_equal(logicals.toarray(), [[1], [0]])
    np.testing.assert_allclose(probabilities, [0.2])

    with pytest.raises(ValueError, match="logical-only source errors"):
        gari.dem_to_matrices(
            stim.DetectorErrorModel("error(0.1) D0 D0 L0")
        )

    _, transform = _tiny_model()
    np.testing.assert_array_equal(
        transform.checks.toarray(),
        [
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0],
            [0, 1, 1, 0, 1],
        ],
    )
    np.testing.assert_array_equal(
        transform.logicals.toarray(),
        [[1, 0, 1, 0, 0], [0, 1, 0, 0, 0]],
    )
    np.testing.assert_array_equal(
        transform.source_to_gari_detectors, [2, 0, 3, 1]
    )


def test_prior_probabilities_and_gari_dem_round_trip():
    source_probabilities, transform = _tiny_model()
    paper_probabilities = gari.paper_prior_probabilities(
        transform, source_probabilities
    )
    np.testing.assert_array_equal(
        paper_probabilities,
        [0.01, 0.02, 0.04, 0.5, 0.5],
    )
    xor_probabilities = gari.tesseract_xor_prior_probabilities(
        transform, source_probabilities
    )
    np.testing.assert_allclose(
        xor_probabilities, [0.01, 0.02, 0.04, 0.0492, 0.0584]
    )
    lp_probabilities = gari.tesseract_lp_max_barred_cost_prior_probabilities(
        transform, source_probabilities
    )
    source_costs = np.log1p(-paper_probabilities[:3]) - np.log(
        paper_probabilities[:3]
    )
    lp_costs = np.log1p(-lp_probabilities) - np.log(lp_probabilities)
    assert np.all(lp_costs > 0)
    np.testing.assert_allclose(
        lp_costs[2:], source_costs[2] / 3, rtol=1e-6
    )
    np.testing.assert_allclose(
        [
            lp_costs[0] + lp_costs[3],
            lp_costs[1] + lp_costs[4],
            lp_costs[2] + lp_costs[3] + lp_costs[4],
        ],
        source_costs,
    )

    gari_dem = gari._build_gari_dem(
        transform,
        source_probabilities,
        prior_function=gari.tesseract_xor_prior_probabilities,
        row_order="block",
    )
    checks, logicals, probabilities = gari.dem_to_matrices(gari_dem)
    assert gari_dem.num_detectors == transform.checks.shape[0]
    assert gari_dem.num_observables == transform.logicals.shape[0]
    assert (checks != transform.checks).nnz == 0
    assert (logicals != transform.logicals).nnz == 0
    np.testing.assert_allclose(probabilities, xor_probabilities)


def test_public_circuit_conversion_and_file_output(tmp_path):
    public_gari = demutil.gari
    circuit = _tiny_circuit()
    gari_dem = public_gari.circuit_to_gari(
        circuit,
        prior_function=public_gari.tesseract_xor_prior_probabilities,
    )
    assert gari_dem.num_detectors == 6
    assert gari_dem.num_observables == 2
    _, transform = _tiny_model()
    checks, _, _ = gari.dem_to_matrices(gari_dem)
    np.testing.assert_array_equal(
        checks.toarray(), transform.checks[[2, 0, 3, 1, 4, 5], :].toarray()
    )
    source_orders = utils.build_det_orders(
        gari._circuit_to_gari_source_dem(circuit),
        2,
        method=utils.DetectorOrderMethod.Coordinate,
        seed=0,
    )
    assert any(
        list(source_order)
        != sorted(range(4), key=source_order.__getitem__)
        for source_order in source_orders
    )
    gari_orders = public_gari.build_detector_orders(
        circuit,
        gari_dem,
        2,
        method=utils.DetectorOrderMethod.Coordinate,
        seed=0,
    )
    assert [order[:4] for order in gari_orders] == source_orders
    assert [order[4:] for order in gari_orders] == [[4, 5], [4, 5]]

    block_dem = public_gari.circuit_to_gari(
        circuit,
        prior_function=public_gari.tesseract_xor_prior_probabilities,
        row_order="block",
    )
    block_checks, _, _ = gari.dem_to_matrices(block_dem)
    assert (block_checks != transform.checks).nnz == 0

    circuit_path = tmp_path / "tiny.stim"
    circuit.to_file(circuit_path)
    output_dir = tmp_path / "gari"
    public_gari.call_gari(str(circuit_path), "xor", str(output_dir))
    output_name = "tiny_gari_xor"
    written_dem = stim.DetectorErrorModel.from_file(
        output_dir / f"{output_name}.dem"
    )
    public_gari.call_gari(
        str(circuit_path), "xor", str(output_dir), row_order="block"
    )
    assert str(written_dem) == str(gari_dem)
    assert (output_dir / f"{output_name}_block.dem").is_file()


def test_build_detector_orders_validates_source_aligned_transform():
    public_gari = demutil.gari
    circuit = _tiny_circuit()
    gari_dem = public_gari.circuit_to_gari(
        circuit,
        prior_function=public_gari.tesseract_xor_prior_probabilities,
    )
    expected_orders = public_gari.build_detector_orders(
        circuit,
        gari_dem,
        2,
        method=utils.DetectorOrderMethod.Coordinate,
        seed=0,
    )

    checks, logicals, probabilities = gari.dem_to_matrices(gari_dem)
    changed_probabilities = np.linspace(0.11, 0.21, checks.shape[1])
    assert not np.allclose(changed_probabilities, probabilities)
    reweighted_dem = gari._matrices_to_gari_dem(
        checks, logicals, changed_probabilities
    )
    assert (
        public_gari.build_detector_orders(
            circuit,
            reweighted_dem,
            2,
            method=utils.DetectorOrderMethod.Coordinate,
            seed=0,
        )
        == expected_orders
    )

    block_dem = public_gari.circuit_to_gari(
        circuit,
        prior_function=public_gari.tesseract_xor_prior_probabilities,
        row_order="block",
    )
    with pytest.raises(ValueError, match="source-aligned GARI transform"):
        public_gari.build_detector_orders(circuit, block_dem, 1)

    unrelated_checks = checks.tolil()
    unrelated_checks[0, 0] = 1 - int(unrelated_checks[0, 0])
    unrelated_dem = gari._matrices_to_gari_dem(
        unrelated_checks.tocsc(), logicals, probabilities
    )
    assert unrelated_dem.num_detectors == gari_dem.num_detectors
    with pytest.raises(ValueError, match="source-aligned GARI transform"):
        public_gari.build_detector_orders(circuit, unrelated_dem, 1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
