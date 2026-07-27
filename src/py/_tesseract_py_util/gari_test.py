import itertools

import numpy as np
import pytest
import scipy.sparse
import stim

from _tesseract_py_util.gari import (
    build_gari_dem,
    dem_to_matrices,
    detector_partition_from_fourth_coordinate,
    gari_transform,
    paper_prior_probabilities,
    tesseract_lp_maximin_prior_probabilities,
    tesseract_xor_prior_probabilities,
)


def _tiny_model():
    source_dem = stim.DetectorErrorModel("""
        error(0.1) D0 D2 L0
        error(0.2) D1 D3 L1
        error(0.3) D0 D1 D2 D3 L0
        detector(0, 0, 0, 0) D0
        detector(0, 0, 0, 3) D1
        detector(0, 0, 0, 2) D2
        detector(0, 0, 0, 4) D3
    """)
    checks, logicals, probabilities = dem_to_matrices(source_dem)
    x_detectors, z_detectors = detector_partition_from_fourth_coordinate(
        source_dem
    )
    transform = gari_transform(
        checks,
        logicals,
        x_detectors=x_detectors,
        z_detectors=z_detectors,
    )
    return checks, logicals, probabilities, transform


def test_tiny_transform():
    with pytest.raises(ValueError, match="decompose_errors=False"):
        dem_to_matrices(stim.DetectorErrorModel("error(0.1) D0 ^ D1"))
    with pytest.raises(ValueError, match="fully flattened"):
        dem_to_matrices(stim.DetectorErrorModel("shift_detectors 1"))

    checks, logicals, _, transform = _tiny_model()
    np.testing.assert_array_equal(transform.e_z_columns, [0])
    np.testing.assert_array_equal(transform.e_x_columns, [1])
    np.testing.assert_array_equal(transform.e_y_columns, [2])
    np.testing.assert_array_equal(transform.u.toarray(), [[1]])
    np.testing.assert_array_equal(transform.v.toarray(), [[1]])
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
        transform.source_to_gari_detectors, [0, 2, 1, 3]
    )
    assert (
        transform.physical_x_rows,
        transform.physical_z_rows,
        transform.virtual_z_rows,
        transform.virtual_x_rows,
    ) == (slice(0, 2), slice(2, 4), slice(4, 5), slice(5, 6))

    for source_error in itertools.product([0, 1], repeat=3):
        e_z, e_x, e_y = source_error
        source_error = np.asarray(source_error, dtype=np.uint8)
        gari_error = np.asarray(
            [e_z, e_x, e_y, e_z ^ e_y, e_x ^ e_y], dtype=np.uint8
        )
        source_syndrome = np.asarray(checks @ source_error).reshape(-1) % 2
        expected = np.concatenate(
            [source_syndrome[[0, 2, 1, 3]], np.zeros(2, dtype=np.uint8)]
        )
        np.testing.assert_array_equal(
            np.asarray(transform.checks @ gari_error).reshape(-1) % 2,
            expected,
        )
        np.testing.assert_array_equal(
            np.asarray(transform.logicals @ gari_error).reshape(-1) % 2,
            np.asarray(logicals @ source_error).reshape(-1) % 2,
        )


def test_prior_probabilities_and_gari_dem_round_trip():
    _, _, source_probabilities, transform = _tiny_model()
    np.testing.assert_array_equal(
        paper_prior_probabilities(transform, source_probabilities),
        [0.1, 0.2, 0.3, 0.5, 0.5],
    )
    xor_probabilities = tesseract_xor_prior_probabilities(
        transform, source_probabilities
    )
    np.testing.assert_allclose(
        xor_probabilities, [0.1, 0.2, 0.3, 0.34, 0.38]
    )

    lp_probabilities = tesseract_lp_maximin_prior_probabilities(
        transform, source_probabilities
    )
    lp_costs = np.log1p(-lp_probabilities) - np.log(lp_probabilities)
    source_costs = np.log1p(-source_probabilities) - np.log(
        source_probabilities
    )
    np.testing.assert_allclose(
        lp_costs[:3] + np.asarray([[1, 0], [0, 1], [1, 1]]) @ lp_costs[3:],
        source_costs,
    )

    gari_dem = build_gari_dem(
        transform,
        source_probabilities,
        prior_function=tesseract_xor_prior_probabilities,
    )
    checks, logicals, probabilities = dem_to_matrices(gari_dem)
    assert gari_dem.num_detectors == transform.checks.shape[0]
    assert gari_dem.num_observables == transform.logicals.shape[0]
    assert (checks != transform.checks).nnz == 0
    assert (logicals != transform.logicals).nnz == 0
    np.testing.assert_allclose(probabilities, xor_probabilities)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
