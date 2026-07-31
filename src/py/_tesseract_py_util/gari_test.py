import numpy as np
import pytest
import stim

from _tesseract_py_util import gari


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
    checks, logicals, probabilities = gari.dem_to_matrices(source_dem)
    x_detectors, z_detectors = gari.detector_partition_from_fourth_coordinate(
        source_dem
    )
    transform = gari.gari_transform(
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
        gari.detector_partition_from_fourth_coordinate(
            stim.DetectorErrorModel("detector(0, 0, 0, 2.5) D0")
        )

    with pytest.raises(ValueError, match="decompose_errors=False"):
        gari.dem_to_matrices(stim.DetectorErrorModel("error(0.1) D0 ^ D1"))

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
        transform.source_to_gari_detectors, [0, 2, 1, 3]
    )


def test_prior_probabilities_and_gari_dem_round_trip():
    source_probabilities, transform = _tiny_model()
    np.testing.assert_array_equal(
        gari.paper_prior_probabilities(transform, source_probabilities),
        [0.1, 0.2, 0.3, 0.5, 0.5],
    )
    xor_probabilities = gari.tesseract_xor_prior_probabilities(
        transform, source_probabilities
    )
    np.testing.assert_allclose(
        xor_probabilities, [0.1, 0.2, 0.3, 0.34, 0.38]
    )

    gari_dem = gari.build_gari_dem(
        transform,
        source_probabilities,
        prior_function=gari.tesseract_xor_prior_probabilities,
    )
    checks, logicals, probabilities = gari.dem_to_matrices(gari_dem)
    assert gari_dem.num_detectors == transform.checks.shape[0]
    assert gari_dem.num_observables == transform.logicals.shape[0]
    assert (checks != transform.checks).nnz == 0
    assert (logicals != transform.logicals).nnz == 0
    np.testing.assert_allclose(probabilities, xor_probabilities)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
