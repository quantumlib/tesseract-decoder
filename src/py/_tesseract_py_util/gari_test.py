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

import itertools

import numpy as np
import pytest
import scipy.optimize
import scipy.sparse
import stim

import _tesseract_py_util.gari as gari_module
from _tesseract_py_util.gari import (
    GariTransform,
    _matrices_to_decoder_dem,
    build_gari_decoder_dem,
    dem_to_matrices,
    detector_partition_from_last_coordinate,
    gari_transform,
    paper_prior_probabilities,
    tesseract_lp_maximin_prior_probabilities,
    tesseract_xor_prior_probabilities,
)


_X_DETECTORS = [0, 2]
_Z_DETECTORS = [1, 3]


def _tiny_source() -> tuple[
    scipy.sparse.csc_matrix, scipy.sparse.csc_matrix
]:
    # Columns are e_Z, e_X, e_Y. Rows are interleaved X, Z, X, Z.
    checks = scipy.sparse.csc_matrix(
        [
            [1, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=np.uint8,
    )
    logicals = scipy.sparse.csc_matrix(
        [
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )
    return checks, logicals


def _tiny_transform() -> GariTransform:
    checks, logicals = _tiny_source()
    return gari_transform(
        checks,
        logicals,
        x_detectors=_X_DETECTORS,
        z_detectors=_Z_DETECTORS,
    )


def _augmented_error(
    transform: GariTransform, source_error: np.ndarray
) -> np.ndarray:
    e_z = source_error[transform.e_z_columns]
    e_x = source_error[transform.e_x_columns]
    e_y = source_error[transform.e_y_columns]
    bar_e_z = (e_z + transform.u @ e_y) % 2
    bar_e_x = (e_x + transform.v @ e_y) % 2
    return np.concatenate([e_z, e_x, e_y, bar_e_z, bar_e_x]).astype(
        np.uint8
    )


def test_exact_tiny_transform():
    transform = _tiny_transform()

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
    assert transform.physical_x_rows == slice(0, 2)
    assert transform.physical_z_rows == slice(2, 4)
    assert transform.virtual_z_rows == slice(4, 5)
    assert transform.virtual_x_rows == slice(5, 6)

    source_probabilities = np.asarray([0.1, 0.2, 0.3])
    np.testing.assert_array_equal(
        paper_prior_probabilities(transform, source_probabilities),
        [0.1, 0.2, 0.3, 0.5, 0.5],
    )
    np.testing.assert_allclose(
        tesseract_xor_prior_probabilities(
            transform, source_probabilities
        ),
        [0.1, 0.2, 0.3, 0.34, 0.38],
    )

    source_checks, source_logicals = _tiny_source()
    source_permutation = [2, 0, 1]
    permuted_transform = gari_transform(
        source_checks[:, source_permutation],
        source_logicals[:, source_permutation],
        x_detectors=_X_DETECTORS,
        z_detectors=_Z_DETECTORS,
    )
    permuted_probabilities = source_probabilities[source_permutation]
    np.testing.assert_array_equal(
        paper_prior_probabilities(
            permuted_transform, permuted_probabilities
        ),
        [0.1, 0.2, 0.3, 0.5, 0.5],
    )
    np.testing.assert_allclose(
        tesseract_xor_prior_probabilities(
            permuted_transform, permuted_probabilities
        ),
        [0.1, 0.2, 0.3, 0.34, 0.38],
    )

    lp_probabilities = tesseract_lp_maximin_prior_probabilities(
        transform, source_probabilities
    )
    lp_costs = np.log1p(-lp_probabilities) - np.log(lp_probabilities)
    source_costs = np.log1p(-source_probabilities) - np.log(
        source_probabilities
    )
    cost_matrix = np.asarray([[1, 0], [0, 1], [1, 1]])
    np.testing.assert_allclose(
        lp_costs[:3] + cost_matrix @ lp_costs[3:], source_costs
    )
    assert np.min(lp_costs) == pytest.approx(source_costs[2] / 3)
    assert np.all(lp_costs >= 0)

    custom_probabilities = np.linspace(
        0.05, 0.25, num=transform.checks.shape[1]
    )

    def custom_prior(callback_transform, callback_probabilities):
        assert callback_transform is transform
        assert not callback_probabilities.flags.writeable
        return custom_probabilities

    custom_dem = build_gari_decoder_dem(
        transform,
        source_probabilities,
        prior_function=custom_prior,
    )
    _, _, serialized_custom_probabilities = dem_to_matrices(custom_dem)
    np.testing.assert_allclose(
        serialized_custom_probabilities, custom_probabilities
    )

    source_dem = stim.DetectorErrorModel("""
        error(0.125) D0 D0 D1 ^ D2 D2 L0 L0 L2
        detector(0, 1) D0
        detector(0, 3) D1
        detector(0, 1) D2
        detector(0, 3) D3
        logical_observable L4
    """)
    checks, logicals, probabilities = dem_to_matrices(source_dem)
    assert checks.shape == (4, 1)
    assert logicals.shape == (5, 1)
    assert _column_support(checks, 0) == [1]
    assert _column_support(logicals, 0) == [2]
    np.testing.assert_array_equal(probabilities, [0.125])
    x_detectors, z_detectors = detector_partition_from_last_coordinate(
        source_dem
    )
    np.testing.assert_array_equal(x_detectors, _X_DETECTORS)
    np.testing.assert_array_equal(z_detectors, _Z_DETECTORS)

    decoder_probabilities = np.linspace(
        0.1, 0.5, num=transform.checks.shape[1]
    )
    decoder_dem = _matrices_to_decoder_dem(
        transform.checks, transform.logicals, decoder_probabilities
    )
    reparsed_dem = stim.DetectorErrorModel(str(decoder_dem))
    round_trip_checks, round_trip_logicals, round_trip_probabilities = (
        dem_to_matrices(reparsed_dem)
    )
    assert reparsed_dem.num_detectors == transform.checks.shape[0]
    assert reparsed_dem.num_observables == transform.logicals.shape[0]
    assert reparsed_dem.num_errors == transform.checks.shape[1]
    assert (round_trip_checks != transform.checks).nnz == 0
    assert (round_trip_logicals != transform.logicals).nnz == 0
    np.testing.assert_allclose(
        round_trip_probabilities, decoder_probabilities
    )


def test_exhaustive_equivalence_and_virtual_constraints():
    checks, logicals = _tiny_source()
    transform = _tiny_transform()

    for bits in itertools.product([0, 1], repeat=checks.shape[1]):
        source_error = np.asarray(bits, dtype=np.uint8)
        gari_error = _augmented_error(transform, source_error)
        source_syndrome = np.asarray(checks @ source_error).reshape(-1) % 2
        expected_syndrome = np.concatenate(
            [source_syndrome[[0, 2]], source_syndrome[[1, 3]], [0, 0]]
        )
        gari_syndrome = np.asarray(transform.checks @ gari_error).reshape(-1) % 2
        np.testing.assert_array_equal(gari_syndrome, expected_syndrome)
        np.testing.assert_array_equal(
            np.asarray(transform.logicals @ gari_error).reshape(-1) % 2,
            np.asarray(logicals @ source_error).reshape(-1) % 2,
        )

    consistent_count = 0
    for bits in itertools.product([0, 1], repeat=transform.checks.shape[1]):
        gari_error = np.asarray(bits, dtype=np.uint8)
        syndrome = np.asarray(transform.checks @ gari_error).reshape(-1) % 2
        if np.any(syndrome[transform.virtual_z_rows]) or np.any(
            syndrome[transform.virtual_x_rows]
        ):
            continue
        consistent_count += 1
        e_z, e_x, e_y, bar_e_z, bar_e_x = gari_error
        assert bar_e_z == (e_z ^ e_y)
        assert bar_e_x == (e_x ^ e_y)
    assert consistent_count == 8


def test_unmatched_pure_columns_still_receive_barred_variables():
    # The second e_Z and e_X columns are not used by the e_Y projection.
    checks = scipy.sparse.csc_matrix(
        [
            [1, 0, 0, 0, 1],
            [0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
        ]
    )
    transform = gari_transform(
        checks,
        scipy.sparse.csc_matrix((1, 5)),
        x_detectors=_X_DETECTORS,
        z_detectors=_Z_DETECTORS,
    )

    np.testing.assert_array_equal(transform.u.toarray(), [[1], [0]])
    np.testing.assert_array_equal(transform.v.toarray(), [[1], [0]])
    # All three original variable blocks remain zero in the physical rows.
    assert transform.checks[:4, :5].nnz == 0
    # The unmatched pure variables are copied by their identity constraints.
    assert _column_support(transform.checks, 1) == [5]
    assert _column_support(transform.checks, 6) == [1, 5]
    assert _column_support(transform.checks, 3) == [7]
    assert _column_support(transform.checks, 8) == [3, 7]

    np.testing.assert_allclose(
        tesseract_xor_prior_probabilities(
            transform, np.asarray([0.1, 0.15, 0.2, 0.25, 0.3])
        ),
        [0.1, 0.15, 0.2, 0.25, 0.3, 0.34, 0.15, 0.38, 0.25],
    )

    repeated_y_transform = gari_transform(
        scipy.sparse.csc_matrix(
            [
                [1, 0, 1, 1],
                [0, 1, 1, 1],
                [1, 0, 1, 1],
                [0, 1, 1, 1],
            ]
        ),
        scipy.sparse.csc_matrix((1, 4)),
        x_detectors=_X_DETECTORS,
        z_detectors=_Z_DETECTORS,
    )
    np.testing.assert_allclose(
        tesseract_xor_prior_probabilities(
            repeated_y_transform, np.asarray([0.1, 0.2, 0.3, 0.4])
        ),
        [0.1, 0.2, 0.3, 0.4, 0.468, 0.476],
    )
    np.testing.assert_array_equal(
        tesseract_xor_prior_probabilities(
            _tiny_transform(), np.asarray([0.1, 0.2, 0.5])
        )[-2:],
        [0.5, 0.5],
    )


def _column_support(matrix: scipy.sparse.csc_matrix, column: int) -> list[int]:
    return matrix[:, column].tocoo().row.tolist()


def _assert_rejected(
    checks,
    logicals,
    message,
    *,
    x_detectors=_X_DETECTORS,
    z_detectors=_Z_DETECTORS,
):
    with pytest.raises(ValueError, match=message):
        gari_transform(
            scipy.sparse.csc_matrix(checks),
            scipy.sparse.csc_matrix(logicals),
            x_detectors=x_detectors,
            z_detectors=z_detectors,
        )


def test_rejects_unsupported_projection_structure():
    for x_projection, z_projection, message in [
        ([1, 0], [1, 1], "X-side projection"),
        ([1, 1], [1, 0], "Z-side projection"),
    ]:
        _assert_rejected(
            [
                [1, 0, x_projection[0]],
                [0, 1, z_projection[0]],
                [1, 0, x_projection[1]],
                [0, 1, z_projection[1]],
            ],
            scipy.sparse.csc_matrix((1, 3)),
            message,
        )

    duplicate_cases = [
        (
            "D_X",
            [[1, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1], [0, 0, 1, 1]],
        ),
        (
            "D_Z",
            [[1, 0, 0, 1], [0, 1, 1, 1], [1, 0, 0, 1], [0, 1, 1, 1]],
        ),
    ]
    for side, checks in duplicate_cases:
        _assert_rejected(
            checks,
            scipy.sparse.csc_matrix((1, 4)),
            f"{side} has duplicate columns",
        )


def test_rejects_invalid_inputs(monkeypatch):
    checks, logicals = _tiny_source()
    for x_detectors, z_detectors, message in [
        ([0], [1, 3], "complete partition"),
        ([0, 2], [1, 2, 3], "disjoint"),
        ([0, 0, 2], [1, 3], "more than once"),
        ([0, 2], [1, 4], "outside the detector range"),
    ]:
        _assert_rejected(
            checks,
            logicals,
            message,
            x_detectors=x_detectors,
            z_detectors=z_detectors,
        )

    detectorless_checks = scipy.sparse.hstack(
        [checks, scipy.sparse.csc_matrix((4, 1))], format="csc"
    )
    detectorless_logicals = scipy.sparse.hstack(
        [logicals, scipy.sparse.csc_matrix([[0], [1]])], format="csc"
    )
    _assert_rejected(
        detectorless_checks,
        detectorless_logicals,
        r"column 3.*logical support is \[1\]",
    )
    _assert_rejected(
        checks,
        scipy.sparse.csc_matrix((1, 4)),
        "same source column count",
    )

    nonbinary_checks = checks.astype(float)
    nonbinary_checks.data[0] = 2
    _assert_rejected(nonbinary_checks, logicals, "binary values")

    for dem_text, message in [
        ("error(0.1) D0\ndetector D0", "has no coordinates"),
        ("error(0.1) D0\ndetector(0, 2) D0", "unknown final coordinate"),
    ]:
        with pytest.raises(ValueError, match=message):
            detector_partition_from_last_coordinate(
                stim.DetectorErrorModel(dem_text)
            )

    transform = _tiny_transform()
    with pytest.raises(ValueError, match="one value per decoder column"):
        _matrices_to_decoder_dem(
            transform.checks,
            transform.logicals,
            np.full(transform.checks.shape[1] - 1, 0.1),
        )
    with pytest.raises(ValueError, match="no detector support"):
        _matrices_to_decoder_dem(
            scipy.sparse.csc_matrix((1, 1)),
            scipy.sparse.csc_matrix([[1]]),
            np.asarray([0.1]),
        )

    transform = _tiny_transform()
    for probabilities, message in [
        (np.asarray([0.1, 0.2]), "one value per source column"),
        (np.asarray([[0.1, 0.2, 0.3]]), "one-dimensional"),
        (np.asarray([0.0, 0.2, 0.3]), r"\(0, 0.5\]"),
        (np.asarray([0.1, 0.2, 0.6]), r"\(0, 0.5\]"),
        (np.asarray([0.1, np.nan, 0.3]), "finite"),
    ]:
        with pytest.raises(ValueError, match=message):
            paper_prior_probabilities(transform, probabilities)

    source_probabilities = np.asarray([0.1, 0.2, 0.3])
    invalid_custom_priors = [
        (lambda _transform, _probabilities: np.asarray([0.1]), "one value"),
        (
            lambda _transform, _probabilities: np.full((1, 5), 0.1),
            "one-dimensional",
        ),
        (
            lambda _transform, _probabilities: np.asarray(
                [0.1, 0.1, 0.1, 0.1, np.nan]
            ),
            "non-finite",
        ),
        (
            lambda _transform, _probabilities: np.asarray(
                [0.1, 0.1, 0.1, 0.1, 0.0]
            ),
            r"\(0, 0.5\]",
        ),
        (
            lambda _transform, _probabilities: np.asarray(
                [0.1, 0.1, 0.1, 0.1, 0.6]
            ),
            r"\(0, 0.5\]",
        ),
    ]
    for prior_function, message in invalid_custom_priors:
        with pytest.raises(ValueError, match=message):
            build_gari_decoder_dem(
                transform,
                source_probabilities,
                prior_function=prior_function,
            )
    with pytest.raises(ValueError, match="must be callable"):
        build_gari_decoder_dem(
            transform,
            source_probabilities,
            prior_function=None,
        )

    failed_result = scipy.optimize.OptimizeResult(
        success=False, message="planned solver failure"
    )
    monkeypatch.setattr(
        gari_module.scipy.optimize,
        "linprog",
        lambda *_args, **_kwargs: failed_result,
    )
    with pytest.raises(RuntimeError, match="planned solver failure"):
        tesseract_lp_maximin_prior_probabilities(
            transform, source_probabilities
        )

    infeasible_result = scipy.optimize.OptimizeResult(
        success=True,
        message="claimed success",
        x=np.asarray([0.0, 0.0, 1.0]),
    )
    monkeypatch.setattr(
        gari_module.scipy.optimize,
        "linprog",
        lambda *_args, **_kwargs: infeasible_result,
    )
    with pytest.raises(RuntimeError, match="maximin constraints"):
        tesseract_lp_maximin_prior_probabilities(
            transform, source_probabilities
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
