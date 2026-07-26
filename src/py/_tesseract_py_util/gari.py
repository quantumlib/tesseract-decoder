"""Graph augmentation and rewiring for inference (GARI).

This module implements the matrix construction from A. S. Maan et al.,
"Decoding correlated errors in quantum LDPC codes," Nature Communications 17,
3965 (2026), https://doi.org/10.1038/s41467-026-70556-3.

For source columns ``e_Z``, ``e_X``, and ``e_Y``, the supported CSS check
matrix has the form

::

                      e_Z          e_X          e_Y
                    +------------+------------+------------+
    X syndrome      |    D_X     |     0      |   D_X U    |
                    +------------+------------+------------+
    Z syndrome      |     0      |    D_Z     |   D_Z V    |
                    +------------+------------+------------+

over GF(2). GARI substitutes

``bar(e)_Z = e_Z XOR U e_Y`` and ``bar(e)_X = e_X XOR V e_Y``.

Columns are emitted as ``[e_Z, e_X, e_Y, bar(e)_Z, bar(e)_X]`` and rows as
``[physical X, physical Z, virtual Z, virtual X]``:

::

                           e_Z  e_X  e_Y  bar(e)_Z  bar(e)_X
                         +----+----+----+---------+---------+
    physical X syndrome  |  0 |  0 |  0 |   D_X   |    0    |
    physical Z syndrome  |  0 |  0 |  0 |    0    |   D_Z   |
    virtual Z constraint |  I |  0 |  U |    I    |    0    |
    virtual X constraint |  0 |  I |  V |    0    |    I    |
                         +----+----+----+---------+---------+

The corresponding decoder syndrome is ``[s_X, s_Z, 0, 0]``. The logical map
stays on the original physical variables:
``[L_eZ, L_eX, L_eY, 0, 0]``. These are the GARI transformed matrices. They can
be stored using Stim's DEM syntax, but the resulting GARI DEM is only a matrix
storage and decoding representation. It is not a physical detector error model
and must not be sampled.

For certain single-basis CSS memory experiments, the paper instead evaluates
the logical observable on ``bar(e)_X`` or ``bar(e)_Z``. That placement is
experiment-specific and is not implemented by this generic transform.

Every pure ``e_Z`` and ``e_X`` column receives a barred counterpart, including
columns that are not the projection of any ``e_Y`` column. Such an unused pure
column has an all-zero row in ``U`` or ``V``; its virtual identity constraint
therefore only copies ``e`` to ``bar(e)``. This deliberate redundancy keeps the
five-block structure uniform and the physical top-left blocks zero.
"""

from __future__ import annotations

import dataclasses
import numbers
from collections.abc import Callable, Sequence

import numpy as np
import scipy.optimize
import scipy.sparse
import stim

from _tesseract_py_util.decompose_errors import (
    undecomposed_error_detectors_and_observables,
)


@dataclasses.dataclass(frozen=True)
class GariTransform:
    """Validated GARI transformed matrices and their block metadata."""

    checks: scipy.sparse.csc_matrix
    logicals: scipy.sparse.csc_matrix
    u: scipy.sparse.csc_matrix
    v: scipy.sparse.csc_matrix
    e_z_columns: np.ndarray
    e_x_columns: np.ndarray
    e_y_columns: np.ndarray
    source_to_gari_detectors: np.ndarray
    physical_x_rows: slice
    physical_z_rows: slice
    virtual_z_rows: slice
    virtual_x_rows: slice


def _canonical_binary_csc(
    matrix: scipy.sparse.spmatrix, *, name: str
) -> scipy.sparse.csc_matrix:
    if not scipy.sparse.issparse(matrix):
        raise ValueError(f"{name} must be a sparse matrix.")
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")

    coordinate_matrix = matrix.tocoo(copy=True)
    stored_values = np.asarray(coordinate_matrix.data)
    if not np.all(np.isfinite(stored_values)):
        raise ValueError(f"{name} must contain only finite binary values.")
    is_binary = (stored_values == 0) | (stored_values == 1)
    if not np.all(is_binary):
        bad_value = stored_values[np.flatnonzero(~is_binary)[0]]
        raise ValueError(
            f"{name} must contain only binary values 0 or 1; found "
            f"{bad_value!r}."
        )

    result = scipy.sparse.coo_matrix(
        (
            stored_values.astype(np.int64),
            (coordinate_matrix.row, coordinate_matrix.col),
        ),
        shape=coordinate_matrix.shape,
        dtype=np.int64,
    ).tocsc()
    result.sum_duplicates()
    duplicate_sum_is_binary = (result.data == 0) | (result.data == 1)
    if not np.all(duplicate_sum_is_binary):
        bad_value = result.data[np.flatnonzero(~duplicate_sum_is_binary)[0]]
        raise ValueError(
            f"{name} must be canonical after combining duplicate entries; "
            f"found stored value {bad_value!r}."
        )
    result.eliminate_zeros()
    result.sort_indices()
    return result.astype(np.uint8)


def _validated_detector_indices(
    detectors: Sequence[int], *, name: str, detector_count: int
) -> np.ndarray:
    try:
        values = list(detectors)
    except TypeError as ex:
        raise ValueError(f"{name} must be a one-dimensional sequence.") from ex

    result: list[int] = []
    seen: dict[int, int] = {}
    for position, value in enumerate(values):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, numbers.Integral
        ):
            raise ValueError(
                f"{name}[{position}] must be an integer detector index; "
                f"found {value!r}."
            )
        index = int(value)
        if index < 0 or index >= detector_count:
            raise ValueError(
                f"{name}[{position}] = {index} is outside the detector "
                f"range [0, {detector_count})."
            )
        if index in seen:
            raise ValueError(
                f"{name} contains detector {index} more than once "
                f"(positions {seen[index]} and {position})."
            )
        seen[index] = position
        result.append(index)
    return np.asarray(result, dtype=np.int64)


def _column_support(
    matrix: scipy.sparse.csc_matrix, column: int
) -> tuple[int, ...]:
    start = matrix.indptr[column]
    stop = matrix.indptr[column + 1]
    return tuple(int(v) for v in matrix.indices[start:stop])


def _projection_lookup(
    projections: scipy.sparse.csc_matrix,
    source_columns: np.ndarray,
    *,
    name: str,
) -> dict[tuple[int, ...], tuple[int, int]]:
    lookup: dict[tuple[int, ...], tuple[int, int]] = {}
    for local_column, source_column in enumerate(source_columns):
        support = _column_support(projections, local_column)
        if support in lookup:
            _, previous_source_column = lookup[support]
            raise ValueError(
                f"{name} has duplicate columns from source columns "
                f"{previous_source_column} and {int(source_column)}."
            )
        lookup[support] = (local_column, int(source_column))
    return lookup


def _gf2_product(
    left: scipy.sparse.csc_matrix, right: scipy.sparse.csc_matrix
) -> scipy.sparse.csc_matrix:
    product = (left @ right).tocsc()
    product.sum_duplicates()
    product.data %= 2
    product.eliminate_zeros()
    product.sort_indices()
    return product.astype(np.uint8)


def _sparse_equal(
    left: scipy.sparse.csc_matrix, right: scipy.sparse.csc_matrix
) -> bool:
    return left.shape == right.shape and (left != right).nnz == 0


def _readonly_int_array(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.int64).copy()
    result.setflags(write=False)
    return result


def dem_to_matrices(
    dem: stim.DetectorErrorModel,
) -> tuple[
    scipy.sparse.csc_matrix, scipy.sparse.csc_matrix, np.ndarray
]:
    """Extracts canonical binary matrices and probabilities from a Stim DEM.

    The DEM is flattened before extraction. Detector and observable targets in
    each error are combined by symmetric difference, so repeated targets
    cancel over GF(2). Declared dimensions are retained even when trailing rows
    are unused. The result has one column and probability per flattened error.
    """
    if not isinstance(dem, stim.DetectorErrorModel):
        raise ValueError("dem must be a stim.DetectorErrorModel.")

    flattened = dem.flattened()
    detector_rows: list[int] = []
    detector_columns: list[int] = []
    logical_rows: list[int] = []
    logical_columns: list[int] = []
    probabilities: list[float] = []

    for instruction in flattened:
        if instruction.type != "error":
            continue
        arguments = instruction.args_copy()
        if len(arguments) != 1:
            raise ValueError(
                "Each Stim error instruction must contain exactly one "
                f"probability; found {len(arguments)} in {instruction}."
            )
        probability = float(arguments[0])
        if not np.isfinite(probability) or probability < 0 or probability > 1:
            raise ValueError(
                f"Stim error probability must be finite and in [0, 1]; "
                f"found {probability!r}."
            )

        detectors, observables = undecomposed_error_detectors_and_observables(
            instruction
        )
        source_column = len(probabilities)
        for detector in detectors:
            detector_rows.append(detector)
            detector_columns.append(source_column)
        for observable in observables:
            logical_rows.append(observable)
            logical_columns.append(source_column)
        probabilities.append(probability)

    source_column_count = len(probabilities)
    checks = scipy.sparse.csc_matrix(
        (
            np.ones(len(detector_rows), dtype=np.uint8),
            (detector_rows, detector_columns),
        ),
        shape=(dem.num_detectors, source_column_count),
        dtype=np.uint8,
    )
    logicals = scipy.sparse.csc_matrix(
        (
            np.ones(len(logical_rows), dtype=np.uint8),
            (logical_rows, logical_columns),
        ),
        shape=(dem.num_observables, source_column_count),
        dtype=np.uint8,
    )
    return checks, logicals, np.asarray(probabilities, dtype=np.float64)


def _matrices_to_gari_dem(
    checks: scipy.sparse.csc_matrix,
    logicals: scipy.sparse.csc_matrix,
    probabilities: np.ndarray,
) -> stim.DetectorErrorModel:
    """Stores GARI transformed matrices using Stim's DEM syntax.

    The result is a GARI matrix representation for decoding and interchange.
    It is not a physical detector error model and must not be sampled to
    generate shots.
    """
    gari_checks = _canonical_binary_csc(checks, name="checks")
    gari_logicals = _canonical_binary_csc(logicals, name="logicals")
    if gari_checks.shape[1] != gari_logicals.shape[1]:
        raise ValueError(
            "checks and logicals must have the same GARI column count; "
            f"found {gari_checks.shape[1]} and {gari_logicals.shape[1]}."
        )
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if probability_array.ndim != 1:
        raise ValueError("probabilities must be one-dimensional.")
    if len(probability_array) != gari_checks.shape[1]:
        raise ValueError(
            "probabilities must contain one value per GARI column; "
            f"found {len(probability_array)} for {gari_checks.shape[1]} "
            "columns."
        )
    if not np.all(np.isfinite(probability_array)):
        raise ValueError("probabilities must contain only finite values.")
    if np.any(probability_array < 0) or np.any(probability_array > 1):
        raise ValueError("probabilities must lie in [0, 1].")

    detector_target = stim.target_relative_detector_id
    logical_target = stim.target_logical_observable_id
    gari_dem = stim.DetectorErrorModel()
    for column, probability in enumerate(probability_array):
        detector_targets = [
            detector_target(detector)
            for detector in _column_support(gari_checks, column)
        ]
        if not detector_targets:
            logical_support = list(_column_support(gari_logicals, column))
            raise ValueError(
                f"GARI column {column} has no detector support; logical "
                f"support is {logical_support}."
            )
        targets = detector_targets
        targets.extend(
            logical_target(observable)
            for observable in _column_support(gari_logicals, column)
        )
        gari_dem.append("error", float(probability), targets)

    # One trailing declaration preserves each dimension after serialization.
    if gari_checks.shape[0]:
        gari_dem.append(
            "detector", [], [detector_target(gari_checks.shape[0] - 1)]
        )
    if gari_logicals.shape[0]:
        gari_dem.append(
            "logical_observable",
            [],
            [logical_target(gari_logicals.shape[0] - 1)],
        )
    return gari_dem


def detector_partition_from_fourth_coordinate(
    dem: stim.DetectorErrorModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Partitions detectors using the repository's fourth-coordinate rule.

    This is the color-code-style convention followed by the test-data circuits
    associated with this repository, not a universal Stim convention. The
    fourth coordinate is a finite integer: values at most ``2`` identify X
    detectors, while values at least ``3`` identify Z detectors.
    """
    if not isinstance(dem, stim.DetectorErrorModel):
        raise ValueError("dem must be a stim.DetectorErrorModel.")

    coordinates = dem.get_detector_coordinates()
    x_detectors: list[int] = []
    z_detectors: list[int] = []
    for detector in range(dem.num_detectors):
        detector_coordinates = coordinates.get(detector)
        if detector_coordinates is None or len(detector_coordinates) < 4:
            raise ValueError(
                f"Detector {detector} must have at least four coordinates."
            )
        role = detector_coordinates[3]
        if not np.isfinite(role) or not float(role).is_integer():
            raise ValueError(
                f"Detector {detector} has invalid fourth coordinate "
                f"{role!r}; expected a finite integer."
            )
        if role <= 2:
            x_detectors.append(detector)
        else:
            z_detectors.append(detector)
    return _readonly_int_array(x_detectors), _readonly_int_array(z_detectors)


def gari_transform(
    checks: scipy.sparse.csc_matrix,
    logicals: scipy.sparse.csc_matrix,
    *,
    x_detectors: Sequence[int],
    z_detectors: Sequence[int],
) -> GariTransform:
    """Constructs validated GARI transformed matrices over GF(2).

    ``x_detectors`` and ``z_detectors`` partition the source detector rows.
    Their sequence order determines the order within the physical X and
    physical Z row blocks, respectively.

    Args:
        checks: Binary source detector-by-error matrix.
        logicals: Binary source observable-by-error matrix.
        x_detectors: Source rows containing X-type checks.
        z_detectors: Source rows containing Z-type checks.

    Returns:
        The transformed checks, physical logical map, projection matrices,
        source column classes, detector mapping, and row block slices.

    Raises:
        ValueError: The inputs do not satisfy the supported correlated CSS
        structure.
    """
    source_checks = _canonical_binary_csc(checks, name="checks")
    source_logicals = _canonical_binary_csc(logicals, name="logicals")
    if source_checks.shape[1] != source_logicals.shape[1]:
        raise ValueError(
            "checks and logicals must have the same source column count; "
            f"found {source_checks.shape[1]} and {source_logicals.shape[1]}."
        )

    detector_count = source_checks.shape[0]
    x_rows = _validated_detector_indices(
        x_detectors, name="x_detectors", detector_count=detector_count
    )
    z_rows = _validated_detector_indices(
        z_detectors, name="z_detectors", detector_count=detector_count
    )
    overlap = sorted(set(x_rows.tolist()) & set(z_rows.tolist()))
    if overlap:
        raise ValueError(
            "x_detectors and z_detectors must be disjoint; detectors "
            f"{overlap} appear in both."
        )
    missing = sorted(
        set(range(detector_count))
        - set(x_rows.tolist())
        - set(z_rows.tolist())
    )
    if missing:
        raise ValueError(
            "x_detectors and z_detectors must form a complete partition; "
            f"missing detectors {missing}."
        )

    x_checks = source_checks[x_rows, :].tocsc()
    z_checks = source_checks[z_rows, :].tocsc()
    x_support_counts = np.diff(x_checks.indptr)
    z_support_counts = np.diff(z_checks.indptr)

    e_z_columns = np.flatnonzero(
        (x_support_counts > 0) & (z_support_counts == 0)
    )
    e_x_columns = np.flatnonzero(
        (x_support_counts == 0) & (z_support_counts > 0)
    )
    e_y_columns = np.flatnonzero(
        (x_support_counts > 0) & (z_support_counts > 0)
    )
    detectorless_columns = np.flatnonzero(
        (x_support_counts == 0) & (z_support_counts == 0)
    )
    if detectorless_columns.size:
        source_column = int(detectorless_columns[0])
        logical_support = list(_column_support(source_logicals, source_column))
        raise ValueError(
            f"Source column {source_column} is detectorless; logical support "
            f"is {logical_support}."
        )

    d_x = x_checks[:, e_z_columns].tocsc()
    d_z = z_checks[:, e_x_columns].tocsc()
    d_x_prime = x_checks[:, e_y_columns].tocsc()
    d_z_prime = z_checks[:, e_y_columns].tocsc()
    d_x_lookup = _projection_lookup(d_x, e_z_columns, name="D_X")
    d_z_lookup = _projection_lookup(d_z, e_x_columns, name="D_Z")

    u_rows: list[int] = []
    v_rows: list[int] = []
    for local_y_column, source_column_value in enumerate(e_y_columns):
        source_column = int(source_column_value)
        x_projection = _column_support(d_x_prime, local_y_column)
        if x_projection not in d_x_lookup:
            raise ValueError(
                f"Source column {source_column} has X-side projection "
                f"{list(x_projection)}, which does not equal a D_X column."
            )
        z_projection = _column_support(d_z_prime, local_y_column)
        if z_projection not in d_z_lookup:
            raise ValueError(
                f"Source column {source_column} has Z-side projection "
                f"{list(z_projection)}, which does not equal a D_Z column."
            )
        u_rows.append(d_x_lookup[x_projection][0])
        v_rows.append(d_z_lookup[z_projection][0])

    y_column_count = len(e_y_columns)
    y_indices = np.arange(y_column_count, dtype=np.int64)
    u = scipy.sparse.csc_matrix(
        (
            np.ones(y_column_count, dtype=np.uint8),
            (np.asarray(u_rows, dtype=np.int64), y_indices),
        ),
        shape=(len(e_z_columns), y_column_count),
        dtype=np.uint8,
    )
    v = scipy.sparse.csc_matrix(
        (
            np.ones(y_column_count, dtype=np.uint8),
            (np.asarray(v_rows, dtype=np.int64), y_indices),
        ),
        shape=(len(e_x_columns), y_column_count),
        dtype=np.uint8,
    )
    for factor, name in ((u, "U"), (v, "V")):
        if not np.all(np.diff(factor.indptr) == 1):
            raise ValueError(
                f"Every {name} column must contain exactly one nonzero."
            )
    for base, factor, projection, message in (
        (d_x, u, d_x_prime, "D_X @ U does not equal the e_Y X-side projection."),
        (d_z, v, d_z_prime, "D_Z @ V does not equal the e_Y Z-side projection."),
    ):
        if not _sparse_equal(_gf2_product(base, factor), projection):
            raise ValueError(message)

    x_row_count = len(x_rows)
    z_row_count = len(z_rows)
    e_z_count = len(e_z_columns)
    e_x_count = len(e_x_columns)
    zero = scipy.sparse.csc_matrix

    # Keep a barred variable for every pure column, even when its row in U or V
    # is zero. In that case the identity blocks add the redundant constraint
    # e = bar(e), preserving the same block form for every supported model.
    identity_z = scipy.sparse.identity(e_z_count, dtype=np.uint8, format="csc")
    identity_x = scipy.sparse.identity(e_x_count, dtype=np.uint8, format="csc")
    augmented_checks = scipy.sparse.bmat(
        [
            [None, None, None, d_x, None],
            [None, None, None, None, d_z],
            [identity_z, None, u, identity_z, None],
            [None, identity_x, v, None, identity_x],
        ],
        format="csc",
        dtype=np.uint8,
    )

    augmented_logicals = scipy.sparse.hstack(
        [
            source_logicals[:, e_z_columns],
            source_logicals[:, e_x_columns],
            source_logicals[:, e_y_columns],
            zero((source_logicals.shape[0], e_z_count), dtype=np.uint8),
            zero((source_logicals.shape[0], e_x_count), dtype=np.uint8),
        ],
        format="csc",
    ).astype(np.uint8)

    physical_x_rows = slice(0, x_row_count)
    physical_z_rows = slice(x_row_count, x_row_count + z_row_count)
    virtual_z_rows = slice(
        physical_z_rows.stop, physical_z_rows.stop + e_z_count
    )
    virtual_x_rows = slice(
        virtual_z_rows.stop, virtual_z_rows.stop + e_x_count
    )

    source_to_gari = np.empty(detector_count, dtype=np.int64)
    source_to_gari[x_rows] = np.arange(x_row_count, dtype=np.int64)
    source_to_gari[z_rows] = x_row_count + np.arange(
        z_row_count, dtype=np.int64
    )
    if len(np.unique(source_to_gari)) != detector_count:
        raise ValueError("The source-to-GARI detector mapping is not injective.")
    if np.any(source_to_gari < 0) or np.any(
        source_to_gari >= x_row_count + z_row_count
    ):
        raise ValueError("The source-to-GARI detector mapping is out of range.")

    return GariTransform(
        checks=augmented_checks,
        logicals=augmented_logicals,
        u=u,
        v=v,
        e_z_columns=_readonly_int_array(e_z_columns),
        e_x_columns=_readonly_int_array(e_x_columns),
        e_y_columns=_readonly_int_array(e_y_columns),
        source_to_gari_detectors=_readonly_int_array(source_to_gari),
        physical_x_rows=physical_x_rows,
        physical_z_rows=physical_z_rows,
        virtual_z_rows=virtual_z_rows,
        virtual_x_rows=virtual_x_rows,
    )


def _validated_probabilities(
    values: np.ndarray,
    *,
    expected_count: int,
    name: str,
    column_kind: str,
) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as ex:
        raise ValueError(f"{name} must be a numeric array.") from ex
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if len(result) != expected_count:
        raise ValueError(
            f"{name} must contain one value per {column_kind} column; "
            f"found {len(result)} for {expected_count} columns."
        )
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains a non-finite probability.")
    if np.any(result <= 0) or np.any(result > 0.5):
        raise ValueError(f"{name} must lie in (0, 0.5].")
    result = result.copy()
    result.setflags(write=False)
    return result


def _validated_source_probabilities(
    transform: GariTransform, source_probabilities: np.ndarray
) -> np.ndarray:
    if not isinstance(transform, GariTransform):
        raise ValueError("transform must be a GariTransform.")
    source_column_count = (
        len(transform.e_z_columns)
        + len(transform.e_x_columns)
        + len(transform.e_y_columns)
    )
    return _validated_probabilities(
        source_probabilities,
        expected_count=source_column_count,
        name="source_probabilities",
        column_kind="source",
    )


def _physical_probability_blocks(
    transform: GariTransform, source_probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = _validated_source_probabilities(
        transform, source_probabilities
    )
    return (
        probabilities[transform.e_z_columns],
        probabilities[transform.e_x_columns],
        probabilities[transform.e_y_columns],
    )


def paper_prior_probabilities(
    transform: GariTransform, source_probabilities: np.ndarray
) -> np.ndarray:
    """Returns the published GARI initialization in GARI column order.

    Physical ``e_Z``, ``e_X``, and ``e_Y`` variables retain their source
    probabilities. Every auxiliary variable is assigned probability exactly
    ``0.5``, giving it zero log-likelihood-ratio cost. This is the literature
    reference policy, but those zero-cost branches can produce a very large
    Tesseract search space.
    """
    p_e_z, p_e_x, p_e_y = _physical_probability_blocks(
        transform, source_probabilities
    )
    return np.concatenate(
        [
            p_e_z,
            p_e_x,
            p_e_y,
            np.full(len(p_e_z), 0.5),
            np.full(len(p_e_x), 0.5),
        ]
    )


def _xor_parity_probability(probabilities: np.ndarray) -> float:
    if len(probabilities) == 1:
        return float(probabilities[0])
    if np.any(probabilities == 0.5):
        return 0.5
    log_even_bias = np.sum(np.log1p(-2 * probabilities), dtype=np.float64)
    return float(-0.5 * np.expm1(log_even_bias))


def _auxiliary_xor_probabilities(
    base_probabilities: np.ndarray,
    y_probabilities: np.ndarray,
    projection_matrix: scipy.sparse.csc_matrix,
) -> np.ndarray:
    projection_rows = projection_matrix.tocsr()
    result = np.empty(len(base_probabilities), dtype=np.float64)
    for row, base_probability in enumerate(base_probabilities):
        start = projection_rows.indptr[row]
        stop = projection_rows.indptr[row + 1]
        y_columns = projection_rows.indices[start:stop]
        parity_probabilities = np.concatenate(
            [np.asarray([base_probability]), y_probabilities[y_columns]]
        )
        result[row] = _xor_parity_probability(parity_probabilities)
    return result


def tesseract_xor_prior_probabilities(
    transform: GariTransform, source_probabilities: np.ndarray
) -> np.ndarray:
    """Returns experimental independent-XOR marginals for Tesseract.

    Each auxiliary probability is the independent Bernoulli parity marginal
    implied by ``bar(e)_Z = e_Z XOR U e_Y`` or
    ``bar(e)_X = e_X XOR V e_Y``. The computation uses log-domain products
    for numerical stability and does not clip invalid inputs.

    This is a Tesseract-specific experimental heuristic, not the published
    GARI prior. It can represent evidence already present in the physical
    variables and virtual constraints, and is not claimed to preserve the
    exact source-model maximum-likelihood objective.
    """
    p_e_z, p_e_x, p_e_y = _physical_probability_blocks(
        transform, source_probabilities
    )
    p_bar_e_z = _auxiliary_xor_probabilities(
        p_e_z, p_e_y, transform.u
    )
    p_bar_e_x = _auxiliary_xor_probabilities(
        p_e_x, p_e_y, transform.v
    )
    return np.concatenate([p_e_z, p_e_x, p_e_y, p_bar_e_z, p_bar_e_x])


def _source_to_auxiliary_cost_matrix(
    transform: GariTransform,
) -> scipy.sparse.csc_matrix:
    e_z_count = len(transform.e_z_columns)
    e_x_count = len(transform.e_x_columns)
    identity = scipy.sparse.identity
    return scipy.sparse.bmat(
        [
            [identity(e_z_count, format="csc"), None],
            [None, identity(e_x_count, format="csc")],
            [transform.u.T, transform.v.T],
        ],
        format="csc",
    )


def tesseract_lp_maximin_prior_probabilities(
    transform: GariTransform, source_probabilities: np.ndarray
) -> np.ndarray:
    """Balances nonnegative physical and auxiliary costs for Tesseract.

    For source costs ``c = log((1-p)/p)`` and auxiliary costs ``g``, this
    experimental policy maximizes a common lower bound ``t`` subject to
    ``A g + t <= c`` and ``-g + t <= 0``. The returned physical costs are the
    residuals ``c - A g`` and the remaining costs are ``g``.

    This is a practical Tesseract adaptation of exploratory mode Q, not part of
    the GARI paper. It changes the search objective and is not claimed to
    preserve exact maximum-likelihood decoding. Solver failure is a hard error;
    there is no fallback or clipping.
    """
    p_e_z, p_e_x, p_e_y = _physical_probability_blocks(
        transform, source_probabilities
    )
    physical_probabilities = np.concatenate([p_e_z, p_e_x, p_e_y])
    source_costs = np.log1p(-physical_probabilities) - np.log(
        physical_probabilities
    )
    cost_matrix = _source_to_auxiliary_cost_matrix(transform)
    auxiliary_count = cost_matrix.shape[1]

    constraints = scipy.sparse.bmat(
        [
            [cost_matrix, np.ones((len(source_costs), 1))],
            [
                -scipy.sparse.identity(auxiliary_count, format="csc"),
                np.ones((auxiliary_count, 1)),
            ],
        ],
        format="csc",
    )
    bounds = np.concatenate(
        [source_costs, np.zeros(auxiliary_count, dtype=np.float64)]
    )
    objective = np.zeros(auxiliary_count + 1, dtype=np.float64)
    objective[-1] = -1
    result = scipy.optimize.linprog(
        objective,
        A_ub=constraints,
        b_ub=bounds,
        bounds=[(0, None)] * (auxiliary_count + 1),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(
            "LP maximin prior solver failed: " + str(result.message)
        )
    solution = np.asarray(
        [] if result.x is None else result.x, dtype=np.float64
    )
    if solution.shape != (auxiliary_count + 1,):
        raise RuntimeError(
            "LP maximin prior solver returned an invalid solution."
        )
    if not np.all(np.isfinite(solution)):
        raise RuntimeError(
            "LP maximin prior solver returned non-finite costs."
        )
    auxiliary_costs = solution[:-1]
    residual_costs = source_costs - np.asarray(
        cost_matrix @ auxiliary_costs
    ).reshape(-1)
    if (
        solution[-1] < 0
        or np.any(auxiliary_costs < 0)
        or np.any(residual_costs < 0)
    ):
        raise RuntimeError(
            "LP maximin prior solver returned negative costs."
        )
    minimum_cost = solution[-1]
    if np.any(auxiliary_costs < minimum_cost - 1e-8) or np.any(
        residual_costs < minimum_cost - 1e-8
    ):
        raise RuntimeError(
            "LP maximin prior solver returned a solution that violates the "
            "maximin constraints."
        )
    gari_costs = np.concatenate([residual_costs, auxiliary_costs])
    return np.exp(-np.logaddexp(0, gari_costs))


def build_gari_dem(
    transform: GariTransform,
    source_probabilities: np.ndarray,
    *,
    prior_function: Callable[[GariTransform, np.ndarray], np.ndarray],
) -> stim.DetectorErrorModel:
    """Builds a GARI DEM using an explicit prior policy.

    ``prior_function`` may be one of this module's three built-in policies or
    a user-defined callable. Its output is validated before serialization.
    Stim's DEM syntax is used only to store the GARI transformed matrices. The
    result is not a physical detector error model and must not be sampled.
    """
    probabilities = _validated_source_probabilities(
        transform, source_probabilities
    )
    if not callable(prior_function):
        raise ValueError("prior_function must be callable.")
    gari_probabilities = _validated_probabilities(
        prior_function(transform, probabilities),
        expected_count=transform.checks.shape[1],
        name="prior_function probabilities",
        column_kind="GARI",
    )
    return _matrices_to_gari_dem(
        transform.checks, transform.logicals, gari_probabilities
    )
