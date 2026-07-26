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
``[L_eZ, L_eX, L_eY, 0, 0]``. The augmented system is a decoder model with
structural variables; it is not a physical noise model to sample directly.

Every pure ``e_Z`` and ``e_X`` column receives a barred counterpart, including
columns that are not the projection of any ``e_Y`` column. Such an unmatched
column has an all-zero row in ``U`` or ``V``; its virtual identity constraint
therefore only copies ``e`` to ``bar(e)``. This deliberate redundancy keeps the
five-block structure uniform and the physical top-left blocks zero.
"""

from __future__ import annotations

import dataclasses
import numbers
from collections.abc import Sequence

import numpy as np
import scipy.sparse


@dataclasses.dataclass(frozen=True)
class GariTransform:
    """A validated GARI augmented check system."""

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
    if stored_values.size and not np.all(np.isfinite(stored_values)):
        raise ValueError(f"{name} must contain only finite binary values.")
    is_binary = (stored_values == 0) | (stored_values == 1)
    if stored_values.size and not np.all(is_binary):
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
    if result.data.size and not np.all(duplicate_sum_is_binary):
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


def gari_transform(
    checks: scipy.sparse.csc_matrix,
    logicals: scipy.sparse.csc_matrix,
    *,
    x_detectors: Sequence[int],
    z_detectors: Sequence[int],
) -> GariTransform:
    """Constructs the validated GARI augmented system over GF(2).

    ``x_detectors`` and ``z_detectors`` partition the source detector rows.
    Their sequence order determines the order within the physical X and
    physical Z row blocks, respectively.

    Args:
        checks: Binary source detector-by-error matrix.
        logicals: Binary source observable-by-error matrix.
        x_detectors: Source rows containing X-type checks.
        z_detectors: Source rows containing Z-type checks.

    Returns:
        The augmented checks, physical logical map, matching matrices, source
        column classes, detector mapping, and row block slices.

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
                f"{list(x_projection)}, which does not match a D_X column."
            )
        z_projection = _column_support(d_z_prime, local_y_column)
        if z_projection not in d_z_lookup:
            raise ValueError(
                f"Source column {source_column} has Z-side projection "
                f"{list(z_projection)}, which does not match a D_Z column."
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
    if not np.all(np.diff(u.indptr) == 1):
        raise ValueError("Every U column must contain exactly one nonzero.")
    if not np.all(np.diff(v.indptr) == 1):
        raise ValueError("Every V column must contain exactly one nonzero.")
    if not _sparse_equal(_gf2_product(d_x, u), d_x_prime):
        raise ValueError("D_X @ U does not equal the e_Y X-side projection.")
    if not _sparse_equal(_gf2_product(d_z, v), d_z_prime):
        raise ValueError("D_Z @ V does not equal the e_Y Z-side projection.")

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
            [
                zero((x_row_count, e_z_count), dtype=np.uint8),
                zero((x_row_count, e_x_count), dtype=np.uint8),
                zero((x_row_count, y_column_count), dtype=np.uint8),
                d_x,
                zero((x_row_count, e_x_count), dtype=np.uint8),
            ],
            [
                zero((z_row_count, e_z_count), dtype=np.uint8),
                zero((z_row_count, e_x_count), dtype=np.uint8),
                zero((z_row_count, y_column_count), dtype=np.uint8),
                zero((z_row_count, e_z_count), dtype=np.uint8),
                d_z,
            ],
            [
                identity_z,
                zero((e_z_count, e_x_count), dtype=np.uint8),
                u,
                identity_z,
                zero((e_z_count, e_x_count), dtype=np.uint8),
            ],
            [
                zero((e_x_count, e_z_count), dtype=np.uint8),
                identity_x,
                v,
                zero((e_x_count, e_z_count), dtype=np.uint8),
                identity_x,
            ],
        ],
        format="csc",
    ).astype(np.uint8)

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
