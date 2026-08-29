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

GARI source DEMs must be undecomposed. Circuit conversion generates them with
``decompose_errors=False`` and ``flatten_loops=True``. Matrix extraction also
flattens its input before treating each Stim ``error`` instruction as one
source matrix column. Instructions containing Stim's ``^`` decomposition
separator are not supported. Repeated detector or logical targets are reduced
modulo two, following Stim's GF(2) parity semantics.

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
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import scipy.optimize
import scipy.sparse
import stim

if __package__:
    from .detector_basis import (
        DetectorBasisClassifier,
        automatic_detector_basis_classifier,
        chromobius_detector_basis_classifier,
        classify_detector_bases,
    )
else:
    from detector_basis import (
        DetectorBasisClassifier,
        automatic_detector_basis_classifier,
        chromobius_detector_basis_classifier,
        classify_detector_bases,
    )


@dataclasses.dataclass
class GariTransform:
    """GARI matrices and their source-column and detector mappings."""

    checks: scipy.sparse.csc_matrix
    logicals: scipy.sparse.csc_matrix
    u: scipy.sparse.csc_matrix
    v: scipy.sparse.csc_matrix
    e_z_columns: np.ndarray
    e_x_columns: np.ndarray
    e_y_columns: np.ndarray
    source_to_gari_detectors: np.ndarray


def _circuit_to_gari_source_dem(
    circuit: stim.Circuit,
) -> stim.DetectorErrorModel:
    """Creates the flattened, undecomposed source DEM required by GARI."""
    return circuit.detector_error_model(
        decompose_errors=False,
        flatten_loops=True,
    ).flattened()


def _nonzero_column_rows(
    matrix: scipy.sparse.csc_matrix, column: int
) -> tuple[int, ...]:
    start = matrix.indptr[column]
    stop = matrix.indptr[column + 1]
    return tuple(int(v) for v in matrix.indices[start:stop])


def _projection_matrix(
    pure_columns: scipy.sparse.csc_matrix,
    mixed_projections: scipy.sparse.csc_matrix,
    mixed_source_columns: np.ndarray,
    *,
    matrix_name: str,
    pure_name: str,
) -> scipy.sparse.csc_matrix:
    lookup: dict[tuple[int, ...], int] = {}
    for local_column in range(pure_columns.shape[1]):
        support = _nonzero_column_rows(pure_columns, local_column)
        if support in lookup:
            raise ValueError(f"{pure_name} has duplicate columns.")
        lookup[support] = local_column

    rows: list[int] = []
    for local_column, source_column in enumerate(mixed_source_columns):
        support = _nonzero_column_rows(mixed_projections, local_column)
        if support not in lookup:
            raise ValueError(
                f"{matrix_name} mixed source column {int(source_column)} has "
                f"no corresponding {pure_name} pure column."
            )
        rows.append(lookup[support])

    column_count = len(mixed_source_columns)
    return scipy.sparse.csc_matrix(
        (
            np.ones(column_count, dtype=np.uint8),
            (
                np.asarray(rows, dtype=np.int64),
                np.arange(column_count, dtype=np.int64),
            ),
        ),
        shape=(pure_columns.shape[1], column_count),
        dtype=np.uint8,
    )


def dem_to_matrices(
    dem: stim.DetectorErrorModel,
) -> tuple[
    scipy.sparse.csc_matrix, scipy.sparse.csc_matrix, np.ndarray
]:
    """Extracts matrices from an undecomposed source DEM.

    Repeat blocks and detector shifts are flattened first. Each resulting Stim
    ``error`` instruction becomes exactly one source matrix column. The input
    should be generated with ``decompose_errors=False``. This function does
    not merge duplicate instructions or reconstruct correlations split across
    instructions. A Stim ``^`` decomposition separator is rejected. Repeated
    detector or logical targets within an instruction are reduced modulo two.
    Resulting no-op errors are omitted, and logical-only errors are rejected.
    """
    dem = dem.flattened()
    detector_rows: list[int] = []
    detector_columns: list[int] = []
    logical_rows: list[int] = []
    logical_columns: list[int] = []
    probabilities: list[float] = []

    for instruction in dem:
        if instruction.type != "error":
            continue
        targets = instruction.targets_copy()
        if any(target.is_separator() for target in targets):
            raise ValueError(
                "GARI requires a DEM generated with decompose_errors=False."
            )
        detectors: set[int] = set()
        logicals: set[int] = set()
        for target in targets:
            if target.is_relative_detector_id():
                detectors ^= {target.val}
            elif target.is_logical_observable_id():
                logicals ^= {target.val}
        if not detectors and not logicals:
            continue
        if not detectors:
            raise ValueError("GARI does not support logical-only source errors.")
        column = len(probabilities)
        detector_rows.extend(sorted(detectors))
        detector_columns.extend([column] * len(detectors))
        logical_rows.extend(sorted(logicals))
        logical_columns.extend([column] * len(logicals))
        probabilities.append(float(instruction.args_copy()[0]))

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
    """Stores GARI transformed matrices using Stim's DEM syntax."""
    detector_target = stim.target_relative_detector_id
    logical_target = stim.target_logical_observable_id
    gari_dem = stim.DetectorErrorModel()
    for column, probability in enumerate(probabilities):
        targets = [
            detector_target(detector)
            for detector in _nonzero_column_rows(checks, column)
        ]
        targets.extend(
            logical_target(observable)
            for observable in _nonzero_column_rows(logicals, column)
        )
        gari_dem.append("error", float(probability), targets)

    # Declare only dimensions not already implied by the error targets.
    if gari_dem.num_detectors < checks.shape[0]:
        gari_dem.append(
            "detector", [], [detector_target(checks.shape[0] - 1)]
        )
    if gari_dem.num_observables < logicals.shape[0]:
        gari_dem.append(
            "logical_observable",
            [],
            [logical_target(logicals.shape[0] - 1)],
        )
    return gari_dem


def _detector_partition_from_fourth_coordinate(
    dem: stim.DetectorErrorModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper for the Chromobius fourth-coordinate rule.

    This is the color-code-style convention followed by the test-data circuits
    associated with this repository, not a universal Stim convention. The
    fourth-coordinate values ``0``, ``1``, or ``2`` identify X detectors,
    while values ``3``, ``4``, or ``5`` identify Z detectors.
    """
    return _detector_partition(
        dem, detector_basis_classifier=chromobius_detector_basis_classifier
    )


def _detector_partition(
    dem: stim.DetectorErrorModel,
    *,
    detector_basis_classifier: DetectorBasisClassifier,
) -> tuple[np.ndarray, np.ndarray]:
    detector_bases = np.asarray(
        classify_detector_bases(
            dem, detector_basis_classifier=detector_basis_classifier
        )
    )
    return np.flatnonzero(detector_bases == "X"), np.flatnonzero(
        detector_bases == "Z"
    )


def _gari_transform(
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
        source column classes, and detector mapping.
    """
    source_checks = checks.tocsc()
    source_logicals = logicals.tocsc()
    if source_checks.shape[1] != source_logicals.shape[1]:
        raise ValueError(
            "checks and logicals must have the same source column count; "
            f"found {source_checks.shape[1]} and {source_logicals.shape[1]}."
        )

    detector_count = source_checks.shape[0]
    x_rows = np.asarray(x_detectors, dtype=np.int64)
    z_rows = np.asarray(z_detectors, dtype=np.int64)
    partition = np.concatenate([x_rows, z_rows])
    if not np.array_equal(np.sort(partition), np.arange(detector_count)):
        raise ValueError(
            "x_detectors and z_detectors must partition all detector rows."
        )

    x_checks = source_checks[x_rows, :]
    z_checks = source_checks[z_rows, :]
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
        raise ValueError(
            f"Source column {int(detectorless_columns[0])} is detectorless."
        )

    d_x = x_checks[:, e_z_columns]
    d_z = z_checks[:, e_x_columns]
    d_x_prime = x_checks[:, e_y_columns]
    d_z_prime = z_checks[:, e_y_columns]
    u = _projection_matrix(
        d_x,
        d_x_prime,
        e_y_columns,
        matrix_name="U",
        pure_name="D_X",
    )
    v = _projection_matrix(
        d_z,
        d_z_prime,
        e_y_columns,
        matrix_name="V",
        pure_name="D_Z",
    )
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

    source_to_gari = np.empty(detector_count, dtype=np.int64)
    source_to_gari[partition] = np.arange(detector_count, dtype=np.int64)
    return GariTransform(
        checks=augmented_checks,
        logicals=augmented_logicals,
        u=u,
        v=v,
        e_z_columns=e_z_columns,
        e_x_columns=e_x_columns,
        e_y_columns=e_y_columns,
        source_to_gari_detectors=source_to_gari,
    )


def _physical_probability_blocks(
    transform: GariTransform, source_probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.asarray(source_probabilities, dtype=np.float64)
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
    ``0.5``. In Tesseract this gives the auxiliary variable zero search cost,
    which can produce a very large search space.
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


def _barred_xor_probabilities(
    base_probabilities: np.ndarray,
    y_probabilities: np.ndarray,
    projection_matrix: scipy.sparse.csc_matrix,
) -> np.ndarray:
    """Returns marginals of ``base XOR projection_matrix @ e_Y``."""
    with np.errstate(divide="ignore"):
        log_even_bias = np.log1p(-2 * base_probabilities) + (
            projection_matrix @ np.log1p(-2 * y_probabilities)
        )
    return -0.5 * np.expm1(log_even_bias)


def tesseract_xor_prior_probabilities(
    transform: GariTransform, source_probabilities: np.ndarray
) -> np.ndarray:
    """Returns experimental independent-XOR marginals for Tesseract.

    Each auxiliary probability is the independent Bernoulli parity marginal
    implied by ``bar(e)_Z = e_Z XOR U e_Y`` or
    ``bar(e)_X = e_X XOR V e_Y``. The computation uses log-domain products
    for numerical stability and does not clip invalid inputs.

    This is a Tesseract-specific experimental heuristic, not the published
    GARI prior. It only defines auxiliary search weights and makes no claim
    about decoding optimality.
    """
    p_e_z, p_e_x, p_e_y = _physical_probability_blocks(
        transform, source_probabilities
    )
    p_bar_e_z = _barred_xor_probabilities(
        p_e_z, p_e_y, transform.u
    )
    p_bar_e_x = _barred_xor_probabilities(
        p_e_x, p_e_y, transform.v
    )
    return np.concatenate([p_e_z, p_e_x, p_e_y, p_bar_e_z, p_bar_e_x])


def tesseract_lp_max_barred_cost_prior_probabilities(
    transform: GariTransform, source_probabilities: np.ndarray
) -> np.ndarray:
    """Balances physical and barred-variable search costs with two LPs.

    The source costs ``c = log((1-p)/p)`` are ordered as ``[e_Z, e_X, e_Y]``.
    The auxiliary costs ``g`` are ordered as ``[bar(e)_Z, bar(e)_X]``. The
    incidence matrix is ``A = [[I, 0], [0, I], [U.T, V.T]]``, so the residual
    physical costs are ``r = c - A g``.

    Maximizing only ``sum(g)`` can put most of the cost on a few variables and
    leave many physical or auxiliary costs at zero. A zero cost becomes
    probability ``0.5`` and gives Tesseract no search preference. The first LP
    instead maximizes a common floor for every entry of ``r`` and ``g``. The
    second LP keeps that floor and then maximizes ``sum(g)``. This gives a more
    balanced set of search costs while still favoring the barred variables.

    The result is ``[r, g]`` converted back to probabilities in GARI column
    order. This experimental policy is not part of the GARI paper. It only
    defines search costs and makes no claim about decoding optimality. Solver
    failure is a hard error; there is no fallback.
    """
    p_e_z, p_e_x, p_e_y = _physical_probability_blocks(
        transform, source_probabilities
    )
    physical_probabilities = np.concatenate([p_e_z, p_e_x, p_e_y])
    source_costs = np.log1p(-physical_probabilities) - np.log(
        physical_probabilities
    )
    # A maps [bar(e)_Z, bar(e)_X] costs into [e_Z, e_X, e_Y] costs.
    identity = scipy.sparse.identity
    cost_matrix = scipy.sparse.bmat(
        [
            [identity(len(p_e_z), format="csc"), None],
            [None, identity(len(p_e_x), format="csc")],
            [transform.u.T, transform.v.T],
        ],
        format="csc",
    )
    auxiliary_count = cost_matrix.shape[1]
    if auxiliary_count == 0:
        return physical_probabilities

    # First maximize t subject to every physical and auxiliary cost being at
    # least t: c - A g >= t and g >= t.
    floor_constraints = scipy.sparse.bmat(
        [
            [cost_matrix, np.ones((len(source_costs), 1))],
            [
                -identity(auxiliary_count, format="csc"),
                np.ones((auxiliary_count, 1)),
            ],
        ],
        format="csc",
    )
    floor_objective = np.zeros(auxiliary_count + 1, dtype=np.float64)
    floor_objective[-1] = -1
    floor_result = scipy.optimize.linprog(
        floor_objective,
        A_ub=floor_constraints,
        b_ub=np.concatenate([source_costs, np.zeros(auxiliary_count)]),
        bounds=(0, None),
        method="highs",
    )
    if not floor_result.success:
        raise RuntimeError(
            "LP prior floor solver failed: " + str(floor_result.message)
        )
    tolerance = 1e-7 * max(
        1.0, float(np.max(source_costs, initial=0.0))
    )
    cost_floor = max(0.0, float(floor_result.x[-1]) - tolerance)

    # Then maximize the total barred cost without lowering the common floor.
    objective = -np.ones(auxiliary_count, dtype=np.float64)
    result = scipy.optimize.linprog(
        objective,
        A_ub=cost_matrix,
        b_ub=source_costs - cost_floor,
        bounds=(cost_floor, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(
            "LP prior barred-cost solver failed: " + str(result.message)
        )
    auxiliary_costs = np.asarray(result.x)
    if np.min(auxiliary_costs) < cost_floor - tolerance:
        raise RuntimeError(
            "LP max-barred-cost solver returned an infeasible solution."
        )
    auxiliary_costs = np.maximum(auxiliary_costs, 0.0)
    residual_costs = source_costs - np.asarray(
        cost_matrix @ auxiliary_costs
    ).reshape(-1)
    if np.min(residual_costs) < cost_floor - tolerance:
        raise RuntimeError(
            "LP max-barred-cost solver returned an infeasible solution."
        )
    # Normalize only active-constraint noise accepted by the LP solver.
    residual_costs = np.maximum(residual_costs, 0.0)
    gari_costs = np.concatenate([residual_costs, auxiliary_costs])
    return np.exp(-np.logaddexp(0, gari_costs))


def _build_gari_dem(
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
    def validated_probabilities(
        values: np.ndarray, expected_count: int, name: str
    ) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (expected_count,) or not np.all(
            (result > 0) & (result <= 0.5)
        ):
            raise ValueError(
                f"{name} must contain {expected_count} finite values in "
                "(0, 0.5]."
            )
        return result

    source_count = (
        len(transform.e_z_columns)
        + len(transform.e_x_columns)
        + len(transform.e_y_columns)
    )
    probabilities = validated_probabilities(
        source_probabilities,
        source_count,
        "source_probabilities",
    )
    gari_probabilities = validated_probabilities(
        prior_function(transform, probabilities),
        transform.checks.shape[1],
        "prior_function probabilities",
    )
    return _matrices_to_gari_dem(
        transform.checks, transform.logicals, gari_probabilities
    )


def circuit_to_gari(
    circuit: stim.Circuit,
    *,
    prior_function: Callable[[GariTransform, np.ndarray], np.ndarray],
    detector_basis_classifier: DetectorBasisClassifier = (
        automatic_detector_basis_classifier
    ),
) -> tuple[stim.DetectorErrorModel, dict[str, object]]:
    """Converts a supported CSS circuit into a GARI matrix DEM and v1 layout.

    The source DEM is generated undecomposed (``decompose_errors=False``) and
    flattened. By default, detector basis metadata is checked before the
    repository's Chromobius fourth-coordinate convention. The returned DEM
    stores transformed matrices for decoding and must not be sampled.
    """
    source_dem = _circuit_to_gari_source_dem(circuit)
    checks, logicals, probabilities = dem_to_matrices(source_dem)
    x_detectors, z_detectors = _detector_partition(
        source_dem, detector_basis_classifier=detector_basis_classifier
    )
    transform = _gari_transform(
        checks,
        logicals,
        x_detectors=x_detectors,
        z_detectors=z_detectors,
    )
    gari_dem = _build_gari_dem(
        transform, probabilities, prior_function=prior_function
    )
    layout = {
        "schema": "tesseract.gari_layout.v1",
        "source_detector_count": len(transform.source_to_gari_detectors),
        "gari_detector_count": transform.checks.shape[0],
        "source_to_gari": transform.source_to_gari_detectors.tolist(),
        "detector_order": "physical_then_virtual",
    }
    return gari_dem, layout


def call_gari(circuit_fname: str, prior_name: str, output_dir: str) -> None:
    """Converts one circuit and writes its GARI DEM and layout files."""
    prior_function = {
        "paper": paper_prior_probabilities,
        "xor": tesseract_xor_prior_probabilities,
        "lp-max-barred-cost": tesseract_lp_max_barred_cost_prior_probabilities,
    }[prior_name]
    gari_dem, layout = circuit_to_gari(
        stim.Circuit.from_file(circuit_fname),
        prior_function=prior_function,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_name = f"{Path(circuit_fname).stem}_gari_{prior_name.replace('-', '_')}"
    gari_dem.to_file(output_path / f"{output_name}.dem")
    (output_path / f"{output_name}_layout.json").write_text(
        json.dumps(layout, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Convert one Stim circuit into a GARI matrix DEM and "
            "detector-layout JSON file."
        )
    )
    parser.add_argument(
        "--circuit", required=True, help="Input Stim circuit file."
    )
    parser.add_argument(
        "--prior",
        choices=("paper", "xor", "lp-max-barred-cost"),
        required=True,
        help="Prior policy used for the GARI matrix probabilities.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help=(
            "Output directory, created if needed. Files are named "
            "<circuit>_gari_<prior>.dem and "
            "<circuit>_gari_<prior>_layout.json."
        ),
    )
    args = parser.parse_args()
    call_gari(args.circuit, args.prior, args.out_dir)


if __name__ == "__main__":
    main()
