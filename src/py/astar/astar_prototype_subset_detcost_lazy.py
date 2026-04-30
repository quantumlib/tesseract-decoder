#!/usr/bin/env python3
"""Prototype A* decoder with lazy subset-LP refinement.

Heuristic modes:
    --opt-subset-detcost-size 0   plain detcost
    --opt-subset-detcost-size 1   lazy optimal singleton LP
    --opt-subset-detcost-size 2   lazy optimal LP over size-1/2 subsets
    --opt-subset-detcost-size 3   lazy optimal LP over size-1/2/3 subsets

For subset size N > 0, the search uses lazy refinement:
    * nodes are first inserted using a cheap lower bound;
    * when popped, the exact subset LP is solved;
    * if the exact LP raises the node key, the node is reinserted;
    * expanded nodes project their exact subset-pattern prices onto children.

The projection step is the main subtlety relative to the singleton case.
The exact parent LP stores prices u_{S,t} for subset/pattern pairs. For a child,
we keep inherited u_{S,t} values on patterns still available, zero out patterns
that have become unavailable, assign zero to newly active subsets, and recompute
child y_S values as the minimum cost of a feasible local signature decomposition
under those inherited prices.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import stim
from scipy import sparse
from scipy.optimize import linprog

INF = math.inf
HEURISTIC_EPS = 1e-9


@dataclass(frozen=True)
class MergedError:
    probability: float
    likelihood_cost: float
    detectors: Tuple[int, ...]
    observables: Tuple[int, ...]


@dataclass
class DecoderData:
    num_detectors: int
    num_observables: int
    errors: List[MergedError]
    detector_to_errors: List[List[int]]
    error_costs: np.ndarray
    error_detectors: List[Tuple[int, ...]]
    error_detector_sets: List[frozenset[int]]
    error_observables: List[Tuple[int, ...]]


@dataclass(frozen=True)
class SubsetLibraryEntry:
    subset_id: int
    detectors: Tuple[int, ...]
    pattern_to_errors: Dict[int, Tuple[int, ...]]
    resolution_combos: Dict[int, Tuple[Tuple[int, ...], ...]]


@dataclass
class ActiveSubsetRecord:
    subset_id: int
    detectors: Tuple[int, ...]
    size: int
    target_mask: int
    available_patterns: Dict[int, Tuple[int, ...]]
    feasible_combos: Tuple[Tuple[int, ...], ...]


@dataclass
class SearchState:
    activated_errors: Tuple[int, ...]
    blocked_errors: np.ndarray
    active_detectors: np.ndarray
    active_detector_counts: np.ndarray
    path_cost: float
    heuristic_cost: float
    heuristic_source: str
    exact_refined: bool
    lp_solution: Optional["SubsetLPSolution"] = None


@dataclass
class DecodeStats:
    num_pq_pushed: int
    num_nodes_popped: int
    max_queue_size: int
    heuristic_calls: int
    plain_heuristic_calls: int
    projection_heuristic_calls: int
    exact_refinement_calls: int
    lp_calls: int
    lp_reinserts: int
    projected_nodes_generated: int
    projected_nodes_refined: int
    projected_nodes_unrefined_at_finish: int
    total_lp_refinement_gain: float
    max_lp_refinement_gain: float
    lp_total_seconds: float
    elapsed_seconds: float
    heuristic_name: str


@dataclass
class DecodeResult:
    activated_errors: Tuple[int, ...]
    path_cost: float
    stats: DecodeStats


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


class LPLogger:
    def __init__(self, path: Path, *, every: int = 1, top_k: int = 10) -> None:
        self.path = path
        self.every = max(1, every)
        self.top_k = max(1, top_k)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def maybe_log(self, *, call_index: int, payload: Dict[str, Any]) -> None:
        if call_index % self.every != 0:
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")


@dataclass
class SubsetLibrary:
    max_subset_size: int
    entries: List[SubsetLibraryEntry]
    subsets_by_detector: List[List[int]]
    num_subsets_by_size: Dict[int, int]


@dataclass
class SubsetLPSolution:
    value: float
    subset_u_values: Dict[int, Dict[int, float]]
    num_active_subsets: int
    num_components: int
    num_variables: int
    num_constraints: int


@dataclass
class SubsetLPSolverStats:
    lp_calls: int = 0
    lp_total_seconds: float = 0.0


class SubsetLPHeuristic:
    def __init__(
        self,
        data: DecoderData,
        subset_library: SubsetLibrary,
        *,
        logger: Optional[LPLogger] = None,
    ) -> None:
        self.data = data
        self.subset_library = subset_library
        self.logger = logger
        self.stats = SubsetLPSolverStats()

    def reset_stats(self) -> None:
        self.stats = SubsetLPSolverStats()

    def _collect_active_subset_records(
        self,
        active_detectors: np.ndarray,
        blocked_errors: np.ndarray,
    ) -> Tuple[Optional[List[ActiveSubsetRecord]], Optional[Dict[int, List[int]]]]:
        active_subset_ids: set[int] = set()
        for detector in np.flatnonzero(active_detectors):
            active_subset_ids.update(self.subset_library.subsets_by_detector[int(detector)])

        subset_records: List[ActiveSubsetRecord] = []
        error_to_subset_positions: Dict[int, List[int]] = defaultdict(list)

        for subset_id in sorted(active_subset_ids):
            entry = self.subset_library.entries[subset_id]
            target_mask = 0
            for bit_index, detector in enumerate(entry.detectors):
                if active_detectors[detector]:
                    target_mask |= 1 << bit_index
            if target_mask == 0:
                continue

            available_patterns: Dict[int, Tuple[int, ...]] = {}
            relevant_errors: set[int] = set()
            for pattern_mask, error_indices in entry.pattern_to_errors.items():
                kept = tuple(error_index for error_index in error_indices if not blocked_errors[error_index])
                if kept:
                    available_patterns[pattern_mask] = kept
                    relevant_errors.update(kept)

            feasible_combos = tuple(
                combo
                for combo in entry.resolution_combos.get(target_mask, ())
                if all(pattern_mask in available_patterns for pattern_mask in combo)
            )
            if not feasible_combos:
                return None, None

            subset_position = len(subset_records)
            subset_records.append(
                ActiveSubsetRecord(
                    subset_id=subset_id,
                    detectors=entry.detectors,
                    size=len(entry.detectors),
                    target_mask=target_mask,
                    available_patterns=available_patterns,
                    feasible_combos=feasible_combos,
                )
            )
            for error_index in sorted(relevant_errors):
                error_to_subset_positions[error_index].append(subset_position)

        return subset_records, error_to_subset_positions

    def solve_exact(
        self,
        active_detectors: np.ndarray,
        blocked_errors: np.ndarray,
    ) -> Tuple[SubsetLPSolution, Dict[str, Any]]:
        t0 = time.perf_counter()
        subset_records, error_to_subset_positions = self._collect_active_subset_records(
            active_detectors=active_detectors,
            blocked_errors=blocked_errors,
        )
        if subset_records is None:
            elapsed = time.perf_counter() - t0
            self.stats.lp_total_seconds += elapsed
            payload = {
                "objective": INF,
                "solve_seconds": elapsed,
                "num_active_subsets": 0,
                "num_components": 0,
                "num_variables": 0,
                "num_constraints": 0,
                "num_active_subsets_by_size": {},
                "contribution_by_subset_size": {},
                "allocated_budget_by_subset_size": {},
                "top_subsets": [],
                "structurally_infeasible": True,
            }
            return (
                SubsetLPSolution(
                    value=INF,
                    subset_u_values={},
                    num_active_subsets=0,
                    num_components=0,
                    num_variables=0,
                    num_constraints=0,
                ),
                payload,
            )

        if not subset_records:
            elapsed = time.perf_counter() - t0
            self.stats.lp_total_seconds += elapsed
            payload = {
                "objective": 0.0,
                "solve_seconds": elapsed,
                "num_active_subsets": 0,
                "num_components": 0,
                "num_variables": 0,
                "num_constraints": 0,
                "num_active_subsets_by_size": {},
                "contribution_by_subset_size": {},
                "allocated_budget_by_subset_size": {},
                "top_subsets": [],
                "structurally_infeasible": False,
            }
            return (
                SubsetLPSolution(
                    value=0.0,
                    subset_u_values={},
                    num_active_subsets=0,
                    num_components=0,
                    num_variables=0,
                    num_constraints=0,
                ),
                payload,
            )

        component_uf = UnionFind(len(subset_records))
        for subset_positions in error_to_subset_positions.values():
            for position in subset_positions[1:]:
                component_uf.union(subset_positions[0], position)
        component_to_subset_positions: Dict[int, List[int]] = defaultdict(list)
        for subset_position in range(len(subset_records)):
            component_to_subset_positions[component_uf.find(subset_position)].append(subset_position)

        total_objective = 0.0
        total_num_variables = 0
        total_num_constraints = 0
        subset_u_values: Dict[int, Dict[int, float]] = {}
        contribution_by_size: Dict[int, float] = defaultdict(float)
        budget_by_size: Dict[int, float] = defaultdict(float)
        active_subset_count_by_size: Dict[int, int] = defaultdict(int)
        top_subset_records: List[Dict[str, Any]] = []
        need_log_details = self.logger is not None

        for component_positions in component_to_subset_positions.values():
            y_var: Dict[int, int] = {}
            u_var: Dict[Tuple[int, int], int] = {}
            error_to_u_vars: Dict[int, List[int]] = defaultdict(list)

            next_var_index = 0
            for subset_position in component_positions:
                y_var[subset_position] = next_var_index
                next_var_index += 1
            for subset_position in component_positions:
                record = subset_records[subset_position]
                active_subset_count_by_size[record.size] += 1
                for pattern_mask, error_indices in sorted(record.available_patterns.items()):
                    variable_index = next_var_index
                    u_var[(subset_position, pattern_mask)] = variable_index
                    next_var_index += 1
                    for error_index in error_indices:
                        error_to_u_vars[error_index].append(variable_index)

            row_indices: List[int] = []
            col_indices: List[int] = []
            values: List[float] = []
            rhs: List[float] = []

            for error_index, variable_indices in sorted(error_to_u_vars.items()):
                row = len(rhs)
                rhs.append(float(self.data.error_costs[error_index]))
                for variable_index in variable_indices:
                    row_indices.append(row)
                    col_indices.append(variable_index)
                    values.append(1.0)

            for subset_position in component_positions:
                record = subset_records[subset_position]
                y_index = y_var[subset_position]
                for combo in record.feasible_combos:
                    row = len(rhs)
                    rhs.append(0.0)
                    row_indices.append(row)
                    col_indices.append(y_index)
                    values.append(1.0)
                    for pattern_mask in combo:
                        row_indices.append(row)
                        col_indices.append(u_var[(subset_position, pattern_mask)])
                        values.append(-1.0)

            total_num_variables += next_var_index
            total_num_constraints += len(rhs)

            a_ub = sparse.csr_matrix(
                (values, (row_indices, col_indices)),
                shape=(len(rhs), next_var_index),
                dtype=np.float64,
            )
            objective = np.zeros(next_var_index, dtype=np.float64)
            for subset_position in component_positions:
                objective[y_var[subset_position]] = -1.0

            self.stats.lp_calls += 1
            result = linprog(
                c=objective,
                A_ub=a_ub,
                b_ub=np.asarray(rhs, dtype=np.float64),
                bounds=[(0.0, None)] * next_var_index,
                method="highs",
            )
            if not result.success:
                raise RuntimeError(
                    f"subset detcost LP solve failed: status={result.status} message={result.message}"
                )
            total_objective += float(-result.fun)
            solution = np.asarray(result.x, dtype=np.float64)

            for subset_position in component_positions:
                record = subset_records[subset_position]
                subset_pattern_values: Dict[int, float] = {}
                total_budget = 0.0
                for pattern_mask in sorted(record.available_patterns):
                    u_value = float(solution[u_var[(subset_position, pattern_mask)]])
                    total_budget += u_value
                    if u_value > 1e-12:
                        subset_pattern_values[pattern_mask] = u_value
                if subset_pattern_values:
                    subset_u_values[record.subset_id] = subset_pattern_values

                if need_log_details:
                    y_value = float(solution[y_var[subset_position]])
                    contribution_by_size[record.size] += y_value
                    budget_by_size[record.size] += total_budget
                    pattern_values = [
                        {
                            "pattern_detectors": [
                                detector
                                for bit_index, detector in enumerate(record.detectors)
                                if pattern_mask & (1 << bit_index)
                            ],
                            "u": float(solution[u_var[(subset_position, pattern_mask)]]),
                            "num_allowed_errors": len(record.available_patterns[pattern_mask]),
                        }
                        for pattern_mask in sorted(record.available_patterns)
                        if solution[u_var[(subset_position, pattern_mask)]] > 1e-12
                    ]
                    top_subset_records.append(
                        {
                            "subset_detectors": list(record.detectors),
                            "subset_size": record.size,
                            "target_active_detectors": [
                                detector
                                for bit_index, detector in enumerate(record.detectors)
                                if record.target_mask & (1 << bit_index)
                            ],
                            "y": y_value,
                            "total_budget": total_budget,
                            "num_available_patterns": len(record.available_patterns),
                            "num_feasible_resolution_combos": len(record.feasible_combos),
                            "patterns": pattern_values,
                        }
                    )

        elapsed = time.perf_counter() - t0
        self.stats.lp_total_seconds += elapsed

        if need_log_details:
            top_subset_records.sort(key=lambda item: (-item["y"], -item["total_budget"], item["subset_detectors"]))
        payload = {
            "objective": total_objective,
            "solve_seconds": elapsed,
            "num_active_subsets": len(subset_records),
            "num_components": len(component_to_subset_positions),
            "num_variables": total_num_variables,
            "num_constraints": total_num_constraints,
            "num_active_subsets_by_size": {
                str(size): active_subset_count_by_size[size] for size in sorted(active_subset_count_by_size)
            },
            "contribution_by_subset_size": (
                {str(size): contribution_by_size[size] for size in sorted(contribution_by_size)}
                if need_log_details
                else {}
            ),
            "allocated_budget_by_subset_size": (
                {str(size): budget_by_size[size] for size in sorted(budget_by_size)}
                if need_log_details
                else {}
            ),
            "top_subsets": top_subset_records[: self.logger.top_k] if self.logger is not None else [],
            "structurally_infeasible": False,
        }
        return (
            SubsetLPSolution(
                value=total_objective,
                subset_u_values=subset_u_values,
                num_active_subsets=len(subset_records),
                num_components=len(component_to_subset_positions),
                num_variables=total_num_variables,
                num_constraints=total_num_constraints,
            ),
            payload,
        )

    def project_from_parent(
        self,
        parent_solution: SubsetLPSolution,
        child_active_detectors: np.ndarray,
        child_blocked_errors: np.ndarray,
    ) -> float:
        total = 0.0
        active_subset_ids: set[int] = set()
        for detector in np.flatnonzero(child_active_detectors):
            active_subset_ids.update(self.subset_library.subsets_by_detector[int(detector)])

        for subset_id in sorted(active_subset_ids):
            entry = self.subset_library.entries[subset_id]
            target_mask = 0
            for bit_index, detector in enumerate(entry.detectors):
                if child_active_detectors[detector]:
                    target_mask |= 1 << bit_index
            if target_mask == 0:
                continue

            combos = entry.resolution_combos.get(target_mask, ())
            if not combos:
                return INF

            parent_u = parent_solution.subset_u_values.get(subset_id, {})
            availability_cache: Dict[int, bool] = {}
            best = INF
            for combo in combos:
                combo_sum = 0.0
                feasible = True
                for pattern_mask in combo:
                    is_available = availability_cache.get(pattern_mask)
                    if is_available is None:
                        is_available = any(
                            not child_blocked_errors[error_index]
                            for error_index in entry.pattern_to_errors.get(pattern_mask, ())
                        )
                        availability_cache[pattern_mask] = is_available
                    if not is_available:
                        feasible = False
                        break
                    combo_sum += parent_u.get(pattern_mask, 0.0)
                if feasible and combo_sum < best:
                    best = combo_sum
            if best == INF:
                return INF
            total += best

        return total


def parse_beam(text: str) -> float:
    lowered = text.strip().lower()
    if lowered in {"inf", "+inf", "infinity", "+infinity"}:
        return INF
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("beam must be non-negative or 'inf'")
    return float(value)


def format_indices(indices: Iterable[int], prefix: str) -> str:
    items = list(indices)
    if not items:
        return "(none)"
    return " ".join(f"{prefix}{i}" for i in items)


def xor_probability(p0: float, p1: float) -> float:
    return p0 * (1 - p1) + (1 - p0) * p1


def iter_dem_errors(dem: stim.DetectorErrorModel) -> Iterable[MergedError]:
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        probability = float(instruction.args_copy()[0])
        if probability <= 0:
            continue
        if probability >= 0.5:
            raise ValueError(
                "This prototype assumes detector-error-model probabilities are in (0, 0.5)."
            )
        detectors: set[int] = set()
        observables: set[int] = set()
        for target in instruction.targets_copy():
            if target.is_separator():
                continue
            if target.is_logical_observable_id():
                if target.val in observables:
                    observables.remove(target.val)
                else:
                    observables.add(target.val)
            else:
                if not target.is_relative_detector_id():
                    raise ValueError(f"Unexpected DEM target: {target!r}")
                if target.val in detectors:
                    detectors.remove(target.val)
                else:
                    detectors.add(target.val)
        yield MergedError(
            probability=probability,
            likelihood_cost=float(-math.log(probability / (1 - probability))),
            detectors=tuple(sorted(detectors)),
            observables=tuple(sorted(observables)),
        )


def merged_errors(dem: stim.DetectorErrorModel) -> List[MergedError]:
    errors_by_symptom: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {}
    for error in iter_dem_errors(dem):
        key = (error.detectors, error.observables)
        previous = errors_by_symptom.get(key)
        if previous is None:
            errors_by_symptom[key] = error.probability
        else:
            errors_by_symptom[key] = xor_probability(previous, error.probability)

    merged: List[MergedError] = []
    for (detectors, observables), probability in errors_by_symptom.items():
        if probability <= 0:
            continue
        if probability >= 0.5:
            raise ValueError(
                "Merged error has probability >= 0.5, which would give a non-positive cost."
            )
        merged.append(
            MergedError(
                probability=probability,
                likelihood_cost=float(-math.log(probability / (1 - probability))),
                detectors=detectors,
                observables=observables,
            )
        )
    return merged


def build_decoder_data(
    dem: stim.DetectorErrorModel,
    *,
    merge_errors_in_dem: bool = True,
) -> DecoderData:
    errors = merged_errors(dem) if merge_errors_in_dem else list(iter_dem_errors(dem))
    detector_to_errors: List[List[int]] = [[] for _ in range(dem.num_detectors)]
    for ei, error in enumerate(errors):
        for d in error.detectors:
            detector_to_errors[d].append(ei)
    return DecoderData(
        num_detectors=dem.num_detectors,
        num_observables=dem.num_observables,
        errors=errors,
        detector_to_errors=detector_to_errors,
        error_costs=np.asarray([e.likelihood_cost for e in errors], dtype=np.float64),
        error_detectors=[e.detectors for e in errors],
        error_detector_sets=[frozenset(e.detectors) for e in errors],
        error_observables=[e.observables for e in errors],
    )


def unpack_bit_packed_rows(bits: np.ndarray, count: int) -> np.ndarray:
    return np.unpackbits(bits, bitorder="little", axis=1, count=count).astype(bool, copy=False)


def initial_detector_counts(data: DecoderData, active_detectors: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(data.errors), dtype=np.int32)
    for d in np.flatnonzero(active_detectors):
        for ei in data.detector_to_errors[int(d)]:
            counts[ei] += 1
    return counts


def apply_error(
    data: DecoderData,
    active_detectors: np.ndarray,
    active_detector_counts: np.ndarray,
    error_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    next_detectors = active_detectors.copy()
    next_counts = active_detector_counts.copy()
    for d in data.error_detectors[error_index]:
        if next_detectors[d]:
            next_detectors[d] = False
            delta = -1
        else:
            next_detectors[d] = True
            delta = 1
        for other_error_index in data.detector_to_errors[d]:
            next_counts[other_error_index] += delta
    return next_detectors, next_counts


def plain_detcost_for_detector(
    data: DecoderData,
    detector: int,
    blocked_errors: np.ndarray,
    active_detector_counts: np.ndarray,
) -> float:
    best = INF
    for ei in data.detector_to_errors[detector]:
        if blocked_errors[ei]:
            continue
        count = int(active_detector_counts[ei])
        assert count > 0
        candidate = float(data.error_costs[ei]) / count
        if candidate < best:
            best = candidate
    return best


def plain_detcost_heuristic(
    data: DecoderData,
    active_detectors: np.ndarray,
    blocked_errors: np.ndarray,
    active_detector_counts: np.ndarray,
) -> float:
    total = 0.0
    for d in np.flatnonzero(active_detectors):
        det_cost = plain_detcost_for_detector(
            data=data,
            detector=int(d),
            blocked_errors=blocked_errors,
            active_detector_counts=active_detector_counts,
        )
        if det_cost == INF:
            return INF
        total += det_cost
    return total


def compute_minimal_resolution_combos(
    available_pattern_masks: Iterable[int],
    subset_size: int,
) -> Dict[int, Tuple[Tuple[int, ...], ...]]:
    patterns = tuple(sorted(set(available_pattern_masks)))
    combos_by_target: Dict[int, List[Tuple[int, ...]]] = {
        target: [] for target in range(1, 1 << subset_size)
    }
    for r in range(1, min(len(patterns), subset_size) + 1):
        for combo in itertools.combinations(patterns, r):
            target_mask = 0
            for pattern_mask in combo:
                target_mask ^= pattern_mask
            if target_mask == 0:
                continue
            combo_set = set(combo)
            existing = combos_by_target[target_mask]
            keep = True
            survivors: List[Tuple[int, ...]] = []
            for old_combo in existing:
                old_set = set(old_combo)
                if combo_set.issuperset(old_set):
                    keep = False
                    survivors.append(old_combo)
                elif old_set.issuperset(combo_set):
                    continue
                else:
                    survivors.append(old_combo)
            if keep:
                survivors.append(combo)
                survivors.sort(key=lambda x: (len(x), x))
                combos_by_target[target_mask] = survivors
    return {
        target_mask: tuple(combos)
        for target_mask, combos in combos_by_target.items()
        if combos
    }


def build_subset_library(data: DecoderData, max_subset_size: int) -> SubsetLibrary:
    library_keys: set[Tuple[int, ...]] = set()
    if max_subset_size >= 1:
        for detector in range(data.num_detectors):
            library_keys.add((detector,))

    for detectors in data.error_detectors:
        limit = min(max_subset_size, len(detectors))
        for subset_size in range(1, limit + 1):
            for subset_detectors in itertools.combinations(detectors, subset_size):
                library_keys.add(tuple(subset_detectors))

    subsets_by_detector: List[List[int]] = [[] for _ in range(data.num_detectors)]
    entries: List[SubsetLibraryEntry] = []
    num_subsets_by_size: Dict[int, int] = defaultdict(int)

    for subset_id, subset_detectors in enumerate(sorted(library_keys, key=lambda t: (len(t), t))):
        pattern_to_errors: Dict[int, List[int]] = defaultdict(list)
        for error_index, detector_set in enumerate(data.error_detector_sets):
            pattern_mask = 0
            for bit_index, detector in enumerate(subset_detectors):
                if detector in detector_set:
                    pattern_mask |= 1 << bit_index
            if pattern_mask != 0:
                pattern_to_errors[pattern_mask].append(error_index)
        frozen_pattern_to_errors = {
            pattern_mask: tuple(error_indices)
            for pattern_mask, error_indices in pattern_to_errors.items()
        }
        entry = SubsetLibraryEntry(
            subset_id=subset_id,
            detectors=subset_detectors,
            pattern_to_errors=frozen_pattern_to_errors,
            resolution_combos=compute_minimal_resolution_combos(
                available_pattern_masks=frozen_pattern_to_errors.keys(),
                subset_size=len(subset_detectors),
            ),
        )
        entries.append(entry)
        num_subsets_by_size[len(subset_detectors)] += 1
        for detector in subset_detectors:
            subsets_by_detector[detector].append(subset_id)

    return SubsetLibrary(
        max_subset_size=max_subset_size,
        entries=entries,
        subsets_by_detector=subsets_by_detector,
        num_subsets_by_size=dict(sorted(num_subsets_by_size.items())),
    )


def detectors_from_solution(data: DecoderData, activated_errors: Sequence[int]) -> np.ndarray:
    detectors = np.zeros(data.num_detectors, dtype=bool)
    for error_index in activated_errors:
        for detector in data.error_detectors[error_index]:
            detectors[detector] ^= True
    return detectors


def observables_from_solution(data: DecoderData, activated_errors: Sequence[int]) -> np.ndarray:
    observables = np.zeros(data.num_observables, dtype=bool)
    for error_index in activated_errors:
        for observable in data.error_observables[error_index]:
            observables[observable] ^= True
    return observables


def decode(
    data: DecoderData,
    detections: np.ndarray,
    *,
    det_beam: float = INF,
    opt_subset_solver: Optional[SubsetLPHeuristic] = None,
    verbose_search: bool = False,
) -> DecodeResult:
    start_time = time.perf_counter()
    if opt_subset_solver is not None:
        opt_subset_solver.reset_stats()

    heuristic_calls = 0
    plain_heuristic_calls = 0
    projection_heuristic_calls = 0
    exact_refinement_calls = 0
    lp_reinserts = 0
    projected_nodes_generated = 0
    projected_nodes_refined = 0
    total_lp_refinement_gain = 0.0
    max_lp_refinement_gain = 0.0

    initial_active_detectors = np.asarray(detections, dtype=bool).copy()
    initial_counts = initial_detector_counts(data, initial_active_detectors)
    initial_blocked = np.zeros(len(data.errors), dtype=bool)
    heuristic_calls += 1
    plain_heuristic_calls += 1
    initial_heuristic = plain_detcost_heuristic(
        data=data,
        active_detectors=initial_active_detectors,
        blocked_errors=initial_blocked,
        active_detector_counts=initial_counts,
    )
    if initial_heuristic == INF:
        raise RuntimeError("Initial residual syndrome is infeasible under the current pruning rule.")

    initial_state = SearchState(
        activated_errors=(),
        blocked_errors=initial_blocked,
        active_detectors=initial_active_detectors,
        active_detector_counts=initial_counts,
        path_cost=0.0,
        heuristic_cost=initial_heuristic,
        heuristic_source="plain",
        exact_refined=(opt_subset_solver is None),
        lp_solution=None,
    )

    priority_queue: List[Tuple[float, int, int, SearchState]] = []
    push_counter = 0
    initial_num_dets = int(initial_active_detectors.sum())
    heapq.heappush(
        priority_queue,
        (initial_state.path_cost + initial_state.heuristic_cost, initial_num_dets, push_counter, initial_state),
    )
    push_counter += 1

    num_pq_pushed = 1
    num_nodes_popped = 0
    max_queue_size = 1
    min_num_dets = initial_num_dets
    max_num_dets = INF if det_beam == INF else min_num_dets + det_beam

    heuristic_name = (
        f"opt_subset_detcost_size_{opt_subset_solver.subset_library.max_subset_size}_lazy_projection"
        if opt_subset_solver is not None
        else "plain_detcost"
    )

    while priority_queue:
        max_queue_size = max(max_queue_size, len(priority_queue))
        f_cost, num_dets, _, state = heapq.heappop(priority_queue)
        num_nodes_popped += 1

        if num_dets > max_num_dets:
            continue

        if num_dets < min_num_dets:
            min_num_dets = num_dets
            max_num_dets = INF if det_beam == INF else min_num_dets + det_beam

        if verbose_search:
            print(
                f"nodes_popped={num_nodes_popped} len(pq)={len(priority_queue)} "
                f"lp_calls={0 if opt_subset_solver is None else opt_subset_solver.stats.lp_calls} "
                f"lp_reinserts={lp_reinserts} proj_generated={projected_nodes_generated} "
                f"proj_refined={projected_nodes_refined} "
                f"proj_unrefined_so_far={projected_nodes_generated - projected_nodes_refined} "
                f"active_dets={num_dets} beam_max={max_num_dets} depth={len(state.activated_errors)} "
                f"f={f_cost:.12g} g={state.path_cost:.12g} h={state.heuristic_cost:.12g} "
                f"h_source={state.heuristic_source} exact_refined={state.exact_refined}"
            )

        if num_dets == 0:
            elapsed_seconds = time.perf_counter() - start_time
            lp_calls = 0 if opt_subset_solver is None else opt_subset_solver.stats.lp_calls
            lp_total_seconds = 0.0 if opt_subset_solver is None else opt_subset_solver.stats.lp_total_seconds
            return DecodeResult(
                activated_errors=state.activated_errors,
                path_cost=state.path_cost,
                stats=DecodeStats(
                    num_pq_pushed=num_pq_pushed,
                    num_nodes_popped=num_nodes_popped,
                    max_queue_size=max_queue_size,
                    heuristic_calls=heuristic_calls,
                    plain_heuristic_calls=plain_heuristic_calls,
                    projection_heuristic_calls=projection_heuristic_calls,
                    exact_refinement_calls=exact_refinement_calls,
                    lp_calls=lp_calls,
                    lp_reinserts=lp_reinserts,
                    projected_nodes_generated=projected_nodes_generated,
                    projected_nodes_refined=projected_nodes_refined,
                    projected_nodes_unrefined_at_finish=projected_nodes_generated - projected_nodes_refined,
                    total_lp_refinement_gain=total_lp_refinement_gain,
                    max_lp_refinement_gain=max_lp_refinement_gain,
                    lp_total_seconds=lp_total_seconds,
                    elapsed_seconds=elapsed_seconds,
                    heuristic_name=heuristic_name,
                ),
            )

        if opt_subset_solver is not None and not state.exact_refined:
            heuristic_calls += 1
            exact_refinement_calls += 1
            previous_h = state.heuristic_cost
            previous_source = state.heuristic_source
            exact_solution, exact_payload = opt_subset_solver.solve_exact(
                active_detectors=state.active_detectors,
                blocked_errors=state.blocked_errors,
            )
            exact_h = exact_solution.value
            reinserted = False
            discarded = False

            if exact_h == INF:
                discarded = True
                if previous_source == "projected":
                    projected_nodes_refined += 1
            else:
                if exact_h + 1e-7 < previous_h:
                    raise AssertionError(
                        f"Exact subset LP lower bound {exact_h} is below stored {previous_source} lower bound {previous_h}."
                    )
                delta = exact_h - previous_h
                total_lp_refinement_gain += delta
                max_lp_refinement_gain = max(max_lp_refinement_gain, delta)
                state.heuristic_cost = exact_h
                state.heuristic_source = "exact"
                state.exact_refined = True
                state.lp_solution = exact_solution
                if previous_source == "projected":
                    projected_nodes_refined += 1
                if delta > HEURISTIC_EPS:
                    reinserted = True
                    lp_reinserts += 1
                    heapq.heappush(
                        priority_queue,
                        (state.path_cost + state.heuristic_cost, num_dets, push_counter, state),
                    )
                    push_counter += 1

            if opt_subset_solver.logger is not None:
                payload = dict(exact_payload)
                payload.update(
                    {
                        "call_index": exact_refinement_calls,
                        "phase": "exact_refinement",
                        "depth": len(state.activated_errors),
                        "nodes_popped": num_nodes_popped,
                        "path_cost": state.path_cost,
                        "active_detector_count": num_dets,
                        "approx_h": previous_h,
                        "exact_h": exact_h,
                        "delta": INF if exact_h == INF else exact_h - previous_h,
                        "heuristic_source_before": previous_source,
                        "reinserted": reinserted,
                        "discarded": discarded,
                    }
                )
                opt_subset_solver.logger.maybe_log(call_index=exact_refinement_calls, payload=payload)

            if verbose_search:
                delta_text = "INF" if exact_h == INF else f"{exact_h - previous_h:.12g}"
                exact_text = "INF" if exact_h == INF else f"{exact_h:.12g}"
                print(
                    f"  lp_refine approx_h={previous_h:.12g} exact_h={exact_text} delta={delta_text} "
                    f"vars={exact_solution.num_variables} constraints={exact_solution.num_constraints} "
                    f"active_subsets={exact_solution.num_active_subsets} comps={exact_solution.num_components} "
                    f"reinserted={reinserted} discarded={discarded}"
                )

            if discarded or reinserted:
                continue

        min_detector = int(np.flatnonzero(state.active_detectors)[0])
        blocked_prefix = state.blocked_errors.copy()
        children_generated = 0
        children_projected = 0
        children_beam_pruned = 0
        children_infeasible = 0

        for error_index in data.detector_to_errors[min_detector]:
            blocked_prefix[error_index] = True
            if state.blocked_errors[error_index]:
                continue

            child_active_detectors, child_active_counts = apply_error(
                data=data,
                active_detectors=state.active_detectors,
                active_detector_counts=state.active_detector_counts,
                error_index=error_index,
            )
            child_num_dets = int(child_active_detectors.sum())
            if child_num_dets > max_num_dets:
                children_beam_pruned += 1
                continue

            child_blocked = blocked_prefix.copy()
            child_path_cost = state.path_cost + float(data.error_costs[error_index])

            if opt_subset_solver is None:
                heuristic_calls += 1
                plain_heuristic_calls += 1
                child_heuristic = plain_detcost_heuristic(
                    data=data,
                    active_detectors=child_active_detectors,
                    blocked_errors=child_blocked,
                    active_detector_counts=child_active_counts,
                )
                child_source = "plain"
                child_exact_refined = True
                child_lp_solution = None
            else:
                if state.lp_solution is None:
                    raise AssertionError("Subset-LP projection requires an exact-refined parent solution.")
                heuristic_calls += 1
                projection_heuristic_calls += 1
                projected_nodes_generated += 1
                children_projected += 1
                child_heuristic = opt_subset_solver.project_from_parent(
                    parent_solution=state.lp_solution,
                    child_active_detectors=child_active_detectors,
                    child_blocked_errors=child_blocked,
                )
                child_source = "projected"
                child_exact_refined = False
                child_lp_solution = None

            if child_heuristic == INF:
                children_infeasible += 1
                continue

            child_state = SearchState(
                activated_errors=state.activated_errors + (error_index,),
                blocked_errors=child_blocked,
                active_detectors=child_active_detectors,
                active_detector_counts=child_active_counts,
                path_cost=child_path_cost,
                heuristic_cost=child_heuristic,
                heuristic_source=child_source,
                exact_refined=child_exact_refined,
                lp_solution=child_lp_solution,
            )
            heapq.heappush(
                priority_queue,
                (child_path_cost + child_heuristic, child_num_dets, push_counter, child_state),
            )
            push_counter += 1
            num_pq_pushed += 1
            children_generated += 1

        if verbose_search:
            print(
                f"  expanded children_generated={children_generated} children_projected={children_projected} "
                f"beam_pruned={children_beam_pruned} infeasible={children_infeasible}"
            )

    raise RuntimeError("Decoding failed to find any completion.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prototype A* decoder for Stim detector error models. "
            "Supports plain detcost and lazy subset-based LP lower bounds."
        )
    )
    parser.add_argument("--circuit", type=Path, required=True, help="Path to a Stim circuit file.")
    parser.add_argument(
        "--shot",
        type=int,
        default=0,
        help="Zero-based sampled shot index to decode.",
    )
    parser.add_argument(
        "--sample-num-shots",
        type=int,
        default=100,
        help="Number of shots to sample before selecting --shot.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=27123839530,
        help="Seed passed to stim.compile_detector_sampler(...).sample(...).",
    )
    parser.add_argument(
        "--det-beam",
        type=parse_beam,
        default=INF,
        help="Beam cutoff on the residual detector count. Use an integer or 'inf'.",
    )
    parser.add_argument(
        "--opt-subset-detcost-size",
        type=int,
        default=0,
        help=(
            "Use the lazy subset-based LP heuristic with library subsets of size at most N. "
            "Use 0 for plain detcost, 1 for the optimal singleton LP, etc."
        ),
    )
    parser.add_argument(
        "--merge-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge indistinguishable DEM errors before decoding (default: enabled).",
    )
    parser.add_argument(
        "--show-shot-detectors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print the sampled shot's active detector IDs before decoding.",
    )
    parser.add_argument(
        "--show-error-indices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print the activated error indices in the final decoding.",
    )
    parser.add_argument(
        "--verbose-search",
        action="store_true",
        help="Print per-node search diagnostics.",
    )
    parser.add_argument(
        "--lp-log-path",
        type=Path,
        default=None,
        help="Optional JSONL file for logging details of each exact subset-LP refinement.",
    )
    parser.add_argument(
        "--lp-log-top-k",
        type=int,
        default=10,
        help="When logging exact LP refinements, include at most this many top subsets.",
    )
    parser.add_argument(
        "--lp-log-every",
        type=int,
        default=1,
        help="When logging exact LP refinements, only write every k-th refinement.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.sample_num_shots <= 0:
        parser.error("--sample-num-shots must be positive.")
    if args.shot < 0:
        parser.error("--shot must be non-negative.")
    if args.opt_subset_detcost_size < 0:
        parser.error("--opt-subset-detcost-size must be non-negative.")
    if args.lp_log_every <= 0:
        parser.error("--lp-log-every must be positive.")
    if args.lp_log_top_k <= 0:
        parser.error("--lp-log-top-k must be positive.")

    circuit = stim.Circuit.from_file(str(args.circuit))
    dem = circuit.detector_error_model(decompose_errors=False)
    data = build_decoder_data(dem, merge_errors_in_dem=args.merge_errors)

    subset_library = None
    subset_solver = None
    if args.opt_subset_detcost_size > 0:
        subset_library = build_subset_library(data, args.opt_subset_detcost_size)
        lp_logger = None
        if args.lp_log_path is not None:
            lp_logger = LPLogger(
                args.lp_log_path,
                every=args.lp_log_every,
                top_k=args.lp_log_top_k,
            )
        subset_solver = SubsetLPHeuristic(data, subset_library, logger=lp_logger)

    dets_packed, obs_packed = circuit.compile_detector_sampler(seed=args.seed).sample(
        shots=args.sample_num_shots,
        separate_observables=True,
        bit_packed=True,
    )
    detections = unpack_bit_packed_rows(dets_packed, count=dem.num_detectors)
    observables = unpack_bit_packed_rows(obs_packed, count=dem.num_observables)

    if args.shot >= detections.shape[0]:
        parser.error(f"--shot={args.shot} is out of range for {detections.shape[0]} sampled shots.")

    shot_detections = detections[args.shot]
    shot_observables = observables[args.shot] if observables.size else np.zeros(0, dtype=bool)

    print(f"circuit = {args.circuit}")
    print(
        "heuristic = "
        + (
            "plain_detcost"
            if subset_solver is None
            else f"opt_subset_detcost_size_{subset_library.max_subset_size}_lazy_projection"
        )
    )
    print(f"shot = {args.shot}")
    print(f"sample_num_shots = {args.sample_num_shots}")
    print(f"num_detectors = {data.num_detectors}")
    print(f"num_observables = {data.num_observables}")
    print(f"num_errors = {len(data.errors)}")
    print(f"beam = {args.det_beam}")
    if subset_library is not None:
        print(f"subset_library_size = {len(subset_library.entries)}")
        print(
            "subset_library_by_size = "
            + ", ".join(
                f"{size}:{count}" for size, count in subset_library.num_subsets_by_size.items()
            )
        )
    if args.show_shot_detectors:
        print(f"shot_detectors = {format_indices(np.flatnonzero(shot_detections), 'D')}")

    result = decode(
        data=data,
        detections=shot_detections,
        det_beam=args.det_beam,
        opt_subset_solver=subset_solver,
        verbose_search=args.verbose_search,
    )

    predicted_observables = observables_from_solution(data, result.activated_errors)
    reproduced_detectors = detectors_from_solution(data, result.activated_errors)
    if not np.array_equal(reproduced_detectors, shot_detections):
        raise AssertionError("Decoded error set does not reproduce the shot's syndrome.")

    print(f"solution_size = {len(result.activated_errors)}")
    print(f"solution_cost = {result.path_cost:.12g}")
    if args.show_error_indices:
        print(f"activated_errors = {format_indices(result.activated_errors, 'E')}")
    print(f"predicted_observables = {format_indices(np.flatnonzero(predicted_observables), 'L')}")
    print(f"sample_observables = {format_indices(np.flatnonzero(shot_observables), 'L')}")
    print(f"observables_match = {bool(np.array_equal(predicted_observables, shot_observables))}")
    print(f"num_pq_pushed = {result.stats.num_pq_pushed}")
    print(f"num_nodes_popped = {result.stats.num_nodes_popped}")
    print(f"max_queue_size = {result.stats.max_queue_size}")
    print(f"heuristic_calls = {result.stats.heuristic_calls}")
    print(f"plain_heuristic_calls = {result.stats.plain_heuristic_calls}")
    print(f"projection_heuristic_calls = {result.stats.projection_heuristic_calls}")
    print(f"exact_refinement_calls = {result.stats.exact_refinement_calls}")
    print(f"lp_calls = {result.stats.lp_calls}")
    print(f"lp_reinserts = {result.stats.lp_reinserts}")
    print(f"projected_nodes_generated = {result.stats.projected_nodes_generated}")
    print(f"projected_nodes_refined = {result.stats.projected_nodes_refined}")
    print(f"projected_nodes_unrefined_at_finish = {result.stats.projected_nodes_unrefined_at_finish}")
    print(f"total_lp_refinement_gain = {result.stats.total_lp_refinement_gain:.12g}")
    print(f"max_lp_refinement_gain = {result.stats.max_lp_refinement_gain:.12g}")
    print(f"lp_total_seconds = {result.stats.lp_total_seconds:.6f}")
    print(f"elapsed_seconds = {result.stats.elapsed_seconds:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
