#!/usr/bin/env python3
"""Prototype A* decoder with detcost and subset-LP heuristics.

This script keeps the basic search structure of the original prototype while
adding a small CLI and a family of stronger admissible heuristics.

Heuristic modes:
    --opt-subset-detcost-size 0   plain detcost
    --opt-subset-detcost-size 1   optimal singleton LP
    --opt-subset-detcost-size 2   optimal LP over singletons and 2-detector subsets
    --opt-subset-detcost-size 3   optimal LP over singletons and 2/3-detector subsets

The subset library is the small-subset closure of DEM supports:
    * every singleton detector subset, and
    * every nonempty subset of D(e) of size at most N, for each DEM error e.

For a library subset S, the local decoder only sees the restriction of errors to
S. Because N is intended to be small (<= 3 in practice), all minimal local
pattern resolutions can be precomputed once.
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


@dataclass
class DecodeStats:
    num_pq_pushed: int
    num_nodes_popped: int
    max_queue_size: int
    heuristic_calls: int
    lp_calls: int
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
        # Truncate eagerly so repeated runs do not append by accident.
        self.path.write_text("")

    def maybe_log(self, *, call_index: int, payload: Dict[str, Any]) -> None:
        if call_index % self.every != 0:
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")


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
    """Precompute inclusion-minimal local pattern combinations for each target.

    For a fixed subset S of size k, an error only matters through its nonzero local
    pattern D(e)∩S, represented as a bit-mask in {1, ..., 2^k-1}. Because local
    budgets are nonnegative, an optimal local resolution never needs to use the same
    local pattern twice, and any combo that strictly contains another combo with the
    same XOR target is dominated.
    """

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


@dataclass
class SubsetLibrary:
    max_subset_size: int
    entries: List[SubsetLibraryEntry]
    subsets_by_detector: List[List[int]]
    num_subsets_by_size: Dict[int, int]


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


@dataclass
class SubsetLPHeuristicStats:
    call_count: int = 0
    lp_call_count: int = 0
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
        self.stats = SubsetLPHeuristicStats()

    def evaluate(
        self,
        active_detectors: np.ndarray,
        blocked_errors: np.ndarray,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        self.stats.call_count += 1
        self.stats.lp_call_count += 1
        t0 = time.perf_counter()

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
                self.stats.lp_total_seconds += time.perf_counter() - t0
                return INF

            record = ActiveSubsetRecord(
                subset_id=subset_id,
                detectors=entry.detectors,
                size=len(entry.detectors),
                target_mask=target_mask,
                available_patterns=available_patterns,
                feasible_combos=feasible_combos,
            )
            subset_position = len(subset_records)
            subset_records.append(record)
            for error_index in sorted(relevant_errors):
                error_to_subset_positions[error_index].append(subset_position)

        if not subset_records:
            elapsed = time.perf_counter() - t0
            self.stats.lp_total_seconds += elapsed
            if self.logger is not None:
                payload: Dict[str, Any] = {
                    "call_index": self.stats.call_count,
                    "objective": 0.0,
                    "solve_seconds": elapsed,
                    "num_active_subsets": 0,
                    "num_components": 0,
                }
                if context is not None:
                    payload.update(context)
                self.logger.maybe_log(call_index=self.stats.call_count, payload=payload)
            return 0.0

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
        contribution_by_size: Dict[int, float] = defaultdict(float)
        budget_by_size: Dict[int, float] = defaultdict(float)
        active_subset_count_by_size: Dict[int, int] = defaultdict(int)
        top_subset_records: List[Dict[str, Any]] = []

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
                y_value = float(solution[y_var[subset_position]])
                total_budget = float(
                    sum(solution[u_var[(subset_position, pattern_mask)]] for pattern_mask in record.available_patterns)
                )
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

        if self.logger is not None:
            top_subset_records.sort(key=lambda item: (-item["y"], -item["total_budget"], item["subset_detectors"]))
            payload = {
                "call_index": self.stats.call_count,
                "objective": total_objective,
                "solve_seconds": elapsed,
                "num_active_subsets": len(subset_records),
                "num_active_subsets_by_size": {
                    str(size): active_subset_count_by_size[size] for size in sorted(active_subset_count_by_size)
                },
                "num_components": len(component_to_subset_positions),
                "num_variables": total_num_variables,
                "num_constraints": total_num_constraints,
                "contribution_by_subset_size": {
                    str(size): contribution_by_size[size] for size in sorted(contribution_by_size)
                },
                "allocated_budget_by_subset_size": {
                    str(size): budget_by_size[size] for size in sorted(budget_by_size)
                },
                "top_subsets": top_subset_records[: self.logger.top_k],
            }
            if context is not None:
                payload.update(context)
            self.logger.maybe_log(call_index=self.stats.call_count, payload=payload)

        return total_objective

def compute_heuristic(
    data: DecoderData,
    active_detectors: np.ndarray,
    blocked_errors: np.ndarray,
    active_detector_counts: np.ndarray,
    *,
    opt_subset_solver: Optional[SubsetLPHeuristic],
    context: Optional[Dict[str, Any]] = None,
) -> float:
    if opt_subset_solver is None:
        return plain_detcost_heuristic(
            data=data,
            active_detectors=active_detectors,
            blocked_errors=blocked_errors,
            active_detector_counts=active_detector_counts,
        )
    del active_detector_counts
    return opt_subset_solver.evaluate(
        active_detectors=active_detectors,
        blocked_errors=blocked_errors,
        context=context,
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
    initial_active_detectors = np.asarray(detections, dtype=bool).copy()
    initial_counts = initial_detector_counts(data, initial_active_detectors)
    initial_blocked = np.zeros(len(data.errors), dtype=bool)
    initial_path_cost = 0.0
    initial_heuristic = compute_heuristic(
        data=data,
        active_detectors=initial_active_detectors,
        blocked_errors=initial_blocked,
        active_detector_counts=initial_counts,
        opt_subset_solver=opt_subset_solver,
        context={
            "phase": "initial",
            "depth": 0,
            "nodes_popped": 0,
            "path_cost": 0.0,
            "active_detector_count": int(initial_active_detectors.sum()),
        },
    )
    if initial_heuristic == INF:
        raise RuntimeError("Initial residual syndrome is infeasible under the current pruning rule.")

    initial_state = SearchState(
        activated_errors=(),
        blocked_errors=initial_blocked,
        active_detectors=initial_active_detectors,
        active_detector_counts=initial_counts,
        path_cost=initial_path_cost,
    )

    priority_queue: List[Tuple[float, int, int, SearchState]] = []
    push_counter = 0
    initial_num_dets = int(initial_active_detectors.sum())
    heapq.heappush(
        priority_queue,
        (initial_path_cost + initial_heuristic, initial_num_dets, push_counter, initial_state),
    )
    push_counter += 1

    num_pq_pushed = 1
    num_nodes_popped = 0
    max_queue_size = 1
    min_num_dets = initial_num_dets
    max_num_dets = INF if det_beam == INF else min_num_dets + det_beam

    heuristic_name = (
        f"opt_subset_detcost_size_{opt_subset_solver.subset_library.max_subset_size}"
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
                f"active_dets={num_dets} beam_max={max_num_dets} depth={len(state.activated_errors)} "
                f"f={f_cost:.12g} g={state.path_cost:.12g}"
            )

        if num_dets == 0:
            elapsed_seconds = time.perf_counter() - start_time
            heuristic_calls = 0 if opt_subset_solver is None else opt_subset_solver.stats.call_count
            lp_calls = 0 if opt_subset_solver is None else opt_subset_solver.stats.lp_call_count
            lp_total_seconds = 0.0 if opt_subset_solver is None else opt_subset_solver.stats.lp_total_seconds
            return DecodeResult(
                activated_errors=state.activated_errors,
                path_cost=state.path_cost,
                stats=DecodeStats(
                    num_pq_pushed=num_pq_pushed,
                    num_nodes_popped=num_nodes_popped,
                    max_queue_size=max_queue_size,
                    heuristic_calls=heuristic_calls,
                    lp_calls=lp_calls,
                    lp_total_seconds=lp_total_seconds,
                    elapsed_seconds=elapsed_seconds,
                    heuristic_name=heuristic_name,
                ),
            )

        min_detector = int(np.flatnonzero(state.active_detectors)[0])
        blocked_prefix = state.blocked_errors.copy()
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
                continue

            child_blocked = blocked_prefix.copy()
            child_path_cost = state.path_cost + float(data.error_costs[error_index])
            child_heuristic = compute_heuristic(
                data=data,
                active_detectors=child_active_detectors,
                blocked_errors=child_blocked,
                active_detector_counts=child_active_counts,
                opt_subset_solver=opt_subset_solver,
                context={
                    "phase": "child",
                    "depth": len(state.activated_errors) + 1,
                    "nodes_popped": num_nodes_popped,
                    "path_cost": child_path_cost,
                    "active_detector_count": child_num_dets,
                    "chosen_error": error_index,
                    "min_detector": min_detector,
                },
            )
            if child_heuristic == INF:
                continue

            child_state = SearchState(
                activated_errors=state.activated_errors + (error_index,),
                blocked_errors=child_blocked,
                active_detectors=child_active_detectors,
                active_detector_counts=child_active_counts,
                path_cost=child_path_cost,
            )
            heapq.heappush(
                priority_queue,
                (
                    child_path_cost + child_heuristic,
                    child_num_dets,
                    push_counter,
                    child_state,
                ),
            )
            push_counter += 1
            num_pq_pushed += 1

    raise RuntimeError("Decoding failed to find any completion.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prototype A* decoder for Stim detector error models. "
            "Supports plain detcost and subset-based LP lower bounds."
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
            "Use the subset-based LP heuristic with library subsets of size at most N. "
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
        help="Optional JSONL file for logging details of each subset-LP solve.",
    )
    parser.add_argument(
        "--lp-log-top-k",
        type=int,
        default=10,
        help="When logging LP solves, include at most this many top subsets per solve.",
    )
    parser.add_argument(
        "--lp-log-every",
        type=int,
        default=1,
        help="When logging LP solves, only write every k-th solve.",
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
            else f"opt_subset_detcost_size_{subset_library.max_subset_size}"
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
    print(f"lp_calls = {result.stats.lp_calls}")
    print(f"lp_total_seconds = {result.stats.lp_total_seconds:.6f}")
    print(f"elapsed_seconds = {result.stats.elapsed_seconds:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
