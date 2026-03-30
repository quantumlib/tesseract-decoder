#!/usr/bin/env python3
"""Instrumented A* prototype for studying the optimal singleton LP heuristic.

This script is intentionally data-heavy and not heavily optimized. It decodes a
set of Stim circuits, samples several shots from each, and writes detailed logs
about every heuristic evaluation during search.

Outputs (written under --output-dir):
    manifest.json
    shot_summaries.jsonl
    node_summaries.jsonl.gz
    component_summaries.jsonl.gz
    sampled_instances.jsonl.gz

The node/component logs are designed to answer questions such as:
  * How often is the singleton LP graphlike (all distinct supports have size <= 2)?
  * How many connected components does the residual support hypergraph have?
  * How many raw allowed errors collapse to the same distinct active support?
  * How sparse are primal/dual LP solutions?
  * Are graphlike components common enough to justify a specialized solver?

The search tree uses the same precedence-style pruning idea as the prototype and
Tesseract paper: at each node, only errors incident to the minimum active
residual detector are expanded, with earlier siblings blocked to keep a unique
path ordering.  The A* heuristic can be plain detcost or the optimal singleton
LP; both values are logged for every created node.
"""

from __future__ import annotations

import argparse
import gzip
import heapq
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import stim
from scipy import sparse
from scipy.optimize import linprog

INF = math.inf
JSON_SEPARATORS = (",", ":")
LP_TOL = 1e-9
RATIONAL_TOL = 1e-7


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
    error_observables: List[Tuple[int, ...]]


@dataclass
class SearchSettings:
    det_beam: float
    search_heuristic: str
    respect_blocked_errors_in_heuristic: bool
    max_nodes_popped: Optional[int]
    max_nodes_pushed: Optional[int]
    sample_raw_nodes_per_shot: int
    verbose_search: bool


@dataclass
class SearchState:
    node_id: int
    parent_node_id: Optional[int]
    incoming_error_index: Optional[int]
    depth: int
    activated_errors: Tuple[int, ...]
    activated_error_mask: np.ndarray
    blocked_errors: np.ndarray
    active_detectors: np.ndarray
    active_detector_counts: np.ndarray
    path_cost: float
    search_h: float
    plain_h: float
    opt_h: float


class JsonlWriter:
    def __init__(self, path: Path, *, gz: bool = False):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if gz:
            self.file = gzip.open(path, "wt", encoding="utf-8")
        else:
            self.file = open(path, "wt", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        self.file.write(json.dumps(record, separators=JSON_SEPARATORS, sort_keys=True))
        self.file.write("\n")

    def flush(self) -> None:
        self.file.flush()

    def close(self) -> None:
        self.file.close()


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


class ShotAggregator:
    def __init__(self) -> None:
        self.nodes_created = 0
        self.nodes_pushed = 0
        self.nodes_infeasible = 0
        self.nodes_graphlike = 0
        self.nodes_with_lp = 0
        self.total_plain_h = 0.0
        self.total_opt_h = 0.0
        self.total_h_gain = 0.0
        self.total_lp_time_sec = 0.0
        self.total_lp_vars = 0
        self.total_lp_constraints = 0
        self.total_raw_allowed_errors = 0
        self.total_distinct_supports = 0
        self.total_components = 0
        self.total_graphlike_components = 0
        self.max_active_detectors = 0
        self.max_distinct_supports = 0
        self.max_component_variables = 0
        self.max_component_constraints = 0

    def absorb_node(self, node_record: Dict[str, Any]) -> None:
        self.nodes_created += 1
        self.nodes_pushed += int(bool(node_record["pushed"]))
        self.nodes_infeasible += int(bool(node_record["opt_infeasible"]))
        self.nodes_graphlike += int(bool(node_record["graphlike_all_components"]))
        self.nodes_with_lp += int(node_record["lp_calls"] > 0)
        self.total_plain_h += float(node_record["plain_h"])
        if not node_record["opt_infeasible"]:
            self.total_opt_h += float(node_record["opt_h"])
            self.total_h_gain += float(node_record["opt_h"] - node_record["plain_h"])
        self.total_lp_time_sec += float(node_record["lp_time_sec"])
        self.total_lp_vars += int(node_record["total_lp_vars"])
        self.total_lp_constraints += int(node_record["total_lp_constraints"])
        self.total_raw_allowed_errors += int(node_record["raw_allowed_errors"])
        self.total_distinct_supports += int(node_record["distinct_supports"])
        self.total_components += int(node_record["num_components"])
        self.total_graphlike_components += int(node_record["num_graphlike_components"])
        self.max_active_detectors = max(self.max_active_detectors, int(node_record["num_active_detectors"]))
        self.max_distinct_supports = max(self.max_distinct_supports, int(node_record["distinct_supports"]))
        self.max_component_variables = max(self.max_component_variables, int(node_record["max_component_variables"]))
        self.max_component_constraints = max(self.max_component_constraints, int(node_record["max_component_constraints"]))

    def finish(self, *, nodes_popped: int, status: str, elapsed_seconds: float) -> Dict[str, Any]:
        n = max(self.nodes_created, 1)
        c = max(self.total_components, 1)
        return {
            "status": status,
            "nodes_created": self.nodes_created,
            "nodes_pushed": self.nodes_pushed,
            "nodes_popped": nodes_popped,
            "nodes_infeasible": self.nodes_infeasible,
            "graphlike_node_fraction": self.nodes_graphlike / n,
            "mean_plain_h": self.total_plain_h / n,
            "mean_opt_h_over_feasible": (self.total_opt_h / max(self.nodes_created - self.nodes_infeasible, 1)),
            "mean_opt_minus_plain_over_feasible": (self.total_h_gain / max(self.nodes_created - self.nodes_infeasible, 1)),
            "total_lp_time_sec": self.total_lp_time_sec,
            "mean_lp_time_per_created_node_sec": self.total_lp_time_sec / n,
            "mean_lp_vars_per_created_node": self.total_lp_vars / n,
            "mean_lp_constraints_per_created_node": self.total_lp_constraints / n,
            "mean_raw_allowed_errors": self.total_raw_allowed_errors / n,
            "mean_distinct_supports": self.total_distinct_supports / n,
            "mean_components": self.total_components / n,
            "graphlike_component_fraction": self.total_graphlike_components / c,
            "max_active_detectors": self.max_active_detectors,
            "max_distinct_supports": self.max_distinct_supports,
            "max_component_variables": self.max_component_variables,
            "max_component_constraints": self.max_component_constraints,
            "elapsed_seconds": elapsed_seconds,
        }


class NodeSampler:
    def __init__(self, sample_raw_nodes_per_shot: int):
        self.sample_raw_nodes_per_shot = sample_raw_nodes_per_shot
        self.seen = 0

    def should_sample(self, node_id: int) -> bool:
        del node_id
        if self.seen < self.sample_raw_nodes_per_shot:
            self.seen += 1
            return True
        return False


class ProbeLogger:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.shot_writer = JsonlWriter(output_dir / "shot_summaries.jsonl", gz=False)
        self.node_writer = JsonlWriter(output_dir / "node_summaries.jsonl.gz", gz=True)
        self.component_writer = JsonlWriter(output_dir / "component_summaries.jsonl.gz", gz=True)
        self.sample_writer = JsonlWriter(output_dir / "sampled_instances.jsonl.gz", gz=True)

    def close(self) -> None:
        self.shot_writer.close()
        self.node_writer.close()
        self.component_writer.close()
        self.sample_writer.close()

    def flush(self) -> None:
        self.shot_writer.flush()
        self.node_writer.flush()
        self.component_writer.flush()
        self.sample_writer.flush()


def parse_optional_int(text: str) -> Optional[int]:
    lowered = text.strip().lower()
    if lowered in {"none", "inf", "infinity", "+inf", "+infinity"}:
        return None
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative or one of: none, inf")
    return value


def parse_beam(text: str) -> float:
    lowered = text.strip().lower()
    if lowered in {"inf", "infinity", "+inf", "+infinity"}:
        return INF
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("beam must be non-negative or 'inf'")
    return float(value)


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
                assert target.is_relative_detector_id()
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
    probabilities: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {}
    for error in iter_dem_errors(dem):
        key = (error.detectors, error.observables)
        prev = probabilities.get(key)
        probabilities[key] = error.probability if prev is None else xor_probability(prev, error.probability)

    out: List[MergedError] = []
    for (detectors, observables), probability in probabilities.items():
        if probability <= 0:
            continue
        if probability >= 0.5:
            raise ValueError("Merged error has probability >= 0.5.")
        out.append(
            MergedError(
                probability=probability,
                likelihood_cost=float(-math.log(probability / (1 - probability))),
                detectors=detectors,
                observables=observables,
            )
        )
    return out


def build_decoder_data(dem: stim.DetectorErrorModel, *, merge_errors_in_dem: bool = True) -> DecoderData:
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
    *,
    activated_error_mask: np.ndarray,
    blocked_errors: np.ndarray,
    active_detector_counts: np.ndarray,
    respect_blocked_errors_in_heuristic: bool,
) -> float:
    best = INF
    for ei in data.detector_to_errors[detector]:
        if respect_blocked_errors_in_heuristic:
            if blocked_errors[ei]:
                continue
        else:
            if activated_error_mask[ei]:
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
    *,
    activated_error_mask: np.ndarray,
    blocked_errors: np.ndarray,
    active_detector_counts: np.ndarray,
    respect_blocked_errors_in_heuristic: bool,
) -> float:
    total = 0.0
    for d in np.flatnonzero(active_detectors):
        det_cost = plain_detcost_for_detector(
            data=data,
            detector=int(d),
            activated_error_mask=activated_error_mask,
            blocked_errors=blocked_errors,
            active_detector_counts=active_detector_counts,
            respect_blocked_errors_in_heuristic=respect_blocked_errors_in_heuristic,
        )
        if det_cost == INF:
            return INF
        total += det_cost
    return total


def grid_fraction(values: np.ndarray, denominator: int, tol: float = RATIONAL_TOL) -> float:
    if values.size == 0:
        return 0.0
    scaled = denominator * values
    return float(np.mean(np.abs(scaled - np.round(scaled)) <= tol))


@dataclass
class LPProbeResult:
    opt_h: float
    node_record: Dict[str, Any]
    component_records: List[Dict[str, Any]]
    sample_record: Optional[Dict[str, Any]]


def probe_opt_singleton_lp(
    *,
    run_id: str,
    circuit_name: str,
    shot_index: int,
    state: SearchState,
    data: DecoderData,
    settings: SearchSettings,
    plain_h: float,
    sample_raw_instance: bool,
) -> LPProbeResult:
    active_detector_ids = np.flatnonzero(state.active_detectors)
    num_active_detectors = int(active_detector_ids.size)
    global_to_local = np.full(data.num_detectors, -1, dtype=np.int32)
    global_to_local[active_detector_ids] = np.arange(num_active_detectors, dtype=np.int32)

    support_to_cost: Dict[Tuple[int, ...], float] = {}
    support_to_multiplicity: Dict[Tuple[int, ...], int] = {}
    covered = np.zeros(num_active_detectors, dtype=bool)

    raw_allowed_errors = 0
    raw_support_size_hist = {"1": 0, "2": 0, "3": 0, "4+": 0}

    for ei, error_detectors in enumerate(data.error_detectors):
        if settings.respect_blocked_errors_in_heuristic:
            if state.blocked_errors[ei]:
                continue
        else:
            if state.activated_error_mask[ei]:
                continue

        count = int(state.active_detector_counts[ei])
        if count == 0:
            continue
        support = tuple(int(global_to_local[d]) for d in error_detectors if state.active_detectors[d])
        assert support
        raw_allowed_errors += 1
        size = len(support)
        if size == 1:
            raw_support_size_hist["1"] += 1
        elif size == 2:
            raw_support_size_hist["2"] += 1
        elif size == 3:
            raw_support_size_hist["3"] += 1
        else:
            raw_support_size_hist["4+"] += 1
        covered[list(support)] = True
        support_to_multiplicity[support] = support_to_multiplicity.get(support, 0) + 1
        cost = float(data.error_costs[ei])
        prev = support_to_cost.get(support)
        if prev is None or cost < prev:
            support_to_cost[support] = cost

    distinct_support_size_hist = {"1": 0, "2": 0, "3": 0, "4+": 0}
    for support in support_to_cost:
        size = len(support)
        if size == 1:
            distinct_support_size_hist["1"] += 1
        elif size == 2:
            distinct_support_size_hist["2"] += 1
        elif size == 3:
            distinct_support_size_hist["3"] += 1
        else:
            distinct_support_size_hist["4+"] += 1

    uncovered_count = int(np.count_nonzero(~covered))
    base_node_record: Dict[str, Any] = {
        "run_id": run_id,
        "circuit": circuit_name,
        "shot": shot_index,
        "node_id": state.node_id,
        "parent_node_id": state.parent_node_id,
        "incoming_error_index": state.incoming_error_index,
        "depth": state.depth,
        "num_active_detectors": num_active_detectors,
        "path_cost": state.path_cost,
        "plain_h": plain_h,
        "raw_allowed_errors": raw_allowed_errors,
        "raw_support_hist": raw_support_size_hist,
        "distinct_supports": len(support_to_cost),
        "distinct_support_hist": distinct_support_size_hist,
        "support_multiplicity_mean": (float(np.mean(list(support_to_multiplicity.values()))) if support_to_multiplicity else 0.0),
        "support_multiplicity_max": (max(support_to_multiplicity.values()) if support_to_multiplicity else 0),
        "uncovered_active_detectors": uncovered_count,
    }

    if uncovered_count > 0:
        base_node_record.update(
            {
                "opt_h": INF,
                "opt_infeasible": True,
                "lp_calls": 0,
                "lp_time_sec": 0.0,
                "total_lp_vars": 0,
                "total_lp_constraints": 0,
                "num_components": 0,
                "num_graphlike_components": 0,
                "graphlike_all_components": False,
                "max_support_size": 0,
                "max_component_variables": 0,
                "max_component_constraints": 0,
                "positive_y_count": 0,
                "tight_constraint_count": 0,
                "positive_dual_count": 0,
            }
        )
        sample_record = None
        if sample_raw_instance:
            sample_record = {
                "run_id": run_id,
                "circuit": circuit_name,
                "shot": shot_index,
                "node_id": state.node_id,
                "parent_node_id": state.parent_node_id,
                "depth": state.depth,
                "opt_infeasible": True,
                "active_detector_ids": active_detector_ids.tolist(),
                "supports": [
                    {
                        "local_support": list(support),
                        "global_support": [int(active_detector_ids[i]) for i in support],
                        "cost": support_to_cost[support],
                        "multiplicity": support_to_multiplicity[support],
                    }
                    for support in sorted(support_to_cost)
                ],
            }
        return LPProbeResult(
            opt_h=INF,
            node_record=base_node_record,
            component_records=[],
            sample_record=sample_record,
        )

    union_find = UnionFind(num_active_detectors)
    for support in support_to_cost:
        first = support[0]
        for detector in support[1:]:
            union_find.union(first, detector)

    detectors_by_root: Dict[int, List[int]] = {}
    for detector in range(num_active_detectors):
        root = union_find.find(detector)
        detectors_by_root.setdefault(root, []).append(detector)

    supports_by_root: Dict[int, List[Tuple[Tuple[int, ...], float, int]]] = {}
    for support, cost in support_to_cost.items():
        root = union_find.find(support[0])
        supports_by_root.setdefault(root, []).append((support, cost, support_to_multiplicity[support]))

    component_records: List[Dict[str, Any]] = []
    sample_components: List[Dict[str, Any]] = []
    total_opt_h = 0.0
    total_lp_time = 0.0
    total_lp_vars = 0
    total_lp_constraints = 0
    total_positive_y = 0
    total_tight_constraints = 0
    total_positive_dual = 0
    num_graphlike_components = 0
    max_component_variables = 0
    max_component_constraints = 0
    max_support_size = max((len(support) for support in support_to_cost), default=0)

    for component_index, (root, component_detectors) in enumerate(sorted(detectors_by_root.items())):
        local_reindex = {detector: i for i, detector in enumerate(component_detectors)}
        component_supports = supports_by_root[root]
        num_vars = len(component_detectors)
        num_constraints = len(component_supports)
        max_component_variables = max(max_component_variables, num_vars)
        max_component_constraints = max(max_component_constraints, num_constraints)
        total_lp_vars += num_vars
        total_lp_constraints += num_constraints

        row_indices: List[int] = []
        col_indices: List[int] = []
        values: List[float] = []
        rhs = np.empty(num_constraints, dtype=np.float64)
        component_global_supports: List[List[int]] = []
        support_sizes = np.empty(num_constraints, dtype=np.int32)
        multiplicities = np.empty(num_constraints, dtype=np.int32)

        graphlike = True
        support_size_hist = {"1": 0, "2": 0, "3": 0, "4+": 0}
        for row, (support, cost, multiplicity) in enumerate(component_supports):
            rhs[row] = cost
            multiplicities[row] = multiplicity
            reindexed_support = [local_reindex[d] for d in support]
            support_sizes[row] = len(reindexed_support)
            if support_sizes[row] == 1:
                support_size_hist["1"] += 1
            elif support_sizes[row] == 2:
                support_size_hist["2"] += 1
            elif support_sizes[row] == 3:
                support_size_hist["3"] += 1
                graphlike = False
            else:
                support_size_hist["4+"] += 1
                graphlike = False
            component_global_supports.append([int(active_detector_ids[d]) for d in support])
            for col in reindexed_support:
                row_indices.append(row)
                col_indices.append(col)
                values.append(1.0)

        a_ub = sparse.csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(num_constraints, num_vars),
            dtype=np.float64,
        )

        t0 = time.perf_counter()
        result = linprog(
            c=-np.ones(num_vars, dtype=np.float64),
            A_ub=a_ub,
            b_ub=rhs,
            bounds=[(0.0, None)] * num_vars,
            method="highs",
        )
        lp_time_sec = time.perf_counter() - t0
        total_lp_time += lp_time_sec
        if not result.success:
            raise RuntimeError(
                f"LP solve failed for circuit={circuit_name} shot={shot_index} node={state.node_id} "
                f"component={component_index}: {result.message}"
            )

        y = np.asarray(result.x, dtype=np.float64)
        total_opt_h += float(-result.fun)
        positive_y_mask = y > LP_TOL
        positive_y_count = int(np.count_nonzero(positive_y_mask))
        total_positive_y += positive_y_count

        if hasattr(result, "ineqlin") and hasattr(result.ineqlin, "residual"):
            residual = np.asarray(result.ineqlin.residual, dtype=np.float64)
        else:
            residual = rhs - a_ub.dot(y)
        tight_mask = residual <= LP_TOL
        tight_count = int(np.count_nonzero(tight_mask))
        total_tight_constraints += tight_count

        if hasattr(result, "ineqlin") and hasattr(result.ineqlin, "marginals"):
            dual = -np.asarray(result.ineqlin.marginals, dtype=np.float64)
        else:
            dual = np.full(num_constraints, np.nan)
        if np.isnan(dual).any():
            positive_dual_mask = np.zeros(num_constraints, dtype=bool)
            positive_dual = np.zeros(0, dtype=np.float64)
        else:
            positive_dual_mask = dual > LP_TOL
            positive_dual = dual[positive_dual_mask]
        positive_dual_count = int(np.count_nonzero(positive_dual_mask))
        total_positive_dual += positive_dual_count

        if graphlike:
            num_graphlike_components += 1

        positive_dual_size_hist = {"1": 0, "2": 0, "3": 0, "4+": 0}
        for size in support_sizes[positive_dual_mask]:
            if size == 1:
                positive_dual_size_hist["1"] += 1
            elif size == 2:
                positive_dual_size_hist["2"] += 1
            elif size == 3:
                positive_dual_size_hist["3"] += 1
            else:
                positive_dual_size_hist["4+"] += 1

        component_record = {
            "run_id": run_id,
            "circuit": circuit_name,
            "shot": shot_index,
            "node_id": state.node_id,
            "component_index": component_index,
            "num_variables": num_vars,
            "num_constraints": num_constraints,
            "objective": float(-result.fun),
            "lp_time_sec": lp_time_sec,
            "graphlike": graphlike,
            "max_support_size": int(np.max(support_sizes) if support_sizes.size else 0),
            "support_hist": support_size_hist,
            "positive_y_count": positive_y_count,
            "tight_constraint_count": tight_count,
            "positive_dual_count": positive_dual_count,
            "dual_integral_fraction": grid_fraction(positive_dual, 1),
            "dual_half_integral_fraction": grid_fraction(positive_dual, 2),
            "dual_third_integral_fraction": grid_fraction(positive_dual, 3),
            "dual_quarter_integral_fraction": grid_fraction(positive_dual, 4),
            "positive_dual_support_hist": positive_dual_size_hist,
            "support_multiplicity_mean": float(np.mean(multiplicities)) if multiplicities.size else 0.0,
            "support_multiplicity_max": int(np.max(multiplicities) if multiplicities.size else 0),
        }
        component_records.append(component_record)

        if sample_raw_instance:
            sample_components.append(
                {
                    "component_index": component_index,
                    "global_detector_ids": [int(active_detector_ids[d]) for d in component_detectors],
                    "supports": [
                        {
                            "global_support": component_global_supports[row],
                            "cost": float(rhs[row]),
                            "multiplicity": int(multiplicities[row]),
                            "dual": float(dual[row]) if not np.isnan(dual[row]) else None,
                            "slack": float(residual[row]),
                        }
                        for row in range(num_constraints)
                    ],
                    "y": [float(v) for v in y],
                }
            )

    base_node_record.update(
        {
            "opt_h": total_opt_h,
            "opt_infeasible": False,
            "lp_calls": len(component_records),
            "lp_time_sec": total_lp_time,
            "total_lp_vars": total_lp_vars,
            "total_lp_constraints": total_lp_constraints,
            "num_components": len(component_records),
            "num_graphlike_components": num_graphlike_components,
            "graphlike_all_components": num_graphlike_components == len(component_records),
            "max_support_size": max_support_size,
            "max_component_variables": max_component_variables,
            "max_component_constraints": max_component_constraints,
            "positive_y_count": total_positive_y,
            "tight_constraint_count": total_tight_constraints,
            "positive_dual_count": total_positive_dual,
        }
    )

    sample_record = None
    if sample_raw_instance:
        sample_record = {
            "run_id": run_id,
            "circuit": circuit_name,
            "shot": shot_index,
            "node_id": state.node_id,
            "parent_node_id": state.parent_node_id,
            "incoming_error_index": state.incoming_error_index,
            "depth": state.depth,
            "path_cost": state.path_cost,
            "plain_h": plain_h,
            "opt_h": total_opt_h,
            "active_detector_ids": active_detector_ids.tolist(),
            "components": sample_components,
        }

    return LPProbeResult(
        opt_h=total_opt_h,
        node_record=base_node_record,
        component_records=component_records,
        sample_record=sample_record,
    )


def compute_node_metrics(
    *,
    run_id: str,
    circuit_name: str,
    shot_index: int,
    state: SearchState,
    data: DecoderData,
    settings: SearchSettings,
    sample_raw_instance: bool,
) -> LPProbeResult:
    plain_h = plain_detcost_heuristic(
        data=data,
        active_detectors=state.active_detectors,
        activated_error_mask=state.activated_error_mask,
        blocked_errors=state.blocked_errors,
        active_detector_counts=state.active_detector_counts,
        respect_blocked_errors_in_heuristic=settings.respect_blocked_errors_in_heuristic,
    )
    lp_probe = probe_opt_singleton_lp(
        run_id=run_id,
        circuit_name=circuit_name,
        shot_index=shot_index,
        state=state,
        data=data,
        settings=settings,
        plain_h=plain_h,
        sample_raw_instance=sample_raw_instance,
    )
    return lp_probe


def observables_from_solution(data: DecoderData, activated_errors: Sequence[int]) -> np.ndarray:
    observables = np.zeros(data.num_observables, dtype=bool)
    for error_index in activated_errors:
        for observable in data.error_observables[error_index]:
            observables[observable] ^= True
    return observables


def detectors_from_solution(data: DecoderData, activated_errors: Sequence[int]) -> np.ndarray:
    detectors = np.zeros(data.num_detectors, dtype=bool)
    for error_index in activated_errors:
        for detector in data.error_detectors[error_index]:
            detectors[detector] ^= True
    return detectors


def heuristic_for_search(settings: SearchSettings, plain_h: float, opt_h: float) -> float:
    if settings.search_heuristic == "plain":
        return plain_h
    if settings.search_heuristic == "opt":
        return opt_h
    raise ValueError(f"Unknown search heuristic: {settings.search_heuristic}")


def decode_and_probe_shot(
    *,
    run_id: str,
    circuit_name: str,
    shot_index: int,
    shot_detections: np.ndarray,
    shot_observables: np.ndarray,
    data: DecoderData,
    settings: SearchSettings,
    logger: ProbeLogger,
) -> Dict[str, Any]:
    shot_start = time.perf_counter()
    sampler = NodeSampler(settings.sample_raw_nodes_per_shot)
    aggregator = ShotAggregator()

    initial_active_detectors = np.asarray(shot_detections, dtype=bool).copy()
    initial_counts = initial_detector_counts(data, initial_active_detectors)
    initial_activated_mask = np.zeros(len(data.errors), dtype=bool)
    initial_blocked = np.zeros(len(data.errors), dtype=bool)

    root_state = SearchState(
        node_id=0,
        parent_node_id=None,
        incoming_error_index=None,
        depth=0,
        activated_errors=(),
        activated_error_mask=initial_activated_mask,
        blocked_errors=initial_blocked,
        active_detectors=initial_active_detectors,
        active_detector_counts=initial_counts,
        path_cost=0.0,
        search_h=0.0,
        plain_h=0.0,
        opt_h=0.0,
    )

    root_probe = compute_node_metrics(
        run_id=run_id,
        circuit_name=circuit_name,
        shot_index=shot_index,
        state=root_state,
        data=data,
        settings=settings,
        sample_raw_instance=sampler.should_sample(root_state.node_id),
    )
    root_state.plain_h = float(root_probe.node_record["plain_h"])
    root_state.opt_h = float(root_probe.node_record["opt_h"])
    root_state.search_h = heuristic_for_search(settings, root_state.plain_h, root_state.opt_h)
    if root_state.search_h == INF:
        raise RuntimeError(
            f"Root node is infeasible for circuit={circuit_name} shot={shot_index}."
        )

    root_record = {
        **root_probe.node_record,
        "search_h": root_state.search_h,
        "f_cost": root_state.path_cost + root_state.search_h,
        "pushed": True,
    }
    logger.node_writer.write(root_record)
    for component_record in root_probe.component_records:
        logger.component_writer.write(component_record)
    if root_probe.sample_record is not None:
        logger.sample_writer.write(root_probe.sample_record)
    aggregator.absorb_node(root_record)

    queue: List[Tuple[float, int, int, SearchState]] = []
    heapq_push_counter = 0
    npush = 1
    popped = 0
    max_queue_size = 1
    min_num_dets = int(initial_active_detectors.sum())
    max_num_dets = INF if settings.det_beam == INF else min_num_dets + settings.det_beam
    heapq.heappush(queue, (root_state.path_cost + root_state.search_h, min_num_dets, heapq_push_counter, root_state))
    heapq_push_counter += 1
    next_node_id = 1

    solution_state: Optional[SearchState] = None
    status = "unknown"

    while queue:
        max_queue_size = max(max_queue_size, len(queue))
        f_cost, num_dets, _, state = heapq.heappop(queue)
        popped += 1

        if settings.max_nodes_popped is not None and popped > settings.max_nodes_popped:
            status = "max_nodes_popped"
            break

        if num_dets > max_num_dets:
            continue

        if settings.verbose_search:
            print(
                f"[{circuit_name} shot={shot_index}] nodes_popped={popped} pq={len(queue)} "
                f"active_dets={num_dets} max_active_dets={max_num_dets} depth={state.depth} "
                f"g={state.path_cost:.12g} h={state.search_h:.12g} f={f_cost:.12g}",
                flush=True,
            )

        if num_dets == 0:
            solution_state = state
            status = "success"
            break

        if num_dets < min_num_dets:
            min_num_dets = num_dets
            max_num_dets = INF if settings.det_beam == INF else min_num_dets + settings.det_beam

        min_detector = int(np.flatnonzero(state.active_detectors)[0])
        blocked_prefix = state.blocked_errors.copy()

        for error_index in data.detector_to_errors[min_detector]:
            blocked_prefix[error_index] = True
            if state.blocked_errors[error_index]:
                continue

            child_active_detectors, child_counts = apply_error(
                data=data,
                active_detectors=state.active_detectors,
                active_detector_counts=state.active_detector_counts,
                error_index=error_index,
            )
            child_num_dets = int(child_active_detectors.sum())
            if child_num_dets > max_num_dets:
                continue

            child_activated_mask = state.activated_error_mask.copy()
            child_activated_mask[error_index] = True
            child_blocked = blocked_prefix.copy()
            child_path_cost = state.path_cost + float(data.error_costs[error_index])

            child_state = SearchState(
                node_id=next_node_id,
                parent_node_id=state.node_id,
                incoming_error_index=error_index,
                depth=state.depth + 1,
                activated_errors=state.activated_errors + (error_index,),
                activated_error_mask=child_activated_mask,
                blocked_errors=child_blocked,
                active_detectors=child_active_detectors,
                active_detector_counts=child_counts,
                path_cost=child_path_cost,
                search_h=0.0,
                plain_h=0.0,
                opt_h=0.0,
            )
            next_node_id += 1

            child_probe = compute_node_metrics(
                run_id=run_id,
                circuit_name=circuit_name,
                shot_index=shot_index,
                state=child_state,
                data=data,
                settings=settings,
                sample_raw_instance=sampler.should_sample(child_state.node_id),
            )
            child_state.plain_h = float(child_probe.node_record["plain_h"])
            child_state.opt_h = float(child_probe.node_record["opt_h"])
            child_state.search_h = heuristic_for_search(settings, child_state.plain_h, child_state.opt_h)

            pushed = child_state.search_h != INF
            child_record = {
                **child_probe.node_record,
                "search_h": child_state.search_h,
                "f_cost": child_state.path_cost + child_state.search_h,
                "pushed": pushed,
            }
            logger.node_writer.write(child_record)
            for component_record in child_probe.component_records:
                logger.component_writer.write(component_record)
            if child_probe.sample_record is not None:
                logger.sample_writer.write(child_probe.sample_record)
            aggregator.absorb_node(child_record)

            if not pushed:
                continue

            heapq.heappush(
                queue,
                (
                    child_state.path_cost + child_state.search_h,
                    child_num_dets,
                    heapq_push_counter,
                    child_state,
                ),
            )
            heapq_push_counter += 1
            npush += 1
            if settings.max_nodes_pushed is not None and npush > settings.max_nodes_pushed:
                status = "max_nodes_pushed"
                queue.clear()
                break

        if status == "max_nodes_pushed":
            break

    if status == "unknown":
        status = "empty_queue"

    elapsed_seconds = time.perf_counter() - shot_start
    predicted_observables: Optional[np.ndarray] = None
    solution_cost: Optional[float] = None
    observables_match: Optional[bool] = None
    solution_size: Optional[int] = None

    if solution_state is not None:
        reproduced_detectors = detectors_from_solution(data, solution_state.activated_errors)
        if not np.array_equal(reproduced_detectors, shot_detections):
            raise AssertionError(
                f"Decoded error set does not reproduce the shot syndrome for circuit={circuit_name} shot={shot_index}."
            )
        predicted_observables = observables_from_solution(data, solution_state.activated_errors)
        observables_match = bool(np.array_equal(predicted_observables, shot_observables))
        solution_cost = float(solution_state.path_cost)
        solution_size = len(solution_state.activated_errors)

    summary = {
        "run_id": run_id,
        "circuit": circuit_name,
        "shot": shot_index,
        **aggregator.finish(nodes_popped=popped, status=status, elapsed_seconds=elapsed_seconds),
        "max_queue_size": max_queue_size,
        "det_beam": settings.det_beam,
        "search_heuristic": settings.search_heuristic,
        "solution_cost": solution_cost,
        "solution_size": solution_size,
        "observables_match": observables_match,
        "predicted_observables": (np.flatnonzero(predicted_observables).tolist() if predicted_observables is not None else None),
        "sample_observables": np.flatnonzero(shot_observables).tolist(),
    }
    logger.shot_writer.write(summary)
    logger.flush()
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the prototype decoder on several circuits and log detailed LP-structure data "
            "for the optimal singleton heuristic."
        )
    )
    parser.add_argument(
        "circuits",
        nargs="+",
        type=Path,
        help="Stim circuit files to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where logs will be written.",
    )
    parser.add_argument(
        "--shots-per-circuit",
        type=int,
        default=10,
        help="Number of sampled shots to decode per circuit (default: 10).",
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
        help="Beam cutoff on residual detector count. Use an integer or 'inf'.",
    )
    parser.add_argument(
        "--max-nodes-popped",
        type=parse_optional_int,
        default=5000,
        help="Stop after this many popped nodes per shot (default: 5000; use 'none' for no limit).",
    )
    parser.add_argument(
        "--max-nodes-pushed",
        type=parse_optional_int,
        default=50000,
        help="Stop after this many pushed nodes per shot (default: 50000; use 'none' for no limit).",
    )
    parser.add_argument(
        "--search-heuristic",
        choices=["plain", "opt"],
        default="opt",
        help="Heuristic used for queue ordering. Both plain and optimal values are always logged.",
    )
    parser.add_argument(
        "--respect-blocked-errors-in-heuristic",
        action="store_true",
        help=(
            "When set, both heuristics exclude precedence-blocked errors as well as already-activated errors. "
            "By default, heuristics only exclude already-activated errors, matching the original prototype."
        ),
    )
    parser.add_argument(
        "--merge-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge indistinguishable DEM errors before decoding (default: enabled).",
    )
    parser.add_argument(
        "--sample-raw-nodes-per-shot",
        type=int,
        default=25,
        help="How many raw LP instances to dump per shot (default: 25).",
    )
    parser.add_argument(
        "--verbose-search",
        action="store_true",
        help="Print one line per popped node.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-shot progress printing.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.shots_per_circuit <= 0:
        parser.error("--shots-per-circuit must be positive.")
    if args.sample_raw_nodes_per_shot < 0:
        parser.error("--sample-raw-nodes-per-shot must be non-negative.")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"singleton_lp_probe_{int(time.time())}"

    manifest = {
        "run_id": run_id,
        "argv": list(argv) if argv is not None else sys.argv[1:],
        "circuits": [str(p) for p in args.circuits],
        "shots_per_circuit": args.shots_per_circuit,
        "seed": args.seed,
        "det_beam": args.det_beam,
        "max_nodes_popped": args.max_nodes_popped,
        "max_nodes_pushed": args.max_nodes_pushed,
        "search_heuristic": args.search_heuristic,
        "respect_blocked_errors_in_heuristic": args.respect_blocked_errors_in_heuristic,
        "merge_errors": args.merge_errors,
        "sample_raw_nodes_per_shot": args.sample_raw_nodes_per_shot,
        "lp_tol": LP_TOL,
        "rational_tol": RATIONAL_TOL,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    logger = ProbeLogger(output_dir)
    settings = SearchSettings(
        det_beam=args.det_beam,
        search_heuristic=args.search_heuristic,
        respect_blocked_errors_in_heuristic=args.respect_blocked_errors_in_heuristic,
        max_nodes_popped=args.max_nodes_popped,
        max_nodes_pushed=args.max_nodes_pushed,
        sample_raw_nodes_per_shot=args.sample_raw_nodes_per_shot,
        verbose_search=args.verbose_search,
    )

    try:
        for circuit_path in args.circuits:
            circuit = stim.Circuit.from_file(str(circuit_path))
            dem = circuit.detector_error_model(decompose_errors=False)
            data = build_decoder_data(dem, merge_errors_in_dem=args.merge_errors)
            dets_packed, obs_packed = circuit.compile_detector_sampler(seed=args.seed).sample(
                shots=args.shots_per_circuit,
                separate_observables=True,
                bit_packed=True,
            )
            detections = unpack_bit_packed_rows(dets_packed, count=dem.num_detectors)
            observables = unpack_bit_packed_rows(obs_packed, count=dem.num_observables)
            circuit_name = circuit_path.name

            for shot_index in range(args.shots_per_circuit):
                if not args.quiet:
                    print(
                        f"[{run_id}] circuit={circuit_name} shot={shot_index} "
                        f"detectors={int(np.count_nonzero(detections[shot_index]))} ...",
                        flush=True,
                    )
                summary = decode_and_probe_shot(
                    run_id=run_id,
                    circuit_name=circuit_name,
                    shot_index=shot_index,
                    shot_detections=detections[shot_index],
                    shot_observables=observables[shot_index] if observables.size else np.zeros(0, dtype=bool),
                    data=data,
                    settings=settings,
                    logger=logger,
                )
                if not args.quiet:
                    print(
                        f"[{run_id}] done circuit={circuit_name} shot={shot_index} status={summary['status']} "
                        f"nodes_popped={summary['nodes_popped']} nodes_created={summary['nodes_created']} "
                        f"total_lp_time_sec={summary['total_lp_time_sec']:.6f}",
                        flush=True,
                    )
    finally:
        logger.close()

    if not args.quiet:
        print(f"Wrote logs under {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
