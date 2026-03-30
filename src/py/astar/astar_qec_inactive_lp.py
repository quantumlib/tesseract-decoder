#!/usr/bin/env python3
"""Prototype A* decoder for Stim circuits using greedy singleton-budget heuristics.

This version keeps the same Stim-facing API as the earlier greedy prototype but
adds lazy reinsertion / parent-y projection, in the same spirit as the lazy
optimal-singleton prototype:

  * nodes are seeded with a cheap feasible lower bound;
  * when a node is popped, the selected heuristic is evaluated on that node;
  * if the refined heuristic raises the node key, the node is reinserted;
  * expanded nodes project their current feasible y-prices onto children;
  * optionally, the projected child bound is maxed with plain detcost.

Supported heuristic choices:
    plain          original detector-wise feasible point
    asc_deg        zero-start saturation ordered by ascending detector degree
    desc_plain     zero-start saturation ordered by descending plain y_d
    plain_sweep    start from plain, then one descending saturation sweep
    best_of_two    max(plain_sweep, asc_deg)
    best_of_three  max(plain_sweep, asc_deg, desc_plain)
    exact_lp        exact optimal singleton LP lower bound
    exact_lp_plus_inactive
                    exact LP lower bound with extra inactive-detector no-one-hot constraints

When --lazy-reinsert-heuristics is enabled (the default), the root is seeded by
plain detcost and only popped nodes are refined with the selected heuristic.
This works directly for the support-only heuristics because each returns a
feasible singleton-budget vector y, and projecting that y to a child by
keeping prices on detectors that remain active and zeroing newly active
detectors is still a feasible child singleton-budget point. For
exact_lp_plus_inactive, the refined LP optimum is not directly projectable to a
child, so lazy mode keeps the current projectable singleton-budget prices for
child seeding and uses the tightened LP only when refining popped nodes.
"""

from __future__ import annotations

import argparse
import heapq
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import stim
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

INF = float("inf")
HEURISTIC_EPS = 1e-9


@dataclass(frozen=True)
class ErrorRecord:
    probability: float
    likelihood_cost: float
    detectors: Tuple[int, ...]
    observables: Tuple[int, ...]


@dataclass
class SupportData:
    active_detectors: List[int]
    supports: List[Tuple[Tuple[int, ...], float]]
    incident: Dict[int, List[int]]


@dataclass
class SearchState:
    errs: np.ndarray
    blocked_errs: np.ndarray
    dets: np.ndarray
    det_counts: np.ndarray
    g_cost: float
    h_cost: float
    h_source: str
    refined: bool
    y_prices: Optional[np.ndarray]


@dataclass
class DecodeResult:
    success: bool
    errs: np.ndarray
    residual_dets: np.ndarray
    cost: float
    nodes_pushed: int
    nodes_popped: int
    max_queue_size: int
    heuristic_calls: int
    plain_heuristic_calls: int
    projection_heuristic_calls: int
    refinement_calls: int
    lp_calls: int
    reinserts: int
    projected_nodes_generated: int
    projected_nodes_refined: int
    projected_nodes_unrefined_at_finish: int
    total_refinement_gain: float
    max_refinement_gain: float
    elapsed_seconds: float


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

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


def xor_probability(p0: float, p1: float) -> float:
    return p0 * (1.0 - p1) + (1.0 - p0) * p1


def iter_dem_errors_from_dem(dem: stim.DetectorErrorModel) -> Iterable[ErrorRecord]:
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        probability = float(instruction.args_copy()[0])
        if probability <= 0:
            continue
        if probability >= 0.5:
            raise ValueError(
                f"Expected flattened error probabilities in (0, 0.5), got {probability}."
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

        yield ErrorRecord(
            probability=probability,
            likelihood_cost=-math.log(probability / (1.0 - probability)),
            detectors=tuple(sorted(detectors)),
            observables=tuple(sorted(observables)),
        )


def merged_errors_from_dem(dem: stim.DetectorErrorModel) -> List[ErrorRecord]:
    errors_by_symptom: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {}
    for error in iter_dem_errors_from_dem(dem):
        key = (error.detectors, error.observables)
        p_old = errors_by_symptom.get(key)
        if p_old is None:
            p_new = error.probability
        else:
            p_new = xor_probability(p_old, error.probability)
        errors_by_symptom[key] = p_new

    merged: List[ErrorRecord] = []
    for (detectors, observables), probability in errors_by_symptom.items():
        if probability <= 0:
            continue
        if probability >= 0.5:
            raise ValueError(
                f"Merged error has probability >= 0.5 ({probability}); cannot assign positive cost."
            )
        merged.append(
            ErrorRecord(
                probability=probability,
                likelihood_cost=-math.log(probability / (1.0 - probability)),
                detectors=detectors,
                observables=observables,
            )
        )
    return merged


class GreedySingletonHeuristicDecoder:
    def __init__(
        self,
        errors: Sequence[ErrorRecord],
        num_detectors: int,
        num_observables: int,
        *,
        heuristic: str = "best_of_two",
        respect_blocked_errors_in_heuristic: bool = True,
        lazy_reinsert_heuristics: bool = True,
        projection_combine_max_plain: bool = True,
        verbose_search: bool = False,
    ) -> None:
        self.errors = list(errors)
        self.num_errors = len(self.errors)
        self.num_detectors = int(num_detectors)
        self.num_observables = int(num_observables)
        self.heuristic_name = heuristic
        self.respect_blocked_errors_in_heuristic = respect_blocked_errors_in_heuristic
        self.lazy_reinsert_heuristics = lazy_reinsert_heuristics
        self.projection_combine_max_plain = projection_combine_max_plain
        self.verbose_search = verbose_search

        self.probabilities = np.array([err.probability for err in self.errors], dtype=np.float64)
        self.weights = np.array([err.likelihood_cost for err in self.errors], dtype=np.float64)
        self.error_detectors: List[Tuple[int, ...]] = [tuple(err.detectors) for err in self.errors]
        self.error_observables: List[Tuple[int, ...]] = [tuple(err.observables) for err in self.errors]

        d2e_lists: List[List[int]] = [[] for _ in range(self.num_detectors)]
        for ei, dets in enumerate(self.error_detectors):
            for d in dets:
                d2e_lists[d].append(ei)
        self.d2e: List[np.ndarray] = [np.array(v, dtype=np.int32) for v in d2e_lists]

        self.reset_stats()

    def reset_stats(self) -> None:
        self.heuristic_calls = 0
        self.plain_heuristic_calls = 0
        self.projection_heuristic_calls = 0
        self.refinement_calls = 0
        self.lp_calls = 0
        self.reinserts = 0
        self.projected_nodes_generated = 0
        self.projected_nodes_refined = 0
        self.total_refinement_gain = 0.0
        self.max_refinement_gain = 0.0

    @property
    def mode_name(self) -> str:
        if self.heuristic_name == "plain":
            return "plain"
        if self.lazy_reinsert_heuristics:
            suffix = "-lazy-projection"
            if self.projection_combine_max_plain:
                suffix += "-maxplain"
            return f"{self.heuristic_name}{suffix}"
        return self.heuristic_name

    @staticmethod
    def heuristic_has_projectable_prices(name: str) -> bool:
        return name != "exact_lp_plus_inactive"

    def _available_errors(self, errs: np.ndarray, blocked_errs: np.ndarray) -> np.ndarray:
        available = ~errs
        if self.respect_blocked_errors_in_heuristic:
            available &= ~blocked_errs
        return available

    def _has_cover_for_all_active_detectors(self, dets: np.ndarray, available_errs: np.ndarray) -> bool:
        for d in np.flatnonzero(dets):
            found = False
            for ei in self.d2e[int(d)]:
                if available_errs[int(ei)]:
                    found = True
                    break
            if not found:
                return False
        return True

    def build_support_data(self, active_dets: np.ndarray, available_errs: np.ndarray) -> SupportData:
        active_list = sorted(map(int, np.flatnonzero(active_dets)))
        incident: Dict[int, List[int]] = {d: [] for d in active_list}
        support_to_weight: Dict[Tuple[int, ...], float] = {}

        for ei in np.flatnonzero(available_errs):
            ei = int(ei)
            support = tuple(d for d in self.error_detectors[ei] if active_dets[d])
            if not support:
                continue
            weight = float(self.weights[ei])
            old = support_to_weight.get(support)
            if old is None or weight < old:
                support_to_weight[support] = weight

        supports = list(support_to_weight.items())
        for i, (support, _weight) in enumerate(supports):
            for d in support:
                if d in incident:
                    incident[d].append(i)

        return SupportData(active_detectors=active_list, supports=supports, incident=incident)

    def _check_coverage(self, support_data: SupportData) -> bool:
        return all(len(support_data.incident[d]) > 0 for d in support_data.active_detectors)

    def plain_detcost_from_counts(
        self,
        dets: np.ndarray,
        available_errs: np.ndarray,
        det_counts: np.ndarray,
    ) -> Tuple[float, Optional[np.ndarray]]:
        self.heuristic_calls += 1
        self.plain_heuristic_calls += 1
        active = np.flatnonzero(dets)
        if active.size == 0:
            return 0.0, np.zeros(self.num_detectors, dtype=np.float64)

        y = np.zeros(self.num_detectors, dtype=np.float64)
        total = 0.0
        for d in active:
            best = INF
            for ei in self.d2e[int(d)]:
                ei = int(ei)
                if not available_errs[ei]:
                    continue
                count = int(det_counts[ei])
                assert count > 0
                value = self.weights[ei] / count
                if value < best:
                    best = value
            if math.isinf(best):
                return INF, None
            y[int(d)] = best
            total += best
        return total, y

    def heuristic_plain(self, support_data: SupportData) -> Tuple[float, Optional[np.ndarray]]:
        if not support_data.active_detectors:
            return 0.0, np.zeros(self.num_detectors, dtype=np.float64)
        if not self._check_coverage(support_data):
            return INF, None
        y = np.zeros(self.num_detectors, dtype=np.float64)
        for d in support_data.active_detectors:
            best = INF
            for i in support_data.incident[d]:
                support, weight = support_data.supports[i]
                best = min(best, weight / len(support))
            y[d] = best
        return float(y[support_data.active_detectors].sum()), y

    def heuristic_saturation_zero(self, support_data: SupportData, *, order_kind: str) -> Tuple[float, Optional[np.ndarray]]:
        if not support_data.active_detectors:
            return 0.0, np.zeros(self.num_detectors, dtype=np.float64)
        if not self._check_coverage(support_data):
            return INF, None

        slack = np.array([weight for _support, weight in support_data.supports], dtype=np.float64)
        y = np.zeros(self.num_detectors, dtype=np.float64)

        if order_kind == "asc_deg":
            order = sorted(support_data.active_detectors, key=lambda d: (len(support_data.incident[d]), d))
        elif order_kind == "desc_plain":
            _plain_value, y_plain = self.heuristic_plain(support_data)
            if y_plain is None:
                return INF, None
            order = sorted(support_data.active_detectors, key=lambda d: (y_plain[d], d), reverse=True)
        else:
            raise ValueError(f"Unknown order_kind={order_kind!r}")

        for d in order:
            value = min(slack[i] for i in support_data.incident[d])
            if value < 0:
                value = 0.0
            y[d] = value
            for i in support_data.incident[d]:
                slack[i] -= value
        return float(y[support_data.active_detectors].sum()), y

    def heuristic_plain_sweep(self, support_data: SupportData) -> Tuple[float, Optional[np.ndarray]]:
        plain_value, y = self.heuristic_plain(support_data)
        if y is None:
            return INF, None
        order = sorted(support_data.active_detectors, key=lambda d: (y[d], d), reverse=True)
        for d in order:
            max_feasible = min(
                weight - sum(y[dd] for dd in support if dd != d)
                for support, weight in support_data.supports
                if d in support
            )
            if max_feasible > y[d]:
                y[d] = max_feasible
        return float(y[support_data.active_detectors].sum()), y

    def heuristic_exact_lp(self, support_data: SupportData) -> Tuple[float, Optional[np.ndarray]]:
        active = support_data.active_detectors
        if not active:
            return 0.0, np.zeros(self.num_detectors, dtype=np.float64)
        if not self._check_coverage(support_data):
            return INF, None

        detector_index = {d: i for i, d in enumerate(active)}
        uf = UnionFind(len(active))
        for support, _weight in support_data.supports:
            if len(support) > 1:
                a = detector_index[support[0]]
                for d in support[1:]:
                    uf.union(a, detector_index[d])

        components: Dict[int, List[int]] = defaultdict(list)
        for d in active:
            components[uf.find(detector_index[d])].append(d)

        y = np.zeros(self.num_detectors, dtype=np.float64)
        total = 0.0
        for component in components.values():
            component_set = set(component)
            local = {d: i for i, d in enumerate(sorted(component))}
            component_supports: List[Tuple[Tuple[int, ...], float]] = []
            for support, weight in support_data.supports:
                if support[0] in component_set:
                    component_supports.append((tuple(local[d] for d in support), weight))

            rows: List[int] = []
            cols: List[int] = []
            data: List[float] = []
            rhs: List[float] = []
            for r, (support, weight) in enumerate(component_supports):
                rhs.append(weight)
                for c in support:
                    rows.append(r)
                    cols.append(c)
                    data.append(1.0)

            a_ub = csr_matrix(
                (data, (rows, cols)),
                shape=(len(component_supports), len(component)),
                dtype=np.float64,
            )
            self.lp_calls += 1
            result = linprog(
                c=-np.ones(len(component), dtype=np.float64),
                A_ub=a_ub,
                b_ub=np.array(rhs, dtype=np.float64),
                bounds=[(0.0, None)] * len(component),
                method="highs",
            )
            if not result.success:
                return INF, None
            total += -float(result.fun)
            for d, value in zip(sorted(component), result.x):
                y[d] = float(value)
        return float(total), y


    def _reachable_available_components(
        self,
        dets: np.ndarray,
        available_errs: np.ndarray,
    ) -> List[Tuple[List[int], List[int], List[int]]]:
        active = sorted(map(int, np.flatnonzero(dets)))
        if not active:
            return []

        det_visited = np.zeros(self.num_detectors, dtype=bool)
        err_visited = np.zeros(self.num_errors, dtype=bool)
        components: List[Tuple[List[int], List[int], List[int]]] = []

        for seed in active:
            if det_visited[seed]:
                continue
            det_visited[seed] = True
            queue: deque[int] = deque([seed])
            component_dets: List[int] = []
            component_errs: List[int] = []
            while queue:
                d = queue.popleft()
                component_dets.append(d)
                for ei in self.d2e[d]:
                    ei = int(ei)
                    if not available_errs[ei] or err_visited[ei]:
                        continue
                    err_visited[ei] = True
                    component_errs.append(ei)
                    for dd in self.error_detectors[ei]:
                        dd = int(dd)
                        if not det_visited[dd]:
                            det_visited[dd] = True
                            queue.append(dd)

            component_active = [d for d in component_dets if dets[d]]
            if not component_active:
                continue
            component_inactive = [d for d in component_dets if not dets[d]]
            components.append((component_active, component_inactive, component_errs))

        return components

    def _solve_component_exact_lp_plus_inactive(
        self,
        component_active: Sequence[int],
        component_inactive: Sequence[int],
        component_errors: Sequence[int],
    ) -> float:
        if not component_active:
            return 0.0
        if not component_errors:
            return INF

        local_errors = list(component_errors)
        det_to_local_errors: Dict[int, List[int]] = defaultdict(list)
        for local_ei, ei in enumerate(local_errors):
            for d in self.error_detectors[ei]:
                det_to_local_errors[int(d)].append(local_ei)

        active_set = set(component_active)
        inactive_set = set(component_inactive)
        deg: Dict[int, int] = {
            d: len(det_to_local_errors.get(d, []))
            for d in active_set | inactive_set
        }
        alive = np.ones(len(local_errors), dtype=bool)
        queue: deque[int] = deque(d for d in component_inactive if deg.get(d, 0) == 1)

        while queue:
            d = queue.popleft()
            if deg.get(d, 0) != 1:
                continue
            forced_local = next(
                (local_ei for local_ei in det_to_local_errors.get(d, []) if alive[local_ei]),
                None,
            )
            if forced_local is None:
                deg[d] = 0
                continue
            if not alive[forced_local]:
                continue
            alive[forced_local] = False
            for dd in self.error_detectors[local_errors[forced_local]]:
                dd = int(dd)
                if dd not in deg or deg[dd] <= 0:
                    continue
                deg[dd] -= 1
                if dd in active_set and deg[dd] == 0:
                    return INF
                if dd in inactive_set and deg[dd] == 1:
                    queue.append(dd)

        for d in component_active:
            if deg.get(d, 0) <= 0:
                return INF

        reduced_errors = [ei for local_ei, ei in enumerate(local_errors) if alive[local_ei]]
        if not reduced_errors:
            return INF

        local_error_index = {ei: local_ei for local_ei, ei in enumerate(reduced_errors)}
        det_to_reduced_errors: Dict[int, List[int]] = defaultdict(list)
        for ei in reduced_errors:
            local_ei = local_error_index[ei]
            for d in self.error_detectors[ei]:
                d = int(d)
                if deg.get(d, 0) > 0:
                    det_to_reduced_errors[d].append(local_ei)

        inactive_with_incidence = [d for d in component_inactive if deg.get(d, 0) > 0]
        num_x = len(reduced_errors)
        num_s = len(inactive_with_incidence)
        num_vars = num_x + num_s
        c = np.zeros(num_vars, dtype=np.float64)
        c[:num_x] = self.weights[reduced_errors]

        inactive_col = {d: num_x + i for i, d in enumerate(inactive_with_incidence)}

        ub_rows: List[int] = []
        ub_cols: List[int] = []
        ub_data: List[float] = []
        b_ub: List[float] = []
        ub_r = 0

        for d in component_active:
            incident = det_to_reduced_errors.get(d, [])
            if not incident:
                return INF
            for local_ei in incident:
                ub_rows.append(ub_r)
                ub_cols.append(local_ei)
                ub_data.append(-1.0)
            b_ub.append(-1.0)
            ub_r += 1

        for d in inactive_with_incidence:
            s_col = inactive_col[d]
            for local_ei in det_to_reduced_errors[d]:
                ub_rows.extend([ub_r, ub_r])
                ub_cols.extend([local_ei, s_col])
                ub_data.extend([2.0, -1.0])
                b_ub.append(0.0)
                ub_r += 1

        eq_rows: List[int] = []
        eq_cols: List[int] = []
        eq_data: List[float] = []
        b_eq: List[float] = []
        eq_r = 0

        for d in inactive_with_incidence:
            s_col = inactive_col[d]
            eq_rows.append(eq_r)
            eq_cols.append(s_col)
            eq_data.append(1.0)
            for local_ei in det_to_reduced_errors[d]:
                eq_rows.append(eq_r)
                eq_cols.append(local_ei)
                eq_data.append(-1.0)
            b_eq.append(0.0)
            eq_r += 1

        a_ub = None
        if ub_r > 0:
            a_ub = csr_matrix(
                (ub_data, (ub_rows, ub_cols)),
                shape=(ub_r, num_vars),
                dtype=np.float64,
            )

        a_eq = None
        if eq_r > 0:
            a_eq = csr_matrix(
                (eq_data, (eq_rows, eq_cols)),
                shape=(eq_r, num_vars),
                dtype=np.float64,
            )

        self.lp_calls += 1
        result = linprog(
            c=c,
            A_ub=a_ub,
            b_ub=np.array(b_ub, dtype=np.float64) if b_ub else None,
            A_eq=a_eq,
            b_eq=np.array(b_eq, dtype=np.float64) if b_eq else None,
            bounds=[(0.0, None)] * num_vars,
            method="highs",
        )
        if not result.success or result.fun is None:
            return INF
        return float(result.fun)

    def heuristic_exact_lp_plus_inactive(
        self,
        dets: np.ndarray,
        available_errs: np.ndarray,
    ) -> Tuple[float, Optional[np.ndarray]]:
        if not np.any(dets):
            return 0.0, None
        if not self._has_cover_for_all_active_detectors(dets, available_errs):
            return INF, None

        total = 0.0
        for component_active, component_inactive, component_errors in self._reachable_available_components(
            dets,
            available_errs,
        ):
            component_value = self._solve_component_exact_lp_plus_inactive(
                component_active,
                component_inactive,
                component_errors,
            )
            if math.isinf(component_value):
                return INF, None
            total += component_value
        return float(total), None

    def evaluate_named_heuristic(self, support_data: SupportData, name: str) -> Tuple[float, Optional[np.ndarray]]:
        if name == "plain":
            return self.heuristic_plain(support_data)
        if name == "asc_deg":
            return self.heuristic_saturation_zero(support_data, order_kind="asc_deg")
        if name == "desc_plain":
            return self.heuristic_saturation_zero(support_data, order_kind="desc_plain")
        if name == "plain_sweep":
            return self.heuristic_plain_sweep(support_data)
        if name == "best_of_two":
            v1, y1 = self.heuristic_plain_sweep(support_data)
            v2, y2 = self.heuristic_saturation_zero(support_data, order_kind="asc_deg")
            if v1 >= v2:
                return v1, y1
            return v2, y2
        if name == "best_of_three":
            candidates = [
                self.heuristic_plain_sweep(support_data),
                self.heuristic_saturation_zero(support_data, order_kind="asc_deg"),
                self.heuristic_saturation_zero(support_data, order_kind="desc_plain"),
            ]
            return max(candidates, key=lambda t: t[0])
        if name == "exact_lp":
            return self.heuristic_exact_lp(support_data)
        raise ValueError(f"Unknown heuristic {name!r}")

    def compute_support_based_heuristic(
        self,
        dets: np.ndarray,
        errs: np.ndarray,
        blocked_errs: np.ndarray,
        *,
        name: Optional[str] = None,
    ) -> Tuple[float, Optional[np.ndarray]]:
        self.heuristic_calls += 1
        available = self._available_errors(errs, blocked_errs)
        heuristic_name = name or self.heuristic_name
        if heuristic_name == "exact_lp_plus_inactive":
            return self.heuristic_exact_lp_plus_inactive(dets, available)
        support_data = self.build_support_data(dets, available)
        return self.evaluate_named_heuristic(support_data, heuristic_name)

    def project_child_y(
        self,
        parent_state: SearchState,
        child_dets: np.ndarray,
        child_errs: np.ndarray,
        child_blocked_errs: np.ndarray,
        child_det_counts: np.ndarray,
        flipped_detectors: Sequence[int],
    ) -> Tuple[float, Optional[np.ndarray], str]:
        if parent_state.y_prices is None:
            raise AssertionError("Expected a stored feasible y vector before projecting to a child.")

        self.heuristic_calls += 1
        self.projection_heuristic_calls += 1
        available = self._available_errors(child_errs, child_blocked_errs)
        if not self._has_cover_for_all_active_detectors(child_dets, available):
            return INF, None, "projected"

        y_projected = np.zeros(self.num_detectors, dtype=np.float64)
        keep = parent_state.dets & child_dets
        y_projected[keep] = parent_state.y_prices[keep]
        projected_value = float(y_projected[np.flatnonzero(child_dets)].sum())
        best_value = projected_value
        best_y = y_projected
        best_source = "projected"

        if self.projection_combine_max_plain:
            plain_value, plain_y = self.plain_detcost_from_counts(child_dets, available, child_det_counts)
            if plain_y is None:
                return INF, None, "plain"
            if plain_value > best_value + HEURISTIC_EPS:
                best_value = plain_value
                best_y = plain_y
                best_source = "plain"

        return best_value, best_y, best_source

    def report_root_heuristics(self, dets: np.ndarray, errs: np.ndarray, blocked_errs: np.ndarray) -> List[Tuple[str, float]]:
        available = self._available_errors(errs, blocked_errs)
        support_data = self.build_support_data(dets, available)
        names = [
            "plain",
            "asc_deg",
            "desc_plain",
            "plain_sweep",
            "best_of_two",
            "best_of_three",
            "exact_lp",
            "exact_lp_plus_inactive",
        ]
        out: List[Tuple[str, float]] = []
        saved_lp_calls = self.lp_calls
        for name in names:
            if name == "exact_lp_plus_inactive":
                value, _ = self.heuristic_exact_lp_plus_inactive(dets, available)
            else:
                value, _ = self.evaluate_named_heuristic(support_data, name)
            out.append((name, value))
        self.lp_calls = saved_lp_calls
        return out

    def _maybe_refine_node(self, state: SearchState) -> Tuple[SearchState, bool]:
        if state.refined or self.heuristic_name == "plain" or not self.lazy_reinsert_heuristics:
            return state, False

        previous_source = state.h_source
        projectable = self.heuristic_has_projectable_prices(self.heuristic_name)
        self.refinement_calls += 1
        new_value, new_y = self.compute_support_based_heuristic(
            state.dets,
            state.errs,
            state.blocked_errs,
            name=self.heuristic_name,
        )
        if math.isinf(new_value):
            if previous_source == "projected":
                self.projected_nodes_refined += 1
            if self.verbose_search:
                print(
                    f"  refine approx_h={state.h_cost:.6f} new_h=INF delta=INF reinserted=False discarded=True"
                )
            state.h_cost = INF
            state.h_source = "refined"
            if projectable:
                state.y_prices = None
            state.refined = True
            return state, True
        if projectable and new_y is None:
            raise AssertionError(f"Expected projectable y-prices from heuristic {self.heuristic_name!r}.")

        delta = new_value - state.h_cost
        self.total_refinement_gain += max(0.0, delta)
        self.max_refinement_gain = max(self.max_refinement_gain, max(0.0, delta))

        if self.heuristic_name in {"exact_lp", "exact_lp_plus_inactive"} and new_value + 1e-7 < state.h_cost:
            raise AssertionError(
                f"Exact LP refinement {new_value} below stored projected value {state.h_cost}."
            )

        if new_value > state.h_cost + HEURISTIC_EPS:
            if previous_source == "projected":
                self.projected_nodes_refined += 1
            state.h_cost = new_value
            state.h_source = "refined"
            if projectable:
                state.y_prices = new_y
            state.refined = True
            self.reinserts += 1
            if self.verbose_search:
                print(
                    f"  refine approx_h={state.h_cost - delta:.6f} new_h={new_value:.6f} delta={delta:.6f} reinserted=True discarded=False"
                )
            return state, True

        # Non-improving recomputation: keep the existing projectable feasible point unless the
        # selected heuristic returned a fresh one that can still be projected to children.
        if previous_source == "projected":
            self.projected_nodes_refined += 1
        if projectable and abs(new_value - state.h_cost) <= HEURISTIC_EPS and new_y is not None:
            state.y_prices = new_y
        state.refined = True
        if self.verbose_search:
            new_text = "INF" if math.isinf(new_value) else f"{new_value:.6f}"
            print(
                f"  refine approx_h={state.h_cost:.6f} new_h={new_text} delta={delta:.6f} reinserted=False discarded=False"
            )
        return state, False

    def decode(self, shot_dets: np.ndarray, det_beam: float = INF) -> DecodeResult:
        start_time = time.perf_counter()
        self.reset_stats()

        dets0 = np.array(shot_dets, dtype=bool, copy=True)
        errs0 = np.zeros(self.num_errors, dtype=bool)
        blocked0 = np.zeros(self.num_errors, dtype=bool)
        det_counts0 = np.zeros(self.num_errors, dtype=np.uint16)
        for d in np.flatnonzero(dets0):
            for ei in self.d2e[int(d)]:
                det_counts0[int(ei)] += 1

        root_h, root_y = self.plain_detcost_from_counts(dets0, self._available_errors(errs0, blocked0), det_counts0)
        if root_y is None or math.isinf(root_h):
            return DecodeResult(
                success=False,
                errs=errs0,
                residual_dets=dets0,
                cost=INF,
                nodes_pushed=1,
                nodes_popped=0,
                max_queue_size=1,
                heuristic_calls=self.heuristic_calls,
                plain_heuristic_calls=self.plain_heuristic_calls,
                projection_heuristic_calls=self.projection_heuristic_calls,
                refinement_calls=self.refinement_calls,
                lp_calls=self.lp_calls,
                reinserts=self.reinserts,
                projected_nodes_generated=self.projected_nodes_generated,
                projected_nodes_refined=self.projected_nodes_refined,
                projected_nodes_unrefined_at_finish=self.projected_nodes_generated - self.projected_nodes_refined,
                total_refinement_gain=self.total_refinement_gain,
                max_refinement_gain=self.max_refinement_gain,
                elapsed_seconds=time.perf_counter() - start_time,
            )

        root_refined = (self.heuristic_name == "plain") or (not self.lazy_reinsert_heuristics)
        if root_refined and self.heuristic_name != "plain":
            # Eager mode: use the selected heuristic immediately.
            eager_h, eager_y = self.compute_support_based_heuristic(dets0, errs0, blocked0, name=self.heuristic_name)
            if math.isinf(eager_h):
                return DecodeResult(
                    success=False,
                    errs=errs0,
                    residual_dets=dets0,
                    cost=INF,
                    nodes_pushed=1,
                    nodes_popped=0,
                    max_queue_size=1,
                    heuristic_calls=self.heuristic_calls,
                    plain_heuristic_calls=self.plain_heuristic_calls,
                    projection_heuristic_calls=self.projection_heuristic_calls,
                    refinement_calls=self.refinement_calls,
                    lp_calls=self.lp_calls,
                    reinserts=self.reinserts,
                    projected_nodes_generated=self.projected_nodes_generated,
                    projected_nodes_refined=self.projected_nodes_refined,
                    projected_nodes_unrefined_at_finish=self.projected_nodes_generated - self.projected_nodes_refined,
                    total_refinement_gain=self.total_refinement_gain,
                    max_refinement_gain=self.max_refinement_gain,
                    elapsed_seconds=time.perf_counter() - start_time,
                )
            if self.heuristic_has_projectable_prices(self.heuristic_name):
                if eager_y is None:
                    raise AssertionError(f"Expected projectable y-prices from heuristic {self.heuristic_name!r}.")
                root_y = eager_y
            root_h = eager_h

        root_state = SearchState(
            errs=errs0,
            blocked_errs=blocked0,
            dets=dets0,
            det_counts=det_counts0,
            g_cost=0.0,
            h_cost=root_h,
            h_source="plain" if not root_refined else ("plain" if self.heuristic_name == "plain" else "refined"),
            refined=root_refined,
            y_prices=root_y,
        )

        heap: List[Tuple[float, int, int, SearchState]] = []
        counter = 0
        heapq.heappush(heap, (root_state.g_cost + root_state.h_cost, int(dets0.sum()), counter, root_state))
        counter += 1
        nodes_pushed = 1
        nodes_popped = 0
        max_queue_size = 1
        min_num_dets = int(dets0.sum())

        while heap:
            max_queue_size = max(max_queue_size, len(heap))
            f_cost, num_dets, _entry_id, state = heapq.heappop(heap)
            nodes_popped += 1
            max_num_dets = min_num_dets + det_beam
            if num_dets > max_num_dets:
                continue
            if num_dets < min_num_dets:
                min_num_dets = num_dets
                max_num_dets = min_num_dets + det_beam

            if self.verbose_search:
                projected_unrefined = self.projected_nodes_generated - self.projected_nodes_refined
                print(
                    f"len(heap)={len(heap)} nodes_pushed={nodes_pushed} nodes_popped={nodes_popped} "
                    f"lp_calls={self.lp_calls} reinserts={self.reinserts} proj_generated={self.projected_nodes_generated} "
                    f"proj_refined={self.projected_nodes_refined} proj_unrefined_so_far={projected_unrefined} "
                    f"num_dets={num_dets} max_num_dets={max_num_dets} f={f_cost:.6f} g={state.g_cost:.6f} "
                    f"h={state.h_cost:.6f} h_source={state.h_source} refined={state.refined}"
                )

            if num_dets == 0:
                return DecodeResult(
                    success=True,
                    errs=state.errs,
                    residual_dets=state.dets,
                    cost=state.g_cost,
                    nodes_pushed=nodes_pushed,
                    nodes_popped=nodes_popped,
                    max_queue_size=max_queue_size,
                    heuristic_calls=self.heuristic_calls,
                    plain_heuristic_calls=self.plain_heuristic_calls,
                    projection_heuristic_calls=self.projection_heuristic_calls,
                    refinement_calls=self.refinement_calls,
                    lp_calls=self.lp_calls,
                    reinserts=self.reinserts,
                    projected_nodes_generated=self.projected_nodes_generated,
                    projected_nodes_refined=self.projected_nodes_refined,
                    projected_nodes_unrefined_at_finish=self.projected_nodes_generated - self.projected_nodes_refined,
                    total_refinement_gain=self.total_refinement_gain,
                    max_refinement_gain=self.max_refinement_gain,
                    elapsed_seconds=time.perf_counter() - start_time,
                )

            state, should_reinsert = self._maybe_refine_node(state)
            if should_reinsert:
                if state.y_prices is None or math.isinf(state.h_cost):
                    if state.h_source == "projected":
                        self.projected_nodes_refined += 1
                    continue
                if state.h_source != "plain":
                    heapq.heappush(heap, (state.g_cost + state.h_cost, num_dets, counter, state))
                    counter += 1
                    continue

            min_det = int(np.flatnonzero(state.dets)[0])
            prefix_blocked = state.blocked_errs.copy()
            children_generated = 0
            children_beam_pruned = 0
            children_infeasible = 0
            children_projected = 0

            for ei in self.d2e[min_det]:
                ei = int(ei)
                prefix_blocked[ei] = True
                if state.errs[ei] or state.blocked_errs[ei]:
                    continue

                child_errs = state.errs.copy()
                child_errs[ei] = True
                child_blocked = prefix_blocked.copy()
                child_dets = state.dets.copy()
                child_det_counts = state.det_counts.copy()
                for d in self.error_detectors[ei]:
                    d = int(d)
                    if child_dets[d]:
                        child_dets[d] = False
                        for oei in self.d2e[d]:
                            child_det_counts[int(oei)] -= 1
                    else:
                        child_dets[d] = True
                        for oei in self.d2e[d]:
                            child_det_counts[int(oei)] += 1

                child_num_dets = int(child_dets.sum())
                if child_num_dets > max_num_dets:
                    children_beam_pruned += 1
                    continue

                child_g = state.g_cost + float(self.weights[ei])
                if self.heuristic_name == "plain" or (not self.lazy_reinsert_heuristics):
                    child_h, child_y = self.compute_support_based_heuristic(
                        child_dets, child_errs, child_blocked, name=self.heuristic_name
                    )
                    child_source = "plain" if self.heuristic_name == "plain" else "refined"
                    child_refined = True
                else:
                    if state.y_prices is None:
                        raise AssertionError("Expected parent feasible y-prices before projecting to child.")
                    child_h, child_y, child_source = self.project_child_y(
                        state,
                        child_dets,
                        child_errs,
                        child_blocked,
                        child_det_counts,
                        self.error_detectors[ei],
                    )
                    self.projected_nodes_generated += 1
                    children_projected += 1
                    child_refined = False

                if math.isinf(child_h):
                    children_infeasible += 1
                    continue
                if (
                    child_refined
                    and self.heuristic_has_projectable_prices(self.heuristic_name)
                    and child_y is None
                ):
                    raise AssertionError(f"Expected projectable y-prices from heuristic {self.heuristic_name!r}.")

                child_state = SearchState(
                    errs=child_errs,
                    blocked_errs=child_blocked,
                    dets=child_dets,
                    det_counts=child_det_counts,
                    g_cost=child_g,
                    h_cost=child_h,
                    h_source=child_source,
                    refined=child_refined,
                    y_prices=child_y,
                )
                heapq.heappush(heap, (child_g + child_h, child_num_dets, counter, child_state))
                counter += 1
                nodes_pushed += 1
                children_generated += 1

            if self.verbose_search:
                projected_unrefined = self.projected_nodes_generated - self.projected_nodes_refined
                print(
                    f"  expanded children_generated={children_generated} children_projected={children_projected} "
                    f"beam_pruned={children_beam_pruned} infeasible={children_infeasible} "
                    f"lp_calls={self.lp_calls} proj_unrefined_so_far={projected_unrefined}"
                )

        return DecodeResult(
            success=False,
            errs=np.zeros(self.num_errors, dtype=bool),
            residual_dets=np.array(shot_dets, dtype=bool, copy=True),
            cost=INF,
            nodes_pushed=nodes_pushed,
            nodes_popped=nodes_popped,
            max_queue_size=max_queue_size,
            heuristic_calls=self.heuristic_calls,
            plain_heuristic_calls=self.plain_heuristic_calls,
            projection_heuristic_calls=self.projection_heuristic_calls,
            refinement_calls=self.refinement_calls,
            lp_calls=self.lp_calls,
            reinserts=self.reinserts,
            projected_nodes_generated=self.projected_nodes_generated,
            projected_nodes_refined=self.projected_nodes_refined,
            projected_nodes_unrefined_at_finish=self.projected_nodes_generated - self.projected_nodes_refined,
            total_refinement_gain=self.total_refinement_gain,
            max_refinement_gain=self.max_refinement_gain,
            elapsed_seconds=time.perf_counter() - start_time,
        )

    def cost_from_errs(self, errs: np.ndarray) -> float:
        return float(self.weights[errs].sum())

    def detectors_from_errs(self, errs: np.ndarray) -> np.ndarray:
        dets = np.zeros(self.num_detectors, dtype=bool)
        for ei in np.flatnonzero(errs):
            for d in self.error_detectors[int(ei)]:
                dets[d] ^= True
        return dets

    def observables_from_errs(self, errs: np.ndarray) -> np.ndarray:
        parity: Dict[int, bool] = {}
        for ei in np.flatnonzero(errs):
            for obs in self.error_observables[int(ei)]:
                parity[int(obs)] = not parity.get(int(obs), False)
        return np.array(sorted(obs for obs, bit in parity.items() if bit), dtype=np.int32)


def sample_detections_and_observables(
    circuit: stim.Circuit,
    *,
    num_shots: int,
    seed: int,
    num_detectors: int,
    num_observables: int,
) -> Tuple[np.ndarray, np.ndarray]:
    sampler = circuit.compile_detector_sampler(seed=seed)
    dets_packed, obs_packed = sampler.sample(
        shots=num_shots,
        separate_observables=True,
        bit_packed=True,
    )
    dets_unpacked = np.unpackbits(
        dets_packed,
        bitorder="little",
        axis=1,
        count=num_detectors,
    )
    obs_unpacked = np.unpackbits(
        obs_packed,
        bitorder="little",
        axis=1,
        count=num_observables,
    )
    return dets_unpacked.astype(bool), obs_unpacked.astype(bool)


def parse_det_beam(text: str) -> float:
    lowered = text.strip().lower()
    if lowered in {"inf", "infinity", "none"}:
        return INF
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("det-beam must be non-negative or 'inf'.")
    return float(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prototype A* decoder for Stim circuits using greedy singleton-budget heuristics."
        )
    )
    parser.add_argument("--circuit", type=Path, required=True, help="Path to a .stim circuit file.")
    parser.add_argument(
        "--dets",
        type=str,
        default=None,
        help="String of shot dets (e.g., 'shot D0 D1 L2') to parse instead of sampling.",
    )
    parser.add_argument(
        "--sample-num-shots",
        type=int,
        default=100,
        help="Number of shots to sample from Stim before selecting --shot (default: 100).",
    )
    parser.add_argument(
        "--shot",
        type=int,
        default=0,
        help="Index of the sampled shot to decode (default: 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=27123839530,
        help="Stim sampler seed (default: 27123839530).",
    )
    parser.add_argument(
        "--det-beam",
        type=parse_det_beam,
        default=INF,
        help="Beam cutoff on the residual detector count; use 'inf' for none.",
    )
    parser.add_argument(
        "--heuristic",
        choices=[
            "plain",
            "asc_deg",
            "desc_plain",
            "plain_sweep",
            "best_of_two",
            "best_of_three",
            "exact_lp",
            "exact_lp_plus_inactive",
        ],
        default="best_of_two",
        help="Lower-bound heuristic to use during A* search (default: best_of_two).",
    )
    parser.add_argument(
        "--lazy-reinsert-heuristics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For non-plain heuristics, seed nodes with plain detcost, refine on pop, and reinsert when the selected "
            "heuristic improves the key (default: enabled)."
        ),
    )
    parser.add_argument(
        "--projection-combine-max-plain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When projecting parent y-prices to a child, take max(projected, plain detcost) (default: enabled).",
    )
    parser.add_argument(
        "--merge-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge indistinguishable DEM errors before decoding (default: enabled).",
    )
    parser.add_argument(
        "--respect-blocked-errors-in-heuristic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude precedence-blocked errors when forming the lower bound (default: enabled).",
    )
    parser.add_argument(
        "--report-all-root-heuristics",
        action="store_true",
        help="Print all root-node heuristic values, including exact_lp and exact_lp_plus_inactive, for the selected shot.",
    )
    parser.add_argument(
        "--skip-decode",
        action="store_true",
        help="Only report root heuristics; do not run A* search.",
    )
    parser.add_argument(
        "--show-shot-detectors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print the selected shot's active detector IDs (default: enabled).",
    )
    parser.add_argument(
        "--show-error-indices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print the decoded merged-error indices when decoding succeeds (default: enabled).",
    )
    parser.add_argument(
        "--verbose-search",
        action="store_true",
        help="Print per-node search diagnostics.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.sample_num_shots <= 0:
        parser.error("--sample-num-shots must be positive.")
    if args.shot < 0:
        parser.error("--shot must be non-negative.")
    if args.shot >= args.sample_num_shots:
        parser.error("--shot must be smaller than --sample-num-shots.")

    circuit = stim.Circuit.from_file(str(args.circuit))
    dem = circuit.detector_error_model(decompose_errors=False)
    errors = merged_errors_from_dem(dem) if args.merge_errors else list(iter_dem_errors_from_dem(dem))

    if args.dets is not None:
        shot_dets = np.zeros(dem.num_detectors, dtype=bool)
        shot_obs = np.zeros(dem.num_observables, dtype=bool)
        for token in args.dets.split():
            if token == "shot":
                continue
            if token.startswith("D") and token[1:].isdigit():
                d_idx = int(token[1:])
                if d_idx < dem.num_detectors:
                    shot_dets[d_idx] = True
            elif token.startswith("L") and token[1:].isdigit():
                l_idx = int(token[1:])
                if l_idx < dem.num_observables:
                    shot_obs[l_idx] = True
    else:
        dets, obs = sample_detections_and_observables(
            circuit,
            num_shots=args.sample_num_shots,
            seed=args.seed,
            num_detectors=dem.num_detectors,
            num_observables=dem.num_observables,
        )
        shot_dets = dets[args.shot]
        shot_obs = obs[args.shot]

    decoder = GreedySingletonHeuristicDecoder(
        errors,
        num_detectors=dem.num_detectors,
        num_observables=dem.num_observables,
        heuristic=args.heuristic,
        respect_blocked_errors_in_heuristic=args.respect_blocked_errors_in_heuristic,
        lazy_reinsert_heuristics=args.lazy_reinsert_heuristics,
        projection_combine_max_plain=args.projection_combine_max_plain,
        verbose_search=args.verbose_search,
    )

    print(f"circuit = {args.circuit}")
    print(f"heuristic = {args.heuristic}")
    print(f"mode = {decoder.mode_name}")
    print(f"sample_num_shots = {args.sample_num_shots}")
    print(f"shot = {args.shot}")
    print(f"num_errors = {decoder.num_errors}")
    print(f"num_detectors = {decoder.num_detectors}")
    print(f"num_observables = {decoder.num_observables}")
    print(f"det_beam = {args.det_beam}")
    print(f"merge_errors = {args.merge_errors}")
    print(f"respect_blocked_errors_in_heuristic = {args.respect_blocked_errors_in_heuristic}")
    print(f"lazy_reinsert_heuristics = {args.lazy_reinsert_heuristics}")
    print(f"projection_combine_max_plain = {args.projection_combine_max_plain}")

    if args.show_shot_detectors:
        active_dets = np.flatnonzero(shot_dets)
        print("shot_detectors =", " ".join(f"D{d}" for d in active_dets))

    if args.report_all_root_heuristics:
        root_errs = np.zeros(decoder.num_errors, dtype=bool)
        root_blocked = np.zeros(decoder.num_errors, dtype=bool)
        report = decoder.report_root_heuristics(shot_dets, root_errs, root_blocked)
        exact = next((v for k, v in report if k == "exact_lp"), None)
        print("root_heuristics:")
        for name, value in report:
            if exact is not None and not math.isinf(exact) and exact > 0:
                ratio = value / exact if not math.isinf(value) else INF
                ratio_text = "INF" if math.isinf(ratio) else f"{ratio:.6f}"
            else:
                ratio_text = "n/a"
            value_text = "INF" if math.isinf(value) else f"{value:.12f}"
            print(f"  {name:>24s}  value={value_text}  ratio_to_exact={ratio_text}")

    if args.skip_decode:
        return 0

    result = decoder.decode(shot_dets, det_beam=args.det_beam)
    print(f"success = {result.success}")
    print(f"nodes_pushed = {result.nodes_pushed}")
    print(f"nodes_popped = {result.nodes_popped}")
    print(f"max_queue_size = {result.max_queue_size}")
    print(f"heuristic_calls = {result.heuristic_calls}")
    print(f"plain_heuristic_calls = {result.plain_heuristic_calls}")
    print(f"projection_heuristic_calls = {result.projection_heuristic_calls}")
    print(f"refinement_calls = {result.refinement_calls}")
    print(f"lp_calls = {result.lp_calls}")
    print(f"reinserts = {result.reinserts}")
    print(f"projected_nodes_generated = {result.projected_nodes_generated}")
    print(f"projected_nodes_refined = {result.projected_nodes_refined}")
    print(f"projected_nodes_unrefined_at_finish = {result.projected_nodes_unrefined_at_finish}")
    print(f"total_refinement_gain = {result.total_refinement_gain:.6f}")
    print(f"max_refinement_gain = {result.max_refinement_gain:.6f}")
    print(f"elapsed_seconds = {result.elapsed_seconds:.6f}")

    if not result.success:
        print("decode failed")
        return 1

    if args.show_error_indices:
        print("decoded_error_indices =", " ".join(map(str, np.flatnonzero(result.errs).tolist())))

    reproduced_dets = decoder.detectors_from_errs(result.errs)
    if not np.array_equal(reproduced_dets, shot_dets):
        raise AssertionError("Decoded errors do not reproduce the sampled detection events.")

    decoded_cost = decoder.cost_from_errs(result.errs)
    predicted_obs = decoder.observables_from_errs(result.errs)
    sampled_obs = np.flatnonzero(shot_obs)

    print(f"num_decoded_errors = {int(result.errs.sum())}")
    print(f"decoded_cost = {decoded_cost:.12f}")
    print("predicted_observables =", " ".join(f"L{o}" for o in predicted_obs.tolist()))
    print("sampled_observables =", " ".join(f"L{o}" for o in sampled_obs.tolist()))
    print(f"observables_match = {bool(np.array_equal(predicted_obs, sampled_obs))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
