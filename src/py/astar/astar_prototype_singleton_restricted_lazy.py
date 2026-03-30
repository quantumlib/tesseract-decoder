#!/usr/bin/env python3
"""Prototype A* decoder with lazy optimal-singleton refinement.

This script is intentionally packaged similarly to astar_prototype_subset_detcost_lazy.py,
but specialized to the singleton LP. It offers three modes:

  --opt-singleton-detcost-mode plain
      Use plain detcost only.

  --opt-singleton-detcost-mode full
      Lazy exact singleton LP on pop, with projected child lower bounds.

  --opt-singleton-detcost-mode restricted
      Lazy exact singleton LP on pop, but solved by a restricted-master /
      separation loop seeded from the parent tight set.

Two "outside the box" ideas are built in:

  1) Parent-primal projection.
     If y_parent is feasible for the parent singleton LP, then setting the child
     detector prices to y_parent on detectors that remain active and 0 on newly
     active detectors is automatically feasible for the child singleton LP.
     That gives a cheap admissible child lower bound.

  2) Local residual projection LP.
     On top of the projected parent prices, we can re-optimize a tiny local LP
     on either the newly active detectors or the neighborhood touched by the
     changed detector set, while keeping the outside detector prices fixed.
     This is still admissible because it is a feasible child primal solution.
"""

from __future__ import annotations

import argparse
import heapq
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
    error_observables: List[Tuple[int, ...]]


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
    lp_solution: Optional["SingletonLPSolution"] = None
    warm_start_solution: Optional["SingletonLPSolution"] = None
    changed_detectors_from_parent: Tuple[int, ...] = ()


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
    projection_local_lp_calls: int
    projection_local_lp_seconds: float
    restricted_total_rounds: int
    restricted_total_added_supports: int
    restricted_total_fallbacks: int
    full_check_calls: int
    full_check_max_abs_delta: float
    elapsed_seconds: float
    heuristic_name: str


@dataclass
class DecodeResult:
    activated_errors: Tuple[int, ...]
    path_cost: float
    stats: DecodeStats


@dataclass(frozen=True)
class RestrictedMasterConfig:
    add_policy: str = "topk"  # one | topk | all
    add_top_k: int = 3
    violation_tol: float = 1e-9
    tight_tol: float = 1e-8
    prune_slack: bool = True
    prune_tol: float = 1e-8
    seed_normalized_global_k: int = 0
    seed_normalized_touching_changed_k: int = 2
    max_rounds: int = 50
    fallback_full: bool = True
    full_check_every: int = 0


@dataclass
class SingletonLPSolution:
    value: float
    active_detectors: Tuple[int, ...]
    y_by_detector: Dict[int, float]
    tight_supports: Tuple[Tuple[int, ...], ...]
    num_components: int
    num_variables: int
    num_constraints: int
    num_selected_constraints: int
    num_rounds: int
    solve_mode: str


@dataclass
class SingletonLPSolverStats:
    lp_calls: int = 0
    lp_total_seconds: float = 0.0
    projection_local_lp_calls: int = 0
    projection_local_lp_seconds: float = 0.0
    restricted_total_rounds: int = 0
    restricted_total_added_supports: int = 0
    restricted_total_fallbacks: int = 0
    full_check_calls: int = 0
    full_check_max_abs_delta: float = 0.0


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
    def __init__(self, path: Path, *, every: int = 1, top_k: int = 12) -> None:
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


class SingletonLPHeuristic:
    def __init__(
        self,
        data: DecoderData,
        *,
        exact_mode: str,
        projection_mode: str,
        projection_combine_max_plain: bool,
        restricted_config: RestrictedMasterConfig,
        logger: Optional[LPLogger] = None,
    ) -> None:
        if exact_mode not in {"full", "restricted"}:
            raise ValueError(f"Unsupported exact_mode: {exact_mode}")
        if projection_mode not in {"plain", "parent_y", "new_only", "changed_neighborhood"}:
            raise ValueError(f"Unsupported projection_mode: {projection_mode}")
        self.data = data
        self.exact_mode = exact_mode
        self.projection_mode = projection_mode
        self.projection_combine_max_plain = projection_combine_max_plain
        self.restricted_config = restricted_config
        self.logger = logger
        self.stats = SingletonLPSolverStats()
        self.exact_solve_calls = 0

    def reset_stats(self) -> None:
        self.stats = SingletonLPSolverStats()
        self.exact_solve_calls = 0

    def solve_exact(
        self,
        active_detectors: np.ndarray,
        blocked_errors: np.ndarray,
        active_detector_counts: np.ndarray,
        *,
        warm_start_solution: Optional[SingletonLPSolution],
        changed_detectors: Tuple[int, ...],
    ) -> Tuple[SingletonLPSolution, Dict[str, Any]]:
        self.exact_solve_calls += 1
        t0 = time.perf_counter()
        active_detector_ids = tuple(int(d) for d in np.flatnonzero(active_detectors))
        support_costs = build_active_support_costs(
            data=self.data,
            active_detectors=active_detectors,
            blocked_errors=blocked_errors,
            active_detector_counts=active_detector_counts,
        )

        if not active_detector_ids:
            elapsed = time.perf_counter() - t0
            payload = {
                "solve_mode": self.exact_mode,
                "objective": 0.0,
                "num_components": 0,
                "num_variables": 0,
                "num_constraints": 0,
                "num_selected_constraints": 0,
                "num_rounds": 0,
                "num_supports_total": 0,
                "solve_seconds": elapsed,
                "structurally_infeasible": False,
            }
            return (
                SingletonLPSolution(
                    value=0.0,
                    active_detectors=(),
                    y_by_detector={},
                    tight_supports=(),
                    num_components=0,
                    num_variables=0,
                    num_constraints=0,
                    num_selected_constraints=0,
                    num_rounds=0,
                    solve_mode=self.exact_mode,
                ),
                payload,
            )

        missing_cover = [
            detector
            for detector in active_detector_ids
            if not any(detector in support for support in support_costs)
        ]
        if missing_cover:
            elapsed = time.perf_counter() - t0
            payload = {
                "solve_mode": self.exact_mode,
                "objective": INF,
                "num_components": 0,
                "num_variables": len(active_detector_ids),
                "num_constraints": len(support_costs),
                "num_selected_constraints": 0,
                "num_rounds": 0,
                "num_supports_total": len(support_costs),
                "solve_seconds": elapsed,
                "structurally_infeasible": True,
                "missing_cover_detectors": missing_cover,
            }
            return (
                SingletonLPSolution(
                    value=INF,
                    active_detectors=active_detector_ids,
                    y_by_detector={},
                    tight_supports=(),
                    num_components=0,
                    num_variables=len(active_detector_ids),
                    num_constraints=len(support_costs),
                    num_selected_constraints=0,
                    num_rounds=0,
                    solve_mode=self.exact_mode,
                ),
                payload,
            )

        if self.exact_mode == "full":
            solution, full_payload = self._solve_full_support_lp(
                active_detector_ids=active_detector_ids,
                support_costs=support_costs,
                solve_mode="full",
            )
            elapsed = time.perf_counter() - t0
            payload = dict(full_payload)
            payload.update(
                {
                    "solve_mode": "full",
                    "num_supports_total": len(support_costs),
                    "solve_seconds": elapsed,
                    "structurally_infeasible": False,
                }
            )
            return solution, payload

        solution, payload = self._solve_restricted_exact(
            active_detector_ids=active_detector_ids,
            support_costs=support_costs,
            warm_start_solution=warm_start_solution,
            changed_detectors=changed_detectors,
        )
        payload.update(
            {
                "solve_mode": "restricted",
                "num_supports_total": len(support_costs),
                "structurally_infeasible": False,
            }
        )
        return solution, payload

    def project_to_child(
        self,
        parent_solution: SingletonLPSolution,
        child_active_detectors: np.ndarray,
        child_blocked_errors: np.ndarray,
        child_active_detector_counts: np.ndarray,
        *,
        changed_detectors: Tuple[int, ...],
    ) -> float:
        if self.projection_mode == "plain":
            projected = plain_detcost_heuristic(
                data=self.data,
                active_detectors=child_active_detectors,
                blocked_errors=child_blocked_errors,
                active_detector_counts=child_active_detector_counts,
            )
            return projected

        parent_active_set = set(parent_solution.active_detectors)
        child_active_ids = tuple(int(d) for d in np.flatnonzero(child_active_detectors))
        parent_y = parent_solution.y_by_detector

        # Fixed outside prices inherited from the parent exact primal solution.
        fixed_outside_y: Dict[int, float] = {}
        region_detectors: set[int] = set()
        if self.projection_mode == "parent_y":
            region_detectors = set()
        elif self.projection_mode == "new_only":
            region_detectors = {d for d in child_active_ids if d not in parent_active_set}
        elif self.projection_mode == "changed_neighborhood":
            changed_set = set(changed_detectors)
            region_detectors = {d for d in child_active_ids if d not in parent_active_set}
            for detector in changed_set:
                for error_index in self.data.detector_to_errors[detector]:
                    if child_blocked_errors[error_index]:
                        continue
                    if child_active_detector_counts[error_index] <= 0:
                        continue
                    for other_detector in self.data.error_detectors[error_index]:
                        if child_active_detectors[other_detector]:
                            region_detectors.add(other_detector)
        else:
            raise AssertionError("unreachable projection mode")

        for detector in child_active_ids:
            if detector in region_detectors:
                continue
            if detector in parent_active_set:
                fixed_outside_y[detector] = parent_y.get(detector, 0.0)
            else:
                fixed_outside_y[detector] = 0.0

        projected = sum(fixed_outside_y.values())

        if region_detectors:
            local_gain = self._solve_local_region_projection_lp(
                child_active_detectors=child_active_detectors,
                child_blocked_errors=child_blocked_errors,
                child_active_detector_counts=child_active_detector_counts,
                region_detectors=region_detectors,
                fixed_outside_y=fixed_outside_y,
            )
            if local_gain == INF:
                projected = INF
            else:
                projected += local_gain

        if self.projection_combine_max_plain:
            plain = plain_detcost_heuristic(
                data=self.data,
                active_detectors=child_active_detectors,
                blocked_errors=child_blocked_errors,
                active_detector_counts=child_active_detector_counts,
            )
            projected = max(projected, plain)
        return projected

    def _solve_local_region_projection_lp(
        self,
        *,
        child_active_detectors: np.ndarray,
        child_blocked_errors: np.ndarray,
        child_active_detector_counts: np.ndarray,
        region_detectors: set[int],
        fixed_outside_y: Dict[int, float],
    ) -> float:
        if not region_detectors:
            return 0.0
        t0 = time.perf_counter()
        region_support_costs: Dict[Tuple[int, ...], float] = {}
        for error_index, error_detectors in enumerate(self.data.error_detectors):
            if child_blocked_errors[error_index]:
                continue
            count = int(child_active_detector_counts[error_index])
            if count <= 0:
                continue
            full_support = tuple(d for d in error_detectors if child_active_detectors[d])
            assert len(full_support) == count
            local_support = tuple(d for d in full_support if d in region_detectors)
            if not local_support:
                continue
            fixed = sum(fixed_outside_y.get(d, 0.0) for d in full_support if d not in region_detectors)
            residual = float(self.data.error_costs[error_index]) - fixed
            if residual < -1e-8:
                raise AssertionError(
                    f"Projected parent y is infeasible for child: residual={residual} on error {error_index}."
                )
            residual = max(0.0, residual)
            previous = region_support_costs.get(local_support)
            if previous is None or residual < previous:
                region_support_costs[local_support] = residual

        region_detector_ids = tuple(sorted(region_detectors))
        if any(not any(detector in support for support in region_support_costs) for detector in region_detector_ids):
            # No admissible gain on uncovered region detectors; keep them at zero.
            elapsed = time.perf_counter() - t0
            self.stats.projection_local_lp_calls += 1
            self.stats.projection_local_lp_seconds += elapsed
            return 0.0

        objective, _, _, _, _ = solve_primal_lp_on_supports(
            detector_ids=region_detector_ids,
            support_costs=region_support_costs,
            record_stats=self.stats,
            count_as_main_lp=False,
        )
        elapsed = time.perf_counter() - t0
        self.stats.projection_local_lp_calls += 1
        self.stats.projection_local_lp_seconds += elapsed
        return objective

    def _solve_full_support_lp(
        self,
        *,
        active_detector_ids: Tuple[int, ...],
        support_costs: Dict[Tuple[int, ...], float],
        solve_mode: str,
    ) -> Tuple[SingletonLPSolution, Dict[str, Any]]:
        components = split_support_costs_into_components(
            active_detector_ids=active_detector_ids,
            support_costs=support_costs,
        )
        total_value = 0.0
        total_num_variables = 0
        total_num_constraints = 0
        y_by_detector: Dict[int, float] = {}
        tight_supports: List[Tuple[int, ...]] = []
        for detector_ids, component_support_costs in components:
            value, component_y, component_tight, num_vars, num_constraints = solve_primal_lp_on_supports(
                detector_ids=detector_ids,
                support_costs=component_support_costs,
                record_stats=self.stats,
                count_as_main_lp=True,
            )
            total_value += value
            total_num_variables += num_vars
            total_num_constraints += num_constraints
            y_by_detector.update(component_y)
            tight_supports.extend(component_tight)

        solution = SingletonLPSolution(
            value=total_value,
            active_detectors=active_detector_ids,
            y_by_detector=y_by_detector,
            tight_supports=tuple(sorted(set(tight_supports))),
            num_components=len(components),
            num_variables=total_num_variables,
            num_constraints=total_num_constraints,
            num_selected_constraints=total_num_constraints,
            num_rounds=1,
            solve_mode=solve_mode,
        )
        payload = {
            "objective": total_value,
            "num_components": len(components),
            "num_variables": total_num_variables,
            "num_constraints": total_num_constraints,
            "num_selected_constraints": total_num_constraints,
            "num_rounds": 1,
            "tight_support_count": len(solution.tight_supports),
            "top_tight_supports": [
                {"support": list(support), "cost": float(support_costs[support])}
                for support in sorted(solution.tight_supports, key=lambda s: (len(s), s))[: self.logger.top_k if self.logger else 12]
            ],
        }
        return solution, payload

    def _solve_restricted_exact(
        self,
        *,
        active_detector_ids: Tuple[int, ...],
        support_costs: Dict[Tuple[int, ...], float],
        warm_start_solution: Optional[SingletonLPSolution],
        changed_detectors: Tuple[int, ...],
    ) -> Tuple[SingletonLPSolution, Dict[str, Any]]:
        t0 = time.perf_counter()
        components = split_support_costs_into_components(
            active_detector_ids=active_detector_ids,
            support_costs=support_costs,
        )
        total_value = 0.0
        total_num_variables = 0
        total_num_constraints = 0
        total_num_selected_constraints = 0
        total_rounds = 0
        y_by_detector: Dict[int, float] = {}
        tight_supports: List[Tuple[int, ...]] = []
        component_payloads: List[Dict[str, Any]] = []
        fallbacks_used = 0

        parent_tight_supports = set() if warm_start_solution is None else set(warm_start_solution.tight_supports)
        changed_set = set(changed_detectors)

        for detector_ids, component_support_costs in components:
            component_result, component_payload = self._solve_restricted_component(
                detector_ids=detector_ids,
                support_costs=component_support_costs,
                parent_tight_supports=parent_tight_supports,
                changed_set=changed_set,
            )
            total_value += component_result["value"]
            total_num_variables += len(detector_ids)
            total_num_constraints += len(component_support_costs)
            total_num_selected_constraints += component_result["num_selected_constraints"]
            total_rounds += component_result["num_rounds"]
            y_by_detector.update(component_result["y_by_detector"])
            tight_supports.extend(component_result["tight_supports"])
            component_payloads.append(component_payload)
            if component_result["used_full_fallback"]:
                fallbacks_used += 1

        self.stats.restricted_total_rounds += total_rounds
        self.stats.restricted_total_fallbacks += fallbacks_used

        solution = SingletonLPSolution(
            value=total_value,
            active_detectors=active_detector_ids,
            y_by_detector=y_by_detector,
            tight_supports=tuple(sorted(set(tight_supports))),
            num_components=len(components),
            num_variables=total_num_variables,
            num_constraints=total_num_constraints,
            num_selected_constraints=total_num_selected_constraints,
            num_rounds=total_rounds,
            solve_mode="restricted",
        )

        if self.restricted_config.full_check_every > 0 and self.exact_solve_calls % self.restricted_config.full_check_every == 0:
            self.stats.full_check_calls += 1
            full_solution, _ = self._solve_full_support_lp(
                active_detector_ids=active_detector_ids,
                support_costs=support_costs,
                solve_mode="full_check",
            )
            delta = abs(full_solution.value - solution.value)
            self.stats.full_check_max_abs_delta = max(self.stats.full_check_max_abs_delta, delta)
            if delta > 1e-7:
                raise AssertionError(
                    f"Restricted exact solver mismatch: restricted={solution.value} full={full_solution.value} delta={delta}"
                )

        payload = {
            "objective": total_value,
            "num_components": len(components),
            "num_variables": total_num_variables,
            "num_constraints": total_num_constraints,
            "num_selected_constraints": total_num_selected_constraints,
            "num_rounds": total_rounds,
            "tight_support_count": len(solution.tight_supports),
            "used_full_fallbacks": fallbacks_used,
            "components": component_payloads,
            "solve_seconds": time.perf_counter() - t0,
        }
        return solution, payload

    def _solve_restricted_component(
        self,
        *,
        detector_ids: Tuple[int, ...],
        support_costs: Dict[Tuple[int, ...], float],
        parent_tight_supports: set[Tuple[int, ...]],
        changed_set: set[int],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        cover_support_for_detector: Dict[int, Tuple[int, ...]] = {}
        supports_touching_changed: List[Tuple[int, ...]] = []
        for support, cost in support_costs.items():
            for detector in support:
                previous = cover_support_for_detector.get(detector)
                if previous is None:
                    cover_support_for_detector[detector] = support
                else:
                    prev_key = (support_costs[previous], len(previous), previous)
                    cur_key = (cost, len(support), support)
                    if cur_key < prev_key:
                        cover_support_for_detector[detector] = support
            if changed_set and any(detector in changed_set for detector in support):
                supports_touching_changed.append(support)

        selected_supports: set[Tuple[int, ...]] = set(cover_support_for_detector.values())
        cover_supports = set(selected_supports)
        surviving_parent_tight = {support for support in parent_tight_supports if support in support_costs}
        selected_supports |= surviving_parent_tight

        if self.restricted_config.seed_normalized_global_k > 0:
            cheapest_norm = sorted(
                support_costs,
                key=lambda support: (support_costs[support] / len(support), support_costs[support], len(support), support),
            )[: self.restricted_config.seed_normalized_global_k]
            selected_supports.update(cheapest_norm)

        if self.restricted_config.seed_normalized_touching_changed_k > 0 and supports_touching_changed:
            touching = sorted(
                supports_touching_changed,
                key=lambda support: (support_costs[support] / len(support), support_costs[support], len(support), support),
            )[: self.restricted_config.seed_normalized_touching_changed_k]
            selected_supports.update(touching)

        rounds = 0
        total_added_supports = 0
        used_full_fallback = False
        payload_rounds: List[Dict[str, Any]] = []

        while True:
            rounds += 1
            selected_supports |= cover_supports
            restricted_support_costs = {support: support_costs[support] for support in selected_supports}
            value, y_by_detector, selected_tight_supports, num_vars, num_selected_constraints = solve_primal_lp_on_supports(
                detector_ids=detector_ids,
                support_costs=restricted_support_costs,
                record_stats=self.stats,
                count_as_main_lp=True,
            )
            slacks: Dict[Tuple[int, ...], float] = {}
            violations: List[Tuple[float, Tuple[int, ...]]] = []
            full_tight_supports: List[Tuple[int, ...]] = []
            for support, cost in support_costs.items():
                lhs = sum(y_by_detector.get(detector, 0.0) for detector in support)
                slack = cost - lhs
                slacks[support] = slack
                if slack < -self.restricted_config.violation_tol:
                    violations.append((-slack, support))
                if abs(slack) <= self.restricted_config.tight_tol:
                    full_tight_supports.append(support)

            payload_rounds.append(
                {
                    "round": rounds,
                    "selected_constraints": len(selected_supports),
                    "restricted_tight_count": len(selected_tight_supports),
                    "full_tight_count": len(full_tight_supports),
                    "max_violation": 0.0 if not violations else float(max(v for v, _ in violations)),
                }
            )

            if not violations:
                self.stats.restricted_total_added_supports += total_added_supports
                component_result = {
                    "value": value,
                    "y_by_detector": y_by_detector,
                    "tight_supports": tuple(sorted(full_tight_supports)),
                    "num_selected_constraints": len(selected_supports),
                    "num_rounds": rounds,
                    "used_full_fallback": False,
                }
                component_payload = {
                    "detectors": list(detector_ids),
                    "supports_total": len(support_costs),
                    "initial_seed_count": len(cover_supports | surviving_parent_tight),
                    "final_selected_constraints": len(selected_supports),
                    "rounds": rounds,
                    "used_full_fallback": False,
                    "parent_tight_survivors": len(surviving_parent_tight),
                    "cover_supports": len(cover_supports),
                    "round_summaries": payload_rounds,
                }
                return component_result, component_payload

            if rounds >= self.restricted_config.max_rounds:
                if not self.restricted_config.fallback_full:
                    raise RuntimeError(
                        f"Restricted singleton LP exceeded max rounds={self.restricted_config.max_rounds} without fallback."
                    )
                used_full_fallback = True
                self.stats.restricted_total_added_supports += total_added_supports
                full_value, full_y, full_tight, _, _ = solve_primal_lp_on_supports(
                    detector_ids=detector_ids,
                    support_costs=support_costs,
                    record_stats=self.stats,
                    count_as_main_lp=True,
                )
                component_result = {
                    "value": full_value,
                    "y_by_detector": full_y,
                    "tight_supports": tuple(sorted(full_tight)),
                    "num_selected_constraints": len(support_costs),
                    "num_rounds": rounds,
                    "used_full_fallback": True,
                }
                component_payload = {
                    "detectors": list(detector_ids),
                    "supports_total": len(support_costs),
                    "initial_seed_count": len(cover_supports | surviving_parent_tight),
                    "final_selected_constraints": len(support_costs),
                    "rounds": rounds,
                    "used_full_fallback": True,
                    "parent_tight_survivors": len(surviving_parent_tight),
                    "cover_supports": len(cover_supports),
                    "round_summaries": payload_rounds,
                }
                return component_result, component_payload

            if self.restricted_config.prune_slack:
                selected_supports = {
                    support
                    for support in selected_supports
                    if slacks.get(support, INF) <= self.restricted_config.prune_tol or support in cover_supports
                }

            violations.sort(key=lambda item: (-item[0], support_costs[item[1]], len(item[1]), item[1]))
            if self.restricted_config.add_policy == "one":
                to_add = [violations[0][1]]
            elif self.restricted_config.add_policy == "topk":
                to_add = [support for _, support in violations[: self.restricted_config.add_top_k]]
            elif self.restricted_config.add_policy == "all":
                to_add = [support for _, support in violations]
            else:
                raise ValueError(f"Unsupported add policy: {self.restricted_config.add_policy}")
            new_supports = [support for support in to_add if support not in selected_supports]
            total_added_supports += len(new_supports)
            selected_supports.update(new_supports)


def xor_probability(p0: float, p1: float) -> float:
    return p0 * (1 - p1) + (1 - p0) * p1


def parse_beam(text: str) -> float:
    lowered = text.strip().lower()
    if lowered in {"inf", "+inf", "infinity", "+infinity"}:
        return INF
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("beam must be non-negative or 'inf'")
    return float(value)


def parse_optional_int(text: str) -> Optional[int]:
    lowered = text.strip().lower()
    if lowered in {"none", "inf", "infinity"}:
        return None
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative or 'none'")
    return value


def format_indices(indices: Iterable[int], prefix: str) -> str:
    items = list(indices)
    if not items:
        return "(none)"
    return " ".join(f"{prefix}{i}" for i in items)


def iter_dem_errors(dem: stim.DetectorErrorModel) -> Iterable[MergedError]:
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        probability = float(instruction.args_copy()[0])
        if probability <= 0:
            continue
        if probability >= 0.5:
            raise ValueError("This prototype assumes DEM probabilities are in (0, 0.5).")
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
            raise ValueError("Merged error has probability >= 0.5, giving a non-positive cost.")
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
    for error_index, error in enumerate(errors):
        for detector in error.detectors:
            detector_to_errors[detector].append(error_index)
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
    for detector in np.flatnonzero(active_detectors):
        for error_index in data.detector_to_errors[int(detector)]:
            counts[error_index] += 1
    return counts


def apply_error(
    data: DecoderData,
    active_detectors: np.ndarray,
    active_detector_counts: np.ndarray,
    error_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    next_detectors = active_detectors.copy()
    next_counts = active_detector_counts.copy()
    for detector in data.error_detectors[error_index]:
        if next_detectors[detector]:
            next_detectors[detector] = False
            delta = -1
        else:
            next_detectors[detector] = True
            delta = 1
        for other_error_index in data.detector_to_errors[detector]:
            next_counts[other_error_index] += delta
    return next_detectors, next_counts


def plain_detcost_for_detector(
    data: DecoderData,
    detector: int,
    blocked_errors: np.ndarray,
    active_detector_counts: np.ndarray,
) -> float:
    best = INF
    for error_index in data.detector_to_errors[detector]:
        if blocked_errors[error_index]:
            continue
        count = int(active_detector_counts[error_index])
        assert count > 0
        candidate = float(data.error_costs[error_index]) / count
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
    for detector in np.flatnonzero(active_detectors):
        det_cost = plain_detcost_for_detector(
            data=data,
            detector=int(detector),
            blocked_errors=blocked_errors,
            active_detector_counts=active_detector_counts,
        )
        if det_cost == INF:
            return INF
        total += det_cost
    return total


def build_active_support_costs(
    data: DecoderData,
    active_detectors: np.ndarray,
    blocked_errors: np.ndarray,
    active_detector_counts: np.ndarray,
) -> Dict[Tuple[int, ...], float]:
    support_costs: Dict[Tuple[int, ...], float] = {}
    for error_index, error_detectors in enumerate(data.error_detectors):
        if blocked_errors[error_index]:
            continue
        count = int(active_detector_counts[error_index])
        if count <= 0:
            continue
        support = tuple(detector for detector in error_detectors if active_detectors[detector])
        assert len(support) == count
        cost = float(data.error_costs[error_index])
        previous = support_costs.get(support)
        if previous is None or cost < previous:
            support_costs[support] = cost
    return support_costs


def split_support_costs_into_components(
    *,
    active_detector_ids: Tuple[int, ...],
    support_costs: Dict[Tuple[int, ...], float],
) -> List[Tuple[Tuple[int, ...], Dict[Tuple[int, ...], float]]]:
    detector_to_local = {detector: i for i, detector in enumerate(active_detector_ids)}
    uf = UnionFind(len(active_detector_ids))
    for support in support_costs:
        if len(support) <= 1:
            continue
        first = detector_to_local[support[0]]
        for detector in support[1:]:
            uf.union(first, detector_to_local[detector])

    detectors_by_root: Dict[int, List[int]] = defaultdict(list)
    for detector in active_detector_ids:
        detectors_by_root[uf.find(detector_to_local[detector])].append(detector)
    supports_by_root: Dict[int, Dict[Tuple[int, ...], float]] = defaultdict(dict)
    for support, cost in support_costs.items():
        root = uf.find(detector_to_local[support[0]])
        supports_by_root[root][support] = cost
    components: List[Tuple[Tuple[int, ...], Dict[Tuple[int, ...], float]]] = []
    for root, detectors in detectors_by_root.items():
        components.append((tuple(sorted(detectors)), supports_by_root[root]))
    components.sort(key=lambda item: (len(item[0]), item[0]))
    return components


def solve_primal_lp_on_supports(
    *,
    detector_ids: Tuple[int, ...],
    support_costs: Dict[Tuple[int, ...], float],
    record_stats: SingletonLPSolverStats,
    count_as_main_lp: bool,
) -> Tuple[float, Dict[int, float], List[Tuple[int, ...]], int, int]:
    detector_to_var = {detector: i for i, detector in enumerate(detector_ids)}
    if any(not any(detector in support for support in support_costs) for detector in detector_ids):
        raise RuntimeError("LP component has an uncovered detector; restricted master lost coverage.")

    row_indices: List[int] = []
    col_indices: List[int] = []
    values: List[float] = []
    rhs = np.empty(len(support_costs), dtype=np.float64)
    supports = sorted(support_costs, key=lambda s: (len(s), s))
    for row, support in enumerate(supports):
        rhs[row] = float(support_costs[support])
        for detector in support:
            row_indices.append(row)
            col_indices.append(detector_to_var[detector])
            values.append(1.0)

    a_ub = sparse.csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(supports), len(detector_ids)),
        dtype=np.float64,
    )
    record_stats.lp_calls += 1 if count_as_main_lp else 0
    t0 = time.perf_counter()
    result = linprog(
        c=-np.ones(len(detector_ids), dtype=np.float64),
        A_ub=a_ub,
        b_ub=rhs,
        bounds=[(0.0, None)] * len(detector_ids),
        method="highs",
    )
    elapsed = time.perf_counter() - t0
    if count_as_main_lp:
        record_stats.lp_total_seconds += elapsed
    if not result.success:
        raise RuntimeError(
            f"singleton LP solve failed: status={result.status} message={result.message}"
        )

    solution = np.asarray(result.x, dtype=np.float64)
    y_by_detector = {
        detector_ids[var_index]: float(solution[var_index])
        for var_index in range(len(detector_ids))
        if solution[var_index] > 1e-12
    }
    tight_supports: List[Tuple[int, ...]] = []
    for row, support in enumerate(supports):
        lhs = float(sum(solution[detector_to_var[detector]] for detector in support))
        if abs(float(rhs[row]) - lhs) <= 1e-8:
            tight_supports.append(support)
    return float(-result.fun), y_by_detector, tight_supports, len(detector_ids), len(supports)


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
    singleton_solver: Optional[SingletonLPHeuristic] = None,
    verbose_search: bool = False,
) -> DecodeResult:
    start_time = time.perf_counter()
    if singleton_solver is not None:
        singleton_solver.reset_stats()

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
        exact_refined=(singleton_solver is None),
        lp_solution=None,
        warm_start_solution=None,
        changed_detectors_from_parent=(),
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

    if singleton_solver is None:
        heuristic_name = "plain_detcost"
    else:
        heuristic_name = f"opt_singleton_{singleton_solver.exact_mode}_lazy_{singleton_solver.projection_mode}"
        if singleton_solver.projection_combine_max_plain:
            heuristic_name += "_maxplain"

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
                f"lp_calls={0 if singleton_solver is None else singleton_solver.stats.lp_calls} "
                f"lp_reinserts={lp_reinserts} proj_generated={projected_nodes_generated} "
                f"proj_refined={projected_nodes_refined} "
                f"proj_unrefined_so_far={projected_nodes_generated - projected_nodes_refined} "
                f"active_dets={num_dets} beam_max={max_num_dets} depth={len(state.activated_errors)} "
                f"f={f_cost:.12g} g={state.path_cost:.12g} h={state.heuristic_cost:.12g} "
                f"h_source={state.heuristic_source} exact_refined={state.exact_refined}"
            )

        if num_dets == 0:
            elapsed_seconds = time.perf_counter() - start_time
            stats = DecodeStats(
                num_pq_pushed=num_pq_pushed,
                num_nodes_popped=num_nodes_popped,
                max_queue_size=max_queue_size,
                heuristic_calls=heuristic_calls,
                plain_heuristic_calls=plain_heuristic_calls,
                projection_heuristic_calls=projection_heuristic_calls,
                exact_refinement_calls=exact_refinement_calls,
                lp_calls=0 if singleton_solver is None else singleton_solver.stats.lp_calls,
                lp_reinserts=lp_reinserts,
                projected_nodes_generated=projected_nodes_generated,
                projected_nodes_refined=projected_nodes_refined,
                projected_nodes_unrefined_at_finish=projected_nodes_generated - projected_nodes_refined,
                total_lp_refinement_gain=total_lp_refinement_gain,
                max_lp_refinement_gain=max_lp_refinement_gain,
                lp_total_seconds=0.0 if singleton_solver is None else singleton_solver.stats.lp_total_seconds,
                projection_local_lp_calls=0 if singleton_solver is None else singleton_solver.stats.projection_local_lp_calls,
                projection_local_lp_seconds=0.0 if singleton_solver is None else singleton_solver.stats.projection_local_lp_seconds,
                restricted_total_rounds=0 if singleton_solver is None else singleton_solver.stats.restricted_total_rounds,
                restricted_total_added_supports=0 if singleton_solver is None else singleton_solver.stats.restricted_total_added_supports,
                restricted_total_fallbacks=0 if singleton_solver is None else singleton_solver.stats.restricted_total_fallbacks,
                full_check_calls=0 if singleton_solver is None else singleton_solver.stats.full_check_calls,
                full_check_max_abs_delta=0.0 if singleton_solver is None else singleton_solver.stats.full_check_max_abs_delta,
                elapsed_seconds=elapsed_seconds,
                heuristic_name=heuristic_name,
            )
            return DecodeResult(
                activated_errors=state.activated_errors,
                path_cost=state.path_cost,
                stats=stats,
            )

        if singleton_solver is not None and not state.exact_refined:
            heuristic_calls += 1
            exact_refinement_calls += 1
            previous_h = state.heuristic_cost
            previous_source = state.heuristic_source
            exact_solution, exact_payload = singleton_solver.solve_exact(
                active_detectors=state.active_detectors,
                blocked_errors=state.blocked_errors,
                active_detector_counts=state.active_detector_counts,
                warm_start_solution=state.warm_start_solution,
                changed_detectors=state.changed_detectors_from_parent,
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
                        f"Exact singleton LP lower bound {exact_h} is below stored {previous_source} lower bound {previous_h}."
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

            if singleton_solver.logger is not None:
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
                singleton_solver.logger.maybe_log(call_index=exact_refinement_calls, payload=payload)

            if verbose_search:
                delta_text = "INF" if exact_h == INF else f"{exact_h - previous_h:.12g}"
                exact_text = "INF" if exact_h == INF else f"{exact_h:.12g}"
                print(
                    f"  lp_refine approx_h={previous_h:.12g} exact_h={exact_text} delta={delta_text} "
                    f"vars={exact_solution.num_variables} constraints={exact_solution.num_constraints} "
                    f"selected={exact_solution.num_selected_constraints} rounds={exact_solution.num_rounds} "
                    f"tight={len(exact_solution.tight_supports)} reinserted={reinserted} discarded={discarded}"
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
            changed_detectors = tuple(sorted(data.error_detectors[error_index]))

            if singleton_solver is None:
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
                child_warm_start_solution = None
            else:
                if state.lp_solution is None:
                    raise AssertionError("Projected singleton heuristic requires an exact-refined parent solution.")
                heuristic_calls += 1
                projection_heuristic_calls += 1
                projected_nodes_generated += 1
                children_projected += 1
                child_heuristic = singleton_solver.project_to_child(
                    parent_solution=state.lp_solution,
                    child_active_detectors=child_active_detectors,
                    child_blocked_errors=child_blocked,
                    child_active_detector_counts=child_active_counts,
                    changed_detectors=changed_detectors,
                )
                child_source = "projected"
                child_exact_refined = False
                child_lp_solution = None
                child_warm_start_solution = state.lp_solution

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
                warm_start_solution=child_warm_start_solution,
                changed_detectors_from_parent=changed_detectors,
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
            "Supports plain detcost, lazy full singleton LP, and a restricted-master singleton LP."
        )
    )
    parser.add_argument("--circuit", type=Path, required=True, help="Path to a Stim circuit file.")
    parser.add_argument("--shot", type=int, default=0, help="Zero-based sampled shot index to decode.")
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
        help="Beam cutoff on residual detector count. Use an integer or 'inf'.",
    )
    parser.add_argument(
        "--opt-singleton-detcost-mode",
        choices=["plain", "full", "restricted"],
        default="plain",
        help="Heuristic mode: plain detcost, lazy full singleton LP, or lazy restricted singleton LP.",
    )
    parser.add_argument(
        "--projection-mode",
        choices=["plain", "parent_y", "new_only", "changed_neighborhood"],
        default="changed_neighborhood",
        help=(
            "How to score child nodes before exact refinement. "
            "'parent_y' reuses parent primal detector prices, 'new_only' solves a tiny residual LP on newly active detectors, "
            "and 'changed_neighborhood' solves a tiny residual LP on a local region around the changed detectors."
        ),
    )
    parser.add_argument(
        "--projection-combine-max-plain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Take max(projected child lower bound, plain detcost).",
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
    parser.add_argument("--verbose-search", action="store_true", help="Print per-node search diagnostics.")
    parser.add_argument(
        "--lp-log-path",
        type=Path,
        default=None,
        help="Optional JSONL file for logging exact singleton-LP refinements.",
    )
    parser.add_argument(
        "--lp-log-top-k",
        type=int,
        default=12,
        help="When logging exact LP refinements, include at most this many top supports.",
    )
    parser.add_argument(
        "--lp-log-every",
        type=int,
        default=1,
        help="When logging exact LP refinements, only write every k-th refinement.",
    )
    parser.add_argument(
        "--restricted-add-policy",
        choices=["one", "topk", "all"],
        default="topk",
        help="Violation separation policy for restricted singleton LP mode.",
    )
    parser.add_argument(
        "--restricted-add-top-k",
        type=int,
        default=3,
        help="When --restricted-add-policy=topk, add this many most violated supports.",
    )
    parser.add_argument(
        "--restricted-max-rounds",
        type=int,
        default=50,
        help="Maximum separation rounds before optional fallback to the full singleton LP.",
    )
    parser.add_argument(
        "--restricted-fallback-full",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If restricted mode hits the round limit, fall back to the full singleton LP.",
    )
    parser.add_argument(
        "--restricted-prune-slack",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prune slack supports from the restricted master between rounds.",
    )
    parser.add_argument(
        "--restricted-prune-tol",
        type=float,
        default=1e-8,
        help="Keep selected supports whose slack is at most this value.",
    )
    parser.add_argument(
        "--restricted-violation-tol",
        type=float,
        default=1e-9,
        help="Violation tolerance used during separation.",
    )
    parser.add_argument(
        "--restricted-tight-tol",
        type=float,
        default=1e-8,
        help="Tolerance for tagging a support as tight in the exact solution.",
    )
    parser.add_argument(
        "--restricted-seed-normalized-global-k",
        type=int,
        default=0,
        help="Add this many globally cheapest supports by cost/size to the initial restricted pool.",
    )
    parser.add_argument(
        "--restricted-seed-normalized-touching-changed-k",
        type=int,
        default=2,
        help="Add this many cheapest cost/size supports touching changed detectors to the initial restricted pool.",
    )
    parser.add_argument(
        "--full-check-every",
        type=int,
        default=0,
        help="In restricted mode, solve the full singleton LP every k exact refinements and assert equality (0 disables).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.sample_num_shots <= 0:
        parser.error("--sample-num-shots must be positive.")
    if args.shot < 0:
        parser.error("--shot must be non-negative.")
    if args.lp_log_every <= 0:
        parser.error("--lp-log-every must be positive.")
    if args.lp_log_top_k <= 0:
        parser.error("--lp-log-top-k must be positive.")
    if args.restricted_add_top_k <= 0:
        parser.error("--restricted-add-top-k must be positive.")
    if args.restricted_max_rounds <= 0:
        parser.error("--restricted-max-rounds must be positive.")

    circuit = stim.Circuit.from_file(str(args.circuit))
    dem = circuit.detector_error_model(decompose_errors=False)
    data = build_decoder_data(dem, merge_errors_in_dem=args.merge_errors)

    singleton_solver = None
    if args.opt_singleton_detcost_mode != "plain":
        logger = None
        if args.lp_log_path is not None:
            logger = LPLogger(
                args.lp_log_path,
                every=args.lp_log_every,
                top_k=args.lp_log_top_k,
            )
        restricted_config = RestrictedMasterConfig(
            add_policy=args.restricted_add_policy,
            add_top_k=args.restricted_add_top_k,
            violation_tol=args.restricted_violation_tol,
            tight_tol=args.restricted_tight_tol,
            prune_slack=args.restricted_prune_slack,
            prune_tol=args.restricted_prune_tol,
            seed_normalized_global_k=args.restricted_seed_normalized_global_k,
            seed_normalized_touching_changed_k=args.restricted_seed_normalized_touching_changed_k,
            max_rounds=args.restricted_max_rounds,
            fallback_full=args.restricted_fallback_full,
            full_check_every=args.full_check_every,
        )
        singleton_solver = SingletonLPHeuristic(
            data,
            exact_mode=args.opt_singleton_detcost_mode,
            projection_mode=args.projection_mode,
            projection_combine_max_plain=args.projection_combine_max_plain,
            restricted_config=restricted_config,
            logger=logger,
        )

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
    if singleton_solver is None:
        print("heuristic = plain_detcost")
    else:
        print(
            "heuristic = "
            + f"opt_singleton_{args.opt_singleton_detcost_mode}_lazy_{args.projection_mode}"
            + ("_maxplain" if args.projection_combine_max_plain else "")
        )
    print(f"shot = {args.shot}")
    print(f"sample_num_shots = {args.sample_num_shots}")
    print(f"num_detectors = {data.num_detectors}")
    print(f"num_observables = {data.num_observables}")
    print(f"num_errors = {len(data.errors)}")
    print(f"beam = {args.det_beam}")
    if args.show_shot_detectors:
        print(f"shot_detectors = {format_indices(np.flatnonzero(shot_detections), 'D')}")

    result = decode(
        data=data,
        detections=shot_detections,
        det_beam=args.det_beam,
        singleton_solver=singleton_solver,
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
    print(f"projection_local_lp_calls = {result.stats.projection_local_lp_calls}")
    print(f"projection_local_lp_seconds = {result.stats.projection_local_lp_seconds:.6f}")
    print(f"restricted_total_rounds = {result.stats.restricted_total_rounds}")
    print(f"restricted_total_added_supports = {result.stats.restricted_total_added_supports}")
    print(f"restricted_total_fallbacks = {result.stats.restricted_total_fallbacks}")
    print(f"full_check_calls = {result.stats.full_check_calls}")
    print(f"full_check_max_abs_delta = {result.stats.full_check_max_abs_delta:.12g}")
    print(f"elapsed_seconds = {result.stats.elapsed_seconds:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
