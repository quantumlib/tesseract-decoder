#!/usr/bin/env python3
"""Prototype A* decoder with projected singleton-LP refinement.

The default heuristic matches the original prototype's plain detector-wise
heuristic. Passing --opt-singleton-detcost enables a lazy version of the exact
optimal singleton detector lower bound:

    * a node is first inserted with a cheap lower bound;
    * when the node is popped, an LP-based refinement is optionally run;
    * if the refined LP value raises the node's key, the node is reinserted;
    * expanded nodes project their refined LP solution onto each child.

By default the refinement is a full singleton LP solve on pop. Two experimental
modes are also available:

    * --local-lp-component: only reoptimize the active support component(s)
      touched by the flipped error used to create the node.
    * --local-lp-radius R: only reoptimize detector prices within an R-hop
      neighborhood of the changed region, freezing all other prices at their
      projected values.

Both restricted modes remain admissible because they optimize over a subset of
variables while keeping the rest fixed at a feasible point.
"""

from __future__ import annotations

import argparse
import heapq
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import stim
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

INF = float("inf")
HEURISTIC_EPS = 1e-9
FEAS_EPS = 1e-9


@dataclass(frozen=True)
class ErrorRecord:
    probability: float
    likelihood_cost: float
    detectors: Tuple[int, ...]
    observables: Tuple[int, ...]


@dataclass
class SupportSystem:
    active_dets: np.ndarray
    support_to_weight: Dict[Tuple[int, ...], float]
    covered_all: bool


@dataclass
class OptSingletonLPResult:
    value: float
    y_full: np.ndarray
    num_active_dets: int
    num_supports: int
    num_free_vars: int
    seed_frontier_size: int
    mode: str


@dataclass
class SearchState:
    errs: np.ndarray
    blocked_errs: np.ndarray
    dets: np.ndarray
    det_counts: np.ndarray
    g_cost: float
    h_cost: float
    h_source: str
    ready_to_expand: bool
    lp_y: Optional[np.ndarray] = None
    seed_detectors: Optional[np.ndarray] = None


@dataclass
class DecodeResult:
    success: bool
    errs: np.ndarray
    residual_dets: np.ndarray
    cost: float
    nodes_pushed: int
    nodes_popped: int
    heuristic_calls: int
    plain_heuristic_calls: int
    projection_heuristic_calls: int
    refinement_calls: int
    lp_calls: int
    full_lp_calls: int
    component_lp_calls: int
    radius_lp_calls: int
    lp_reinserts: int
    projected_nodes_generated: int
    projected_nodes_refined: int
    projected_nodes_unrefined_at_finish: int
    total_lp_refinement_gain: float
    max_lp_refinement_gain: float
    elapsed_seconds: float


class AStarPrototypeDecoder:
    def __init__(
        self,
        errors: Sequence[ErrorRecord],
        num_detectors: int,
        *,
        use_opt_singleton_detcost: bool = False,
        respect_blocked_errors_in_heuristic: bool = False,
        verbose_search: bool = False,
        local_lp_radius: Optional[int] = None,
        local_lp_component: bool = False,
    ) -> None:
        self.errors = list(errors)
        self.num_detectors = int(num_detectors)
        self.num_errors = len(self.errors)
        self.use_opt_singleton_detcost = use_opt_singleton_detcost
        self.respect_blocked_errors_in_heuristic = respect_blocked_errors_in_heuristic
        self.verbose_search = verbose_search
        self.local_lp_radius = local_lp_radius
        self.local_lp_component = local_lp_component

        if self.local_lp_radius is not None and self.local_lp_radius < 0:
            raise ValueError("local_lp_radius must be non-negative or None")
        if self.local_lp_radius is not None and self.local_lp_component:
            raise ValueError("Choose at most one of local_lp_radius and local_lp_component")

        self.ecosts = np.array([err.likelihood_cost for err in self.errors], dtype=np.float64)
        self.edets: List[np.ndarray] = [
            np.array(err.detectors, dtype=np.int32) for err in self.errors
        ]
        self.eobs: List[np.ndarray] = [
            np.array(err.observables, dtype=np.int32) for err in self.errors
        ]

        d2e_lists: List[List[int]] = [[] for _ in range(self.num_detectors)]
        for ei, dets in enumerate(self.edets):
            for d in dets:
                d2e_lists[int(d)].append(ei)
        self.d2e: List[np.ndarray] = [np.array(v, dtype=np.int32) for v in d2e_lists]

        self.reset_stats()

    def reset_stats(self) -> None:
        self.heuristic_calls = 0
        self.plain_heuristic_calls = 0
        self.projection_heuristic_calls = 0
        self.refinement_calls = 0
        self.lp_calls = 0
        self.full_lp_calls = 0
        self.component_lp_calls = 0
        self.radius_lp_calls = 0
        self.lp_reinserts = 0
        self.projected_nodes_generated = 0
        self.projected_nodes_refined = 0
        self.total_lp_refinement_gain = 0.0
        self.max_lp_refinement_gain = 0.0

    @property
    def heuristic_name(self) -> str:
        if not self.use_opt_singleton_detcost:
            return "plain-detcost"
        if self.local_lp_component:
            return "opt-singleton-detcost-lazy-projection-component"
        if self.local_lp_radius is not None:
            return f"opt-singleton-detcost-lazy-projection-radius{self.local_lp_radius}"
        return "opt-singleton-detcost-lazy-projection-full"

    def _available_errors(self, errs: np.ndarray, blocked_errs: np.ndarray) -> np.ndarray:
        available = ~errs
        if self.respect_blocked_errors_in_heuristic:
            available &= ~blocked_errs
        return available

    def _plain_detcost_heuristic(
        self,
        available_errs: np.ndarray,
        dets: np.ndarray,
        det_counts: np.ndarray,
    ) -> float:
        self.heuristic_calls += 1
        self.plain_heuristic_calls += 1

        total = 0.0
        for d in np.flatnonzero(dets):
            best = INF
            for ei in self.d2e[int(d)]:
                ei = int(ei)
                if not available_errs[ei]:
                    continue
                count = int(det_counts[ei])
                assert count > 0
                value = self.ecosts[ei] / count
                if value < best:
                    best = value
            if math.isinf(best):
                return INF
            total += best
        return total

    def _build_support_system(
        self,
        available_errs: np.ndarray,
        dets: np.ndarray,
        det_counts: np.ndarray,
    ) -> SupportSystem:
        active_dets = np.flatnonzero(dets)
        if active_dets.size == 0:
            return SupportSystem(
                active_dets=active_dets,
                support_to_weight={},
                covered_all=True,
            )

        covered = np.zeros(self.num_detectors, dtype=bool)
        support_to_weight: Dict[Tuple[int, ...], float] = {}

        for ei in np.flatnonzero(available_errs):
            ei = int(ei)
            if int(det_counts[ei]) == 0:
                continue
            support = tuple(int(d) for d in self.edets[ei] if dets[int(d)])
            if not support:
                continue
            for d in support:
                covered[d] = True
            weight = float(self.ecosts[ei])
            old = support_to_weight.get(support)
            if old is None or weight < old:
                support_to_weight[support] = weight

        covered_all = bool(np.all(covered[active_dets]))
        return SupportSystem(
            active_dets=active_dets,
            support_to_weight=support_to_weight,
            covered_all=covered_all,
        )

    def _build_active_neighbors(
        self,
        supports: Iterable[Tuple[int, ...]],
        active_dets: np.ndarray,
    ) -> Dict[int, Set[int]]:
        neighbors: Dict[int, Set[int]] = {int(d): set() for d in active_dets.tolist()}
        for support in supports:
            if len(support) <= 1:
                continue
            support_list = list(support)
            for i, d in enumerate(support_list):
                nbrs = neighbors[int(d)]
                for od in support_list[:i]:
                    nbrs.add(int(od))
                for od in support_list[i + 1 :]:
                    nbrs.add(int(od))
        return neighbors

    def _seed_frontier_from_flipped_detectors(
        self,
        flipped_detectors: np.ndarray,
        dets: np.ndarray,
    ) -> np.ndarray:
        if flipped_detectors is None or len(flipped_detectors) == 0:
            return np.zeros(0, dtype=np.int32)

        frontier: Set[int] = set()
        seen_errors: Set[int] = set()
        for fd in flipped_detectors:
            for ei in self.d2e[int(fd)]:
                ei = int(ei)
                if ei in seen_errors:
                    continue
                seen_errors.add(ei)
                for d in self.edets[ei]:
                    d = int(d)
                    if dets[d]:
                        frontier.add(d)
        if not frontier:
            return np.zeros(0, dtype=np.int32)
        return np.array(sorted(frontier), dtype=np.int32)

    def _free_dets_from_scope(
        self,
        support_system: SupportSystem,
        dets: np.ndarray,
        flipped_detectors: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, int]:
        active_dets = support_system.active_dets
        if active_dets.size == 0:
            return np.zeros(0, dtype=np.int32), 0
        if flipped_detectors is None or len(flipped_detectors) == 0:
            return np.array(active_dets, copy=True), int(active_dets.size)

        seed_frontier = self._seed_frontier_from_flipped_detectors(flipped_detectors, dets)
        seed_frontier_size = int(seed_frontier.size)
        if seed_frontier_size == 0:
            return np.zeros(0, dtype=np.int32), 0

        neighbors = self._build_active_neighbors(support_system.support_to_weight.keys(), active_dets)
        if self.local_lp_component:
            radius_limit: Optional[int] = None
        elif self.local_lp_radius is not None:
            radius_limit = int(self.local_lp_radius)
        else:
            return np.array(active_dets, copy=True), seed_frontier_size

        visited: Set[int] = set(int(d) for d in seed_frontier.tolist())
        frontier: List[int] = [int(d) for d in seed_frontier.tolist()]
        depth = 0

        while frontier and (radius_limit is None or depth < radius_limit):
            next_frontier: List[int] = []
            for d in frontier:
                for od in neighbors.get(d, ()):  # detector adjacency in active support graph
                    if od in visited:
                        continue
                    visited.add(od)
                    next_frontier.append(od)
            frontier = next_frontier
            depth += 1

        if not visited:
            return np.zeros(0, dtype=np.int32), seed_frontier_size
        return np.array(sorted(visited), dtype=np.int32), seed_frontier_size

    def _solve_lp_with_fixed_outside(
        self,
        support_system: SupportSystem,
        base_y: np.ndarray,
        free_dets: np.ndarray,
        *,
        mode: str,
        seed_frontier_size: int,
    ) -> OptSingletonLPResult:
        self.heuristic_calls += 1
        self.refinement_calls += 1
        self.lp_calls += 1
        if mode == "full":
            self.full_lp_calls += 1
        elif mode == "component":
            self.component_lp_calls += 1
        elif mode.startswith("radius"):
            self.radius_lp_calls += 1
        else:
            raise ValueError(f"Unknown LP mode: {mode}")

        active_dets = support_system.active_dets
        num_active_dets = int(active_dets.size)
        num_supports = int(len(support_system.support_to_weight))

        if num_active_dets == 0:
            return OptSingletonLPResult(
                value=0.0,
                y_full=np.zeros(self.num_detectors, dtype=np.float64),
                num_active_dets=0,
                num_supports=0,
                num_free_vars=0,
                seed_frontier_size=seed_frontier_size,
                mode=mode,
            )

        if not support_system.covered_all:
            return OptSingletonLPResult(
                value=INF,
                y_full=np.array(base_y, copy=True),
                num_active_dets=num_active_dets,
                num_supports=num_supports,
                num_free_vars=int(free_dets.size),
                seed_frontier_size=seed_frontier_size,
                mode=mode,
            )

        y_full = np.array(base_y, copy=True)
        free_dets = np.array(sorted(set(int(d) for d in free_dets.tolist() if bool(base_y.shape[0] > d))), dtype=np.int32)
        if free_dets.size == 0:
            return OptSingletonLPResult(
                value=float(y_full[active_dets].sum()),
                y_full=y_full,
                num_active_dets=num_active_dets,
                num_supports=num_supports,
                num_free_vars=0,
                seed_frontier_size=seed_frontier_size,
                mode=mode,
            )

        free_set = set(int(d) for d in free_dets.tolist())
        det_to_var = {int(d): i for i, d in enumerate(free_dets.tolist())}

        row_indices: List[int] = []
        col_indices: List[int] = []
        data: List[float] = []
        rhs: List[float] = []
        row = 0

        for support, weight in support_system.support_to_weight.items():
            fixed_sum = 0.0
            free_support_vars: List[int] = []
            for d in support:
                d = int(d)
                if d in free_set:
                    free_support_vars.append(det_to_var[d])
                else:
                    fixed_sum += float(y_full[d])
            remaining = float(weight) - fixed_sum
            if remaining < -FEAS_EPS:
                raise AssertionError(
                    f"Base y is infeasible in restricted LP: remaining={remaining} mode={mode}"
                )
            remaining = max(0.0, remaining)
            if not free_support_vars:
                continue
            rhs.append(remaining)
            row_indices.extend([row] * len(free_support_vars))
            col_indices.extend(free_support_vars)
            data.extend([1.0] * len(free_support_vars))
            row += 1

        if row == 0:
            return OptSingletonLPResult(
                value=float(y_full[active_dets].sum()),
                y_full=y_full,
                num_active_dets=num_active_dets,
                num_supports=num_supports,
                num_free_vars=int(free_dets.size),
                seed_frontier_size=seed_frontier_size,
                mode=mode,
            )

        a_ub = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(row, int(free_dets.size)),
            dtype=np.float64,
        )

        result = linprog(
            c=-np.ones(int(free_dets.size), dtype=np.float64),
            A_ub=a_ub,
            b_ub=np.array(rhs, dtype=np.float64),
            bounds=[(0.0, None)] * int(free_dets.size),
            method="highs",
        )
        if result.status == 0:
            y_full[free_dets] = np.asarray(result.x, dtype=np.float64)
            return OptSingletonLPResult(
                value=float(y_full[active_dets].sum()),
                y_full=y_full,
                num_active_dets=num_active_dets,
                num_supports=num_supports,
                num_free_vars=int(free_dets.size),
                seed_frontier_size=seed_frontier_size,
                mode=mode,
            )
        if result.status in {2, 3}:  # infeasible or unbounded
            return OptSingletonLPResult(
                value=INF,
                y_full=y_full,
                num_active_dets=num_active_dets,
                num_supports=num_supports,
                num_free_vars=int(free_dets.size),
                seed_frontier_size=seed_frontier_size,
                mode=mode,
            )
        raise RuntimeError(f"linprog failed with status={result.status}: {result.message}")

    def _plain_heuristic_from_state(self, state: SearchState) -> float:
        available = self._available_errors(state.errs, state.blocked_errs)
        return self._plain_detcost_heuristic(available, state.dets, state.det_counts)

    def _project_child_solution_and_heuristic(
        self,
        parent_state: SearchState,
        flipped_detectors: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        if parent_state.lp_y is None:
            raise AssertionError("Expected parent LP solution before projecting to children.")

        self.heuristic_calls += 1
        self.projection_heuristic_calls += 1

        child_y = np.array(parent_state.lp_y, copy=True)
        for d in flipped_detectors:
            d = int(d)
            if parent_state.dets[d]:
                child_y[d] = 0.0
        value = float(parent_state.h_cost)
        for d in flipped_detectors:
            d = int(d)
            if parent_state.dets[d]:
                value -= float(parent_state.lp_y[d])
        if value < -HEURISTIC_EPS:
            raise AssertionError(f"Projected heuristic became negative: {value}")
        return child_y, max(0.0, value)

    def _refine_scope_name(self) -> str:
        if self.local_lp_component:
            return "component"
        if self.local_lp_radius is not None:
            return f"radius{self.local_lp_radius}"
        return "full"

    def _refine_node_lp(
        self,
        state: SearchState,
    ) -> OptSingletonLPResult:
        available = self._available_errors(state.errs, state.blocked_errs)
        support_system = self._build_support_system(available, state.dets, state.det_counts)

        # Root or any node without a projected parent solution falls back to a full LP.
        if state.lp_y is None or state.seed_detectors is None:
            base_y = np.zeros(self.num_detectors, dtype=np.float64)
            return self._solve_lp_with_fixed_outside(
                support_system,
                base_y,
                support_system.active_dets,
                mode="full",
                seed_frontier_size=0,
            )

        if self.local_lp_component or self.local_lp_radius is not None:
            free_dets, seed_frontier_size = self._free_dets_from_scope(
                support_system,
                state.dets,
                state.seed_detectors,
            )
            mode = self._refine_scope_name()
            return self._solve_lp_with_fixed_outside(
                support_system,
                state.lp_y,
                free_dets,
                mode=mode,
                seed_frontier_size=seed_frontier_size,
            )

        return self._solve_lp_with_fixed_outside(
            support_system,
            np.zeros(self.num_detectors, dtype=np.float64),
            support_system.active_dets,
            mode="full",
            seed_frontier_size=0,
        )

    def _maybe_refine_node_with_lp(
        self,
        node_id: int,
        state: SearchState,
        num_dets: int,
    ) -> Tuple[SearchState, Optional[Tuple[float, int]], Optional[Dict[str, float | str]]]:
        if not self.use_opt_singleton_detcost or state.ready_to_expand:
            return state, None, None

        prev_h = state.h_cost
        prev_source = state.h_source
        lp_result = self._refine_node_lp(state)
        refined_h = lp_result.value

        if math.isinf(refined_h):
            refine_info = {
                "approx_h": prev_h,
                "exact_h": refined_h,
                "delta": INF,
                "num_vars": float(lp_result.num_active_dets),
                "num_supports": float(lp_result.num_supports),
                "num_free_vars": float(lp_result.num_free_vars),
                "seed_frontier_size": float(lp_result.seed_frontier_size),
                "reinserted": 0.0,
                "discarded": 1.0,
                "mode": lp_result.mode,
            }
            if prev_source == "projected":
                self.projected_nodes_refined += 1
            return state, None, refine_info

        if refined_h + 1e-7 < prev_h:
            raise AssertionError(
                f"Refined LP lower bound {refined_h} is below stored {prev_source} lower bound {prev_h}."
            )

        delta = refined_h - prev_h
        if prev_source == "projected":
            self.projected_nodes_refined += 1
        self.total_lp_refinement_gain += delta
        self.max_lp_refinement_gain = max(self.max_lp_refinement_gain, delta)

        state.h_cost = refined_h
        state.h_source = lp_result.mode
        state.ready_to_expand = True
        state.lp_y = lp_result.y_full

        should_reinsert = delta > HEURISTIC_EPS
        reinsert_entry = (state.g_cost + refined_h, num_dets) if should_reinsert else None
        if should_reinsert:
            self.lp_reinserts += 1

        refine_info = {
            "approx_h": prev_h,
            "exact_h": refined_h,
            "delta": delta,
            "num_vars": float(lp_result.num_active_dets),
            "num_supports": float(lp_result.num_supports),
            "num_free_vars": float(lp_result.num_free_vars),
            "seed_frontier_size": float(lp_result.seed_frontier_size),
            "reinserted": 1.0 if should_reinsert else 0.0,
            "discarded": 0.0,
            "mode": lp_result.mode,
        }
        return state, reinsert_entry, refine_info

    def _log_pop(
        self,
        *,
        heap_len: int,
        nodes_pushed: int,
        nodes_popped: int,
        num_dets: int,
        max_num_dets: float,
        f_cost: float,
        state: SearchState,
    ) -> None:
        if not self.verbose_search:
            return
        projected_unrefined = self.projected_nodes_generated - self.projected_nodes_refined
        print(
            f"len(heap)={heap_len} nodes_pushed={nodes_pushed} nodes_popped={nodes_popped} "
            f"lp_calls={self.lp_calls} full_lp_calls={self.full_lp_calls} "
            f"component_lp_calls={self.component_lp_calls} radius_lp_calls={self.radius_lp_calls} "
            f"lp_reinserts={self.lp_reinserts} proj_generated={self.projected_nodes_generated} "
            f"proj_refined={self.projected_nodes_refined} proj_unrefined_so_far={projected_unrefined} "
            f"num_dets={num_dets} max_num_dets={max_num_dets} f={f_cost:.6f} g={state.g_cost:.6f} "
            f"h={state.h_cost:.6f} h_source={state.h_source} ready_to_expand={state.ready_to_expand}"
        )

    def _log_refine(self, node_id: int, info: Dict[str, float | str]) -> None:
        if not self.verbose_search:
            return
        exact_h = float(info["exact_h"])
        exact_text = "INF" if math.isinf(exact_h) else f"{exact_h:.6f}"
        delta = float(info["delta"])
        delta_text = "INF" if math.isinf(delta) else f"{delta:.6f}"
        print(
            f"  lp_refine node={node_id} mode={info['mode']} approx_h={float(info['approx_h']):.6f} "
            f"refined_h={exact_text} delta={delta_text} vars={int(float(info['num_vars']))} "
            f"supports={int(float(info['num_supports']))} free_vars={int(float(info['num_free_vars']))} "
            f"seed_frontier={int(float(info['seed_frontier_size']))} "
            f"reinserted={bool(info['reinserted'])} discarded={bool(info['discarded'])}"
        )

    def _log_expand(
        self,
        *,
        node_id: int,
        children_generated: int,
        children_projected: int,
        children_beam_pruned: int,
        children_infeasible: int,
    ) -> None:
        if not self.verbose_search:
            return
        projected_unrefined = self.projected_nodes_generated - self.projected_nodes_refined
        print(
            f"  expanded node={node_id} children_generated={children_generated} "
            f"children_projected={children_projected} beam_pruned={children_beam_pruned} "
            f"infeasible={children_infeasible} lp_calls={self.lp_calls} full_lp_calls={self.full_lp_calls} "
            f"component_lp_calls={self.component_lp_calls} radius_lp_calls={self.radius_lp_calls} "
            f"proj_unrefined_so_far={projected_unrefined}"
        )

    def _result(
        self,
        *,
        success: bool,
        errs: np.ndarray,
        residual_dets: np.ndarray,
        cost: float,
        nodes_pushed: int,
        nodes_popped: int,
        start_time: float,
    ) -> DecodeResult:
        return DecodeResult(
            success=success,
            errs=errs,
            residual_dets=residual_dets,
            cost=cost,
            nodes_pushed=nodes_pushed,
            nodes_popped=nodes_popped,
            heuristic_calls=self.heuristic_calls,
            plain_heuristic_calls=self.plain_heuristic_calls,
            projection_heuristic_calls=self.projection_heuristic_calls,
            refinement_calls=self.refinement_calls,
            lp_calls=self.lp_calls,
            full_lp_calls=self.full_lp_calls,
            component_lp_calls=self.component_lp_calls,
            radius_lp_calls=self.radius_lp_calls,
            lp_reinserts=self.lp_reinserts,
            projected_nodes_generated=self.projected_nodes_generated,
            projected_nodes_refined=self.projected_nodes_refined,
            projected_nodes_unrefined_at_finish=(
                self.projected_nodes_generated - self.projected_nodes_refined
            ),
            total_lp_refinement_gain=self.total_lp_refinement_gain,
            max_lp_refinement_gain=self.max_lp_refinement_gain,
            elapsed_seconds=time.perf_counter() - start_time,
        )

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

        root_state = SearchState(
            errs=errs0,
            blocked_errs=blocked0,
            dets=dets0,
            det_counts=det_counts0,
            g_cost=0.0,
            h_cost=0.0,
            h_source="plain",
            ready_to_expand=not self.use_opt_singleton_detcost,
            lp_y=None,
            seed_detectors=None,
        )
        root_state.h_cost = self._plain_heuristic_from_state(root_state)
        if math.isinf(root_state.h_cost):
            return self._result(
                success=False,
                errs=errs0,
                residual_dets=dets0,
                cost=INF,
                nodes_pushed=1,
                nodes_popped=0,
                start_time=start_time,
            )

        next_node_id = 1
        heap: List[Tuple[float, int, int]] = [
            (root_state.g_cost + root_state.h_cost, int(dets0.sum()), 0)
        ]
        node_data: Dict[int, SearchState] = {0: root_state}

        nodes_pushed = 1
        nodes_popped = 0
        min_num_dets = int(dets0.sum())

        while heap:
            f_cost, num_dets, node_id = heapq.heappop(heap)
            state = node_data.pop(node_id, None)
            if state is None:
                continue
            nodes_popped += 1

            max_num_dets = min_num_dets + det_beam
            if num_dets > max_num_dets:
                continue
            if num_dets < min_num_dets:
                min_num_dets = num_dets
                max_num_dets = min_num_dets + det_beam

            self._log_pop(
                heap_len=len(heap),
                nodes_pushed=nodes_pushed,
                nodes_popped=nodes_popped,
                num_dets=num_dets,
                max_num_dets=max_num_dets,
                f_cost=f_cost,
                state=state,
            )

            if num_dets == 0:
                return self._result(
                    success=True,
                    errs=state.errs,
                    residual_dets=state.dets,
                    cost=state.g_cost,
                    nodes_pushed=nodes_pushed,
                    nodes_popped=nodes_popped,
                    start_time=start_time,
                )

            state, reinsert_entry, refine_info = self._maybe_refine_node_with_lp(
                node_id=node_id,
                state=state,
                num_dets=num_dets,
            )
            if refine_info is not None:
                self._log_refine(node_id, refine_info)
                if bool(refine_info["discarded"]):
                    continue
                if reinsert_entry is not None:
                    node_data[node_id] = state
                    heapq.heappush(heap, (reinsert_entry[0], reinsert_entry[1], node_id))
                    continue

            if self.use_opt_singleton_detcost and not state.ready_to_expand:
                raise AssertionError("Opt-singleton mode should only expand refined nodes.")

            min_det = int(np.flatnonzero(state.dets)[0])
            prefix_blocked_errs = state.blocked_errs.copy()

            children_generated = 0
            children_beam_pruned = 0
            children_infeasible = 0
            children_projected = 0

            for ei in self.d2e[min_det]:
                ei = int(ei)
                prefix_blocked_errs[ei] = True

                if state.errs[ei] or state.blocked_errs[ei]:
                    continue

                child_errs = state.errs.copy()
                child_errs[ei] = True
                child_blocked_errs = prefix_blocked_errs.copy()
                child_dets = state.dets.copy()
                child_det_counts = state.det_counts.copy()

                for d in self.edets[ei]:
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

                child_g = state.g_cost + float(self.ecosts[ei])

                if self.use_opt_singleton_detcost:
                    if state.lp_y is None:
                        raise AssertionError("Expected a refined parent LP solution before projection.")
                    child_lp_y, child_h = self._project_child_solution_and_heuristic(
                        state,
                        self.edets[ei],
                    )
                    child_h_source = "projected"
                    child_ready_to_expand = False
                    child_seed_detectors = np.array(self.edets[ei], copy=True)
                    self.projected_nodes_generated += 1
                    children_projected += 1
                else:
                    child_tmp_state = SearchState(
                        errs=child_errs,
                        blocked_errs=child_blocked_errs,
                        dets=child_dets,
                        det_counts=child_det_counts,
                        g_cost=child_g,
                        h_cost=0.0,
                        h_source="plain",
                        ready_to_expand=True,
                        lp_y=None,
                        seed_detectors=None,
                    )
                    child_h = self._plain_heuristic_from_state(child_tmp_state)
                    child_h_source = "plain"
                    child_ready_to_expand = True
                    child_lp_y = None
                    child_seed_detectors = None
                    if math.isinf(child_h):
                        children_infeasible += 1
                        continue

                child_id = next_node_id
                next_node_id += 1
                node_data[child_id] = SearchState(
                    errs=child_errs,
                    blocked_errs=child_blocked_errs,
                    dets=child_dets,
                    det_counts=child_det_counts,
                    g_cost=child_g,
                    h_cost=child_h,
                    h_source=child_h_source,
                    ready_to_expand=child_ready_to_expand,
                    lp_y=child_lp_y,
                    seed_detectors=child_seed_detectors,
                )
                heapq.heappush(heap, (child_g + child_h, child_num_dets, child_id))
                nodes_pushed += 1
                children_generated += 1

            self._log_expand(
                node_id=node_id,
                children_generated=children_generated,
                children_projected=children_projected,
                children_beam_pruned=children_beam_pruned,
                children_infeasible=children_infeasible,
            )

        return self._result(
            success=False,
            errs=np.zeros(self.num_errors, dtype=bool),
            residual_dets=np.array(shot_dets, dtype=bool, copy=True),
            cost=INF,
            nodes_pushed=nodes_pushed,
            nodes_popped=nodes_popped,
            start_time=start_time,
        )

    def cost_from_errs(self, errs: np.ndarray) -> float:
        return float(self.ecosts[errs].sum())

    def observables_from_errs(self, errs: np.ndarray) -> np.ndarray:
        parity: Dict[int, bool] = {}
        for ei in np.flatnonzero(errs):
            for obs in self.eobs[int(ei)]:
                obs = int(obs)
                parity[obs] = not parity.get(obs, False)
        return np.array(sorted(obs for obs, bit in parity.items() if bit), dtype=np.int32)

    def detectors_from_errs(self, errs: np.ndarray) -> np.ndarray:
        dets = np.zeros(self.num_detectors, dtype=bool)
        for ei in np.flatnonzero(errs):
            for d in self.edets[int(ei)]:
                dets[int(d)] ^= True
        return dets


def merged_errors_from_dem(dem) -> List[ErrorRecord]:
    errors_by_symptom: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {}

    for error in dem.flattened():
        if error.type != "error":
            continue

        probability = float(error.args_copy()[0])
        if probability <= 0:
            continue
        if probability > 0.5:
            raise ValueError(
                f"Expected flattened error probabilities in (0, 0.5], got {probability}."
            )

        detectors: set[int] = set()
        observables: set[int] = set()
        for target in error.targets_copy():
            if target.is_separator():
                continue
            if target.is_logical_observable_id():
                if target.val in observables:
                    observables.remove(target.val)
                else:
                    observables.add(target.val)
            else:
                if not target.is_relative_detector_id():
                    raise ValueError(f"Unexpected target type: {target!r}")
                if target.val in detectors:
                    detectors.remove(target.val)
                else:
                    detectors.add(target.val)

        key = (tuple(sorted(detectors)), tuple(sorted(observables)))
        p_old = errors_by_symptom.get(key)
        if p_old is None:
            p_new = probability
        else:
            p_new = p_old * (1.0 - probability) + (1.0 - p_old) * probability
        errors_by_symptom[key] = p_new

    merged: List[ErrorRecord] = []
    for (detectors, observables), probability in errors_by_symptom.items():
        merged.append(
            ErrorRecord(
                probability=probability,
                likelihood_cost=-math.log(probability / (1.0 - probability)),
                detectors=detectors,
                observables=observables,
            )
        )
    return merged


def sample_detections_and_observables(circuit, num_shots: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
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
        count=circuit.num_detectors,
    )
    obs_unpacked = np.unpackbits(
        obs_packed,
        bitorder="little",
        axis=1,
        count=circuit.num_observables,
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


def parse_nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("Expected a non-negative integer.")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prototype A* decoder using the plain detector-wise heuristic or a lazy "
            "projected version of the optimal singleton detector heuristic."
        )
    )
    parser.add_argument("--circuit", type=Path, required=True, help="Path to a .stim circuit file.")
    parser.add_argument(
        "--shot",
        type=int,
        default=0,
        help="Shot index to decode after sampling --sample-num-shots shots (default: 0).",
    )
    parser.add_argument(
        "--sample-num-shots",
        type=int,
        default=100,
        help="Number of shots to sample before selecting --shot (default: 100).",
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
        help="Beam cutoff on the number of residual detections; use 'inf' for none (default: inf).",
    )
    parser.add_argument(
        "--opt-singleton-detcost",
        action="store_true",
        help=(
            "Use lazy refinement of the optimal singleton detector-cost lower bound. "
            "Nodes are seeded with projected LP prices from their parent and refined "
            "when popped."
        ),
    )
    parser.add_argument(
        "--local-lp-component",
        action="store_true",
        help=(
            "Instead of a full LP on pop, only reoptimize the active support component(s) "
            "touched by the flipped error that created the node."
        ),
    )
    parser.add_argument(
        "--local-lp-radius",
        type=parse_nonnegative_int,
        default=None,
        help=(
            "Instead of a full LP on pop, only reoptimize detector prices within this "
            "radius in the active support graph around the changed region."
        ),
    )
    parser.add_argument(
        "--respect-blocked-errors-in-heuristic",
        action="store_true",
        help=(
            "Exclude precedence-blocked errors from the heuristic. By default the script "
            "preserves the original prototype's behavior and only excludes already-activated errors."
        ),
    )
    parser.add_argument(
        "--show-detections",
        action="store_true",
        help="Print the selected shot's detection events before decoding.",
    )
    parser.add_argument(
        "--show-error-indices",
        action="store_true",
        help="Print the decoded merged-error indices.",
    )
    parser.add_argument(
        "--verbose-search",
        action="store_true",
        help="Print detailed search, LP-refinement, projection, and locality statistics during A* search.",
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
    if args.local_lp_component and args.local_lp_radius is not None:
        parser.error("Choose at most one of --local-lp-component and --local-lp-radius.")
    if (args.local_lp_component or args.local_lp_radius is not None) and not args.opt_singleton_detcost:
        parser.error("Local LP refinement flags require --opt-singleton-detcost.")

    circuit = stim.Circuit.from_file(str(args.circuit))
    dem = circuit.detector_error_model(decompose_errors=False)
    errors = merged_errors_from_dem(dem)

    dets_unpacked, obs_unpacked = sample_detections_and_observables(
        circuit,
        num_shots=args.sample_num_shots,
        seed=args.seed,
    )
    shot_dets = dets_unpacked[args.shot]
    shot_obs = obs_unpacked[args.shot]

    if args.show_detections:
        active_dets = np.flatnonzero(shot_dets)
        print("detections:", " ".join(f"D{d}" for d in active_dets))

    decoder = AStarPrototypeDecoder(
        errors,
        dem.num_detectors,
        use_opt_singleton_detcost=args.opt_singleton_detcost,
        respect_blocked_errors_in_heuristic=args.respect_blocked_errors_in_heuristic,
        verbose_search=args.verbose_search,
        local_lp_radius=args.local_lp_radius,
        local_lp_component=args.local_lp_component,
    )
    result = decoder.decode(shot_dets, det_beam=args.det_beam)

    print(f"heuristic: {decoder.heuristic_name}")
    print(f"shot: {args.shot} / {args.sample_num_shots}")
    print(f"success: {result.success}")
    print(f"nodes_pushed: {result.nodes_pushed}")
    print(f"nodes_popped: {result.nodes_popped}")
    print(f"heuristic_calls: {result.heuristic_calls}")
    print(f"plain_heuristic_calls: {result.plain_heuristic_calls}")
    print(f"projection_heuristic_calls: {result.projection_heuristic_calls}")
    print(f"refinement_calls: {result.refinement_calls}")
    print(f"lp_calls: {result.lp_calls}")
    print(f"full_lp_calls: {result.full_lp_calls}")
    print(f"component_lp_calls: {result.component_lp_calls}")
    print(f"radius_lp_calls: {result.radius_lp_calls}")
    print(f"lp_reinserts: {result.lp_reinserts}")
    print(f"projected_nodes_generated: {result.projected_nodes_generated}")
    print(f"projected_nodes_refined: {result.projected_nodes_refined}")
    print(f"projected_nodes_unrefined_at_finish: {result.projected_nodes_unrefined_at_finish}")
    print(f"total_lp_refinement_gain: {result.total_lp_refinement_gain:.6f}")
    print(f"max_lp_refinement_gain: {result.max_lp_refinement_gain:.6f}")
    print(f"elapsed_seconds: {result.elapsed_seconds:.6f}")

    if not result.success:
        print("decode failed")
        return 1

    decoded_err_indices = np.flatnonzero(result.errs)
    if args.show_error_indices:
        print("decoded_error_indices:", " ".join(map(str, decoded_err_indices.tolist())))

    reproduced_dets = decoder.detectors_from_errs(result.errs)
    if not np.array_equal(reproduced_dets, shot_dets):
        raise AssertionError("Decoded errors do not reproduce the sampled detection events.")

    reproduced_cost = decoder.cost_from_errs(result.errs)
    predicted_obs = decoder.observables_from_errs(result.errs)
    actual_obs = np.flatnonzero(shot_obs)

    print(f"num_decoded_errors: {int(result.errs.sum())}")
    print(f"decoded_cost: {reproduced_cost:.12f}")
    print("predicted_observables:", " ".join(f"L{o}" for o in predicted_obs.tolist()))
    print("sampled_observables:", " ".join(f"L{o}" for o in actual_obs.tolist()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
