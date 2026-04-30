#!/usr/bin/env python3
"""Prototype A* decoder with incremental greedy singleton heuristics.

Heuristic modes:
    --heuristic plain         exact plain detcost via incremental support updates
    --heuristic asc-deg       exact ascending-degree saturation heuristic
    --heuristic plain-sweep   exact plain+one-sweep saturation heuristic
    --heuristic best-of-two   max(asc-deg, plain-sweep)

All four heuristics are maintained incrementally:
    * the deduplicated active-support dictionary W(T) is updated from parent to
      child using only errors touching flipped detectors;
    * heuristic values are recomputed only on the union of touched connected
      components of the active-support hypergraph;
    * untouched components inherit their detector prices exactly.

This stays inside the singleton-family lower-bound framework, but avoids any LP
solves while still being much tighter than basic detcost in practice.
"""

from __future__ import annotations

import argparse
import heapq
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import stim

INF = math.inf

SupportKey = Tuple[int, ...]


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
    detector_to_errors: List[np.ndarray]
    error_costs: np.ndarray
    error_detectors: List[np.ndarray]
    error_observables: List[np.ndarray]


@dataclass
class SupportState:
    support_to_errors: Dict[SupportKey, FrozenSet[int]]
    support_to_weight: Dict[SupportKey, float]
    detector_to_supports: Dict[int, FrozenSet[SupportKey]]


@dataclass
class HeuristicCache:
    support_state: SupportState
    h_value: float
    y_plain: Optional[np.ndarray] = None
    y_asc: Optional[np.ndarray] = None
    y_sweep: Optional[np.ndarray] = None


@dataclass
class SearchState:
    activated_errors: Tuple[int, ...]
    errs: np.ndarray
    blocked_errors: np.ndarray
    active_detectors: np.ndarray
    path_cost: float
    heuristic_cache: HeuristicCache


@dataclass
class DecodeStats:
    num_pq_pushed: int
    num_nodes_popped: int
    max_queue_size: int
    heuristic_evaluations: int
    support_build_calls: int
    support_build_seconds: float
    support_update_calls: int
    support_update_seconds: float
    component_recompute_calls: int
    component_recompute_seconds: float
    incremental_children: int
    changed_supports_total: int
    touched_detectors_total: int
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


class IncrementalGreedyHeuristic:
    def __init__(
        self,
        data: DecoderData,
        *,
        mode: str,
    ) -> None:
        valid_modes = {"plain", "asc-deg", "plain-sweep", "best-of-two"}
        if mode not in valid_modes:
            raise ValueError(f"Unknown heuristic mode: {mode!r}")
        self.data = data
        self.mode = mode
        self.reset_stats()

    def reset_stats(self) -> None:
        self.heuristic_evaluations = 0
        self.support_build_calls = 0
        self.support_build_seconds = 0.0
        self.support_update_calls = 0
        self.support_update_seconds = 0.0
        self.component_recompute_calls = 0
        self.component_recompute_seconds = 0.0
        self.incremental_children = 0
        self.changed_supports_total = 0
        self.touched_detectors_total = 0

    @property
    def heuristic_name(self) -> str:
        return f"{self.mode}-incremental"

    def _active_support(self, active_detectors: np.ndarray, error_index: int) -> Optional[SupportKey]:
        support = tuple(int(d) for d in self.data.error_detectors[error_index] if active_detectors[int(d)])
        return support if support else None

    def _build_support_state_from_scratch(
        self,
        errs: np.ndarray,
        blocked_errors: np.ndarray,
        active_detectors: np.ndarray,
    ) -> SupportState:
        t0 = time.perf_counter()
        self.support_build_calls += 1

        support_to_errors_mut: Dict[SupportKey, Set[int]] = {}
        for error_index in range(len(self.data.errors)):
            if errs[error_index] or blocked_errors[error_index]:
                continue
            support = self._active_support(active_detectors, error_index)
            if support is None:
                continue
            bucket = support_to_errors_mut.setdefault(support, set())
            bucket.add(error_index)

        support_to_errors: Dict[SupportKey, FrozenSet[int]] = {}
        support_to_weight: Dict[SupportKey, float] = {}
        detector_to_supports_mut: Dict[int, Set[SupportKey]] = defaultdict(set)
        for support, bucket in support_to_errors_mut.items():
            frozen = frozenset(bucket)
            support_to_errors[support] = frozen
            support_to_weight[support] = float(min(self.data.error_costs[ei] for ei in frozen))
            for detector in support:
                detector_to_supports_mut[detector].add(support)
        detector_to_supports = {
            detector: frozenset(supports)
            for detector, supports in detector_to_supports_mut.items()
            if supports
        }

        self.support_build_seconds += time.perf_counter() - t0
        return SupportState(
            support_to_errors=support_to_errors,
            support_to_weight=support_to_weight,
            detector_to_supports=detector_to_supports,
        )

    def _update_support_state_incremental(
        self,
        parent_support_state: SupportState,
        parent_errs: np.ndarray,
        child_errs: np.ndarray,
        parent_blocked: np.ndarray,
        child_blocked: np.ndarray,
        parent_active_detectors: np.ndarray,
        child_active_detectors: np.ndarray,
        flipped_detectors: np.ndarray,
    ) -> Tuple[SupportState, Set[SupportKey], Set[int]]:
        t0 = time.perf_counter()
        self.support_update_calls += 1

        affected_errors: Set[int] = set()
        for detector in flipped_detectors:
            for error_index in self.data.detector_to_errors[int(detector)]:
                affected_errors.add(int(error_index))

        child_support_to_errors = dict(parent_support_state.support_to_errors)
        child_support_to_weight = dict(parent_support_state.support_to_weight)
        touched_buckets: Dict[SupportKey, Set[int]] = {}

        def get_touched_bucket(support: SupportKey) -> Set[int]:
            bucket = touched_buckets.get(support)
            if bucket is None:
                bucket = set(parent_support_state.support_to_errors.get(support, frozenset()))
                touched_buckets[support] = bucket
            return bucket

        for error_index in affected_errors:
            old_available = (not parent_errs[error_index]) and (not parent_blocked[error_index])
            new_available = (not child_errs[error_index]) and (not child_blocked[error_index])
            old_support = self._active_support(parent_active_detectors, error_index) if old_available else None
            new_support = self._active_support(child_active_detectors, error_index) if new_available else None
            if old_support == new_support:
                continue
            if old_support is not None:
                get_touched_bucket(old_support).discard(error_index)
            if new_support is not None:
                get_touched_bucket(new_support).add(error_index)

        changed_supports: Set[SupportKey] = set()
        touched_detectors: Set[int] = set()

        child_detector_to_supports = dict(parent_support_state.detector_to_supports)
        touched_detector_sets: Dict[int, Set[SupportKey]] = {}

        for support, bucket in touched_buckets.items():
            old_bucket = parent_support_state.support_to_errors.get(support, frozenset())
            old_present = support in parent_support_state.support_to_weight
            new_present = bool(bucket)

            if new_present:
                frozen_bucket = frozenset(bucket)
                child_support_to_errors[support] = frozen_bucket
                new_weight = float(min(self.data.error_costs[ei] for ei in frozen_bucket))
                child_support_to_weight[support] = new_weight
                if (not old_present) or frozen_bucket != old_bucket or abs(new_weight - parent_support_state.support_to_weight.get(support, 0.0)) > 1e-12:
                    changed_supports.add(support)
            else:
                child_support_to_errors.pop(support, None)
                if old_present:
                    child_support_to_weight.pop(support, None)
                    changed_supports.add(support)

            if old_present != new_present:
                for detector in support:
                    detector_bucket = touched_detector_sets.get(detector)
                    if detector_bucket is None:
                        detector_bucket = set(parent_support_state.detector_to_supports.get(detector, frozenset()))
                        touched_detector_sets[detector] = detector_bucket
                    if new_present:
                        detector_bucket.add(support)
                    else:
                        detector_bucket.discard(support)

        for support in changed_supports:
            touched_detectors.update(support)

        for detector, supports in touched_detector_sets.items():
            if supports:
                child_detector_to_supports[detector] = frozenset(supports)
            else:
                child_detector_to_supports.pop(detector, None)

        self.incremental_children += 1
        self.changed_supports_total += len(changed_supports)
        self.touched_detectors_total += len(touched_detectors)
        self.support_update_seconds += time.perf_counter() - t0

        return (
            SupportState(
                support_to_errors=child_support_to_errors,
                support_to_weight=child_support_to_weight,
                detector_to_supports=child_detector_to_supports,
            ),
            changed_supports,
            touched_detectors,
        )

    def _component_from_seed_detectors(
        self,
        support_state: SupportState,
        seed_detectors: Iterable[int],
        active_detectors: np.ndarray,
    ) -> Tuple[Set[int], Set[SupportKey]]:
        seen_detectors: Set[int] = set()
        seen_supports: Set[SupportKey] = set()
        stack = [int(d) for d in seed_detectors if active_detectors[int(d)] and int(d) in support_state.detector_to_supports]

        while stack:
            detector = stack.pop()
            if detector in seen_detectors:
                continue
            seen_detectors.add(detector)
            for support in support_state.detector_to_supports.get(detector, frozenset()):
                if support in seen_supports:
                    continue
                seen_supports.add(support)
                for other_detector in support:
                    if active_detectors[other_detector] and other_detector not in seen_detectors:
                        stack.append(other_detector)
        return seen_detectors, seen_supports

    def _all_components(self, support_state: SupportState) -> List[Tuple[Set[int], Set[SupportKey]]]:
        components: List[Tuple[Set[int], Set[SupportKey]]] = []
        seen_detectors: Set[int] = set()
        for detector in sorted(support_state.detector_to_supports):
            if detector in seen_detectors:
                continue
            dets, supports = self._component_from_seed_detectors(
                support_state=support_state,
                seed_detectors=[detector],
                active_detectors=np.ones(self.data.num_detectors, dtype=bool),
            )
            seen_detectors.update(dets)
            components.append((dets, supports))
        return components

    def _component_incidence(
        self,
        component_detectors: Set[int],
        component_supports: Set[SupportKey],
        support_state: SupportState,
    ) -> Dict[int, List[SupportKey]]:
        component_supports_set = set(component_supports)
        incidence: Dict[int, List[SupportKey]] = {}
        for detector in component_detectors:
            local_supports = [
                support
                for support in support_state.detector_to_supports.get(detector, frozenset())
                if support in component_supports_set
            ]
            incidence[detector] = local_supports
        return incidence

    def _compute_plain_component(
        self,
        component_detectors: Set[int],
        component_supports: Set[SupportKey],
        support_state: SupportState,
    ) -> Dict[int, float]:
        incidence = self._component_incidence(component_detectors, component_supports, support_state)
        y: Dict[int, float] = {}
        for detector in component_detectors:
            best = INF
            for support in incidence[detector]:
                candidate = support_state.support_to_weight[support] / len(support)
                if candidate < best:
                    best = candidate
            if math.isinf(best):
                raise RuntimeError("Detector in active support component has no incident support.")
            y[detector] = best
        return y

    def _compute_asc_component(
        self,
        component_detectors: Set[int],
        component_supports: Set[SupportKey],
        support_state: SupportState,
    ) -> Dict[int, float]:
        incidence = self._component_incidence(component_detectors, component_supports, support_state)
        order = sorted(component_detectors, key=lambda d: (len(incidence[d]), d))
        slacks = {support: float(support_state.support_to_weight[support]) for support in component_supports}
        y: Dict[int, float] = {}
        for detector in order:
            value = min(slacks[support] for support in incidence[detector])
            y[detector] = value
            for support in incidence[detector]:
                slacks[support] -= value
        return y

    def _compute_plain_sweep_component(
        self,
        component_detectors: Set[int],
        component_supports: Set[SupportKey],
        support_state: SupportState,
    ) -> Dict[int, float]:
        incidence = self._component_incidence(component_detectors, component_supports, support_state)
        y = self._compute_plain_component(component_detectors, component_supports, support_state)
        slacks = {
            support: float(support_state.support_to_weight[support]) - sum(y[detector] for detector in support)
            for support in component_supports
        }
        order = sorted(component_detectors, key=lambda d: (-y[d], d))
        for detector in order:
            delta = min(slacks[support] for support in incidence[detector])
            y[detector] += delta
            for support in incidence[detector]:
                slacks[support] -= delta
        return y

    def _build_cache_from_support_state(self, support_state: SupportState) -> HeuristicCache:
        t0 = time.perf_counter()
        self.heuristic_evaluations += 1
        self.component_recompute_calls += 1

        y_plain = np.zeros(self.data.num_detectors, dtype=np.float64) if self.mode == "plain" else None
        y_asc = np.zeros(self.data.num_detectors, dtype=np.float64) if self.mode in {"asc-deg", "best-of-two"} else None
        y_sweep = np.zeros(self.data.num_detectors, dtype=np.float64) if self.mode in {"plain-sweep", "best-of-two"} else None

        for component_detectors, component_supports in self._all_components(support_state):
            if self.mode == "plain":
                comp = self._compute_plain_component(component_detectors, component_supports, support_state)
                for detector, value in comp.items():
                    y_plain[detector] = value
            elif self.mode == "asc-deg":
                comp = self._compute_asc_component(component_detectors, component_supports, support_state)
                for detector, value in comp.items():
                    y_asc[detector] = value
            elif self.mode == "plain-sweep":
                comp = self._compute_plain_sweep_component(component_detectors, component_supports, support_state)
                for detector, value in comp.items():
                    y_sweep[detector] = value
            elif self.mode == "best-of-two":
                comp_asc = self._compute_asc_component(component_detectors, component_supports, support_state)
                comp_sweep = self._compute_plain_sweep_component(component_detectors, component_supports, support_state)
                for detector, value in comp_asc.items():
                    y_asc[detector] = value
                for detector, value in comp_sweep.items():
                    y_sweep[detector] = value
            else:
                raise AssertionError("unreachable")

        if self.mode == "plain":
            h_value = float(y_plain.sum())
        elif self.mode == "asc-deg":
            h_value = float(y_asc.sum())
        elif self.mode == "plain-sweep":
            h_value = float(y_sweep.sum())
        else:
            h_value = float(max(y_asc.sum(), y_sweep.sum()))

        self.component_recompute_seconds += time.perf_counter() - t0
        return HeuristicCache(
            support_state=support_state,
            h_value=h_value,
            y_plain=y_plain,
            y_asc=y_asc,
            y_sweep=y_sweep,
        )

    def _incremental_child_cache(
        self,
        parent_cache: HeuristicCache,
        child_support_state: SupportState,
        touched_detectors: Set[int],
        child_active_detectors: np.ndarray,
        flipped_detectors: np.ndarray,
    ) -> HeuristicCache:
        t0 = time.perf_counter()
        self.heuristic_evaluations += 1
        self.component_recompute_calls += 1

        touched_component_detectors, touched_component_supports = self._component_from_seed_detectors(
            support_state=child_support_state,
            seed_detectors=touched_detectors,
            active_detectors=child_active_detectors,
        )

        y_plain = None if parent_cache.y_plain is None else parent_cache.y_plain.copy()
        y_asc = None if parent_cache.y_asc is None else parent_cache.y_asc.copy()
        y_sweep = None if parent_cache.y_sweep is None else parent_cache.y_sweep.copy()

        for detector in flipped_detectors:
            detector = int(detector)
            if not child_active_detectors[detector]:
                if y_plain is not None:
                    y_plain[detector] = 0.0
                if y_asc is not None:
                    y_asc[detector] = 0.0
                if y_sweep is not None:
                    y_sweep[detector] = 0.0

        for detector in touched_component_detectors:
            if y_plain is not None:
                y_plain[detector] = 0.0
            if y_asc is not None:
                y_asc[detector] = 0.0
            if y_sweep is not None:
                y_sweep[detector] = 0.0

        if touched_component_detectors:
            if self.mode == "plain":
                comp = self._compute_plain_component(touched_component_detectors, touched_component_supports, child_support_state)
                for detector, value in comp.items():
                    y_plain[detector] = value
            elif self.mode == "asc-deg":
                comp = self._compute_asc_component(touched_component_detectors, touched_component_supports, child_support_state)
                for detector, value in comp.items():
                    y_asc[detector] = value
            elif self.mode == "plain-sweep":
                comp = self._compute_plain_sweep_component(touched_component_detectors, touched_component_supports, child_support_state)
                for detector, value in comp.items():
                    y_sweep[detector] = value
            elif self.mode == "best-of-two":
                comp_asc = self._compute_asc_component(touched_component_detectors, touched_component_supports, child_support_state)
                comp_sweep = self._compute_plain_sweep_component(touched_component_detectors, touched_component_supports, child_support_state)
                for detector, value in comp_asc.items():
                    y_asc[detector] = value
                for detector, value in comp_sweep.items():
                    y_sweep[detector] = value
            else:
                raise AssertionError("unreachable")

        if self.mode == "plain":
            h_value = float(y_plain.sum())
        elif self.mode == "asc-deg":
            h_value = float(y_asc.sum())
        elif self.mode == "plain-sweep":
            h_value = float(y_sweep.sum())
        else:
            h_value = float(max(y_asc.sum(), y_sweep.sum()))

        self.component_recompute_seconds += time.perf_counter() - t0
        return HeuristicCache(
            support_state=child_support_state,
            h_value=h_value,
            y_plain=y_plain,
            y_asc=y_asc,
            y_sweep=y_sweep,
        )

    def build_root_cache(
        self,
        errs: np.ndarray,
        blocked_errors: np.ndarray,
        active_detectors: np.ndarray,
    ) -> HeuristicCache:
        support_state = self._build_support_state_from_scratch(errs, blocked_errors, active_detectors)
        return self._build_cache_from_support_state(support_state)

    def build_child_cache(
        self,
        parent_state: SearchState,
        child_errs: np.ndarray,
        child_blocked_errors: np.ndarray,
        child_active_detectors: np.ndarray,
        flipped_detectors: np.ndarray,
    ) -> HeuristicCache:
        child_support_state, _changed_supports, touched_detectors = self._update_support_state_incremental(
            parent_support_state=parent_state.heuristic_cache.support_state,
            parent_errs=parent_state.errs,
            child_errs=child_errs,
            parent_blocked=parent_state.blocked_errors,
            child_blocked=child_blocked_errors,
            parent_active_detectors=parent_state.active_detectors,
            child_active_detectors=child_active_detectors,
            flipped_detectors=flipped_detectors,
        )
        return self._incremental_child_cache(
            parent_cache=parent_state.heuristic_cache,
            child_support_state=child_support_state,
            touched_detectors=touched_detectors,
            child_active_detectors=child_active_detectors,
            flipped_detectors=flipped_detectors,
        )


def xor_probability(p0: float, p1: float) -> float:
    return p0 * (1.0 - p1) + (1.0 - p0) * p1


def iter_dem_errors(dem: stim.DetectorErrorModel) -> Iterable[MergedError]:
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        probability = float(instruction.args_copy()[0])
        if probability <= 0:
            continue
        if probability >= 0.5:
            raise ValueError("This prototype assumes DEM probabilities in (0, 0.5).")
        detectors: Set[int] = set()
        observables: Set[int] = set()
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
            likelihood_cost=float(-math.log(probability / (1.0 - probability))),
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
            raise ValueError("Merged error has probability >= 0.5.")
        merged.append(
            MergedError(
                probability=probability,
                likelihood_cost=float(-math.log(probability / (1.0 - probability))),
                detectors=detectors,
                observables=observables,
            )
        )
    return merged


def build_decoder_data(dem: stim.DetectorErrorModel, *, merge_errors_in_dem: bool = True) -> DecoderData:
    errors = merged_errors(dem) if merge_errors_in_dem else list(iter_dem_errors(dem))
    detector_to_errors_lists: List[List[int]] = [[] for _ in range(dem.num_detectors)]
    for error_index, error in enumerate(errors):
        for detector in error.detectors:
            detector_to_errors_lists[detector].append(error_index)
    return DecoderData(
        num_detectors=dem.num_detectors,
        num_observables=dem.num_observables,
        errors=errors,
        detector_to_errors=[np.asarray(v, dtype=np.int32) for v in detector_to_errors_lists],
        error_costs=np.asarray([err.likelihood_cost for err in errors], dtype=np.float64),
        error_detectors=[np.asarray(err.detectors, dtype=np.int32) for err in errors],
        error_observables=[np.asarray(err.observables, dtype=np.int32) for err in errors],
    )


def unpack_bit_packed_rows(bits: np.ndarray, count: int) -> np.ndarray:
    return np.unpackbits(bits, bitorder="little", axis=1, count=count).astype(bool, copy=False)


def detectors_from_solution(data: DecoderData, activated_errors: Sequence[int]) -> np.ndarray:
    detectors = np.zeros(data.num_detectors, dtype=bool)
    for error_index in activated_errors:
        for detector in data.error_detectors[error_index]:
            detectors[int(detector)] ^= True
    return detectors


def observables_from_solution(data: DecoderData, activated_errors: Sequence[int]) -> np.ndarray:
    observables = np.zeros(data.num_observables, dtype=bool)
    for error_index in activated_errors:
        for observable in data.error_observables[error_index]:
            observables[int(observable)] ^= True
    return observables


def parse_beam(text: str) -> float:
    lowered = text.strip().lower()
    if lowered in {"inf", "+inf", "infinity", "+infinity", "none"}:
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


def decode(
    data: DecoderData,
    detections: np.ndarray,
    *,
    det_beam: float,
    heuristic: IncrementalGreedyHeuristic,
    verbose_search: bool = False,
) -> DecodeResult:
    start_time = time.perf_counter()
    heuristic.reset_stats()

    root_dets = np.asarray(detections, dtype=bool).copy()
    root_errs = np.zeros(len(data.errors), dtype=bool)
    root_blocked = np.zeros(len(data.errors), dtype=bool)
    root_cache = heuristic.build_root_cache(root_errs, root_blocked, root_dets)
    root_state = SearchState(
        activated_errors=(),
        errs=root_errs,
        blocked_errors=root_blocked,
        active_detectors=root_dets,
        path_cost=0.0,
        heuristic_cache=root_cache,
    )

    heap: List[Tuple[float, int, int]] = [(root_state.path_cost + root_state.heuristic_cache.h_value, int(root_dets.sum()), 0)]
    node_data: Dict[int, SearchState] = {0: root_state}
    next_node_id = 1

    num_pq_pushed = 1
    num_nodes_popped = 0
    max_queue_size = 1
    min_num_dets = int(root_dets.sum())

    while heap:
        max_queue_size = max(max_queue_size, len(heap))
        f_cost, num_dets, node_id = heapq.heappop(heap)
        state = node_data.pop(node_id, None)
        if state is None:
            continue
        num_nodes_popped += 1

        max_num_dets = INF if det_beam == INF else min_num_dets + det_beam
        if num_dets > max_num_dets:
            continue
        if num_dets < min_num_dets:
            min_num_dets = num_dets
            max_num_dets = INF if det_beam == INF else min_num_dets + det_beam

        if verbose_search:
            print(
                f"len(heap)={len(heap)} nodes_pushed={num_pq_pushed} nodes_popped={num_nodes_popped} "
                f"active_dets={num_dets} beam_max={max_num_dets} depth={len(state.activated_errors)} "
                f"f={f_cost:.12g} g={state.path_cost:.12g} h={state.heuristic_cache.h_value:.12g}"
            )

        if num_dets == 0:
            elapsed = time.perf_counter() - start_time
            return DecodeResult(
                activated_errors=state.activated_errors,
                path_cost=state.path_cost,
                stats=DecodeStats(
                    num_pq_pushed=num_pq_pushed,
                    num_nodes_popped=num_nodes_popped,
                    max_queue_size=max_queue_size,
                    heuristic_evaluations=heuristic.heuristic_evaluations,
                    support_build_calls=heuristic.support_build_calls,
                    support_build_seconds=heuristic.support_build_seconds,
                    support_update_calls=heuristic.support_update_calls,
                    support_update_seconds=heuristic.support_update_seconds,
                    component_recompute_calls=heuristic.component_recompute_calls,
                    component_recompute_seconds=heuristic.component_recompute_seconds,
                    incremental_children=heuristic.incremental_children,
                    changed_supports_total=heuristic.changed_supports_total,
                    touched_detectors_total=heuristic.touched_detectors_total,
                    elapsed_seconds=elapsed,
                    heuristic_name=heuristic.heuristic_name,
                ),
            )

        min_detector = int(np.flatnonzero(state.active_detectors)[0])
        blocked_prefix = state.blocked_errors.copy()

        children_generated = 0
        children_beam_pruned = 0
        for error_index in data.detector_to_errors[min_detector]:
            error_index = int(error_index)
            blocked_prefix[error_index] = True
            if state.errs[error_index] or state.blocked_errors[error_index]:
                continue

            child_errs = state.errs.copy()
            child_errs[error_index] = True
            child_blocked = blocked_prefix.copy()
            child_active_detectors = state.active_detectors.copy()
            flipped_detectors = data.error_detectors[error_index]
            for detector in flipped_detectors:
                child_active_detectors[int(detector)] = ~child_active_detectors[int(detector)]

            child_num_dets = int(child_active_detectors.sum())
            if child_num_dets > max_num_dets:
                children_beam_pruned += 1
                continue

            child_cache = heuristic.build_child_cache(
                parent_state=state,
                child_errs=child_errs,
                child_blocked_errors=child_blocked,
                child_active_detectors=child_active_detectors,
                flipped_detectors=flipped_detectors,
            )
            child_state = SearchState(
                activated_errors=state.activated_errors + (error_index,),
                errs=child_errs,
                blocked_errors=child_blocked,
                active_detectors=child_active_detectors,
                path_cost=state.path_cost + float(data.error_costs[error_index]),
                heuristic_cache=child_cache,
            )
            child_id = next_node_id
            next_node_id += 1
            node_data[child_id] = child_state
            heapq.heappush(heap, (child_state.path_cost + child_cache.h_value, child_num_dets, child_id))
            num_pq_pushed += 1
            children_generated += 1

        if verbose_search:
            print(
                f"  expanded node={node_id} children_generated={children_generated} "
                f"beam_pruned={children_beam_pruned} support_updates={heuristic.support_update_calls}"
            )

    raise RuntimeError("Decoding failed to find any completion.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stim-compatible A* prototype with incrementally maintained greedy singleton-family lower bounds."
        )
    )
    parser.add_argument("--circuit", type=Path, required=True, help="Path to a Stim circuit file.")
    parser.add_argument("--shot", type=int, default=0, help="Zero-based sampled shot index to decode.")
    parser.add_argument("--sample-num-shots", type=int, default=100, help="Number of shots to sample before selecting --shot.")
    parser.add_argument("--seed", type=int, default=27123839530, help="Seed passed to stim.compile_detector_sampler(...).sample(...).")
    parser.add_argument("--det-beam", type=parse_beam, default=INF, help="Beam cutoff on the residual detector count. Use an integer or 'inf'.")
    parser.add_argument(
        "--heuristic",
        choices=["plain", "asc-deg", "plain-sweep", "best-of-two"],
        default="plain-sweep",
        help="Incremental singleton-family heuristic to use.",
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
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.sample_num_shots <= 0:
        parser.error("--sample-num-shots must be positive.")
    if args.shot < 0:
        parser.error("--shot must be non-negative.")

    circuit = stim.Circuit.from_file(str(args.circuit))
    dem = circuit.detector_error_model(decompose_errors=False)
    data = build_decoder_data(dem, merge_errors_in_dem=args.merge_errors)

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

    heuristic = IncrementalGreedyHeuristic(data, mode=args.heuristic)

    print(f"circuit = {args.circuit}")
    print(f"heuristic = {heuristic.heuristic_name}")
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
        heuristic=heuristic,
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
    print(f"heuristic_evaluations = {result.stats.heuristic_evaluations}")
    print(f"support_build_calls = {result.stats.support_build_calls}")
    print(f"support_build_seconds = {result.stats.support_build_seconds:.6f}")
    print(f"support_update_calls = {result.stats.support_update_calls}")
    print(f"support_update_seconds = {result.stats.support_update_seconds:.6f}")
    print(f"component_recompute_calls = {result.stats.component_recompute_calls}")
    print(f"component_recompute_seconds = {result.stats.component_recompute_seconds:.6f}")
    print(f"incremental_children = {result.stats.incremental_children}")
    mean_changed_supports = (
        result.stats.changed_supports_total / result.stats.incremental_children
        if result.stats.incremental_children else 0.0
    )
    mean_touched_detectors = (
        result.stats.touched_detectors_total / result.stats.incremental_children
        if result.stats.incremental_children else 0.0
    )
    print(f"mean_changed_supports = {mean_changed_supports:.6f}")
    print(f"mean_touched_detectors = {mean_touched_detectors:.6f}")
    print(f"elapsed_seconds = {result.stats.elapsed_seconds:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
