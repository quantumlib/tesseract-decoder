#!/usr/bin/env python3
"""Prototype A* decoder for experimenting with fast singleton-budget heuristics.

This version mirrors the earlier Stim-based prototypes:
  * load a .stim circuit,
  * extract its detector error model with decompose_errors=False,
  * optionally merge indistinguishable errors,
  * sample detector shots from Stim,
  * run precedence-pruned A* with a selectable singleton lower-bound heuristic.

Supported heuristic choices:
    plain          original detector-wise feasible point
    asc_deg        zero-start saturation ordered by ascending detector degree
    desc_plain     zero-start saturation ordered by descending plain y_d
    plain_sweep    start from plain, then one descending saturation sweep
    best_of_two    max(plain_sweep, asc_deg)
    best_of_three  max(plain_sweep, asc_deg, desc_plain)
    exact_lp       exact optimal singleton LP lower bound

The greedy heuristics are derived from feasible points of the singleton LP

    max sum_d y_d
    s.t. sum_{d in T} y_d <= W(T)
         y_d >= 0,

where W(T) is the cheapest available error whose active support is T.
"""

from __future__ import annotations

import argparse
import heapq
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import stim
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

INF = float("inf")


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
    g_cost: float


@dataclass
class DecodeResult:
    success: bool
    errs: np.ndarray
    residual_dets: np.ndarray
    cost: float
    nodes_pushed: int
    nodes_popped: int
    heuristic_calls: int
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
        verbose_search: bool = False,
    ) -> None:
        self.errors = list(errors)
        self.num_errors = len(self.errors)
        self.num_detectors = int(num_detectors)
        self.num_observables = int(num_observables)
        self.heuristic_name = heuristic
        self.respect_blocked_errors_in_heuristic = respect_blocked_errors_in_heuristic
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

        self.heuristic_calls = 0

    def reset_stats(self) -> None:
        self.heuristic_calls = 0

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

    def compute_heuristic(self, dets: np.ndarray, errs: np.ndarray, blocked_errs: np.ndarray) -> float:
        self.heuristic_calls += 1
        available = ~errs
        if self.respect_blocked_errors_in_heuristic:
            available &= ~blocked_errs
        support_data = self.build_support_data(dets, available)
        value, _ = self.evaluate_named_heuristic(support_data, self.heuristic_name)
        return value

    def report_root_heuristics(self, dets: np.ndarray, errs: np.ndarray, blocked_errs: np.ndarray) -> List[Tuple[str, float]]:
        available = ~errs
        if self.respect_blocked_errors_in_heuristic:
            available &= ~blocked_errs
        support_data = self.build_support_data(dets, available)
        names = ["plain", "asc_deg", "desc_plain", "plain_sweep", "best_of_two", "best_of_three", "exact_lp"]
        out: List[Tuple[str, float]] = []
        for name in names:
            value, _ = self.evaluate_named_heuristic(support_data, name)
            out.append((name, value))
        return out

    def decode(self, shot_dets: np.ndarray, det_beam: float = INF) -> DecodeResult:
        start_time = time.perf_counter()
        self.reset_stats()

        dets0 = np.array(shot_dets, dtype=bool, copy=True)
        errs0 = np.zeros(self.num_errors, dtype=bool)
        blocked0 = np.zeros(self.num_errors, dtype=bool)
        h0 = self.compute_heuristic(dets0, errs0, blocked0)
        if math.isinf(h0):
            return DecodeResult(
                success=False,
                errs=errs0,
                residual_dets=dets0,
                cost=INF,
                nodes_pushed=1,
                nodes_popped=0,
                heuristic_calls=self.heuristic_calls,
                elapsed_seconds=time.perf_counter() - start_time,
            )

        heap: List[Tuple[float, int, int, SearchState]] = []
        counter = 0
        root_state = SearchState(errs=errs0, blocked_errs=blocked0, dets=dets0, g_cost=0.0)
        heapq.heappush(heap, (h0, int(dets0.sum()), counter, root_state))
        counter += 1
        nodes_pushed = 1
        nodes_popped = 0
        min_num_dets = int(dets0.sum())

        while heap:
            f_cost, num_dets, _entry_id, state = heapq.heappop(heap)
            nodes_popped += 1
            max_num_dets = min_num_dets + det_beam
            if num_dets > max_num_dets:
                continue
            if num_dets < min_num_dets:
                min_num_dets = num_dets
                max_num_dets = min_num_dets + det_beam

            if self.verbose_search:
                print(
                    f"len(heap)={len(heap)} nodes_pushed={nodes_pushed} nodes_popped={nodes_popped} "
                    f"num_dets={num_dets} max_num_dets={max_num_dets} f={f_cost:.6f} g={state.g_cost:.6f}"
                )

            if num_dets == 0:
                return DecodeResult(
                    success=True,
                    errs=state.errs,
                    residual_dets=state.dets,
                    cost=state.g_cost,
                    nodes_pushed=nodes_pushed,
                    nodes_popped=nodes_popped,
                    heuristic_calls=self.heuristic_calls,
                    elapsed_seconds=time.perf_counter() - start_time,
                )

            min_det = int(np.flatnonzero(state.dets)[0])
            prefix_blocked = state.blocked_errs.copy()
            children_generated = 0
            children_beam_pruned = 0
            children_infeasible = 0

            for ei in self.d2e[min_det]:
                ei = int(ei)
                prefix_blocked[ei] = True
                if state.errs[ei] or state.blocked_errs[ei]:
                    continue

                child_errs = state.errs.copy()
                child_errs[ei] = True
                child_blocked = prefix_blocked.copy()
                child_dets = state.dets.copy()
                for d in self.error_detectors[ei]:
                    child_dets[d] ^= True
                child_num_dets = int(child_dets.sum())
                if child_num_dets > max_num_dets:
                    children_beam_pruned += 1
                    continue
                child_g = state.g_cost + float(self.weights[ei])
                child_h = self.compute_heuristic(child_dets, child_errs, child_blocked)
                if math.isinf(child_h):
                    children_infeasible += 1
                    continue
                child_state = SearchState(
                    errs=child_errs,
                    blocked_errs=child_blocked,
                    dets=child_dets,
                    g_cost=child_g,
                )
                heapq.heappush(heap, (child_g + child_h, child_num_dets, counter, child_state))
                counter += 1
                nodes_pushed += 1
                children_generated += 1

            if self.verbose_search:
                print(
                    f"  expanded children_generated={children_generated} beam_pruned={children_beam_pruned} "
                    f"infeasible={children_infeasible}"
                )

        return DecodeResult(
            success=False,
            errs=np.zeros(self.num_errors, dtype=bool),
            residual_dets=np.array(shot_dets, dtype=bool, copy=True),
            cost=INF,
            nodes_pushed=nodes_pushed,
            nodes_popped=nodes_popped,
            heuristic_calls=self.heuristic_calls,
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
        choices=["plain", "asc_deg", "desc_plain", "plain_sweep", "best_of_two", "best_of_three", "exact_lp"],
        default="best_of_two",
        help="Lower-bound heuristic to use during A* search (default: best_of_two).",
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
        help="Print all root-node heuristic values, including exact_lp, for the selected shot.",
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
        verbose_search=args.verbose_search,
    )

    print(f"circuit = {args.circuit}")
    print(f"heuristic = {args.heuristic}")
    print(f"sample_num_shots = {args.sample_num_shots}")
    print(f"shot = {args.shot}")
    print(f"num_errors = {decoder.num_errors}")
    print(f"num_detectors = {decoder.num_detectors}")
    print(f"num_observables = {decoder.num_observables}")
    print(f"det_beam = {args.det_beam}")
    print(f"merge_errors = {args.merge_errors}")
    print(f"respect_blocked_errors_in_heuristic = {args.respect_blocked_errors_in_heuristic}")

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
            print(f"  {name:>12s}  value={value_text}  ratio_to_exact={ratio_text}")

    if args.skip_decode:
        return 0

    result = decoder.decode(shot_dets, det_beam=args.det_beam)
    print(f"success = {result.success}")
    print(f"nodes_pushed = {result.nodes_pushed}")
    print(f"nodes_popped = {result.nodes_popped}")
    print(f"heuristic_calls = {result.heuristic_calls}")
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
