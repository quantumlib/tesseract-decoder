#!/usr/bin/env python3
"""Prototype A* decoder with plain detcost or optimal singleton detcost.

The default heuristic matches the original prototype's plain detector-wise
heuristic. Passing --opt-singleton-detcost switches to the exact optimal
singleton lower bound, solved as a small LP over the currently active
residual detectors.

Notes:
    * The search still uses the precedence-based tree pruning from the
      prototype.
    * By default, the heuristic ignores precedence-blocked errors in order to
      preserve the original prototype's behavior. Use
      --respect-blocked-errors-in-heuristic to exclude blocked errors from the
      heuristic as well.
    * The optimal singleton heuristic requires SciPy (``scipy.optimize.linprog``).
"""

from __future__ import annotations

import argparse
import heapq
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
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
class SearchState:
    errs: np.ndarray
    blocked_errs: np.ndarray
    dets: np.ndarray
    det_counts: np.ndarray
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
    lp_calls: int
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
    ) -> None:
        self.errors = list(errors)
        self.num_detectors = int(num_detectors)
        self.num_errors = len(self.errors)
        self.use_opt_singleton_detcost = use_opt_singleton_detcost
        self.respect_blocked_errors_in_heuristic = respect_blocked_errors_in_heuristic
        self.verbose_search = verbose_search

        if self.use_opt_singleton_detcost and linprog is None:
            raise RuntimeError(
                "--opt-singleton-detcost requires scipy. Install scipy and rerun."
            )

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

        self.heuristic_calls = 0
        self.lp_calls = 0

    @property
    def heuristic_name(self) -> str:
        if self.use_opt_singleton_detcost:
            return "opt-singleton-detcost"
        return "plain-detcost"

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

    def _opt_singleton_detcost_heuristic(
        self,
        available_errs: np.ndarray,
        dets: np.ndarray,
        det_counts: np.ndarray,
    ) -> float:
        active_dets = np.flatnonzero(dets)
        if active_dets.size == 0:
            return 0.0

        det_to_var = {int(d): i for i, d in enumerate(active_dets.tolist())}
        support_to_weight: Dict[Tuple[int, ...], float] = {}
        covered = np.zeros(active_dets.size, dtype=bool)

        for ei in np.flatnonzero(available_errs):
            ei = int(ei)
            if int(det_counts[ei]) == 0:
                continue
            support = tuple(det_to_var[int(d)] for d in self.edets[ei] if dets[int(d)])
            if not support:
                continue
            for var in support:
                covered[var] = True
            weight = float(self.ecosts[ei])
            old = support_to_weight.get(support)
            if old is None or weight < old:
                support_to_weight[support] = weight

        if not np.all(covered):
            return INF

        num_vars = active_dets.size
        supports = list(support_to_weight.keys())
        weights = np.array([support_to_weight[s] for s in supports], dtype=np.float64)

        row_indices: List[int] = []
        col_indices: List[int] = []
        data: List[float] = []
        for row, support in enumerate(supports):
            row_indices.extend([row] * len(support))
            col_indices.extend(support)
            data.extend([1.0] * len(support))


        a_ub = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(len(supports), num_vars),
            dtype=np.float64,
        )

        self.lp_calls += 1
        result = linprog(
            c=-np.ones(num_vars, dtype=np.float64),
            A_ub=a_ub,
            b_ub=weights,
            bounds=[(0.0, None)] * num_vars,
            method="highs",
        )
        if result.status == 0:
            return max(0.0, float(-result.fun))
        if result.status in {2, 3}:  # infeasible or unbounded
            return INF
        raise RuntimeError(f"linprog failed with status={result.status}: {result.message}")

    def heuristic_cost(
        self,
        errs: np.ndarray,
        blocked_errs: np.ndarray,
        dets: np.ndarray,
        det_counts: np.ndarray,
    ) -> float:
        self.heuristic_calls += 1
        available = self._available_errors(errs, blocked_errs)
        if self.use_opt_singleton_detcost:
            return self._opt_singleton_detcost_heuristic(available, dets, det_counts)
        return self._plain_detcost_heuristic(available, dets, det_counts)

    def decode(self, shot_dets: np.ndarray, det_beam: float = INF) -> DecodeResult:
        start_time = time.perf_counter()
        self.heuristic_calls = 0
        self.lp_calls = 0

        dets0 = np.array(shot_dets, dtype=bool, copy=True)
        errs0 = np.zeros(self.num_errors, dtype=bool)
        blocked0 = np.zeros(self.num_errors, dtype=bool)
        det_counts0 = np.zeros(self.num_errors, dtype=np.uint16)
        for d in np.flatnonzero(dets0):
            for ei in self.d2e[int(d)]:
                det_counts0[int(ei)] += 1

        h0 = self.heuristic_cost(errs0, blocked0, dets0, det_counts0)
        if math.isinf(h0):
            return DecodeResult(
                success=False,
                errs=errs0,
                residual_dets=dets0,
                cost=INF,
                nodes_pushed=1,
                nodes_popped=0,
                heuristic_calls=self.heuristic_calls,
                lp_calls=self.lp_calls,
                elapsed_seconds=time.perf_counter() - start_time,
            )

        next_node_id = 1
        heap: List[Tuple[float, int, int]] = [(h0, int(dets0.sum()), 0)]
        node_data: Dict[int, SearchState] = {
            0: SearchState(
                errs=errs0,
                blocked_errs=blocked0,
                dets=dets0,
                det_counts=det_counts0,
                g_cost=0.0,
            )
        }

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

            errs = state.errs
            blocked_errs = state.blocked_errs
            dets = state.dets
            det_counts = state.det_counts
            g_cost = state.g_cost

            if self.verbose_search:
                print(
                    f"len(heap)={len(heap)} nodes_pushed={nodes_pushed} nodes_popped={nodes_popped} "
                    f"num_dets={num_dets} max_num_dets={max_num_dets} f={f_cost:.6f} g={g_cost:.6f}"
                )

            if num_dets == 0:
                return DecodeResult(
                    success=True,
                    errs=errs,
                    residual_dets=dets,
                    cost=g_cost,
                    nodes_pushed=nodes_pushed,
                    nodes_popped=nodes_popped,
                    heuristic_calls=self.heuristic_calls,
                    lp_calls=self.lp_calls,
                    elapsed_seconds=time.perf_counter() - start_time,
                )

            min_det = int(np.flatnonzero(dets)[0])
            prefix_blocked_errs = blocked_errs.copy()

            for ei in self.d2e[min_det]:
                ei = int(ei)
                prefix_blocked_errs[ei] = True

                if errs[ei] or blocked_errs[ei]:
                    continue

                child_errs = errs.copy()
                child_errs[ei] = True
                child_blocked_errs = prefix_blocked_errs.copy()
                child_dets = dets.copy()
                child_det_counts = det_counts.copy()

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
                    continue

                child_g = g_cost + float(self.ecosts[ei])
                child_h = self.heuristic_cost(
                    child_errs,
                    child_blocked_errs,
                    child_dets,
                    child_det_counts,
                )
                if math.isinf(child_h):
                    continue

                child_id = next_node_id
                next_node_id += 1
                node_data[child_id] = SearchState(
                    errs=child_errs,
                    blocked_errs=child_blocked_errs,
                    dets=child_dets,
                    det_counts=child_det_counts,
                    g_cost=child_g,
                )
                heapq.heappush(heap, (child_g + child_h, child_num_dets, child_id))
                nodes_pushed += 1

        return DecodeResult(
            success=False,
            errs=np.zeros(self.num_errors, dtype=bool),
            residual_dets=np.array(shot_dets, dtype=bool, copy=True),
            cost=INF,
            nodes_pushed=nodes_pushed,
            nodes_popped=nodes_popped,
            heuristic_calls=self.heuristic_calls,
            lp_calls=self.lp_calls,
            elapsed_seconds=time.perf_counter() - start_time,
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
            # Two independent identical symptoms combine by XORing their parity.
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prototype A* decoder using the plain detector-wise heuristic or the "
            "optimal singleton detector heuristic."
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
            "Use the exact optimal singleton detector-cost lower bound instead of the "
            "plain detector-wise lower bound. Requires scipy."
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
        help="Print one line per expanded node during A* search.",
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

    try:
        import stim
    except ImportError as exc:  # pragma: no cover - depends on runtime environment.
        raise SystemExit("This script requires the 'stim' package to be installed.") from exc

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
    )
    result = decoder.decode(shot_dets, det_beam=args.det_beam)

    print(f"heuristic: {decoder.heuristic_name}")
    print(f"shot: {args.shot} / {args.sample_num_shots}")
    print(f"success: {result.success}")
    print(f"nodes_pushed: {result.nodes_pushed}")
    print(f"nodes_popped: {result.nodes_popped}")
    print(f"heuristic_calls: {result.heuristic_calls}")
    print(f"lp_calls: {result.lp_calls}")
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
