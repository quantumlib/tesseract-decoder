#!/usr/bin/env python3

import argparse
import math
import time
from dataclasses import dataclass

import stim


@dataclass(frozen=True)
class Fault:
    q: float
    p: float
    delta_scale: float
    det_mask: int
    likelihood_cost: float


@dataclass(frozen=True)
class BeamDecodeResult:
    predicted_logical: bool | None
    certified: bool
    margin: float
    discarded_mass: float
    max_width: int
    elapsed_seconds: float


@dataclass(frozen=True)
class SampledShot:
    det_mask: int
    actual_logical: bool


def _likelihood_cost(probability: float) -> float:
    if probability <= 0.0:
        return math.inf
    if probability >= 1.0:
        return 0.0
    return -math.log(probability / (1.0 - probability))


def _detectors_from_mask(mask: int) -> list[int]:
    detectors: list[int] = []
    while mask:
        low_bit = mask & -mask
        detectors.append(low_bit.bit_length() - 1)
        mask ^= low_bit
    return detectors


def _parse_circuit(circuit: stim.Circuit) -> tuple[list[Fault], list[int], list[int], int]:
    dem = circuit.detector_error_model(decompose_errors=False).flattened()

    faults: list[Fault] = []
    all_possible_dets_mask = 0
    last_seen_index: dict[int, int] = {}

    for inst in dem:
        if inst.type != "error":
            continue

        p = float(inst.args_copy()[0])
        det_mask = 0
        flip_l0 = 0
        for target in inst.targets_copy():
            if target.is_separator():
                continue
            if target.is_relative_detector_id():
                det_mask ^= 1 << target.val
            elif target.is_logical_observable_id() and target.val == 0:
                flip_l0 ^= 1

        faults.append(
            Fault(
                q=1.0 - p,
                p=p,
                delta_scale=(-p if flip_l0 else p),
                det_mask=det_mask,
                likelihood_cost=_likelihood_cost(p),
            )
        )
        all_possible_dets_mask |= det_mask

        for det_id in _detectors_from_mask(det_mask):
            last_seen_index[det_id] = len(faults) - 1

    retiring_masks = [0] * len(faults)
    for det_id, index in last_seen_index.items():
        retiring_masks[index] |= 1 << det_id

    live_masks_after = [0] * (len(faults) + 1)
    active_mask = 0
    max_width = 0
    for i, fault in enumerate(faults):
        active_mask |= fault.det_mask
        max_width = max(max_width, active_mask.bit_count())
        active_mask &= ~retiring_masks[i]
        live_masks_after[i + 1] = active_mask

    return faults, retiring_masks, live_masks_after, max_width


def _future_detcost_by_detector(faults: list[Fault], num_detectors: int) -> list[list[float]]:
    future_detcost = [[math.inf] * num_detectors for _ in range(len(faults) + 1)]
    next_row = future_detcost[-1]
    for fault_index in range(len(faults) - 1, -1, -1):
        row = next_row.copy()
        fault = faults[fault_index]
        det_count = fault.det_mask.bit_count()
        if det_count:
            ecost = fault.likelihood_cost / det_count
            for det_id in _detectors_from_mask(fault.det_mask):
                if ecost < row[det_id]:
                    row[det_id] = ecost
        future_detcost[fault_index] = row
        next_row = row
    return future_detcost


def _detcost_penalty(mismatch_mask: int, future_detcost: list[float]) -> float:
    total = 0.0
    pending = mismatch_mask

    while pending:
        low_bit = pending & -pending
        detector = low_bit.bit_length() - 1
        pending ^= low_bit

        best = future_detcost[detector]
        if best == math.inf:
            return math.inf
        total += best

    return total


def sample_shots(circuit: stim.Circuit, shots: int, seed: int | None) -> list[SampledShot]:
    sampler = circuit.compile_detector_sampler(seed=seed)
    syndromes, logicals = sampler.sample(shots=shots, separate_observables=True)
    out: list[SampledShot] = []
    for shot_index in range(shots):
        det_mask = 0
        for detector, fired in enumerate(syndromes[shot_index]):
            if fired:
                det_mask ^= 1 << detector
        out.append(SampledShot(det_mask=det_mask, actual_logical=bool(logicals[shot_index][0])))
    return out


def decode_beam_search_detcost_ranked(
    circuit: stim.Circuit,
    actual_dets: set[int],
    L: int,
) -> BeamDecodeResult:
    start_time = time.perf_counter()

    faults, retiring_masks, live_masks_after, max_width = _parse_circuit(circuit)

    actual_dets_mask = 0
    for detector in actual_dets:
        actual_dets_mask ^= 1 << detector

    all_possible_dets_mask = 0
    for fault in faults:
        all_possible_dets_mask |= fault.det_mask
    if (actual_dets_mask & ~all_possible_dets_mask) != 0:
        return BeamDecodeResult(
            predicted_logical=None,
            certified=False,
            margin=0.0,
            discarded_mass=0.0,
            max_width=0,
            elapsed_seconds=time.perf_counter() - start_time,
        )

    future_detcost = _future_detcost_by_detector(faults, circuit.num_detectors)

    beam = [(0, 1.0, 1.0)]
    discarded_mass = 0.0

    for i, fault in enumerate(faults):
        collapsed_probs: dict[int, list[float]] = {}
        total_mass = 0.0
        retiring_mask = retiring_masks[i]

        if retiring_mask == 0:
            for state, total, delta in beam:
                absent_total = total * fault.q
                absent_delta = delta * fault.q
                total_mass += absent_total
                entry = collapsed_probs.get(state)
                if entry is None:
                    collapsed_probs[state] = [absent_total, absent_delta]
                else:
                    entry[0] += absent_total
                    entry[1] += absent_delta

                toggled = state ^ fault.det_mask
                present_total = total * fault.p
                present_delta = delta * fault.delta_scale
                total_mass += present_total
                entry = collapsed_probs.get(toggled)
                if entry is None:
                    collapsed_probs[toggled] = [present_total, present_delta]
                else:
                    entry[0] += present_total
                    entry[1] += present_delta
        else:
            expected_bits = actual_dets_mask & retiring_mask
            keep_mask = ~retiring_mask
            for state, total, delta in beam:
                absent_total = total * fault.q
                absent_delta = delta * fault.q
                if (state & retiring_mask) == expected_bits:
                    shrunk = state & keep_mask
                    total_mass += absent_total
                    entry = collapsed_probs.get(shrunk)
                    if entry is None:
                        collapsed_probs[shrunk] = [absent_total, absent_delta]
                    else:
                        entry[0] += absent_total
                        entry[1] += absent_delta

                toggled = state ^ fault.det_mask
                present_total = total * fault.p
                present_delta = delta * fault.delta_scale
                if (toggled & retiring_mask) == expected_bits:
                    shrunk = toggled & keep_mask
                    total_mass += present_total
                    entry = collapsed_probs.get(shrunk)
                    if entry is None:
                        collapsed_probs[shrunk] = [present_total, present_delta]
                    else:
                        entry[0] += present_total
                        entry[1] += present_delta

        if total_mass == 0.0:
            return BeamDecodeResult(
                predicted_logical=None,
                certified=False,
                margin=0.0,
                discarded_mass=discarded_mass,
                max_width=max_width,
                elapsed_seconds=time.perf_counter() - start_time,
            )

        ranked_states: list[tuple[float, float, int, float]] = []
        live_target_mask = actual_dets_mask & live_masks_after[i + 1]
        next_future_detcost = future_detcost[i + 1]
        for state, (total, delta) in collapsed_probs.items():
            mismatch_mask = state ^ live_target_mask
            penalty = _detcost_penalty(mismatch_mask=mismatch_mask, future_detcost=next_future_detcost)
            if penalty == math.inf:
                rank_score = -math.inf
            else:
                rank_score = math.log(total) - penalty
            ranked_states.append((rank_score, total, state, delta))

        dropped_mass = 0.0
        if len(ranked_states) > L:
            ranked_states.sort(reverse=True)
            kept = ranked_states[:L]
            beam = [(state, total, delta) for _, total, state, delta in kept]
            kept_mass = sum(total for _, total, _, _ in kept)
            dropped_mass = total_mass - kept_mass
        else:
            beam = [(state, total, delta) for _, total, state, delta in ranked_states]

        inv_total_mass = 1.0 / total_mass
        discarded_mass = (discarded_mass + dropped_mass) * inv_total_mass
        beam = [
            (state, total * inv_total_mass, delta * inv_total_mass)
            for state, total, delta in beam
        ]

    _, _, final_delta = next((entry for entry in beam if entry[0] == 0), (0, 0.0, 0.0))
    margin = abs(final_delta)
    certified = margin > discarded_mass

    if final_delta == 0.0:
        return BeamDecodeResult(
            predicted_logical=None,
            certified=False,
            margin=margin,
            discarded_mass=discarded_mass,
            max_width=max_width,
            elapsed_seconds=time.perf_counter() - start_time,
        )
    return BeamDecodeResult(
        predicted_logical=final_delta < 0.0,
        certified=certified,
        margin=margin,
        discarded_mass=discarded_mass,
        max_width=max_width,
        elapsed_seconds=time.perf_counter() - start_time,
    )


def run_experiment(
    circuit_fname: str,
    L: int,
    sample_num_shots: int,
    sample_seed: int | None = None,
    print_per_shot: bool = False,
) -> None:
    circuit = stim.Circuit.from_file(circuit_fname)
    shots = sample_shots(circuit, shots=sample_num_shots, seed=sample_seed)

    print(f"Running on circuit {circuit_fname}")
    print(f"Total Detectors:      {circuit.num_detectors}")
    print(f"Sample Seed:          {sample_seed}")
    print(f"Num Shots:            {len(shots)}")

    num_errors = 0
    num_low_confidence = 0
    num_certified = 0
    total_elapsed = 0.0
    total_triggered = 0
    max_width_seen = 0

    for shot_index, shot in enumerate(shots):
        actual_dets = set(_detectors_from_mask(shot.det_mask))
        result = decode_beam_search_detcost_ranked(circuit, actual_dets, L)
        success = result.predicted_logical == shot.actual_logical if result.predicted_logical is not None else False
        low_confidence = result.predicted_logical is None

        if low_confidence:
            num_low_confidence += 1
        elif not success:
            num_errors += 1
        if result.certified:
            num_certified += 1

        total_elapsed += result.elapsed_seconds
        total_triggered += len(actual_dets)
        max_width_seen = max(max_width_seen, result.max_width)

        shots_done = shot_index + 1
        resolved_shots = shots_done - num_low_confidence
        error_rate_so_far = num_errors / resolved_shots if resolved_shots else 0.0
        print(
            f"progress shots_done={shots_done}/{len(shots)} errors_so_far={num_errors} "
            f"low_conf_so_far={num_low_confidence} error_rate_so_far={error_rate_so_far:.6f} "
            f"elapsed_total_seconds={total_elapsed:.6f}"
        )

        if print_per_shot:
            print(
                f"shot={shot_index} triggered_detectors={len(actual_dets)} "
                f"predicted_logical={result.predicted_logical} actual_logical={shot.actual_logical} "
                f"success={success} certified={result.certified} "
                f"margin={result.margin:.6e} discarded_mass={result.discarded_mass:.6e} "
                f"elapsed_seconds={result.elapsed_seconds:.6f}"
            )

    print(f"Beam:                 {L}")
    print(f"Mean Triggered Dets:  {total_triggered / max(1, len(shots)):.2f}")
    print(f"Max Width:            {max_width_seen}")
    print(f"Certified Shots:      {num_certified}")
    print(f"Low Confidence:       {num_low_confidence}")
    print(f"Logical Errors:       {num_errors}")
    print(f"Total Seconds:        {total_elapsed:.6f}")
    print(f"Mean Seconds/Shot:    {total_elapsed / max(1, len(shots)):.6f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run trellis beam decoding ranked by mass minus a detcost-style future penalty."
    )
    parser.add_argument("--circuit", required=True, help="Path to the .stim circuit file.")
    parser.add_argument("--beam", type=int, default=1000, help="Beam width cutoff.")
    parser.add_argument("--sample-num-shots", type=int, default=1, help="Number of sampled shots.")
    parser.add_argument("--sample-seed", type=int, default=None, help="Stim sampler seed.")
    parser.add_argument("--print-per-shot", action="store_true", help="Print a detailed line per decoded shot.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.sample_num_shots <= 0:
        raise ValueError("--sample-num-shots must be positive.")
    run_experiment(
        args.circuit,
        L=args.beam,
        sample_num_shots=args.sample_num_shots,
        sample_seed=args.sample_seed,
        print_per_shot=args.print_per_shot,
    )
