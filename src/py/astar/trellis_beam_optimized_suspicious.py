import argparse
import time
from dataclasses import dataclass

import stim


@dataclass(frozen=True)
class BeamDecodeResult:
    predicted_logical: bool | None
    certified: bool
    margin: float
    discarded_mass: float
    max_width: int
    elapsed_seconds: float


def decode_beam_search(circuit: stim.Circuit, actual_dets: set[int], L: int) -> BeamDecodeResult:
    """
    Decodes a syndrome using a dynamic programming sweep with a Top-L beam cutoff.
    """
    start_time = time.perf_counter()

    dem = circuit.detector_error_model(decompose_errors=False).flattened()

    faults = []
    all_possible_dets_mask = 0

    for inst in dem:
        if inst.type != "error":
            continue

        p = inst.args_copy()[0]
        det_mask = 0
        flip_l0 = 0

        for t in inst.targets_copy():
            if t.is_separator():
                continue
            if t.is_relative_detector_id():
                det_mask ^= (1 << t.val)
            elif t.is_logical_observable_id() and t.val == 0:
                flip_l0 ^= 1

        q = 1.0 - p
        delta_scale = -p if flip_l0 else p
        faults.append((q, p, delta_scale, det_mask))
        all_possible_dets_mask |= det_mask

    actual_dets_mask = 0
    for d in actual_dets:
        actual_dets_mask ^= (1 << d)

    if (actual_dets_mask & ~all_possible_dets_mask) != 0:
        return BeamDecodeResult(
            predicted_logical=None,
            certified=False,
            margin=0.0,
            discarded_mass=0.0,
            max_width=0,
            elapsed_seconds=time.perf_counter() - start_time,
        )

    retiring_masks = [0] * len(faults)
    last_seen_index = {}

    for idx, (_, _, _, det_mask) in enumerate(faults):
        temp = det_mask
        d_id = 0
        while temp > 0:
            if temp & 1:
                last_seen_index[d_id] = idx
            temp >>= 1
            d_id += 1

    for d_id, idx in last_seen_index.items():
        retiring_masks[idx] |= (1 << d_id)

    active_mask = 0
    max_width = 0
    for i, (_, _, _, det_mask) in enumerate(faults):
        active_mask |= det_mask
        max_width = max(max_width, active_mask.bit_count())
        active_mask &= ~retiring_masks[i]

    beam = [(0, 1.0, 1.0)]
    discarded_mass = 0.0

    for i, (q, p, delta_scale, det_mask) in enumerate(faults):
        collapsed_probs: dict[int, list[float]] = {}
        total_mass = 0.0
        retiring_mask = retiring_masks[i]

        if retiring_mask == 0:
            for s, total, delta in beam:
                absent_total = total * q
                absent_delta = delta * q
                total_mass += absent_total
                entry = collapsed_probs.get(s)
                if entry is None:
                    collapsed_probs[s] = [absent_total, absent_delta]
                else:
                    entry[0] += absent_total
                    entry[1] += absent_delta

                t = s ^ det_mask
                present_total = total * p
                present_delta = delta * delta_scale
                total_mass += present_total
                entry = collapsed_probs.get(t)
                if entry is None:
                    collapsed_probs[t] = [present_total, present_delta]
                else:
                    entry[0] += present_total
                    entry[1] += present_delta
        else:
            expected_bits = actual_dets_mask & retiring_mask
            keep_mask = ~retiring_mask
            for s, total, delta in beam:
                absent_total = total * q
                absent_delta = delta * q
                if (s & retiring_mask) == expected_bits:
                    shrunk_s = s & keep_mask
                    total_mass += absent_total
                    entry = collapsed_probs.get(shrunk_s)
                    if entry is None:
                        collapsed_probs[shrunk_s] = [absent_total, absent_delta]
                    else:
                        entry[0] += absent_total
                        entry[1] += absent_delta

                t = s ^ det_mask
                present_total = total * p
                present_delta = delta * delta_scale
                if (t & retiring_mask) == expected_bits:
                    shrunk_t = t & keep_mask
                    total_mass += present_total
                    entry = collapsed_probs.get(shrunk_t)
                    if entry is None:
                        collapsed_probs[shrunk_t] = [present_total, present_delta]
                    else:
                        entry[0] += present_total
                        entry[1] += present_delta

        ranked_states = [(total, state, delta) for state, (total, delta) in collapsed_probs.items()]
        if total_mass == 0.0:
            return BeamDecodeResult(
                predicted_logical=None,
                certified=False,
                margin=0.0,
                discarded_mass=discarded_mass,
                max_width=max_width,
                elapsed_seconds=time.perf_counter() - start_time,
            )

        dropped_mass = 0.0
        if len(ranked_states) > L:
            ranked_states.sort(reverse=True)
            kept = ranked_states[:L]
            beam = [(state, total, delta) for total, state, delta in kept]
            kept_mass = sum(total for total, _, _ in kept)
            dropped_mass = total_mass - kept_mass
        else:
            beam = [(state, total, delta) for total, state, delta in ranked_states]

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


def run_experiment(circuit_fname: str, L: int, seed=None):
    print(f"Running on circuit {circuit_fname}")

    circuit = stim.Circuit.from_file(circuit_fname)

    sampler = circuit.compile_detector_sampler(seed=seed)
    syndromes, logicals = sampler.sample(shots=1, separate_observables=True)

    actual_dets = set(i for i, triggered in enumerate(syndromes[0]) if triggered)
    actual_logical = logicals[0][0]

    result = decode_beam_search(circuit, actual_dets, L)

    print(f"Total Detectors:      {circuit.num_detectors}")
    print(f"Seed:                 {seed}")
    print(f"Triggered Detectors:  {len(actual_dets)}")
    print(f"Width:                {result.max_width}")
    print(f"Predicted Logical:    {result.predicted_logical}")
    print(f"Actual Logical:       {bool(actual_logical)}")
    print(f"Certified:            {result.certified}")
    print(f"Margin:               {result.margin:.6e}")
    print(f"Discarded Mass:       {result.discarded_mass:.6e}")
    print(f"Elapsed Seconds:      {result.elapsed_seconds:.6f}")

    if result.predicted_logical is None:
        print("Result:               DECODE FAILED (Tie or Beam too narrow)")
    else:
        print(f"Result:               {'SUCCESS' if result.predicted_logical == actual_logical else 'LOGICAL ERROR'}")
    print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-shot trellis beam decoding on a Stim circuit.")
    parser.add_argument("--circuit", required=True, help="Path to the .stim circuit file.")
    parser.add_argument("--beam", type=int, default=1000, help="Beam width cutoff.")
    parser.add_argument("--seed", type=int, default=None, help="Sampler seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_experiment(args.circuit, L=args.beam, seed=args.seed)
