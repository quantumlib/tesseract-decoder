import heapq
import sys
from operator import itemgetter

import stim


def decode_beam_search(circuit: stim.Circuit, actual_dets: set[int], L: int) -> bool | None:
    """
    Decodes a syndrome using a dynamic programming sweep with a Top-L beam cutoff.
    """
    # 1. Extract the Detector Error Model (flattened, decompose_errors=False)
    dem = circuit.detector_error_model(decompose_errors=False).flattened()

    # 2. Parse the DEM into a list of faults
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

    # 3. Convert observed syndrome set to an integer bitmask
    actual_dets_mask = 0
    for d in actual_dets:
        actual_dets_mask ^= (1 << d)

    # If the quantum computer triggered a detector that our error model says 
    # is mathematically impossible to trigger, the syndrome is invalid.
    if (actual_dets_mask & ~all_possible_dets_mask) != 0:
        return None 

    # 4. Pre-calculate retirement schedules
    # retiring_masks[i] stores the bits of detectors that see their final fault at index i.
    retiring_masks = [0] * len(faults)
    last_seen_index = {}
    
    for idx, (_, _, _, det_mask) in enumerate(faults):
        temp = det_mask
        d_id = 0
        # Extract which bits are set in the mask to find the latest index for each detector
        while temp > 0:
            if temp & 1:
                last_seen_index[d_id] = idx
            temp >>= 1
            d_id += 1
            
    for d_id, idx in last_seen_index.items():
        retiring_masks[idx] |= (1 << d_id)

    # 5. The Beam Search Sweep
    # Each beam entry is (active_syndrome_mask, total_probability, logical_bias),
    # where logical_bias = P(L0) - P(L1). Total probability is enough for beam
    # ranking, and the bias preserves the final logical comparison.
    beam = [(0, 1.0, 1.0)]

    for i, (q, p, delta_scale, det_mask) in enumerate(faults):
        next_probs: dict[int, list[float]] = {}

        # A. Expand the beam
        for s, total, delta in beam:
            # Fault absent
            entry = next_probs.get(s)
            absent_total = total * q
            absent_delta = delta * q
            if entry is None:
                next_probs[s] = [absent_total, absent_delta]
            else:
                entry[0] += absent_total
                entry[1] += absent_delta

            # Fault present
            t = s ^ det_mask
            present_total = total * p
            present_delta = delta * delta_scale
            if t == s:
                entry = next_probs[s]
                entry[0] += present_total
                entry[1] += present_delta
            else:
                entry = next_probs.get(t)
                if entry is None:
                    next_probs[t] = [present_total, present_delta]
                else:
                    entry[0] += present_total
                    entry[1] += present_delta

        # B. Enforce Reality & Collapse the State Space
        retiring_mask = retiring_masks[i]
        if retiring_mask != 0:
            collapsed_probs: dict[int, list[float]] = {}
            expected_bits = actual_dets_mask & retiring_mask
            keep_mask = ~retiring_mask

            for s, (total, delta) in next_probs.items():
                # If the retiring bits don't match our actual observation, kill the state.
                if (s & retiring_mask) != expected_bits:
                    continue

                shrunk_s = s & keep_mask
                entry = collapsed_probs.get(shrunk_s)
                if entry is None:
                    collapsed_probs[shrunk_s] = [total, delta]
                else:
                    entry[0] += total
                    entry[1] += delta
        else:
            collapsed_probs = next_probs

        # C. Truncate the Beam (Top L Cutoff)
        if len(collapsed_probs) > L:
            beam = heapq.nlargest(
                L,
                (
                    (state, total, delta)
                    for state, (total, delta) in collapsed_probs.items()
                ),
                key=itemgetter(1),
            )
        else:
            beam = [
                (state, total, delta)
                for state, (total, delta) in collapsed_probs.items()
            ]

    # 6. Final Likelihood Comparison
    # Since all bits are retired, the only surviving state mask should be exactly 0.
    _, _, final_delta = next((entry for entry in beam if entry[0] == 0), (0, 0.0, 0.0))

    if final_delta == 0.0:
        return None  # Tie or beam missed the correct path entirely
    return final_delta < 0.0


def run_experiment(circuit_fname: str, L: int):
    """
    Generates a surface code, samples an error, and decodes it using the beam search.
    """
    # print(f"--- Running Distance {d}, Rounds {r}, Beam Size {L} ---")
    print(f'Running on circuit {circuit_fname}')
 
    circuit = stim.Circuit.from_file(circuit_fname)

    sampler = circuit.compile_detector_sampler()
    syndromes, logicals = sampler.sample(shots=1, separate_observables=True)

    actual_dets = set(i for i, triggered in enumerate(syndromes[0]) if triggered)
    actual_logical = logicals[0][0]

    predicted_logical = decode_beam_search(circuit, actual_dets, L)

    print(f"Total Detectors:      {circuit.num_detectors}")
    print(f"Triggered Detectors:     {len(actual_dets)}")
    print(f"Predicted Logical:    {predicted_logical}")
    print(f"Actual Logical:       {bool(actual_logical)}")
    
    if predicted_logical is None:
        print("Result:               DECODE FAILED (Tie or Beam too narrow)")
    else:
        print(f"Result:               {'SUCCESS' if predicted_logical == actual_logical else 'LOGICAL ERROR'}")
    print()

if __name__ == '__main__':
    run_experiment(sys.argv[1], L=1000)
