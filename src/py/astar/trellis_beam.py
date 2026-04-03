import stim
from collections import defaultdict
import sys


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

        faults.append((p, det_mask, flip_l0))
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
    
    for idx, (_, det_mask, _) in enumerate(faults):
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
    state_probs = {0: [1.0, 0.0]}  # active_syndrome_mask -> [P(L0), P(L1)]

    for i, (p, det_mask, flip_l0) in enumerate(faults):
        q = 1.0 - p
        next_probs = defaultdict(lambda: [0.0, 0.0])

        # A. Expand the beam
        for s, (p0, p1) in state_probs.items():
            # Fault absent
            next_probs[s][0] += p0 * q
            next_probs[s][1] += p1 * q

            # Fault present
            t = s ^ det_mask
            if flip_l0:
                next_probs[t][0] += p1 * p
                next_probs[t][1] += p0 * p
            else:
                next_probs[t][0] += p0 * p
                next_probs[t][1] += p1 * p

        # B. Enforce Reality & Collapse the State Space
        retiring_mask = retiring_masks[i]
        collapsed_probs = defaultdict(lambda: [0.0, 0.0])
        
        for s, (p0, p1) in next_probs.items():
            if retiring_mask != 0:
                # If the retiring bits don't match our actual observation, kill the state
                if (s & retiring_mask) != (actual_dets_mask & retiring_mask):
                    continue 
            
            # Zero out the retired bits so states merge properly in the dictionary
            shrunk_s = s & ~retiring_mask
            collapsed_probs[shrunk_s][0] += p0
            collapsed_probs[shrunk_s][1] += p1

        # C. Truncate the Beam (Top L Cutoff)
        if len(collapsed_probs) > L:
            # Sort by total marginal probability: P(L0) + P(L1)
            sorted_states = sorted(
                collapsed_probs.items(), 
                key=lambda kv: kv[1][0] + kv[1][1], 
                reverse=True
            )
            state_probs = dict(sorted_states[:L])
        else:
            state_probs = dict(collapsed_probs)

    # 6. Final Likelihood Comparison
    # Since all bits are retired, the only surviving state mask should be exactly 0.
    p0, p1 = state_probs.get(0, (0.0, 0.0))
    
    if p0 == p1:
        return None  # Tie or beam missed the correct path entirely
    return p1 > p0


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
