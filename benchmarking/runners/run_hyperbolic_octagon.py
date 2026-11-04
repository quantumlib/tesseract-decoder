#!/usr/bin/env python3
"""Hyperbolic Octagon cat-flagged circuits (r=16, SI1000 p=0.001, p_seam=0.01).

  iter=32  [[144,8]]   iter=40 [[288,12]]   iter=58 [[1152,36]]   iter=109 [[3600,100]]

Headline [O]/[C] error counts and s/shot come from the 330,000-shot `dripto3`
cluster suite (constants below, as transcribed in coordinator brief #128); the
[AQ] rows are the AlphaQubit reference counts. These cannot be reproduced on a
workstation in reasonable time, so each [O]/[C] record carries a *separate*
local telemetry run (same circuit, canonical flags) whose own command line,
shots, errors, seed and CPU time are stored in `stage_breakdown.telemetry_*`.
`telemetry_same_run_as_headline` is False for these records.

  bazel build --jobs=1 //src:bp
  python3 benchmarking/runners/run_hyperbolic_octagon.py --threads 64
  python3 benchmarking/runners/normalize_datasets.py
"""

import argparse
import json

import bench_common as bc

DATA_OUT = bc.DATA_DIR / "hyperbolic_octagon.json"
FAMILY = "hyperbolic_octagon"
R = 16
CIRCUITS = [
    {"tag": "hyp_oct_iter32", "iter": 32, "n": 144, "k": 8},
    {"tag": "hyp_oct_iter40", "iter": 40, "n": 288, "k": 12},
    {"tag": "hyp_oct_iter58", "iter": 58, "n": 1152, "k": 36},
    {"tag": "hyp_oct_iter109", "iter": 109, "n": 3600, "k": 100},
]
# (errors, shots, cpu s/shot or None).
CLUSTER = {
    32: {"[AQ]": (20539, 1000448, None), "[O]": (2071, 102000, 0.0505), "[C]": (1770, 102000, 0.6087)},
    40: {"[AQ]": (3118, 1000448, None), "[O]": (312, 102000, 0.1794), "[C]": (244, 102000, 2.0251)},
    58: {"[AQ]": (227, 1000192, None), "[O]": (117, 102000, 1.7188), "[C]": (93, 102000, 1.8592)},
    109: {"[AQ]": (196, 1000064, None), "[O]": (22, 24000, 5.5614), "[C]": (11, 24000, 6.4502)},
}
CLUSTER_SOURCE = (
    "dripto3 cluster 330k-shot hyperbolic suite; counts and CPU s/shot as transcribed "
    "in coordinator brief #128. Exact cluster command line was not archived (the "
    "earlier combined_v2 sweep in board #86 used --max-iter 50 --osd-order 60)."
)
AQ_SOURCE = "AlphaQubit reference counts (coordinator brief #128 / board #86)"
TELEMETRY_SHOTS = {32: 20480, 40: 20480, 58: 4096, 109: 4096}


def circuit_path(c):
    return (
        "cat-flagged-circuits-octagon-L2-aug13-2026/r=8,g=8,b=8,iter={iter},side_length=2,"
        "module_size=8,module_color=0,logical_basis=X,num_rounds=16,total_qubits={n},k={k},"
        "canonical_layout=OCTAGON_canonical,noise=SI1000,p=0.00100,p_seam=0.01000.stim"
    ).format(**c)


def headline(c, code, commit):
    nerr, nshots, sps = CLUSTER[c["iter"]][code]
    key, mode, label = bc.CODE_META[code]
    p = nerr / nshots
    cpu = None if sps is None else nshots * sps
    return {
        "tag": c["tag"],
        "label": f"Hyperbolic Octagon iter={c['iter']} [[{c['n']}, {c['k']}]]",
        "family": FAMILY,
        "iter": c["iter"],
        "n": c["n"],
        "q": c["n"],
        "k": c["k"],
        "r": R,
        "p": 0.001,
        "p_seam": 0.01,
        "decoder_key": key,
        "mode": mode,
        "decoder_code": code,
        "decoder_label": label,
        "canonical": True,
        "source": AQ_SOURCE if code == "[AQ]" else CLUSTER_SOURCE,
        "git_commit": commit,
        "command_line": "n/a (AlphaQubit reference)" if code == "[AQ]" else "n/a (dripto3 cluster job, not archived)",
        "circuit_path": circuit_path(c),
        "sample_seed": None,
        "num_shots": nshots,
        "num_errors": nerr,
        "shot_error_rate": p,
        "ler_per_round": bc.round_ler(p, R),
        "ler_per_round_per_logical": bc.round_ler(p, R * c["k"]),
        "cpu_time_seconds": cpu,
        "total_time_seconds": cpu,
        "time_per_shot_seconds": sps,
        "time_per_round": None if sps is None else sps / R,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default="O,C")
    ap.add_argument("--iters", default="32,40,58,109")
    ap.add_argument("--shots", type=int, default=0, help="override telemetry shots")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument(
        "--from-raw",
        action="store_true",
        help="rebuild records from benchmarking/data/raw/ instead of re-running",
    )
    args = ap.parse_args()

    commit = bc.git_commit()
    codes = [f"[{c.strip()}]" for c in args.codes.split(",")]
    iters = [int(x) for x in args.iters.split(",")]
    ds = bc.load_dataset(DATA_OUT)
    ds.update(
        {
            "family": FAMILY,
            "description": "Hyperbolic octagon cat-flagged circuits, r=16, SI1000 p=0.001, p_seam=0.01.",
            "git_commit": commit,
            "telemetry_note": (
                "Headline [O]/[C] = dripto3 cluster runs; stage_breakdown = separate local "
                "telemetry run (see stage_breakdown.telemetry_*). Stage percentages come from "
                "the telemetry run only."
            ),
        }
    )
    order = bc.record_order([c["tag"] for c in CIRCUITS])
    for c in CIRCUITS:
        if c["iter"] not in iters:
            continue
        bc.upsert(ds, headline(c, "[AQ]", commit))
        for code in codes:
            rec = headline(c, code, commit)
            shots = args.shots or TELEMETRY_SHOTS[c["iter"]]
            raw_p = bc.RAW_DIR / FAMILY / f"{c['tag']}_{code.strip('[]')}.json"
            if args.from_raw:
                raw = json.loads(raw_p.read_text())
                cmd, stats = raw["command_line"], raw["stats"]
            else:
                cmd, stats = bc.run_decoder(
                    code,
                    ["--circuit", circuit_path(c)],
                    ["--sample-num-shots", str(shots), "--sample-seed", str(args.seed)],
                    args.threads,
                )
            sb = bc.compute_stage_breakdown(stats, cmd)
            sb["telemetry_same_run_as_headline"] = False
            sb["telemetry_seed"] = args.seed
            rec["stage_breakdown"] = sb
            rec["raw_stats_path"] = bc.archive_raw(FAMILY, c["tag"], code, cmd, stats)
            bc.upsert(ds, rec)
            bc.save_dataset(DATA_OUT, ds, order)
            print(
                f"  {c['tag']} {code} telemetry: {stats['num_errors']}/{stats['num_shots']} "
                f"cpu={stats['total_time_seconds']:.1f}s",
                flush=True,
            )
    bc.save_dataset(DATA_OUT, ds, order)


if __name__ == "__main__":
    main()
