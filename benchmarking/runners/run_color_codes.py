#!/usr/bin/env python3
"""Reproducible runner: Superdense Color Codes (d=3..23, r=d, SI1000 p=0.001).

Runs all 7 canonical decoders across odd distances d=3..23:
  - [O] Tesseract-BP-OSD
  - [C] Tesseract-Cascade
  - [M] Multi-Pass Tesseract (beam 8, pqlimit 100k)
  - [G] Gari (Gari DEM + detector orders, beam 5, pqlimit 1M)
  - [B] Baseline Tesseract (beam 20, 16 det orders, pqlimit 200k)
  - [R] Relay-BP (external relay_bp)
  - [T] Telescoping-BP (external telescoping_decoder, preset_balanced)

Headline numbers and `stage_breakdown` come from the same `--stats-out` run.
All records include 1-sigma (`ler_1sigma`) and Wilson 95% (`ler_lo`, `ler_hi`)
error bounds and CPU time per round (`time_per_round`).

Run from the repo root:
  bazel build --jobs=1 //src:bp //src:tesseract
  python3 benchmarking/runners/run_color_codes.py --codes O,C,M,G,B,R,T --threads 64
  python3 benchmarking/runners/normalize_datasets.py
"""

import argparse
import glob
import json
import math
import os
import pathlib
import resource
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

RUNNER_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = RUNNER_DIR.parents[1]

sys.path[:0] = (
    [
        str(RUNNER_DIR),
        "/tmp/external_bp_bench/yarn/telescoping-decoder",
        str(REPO_ROOT / "bazel-bin/src"),
        str(REPO_ROOT / "src/py"),
        str(REPO_ROOT / "src"),
    ]
    + glob.glob(str(REPO_ROOT / "venv/lib/python*/site-packages"))
)

import bench_common as bc

DATA_OUT = bc.DATA_DIR / "color_codes.json"
SHOWDOWN_JSON = pathlib.Path(
    "/google/data/rw/users/ar/ariash/tesseract-bp/benchmarks/ultimate_showdown/results.json"
)
FAMILY = "color_codes"
ALL_DECODERS = ["[B]", "[G]", "[M]", "[R]", "[T]", "[O]", "[C]"]

CIRCUITS = [
    {"tag": "cc_d3", "d": 3, "q": 13, "n": 7},
    {"tag": "cc_d5", "d": 5, "q": 37, "n": 19},
    {"tag": "cc_d7", "d": 7, "q": 73, "n": 37},
    {"tag": "cc_d9", "d": 9, "q": 121, "n": 61},
    {"tag": "cc_d11", "d": 11, "q": 181, "n": 91},
    {"tag": "cc_d13", "d": 13, "q": 253, "n": 127},
    {"tag": "cc_d15", "d": 15, "q": 337, "n": 169},
    {"tag": "cc_d17", "d": 17, "q": 433, "n": 217},
    {"tag": "cc_d19", "d": 19, "q": 541, "n": 271},
    {"tag": "cc_d21", "d": 21, "q": 661, "n": 331},
    {"tag": "cc_d23", "d": 23, "q": 793, "n": 397},
]
CIRCUIT_BY_TAG = {c["tag"]: c for c in CIRCUITS}

# Default shot counts per (code, tag).
SHOTS: Dict[str, Dict[str, int]] = {
    "[O]": {
        "cc_d3": 50000,
        "cc_d5": 100000,
        "cc_d7": 100000,
        "cc_d9": 150000,
        "cc_d11": 100000,
        "cc_d13": 20000,
        "cc_d15": 10000,
        "cc_d17": 4096,
        "cc_d19": 4096,
        "cc_d21": 2048,
        "cc_d23": 2048,
    },
    "[C]": {
        "cc_d3": 50000,
        "cc_d5": 100000,
        "cc_d7": 100000,
        "cc_d9": 150000,
        "cc_d11": 100000,
        "cc_d13": 20000,
        "cc_d15": 10000,
        "cc_d17": 4096,
        "cc_d19": 4096,
        "cc_d21": 2048,
        "cc_d23": 2048,
    },
    "[M]": {
        "cc_d3": 100000,
        "cc_d5": 100000,
        "cc_d7": 100000,
        "cc_d9": 100000,
        "cc_d11": 25000,
        "cc_d13": 8000,
        "cc_d15": 4000,
        "cc_d17": 2000,
        "cc_d19": 1000,
        "cc_d21": 500,
        "cc_d23": 300,
    },
    "[G]": {
        "cc_d3": 50000,
        "cc_d5": 30000,
        "cc_d7": 50000,
        "cc_d9": 50000,
        "cc_d11": 2000,
        "cc_d13": 500,
        "cc_d15": 256,
        "cc_d17": 128,
        "cc_d19": 32,
        "cc_d21": 16,
        "cc_d23": 16,
    },
    "[B]": {
        "cc_d3": 30000,
        "cc_d5": 20000,
        "cc_d7": 8000,
        "cc_d9": 18000,
        "cc_d11": 256,
        "cc_d13": 128,
        "cc_d15": 64,
        "cc_d17": 16,
        "cc_d19": 16,
        "cc_d21": 16,
        "cc_d23": 16,
    },
    "[R]": {
        "cc_d3": 30000,
        "cc_d5": 20000,
        "cc_d7": 20000,
        "cc_d9": 15000,
        "cc_d11": 3000,
        "cc_d13": 1500,
        "cc_d15": 3000,
        "cc_d17": 500,
        "cc_d19": 200,
        "cc_d21": 150,
        "cc_d23": 500,
    },
    "[T]": {
        "cc_d3": 30000,
        "cc_d5": 20000,
        "cc_d7": 20000,
        "cc_d9": 15000,
        "cc_d11": 1500,
        "cc_d13": 500,
        "cc_d15": 500,
        "cc_d17": 150,
        "cc_d19": 100,
        "cc_d21": 64,
        "cc_d23": 300,
    },
}

MP_DEM_RECIPE = (
    "demutil.annotate_detector_bases(circuit.detector_error_model(decompose_errors=False, "
    "flatten_loops=True, allow_gauge_detectors=True, approximate_disjoint_errors=1.0))"
)
GARI_DEM_RECIPE = (
    "demutil.gari.circuit_to_gari(circuit, "
    "prior_function=demutil.gari.tesseract_xor_prior_probabilities)"
)
STAGE3_DECODER_STR = "Tesseract (--fallback-det-beam, capped at beam 14)"

_DET_CACHE: Dict[str, Tuple[int, int]] = {}


def circuit_path(c: dict) -> str:
    d = c["d"]
    return (
        f"testdata/colorcodes/r={d},d={d},p=0.001,noise=si1000,"
        f"c=superdense_color_code_Z,q={c['q']},gates=cz.stim"
    )


def get_circuit_size(rel_path: str) -> Tuple[int, int]:
    if rel_path not in _DET_CACHE:
        import stim

        ckt = stim.Circuit.from_file(REPO_ROOT / rel_path)
        _DET_CACHE[rel_path] = (ckt.num_detectors, ckt.num_observables)
    return _DET_CACHE[rel_path]


def wilson_interval(failures: int, shots: int, confidence: float = 0.95) -> Tuple[float, float]:
    if shots <= 0:
        return 0.0, 1.0
    z = 1.959963984540054 if abs(confidence - 0.95) < 1e-6 else 1.959963984540054
    n = float(shots)
    p = failures / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def enrich_stats_fields(rec: dict) -> dict:
    """Adds normalized LER bounds (ler_lo, ler_hi, ler_1sigma) and timing fields."""
    n = int(rec["num_shots"])
    n_err = int(rec["num_errors"])
    n_lc = int(rec.get("num_low_confidence", 0))
    r = int(rec.get("r") or rec.get("d") or 1)
    f = n_err + n_lc
    p_shot = f / n if n > 0 else 0.0
    p_lo, p_hi = wilson_interval(f, n, confidence=0.95)
    ler = bc.round_ler(p_shot, r)
    ler_lo = bc.round_ler(p_lo, r)
    ler_hi = bc.round_ler(p_hi, r)
    se = math.sqrt(p_shot * (1.0 - p_shot) / n) if f > 0 and n > 0 else 0.0
    ler_1sigma = abs(bc.round_ler(min(0.499, p_shot + se), r) - ler) if f > 0 else 0.0

    cpu = float(rec.get("cpu_time_seconds") or rec.get("total_time_seconds") or 0.0)
    tps = cpu / n if n > 0 else 0.0
    tpr = cpu / (n * r) if n > 0 and r > 0 else 0.0

    rec["shot_error_rate"] = p_shot
    rec["ler_per_round"] = ler
    rec["ler_lo"] = ler_lo
    rec["ler_hi"] = ler_hi
    rec["ler_1sigma"] = ler_1sigma
    rec["ler_upper_bound"] = (f == 0)
    rec["cpu_time_seconds"] = cpu
    rec["total_time_seconds"] = cpu
    rec["time_per_shot_seconds"] = tps
    rec["time_per_shot"] = tps
    rec["time_per_round"] = tpr
    if rec.get("circuit_path"):
        nd, no = get_circuit_size(rec["circuit_path"])
        rec["num_detectors"] = nd
        rec["num_observables"] = no
    if rec.get("decoder_code") == "[C]":
        rec["stage3_decoder"] = STAGE3_DECODER_STR
    return rec


def ensure_dems(c: dict, dem_dir: pathlib.Path) -> Tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    import stim
    from tesseract_decoder import demutil

    dem_dir.mkdir(parents=True, exist_ok=True)
    tag = c["tag"]
    rel = circuit_path(c)
    mp_p = dem_dir / f"{tag}_mp.dem"
    gari_p = dem_dir / f"{tag}_gari.dem"
    gari_ord_p = dem_dir / f"{tag}_gari_ord.json"

    if not mp_p.exists() or not gari_p.exists() or not gari_ord_p.exists():
        ckt = stim.Circuit.from_file(REPO_ROOT / rel)
        if not mp_p.exists():
            dem = demutil.annotate_detector_bases(
                ckt.detector_error_model(
                    decompose_errors=False,
                    flatten_loops=True,
                    allow_gauge_detectors=True,
                    approximate_disjoint_errors=1.0,
                )
            )
            dem.to_file(mp_p)
        if not gari_p.exists() or not gari_ord_p.exists():
            gari_dem = demutil.gari.circuit_to_gari(
                ckt,
                prior_function=demutil.gari.tesseract_xor_prior_probabilities,
            )
            gari_dem.to_file(gari_p)
            orders = demutil.gari.build_detector_orders(ckt, gari_dem, 1, seed=0)
            gari_ord_p.write_text(json.dumps(orders))
    return mp_p, gari_p, gari_ord_p


def dem_to_sparse_matrices(dem):
    import numpy as np
    from scipy.sparse import csc_matrix

    det_rows, det_cols = [], []
    obs_rows, obs_cols = [], []
    priors = []
    col_idx = 0
    for inst in dem.flattened():
        if inst.type == "error":
            p = inst.args_copy()[0]
            priors.append(p)
            dets = set()
            obss = set()
            for t in inst.targets_copy():
                if t.is_relative_detector_id():
                    if t.val in dets:
                        dets.remove(t.val)
                    else:
                        dets.add(t.val)
                elif t.is_logical_observable_id():
                    if t.val in obss:
                        obss.remove(t.val)
                    else:
                        obss.add(t.val)
            for d in dets:
                det_rows.append(d)
                det_cols.append(col_idx)
            for o in obss:
                obs_rows.append(o)
                obs_cols.append(col_idx)
            col_idx += 1
    H = csc_matrix(
        (np.ones(len(det_rows), dtype=np.uint8), (det_rows, det_cols)),
        shape=(dem.num_detectors, col_idx),
    )
    L = csc_matrix(
        (np.ones(len(obs_rows), dtype=np.uint8), (obs_rows, obs_cols)),
        shape=(dem.num_observables, col_idx),
    )
    return H, L, np.array(priors, dtype=np.float64)


def run_cli_decoder(
    code: str,
    c: dict,
    shots: int,
    seed: int,
    threads: int,
    dem_dir: pathlib.Path,
) -> Tuple[str, dict]:
    rel = circuit_path(c)
    if code in ("[O]", "[C]"):
        return bc.run_decoder(
            code,
            ["--circuit", rel],
            ["--sample-num-shots", str(shots), "--sample-seed", str(seed)],
            threads,
        )
    if code == "[M]":
        mp_p, _, _ = ensure_dems(c, dem_dir)
        return bc.run_decoder(
            code,
            ["--circuit", rel],
            ["--sample-num-shots", str(shots), "--sample-seed", str(seed)],
            threads,
            ["--dem", str(mp_p)],
        )
    if code == "[G]":
        _, gari_p, gari_ord_p = ensure_dems(c, dem_dir)
        cmd = [
            bc.TESSERACT_BIN,
            "--circuit",
            rel,
            "--dem",
            str(gari_p),
            "--detector-orders",
            str(gari_ord_p),
            "--sample-num-shots",
            str(shots),
            "--sample-seed",
            str(seed),
            "--threads",
            str(threads),
            "--stats-out",
            "-",
            "--beam",
            "5",
            "--beam-climbing",
            "--no-revisit-dets",
            "--pqlimit",
            "1000000",
        ]
        print("RUN", " ".join(cmd), flush=True)
        t0 = time.perf_counter()
        out = subprocess.check_output(cmd, text=True, cwd=REPO_ROOT)
        wall_s = time.perf_counter() - t0
        stats = json.loads(out.strip().splitlines()[-1])
        stats.setdefault("wall_time_seconds", wall_s)
        return " ".join(cmd), stats
    if code == "[B]":
        cmd = [
            bc.TESSERACT_BIN,
            "--circuit",
            rel,
            "--sample-num-shots",
            str(shots),
            "--sample-seed",
            str(seed),
            "--threads",
            str(threads),
            "--stats-out",
            "-",
            "--beam",
            "20",
            "--beam-climbing",
            "--no-revisit-dets",
            "--num-det-orders",
            "16",
            "--pqlimit",
            "200000",
        ]
        print("RUN", " ".join(cmd), flush=True)
        t0 = time.perf_counter()
        out = subprocess.check_output(cmd, text=True, cwd=REPO_ROOT)
        wall_s = time.perf_counter() - t0
        stats = json.loads(out.strip().splitlines()[-1])
        stats.setdefault("wall_time_seconds", wall_s)
        return " ".join(cmd), stats
    raise ValueError(f"Unsupported CLI decoder code: {code}")


def run_relay_bp_decoder(c: dict, shots: int, seed: int, threads: int) -> Tuple[str, dict]:
    import numpy as np
    import relay_bp
    import stim

    os.environ["RAYON_NUM_THREADS"] = str(threads)
    rel = circuit_path(c)
    ckt = stim.Circuit.from_file(REPO_ROOT / rel)
    dem = ckt.detector_error_model(
        decompose_errors=False,
        flatten_loops=True,
        allow_gauge_detectors=True,
        approximate_disjoint_errors=1.0,
    )
    H, L, priors = dem_to_sparse_matrices(dem)
    syndromes, actual_obs = ckt.compile_detector_sampler(seed=seed).sample(
        shots=shots, separate_observables=True
    )
    synd_u8 = syndromes.astype(np.uint8)
    obs_u8 = actual_obs.astype(np.uint8)

    dec = relay_bp.RelayDecoderF32(
        H,
        error_priors=priors,
        gamma0=0.1,
        pre_iter=80,
        num_sets=5,
        set_max_iter=60,
        gamma_dist_interval=(-0.24, 0.66),
        stop_nconv=1,
    )
    runner = relay_bp.ObservableDecoderRunner(dec, L, include_decode_result=False)
    t_wall0 = time.perf_counter()
    t_cpu0 = time.process_time()
    preds = runner.decode_observables_batch(synd_u8, parallel=True, progress_bar=False)
    cpu_s = time.process_time() - t_cpu0
    wall_s = time.perf_counter() - t_wall0

    errs = int(np.any(preds != obs_u8, axis=1).sum())
    cmd_str = (
        f'relay_bp.RelayDecoderF32(gamma0=0.1, pre_iter=80, num_sets=5, set_max_iter=60) '
        f'--circuit "{rel}" --sample-num-shots {shots} --sample-seed {seed}'
    )
    stats = {
        "num_shots": shots,
        "num_errors": errs,
        "num_low_confidence": 0,
        "total_time_seconds": cpu_s,
        "wall_time_seconds": wall_s,
        "sample_seed": seed,
    }
    return cmd_str, stats


def run_telescoping_bp_decoder(c: dict, shots: int, seed: int, threads: int) -> Tuple[str, dict]:
    import numpy as np
    import stim
    from telescoping_decoder import TelescopeConfig, TelescopingDecoder, circuit_to_dem

    os.environ["XDG_CACHE_HOME"] = "/tmp/external_bp_bench/telescoping_cache"
    os.environ["TELESCOPING_DECODER_MARCH_NATIVE"] = "1"
    rel = circuit_path(c)
    ckt = stim.Circuit.from_file(REPO_ROOT / rel)
    dem = circuit_to_dem(ckt)
    coords = dem.get_detector_coordinates()
    mask = np.array(
        [int(round(coords[i][3])) in (0, 1, 2) for i in range(dem.num_detectors)],
        dtype=bool,
    )
    syndromes, actual_obs = ckt.compile_detector_sampler(seed=seed).sample(
        shots=shots, separate_observables=True
    )
    synd_u8 = syndromes.astype(np.uint8)
    obs_u8 = actual_obs.astype(np.uint8)

    cfg = TelescopeConfig(use_gari=True)
    cfg.s1.enabled = False
    cfg.s2.enabled = False
    cfg.s3.n_procs = threads
    cfg.s3.mp_start_method = "fork"
    cfg.s3.system = "auto"
    cfg.s4.enabled = False

    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t_wall0 = time.perf_counter()
    with TelescopingDecoder.from_dem(
        dem, is_x_detector=mask, init_basis="Z", config=cfg
    ) as tdec:
        res = tdec.decode(synd_u8, true_obs=obs_u8)
    wall_s = time.perf_counter() - t_wall0
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)

    errs = int(res.le.sum())
    n_nc = int(res.diagnostics.get("n_nc", 0))
    cmd_str = (
        f'TelescopingDecoder(TelescopeConfig.preset_balanced(num_threads={threads})) '
        f'--circuit "{rel}" --sample-num-shots {shots} --sample-seed {seed}'
    )
    stats = {
        "num_shots": shots,
        "num_errors": errs,
        "num_low_confidence": n_nc,
        "total_time_seconds": child_cpu,
        "wall_time_seconds": wall_s,
        "sample_seed": seed,
    }
    return cmd_str, stats


def import_showdown_external_records(ds: dict, commit: str, threads: int = 64) -> int:
    """Imports validated [R] and [T] (and [M5]) records for cc_d15 and cc_d23 from showdown results.json."""
    if not SHOWDOWN_JSON.exists():
        return 0
    rows = json.loads(SHOWDOWN_JSON.read_text())
    existing = {(r["tag"], r["decoder_code"]): r for r in ds.get("records", [])}
    imported = 0
    mode_to_code = {
        "relay_bp": "[R]",
        "telescoping_bp": "[T]",
        "mp_tesseract_b5": "[M5]",
    }
    for row in rows:
        tag = row.get("tag")
        mode = row.get("mode")
        if tag not in ("cc_d15", "cc_d23") or mode not in mode_to_code:
            continue
        code = mode_to_code[mode]
        if (tag, code) in existing and existing[(tag, code)].get("num_shots", 0) >= row["num_shots"]:
            continue
        c = CIRCUIT_BY_TAG[tag]
        rel = circuit_path(c)
        shots = int(row["num_shots"])
        seed = 12345
        if code == "[R]":
            cmd = (
                f'relay_bp.RelayDecoderF32(gamma0=0.1, pre_iter=80, num_sets=5, set_max_iter=60) '
                f'--circuit "{rel}" --sample-num-shots {shots} --sample-seed {seed}'
            )
        elif code == "[T]":
            cmd = (
                f'TelescopingDecoder(TelescopeConfig.preset_balanced(num_threads={threads})) '
                f'--circuit "{rel}" --sample-num-shots {shots} --sample-seed {seed}'
            )
        else:
            cmd = (
                f'./bazel-bin/src/tesseract --dem <mp.dem> --multipass --num-passes 2 '
                f'--pqlimit 1000000 --det-beam 5 --beam-climbing --no-revisit-dets '
                f'--circuit "{rel}" --sample-num-shots {shots} --sample-seed {seed}'
            )
        meta = {
            "tag": tag,
            "label": f"color code, d={c['d']}, q={c['q']}",
            "family": FAMILY,
            "d": c["d"],
            "r": c["d"],
            "q": c["q"],
            "n": c["n"],
            "k": 1,
            "p": 0.001,
            "circuit_path": rel,
        }
        stats = {
            "num_shots": shots,
            "num_errors": int(row["num_errors"]),
            "num_low_confidence": int(row.get("num_low_confidence", 0)),
            "total_time_seconds": float(row["total_time_seconds"]),
            "wall_time_seconds": float(row.get("wall_time_seconds", 0.0)),
            "sample_seed": seed,
        }
        rec = bc.make_record(meta, code, cmd, stats, seed, commit)
        rec["source"] = "benchmarking/ultimate_showdown/results.json (validated 64-core run)"
        rec["raw_stats_path"] = bc.archive_raw(FAMILY, tag, code, cmd, stats)
        enrich_stats_fields(rec)
        bc.upsert(ds, rec)
        imported += 1
    return imported


def update_distance_sweep(ds: dict) -> None:
    ds["records"] = [enrich_stats_fields(r) for r in ds.get("records", [])]
    ds["distance_sweep"] = [
        r
        for r in ds["records"]
        if r["decoder_code"] in ("[O]", "[C]") and r.get("d") in (5, 7, 9, 11, 13, 15, 17, 19, 21, 23)
    ]
    ds["distance_sweep_note"] = (
        "Copies of the canonical [O]/[C] records at d=5..23 (same runs as `records`)."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default="O,C,M,G,B,R,T")
    ap.add_argument("--tags", default=",".join(c["tag"] for c in CIRCUITS))
    ap.add_argument("--shots", type=int, default=0, help="override default shots")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--dem-dir", default="/tmp/tesseract_bench_dems")
    ap.add_argument(
        "--only-missing-or-smaller",
        action="store_true",
        help="Skip (tag, code) pairs that already have >= target shots in color_codes.json",
    )
    ap.add_argument(
        "--import-showdown",
        action="store_true",
        default=True,
        help="Import validated cc_d15/cc_d23 [R]/[T] records from ultimate_showdown/results.json",
    )
    args = ap.parse_args()

    commit = bc.git_commit()
    codes = [f"[{c.strip().strip('[]')}]" for c in args.codes.split(",") if c.strip()]
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    dem_dir = pathlib.Path(args.dem_dir)

    ds = bc.load_dataset(DATA_OUT)
    ds.update(
        {
            "family": FAMILY,
            "description": (
                "Superdense Color Code Z-memory circuits (SI1000, p=0.001, r=d), d=3..23, "
                "across all 7 decoders ([B], [G], [M], [R], [T], [O], [C])."
            ),
            "git_commit": commit,
            "ler_per_round_formula": bc.LER_FORMULA,
            "decoder_codes": {c: m[2] for c, m in bc.CODE_META.items()},
        }
    )

    if args.import_showdown:
        n_imp = import_showdown_external_records(ds, commit, args.threads)
        if n_imp:
            print(f"Imported {n_imp} validated showdown records into color_codes.json", flush=True)

    update_distance_sweep(ds)
    bc.save_dataset(DATA_OUT, ds, bc.record_order([x["tag"] for x in CIRCUITS]))

    existing_map = {(r["tag"], r["decoder_code"]): r for r in ds.get("records", [])}

    for c in CIRCUITS:
        if c["tag"] not in tags:
            continue
        rel = circuit_path(c)
        meta = {
            "tag": c["tag"],
            "label": f"color code, d={c['d']}, q={c['q']}",
            "family": FAMILY,
            "d": c["d"],
            "r": c["d"],
            "q": c["q"],
            "n": c["n"],
            "k": 1,
            "p": 0.001,
            "circuit_path": rel,
        }
        for code in codes:
            shots = args.shots or SHOTS.get(code, {}).get(c["tag"], 0)
            if not shots:
                continue
            prev = existing_map.get((c["tag"], code))
            if args.only_missing_or_smaller and prev is not None and int(prev.get("num_shots", 0)) >= shots:
                print(
                    f"  [SKIP existing] {c['tag']} {code}: has {prev['num_shots']} shots (>= {shots})",
                    flush=True,
                )
                continue

            if code in ("[O]", "[C]", "[M]", "[G]", "[B]"):
                cmd, stats = run_cli_decoder(code, c, shots, args.seed, args.threads, dem_dir)
            elif code == "[R]":
                cmd, stats = run_relay_bp_decoder(c, shots, args.seed, args.threads)
            elif code == "[T]":
                cmd, stats = run_telescoping_bp_decoder(c, shots, args.seed, args.threads)
            else:
                raise ValueError(f"Unknown decoder code: {code}")

            rec = bc.make_record(meta, code, cmd, stats, args.seed, commit)
            if code == "[M]":
                rec["dem_recipe"] = MP_DEM_RECIPE
            elif code == "[G]":
                rec["dem_recipe"] = GARI_DEM_RECIPE
            rec["raw_stats_path"] = bc.archive_raw(FAMILY, c["tag"], code, cmd, stats)
            enrich_stats_fields(rec)
            bc.upsert(ds, rec)
            existing_map[(c["tag"], code)] = rec
            update_distance_sweep(ds)
            bc.save_dataset(DATA_OUT, ds, bc.record_order([x["tag"] for x in CIRCUITS]))
            print(
                f"  [DONE] {c['tag']} {code}: errs={rec['num_errors']}(lc={rec['num_low_confidence']})/{rec['num_shots']} "
                f"ler/r={rec['ler_per_round']:.3e} [lo={rec['ler_lo']:.3e}, hi={rec['ler_hi']:.3e}, 1sig={rec['ler_1sigma']:.3e}] "
                f"t/r={rec['time_per_round']:.3e}s cpu={rec['cpu_time_seconds']:.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
