#!/usr/bin/env python3
"""Reproducible runner: Bivariate Bicycle Gross Code [[144, 12, 12]] physical error rate (p) sweep.

Collects benchmark records for the Gross code [[144, 12, 12]] (d=12, r=12, q=288, k=12, n=144)
across physical error rates p in {0.0005, 0.001, 0.002, 0.003, 0.004, 0.005} for all 7 decoders:
  - [O] Tesseract-BP-OSD
  - [C] Tesseract-Cascade
  - [M] Multi-Pass Tesseract (beam 8, pqlimit 100k)
  - [G] Gari (Gari DEM + detector orders, beam 5, pqlimit 1M)
  - [B] Baseline Tesseract (beam 20, 16 det orders, pqlimit 200k)
  - [R] Relay-BP (external relay_bp)
  - [T] Telescoping-BP (external telescoping_decoder)

Reuses existing p = 0.001 records from benchmarking/data/bivariate_bicycle.json so p = 0.001
matches 100%. Saves normalized records to benchmarking/data/bb_gross_psweep.json and raw stats
to benchmarking/data/raw/bb_gross_psweep/.
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

DATA_OUT = bc.DATA_DIR / "bb_gross_psweep.json"
BB_FIXED_JSON = bc.DATA_DIR / "bivariate_bicycle.json"
FAMILY = "bb_gross_psweep"
BASE_TAG = "bb_d12_gross"

CODE_PARAMS = {
    "base_tag": BASE_TAG,
    "code": BASE_TAG,
    "family": "bivariate_bicycle",
    "d": 12,
    "r": 12,
    "q": 288,
    "n": 144,
    "k": 12,
    "num_detectors": 1728,
    "num_observables": 12,
}

P_VALUES = [0.0005, 0.001, 0.002, 0.003, 0.004, 0.005]

ALL_DECODERS = ["[B]", "[G]", "[M]", "[R]", "[T]", "[O]", "[C]"]

# Default shot counts per decoder and p value
DEFAULT_SHOTS: Dict[str, Dict[float, int]] = {
    "[O]": {
        0.0005: 50000,
        0.001: 50000,
        0.002: 20000,
        0.003: 2048,
        0.004: 1024,
        0.005: 1024,
    },
    "[C]": {
        0.0005: 50000,
        0.001: 50000,
        0.002: 20000,
        0.003: 2048,
        0.004: 128,
        0.005: 128,
    },
    "[M]": {
        0.0005: 5000,
        0.001: 50000,
        0.002: 2000,
        0.003: 500,
        0.004: 200,
        0.005: 128,
    },
    "[G]": {
        0.0005: 500,
        0.001: 1000,
        0.002: 256,
        0.003: 128,
        0.004: 64,
        0.005: 64,
    },
    "[B]": {
        0.0005: 64,
        0.001: 200,
        0.002: 64,
        0.003: 64,
        0.004: 64,
        0.005: 64,
    },
    "[R]": {
        0.0005: 3000,
        0.001: 3000,
        0.002: 2000,
        0.003: 1000,
        0.004: 500,
        0.005: 128,
    },
    "[T]": {
        0.0005: 5000,
        0.001: 5000,
        0.002: 3000,
        0.003: 1500,
        0.004: 1000,
        0.005: 128,
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
STAGE3_DECODER_STR = "Tesseract (beam 12, 4 detector orders, pqlimit 5e4)"


def circuit_rel_path(p: float) -> str:
    return (
        f"testdata/bivariatebicyclecodes/r=12,d=12,p={p:g},noise=si1000,"
        "c=bivariate_bicycle_Z,nkd=[[144,12,12]],q=288,iscolored=True,"
        "A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim"
    )


def tag_for_p(p: float) -> str:
    if abs(p - 0.001) < 1e-9:
        return BASE_TAG
    return f"{BASE_TAG}_p{p:g}"


def wilson_interval(failures: int, shots: int, confidence: float = 0.95) -> Tuple[float, float]:
    if shots <= 0:
        return 0.0, 1.0
    # z for 95% confidence = 1.959963984540054
    z = 1.959963984540054 if abs(confidence - 0.95) < 1e-6 else 1.959963984540054
    n = float(shots)
    p = failures / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def enrich_stats_fields(rec: dict) -> dict:
    """Adds normalized LER bounds and timing fields expected by export_static.py and verifiers."""
    n = int(rec["num_shots"])
    n_err = int(rec["num_errors"])
    n_lc = int(rec.get("num_low_confidence", 0))
    r = int(rec.get("r", 12))
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

    rec["base_tag"] = BASE_TAG
    rec["code"] = BASE_TAG
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
    rec["num_detectors"] = CODE_PARAMS["num_detectors"]
    rec["num_observables"] = CODE_PARAMS["num_observables"]
    if rec.get("decoder_code") == "[C]":
        rec["stage3_decoder"] = STAGE3_DECODER_STR
    return rec


def ensure_dems(p: float, dem_dir: pathlib.Path) -> Tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    import stim
    from tesseract_decoder import demutil

    dem_dir.mkdir(parents=True, exist_ok=True)
    rel = circuit_rel_path(p)
    mp_p = dem_dir / f"bb_d12_gross_p{p:g}_mp.dem"
    gari_p = dem_dir / f"bb_d12_gross_p{p:g}_gari.dem"
    gari_ord_p = dem_dir / f"bb_d12_gross_p{p:g}_gari_ord.json"

    if not mp_p.exists() or not gari_p.exists() or not gari_ord_p.exists():
        ckt = stim.Circuit.from_file(REPO_ROOT / rel)
        if not mp_p.exists():
            src_dem = ckt.detector_error_model(
                decompose_errors=False,
                flatten_loops=True,
                allow_gauge_detectors=True,
                approximate_disjoint_errors=1.0,
            )
            mp_dem = demutil.annotate_detector_bases(src_dem)
            mp_dem.to_file(mp_p)
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
    p: float,
    shots: int,
    seed: int,
    threads: int,
    dem_dir: pathlib.Path,
) -> Tuple[str, dict]:
    rel = circuit_rel_path(p)
    if code in ("[O]", "[C]"):
        return bc.run_decoder(
            code,
            ["--circuit", rel],
            ["--sample-num-shots", str(shots), "--sample-seed", str(seed)],
            threads,
        )
    if code == "[M]":
        mp_p, _, _ = ensure_dems(p, dem_dir)
        return bc.run_decoder(
            code,
            ["--circuit", rel],
            ["--sample-num-shots", str(shots), "--sample-seed", str(seed)],
            threads,
            ["--dem", str(mp_p)],
        )
    if code == "[G]":
        _, gari_p, gari_ord_p = ensure_dems(p, dem_dir)
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


def run_relay_bp_decoder(p: float, shots: int, seed: int, threads: int) -> Tuple[str, dict]:
    import numpy as np
    import relay_bp
    import stim

    os.environ["RAYON_NUM_THREADS"] = str(threads)
    rel = circuit_rel_path(p)
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


def run_telescoping_bp_decoder(p: float, shots: int, seed: int, threads: int) -> Tuple[str, dict]:
    import numpy as np
    import stim
    from telescoping_decoder import TelescopeConfig, TelescopingDecoder, circuit_to_dem

    os.environ["XDG_CACHE_HOME"] = "/tmp/external_bp_bench/telescoping_cache"
    os.environ["TELESCOPING_DECODER_MARCH_NATIVE"] = "1"
    rel = circuit_rel_path(p)
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


def load_p001_from_bivariate_bicycle() -> Dict[str, dict]:
    """Loads the canonical p=0.001 Gross code records from benchmarking/data/bivariate_bicycle.json."""
    out: Dict[str, dict] = {}
    if not BB_FIXED_JSON.exists():
        return out
    data = json.loads(BB_FIXED_JSON.read_text())
    for row in data.get("records", []):
        if row.get("tag") == BASE_TAG and row.get("decoder_code") in ALL_DECODERS:
            rec = dict(row)
            enrich_stats_fields(rec)
            out[rec["decoder_code"]] = rec
    return out


def sort_records(records: List[dict]) -> List[dict]:
    def key(r: dict):
        p_val = float(r.get("p", 0.001))
        p_idx = next((i for i, pv in enumerate(P_VALUES) if abs(pv - p_val) < 1e-7), 99)
        c_idx = bc.CODE_ORDER.index(r["decoder_code"]) if r["decoder_code"] in bc.CODE_ORDER else 99
        return (p_idx, p_val, c_idx)

    return sorted(records, key=key)


def upsert_psweep(ds: dict, rec: dict) -> None:
    p_val = round(float(rec["p"]), 7)
    code = rec["decoder_code"]
    ds["records"] = [
        r
        for r in ds.get("records", [])
        if not (abs(float(r.get("p", 0.0)) - p_val) < 1e-7 and r.get("decoder_code") == code)
    ] + [rec]
    ds["records"] = sort_records(ds["records"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Bivariate Bicycle Gross [[144,12,12]] p-sweep.")
    ap.add_argument("--codes", default=",".join(c.strip("[]") for c in ALL_DECODERS))
    ap.add_argument("--ps", default=",".join(f"{p:g}" for p in P_VALUES))
    ap.add_argument("--shots", type=int, default=0, help="Override default shot count if > 0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--dem-dir", default="/tmp/tesseract_bench_dems")
    ap.add_argument("--force-p001", action="store_true", help="Re-run p=0.001 instead of reusing bivariate_bicycle.json")
    ap.add_argument("--skip-existing", action="store_true", help="Skip (p, code) pairs already present in bb_gross_psweep.json")
    args = ap.parse_args()

    commit = bc.git_commit()
    codes = [f"[{c.strip().strip('[]')}]" for c in args.codes.split(",") if c.strip()]
    ps = [float(x.strip()) for x in args.ps.split(",") if x.strip()]
    dem_dir = pathlib.Path(args.dem_dir)

    ds = bc.load_dataset(DATA_OUT)
    ds.update(
        {
            "family": FAMILY,
            "code": BASE_TAG,
            "description": (
                "Bivariate Bicycle Gross Code [[144, 12, 12]] (d=12, r=12, q=288, k=12, SI1000) "
                "physical error rate sweep across p in {0.0005, 0.001, 0.002, 0.003, 0.004, 0.005} "
                "for all 7 decoders ([B], [G], [M], [R], [T], [O], [C])."
            ),
            "git_commit": commit,
            "ler_per_round_formula": bc.LER_FORMULA,
            "decoder_codes": {c: bc.CODE_META[c][2] for c in ALL_DECODERS},
        }
    )

    p001_canonical = load_p001_from_bivariate_bicycle()

    existing_keys = {
        (round(float(r.get("p", 0.0)), 7), r.get("decoder_code"))
        for r in ds.get("records", [])
    }

    for p in ps:
        p_round = round(p, 7)
        tag = tag_for_p(p)
        rel = circuit_rel_path(p)
        meta = {
            "tag": tag,
            "base_tag": BASE_TAG,
            "code": BASE_TAG,
            "label": f"Gross bivariate bicycle [[144,12,12]], d=12, q=288, p={p:g}",
            "family": "bivariate_bicycle",
            "d": 12,
            "r": 12,
            "q": 288,
            "n": 144,
            "k": 12,
            "p": p,
            "circuit_path": rel,
        }

        for code in codes:
            if abs(p - 0.001) < 1e-7 and not args.force_p001 and code in p001_canonical:
                rec = dict(p001_canonical[code])
                upsert_psweep(ds, rec)
                bc.save_dataset(DATA_OUT, ds)
                print(
                    f"  [REUSE p=0.001] {tag} {code}: {rec['num_errors']}/{rec['num_shots']} "
                    f"ler/r={rec['ler_per_round']:.3e} t/r={rec['time_per_round']:.3e}s",
                    flush=True,
                )
                continue

            if args.skip_existing and (p_round, code) in existing_keys:
                print(f"  [SKIP existing] {tag} {code}", flush=True)
                continue

            shots = args.shots or DEFAULT_SHOTS.get(code, {}).get(p, 1000)
            if code in ("[O]", "[C]", "[M]", "[G]", "[B]"):
                cmd, stats = run_cli_decoder(code, p, shots, args.seed, args.threads, dem_dir)
            elif code == "[R]":
                cmd, stats = run_relay_bp_decoder(p, shots, args.seed, args.threads)
            elif code == "[T]":
                cmd, stats = run_telescoping_bp_decoder(p, shots, args.seed, args.threads)
            else:
                raise ValueError(f"Unknown decoder code: {code}")

            rec = bc.make_record(meta, code, cmd, stats, args.seed, commit)
            if code == "[M]":
                rec["dem_recipe"] = MP_DEM_RECIPE
            elif code == "[G]":
                rec["dem_recipe"] = GARI_DEM_RECIPE
            rec["raw_stats_path"] = bc.archive_raw(FAMILY, tag, code, cmd, stats)
            enrich_stats_fields(rec)
            upsert_psweep(ds, rec)
            bc.save_dataset(DATA_OUT, ds)
            print(
                f"  [DONE] {tag} (p={p:g}) {code}: errs={rec['num_errors']}(lc={rec['num_low_confidence']})/{rec['num_shots']} "
                f"ler/r={rec['ler_per_round']:.3e} [lo={rec['ler_lo']:.3e}, hi={rec['ler_hi']:.3e}, 1sig={rec['ler_1sigma']:.3e}] "
                f"t/r={rec['time_per_round']:.3e}s cpu={rec['cpu_time_seconds']:.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
