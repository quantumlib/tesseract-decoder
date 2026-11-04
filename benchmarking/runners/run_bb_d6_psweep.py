#!/usr/bin/env python3
"""Reproducible physical error rate (p) sweep runner for Bivariate Bicycle [[72, 12, 6]] (bb_d6).

Sweeps physical error rate p across {0.0001, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005}
on SI1000 circuits:
  testdata/bivariatebicyclecodes/r=6,d=6,p=<p>,noise=si1000,c=bivariate_bicycle_Z,nkd=[[72,12,6]],q=144,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim

Includes all 7 decoders:
  [O] Tesseract-BP-OSD
  [C] Tesseract-Cascade ([O] + --tesseract-fallback)
  [M] Multi-Pass Tesseract (--dem <mp.dem> --multipass --num-passes 2 --beam 8 --beam-climbing --no-revisit-dets --pqlimit 100000)
  [G] Gari (--dem <gari.dem> --detector-orders <gari_ord.json> --beam 5 --beam-climbing --no-revisit-dets --pqlimit 1000000)
  [B] Baseline Tesseract (--beam 20 --beam-climbing --no-revisit-dets --num-det-orders 16 --pqlimit 200000)
  [R] Relay-BP (RelayDecoderF32(gamma0=0.1, pre_iter=80, num_sets=5, set_max_iter=60, gamma_dist_interval=(-0.24, 0.66), stop_nconv=1))
  [T] Telescoping-BP (TelescopingDecoder with TelescopeConfig(use_gari=True), s3 auto)

At p = 0.001, reuses existing canonical records from benchmarking/data/bivariate_bicycle.json
(and benchmarking/multipass/results.json for [R] and [T]) so p = 0.001 matches 100%.
"""

import argparse
import copy
import json
import math
import os
import pathlib
import resource
import sys
import time
from typing import Dict, List, Optional, Tuple

import bench_common as bc

REPO_ROOT = bc.REPO_ROOT
for _p in (
    str(REPO_ROOT / "bazel-bin/src"),
    str(REPO_ROOT / "src/py"),
    "/tmp/external_bp_bench/yarn/telescoping-decoder",
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import stim
from scipy.sparse import csc_matrix
from tesseract_decoder import demutil

DATA_OUT = bc.DATA_DIR / "bb_d6_psweep.json"
BB_DATA_PATH = bc.DATA_DIR / "bivariate_bicycle.json"
MP_RESULTS_PATH = REPO_ROOT / "benchmarking/multipass/results.json"
FAMILY = "bb_d6_psweep"

BASE_META = {
    "tag": "bb_d6",
    "base_tag": "bb_d6",
    "code": "bb_d6",
    "label": "bivariate bicycle [[72,12,6]], d=6, q=144",
    "family": "bivariate_bicycle",
    "d": 6,
    "r": 6,
    "q": 144,
    "n": 72,
    "k": 12,
    "num_detectors": 432,
    "num_observables": 12,
}

P_VALUES = [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005]

# Default shot counts per decoder and p value.
SHOTS: Dict[str, Dict[float, int]] = {
    "[O]": {
        0.0001: 1000000,
        0.0005: 250000,
        0.001: 100000,
        0.002: 50000,
        0.003: 20000,
        0.004: 20000,
        0.005: 20000,
    },
    "[C]": {
        0.0001: 1000000,
        0.0005: 250000,
        0.001: 100000,
        0.002: 25000,
        0.003: 5000,
        0.004: 2000,
        0.005: 2000,
    },
    "[M]": {
        0.0001: 100000,
        0.0005: 200000,
        0.001: 100000,
        0.002: 10000,
        0.003: 2000,
        0.004: 1000,
        0.005: 1000,
    },
    "[G]": {
        0.0001: 50000,
        0.0005: 100000,
        0.001: 50000,
        0.002: 5000,
        0.003: 1000,
        0.004: 500,
        0.005: 500,
    },
    "[B]": {
        0.0001: 20000,
        0.0005: 10000,
        0.001: 6000,
        0.002: 500,
        0.003: 200,
        0.004: 128,
        0.005: 128,
    },
    "[R]": {
        0.0001: 50000,
        0.0005: 100000,
        0.001: 100000,
        0.002: 20000,
        0.003: 10000,
        0.004: 5000,
        0.005: 5000,
    },
    "[T]": {
        0.0001: 40000,
        0.0005: 100000,
        0.001: 20000,
        0.002: 10000,
        0.003: 5000,
        0.004: 5000,
        0.005: 5000,
    },
}

MP_DEM_RECIPE = (
    "demutil.annotate_detector_bases(circuit.detector_error_model("
    "decompose_errors=False, flatten_loops=True, allow_gauge_detectors=True, "
    "approximate_disjoint_errors=1.0))"
)
GARI_DEM_RECIPE = (
    "demutil.gari.circuit_to_gari(circuit, "
    "prior_function=demutil.gari.tesseract_xor_prior_probabilities); "
    "demutil.gari.build_detector_orders(circuit, gari_dem, 1, seed=0)"
)

EXTRA_DECODERS = {
    "[G]": {
        "binary": bc.TESSERACT_BIN,
        "flags": [
            "--beam",
            "5",
            "--beam-climbing",
            "--no-revisit-dets",
            "--pqlimit",
            "1000000",
        ],
    },
    "[B]": {
        "binary": bc.TESSERACT_BIN,
        "flags": [
            "--beam",
            "20",
            "--beam-climbing",
            "--no-revisit-dets",
            "--num-det-orders",
            "16",
            "--pqlimit",
            "200000",
        ],
    },
}


def circuit_rel_path(p: float) -> str:
    return (
        f"testdata/bivariatebicyclecodes/r=6,d=6,p={p:g},noise=si1000,"
        "c=bivariate_bicycle_Z,nkd=[[72,12,6]],q=144,iscolored=True,"
        "A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim"
    )


def wilson_interval(f: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = f / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(max(0.0, p * (1.0 - p) / n + z * z / (4.0 * n * n)))
    return (max(0.0, center - half), min(1.0, center + half))


def enrich_stats_fields(rec: dict) -> dict:
    """Computes normalized LER + Wilson bounds + 1-sigma + timing fields in-place."""
    n = int(rec["num_shots"])
    f = int(rec["num_errors"]) + int(rec.get("num_low_confidence", 0))
    r = int(rec.get("r", 6))
    k = int(rec.get("k", 12))
    p_shot = f / n if n > 0 else 0.0
    p_lo, p_hi = wilson_interval(f, n)
    ler = bc.round_ler(p_shot, r)
    lo = bc.round_ler(p_lo, r)
    hi = bc.round_ler(p_hi, r)
    se = math.sqrt(p_shot * (1.0 - p_shot) / n) if (f > 0 and n > 0) else 0.0
    sig = abs(bc.round_ler(min(0.499, p_shot + se), r) - ler) if f > 0 else 0.0

    cpu = rec.get("cpu_time_seconds")
    if cpu is None:
        cpu = rec.get("total_time_seconds")
    if cpu is not None:
        cpu = float(cpu)
        t_shot = cpu / n if n > 0 else 0.0
        t_round = cpu / (n * r) if (n > 0 and r > 0) else 0.0
    else:
        t_round = float(rec.get("time_per_round", 0.0))
        t_shot = float(rec.get("time_per_shot") or rec.get("time_per_shot_seconds") or (t_round * r))
        cpu = t_shot * n

    rec["tag"] = "bb_d6"
    rec["base_tag"] = "bb_d6"
    rec["code"] = "bb_d6"
    rec["d"] = 6
    rec["r"] = r
    rec["q"] = 144
    rec["n"] = 72
    rec["k"] = k
    rec["num_detectors"] = 432
    rec["num_observables"] = 12
    rec["decoder_id"] = rec["decoder_code"].strip("[]")
    rec["shot_error_rate"] = p_shot
    rec["ler_per_round"] = ler
    rec["ler_lo"] = lo
    rec["ler_hi"] = hi
    rec["ler_1sigma"] = sig
    rec["ler_per_round_per_logical"] = bc.round_ler(p_shot, r * k)
    rec["cpu_time_seconds"] = cpu
    rec["total_time_seconds"] = cpu
    rec["time_per_shot"] = t_shot
    rec["time_per_shot_seconds"] = t_shot
    rec["time_per_round"] = t_round
    if rec["decoder_code"] == "[C]":
        rec.setdefault("stage3_decoder", "Tesseract (--fallback-det-beam, capped at beam 14)")
    return rec


def write_mp_dem(circuit_rel: str, out_path: pathlib.Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckt = stim.Circuit.from_file(REPO_ROOT / circuit_rel)
    dem = demutil.annotate_detector_bases(
        ckt.detector_error_model(
            decompose_errors=False,
            flatten_loops=True,
            allow_gauge_detectors=True,
            approximate_disjoint_errors=1.0,
        )
    )
    dem.to_file(out_path)


def write_gari_files(circuit_rel: str, dem_path: pathlib.Path, ord_path: pathlib.Path):
    dem_path.parent.mkdir(parents=True, exist_ok=True)
    ckt = stim.Circuit.from_file(REPO_ROOT / circuit_rel)
    gari_dem = demutil.gari.circuit_to_gari(
        ckt,
        prior_function=demutil.gari.tesseract_xor_prior_probabilities,
    )
    gari_dem.to_file(dem_path)
    orders = demutil.gari.build_detector_orders(ckt, gari_dem, 1, seed=0)
    ord_path.write_text(json.dumps(orders))


def run_cli_decoder(
    code: str,
    circuit_rel: str,
    shots: int,
    seed: int,
    threads: int,
    extra_flags: Optional[List[str]] = None,
) -> Tuple[str, dict]:
    if code in bc.DECODERS:
        return bc.run_decoder(
            code,
            ["--circuit", circuit_rel],
            ["--sample-num-shots", str(shots), "--sample-seed", str(seed)],
            threads,
            extra_flags,
        )
    dec = EXTRA_DECODERS[code]
    cmd = (
        [dec["binary"]]
        + ["--circuit", circuit_rel]
        + ["--sample-num-shots", str(shots), "--sample-seed", str(seed)]
        + ["--threads", str(threads), "--stats-out", "-"]
        + (extra_flags or [])
        + dec["flags"]
    )
    print("RUN", " ".join(cmd), flush=True)
    import subprocess
    out = subprocess.check_output(cmd, text=True, cwd=REPO_ROOT)
    stats = json.loads(out.strip().splitlines()[-1])
    return " ".join(cmd), stats


def run_relay_bp(
    circuit_rel: str,
    shots: int,
    seed: int,
    threads: int,
) -> Tuple[str, dict]:
    import relay_bp
    from telescoping_decoder import circuit_to_dem, dem_to_sparse_matrices

    os.environ["RAYON_NUM_THREADS"] = str(threads)
    ckt = stim.Circuit.from_file(str(REPO_ROOT / circuit_rel))
    dem = circuit_to_dem(ckt)
    H_csr, L_csr, priors = dem_to_sparse_matrices(dem)
    syndromes, actual_obs = ckt.compile_detector_sampler(seed=seed).sample(
        shots=shots, separate_observables=True
    )
    synd_u8 = syndromes.astype(np.uint8)
    obs_u8 = actual_obs.astype(np.uint8)
    dec = relay_bp.RelayDecoderF32(
        csc_matrix(H_csr),
        error_priors=priors.astype(np.float64),
        gamma0=0.1,
        pre_iter=80,
        num_sets=5,
        set_max_iter=60,
        gamma_dist_interval=(-0.24, 0.66),
        stop_nconv=1,
    )
    runner = relay_bp.ObservableDecoderRunner(
        dec, csc_matrix(L_csr), include_decode_result=False
    )
    cmd = (
        'python3 -c "import relay_bp; ... '
        'RelayDecoderF32(gamma0=0.1, pre_iter=80, num_sets=5, set_max_iter=60, '
        f'gamma_dist_interval=(-0.24, 0.66), stop_nconv=1)" # circuit={circuit_rel} '
        f"shots={shots} seed={seed} threads={threads}"
    )
    print("RUN", cmd, flush=True)
    t_wall0 = time.perf_counter()
    t_cpu0 = time.process_time()
    preds = runner.decode_observables_batch(
        synd_u8, parallel=True, progress_bar=False
    )
    cpu_s = time.process_time() - t_cpu0
    wall_s = time.perf_counter() - t_wall0
    errs = int(np.any(preds != obs_u8, axis=1).sum())
    stats = {
        "num_shots": shots,
        "num_errors": errs,
        "num_low_confidence": 0,
        "total_time_seconds": cpu_s,
        "wall_time_seconds": wall_s,
        "sample_seed": seed,
    }
    return cmd, stats


def run_telescoping_bp(
    circuit_rel: str,
    shots: int,
    seed: int,
    threads: int,
) -> Tuple[str, dict]:
    from telescoping_decoder import TelescopeConfig, TelescopingDecoder, circuit_to_dem

    os.environ["XDG_CACHE_HOME"] = "/tmp/external_bp_bench/cache"
    os.environ["TELESCOPING_DECODER_MARCH_NATIVE"] = "1"
    ckt = stim.Circuit.from_file(str(REPO_ROOT / circuit_rel))
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

    cmd = (
        'python3 -c "from telescoping_decoder import TelescopingDecoder, TelescopeConfig; '
        f'TelescopingDecoder(TelescopeConfig(use_gari=True), n_procs={threads})" '
        f"# circuit={circuit_rel} shots={shots} seed={seed} threads={threads}"
    )
    print("RUN", cmd, flush=True)
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t_cpu0 = time.process_time()
    t_wall0 = time.perf_counter()
    with TelescopingDecoder.from_dem(
        dem, is_x_detector=mask, init_basis="Z", config=cfg
    ) as tdec:
        res = tdec.decode(synd_u8, true_obs=obs_u8)
    wall_s = time.perf_counter() - t_wall0
    self_cpu = time.process_time() - t_cpu0
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    cpu_s = child_cpu if child_cpu > 0.1 else (self_cpu + child_cpu)
    errs = int(res.le.sum())
    n_nc = int(res.diagnostics.get("n_nc", 0))
    stats = {
        "num_shots": shots,
        "num_errors": errs,
        "num_low_confidence": 0,
        "num_non_converged_diagnostics": n_nc,
        "total_time_seconds": cpu_s,
        "wall_time_seconds": wall_s,
        "sample_seed": seed,
    }
    return cmd, stats


def load_p001_records(commit: str) -> List[dict]:
    """Loads existing p=0.001 records for bb_d6 from bivariate_bicycle.json and multipass/results.json."""
    out: Dict[str, dict] = {}
    if BB_DATA_PATH.exists():
        bb_ds = json.loads(BB_DATA_PATH.read_text())
        for r in bb_ds.get("records", []):
            if r.get("tag") == "bb_d6" and r.get("decoder_code") in (
                "[B]",
                "[G]",
                "[M]",
                "[O]",
                "[C]",
                "[R]",
                "[T]",
            ):
                rec = copy.deepcopy(r)
                rec["p"] = 0.001
                enrich_stats_fields(rec)
                out[rec["decoder_code"]] = rec

    if MP_RESULTS_PATH.exists():
        mp_rows = json.loads(MP_RESULTS_PATH.read_text())
        for r in mp_rows:
            if r.get("tag") != "bb_d6":
                continue
            mode = r.get("mode")
            code = "[R]" if mode == "relay_bp" else ("[T]" if mode == "telescoping_bp" else None)
            if code and code not in out:
                key, mode_name, label = bc.CODE_META[code]
                cpu_val = float(
                    r.get("cpu_time_seconds")
                    or r.get("total_time_seconds")
                    or (float(r["time_per_round"]) * int(r["num_shots"]) * 6)
                )
                rec = dict(BASE_META)
                rec.update(
                    {
                        "p": 0.001,
                        "circuit_path": circuit_rel_path(0.001),
                        "decoder_key": key,
                        "mode": mode_name,
                        "decoder_code": code,
                        "decoder_label": label,
                        "canonical": True,
                        "source": "reused from benchmarking/multipass/results.json (p=0.001 canonical)",
                        "git_commit": commit,
                        "command_line": r.get("command_line", ""),
                        "sample_seed": 42,
                        "num_shots": int(r["num_shots"]),
                        "num_errors": int(r["num_errors"]),
                        "num_low_confidence": int(r.get("num_low_confidence", 0)),
                        "cpu_time_seconds": cpu_val,
                        "total_time_seconds": cpu_val,
                        "wall_time_seconds": 0.0,
                    }
                )
                enrich_stats_fields(rec)
                out[code] = rec

    return list(out.values())


def upsert_psweep(dataset: dict, rec: dict):
    """Replaces any record with the same (round(p, 8), decoder_code)."""
    k = (round(float(rec["p"]), 8), rec["decoder_code"])
    dataset["records"] = [
        r
        for r in dataset["records"]
        if (round(float(r["p"]), 8), r["decoder_code"]) != k
    ] + [rec]


def psweep_sort_key(r: dict):
    p_val = round(float(r.get("p", 0.0)), 8)
    c_idx = (
        bc.CODE_ORDER.index(r["decoder_code"])
        if r.get("decoder_code") in bc.CODE_ORDER
        else 99
    )
    return (p_val, c_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default="O,C,M,G,B,R,T")
    ap.add_argument(
        "--p-values",
        default=",".join(f"{p:g}" for p in P_VALUES),
        help="Comma-separated physical error rates",
    )
    ap.add_argument("--shots", type=int, default=0, help="Override shot count")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--dem-dir", default="/tmp/tesseract_bench_dems/bb_d6_psweep")
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip (p, decoder) pairs already present in bb_d6_psweep.json",
    )
    ap.add_argument(
        "--force-rerun-p001",
        action="store_true",
        help="Do not reuse existing p=0.001 records; re-run them from scratch",
    )
    args = ap.parse_args()

    commit = bc.git_commit()
    codes = [f"[{c.strip().strip('[]')}]" for c in args.codes.split(",") if c.strip()]
    p_values = [float(x.strip()) for x in args.p_values.split(",") if x.strip()]

    ds = bc.load_dataset(DATA_OUT)
    ds.update(
        {
            "family": FAMILY,
            "code": "bb_d6",
            "description": (
                "Physical error rate (p) sweep for the Bivariate Bicycle [[72, 12, 6]] code "
                "(d=6, r=6, q=144, k=12, SI1000 noise) across all 7 decoders: "
                "[B] Baseline Tesseract, [G] Gari, [M] Multi-Pass Tesseract, "
                "[O] Tesseract-BP-OSD, [C] Tesseract-Cascade, [R] Relay-BP, [T] Telescoping-BP."
            ),
            "git_commit": commit,
            "ler_per_round_formula": bc.LER_FORMULA,
        }
    )

    if 0.001 in [round(p, 8) for p in p_values] and not args.force_rerun_p001:
        for rec in load_p001_records(commit):
            if rec["decoder_code"] in codes:
                upsert_psweep(ds, rec)
                print(
                    f"  [reused p=0.001] {rec['decoder_code']}: "
                    f"{rec['num_errors']}/{rec['num_shots']} "
                    f"ler={rec['ler_per_round']:.3e} t/round={rec['time_per_round']*1e3:.3f}ms",
                    flush=True,
                )
        bc.save_dataset(DATA_OUT, ds, psweep_sort_key)

    existing_keys = {
        (round(float(r["p"]), 8), r["decoder_code"]) for r in ds.get("records", [])
    }

    dem_dir = pathlib.Path(args.dem_dir)
    dem_dir.mkdir(parents=True, exist_ok=True)

    for p in p_values:
        if round(p, 8) == 0.001 and not args.force_rerun_p001:
            continue
        c_rel = circuit_rel_path(p)
        p_tag = f"bb_d6_p{p:g}"
        meta = dict(BASE_META)
        meta.update({"p": p, "circuit_path": c_rel})

        for code in codes:
            if args.skip_existing and (round(p, 8), code) in existing_keys:
                print(f"  [skip existing] p={p:g} {code}", flush=True)
                continue
            shots = args.shots or SHOTS.get(code, {}).get(round(p, 8), 0)
            if not shots:
                continue

            if code in ("[O]", "[C]", "[M]", "[G]", "[B]"):
                extra: List[str] = []
                if code == "[M]":
                    mp_dem = dem_dir / f"{p_tag}_mp.dem"
                    write_mp_dem(c_rel, mp_dem)
                    extra = ["--dem", str(mp_dem)]
                elif code == "[G]":
                    gari_dem = dem_dir / f"{p_tag}_gari.dem"
                    gari_ord = dem_dir / f"{p_tag}_gari_ord.json"
                    write_gari_files(c_rel, gari_dem, gari_ord)
                    extra = ["--dem", str(gari_dem), "--detector-orders", str(gari_ord)]
                cmd, stats = run_cli_decoder(
                    code, c_rel, shots, args.seed, args.threads, extra
                )
            elif code == "[R]":
                cmd, stats = run_relay_bp(c_rel, shots, args.seed, args.threads)
            elif code == "[T]":
                cmd, stats = run_telescoping_bp(c_rel, shots, args.seed, args.threads)
            else:
                raise ValueError(f"Unknown decoder code: {code}")

            rec = bc.make_record(meta, code, cmd, stats, args.seed, commit, pieces_per_logical=12)
            if code == "[M]":
                rec["dem_recipe"] = MP_DEM_RECIPE
            elif code == "[G]":
                rec["dem_recipe"] = GARI_DEM_RECIPE
            rec["raw_stats_path"] = bc.archive_raw(FAMILY, p_tag, code, cmd, stats)
            enrich_stats_fields(rec)
            upsert_psweep(ds, rec)
            bc.save_dataset(DATA_OUT, ds, psweep_sort_key)
            print(
                f"  p={p:g} {code}: {rec['num_errors']}/{rec['num_shots']} "
                f"ler={rec['ler_per_round']:.3e} [{rec['ler_lo']:.2e}, {rec['ler_hi']:.2e}] "
                f"cpu={rec['cpu_time_seconds']:.1f}s t/round={rec['time_per_round']*1e3:.3f}ms",
                flush=True,
            )


if __name__ == "__main__":
    main()
