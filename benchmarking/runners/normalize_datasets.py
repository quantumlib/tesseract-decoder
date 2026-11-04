#!/usr/bin/env python3
"""Normalizes and validates benchmarking/data/*.json (idempotent).

  * One canonical definition per decoder code (see bench_common.py): legacy
    Multi-Pass Tesseract records are recoded by beam width ([M]=beam 8,
    [M5]=beam 5, [M20]=beam 20).
  * At most one record per (tag, decoder_code); duplicates are an error.
  * ler_per_round is recomputed from shot_error_rate with one formula.
  * Every stage_breakdown must come from the same run as its headline record
    (telemetry_same_run_as_headline) or carry its own telemetry_* provenance.
  * color_codes.json `distance_sweep` is rebuilt from the canonical [O]/[C]
    records at d=5,7,9,11 (no separate runs, so the numbers are identical).

Usage: python3 benchmarking/runners/normalize_datasets.py [--check]
"""

import argparse
import json
import re
import sys

from bench_common import CODE_META, DATA_DIR, LER_FORMULA, REPO_ROOT, round_ler

FILES = ["color_codes", "bivariate_bicycle", "mitten", "hyperbolic_octagon"]
LEGACY_SOURCE = {
    "color_codes": "benchmarking/multipass/results.json (validated legacy run)",
}


def multipass_beam(cmd: str):
    m = re.search(r"--(?:det-)?beam (\d+)", cmd or "")
    return int(m.group(1)) if m else None


_SIZE_CACHE = {}


def circuit_size(rel_path: str):
    """(num_detectors, num_observables) of a circuit or DEM; None without stim."""
    if rel_path not in _SIZE_CACHE:
        try:
            import stim
        except ImportError:
            return None
        p = REPO_ROOT / rel_path
        obj = (
            stim.DetectorErrorModel.from_file(p)
            if rel_path.endswith(".dem")
            else stim.Circuit.from_file(p)
        )
        _SIZE_CACHE[rel_path] = (obj.num_detectors, obj.num_observables)
    return _SIZE_CACHE[rel_path]


def stage3_decoder(num_dets: int, num_obs: int) -> str:
    """Mirrors the Stage-3 dispatch in src/bp_main.cc (deferred cascade)."""
    if num_dets > 6000:
        return "Diversified MP-BP-OSD (random schedule, max_iter>=100, osd_order>=80); no Tesseract"
    if num_dets <= 480 or num_obs <= 1:
        return "Tesseract (--fallback-det-beam, capped at beam 14)"
    if num_dets <= 1500:
        return "Multi-Pass Tesseract (beam 8, pqlimit 1e5)"
    if num_dets <= 3000:
        return "Tesseract (beam 12, 4 detector orders, pqlimit 5e4)"
    return "Multi-Pass Tesseract (beam 15, 16 detector orders, pqlimit 2e5)"


def recode(rec: dict):
    code = rec["decoder_code"]
    if code in ("[M]", "[M5]", "[M20]") and "--multipass" in (rec.get("command_line") or ""):
        beam = multipass_beam(rec["command_line"])
        code = {8: "[M]", 5: "[M5]", 20: "[M20]"}[beam]
    key, mode, label = CODE_META[code]
    rec.update(
        {"decoder_code": code, "decoder_key": key, "mode": mode, "decoder_label": label}
    )


def validate(family: str, ds: dict) -> list:
    errs = []
    seen = set()
    for r in ds["records"]:
        k = (r["tag"], r["decoder_code"])
        if k in seen:
            errs.append(f"{family}: duplicate {k}")
        seen.add(k)
        for f in ("git_commit", "command_line", "circuit_path", "num_shots", "num_errors", "source"):
            if f not in r:
                errs.append(f"{family}: {k} missing {f}")
        sb = r.get("stage_breakdown")
        if not sb:
            continue
        if sb.get("telemetry_same_run_as_headline"):
            if sb["telemetry_num_shots"] != r["num_shots"]:
                errs.append(f"{family}: {k} telemetry shots != headline")
            if sb["telemetry_num_errors"] != r["num_errors"]:
                errs.append(f"{family}: {k} telemetry errors != headline")
            if abs(sb["telemetry_cpu_seconds"] - r["cpu_time_seconds"]) > 1e-6:
                errs.append(f"{family}: {k} telemetry cpu != headline")
        else:
            missing = [
                f
                for f in (
                    "telemetry_command_line",
                    "telemetry_num_shots",
                    "telemetry_num_errors",
                    "telemetry_seed",
                    "telemetry_cpu_seconds",
                )
                if f not in sb
            ]
            if missing:
                errs.append(f"{family}: {k} stage_breakdown missing {missing}")
                continue
        t = sb["stage1_time_seconds"] + sb["stage2_time_seconds"] + sb["stage3_time_seconds"]
        if abs(t - sb["telemetry_cpu_seconds"]) > 1e-3 + 1e-6 * t:
            errs.append(f"{family}: {k} stage times {t} != telemetry cpu")
        n = sb["telemetry_num_shots"]
        parts = [
            sb["resolved_stage1_bp_shots"],
            sb["resolved_stage1_osd_shots"],
            sb["resolved_stage2_shots"],
            sb["resolved_stage3_shots"],
        ]
        if sum(parts) != n or min(parts) < 0:
            errs.append(f"{family}: {k} resolved partition {parts} != {n}")
        pct = sb["pct_cpu_stage1_bp"] + sb["pct_cpu_stage1_osd"] + sb["pct_cpu_stage2"] + sb["pct_cpu_stage3"]
        if abs(pct - 100.0) > 1e-6:
            errs.append(f"{family}: {k} cpu pct sums to {pct}")
        if r["decoder_code"] == "[O]" and (sb["num_stage2_triggered"] or sb["stage2_time_seconds"]):
            errs.append(f"{family}: {k} [O] has Stage-2 activity")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="validate only")
    args = ap.parse_args()
    all_errs = []
    for fam in FILES:
        path = DATA_DIR / f"{fam}.json"
        ds = json.loads(path.read_text())
        if not args.check:
            ds["ler_per_round_formula"] = LER_FORMULA
            for r in ds["records"]:
                recode(r)
                r.setdefault("source", LEGACY_SOURCE.get(fam, "earlier local run (command_line recorded)"))
                r["canonical"] = True
                if r.get("shot_error_rate") is not None:
                    r["ler_per_round"] = round_ler(r["shot_error_rate"], r["r"])
                    if "k" in r and fam == "hyperbolic_octagon":
                        r["ler_per_round_per_logical"] = round_ler(r["shot_error_rate"], r["r"] * r["k"])
                size = circuit_size(r["circuit_path"])
                if size:
                    r["num_detectors"], r["num_observables"] = size
                    if r["decoder_code"] == "[C]":
                        r["stage3_decoder"] = stage3_decoder(*size)
            ds.pop("decoders", None)
            ds["decoder_codes"] = {c: m[2] for c, m in CODE_META.items()}
            if fam == "color_codes":
                ds["distance_sweep"] = [
                    r
                    for r in ds["records"]
                    if r["decoder_code"] in ("[O]", "[C]") and r.get("d") in (5, 7, 9, 11, 13, 15, 17, 19, 21, 23)
                ]
                ds["distance_sweep_note"] = (
                    "Copies of the canonical [O]/[C] records at d=5..23 "
                    "(same runs as `records`)."
                )
            path.write_text(json.dumps(ds, indent=2) + "\n")
        all_errs += validate(fam, ds)
    for e in all_errs:
        print("ERROR", e)
    print(f"{len(all_errs)} validation errors")
    sys.exit(1 if all_errs else 0)


if __name__ == "__main__":
    main()
