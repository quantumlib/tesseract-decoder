#!/usr/bin/env python3
"""Shared helpers for the reproducible benchmark runners in this directory.

Every fresh run executes exactly one decoder binary with `--stats-out -`, and
both the headline record (shots, errors, CPU time) and its `stage_breakdown`
are derived from that single stats JSON. The raw stats JSON is archived next to
the dataset under `benchmarking/data/raw/<family>/` so every number in
`benchmarking/data/*.json` can be traced back to one command line.

Canonical decoder codes (one definition per code across all families):
  [B]   Baseline Tesseract
  [G]   Gari
  [M]   Multi-Pass Tesseract, --beam 8 --pqlimit 100000 (canonical)
  [M5]  Multi-Pass Tesseract, --beam 5 (ablation)
  [M20] Multi-Pass Tesseract, --beam 20 --pqlimit 1000000 (ablation)
  [O]   Tesseract-BP-OSD
  [C]   Tesseract-Cascade
  [R]   Relay-BP (external reference)
  [T]   Telescoping-BP (external reference)
  [AQ]  AlphaQubit reference (hyperbolic only)
"""

import json
import pathlib
import subprocess
from typing import Callable, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "benchmarking/data"
RAW_DIR = DATA_DIR / "raw"
BP_BIN = "./bazel-bin/src/bp"
TESSERACT_BIN = "./bazel-bin/src/tesseract"

# Shared BP-OSD flags of [O] and [C].
BP_OSD_FLAGS = [
    "--multipass",
    "--num-passes",
    "2",
    "--random-schedule",
    "--max-iter",
    "20",
    "--normalization-factor",
    "0.75",
    "--osd-order",
    "10",
    "--osd-weight",
    "2",
]

DECODERS = {
    "[O]": {
        "decoder_key": "tesseract_bp_osd",
        "mode": "mp_bp_osd",
        "decoder_label": "[O] Tesseract-BP-OSD",
        "binary": BP_BIN,
        "flags": BP_OSD_FLAGS,
    },
    "[C]": {
        "decoder_key": "tesseract_cascade",
        "mode": "mp_bp_osd_cascade",
        "decoder_label": "[C] Tesseract-Cascade",
        "binary": BP_BIN,
        "flags": BP_OSD_FLAGS + ["--tesseract-fallback"],
    },
    "[M]": {
        "decoder_key": "multipass_tesseract",
        "mode": "multipass_tesseract",
        "decoder_label": "[M] Multi-Pass Tesseract",
        "binary": TESSERACT_BIN,
        # Requires a basis-annotated `--dem` (added by the caller).
        "flags": [
            "--multipass",
            "--num-passes",
            "2",
            "--beam",
            "8",
            "--beam-climbing",
            "--no-revisit-dets",
            "--pqlimit",
            "100000",
        ],
    },
}

# Labels for codes that are not re-run by default (legacy / external records).
CODE_META = {
    "[B]": ("baseline", "baseline", "[B] Baseline Tesseract"),
    "[G]": ("gari", "gari", "[G] Gari"),
    "[M]": ("multipass_tesseract", "multipass_tesseract", "[M] Multi-Pass Tesseract"),
    "[M5]": (
        "multipass_tesseract_b5",
        "multipass_tesseract_b5",
        "[M5] Multi-Pass Tesseract (beam 5)",
    ),
    "[M20]": (
        "multipass_tesseract_b20",
        "multipass_tesseract_b20",
        "[M20] Multi-Pass Tesseract (beam 20)",
    ),
    "[O]": ("tesseract_bp_osd", "mp_bp_osd", "[O] Tesseract-BP-OSD"),
    "[C]": ("tesseract_cascade", "mp_bp_osd_cascade", "[C] Tesseract-Cascade"),
    "[R]": ("relay_bp", "relay_bp", "[R] Relay-BP"),
    "[T]": ("telescoping_bp", "telescoping_bp", "[T] Telescoping-BP"),
    "[AQ]": ("aq_baseline", "aq_baseline", "AQ Baseline"),
}

LER_FORMULA = (
    "sinter piece error rate: ler_per_round = 0.5*(1-(1-2*shot_error_rate)^(1/r))"
)


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def round_ler(p_shot: Optional[float], pieces: int) -> Optional[float]:
    """sinter.shot_error_rate_to_piece_error_rate (values=1)."""
    if p_shot is None:
        return None
    if p_shot <= 0.0:
        return 0.0
    if p_shot >= 0.5:
        return 0.5
    return 0.5 * (1.0 - (1.0 - 2.0 * p_shot) ** (1.0 / pieces))


def compute_stage_breakdown(stats: dict, command_line: str) -> dict:
    """Per-stage shot / CPU split derived from ONE `--stats-out` JSON.

    Inclusive counts (as emitted by src/bp):
      stage1_bp_converged_shots + stage1_osd_shots == num_shots
      num_stage2_triggered  = shots flagged low-confidence after Stage 1
      num_fallback_triggered = Stage-2 shots escalated to Stage 3 (subset)
    Stage-2 trigger reason 5 is the only reason raised on a shot where both BP
    passes converged; all other reasons imply Stage-1 OSD ran. This gives an
    exact partition into the stage at which each shot was finally resolved.
    """
    n = int(stats["num_shots"])
    s1_bp = int(stats.get("stage1_bp_converged_shots", 0))
    s1_osd = int(stats.get("stage1_osd_shots", 0))
    s2 = int(stats.get("num_stage2_triggered", 0))
    s2_from_bp = int(stats.get("num_stage2_from_bp_converged", 0))
    s3 = int(stats.get("num_fallback_triggered", 0))
    s3_sub = int(stats.get("num_fallback_substituted", 0))
    assert s1_bp + s1_osd == n, (s1_bp, s1_osd, n)
    assert 0 <= s3 <= s2 <= n, (s2, s3, n)

    res_s1_bp = s1_bp - s2_from_bp
    res_s1_osd = s1_osd - (s2 - s2_from_bp)
    res_s2 = s2 - s3
    res_s3 = s3
    assert min(res_s1_bp, res_s1_osd, res_s2, res_s3) >= 0
    assert res_s1_bp + res_s1_osd + res_s2 + res_s3 == n

    t_s1 = float(stats.get("stage1_time_seconds", 0.0))
    t_s2 = float(stats.get("stage2_time_seconds", 0.0))
    t_s3 = float(stats.get("stage3_time_seconds", 0.0))
    t_total = float(stats["total_time_seconds"])
    assert abs((t_s1 + t_s2 + t_s3) - t_total) <= 1e-6 * max(1.0, t_total) + 1e-3, (
        t_s1,
        t_s2,
        t_s3,
        t_total,
    )
    t_bp = float(stats.get("t_bp", 0.0))
    t_post = float(stats.get("t_post", 0.0))
    # Stage-1 wall-clock per batch is split between BP and OSD in proportion to
    # the component profiler's t_bp : t_post (construct/deinterleave overhead
    # is attributed proportionally).
    if t_bp + t_post > 0:
        t_s1_bp = t_s1 * t_bp / (t_bp + t_post)
    else:
        t_s1_bp = t_s1
    t_s1_osd = t_s1 - t_s1_bp
    tt = max(t_total, 1e-12)

    return {
        "telemetry_same_run_as_headline": True,
        "telemetry_command_line": command_line,
        "telemetry_num_shots": n,
        "telemetry_num_errors": stats.get("num_errors"),
        "telemetry_seed": stats.get("sample_seed"),
        "telemetry_cpu_seconds": t_total,
        # Inclusive counters.
        "stage1_bp_converged_shots": s1_bp,
        "stage1_osd_shots": s1_osd,
        "num_stage2_triggered": s2,
        "num_stage2_from_bp_converged": s2_from_bp,
        "num_stage2_by_reason": stats.get("num_stage2_by_reason"),
        "num_fallback_triggered": s3,
        "num_fallback_substituted": s3_sub,
        "pct_shots_stage1_bp": 100.0 * s1_bp / n,
        "pct_shots_stage1_osd": 100.0 * s1_osd / n,
        "pct_shots_stage2": 100.0 * s2 / n,
        "pct_shots_stage3": 100.0 * s3 / n,
        # Exclusive partition: stage at which each shot was finally resolved.
        "resolved_stage1_bp_shots": res_s1_bp,
        "resolved_stage1_osd_shots": res_s1_osd,
        "resolved_stage2_shots": res_s2,
        "resolved_stage3_shots": res_s3,
        "pct_shots_resolved_stage1_bp": 100.0 * res_s1_bp / n,
        "pct_shots_resolved_stage1_osd": 100.0 * res_s1_osd / n,
        "pct_shots_resolved_stage2": 100.0 * res_s2 / n,
        "pct_shots_resolved_stage3": 100.0 * res_s3 / n,
        # CPU split (sums to telemetry_cpu_seconds).
        "stage1_time_seconds": t_s1,
        "stage1_bp_time_seconds": t_s1_bp,
        "stage1_osd_time_seconds": t_s1_osd,
        "stage2_time_seconds": t_s2,
        "stage3_time_seconds": t_s3,
        "t_bp": t_bp,
        "t_post": t_post,
        "t_construct": float(stats.get("t_construct", 0.0)),
        "t_deinterleave": float(stats.get("t_deinterleave", 0.0)),
        "pct_cpu_stage1_bp": 100.0 * t_s1_bp / tt,
        "pct_cpu_stage1_osd": 100.0 * t_s1_osd / tt,
        "pct_cpu_stage2": 100.0 * t_s2 / tt,
        "pct_cpu_stage3": 100.0 * t_s3 / tt,
    }


def run_decoder(
    code: str,
    input_args: list,
    shots_args: list,
    threads: int,
    extra_flags: Optional[list] = None,
) -> tuple:
    """Runs one canonical decoder; returns (command_line, stats_json)."""
    dec = DECODERS[code]
    cmd = (
        [dec["binary"]]
        + input_args
        + shots_args
        + ["--threads", str(threads), "--stats-out", "-"]
        + (extra_flags or [])
        + dec["flags"]
    )
    print("RUN", " ".join(cmd), flush=True)
    out = subprocess.check_output(cmd, text=True, cwd=REPO_ROOT)
    # `--stats-out -` prints the JSON as the last stdout line.
    stats = json.loads(out.strip().splitlines()[-1])
    return " ".join(cmd), stats


def make_record(
    meta: dict,
    code: str,
    command_line: str,
    stats: dict,
    seed: int,
    commit: str,
    pieces_per_logical: Optional[int] = None,
) -> dict:
    key, mode, label = CODE_META[code]
    n = int(stats["num_shots"])
    nerr = int(stats["num_errors"])
    low_c = int(stats.get("num_low_confidence", 0))
    p = (nerr + low_c) / n
    cpu = float(stats["total_time_seconds"])
    r = meta["r"]
    rec = dict(meta)
    rec.update(
        {
            "decoder_key": key,
            "mode": mode,
            "decoder_code": code,
            "decoder_label": label,
            "canonical": True,
            "source": "fresh run via benchmarking/runners (raw stats archived)",
            "git_commit": commit,
            "command_line": command_line,
            "sample_seed": seed,
            "num_shots": n,
            "num_errors": nerr,
            "num_low_confidence": low_c,
            "shot_error_rate": p,
            "ler_per_round": round_ler(p, r),
            "cpu_time_seconds": cpu,
            "total_time_seconds": cpu,
            "wall_time_seconds": float(stats.get("wall_time_seconds", 0.0)),
            "time_per_shot_seconds": cpu / n,
            "time_per_round": cpu / (n * r),
        }
    )
    if pieces_per_logical:
        rec["ler_per_round_per_logical"] = round_ler(p, r * pieces_per_logical)
    if "stage1_bp_converged_shots" in stats:
        rec["stage_breakdown"] = compute_stage_breakdown(stats, command_line)
        rec["stage_breakdown"]["telemetry_seed"] = seed
    return rec


def archive_raw(family: str, tag: str, code: str, command_line: str, stats: dict):
    d = RAW_DIR / family
    d.mkdir(parents=True, exist_ok=True)
    name = f"{tag}_{code.strip('[]')}.json"
    (d / name).write_text(
        json.dumps({"command_line": command_line, "stats": stats}, indent=2) + "\n"
    )
    return str((d / name).relative_to(REPO_ROOT))


def load_dataset(path: pathlib.Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"records": []}


def upsert(dataset: dict, rec: dict):
    """Replaces any record with the same (tag, decoder_code)."""
    k = (rec["tag"], rec["decoder_code"])
    dataset["records"] = [
        r for r in dataset["records"] if (r["tag"], r["decoder_code"]) != k
    ] + [rec]


def save_dataset(path: pathlib.Path, dataset: dict, order: Callable = None):
    if order is not None:
        dataset["records"].sort(key=order)
    path.write_text(json.dumps(dataset, indent=2) + "\n")


CODE_ORDER = ["[AQ]", "[B]", "[G]", "[M]", "[M5]", "[M20]", "[R]", "[T]", "[O]", "[C]"]


def record_order(tag_order: list):
    def key(r):
        t = tag_order.index(r["tag"]) if r["tag"] in tag_order else 99
        c = CODE_ORDER.index(r["decoder_code"]) if r["decoder_code"] in CODE_ORDER else 99
        return (t, c)

    return key
