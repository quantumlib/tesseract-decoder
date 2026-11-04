#!/usr/bin/env python3
"""Unified Deterministic Static Figure & Dataset Exporter for Tesseract-BP-OSD.

Generates:
1. High-resolution static figures (.png and .pdf) with canonical self-describing filenames:
   - tradeoff__code=cc__d=7_9__dec=B_G_M_O__x=log.{png,pdf}
   - tradeoff__code=cc__d=5_7_9_11__dec=B_G_M_O__x=log.{png,pdf}
   - tradeoff__code=bb__d=6_12_gross__dec=B_G_M_O__x=log.{png,pdf}
   - tradeoff__code=bb_d6__dec=B_G_M_O__x=log.{png,pdf}
   - tradeoff__code=gross_d12__dec=B_G_M_O__x=log.{png,pdf}
   - tradeoff__code=mitten__d=10__dec=B_G_M_O__x=log.{png,pdf}
   - hyperbolic_ler__iters=32_40_58_109__dec=AQ_BPOSD__norm=block.{png,pdf}
   - hyperbolic_ler__iters=32_40_58_109__dec=AQ_BPOSD__norm=per_logical.{png,pdf}
   - stage_breakdown__code=cc__d=5_7_9_11__dec=BPOSD.{png,pdf}
   - stage_breakdown__codes=all_families__dec=BPOSD.{png,pdf}
   - arch_flow__decoder=Tesseract_BPOSD.{png,pdf}
2. LaTeX provenance mapping file (`figures_provenance.tex`) with `% === PROVENANCE ===` blocks.
3. Interactive JS dataset bundle (`benchmarking/viz/datasets.js`) for `plot_widget.js` and `explorer.html`.
"""

import argparse
import glob
import json
import math
import pathlib
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

RF = "bazel-bin/benchmarking/sparsify_errors/plot.runfiles"
sys.path[:0] = (
    [f"{RF}/_main/src", f"{RF}/_main/src/py", "bazel-bin/src", "src/py"]
    + glob.glob(f"{RF}/*/site-packages")
    + glob.glob("venv/lib/python*/site-packages")
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

VIZ_DIR = pathlib.Path(__file__).resolve().parent
BENCH_DIR = VIZ_DIR.parent
REPO_ROOT = BENCH_DIR.parent
DATA_DIR = BENCH_DIR / "data"
MULTIPASS_DIR = BENCH_DIR / "multipass"
MULTIPASS_PLOTS_DIR = MULTIPASS_DIR / "plots"
PAPER_FIGURES_DIR = pathlib.Path(
    "/usr/local/google/home/ariash/.gemini/jetski/brain/67c42a3b-49c1-44cb-a2ce-ec1851dd8f89/scratch/paper_work/figures"
)
TMP_FIGURES_DIR = pathlib.Path("/tmp/tesseract_paper_figures")
TMP_PAPER_BUILD_FIGURES_DIR = pathlib.Path("/tmp/tesseract_paper_build/figures")


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "8e4522f"


GIT_COMMIT = get_git_commit()


def wilson_interval(failures: int, shots: int, confidence: float = 0.95) -> Tuple[float, float]:
    if shots <= 0:
        return 0.0, 1.0
    z = float(norm.ppf(0.5 + confidence / 2.0))
    n = float(shots)
    p = failures / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def shot_to_round_ler(p_shot: float, rounds: int) -> float:
    if rounds <= 0 or p_shot <= 0.0:
        return 0.0
    if p_shot >= 0.5:
        return 0.5
    return 0.5 * (1.0 - (1.0 - 2.0 * p_shot) ** (1.0 / rounds))


def shot_to_logical_round_ler(p_shot: float, rounds: int, k: int) -> float:
    eff_units = max(1, rounds * max(1, k))
    if p_shot <= 0.0:
        return 0.0
    if p_shot >= 0.5:
        return 0.5
    return 0.5 * (1.0 - (1.0 - 2.0 * p_shot) ** (1.0 / eff_units))


def _dec(id_, mode, short, legend, color, offset=(9, 9)):
    return {
        "id": id_,
        "code_key": mode,
        "short_label": short,
        "legend_label": legend,
        "ring_color": color,
        "badge_offset": offset,
    }


# Decoder codes match `decoder_code` in benchmarking/data/*.json. The exact
# flags for every point are in each record's `command_line`.
DECODER_META: Dict[str, Dict[str, Any]] = {
    "B": _dec("B", "baseline", "[B] Baseline Tesseract", "[B] Baseline Tesseract (beam 20)", "#111111"),
    "G": _dec("G", "gari", "[G] Gari", "[G] Gari (Gari DEM + detector orders, beam 5)", "#108000"),
    "M": _dec("M", "multipass_tesseract", "[M] Multi-Pass Tesseract", "[M] Multi-Pass Tesseract (2 passes, beam 8)", "#0055cc"),
    "M5": _dec("M5", "multipass_tesseract_b5", "[M5] Multi-Pass Tesseract (beam 5)", "[M5] Multi-Pass Tesseract (2 passes, beam 5)", "#5b8def"),
    "M20": _dec("M20", "multipass_tesseract_b20", "[M20] Multi-Pass Tesseract (beam 20)", "[M20] Multi-Pass Tesseract (2 passes, beam 20)", "#002d6b"),
    "O": _dec("O", "mp_bp_osd", "[O] Tesseract-BP-OSD", "[O] Tesseract-BP-OSD (2-pass BP + OSD)", "#008080"),
    "R": _dec("R", "relay_bp", "[R] Relay-BP", "[R] Relay-BP (external, relay_bp)", "#e66100"),
    "T": _dec("T", "telescoping_bp", "[T] Telescoping-BP", "[T] Telescoping-BP (external, preset_balanced)", "#7b3f00"),
    "AQ": _dec("AQ", "aq_baseline", "AQ Baseline", "AQ Baseline (AlphaQubit)", "#283142", (8, 8)),
}

CIRCUIT_META: Dict[str, Dict[str, Any]] = {
    "cc_d3": {
        "tag": "cc_d3",
        "label": "cc, d=3",
        "title_label": "Color Code [[7, 1, 3]] (d=3, q=13)",
        "family": "cc",
        "n": 7,
        "k": 1,
        "d": 3,
        "r": 3,
        "q": 13,
        "marker": "o",
        "fill_color": "#f5c242",
        "edge_color": "#b58900",
        "circuit_path": "testdata/colorcodes/r=3,d=3,p=0.001,noise=si1000,c=superdense_color_code_Z,q=13,gates=cz.stim",
        "json_source": "benchmarking/data/color_codes.json",
    },
    "cc_d5": {
        "tag": "cc_d5",
        "label": "cc, d=5",
        "title_label": "Color Code [[19, 1, 5]] (d=5, q=37)",
        "family": "cc",
        "n": 19,
        "k": 1,
        "d": 5,
        "r": 5,
        "q": 37,
        "marker": "^",
        "fill_color": "#f5c242",
        "edge_color": "#b58900",
        "circuit_path": "testdata/colorcodes/r=5,d=5,p=0.001,noise=si1000,c=superdense_color_code_Z,q=37,gates=cz.stim",
        "json_source": "benchmarking/data/color_codes.json",
    },
    "cc_d7": {
        "tag": "cc_d7",
        "label": "cc, d=7",
        "title_label": "Color Code [[37, 1, 7]] (d=7, q=73)",
        "family": "cc",
        "n": 37,
        "k": 1,
        "d": 7,
        "r": 7,
        "q": 73,
        "marker": "*",
        "fill_color": "#f5c242",
        "edge_color": "#b58900",
        "circuit_path": "testdata/colorcodes/r=7,d=7,p=0.001,noise=si1000,c=superdense_color_code_Z,q=73,gates=cz.stim",
        "json_source": "benchmarking/data/color_codes.json",
    },
    "cc_d9": {
        "tag": "cc_d9",
        "label": "cc, d=9",
        "title_label": "Color Code [[61, 1, 9]] (d=9, q=121)",
        "family": "cc",
        "n": 61,
        "k": 1,
        "d": 9,
        "r": 9,
        "q": 121,
        "marker": "D",
        "fill_color": "#f5c242",
        "edge_color": "#b58900",
        "circuit_path": "testdata/colorcodes/r=9,d=9,p=0.001,noise=si1000,c=superdense_color_code_Z,q=121,gates=cz.stim",
        "json_source": "benchmarking/data/color_codes.json",
    },
    "cc_d11": {
        "tag": "cc_d11",
        "label": "cc, d=11",
        "title_label": "Color Code [[91, 1, 11]] (d=11, q=181)",
        "family": "cc",
        "n": 91,
        "k": 1,
        "d": 11,
        "r": 11,
        "q": 181,
        "marker": "p",
        "fill_color": "#f5c242",
        "edge_color": "#b58900",
        "circuit_path": "testdata/colorcodes/r=11,d=11,p=0.001,noise=si1000,c=superdense_color_code_Z,q=181,gates=cz.stim",
        "json_source": "benchmarking/data/color_codes.json",
    },
    "cc_d13": {
        "tag": "cc_d13",
        "label": "cc, d=13",
        "title_label": "Color Code [[127, 1, 13]] (d=13, q=253)",
        "family": "cc",
        "n": 127,
        "k": 1,
        "d": 13,
        "r": 13,
        "q": 253,
        "marker": "s",
        "fill_color": "#f5c242",
        "edge_color": "#b58900",
        "circuit_path": "testdata/colorcodes/r=13,d=13,p=0.001,noise=si1000,c=superdense_color_code_Z,q=253,gates=cz.stim",
        "json_source": "benchmarking/data/color_codes.json",
    },
    "cc_d15": {
        "tag": "cc_d15",
        "label": "cc, d=15",
        "title_label": "Color Code [[169, 1, 15]] (d=15, q=337)",
        "family": "cc",
        "n": 169,
        "k": 1,
        "d": 15,
        "r": 15,
        "q": 337,
        "marker": "h",
        "fill_color": "#f5c242",
        "edge_color": "#b58900",
        "circuit_path": "testdata/colorcodes/r=15,d=15,p=0.001,noise=si1000,c=superdense_color_code_Z,q=337,gates=cz.stim",
        "json_source": "benchmarking/data/color_codes.json",
    },
    "cc_d17": {
        "tag": "cc_d17",
        "label": "cc, d=17",
        "title_label": "Color Code [[217, 1, 17]] (d=17, q=433)",
        "family": "cc",
        "n": 217,
        "k": 1,
        "d": 17,
        "r": 17,
        "q": 433,
        "marker": "v",
        "fill_color": "#f5c242",
        "edge_color": "#b58900",
        "circuit_path": "testdata/colorcodes/r=17,d=17,p=0.001,noise=si1000,c=superdense_color_code_Z,q=433,gates=cz.stim",
        "json_source": "benchmarking/data/color_codes.json",
    },
    "cc_d19": {
        "tag": "cc_d19",
        "label": "cc, d=19",
        "title_label": "Color Code [[271, 1, 19]] (d=19, q=541)",
        "family": "cc",
        "n": 271,
        "k": 1,
        "d": 19,
        "r": 19,
        "q": 541,
        "marker": "<",
        "fill_color": "#f5c242",
        "edge_color": "#b58900",
        "circuit_path": "testdata/colorcodes/r=19,d=19,p=0.001,noise=si1000,c=superdense_color_code_Z,q=541,gates=cz.stim",
        "json_source": "benchmarking/data/color_codes.json",
    },
    "cc_d21": {
        "tag": "cc_d21",
        "label": "cc, d=21",
        "title_label": "Color Code [[331, 1, 21]] (d=21, q=661)",
        "family": "cc",
        "n": 331,
        "k": 1,
        "d": 21,
        "r": 21,
        "q": 661,
        "marker": ">",
        "fill_color": "#f5c242",
        "edge_color": "#b58900",
        "circuit_path": "testdata/colorcodes/r=21,d=21,p=0.001,noise=si1000,c=superdense_color_code_Z,q=661,gates=cz.stim",
        "json_source": "benchmarking/data/color_codes.json",
    },
    "cc_d23": {
        "tag": "cc_d23",
        "label": "cc, d=23",
        "title_label": "Color Code [[397, 1, 23]] (d=23, q=793)",
        "family": "cc",
        "n": 397,
        "k": 1,
        "d": 23,
        "r": 23,
        "q": 793,
        "marker": "d",
        "fill_color": "#f5c242",
        "edge_color": "#b58900",
        "circuit_path": "testdata/colorcodes/r=23,d=23,p=0.001,noise=si1000,c=superdense_color_code_Z,q=793,gates=cz.stim",
        "json_source": "benchmarking/data/color_codes.json",
    },
    "bb_d6": {
        "tag": "bb_d6",
        "label": "BB [[72, 12, 6]] (d=6, q=144)",
        "title_label": "Bivariate Bicycle [[72, 12, 6]] (d=6, q=144)",
        "family": "bb",
        "n": 72,
        "k": 12,
        "d": 6,
        "r": 6,
        "q": 144,
        "marker": "o",
        "fill_color": "#ff00ff",
        "edge_color": "#aa00aa",
        "circuit_path": "testdata/bivariatebicyclecodes/r=6,d=6,p=0.001,noise=si1000,c=bivariate_bicycle_Z,nkd=[[72,12,6]],q=144,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim",
        "json_source": "benchmarking/data/bivariate_bicycle.json",
    },
    "bb_d12_gross": {
        "tag": "bb_d12_gross",
        "label": "Gross [[144, 12, 12]] (d=12, q=288)",
        "title_label": "Gross Code [[144, 12, 12]] (d=12, q=288)",
        "family": "bb",
        "n": 144,
        "k": 12,
        "d": 12,
        "r": 12,
        "q": 288,
        "marker": "o",
        "fill_color": "#6366f1",
        "edge_color": "#312e81",
        "circuit_path": "testdata/bivariatebicyclecodes/r=12,d=12,p=0.001,noise=si1000,c=bivariate_bicycle_Z,nkd=[[144,12,12]],q=288,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim",
        "json_source": "benchmarking/data/bivariate_bicycle.json",
    },
    "mitten_d10": {
        "tag": "mitten_d10",
        "label": "mitten, d=10, q=270, [[150, 30, 10]]",
        "title_label": "Mitten Code [[150, 30, 10]] (d=10, q=270)",
        "family": "mitten",
        "n": 150,
        "k": 30,
        "d": 10,
        "r": 10,
        "q": 270,
        "marker": "o",
        "fill_color": "#38bdf8",
        "edge_color": "#0284c7",
        "circuit_path": "testdata/mittencodes/r=10,d=10,p=0.001,noise=xyz_hookfree,c=mitten_X,nkd=[[150,30,10]],q=270.stim",
        "json_source": "benchmarking/data/mitten.json",
    },
    "hyp_iter32": {
        "tag": "hyp_iter32",
        "label": "iter=32\n(n=144, k=8)",
        "short_label": "Hyp iter=32",
        "title_label": "Hyperbolic Octagon iter=32 ([[144, 8]], R=16)",
        "family": "hyperbolic",
        "iter": 32,
        "n": 144,
        "k": 8,
        "d": None,
        "r": 16,
        "q": 432,
        "marker": "o",
        "fill_color": "#10b981",
        "edge_color": "#047857",
        "circuit_path": "dripto3:/home/ariash_google_com/hyp_oct_cluster/circuits/hyp_iter32_n144_k8_p0.001_pseam0.01.stim",
        "json_source": "benchmarking/data/hyperbolic_octagon.json",
    },
    "hyp_iter40": {
        "tag": "hyp_iter40",
        "label": "iter=40\n(n=288, k=12)",
        "short_label": "Hyp iter=40",
        "title_label": "Hyperbolic Octagon iter=40 ([[288, 12]], R=16)",
        "family": "hyperbolic",
        "iter": 40,
        "n": 288,
        "k": 12,
        "d": None,
        "r": 16,
        "q": 864,
        "marker": "o",
        "fill_color": "#10b981",
        "edge_color": "#047857",
        "circuit_path": "dripto3:/home/ariash_google_com/hyp_oct_cluster/circuits/hyp_iter40_n288_k12_p0.001_pseam0.01.stim",
        "json_source": "benchmarking/data/hyperbolic_octagon.json",
    },
    "hyp_iter58": {
        "tag": "hyp_iter58",
        "label": "iter=58\n(n=1152, k=36)",
        "short_label": "Hyp iter=58",
        "title_label": "Hyperbolic Octagon iter=58 ([[1152, 36]], R=16)",
        "family": "hyperbolic",
        "iter": 58,
        "n": 1152,
        "k": 36,
        "d": None,
        "r": 16,
        "q": 3456,
        "marker": "o",
        "fill_color": "#10b981",
        "edge_color": "#047857",
        "circuit_path": "dripto3:/home/ariash_google_com/hyp_oct_cluster/circuits/hyp_iter58_n1152_k36_p0.001_pseam0.01.stim",
        "json_source": "benchmarking/data/hyperbolic_octagon.json",
    },
    "hyp_iter109": {
        "tag": "hyp_iter109",
        "label": "iter=109\n(n=3600, k=100)",
        "short_label": "Hyp iter=109",
        "title_label": "Hyperbolic Octagon iter=109 ([[3600, 100]], R=16)",
        "family": "hyperbolic",
        "iter": 109,
        "n": 3600,
        "k": 100,
        "d": None,
        "r": 16,
        "q": 10800,
        "marker": "o",
        "fill_color": "#10b981",
        "edge_color": "#047857",
        "circuit_path": "dripto3:/home/ariash_google_com/hyp_oct_cluster/circuits/hyp_iter109_n3600_k100_p0.001_pseam0.01.stim",
        "json_source": "benchmarking/data/hyperbolic_octagon.json",
    },
}


# -----------------------------------------------------------------------------
# Data loading. Every plotted point comes from benchmarking/data/*.json
# ("records" lists only). There are no hard-coded fallback records and no
# default stage breakdowns. If a record has no telemetry, its stage fields are
# None and nothing is plotted for it.
# -----------------------------------------------------------------------------

DECODER_CODE_TO_ID: Dict[str, str] = {
    "[B]": "B",
    "[G]": "G",
    "[M]": "M",
    "[M5]": "M5",
    "[M20]": "M20",
    "[O]": "O",
    "[R]": "R",
    "[T]": "T",
    "[AQ]": "AQ",
    "B": "B",
    "G": "G",
    "M": "M",
    "M5": "M5",
    "M20": "M20",
    "O": "O",
    "R": "R",
    "T": "T",
    "AQ": "AQ",
    "baseline": "B",
    "gari": "G",
    "multipass_tesseract": "M",
    "mp_bp_osd": "O",
    "relay_bp": "R",
    "telescoping_bp": "T",
}

TAG_ALIASES: Dict[str, str] = {
    "hyp_oct_iter32": "hyp_iter32",
    "hyp_oct_iter40": "hyp_iter40",
    "hyp_oct_iter58": "hyp_iter58",
    "hyp_oct_iter109": "hyp_iter109",
    "gross_d12": "bb_d12_gross",
    "bb_d12": "bb_d12_gross",
    "bb_gross": "bb_d12_gross",
}

# Telemetry runs with fewer shots than this are too small to quote stage
# percentages; their stage bars are not drawn.
MIN_TELEMETRY_SHOTS = 1024

STAGE_KEYS = (
    "stage1_bp_shots_pct",
    "stage1_osd_shots_pct",
    "stage2_shots_pct",
    "stage3_shots_pct",
    "stage1_bp_cpu_pct",
    "stage1_osd_cpu_pct",
    "stage2_cpu_pct",
    "stage3_cpu_pct",
    "stage2_triggered_pct",
    "stage3_triggered_pct",
    "stage3_substituted_shots",
    "telemetry_num_shots",
    "telemetry_same_run_as_headline",
    "telemetry_command_line",
    "telemetry_ok",
)


class DataError(RuntimeError):
    pass


def _first(d: Dict[str, Any], *keys: str) -> Optional[float]:
    for k in keys:
        v = d.get(k)
        if v is not None:
            return float(v)
    return None


def _stage_fields(row: Dict[str, Any], dec_id: str) -> Dict[str, Any]:
    """Extracts per-stage shot and CPU shares from a record's `stage_breakdown`."""
    out: Dict[str, Any] = {k: None for k in STAGE_KEYS}
    out["telemetry_ok"] = False
    sb = row.get("stage_breakdown")
    if dec_id != "O" or not isinstance(sb, dict):
        return out

    same_run = bool(sb.get("telemetry_same_run_as_headline"))
    tel_shots = sb.get("telemetry_num_shots") or row.get("telemetry_sample_shots")
    if tel_shots is None and same_run:
        tel_shots = row.get("num_shots")
    out["telemetry_num_shots"] = int(tel_shots) if tel_shots is not None else None
    out["telemetry_same_run_as_headline"] = same_run
    out["telemetry_command_line"] = sb.get("telemetry_command_line") or (
        row.get("command_line") if same_run else None
    )

    s1_bp = _first(sb, "pct_shots_resolved_stage1_bp", "pct_shots_stage1_bp")
    if s1_bp is None:
        return out
    s1_osd = max(0.0, 100.0 - s1_bp)

    cpu = [
        _first(sb, "pct_cpu_stage1_bp"),
        _first(sb, "pct_cpu_stage1_osd"),
        0.0,
        0.0,
    ]
    cpu = [0.0 if c is None else c for c in cpu]

    out.update(
        {
            "stage1_bp_shots_pct": round(s1_bp, 3),
            "stage1_osd_shots_pct": round(s1_osd, 3),
            "stage2_shots_pct": 0.0,
            "stage3_shots_pct": 0.0,
            "stage1_bp_cpu_pct": round(cpu[0], 2),
            "stage1_osd_cpu_pct": round(cpu[1], 2),
            "stage2_cpu_pct": 0.0,
            "stage3_cpu_pct": 0.0,
            "stage2_triggered_pct": 0.0,
            "stage3_triggered_pct": 0.0,
            "stage3_substituted_shots": 0,
        }
    )
    tel_ok = out["telemetry_num_shots"] is not None and out["telemetry_num_shots"] >= MIN_TELEMETRY_SHOTS
    out["telemetry_ok"] = bool(tel_ok)
    return out


def _err_bars(f: int, n: int, r: int, k: int) -> Dict[str, float]:
    """Sinter per-round LER (and per-logical variant) with Wilson 95% bounds and 1-sigma."""
    p = f / n
    p_lo, p_hi = wilson_interval(f, n)
    ler = shot_to_round_ler(p, r)
    lo, hi = shot_to_round_ler(p_lo, r), shot_to_round_ler(p_hi, r)
    lerl = shot_to_logical_round_ler(p, r, k)
    lol, hil = shot_to_logical_round_ler(p_lo, r, k), shot_to_logical_round_ler(p_hi, r, k)
    se = math.sqrt(p * (1.0 - p) / n) if f > 0 else 0.0
    sig = abs(shot_to_round_ler(min(0.499, p + se), r) - ler) if f > 0 else 0.0
    sigl = abs(shot_to_logical_round_ler(min(0.499, p + se), r, k) - lerl) if f > 0 else 0.0
    upper = f == 0
    return {
        "p_shot": p,
        # Zero-error runs are drawn at their 95% upper bound with a down arrow.
        "ler_upper_bound": upper,
        "ler": hi if upper else ler,
        "ler_raw": ler,
        "ler_lo": lo,
        "ler_hi": hi,
        "ler_err_minus": 0.0 if upper else ler - lo,
        "ler_err_plus": 0.0 if upper else hi - ler,
        "ler_1sigma": sig,
        "ler_per_logical": hil if upper else lerl,
        "ler_per_logical_raw": lerl,
        "ler_per_logical_lo": lol,
        "ler_per_logical_hi": hil,
        "ler_per_logical_1sigma": sigl,
    }


def _resolve_dec_id(row: Dict[str, Any]) -> Optional[str]:
    for k in ("decoder_code", "decoder_id", "decoder_key", "mode"):
        val = row.get(k)
        if val is not None and str(val) in DECODER_CODE_TO_ID:
            return DECODER_CODE_TO_ID[str(val)]
    return None


def _enrich_single_row(
    row: Dict[str, Any],
    base_tag: str,
    dec_id: str,
    json_source: str,
    default_git_commit: str,
    p_override: Optional[float] = None,
) -> Dict[str, Any]:
    cmeta = CIRCUIT_META[base_tag]
    dmeta = DECODER_META[dec_id]
    r = int(row.get("r") or cmeta["r"])
    k = int(row.get("k") or cmeta["k"])
    n_shots = int(row["num_shots"])
    n_errs = int(row["num_errors"])
    n_lc = int(row.get("num_low_confidence") or 0)

    p_val = p_override if p_override is not None else float(row.get("p") or 0.001)

    cpu = row.get("cpu_time_seconds")
    if cpu is None:
        cpu = row.get("total_time_seconds")
    cpu = float(cpu) if cpu is not None else None
    t_shot_raw = row.get("time_per_shot")
    if t_shot_raw is None:
        t_shot_raw = row.get("time_per_shot_seconds")
    t_round_raw = row.get("time_per_round")

    if cpu is None and t_shot_raw is not None:
        cpu = float(t_shot_raw) * n_shots
    elif cpu is None and t_round_raw is not None:
        cpu = float(t_round_raw) * n_shots * r

    time_per_shot = (
        cpu / n_shots
        if cpu is not None and n_shots > 0
        else (float(t_shot_raw) if t_shot_raw is not None else (float(t_round_raw) * r if t_round_raw is not None else None))
    )
    time_per_round = (
        cpu / (n_shots * r)
        if cpu is not None and n_shots > 0 and r > 0
        else (float(t_round_raw) if t_round_raw is not None else (time_per_shot / r if time_per_shot is not None and r > 0 else None))
    )

    wall = row.get("wall_time_seconds")
    wall = float(wall) if wall else None

    rec: Dict[str, Any] = {
        "tag": base_tag,
        "base_tag": base_tag,
        "p": p_val,
        "label": cmeta["label"],
        "title_label": cmeta["title_label"],
        "family": cmeta["family"],
        "n": int(row.get("n") or cmeta["n"]),
        "k": k,
        "d": row.get("d", cmeta.get("d")),
        "r": r,
        "q": int(row.get("q") or cmeta["q"]),
        "iter": cmeta.get("iter"),
        "marker": cmeta["marker"],
        "fill_color": cmeta["fill_color"],
        "edge_color": cmeta["edge_color"],
        "decoder_id": dec_id,
        "decoder_code": f"[{dec_id}]",
        "mode": row.get("mode") or dmeta["code_key"],
        "decoder_short": dmeta["short_label"],
        "decoder_legend": dmeta["legend_label"],
        "ring_color": dmeta["ring_color"],
        "num_shots": n_shots,
        "num_errors": n_errs,
        "num_low_confidence": n_lc,
        # Low-confidence (discarded) shots count as failures, as in the JSON headline LER.
        "eff_failures": n_errs + n_lc,
        "total_time_seconds": cpu,
        "wall_time_seconds": wall,
        "time_per_shot": time_per_shot,
        "time_per_round": time_per_round,
        "command_line": row.get("command_line") or "n/a",
        "circuit_path": row.get("circuit_path") or cmeta["circuit_path"],
        "json_source": json_source,
        "source_note": row.get("source"),
        "raw_stats_path": row.get("raw_stats_path"),
        "git_commit": row.get("git_commit") or default_git_commit or GIT_COMMIT,
        "sample_seed": row.get("sample_seed"),
        "num_detectors": row.get("num_detectors"),
        "stage3_decoder": row.get("stage3_decoder"),
    }
    rec.update(_err_bars(n_errs + n_lc, n_shots, r, k))
    rec.update(_stage_fields(row, dec_id))
    return rec


def load_all_enriched_data() -> Dict[Tuple[str, str], Dict[str, Any]]:
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not DATA_DIR.exists():
        raise DataError(f"{DATA_DIR} does not exist")

    fixed_p_files = ["color_codes.json", "bivariate_bicycle.json", "mitten.json", "hyperbolic_octagon.json"]
    for fname in fixed_p_files:
        jpath = DATA_DIR / fname
        if not jpath.exists():
            continue
        jdata = json.loads(jpath.read_text())
        rows = jdata.get("records", []) if isinstance(jdata, dict) else jdata
        json_source = f"benchmarking/data/{jpath.name}"
        default_commit = (jdata.get("git_commit") if isinstance(jdata, dict) else None) or GIT_COMMIT
        for row in rows:
            if not isinstance(row, dict) or row.get("canonical") is False:
                continue
            raw_tag = str(row.get("tag") or "")
            tag = TAG_ALIASES.get(raw_tag, raw_tag)
            dec_id = _resolve_dec_id(row)
            if tag not in CIRCUIT_META or dec_id is None:
                continue
            if (tag, dec_id) in by_key:
                raise DataError(
                    f"duplicate canonical record for ({tag}, {dec_id}) in {json_source} and "
                    f"{by_key[(tag, dec_id)]['json_source']}"
                )

            rec = _enrich_single_row(row, tag, dec_id, json_source, default_commit)
            cmeta = CIRCUIT_META[tag]

            # Cross-check against the JSON's own per-round LER.
            j_ler = row.get("ler_per_round")
            if j_ler and rec["num_errors"] > 0 and abs(j_ler - rec["ler_raw"]) > 1e-3 * j_ler:
                print(f"Warning: {tag} {dec_id} ler_per_round {j_ler} != recomputed {rec['ler_raw']}")
            by_key[(tag, dec_id)] = rec

            # Keep circuit metadata in sync with the JSON (used by datasets.js).
            cmeta["circuit_path"] = rec["circuit_path"]
            cmeta["json_source"] = json_source
            cmeta["n"], cmeta["k"], cmeta["q"], cmeta["r"] = rec["n"], rec["k"], rec["q"], rec["r"]

    # Also backfill p=0.001 records from bb_d6_psweep.json / bb_gross_psweep.json if any decoder (e.g. R, T)
    # is present there at p=0.001 and not yet in bivariate_bicycle.json.
    for ps_fname, default_base_tag in [("bb_d6_psweep.json", "bb_d6"), ("bb_gross_psweep.json", "bb_d12_gross")]:
        ps_path = DATA_DIR / ps_fname
        if not ps_path.exists():
            continue
        try:
            ps_data = json.loads(ps_path.read_text())
            ps_rows = ps_data.get("records", []) if isinstance(ps_data, dict) else ps_data
            ps_commit = (ps_data.get("git_commit") if isinstance(ps_data, dict) else None) or GIT_COMMIT
            for row in ps_rows:
                if not isinstance(row, dict) or row.get("canonical") is False:
                    continue
                p_val = float(row.get("p") or 0.0)
                if abs(p_val - 0.001) > 1e-7:
                    continue
                dec_id = _resolve_dec_id(row)
                if dec_id and (default_base_tag, dec_id) not in by_key:
                    by_key[(default_base_tag, dec_id)] = _enrich_single_row(
                        row, default_base_tag, dec_id, f"benchmarking/data/{ps_fname}", ps_commit, 0.001
                    )
        except Exception as exc:
            print(f"Note: could not inspect {ps_fname} for p=0.001 backfill ({exc})")

    for (tag, dec_id), rec in by_key.items():
        b_rec = by_key.get((tag, "B"))
        g_rec = by_key.get((tag, "G"))
        aq_rec = by_key.get((tag, "AQ"))
        t = rec["time_per_round"]
        rec["speedup_vs_B"] = (
            round(b_rec["time_per_round"] / t, 2) if b_rec and b_rec["time_per_round"] and t else None
        )
        rec["speedup_vs_G"] = (
            round(g_rec["time_per_round"] / t, 2) if g_rec and g_rec["time_per_round"] and t else None
        )
        rec["ratio_vs_AQ"] = (
            round(rec["ler_raw"] / aq_rec["ler_raw"], 3)
            if aq_rec and aq_rec["ler_raw"] > 0 and not rec["ler_upper_bound"]
            else None
        )
    return by_key


def load_psweep_records(by_key: Dict[Tuple[str, str], Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Loads physical-error-rate sweep records from bb_d6_psweep.json and bb_gross_psweep.json."""
    import re as _re

    psweep_map: Dict[Tuple[str, float, str], Dict[str, Any]] = {}

    # Seed p=0.001 from by_key first so p=0.001 is always available even before psweep files finish
    for base_tag in ("bb_d6", "bb_d12_gross"):
        for dec_id in DECODER_META:
            if (base_tag, dec_id) in by_key:
                r_copy = dict(by_key[(base_tag, dec_id)])
                r_copy["base_tag"] = base_tag
                r_copy["p"] = 0.001
                psweep_map[(base_tag, 0.001, dec_id)] = r_copy

    for ps_fname, default_base_tag in [("bb_d6_psweep.json", "bb_d6"), ("bb_gross_psweep.json", "bb_d12_gross")]:
        ps_path = DATA_DIR / ps_fname
        if not ps_path.exists():
            continue
        jdata = json.loads(ps_path.read_text())
        rows = jdata.get("records", []) if isinstance(jdata, dict) else jdata
        json_source = f"benchmarking/data/{ps_fname}"
        default_commit = (jdata.get("git_commit") if isinstance(jdata, dict) else None) or GIT_COMMIT

        for row in rows:
            if not isinstance(row, dict) or row.get("canonical") is False:
                continue
            dec_id = _resolve_dec_id(row)
            if dec_id is None:
                continue

            raw_tag = str(row.get("tag") or "")
            if "gross" in raw_tag or "d12" in raw_tag or row.get("d") == 12:
                base_tag = "bb_d12_gross"
            elif "d6" in raw_tag or row.get("d") == 6:
                base_tag = "bb_d6"
            else:
                base_tag = default_base_tag

            p_raw = row.get("p")
            if p_raw is None:
                m = _re.search(r"p=([0-9.eE+-]+)", str(row.get("circuit_path") or raw_tag))
                p_val = float(m.group(1)) if m else 0.001
            else:
                p_val = round(float(p_raw), 7)

            rec = _enrich_single_row(row, base_tag, dec_id, json_source, default_commit, p_val)
            rec["tag"] = base_tag if abs(p_val - 0.001) < 1e-7 else f"{base_tag}_p{p_val:g}"
            rec["base_tag"] = base_tag
            psweep_map[(base_tag, p_val, dec_id)] = rec

    # Compute speedup_vs_B and speedup_vs_G at each (base_tag, p_val)
    for (base_tag, p_val, dec_id), rec in psweep_map.items():
        b_rec = psweep_map.get((base_tag, p_val, "B"))
        g_rec = psweep_map.get((base_tag, p_val, "G"))
        t = rec.get("time_per_round")
        rec["speedup_vs_B"] = (
            round(b_rec["time_per_round"] / t, 2) if b_rec and b_rec.get("time_per_round") and t else None
        )
        rec["speedup_vs_G"] = (
            round(g_rec["time_per_round"] / t, 2) if g_rec and g_rec.get("time_per_round") and t else None
        )

    return [psweep_map[k] for k in sorted(psweep_map.keys())]


def fmt_time(sec: Optional[float]) -> str:
    if sec is None:
        return "n/a"
    if sec >= 1.0:
        return f"{sec:.2f} s"
    if sec >= 1e-3:
        return f"{sec * 1e3:.2f} ms"
    return f"{sec * 1e6:.0f} µs"


def compare_to_aq(rec: Dict[str, Any], aq: Dict[str, Any]) -> str:
    """Honest comparison against AQ: z-score of the log rate ratio with Poisson errors."""
    ratio = rec["ler_raw"] / aq["ler_raw"]
    f1, f2 = rec["eff_failures"], aq["eff_failures"]
    sig = math.sqrt(1.0 / f1 + 1.0 / f2) if f1 > 0 and f2 > 0 else 0.0
    z = math.log(ratio) / sig if sig > 0 and ratio > 0 else 0.0
    if abs(z) < 2.0:
        verdict = f"tie with AQ ({abs(z):.1f}σ)"
    elif z < 0:
        verdict = f"lower LER than AQ ({abs(z):.1f}σ)"
    else:
        verdict = f"AQ has {ratio:.1f}x lower LER ({abs(z):.1f}σ)"
    return f"{ratio:.2f}x AQ, {verdict}"


def save_figure_everywhere(
    fig: plt.Figure,
    canonical_stem: str,
    out_dirs: List[pathlib.Path],
    legacy_names: Optional[List[str]] = None,
    dpi: int = 220,
) -> List[str]:
    saved_paths: List[str] = []
    for d in out_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            for ext in ("png", "pdf"):
                p = d / f"{canonical_stem}.{ext}"
                fig.savefig(p, dpi=dpi, bbox_inches="tight")
                saved_paths.append(str(p))
            if legacy_names:
                for lname in legacy_names:
                    lp = d / lname
                    fig.savefig(lp, dpi=dpi, bbox_inches="tight")
        except Exception as exc:
            # Inside standard sandbox, PAPER_FIGURES_DIR may not be mounted; TMP_FIGURES_DIR and MULTIPASS_PLOTS_DIR always succeed
            print(f"Note: skipped direct write to {d} ({exc})")
    plt.close(fig)
    return saved_paths


def style_tradeoff_axes(ax: plt.Axes, title: str, xscale: str = "log") -> None:
    ax.set_xscale(xscale)
    ax.set_yscale("log")
    ax.set_xlabel("Decode CPU Time per Round (seconds)", fontsize=12, fontstyle="italic", color="#222222")
    ax.set_ylabel("Logical Error Rate per Round (Block LER)", fontsize=12, fontstyle="italic", color="#222222")
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color="#222222", pad=10)
    ax.grid(True, which="major", color="#dcdcdc", linewidth=0.9)
    ax.grid(True, which="minor", color="#f0f0f0", linewidth=0.5, linestyle=":")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_color("#dcdcdc")
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#222222")
        ax.spines[spine].set_linewidth(1.1)


def draw_point(
    ax: plt.Axes,
    pt: Dict[str, Any],
    marker: str,
    fill_color: str,
    ring_color: str,
    letter: str,
    size: int = 120,
    ring_size: int = 280,
    badge_offset: Tuple[int, int] = (9, 9),
    zorder: int = 5,
) -> None:
    x = pt["time_per_round"]
    y = pt["ler"]
    ecolor = fill_color if fill_color != "#f5c242" else "#d99b00"
    ylo_axis = ax.get_ylim()[0]
    is_ub = bool(pt.get("ler_upper_bound"))
    clamp_bottom = (
        is_ub
        or (pt["num_errors"] <= 6 and pt["num_shots"] >= 5000 and y <= ylo_axis * 25.0)
        or (y - pt["ler_err_minus"] <= ylo_axis)
    )
    lo_val = ylo_axis if clamp_bottom else (y - pt["ler_err_minus"])
    yerr_minus = max(0.0, y - lo_val)
    yerr_plus = 0.0 if is_ub else pt["ler_err_plus"]
    eb = ax.errorbar(
        [x],
        [y],
        yerr=[[yerr_minus], [yerr_plus]],
        fmt="none",
        ecolor=ecolor,
        elinewidth=1.4,
        alpha=0.85,
        zorder=zorder - 1,
    )
    if is_ub:
        for barcol in eb[2]:
            barcol.set_linestyle("--")
    ax.scatter([x], [y], marker=marker, s=ring_size, facecolors="none", edgecolors=ring_color, linewidths=2.6, zorder=zorder + 1)
    ax.scatter(
        [x],
        [y],
        marker=marker,
        s=size,
        facecolors=fill_color,
        edgecolors="#222222" if ring_color == "#111111" else ring_color,
        linewidths=1.0,
        zorder=zorder,
    )
    badge_letter = letter
    ax.annotate(
        badge_letter,
        xy=(x, y),
        xytext=badge_offset,
        textcoords="offset points",
        fontsize=8.0,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        bbox=dict(boxstyle="circle,pad=0.17", fc=ring_color, ec="white", lw=0.9),
        zorder=zorder + 3,
    )


def draw_arrow_segment(
    ax: plt.Axes,
    pt_from: Dict[str, Any],
    pt_to: Dict[str, Any],
    color: str = "#555555",
    lw: float = 1.85,
    linestyle: str = "-",
    alpha: float = 0.88,
    rad: float = 0.0,
) -> None:
    ax.annotate(
        "",
        xy=(pt_to["time_per_round"], pt_to["ler"]),
        xytext=(pt_from["time_per_round"], pt_from["ler"]),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            ls=linestyle,
            alpha=alpha,
            shrinkA=11,
            shrinkB=11,
            mutation_scale=12,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=3,
    )


def _badge_offsets(pts: Dict[str, Dict[str, Any]], chain_ids: List[str], ax: plt.Axes, dist: float = 13.0) -> Dict[str, Tuple[float, float]]:
    """Places each letter badge on the side away from its chain neighbours (display coords)."""
    disp = {d: np.array(ax.transData.transform((p["time_per_round"], p["ler"]))) for d, p in pts.items()}
    out: Dict[str, Tuple[float, float]] = {}
    for d, xy in disp.items():
        vx, vy = 0.0, 0.0
        nbrs: List[str] = []
        if d in chain_ids:
            i = chain_ids.index(d)
            nbrs = [chain_ids[j] for j in (i - 1, i + 1) if 0 <= j < len(chain_ids)]
        nbrs += [o for o in disp if o != d and o not in nbrs and np.hypot(*(disp[o] - xy)) < 60]
        for o in nbrs:
            dx, dy = disp[o] - xy
            n = math.hypot(dx, dy) or 1.0
            vx, vy = vx + dx / n, vy + dy / n
        n = math.hypot(vx, vy)
        if n < 1e-6:
            out[d] = (dist * 0.71, dist * 0.71)
        else:
            out[d] = (-vx / n * dist, -vy / n * dist)
    return out


def build_decoder_legend_handles(dec_order: List[str]) -> List[mlines.Line2D]:
    handles: List[mlines.Line2D] = []
    for d_id in dec_order:
        dm = DECODER_META[d_id]
        handles.append(
            mlines.Line2D(
                [],
                [],
                color="none",
                marker="o",
                markerfacecolor="#cccccc",
                markeredgecolor=dm["ring_color"],
                markeredgewidth=2.6,
                markersize=9.8,
                label=dm["legend_label"],
            )
        )
    return handles


def _log_limits(vals: List[float], lo_pad: float, hi_pad: float) -> Tuple[float, float]:
    vals = [v for v in vals if v and v > 0]
    lo, hi = min(vals), max(vals)
    return 10 ** (math.floor(math.log10(lo / lo_pad) * 2) / 2), 10 ** (math.ceil(math.log10(hi * hi_pad) * 2) / 2)


def plot_tradeoff_family(
    by_key: Dict[Tuple[str, str], Dict[str, Any]],
    tags: List[str],
    title: str,
    canonical_stem: str,
    out_dirs: List[pathlib.Path],
    include_external: bool = False,
    legacy_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    main_chain = ["B", "G", "M", "O"]
    external = ["R", "T"] if include_external else []
    fig, ax = plt.subplots(figsize=(14.6, 7.6), dpi=220)
    style_tradeoff_axes(ax, title)

    code_handles: List[mlines.Line2D] = []
    used: List[Dict[str, Any]] = []
    xs: List[float] = []
    ys: List[float] = []
    present_decs: List[str] = []

    allow_ub = len(tags) == 1
    all_pts = [
        by_key[(t, d)] for t in tags for d in main_chain + external
        if (t, d) in by_key
        and by_key[(t, d)]["time_per_round"]
        and (allow_ub or (by_key[(t, d)]["num_errors"] > 0 and not by_key[(t, d)]["ler_upper_bound"]))
    ]
    for pt in all_pts:
        xs.append(pt["time_per_round"])
        ys.append(max(pt["ler"] - pt["ler_err_minus"], pt["ler"] / 4.0))
        ys.append(pt["ler"] + pt["ler_err_plus"])
    ax.set_xlim(*_log_limits(xs, 2.5, 2.5))
    ax.set_ylim(*_log_limits(ys, 1.6, 1.8))
    fig.canvas.draw()

    for tag in tags:
        cm = CIRCUIT_META[tag]
        marker = cm["marker"]
        code_handles.append(
            mlines.Line2D(
                [], [], color="none", marker=marker, markerfacecolor=cm["fill_color"],
                markeredgecolor=cm["edge_color"], markersize=10.5 if marker == "*" else 9.5, label=cm["label"],
            )
        )
        pts = {d: by_key.get((tag, d)) for d in main_chain + external}
        pts = {
            d: p for d, p in pts.items()
            if p is not None and p["time_per_round"] and (allow_ub or (p["num_errors"] > 0 and not p["ler_upper_bound"]))
        }
        chain_ids = [d for d in main_chain if d in pts]
        offsets = _badge_offsets(pts, chain_ids, ax)
        for d_id, pt in pts.items():
            draw_point(ax, pt, marker, cm["fill_color"], DECODER_META[d_id]["ring_color"], d_id,
                       size=125, ring_size=290, badge_offset=offsets[d_id])
            used.append(pt)
            if d_id not in present_decs:
                present_decs.append(d_id)

        for i in range(len(chain_ids) - 1):
            a_id, b_id = chain_ids[i], chain_ids[i + 1]
            draw_arrow_segment(ax, pts[a_id], pts[b_id], color=pts[b_id]["ring_color"], lw=1.9, rad=0.0)
        if include_external and pts.get("B") and pts.get("R") and pts.get("T"):
            draw_arrow_segment(ax, pts["B"], pts["R"], color="#e66100", lw=1.35, linestyle="--", alpha=0.75)
            draw_arrow_segment(ax, pts["R"], pts["T"], color="#7b3f00", lw=1.35, linestyle="--", alpha=0.75)

    dec_order = [d for d in main_chain + external if d in present_decs]
    legend_handles = code_handles + [mlines.Line2D([], [], color="none", label="")] + build_decoder_legend_handles(dec_order)
    ax.legend(
        handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.7,
        frameon=True, facecolor="white", edgecolor="#dddddd", borderpad=0.8,
    )
    plt.tight_layout()
    save_figure_everywhere(fig, canonical_stem, out_dirs, legacy_names=legacy_names)
    return {
        "stem": canonical_stem,
        "tags": tags,
        "decoders": dec_order,
        "json_sources": sorted({p["json_source"] for p in used}),
        "commands": [p["command_line"] for p in used],
        "export_cmd": f"python3 benchmarking/viz/export_static.py --figure {canonical_stem}",
    }


HYP_TAGS = ["hyp_iter32", "hyp_iter40", "hyp_iter58", "hyp_iter109"]


def plot_hyperbolic_ler(
    by_key: Dict[Tuple[str, str], Dict[str, Any]],
    norm_mode: str,
    canonical_stem: str,
    out_dirs: List[pathlib.Path],
    legacy_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """2-panel Hyperbolic Octagon LER comparison: AQ Baseline vs [O]."""
    tags = [t for t in HYP_TAGS if all((t, d) in by_key for d in ("AQ", "O"))]
    x_labels = [CIRCUIT_META[t]["label"] for t in tags]
    x_idx = np.arange(len(tags))
    is_logical = norm_mode == "per_logical"
    metric_key = "ler_per_logical" if is_logical else "ler"
    sigma_key = "ler_per_logical_1sigma" if is_logical else "ler_1sigma"
    ylab = "Per-Round Per-Logical LER (block LER / k)" if is_logical else "Per-Round Block LER"

    fig, (ax_line, ax_bar) = plt.subplots(1, 2, figsize=(15.2, 6.9), dpi=220)
    fig.suptitle(
        f"Hyperbolic Octagon Cat-Flagged Circuits (r=16, p=0.001, p_seam=0.01): {ylab}",
        fontsize=13.5, fontweight="bold", y=0.985, color="#111111",
    )
    series_cfg = [
        ("AQ", "AQ Baseline (AlphaQubit)", "#283142", "D", "--"),
        ("O", "[O] Tesseract-BP-OSD", DECODER_META["O"]["ring_color"], "s", "-"),
    ]
    all_y: List[float] = []
    for d_id, label, color, marker, ls in series_cfg:
        ys = [by_key[(t, d_id)][metric_key] for t in tags]
        es = [by_key[(t, d_id)][sigma_key] for t in tags]
        all_y += [y - e for y, e in zip(ys, es)] + [y + e for y, e in zip(ys, es)]
        ax_line.errorbar(x_idx, ys, yerr=es, label=label, color=color, marker=marker, markersize=7.5,
                         linestyle=ls, linewidth=2.3, capsize=3.5, zorder=5 if d_id == "O" else 4)
    ylo, yhi = _log_limits(all_y, 2.0, 2.0)
    for ax in (ax_line, ax_bar):
        ax.set_yscale("log")
        ax.set_xticks(x_idx)
        ax.set_xticklabels(x_labels, fontsize=9.8)
        ax.set_xlabel("Hyperbolic octagon circuit", fontsize=10.5, fontweight="bold")
        ax.set_ylabel(ylab + " (±1σ)", fontsize=10.8, fontweight="bold")
        ax.grid(True, which="major", axis="y", linestyle="--", color="#d0d5dd", alpha=0.8)
    ax_line.set_ylim(ylo, yhi)
    ax_bar.set_ylim(ylo, yhi * 30)
    ax_line.set_title("(a) LER vs. circuit size", fontsize=11.5, fontweight="bold", pad=8)
    ax_line.legend(loc="lower left", fontsize=9.5, frameon=True, facecolor="white", edgecolor="#cccccc")

    width = 0.32
    for idx_s, (d_id, label, color, _, _) in enumerate(series_cfg):
        off = (idx_s - 0.5) * width
        ys = [by_key[(t, d_id)][metric_key] for t in tags]
        es = [by_key[(t, d_id)][sigma_key] for t in tags]
        ax_bar.bar(x_idx + off, ys, width=width * 0.92, yerr=es, label=label, color=color,
                   edgecolor="#111111", linewidth=0.7, capsize=3, alpha=0.94, zorder=3)
    ax_bar.set_title("(b) LER comparison across circuit sizes", fontsize=11.5, fontweight="bold", pad=8)
    ax_bar.legend(loc="upper right", fontsize=9.0, frameon=True, facecolor="white", edgecolor="#cccccc")

    lines = ["Ratio < 1 means lower LER than AQ; |z| < 2σ is reported as a tie (σ from binomial shot noise)."]
    for t in tags:
        it = CIRCUIT_META[t]["iter"]
        aq = by_key[(t, "AQ")]
        lines.append(
            f"iter={it}:  [O] {compare_to_aq(by_key[(t, 'O')], aq)}"
        )
    fig.text(0.5, 0.01, "\n".join(lines), ha="center", va="bottom", fontsize=8.4,
             bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8fafc", edgecolor="#94a3b8", lw=0.9))
    plt.tight_layout(rect=[0.0, 0.13, 1.0, 0.95])
    save_figure_everywhere(fig, canonical_stem, out_dirs, legacy_names=legacy_names)
    return {
        "stem": canonical_stem,
        "tags": tags,
        "decoders": ["AQ", "O"],
        "json_sources": sorted({by_key[(t, d)]["json_source"] for t in tags for d in ("AQ", "O")}),
        "commands": [by_key[(t, d)]["command_line"] for t in tags for d in ("AQ", "O")],
        "export_cmd": f"python3 benchmarking/viz/export_static.py --figure {canonical_stem}",
    }


def short_code_label(t: str) -> str:
    cm = CIRCUIT_META[t]
    if cm["family"] == "cc":
        return f"CC d={cm['d']}"
    if t == "bb_d6":
        return "BB [[72,12,6]]"
    if t == "bb_d12_gross":
        return "Gross [[144,12,12]]"
    if t == "mitten_d10":
        return "Mitten [[150,30,10]]"
    return f"Hyp iter={cm['iter']}"


def plot_cc_scaling(
    by_key: Dict[Tuple[str, str], Dict[str, Any]],
    canonical_stem: str,
    out_dirs: List[pathlib.Path],
    legacy_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """2-panel Color Code distance scaling (d=3..23): (a) LER/round vs d, (b) CPU time/round vs d."""
    cc_tags = [f"cc_d{d}" for d in (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23)]
    dec_order = ["B", "G", "M", "O", "R", "T"]
    markers = {"B": "o", "G": "^", "M": "s", "O": "D", "R": "v", "T": "P"}
    linestyles = {"B": "-", "G": "--", "M": "-.", "O": "-", "R": ":", "T": "--"}

    fig, (ax_ler, ax_time) = plt.subplots(1, 2, figsize=(15.4, 6.2), dpi=220)
    fig.suptitle(
        "Superdense Color Code Distance Scaling (d = 3 .. 23, SI1000 p = 0.001, r = d)",
        fontsize=13.5, fontweight="bold", y=0.985, color="#111111",
    )
    used: List[Dict[str, Any]] = []
    ds_all = [CIRCUIT_META[t]["d"] for t in cc_tags]

    all_ler_pts = [
        by_key[(t, d_id)]
        for t in cc_tags
        for d_id in dec_order
        if (t, d_id) in by_key
        and by_key[(t, d_id)]["num_errors"] > 0
        and not by_key[(t, d_id)]["ler_upper_bound"]
    ]
    ler_vals_for_lim = [p["ler"] for p in all_ler_pts] + [p["ler"] + p["ler_err_plus"] for p in all_ler_pts]
    ylo_ler, yhi_ler = _log_limits(ler_vals_for_lim, 2.0, 2.0) if ler_vals_for_lim else (1e-6, 1e-1)
    ax_ler.set_yscale("log")
    ax_ler.set_ylim(ylo_ler, yhi_ler)

    for d_id in dec_order:
        dm = DECODER_META[d_id]
        color = dm["ring_color"]
        pts = [by_key[(t, d_id)] for t in cc_tags if (t, d_id) in by_key]
        if not pts:
            continue
        used.extend(pts)

        # Panel (a): LER per round vs d — NEVER plot points where num_errors == 0
        pts_ler = [p for p in pts if p["num_errors"] > 0 and not p["ler_upper_bound"]]
        if pts_ler:
            xs_ler = [p["d"] for p in pts_ler]
            ys_ler = [p["ler"] for p in pts_ler]
            y_lo = [
                ylo_ler
                if (
                    (p["num_errors"] <= 6 and p["num_shots"] >= 5000 and p["ler"] <= ylo_ler * 25.0)
                    or (p["ler"] - p["ler_err_minus"] <= ylo_ler)
                )
                else (p["ler"] - p["ler_err_minus"])
                for p in pts_ler
            ]
            y_hi = [p["ler"] + p["ler_err_plus"] for p in pts_ler]
            yerr_minus = [max(0.0, y - lo) for y, lo in zip(ys_ler, y_lo)]
            yerr_plus = [max(0.0, hi - y) for y, hi in zip(ys_ler, y_hi)]

            ax_ler.plot(xs_ler, ys_ler, color=color, linestyle=linestyles[d_id], linewidth=2.3 if d_id == "O" else 2.1,
                        label=dm["short_label"], zorder=5 if d_id == "O" else 3)
            ax_ler.fill_between(xs_ler, y_lo, y_hi, color=color, alpha=0.13, zorder=2)
            ax_ler.errorbar(
                xs_ler, ys_ler, yerr=[yerr_minus, yerr_plus], fmt="none", ecolor=color,
                elinewidth=1.25, capsize=2.8, alpha=0.85, zorder=4
            )
            ax_ler.scatter(
                xs_ler, ys_ler,
                marker=markers[d_id],
                s=75 if d_id == "O" else 58,
                facecolors=color,
                edgecolors=color,
                linewidths=1.0,
                zorder=6 if d_id == "O" else 4,
            )

        # Panel (b): Decode CPU time per round vs d — KEEP all measured points (d=3..23)
        pts_time = [p for p in pts if p.get("time_per_round")]
        if pts_time:
            xs_time = [p["d"] for p in pts_time]
            ts = [p["time_per_round"] for p in pts_time]
            ax_time.plot(xs_time, ts, color=color, linestyle=linestyles[d_id], marker=markers[d_id],
                         markersize=6.8 if d_id == "O" else 6.2, linewidth=2.3 if d_id == "O" else 2.1, label=dm["short_label"],
                         zorder=5 if d_id == "O" else 3)

    for ax in (ax_ler, ax_time):
        ax.set_yscale("log")
        ax.set_xticks(ds_all)
        ax.set_xlabel("Code Distance d (rounds r = d)", fontsize=10.8, fontweight="bold")
        ax.grid(True, which="major", linestyle="--", color="#d0d5dd", alpha=0.8)
        ax.grid(True, which="minor", linestyle=":", color="#eef2f6", alpha=0.6)
    ax_ler.set_ylim(ylo_ler, yhi_ler)
    ax_ler.set_ylabel("Logical Error Rate per Round (Block LER)", fontsize=10.8, fontweight="bold")
    ax_ler.set_title("(a) LER per round vs. code distance d", fontsize=11.2, fontweight="bold", pad=8)
    ax_time.set_ylabel("Decode CPU Time per Round (seconds)", fontsize=10.8, fontweight="bold")
    ax_time.set_title("(b) Decode CPU time per round vs. code distance d", fontsize=11.2, fontweight="bold", pad=8)

    handles, labels = ax_time.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9.2, frameon=True,
               facecolor="white", edgecolor="#cbd5e1", bbox_to_anchor=(0.5, 0.005))
    plt.tight_layout(rect=[0.0, 0.09, 1.0, 0.95])
    save_figure_everywhere(fig, canonical_stem, out_dirs, legacy_names=legacy_names)
    return {
        "stem": canonical_stem,
        "tags": cc_tags,
        "decoders": dec_order,
        "json_sources": sorted({r["json_source"] for r in used}),
        "commands": [r["command_line"] for r in used],
        "export_cmd": f"python3 benchmarking/viz/export_static.py --figure {canonical_stem}",
    }


def plot_bb_psweep(
    psweep_records: List[Dict[str, Any]],
    base_tag: str,
    title: str,
    canonical_stem: str,
    out_dirs: List[pathlib.Path],
) -> Dict[str, Any]:
    """2-panel physical error rate p-sweep for BB [[72,12,6]] or Gross [[144,12,12]]."""
    recs = [r for r in psweep_records if r.get("base_tag") == base_tag]
    dec_order = ["B", "G", "M", "O", "R", "T"]
    markers = {"B": "o", "G": "^", "M": "s", "O": "D", "R": "v", "T": "P"}
    linestyles = {"B": "-", "G": "--", "M": "-.", "O": "-", "R": ":", "T": "--"}

    fig, (ax_ler, ax_time) = plt.subplots(1, 2, figsize=(15.4, 6.2), dpi=220)
    fig.suptitle(title, fontsize=13.2, fontweight="bold", y=0.985, color="#111111")
    used: List[Dict[str, Any]] = []

    all_ler_pts = [
        r for r in recs
        if r["decoder_id"] in dec_order
        and r.get("time_per_round")
        and r["num_errors"] > 0
        and not r["ler_upper_bound"]
    ]
    ler_vals_for_lim = [p["ler"] for p in all_ler_pts] + [p["ler"] + p["ler_err_plus"] for p in all_ler_pts]
    ylo_ler, yhi_ler = _log_limits(ler_vals_for_lim, 2.0, 2.0) if ler_vals_for_lim else (1e-6, 1e-1)
    ax_ler.set_yscale("log")
    ax_ler.set_ylim(ylo_ler, yhi_ler)

    for d_id in dec_order:
        dm = DECODER_META[d_id]
        color = dm["ring_color"]
        pts = sorted([r for r in recs if r["decoder_id"] == d_id and r.get("time_per_round")], key=lambda x: float(x["p"]))
        if not pts:
            continue
        used.extend(pts)

        # Panel (a): LER per round vs p — NEVER plot points where num_errors == 0
        pts_ler = [p for p in pts if p["num_errors"] > 0 and not p["ler_upper_bound"]]
        if pts_ler:
            xs_ler = [float(p["p"]) for p in pts_ler]
            ys_ler = [p["ler"] for p in pts_ler]
            y_lo = [
                ylo_ler
                if (
                    (p["num_errors"] <= 6 and p["num_shots"] >= 5000 and p["ler"] <= ylo_ler * 25.0)
                    or (p["ler"] - p["ler_err_minus"] <= ylo_ler)
                )
                else (p["ler"] - p["ler_err_minus"])
                for p in pts_ler
            ]
            y_hi = [p["ler"] + p["ler_err_plus"] for p in pts_ler]
            yerr_minus = [max(0.0, y - lo) for y, lo in zip(ys_ler, y_lo)]
            yerr_plus = [max(0.0, hi - y) for y, hi in zip(ys_ler, y_hi)]

            ax_ler.plot(xs_ler, ys_ler, color=color, linestyle=linestyles[d_id], linewidth=2.3 if d_id == "O" else 2.1,
                        label=dm["short_label"], zorder=5 if d_id == "O" else 3)
            ax_ler.fill_between(xs_ler, y_lo, y_hi, color=color, alpha=0.13, zorder=2)
            ax_ler.errorbar(
                xs_ler, ys_ler, yerr=[yerr_minus, yerr_plus], fmt="none", ecolor=color,
                elinewidth=1.25, capsize=2.8, alpha=0.85, zorder=4
            )
            ax_ler.scatter(
                xs_ler, ys_ler,
                marker=markers[d_id],
                s=75 if d_id == "O" else 58,
                facecolors=color,
                edgecolors=color,
                linewidths=1.0,
                zorder=6 if d_id == "O" else 4,
            )

        # Panel (b): Decode CPU time per round vs p — KEEP all measured points
        xs_time = [float(p["p"]) for p in pts]
        ts = [p["time_per_round"] for p in pts]
        ax_time.plot(xs_time, ts, color=color, linestyle=linestyles[d_id], marker=markers[d_id],
                     markersize=6.8 if d_id == "O" else 6.2, linewidth=2.3 if d_id == "O" else 2.1, label=dm["short_label"],
                     zorder=5 if d_id == "O" else 3)

    for ax in (ax_ler, ax_time):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Physical Error Rate p (SI1000 circuit noise)", fontsize=10.8, fontweight="bold")
        ax.grid(True, which="major", linestyle="--", color="#d0d5dd", alpha=0.8)
        ax.grid(True, which="minor", linestyle=":", color="#eef2f6", alpha=0.6)
    ax_ler.set_ylim(ylo_ler, yhi_ler)
    ax_ler.set_ylabel("Logical Error Rate per Round (Block LER)", fontsize=10.8, fontweight="bold")
    ax_ler.set_title("(a) LER per round vs. physical error rate p", fontsize=11.2, fontweight="bold", pad=8)
    ax_time.set_ylabel("Decode CPU Time per Round (seconds)", fontsize=10.8, fontweight="bold")
    ax_time.set_title("(b) Decode CPU time per round vs. physical error rate p", fontsize=11.2, fontweight="bold", pad=8)

    handles, labels = ax_time.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9.2, frameon=True,
               facecolor="white", edgecolor="#cbd5e1", bbox_to_anchor=(0.5, 0.005))
    plt.tight_layout(rect=[0.0, 0.09, 1.0, 0.95])
    save_figure_everywhere(fig, canonical_stem, out_dirs)
    return {
        "stem": canonical_stem,
        "tags": sorted({r["tag"] for r in used}),
        "decoders": dec_order,
        "json_sources": sorted({r["json_source"] for r in used}),
        "commands": [r["command_line"] for r in used],
        "export_cmd": f"python3 benchmarking/viz/export_static.py --figure {canonical_stem}",
    }


def plot_stage_breakdown(
    by_key: Dict[Tuple[str, str], Dict[str, Any]],
    tags: List[str],
    title: str,
    canonical_stem: str,
    out_dirs: List[pathlib.Path],
) -> Dict[str, Any]:
    """2-panel bar chart: (a) % shots resolved by BP vs OSD, (b) % CPU time per stage, for [O]."""
    tags = [t for t in tags if (t, "O") in by_key]
    is_wide = len(tags) > 5
    fig, (ax_shots, ax_cpu) = plt.subplots(1, 2, figsize=(17.2 if is_wide else 13.6, 7.0 if is_wide else 6.4), dpi=220)
    fig.suptitle(title, fontsize=13.5, fontweight="bold", y=0.985, color="#111111")
    x_idx = np.arange(len(tags))
    x_labels = []
    for t in tags:
        rec = by_key[(t, "O")]
        n_tel = rec["telemetry_num_shots"]
        n_str = "n/a" if n_tel is None else (f"{n_tel / 1000:.0f}k" if n_tel >= 1000 else str(n_tel))
        x_labels.append(f"{short_code_label(t)}\nn={n_str}")

    stage_defs = [
        ("stage1_bp", "Pass 0+1: BP converged", "#10b981"),
        ("stage1_osd", "OSD post-processing", "#3b82f6"),
    ]
    ok = [bool(by_key[(t, "O")]["telemetry_ok"]) for t in tags]

    for ax, suffix, ylab, ptitle in (
        (ax_shots, "_shots_pct", "Share of shots (%)", "(a) Share of shots resolved by BP vs. OSD"),
        (ax_cpu, "_cpu_pct", "Share of total decode CPU time (%)", "(b) Share of CPU time in BP vs. OSD"),
    ):
        bottom = np.zeros(len(tags))
        for s_key, s_label, s_color in stage_defs:
            vals = np.array([by_key[(t, "O")][s_key + suffix] if ok[i] else 0.0 for i, t in enumerate(tags)], dtype=float)
            ax.bar(x_idx, vals, bottom=bottom, width=0.6, label=s_label, color=s_color, edgecolor="white", linewidth=0.8)
            for i, v in enumerate(vals):
                if v >= 9.0:
                    ax.text(x_idx[i], bottom[i] + v * 0.5, f"{v:.1f}%", ha="center", va="center",
                            fontsize=7.4 if is_wide else 8.5, fontweight="bold", color="white")
            bottom += vals
        for i, t in enumerate(tags):
            rec = by_key[(t, "O")]
            if not ok[i]:
                ax.text(x_idx[i], 50, "telemetry\ntoo small /\nunavailable", ha="center", va="center",
                        fontsize=7.4, color="#64748b", rotation=90 if is_wide else 0)
                continue
            if suffix == "_shots_pct":
                badge = f"OSD {rec['stage1_osd_shots_pct']:.1f}%"
            else:
                badge = f"{fmt_time(rec['time_per_shot'])}/shot"
            ax.annotate(badge, xy=(x_idx[i], 101.5), rotation=90 if is_wide else 0, ha="center", va="bottom",
                        fontsize=6.9 if is_wide else 7.8, fontweight="bold", color="#1e293b",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#f8fafc", edgecolor="#cbd5e1", lw=0.7))
        ax.set_xticks(x_idx)
        ax.set_xticklabels(x_labels, fontsize=8.0 if is_wide else 9.0, rotation=30 if is_wide else 0,
                           ha="right" if is_wide else "center")
        ax.set_ylim(0, 140 if is_wide else 122)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_ylabel(ylab, fontsize=10.8, fontweight="bold")
        ax.set_title(ptitle, fontsize=11.4, fontweight="bold", pad=8)
        ax.grid(True, axis="y", linestyle="--", color="#d0d5dd", alpha=0.7)
        ax.set_axisbelow(True)

    handles, labels = ax_shots.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9.3, frameon=True, facecolor="white",
               edgecolor="#cbd5e1", bbox_to_anchor=(0.5, 0.005))
    plt.tight_layout(rect=[0.0, 0.06, 1.0, 0.95])
    save_figure_everywhere(fig, canonical_stem, out_dirs)
    recs = [by_key[(t, "O")] for i, t in enumerate(tags) if ok[i]]
    return {
        "stem": canonical_stem,
        "tags": [t for i, t in enumerate(tags) if ok[i]],
        "decoders": ["O"],
        "json_sources": sorted({r["json_source"] for r in recs}),
        "commands": [r["telemetry_command_line"] or r["command_line"] for r in recs],
        "export_cmd": f"python3 benchmarking/viz/export_static.py --figure {canonical_stem}",
    }


def plot_architecture_flow(
    by_key: Dict[Tuple[str, str], Dict[str, Any]],
    canonical_stem: str,
    out_dirs: List[pathlib.Path],
    ref_tag: str = "cc_d7",
) -> Dict[str, Any]:
    """Block & shot-flow diagram of [O] Tesseract-BP-OSD."""
    o_ref = by_key[(ref_tag, "O")]
    ref_name = short_code_label(ref_tag)
    BODY, TITLE = 12.5, 13.5

    def pct(v: Optional[float], nd: int = 2) -> str:
        return "n/a" if v is None else f"{v:.{nd}f}%"

    fig, ax = plt.subplots(figsize=(13.0, 7.5), dpi=220)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(50, 96.0, "[O] Tesseract-BP-OSD: 2-Pass BP + Multi-Seed OSD Pipeline",
            ha="center", va="center", fontsize=15.5, fontweight="bold", color="#0f172a")
    ax.text(50, 91.5,
            f"{ref_name}, {o_ref['num_shots']:,} shots ({o_ref['json_source']})",
            ha="center", va="center", fontsize=11.5, fontstyle="italic", color="#475569")

    def add_box(x, y, w, h, title, lines, fc, ec, title_color, lw=1.8, body=BODY):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.2",
                                    facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
        ax.text(x + w / 2.0, y + h - 1.4, title, ha="center", va="top", fontsize=TITLE, fontweight="bold",
                color=title_color, zorder=3)
        ax.text(x + w / 2.0, y + (h - 4.2) / 2.0, "\n".join(lines), ha="center", va="center", fontsize=body,
                color="#1e293b", linespacing=1.35, zorder=3)

    def arrow(x0, y0, x1, y1, color):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18, linewidth=2.2,
                                     color=color, zorder=4))

    s1_bp = o_ref["stage1_bp_shots_pct"]
    s1_osd_inv = 100.0 - s1_bp if s1_bp is not None else None

    y1, h = 48.0, 34.0
    add_box(1.5, y1, 30.0, h, "Stage 1: 2-Pass BP", [
        "Multi-pass DEM, serial min-sum",
        f"Converges: {pct(s1_bp)}",
        f"CPU share: {pct(o_ref['stage1_bp_cpu_pct'], 1)}",
    ], "#ecfdf5", "#047857", "#065f46")
    add_box(36.0, y1, 30.0, h, "Stage 2: Multi-Seed OSD", [
        "When BP does not converge",
        f"Invoked: {pct(s1_osd_inv)}",
        f"CPU share: {pct(o_ref['stage1_osd_cpu_pct'], 1)}",
    ], "#f0f9ff", "#0369a1", "#0c4a6e")
    o_lines = [
        "Minimum-weight codeword",
        f"{o_ref['num_errors']}/{o_ref['num_shots']:,} errors",
        f"{fmt_time(o_ref['time_per_round'])}/round CPU",
    ]
    if o_ref.get("speedup_vs_B"):
        o_lines.append(f"{o_ref['speedup_vs_B']:,.0f}x faster than [B]")
    add_box(70.5, y1, 28.0, h, "[O] Output", o_lines, "#f0fdfa", "#0f766e", "#115e59")
    arrow(32.0, y1 + h / 2, 35.5, y1 + h / 2, "#047857")
    arrow(66.5, y1 + h / 2, 70.0, y1 + h / 2, "#0f766e")

    summary = []
    for t in ("cc_d7", "cc_d9", "bb_d6", "mitten_d10"):
        o = by_key.get((t, "O"))
        if not (o and o["time_per_shot"]):
            continue
        spd = f" ({o['speedup_vs_B']:,.0f}x vs [B])" if o.get("speedup_vs_B") else ""
        summary.append(
            f"{short_code_label(t)}:  [O] {o['num_errors']}/{o['num_shots']:,} errors, {fmt_time(o['time_per_round'])}/round{spd}"
        )
    add_box(1.5, 4.0, 97.0, 34.0, "[O] Tesseract-BP-OSD across code families (canonical records)", summary,
            "#f8fafc", "#0f172a", "#0f172a", body=BODY - 1)

    save_figure_everywhere(fig, canonical_stem, out_dirs)
    used = [o_ref] + [by_key[(t, "O")] for t in ("cc_d9", "bb_d6", "mitten_d10") if (t, "O") in by_key]
    return {
        "stem": canonical_stem,
        "tags": sorted({r["tag"] for r in used}),
        "decoders": ["O"],
        "json_sources": sorted({r["json_source"] for r in used}),
        "commands": [r["command_line"] for r in used],
        "export_cmd": f"python3 benchmarking/viz/export_static.py --figure {canonical_stem}",
    }


def plot_dem_transforms(
    by_key: Dict[Tuple[str, str], Dict[str, Any]],
    canonical_stem: str,
    out_dirs: List[pathlib.Path],
) -> Dict[str, Any]:
    """3-panel engineering diagram comparing (a) [B] Monolithic DEM, (b) [G] Gari Gauge Elimination,
    and (c) [M] / [O] Multi-Pass DEM Shattering.
    """
    b_cc7 = by_key["cc_d7", "B"]
    g_cc7 = by_key["cc_d7", "G"]
    m_cc7 = by_key["cc_d7", "M"]
    o_cc7 = by_key["cc_d7", "O"]
    b_cc5 = by_key["cc_d5", "B"]
    g_cc5 = by_key["cc_d5", "G"]
    b_bb6 = by_key["bb_d6", "B"]
    g_bb6 = by_key["bb_d6", "G"]
    m_bb6 = by_key["bb_d6", "M"]
    b_mit = by_key["mitten_d10", "B"]
    g_mit = by_key["mitten_d10", "G"]
    o_mit = by_key["mitten_d10", "O"]

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 8.6), dpi=220)
    fig.suptitle(
        "Detector Error Model (DEM) Structural Transformations: [B] Monolithic vs [G] Gari Gauge Elimination vs [M]/[O] Multi-Pass Shattering",
        fontsize=13.5,
        fontweight="bold",
        color="#0f172a",
        y=0.985,
    )

    def setup_panel(ax: plt.Axes, title: str, subtitle: str, border_color: str, bg_color: str) -> None:
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_autoscale_on(False)
        ax.axis("off")
        ax.add_patch(
            FancyBboxPatch(
                (1.0, 1.0),
                98.0,
                97.5,
                boxstyle="round,pad=0.4,rounding_size=1.6",
                facecolor=bg_color,
                edgecolor=border_color,
                linewidth=2.2,
                zorder=1,
            )
        )
        ax.text(
            50.0,
            96.0,
            title,
            ha="center",
            va="top",
            fontsize=10.1,
            fontweight="bold",
            color=border_color,
            zorder=3,
        )
        ax.text(
            50.0,
            92.0,
            subtitle,
            ha="center",
            va="top",
            fontsize=8.0,
            fontstyle="italic",
            color="#334155",
            zorder=3,
        )

    def draw_matrix_grid(
        ax: plt.Axes,
        x0: float,
        y0: float,
        w: float,
        h: float,
        mat: np.ndarray,
        color_map: Dict[int, str],
        title: str,
    ) -> None:
        rows, cols = mat.shape
        ax.add_patch(
            FancyBboxPatch(
                (x0 - 2.5, y0 - 1.5),
                w + 5.0,
                h + 6.2,
                boxstyle="round,pad=0.2,rounding_size=0.9",
                facecolor="white",
                edgecolor="#cbd5e1",
                linewidth=1.2,
                zorder=2,
            )
        )
        ax.text(
            x0 + w / 2.0,
            y0 + h + 2.8,
            title,
            ha="center",
            va="center",
            fontsize=8.1,
            fontweight="bold",
            color="#0f172a",
            zorder=4,
        )
        cw = w / cols
        ch = h / rows
        for r in range(rows):
            for c in range(cols):
                val = int(mat[r, c])
                fc = color_map.get(val, "#f1f5f9")
                ec = "#e2e8f0" if val == 0 else "#0f172a"
                lw = 0.35 if val == 0 else 0.65
                ax.add_patch(
                    mpatches.Rectangle(
                        (x0 + c * cw, y0 + (rows - 1 - r) * ch),
                        cw * 0.88,
                        ch * 0.86,
                        facecolor=fc,
                        edgecolor=ec,
                        linewidth=lw,
                        zorder=3,
                    )
                )

    # -------------------------------------------------------------------------
    # PANEL (a): [B] Monolithic DEM
    # -------------------------------------------------------------------------
    ax_a = axes[0]
    setup_panel(
        ax_a,
        "(a) [B] Monolithic DEM (Coupled X/Z Hypergraph)",
        "Degree 4–6 cross-basis Y/hook hyperedges & dense short 4-cycles (C4)",
        "#1e293b",
        "#f8fafc",
    )

    # Upper schematic: coupled X & Z 3D lattice with degree-6 hyperedges & C4 cycles
    ax_a.add_patch(
        FancyBboxPatch(
            (5.0, 56.5),
            90.0,
            31.0,
            boxstyle="round,pad=0.3,rounding_size=1.0",
            facecolor="#0f172a",
            edgecolor="#475569",
            linewidth=1.3,
            zorder=2,
        )
    )
    ax_a.text(50.0, 85.2, "3D Spacetime Detector Hypergraph (Coupled X + Z)", ha="center", va="center",
              fontsize=9.2, fontweight="bold", color="#f8fafc", zorder=4)

    x_nodes_a = [(18, 76), (40, 78), (64, 76), (84, 77), (26, 62), (52, 61), (76, 62)]
    z_nodes_a = [(28, 74), (52, 75), (74, 73), (16, 63), (40, 64), (64, 63), (85, 64)]

    # Draw coupled hyperedges (red/magenta polygons & C4 rectangle)
    poly1 = mpatches.Polygon([(18, 76), (40, 78), (28, 74), (40, 64), (26, 62)], closed=True,
                             facecolor="#f43f5e", alpha=0.28, edgecolor="#fb7185", linewidth=1.6, zorder=3)
    poly2 = mpatches.Polygon([(64, 76), (84, 77), (74, 73), (64, 63), (76, 62)], closed=True,
                             facecolor="#f43f5e", alpha=0.28, edgecolor="#fb7185", linewidth=1.6, zorder=3)
    ax_a.add_patch(poly1)
    ax_a.add_patch(poly2)

    # Highlight a cross-basis 4-cycle C4
    c4_pts = [(40, 78), (64, 76), (64, 63), (40, 64)]
    ax_a.add_patch(
        mpatches.Polygon(c4_pts, closed=True, facecolor="none", edgecolor="#facc15",
                         linewidth=2.3, linestyle="--", zorder=4)
    )
    ax_a.text(52.0, 69.5, "Cross-basis C4 cycle\n(w_H = 4–6 fault)", ha="center", va="center",
              fontsize=8.0, fontweight="bold", color="#fef08a",
              bbox=dict(boxstyle="round,pad=0.2", fc="#1e293b", ec="#facc15", lw=0.9), zorder=5)

    for nx, ny in x_nodes_a:
        ax_a.scatter([nx], [ny], s=95, color="#38bdf8", edgecolors="white", linewidths=1.1, zorder=5)
    for nx, ny in z_nodes_a:
        ax_a.scatter([nx], [ny], s=95, color="#f59e0b", edgecolors="white", linewidths=1.1, zorder=5)

    ax_a.text(8.0, 58.5, "● X-detectors", fontsize=7.6, fontweight="bold", color="#38bdf8", zorder=5)
    ax_a.text(38.0, 58.5, "● Z-detectors", fontsize=7.6, fontweight="bold", color="#f59e0b", zorder=5)
    ax_a.text(66.0, 58.5, "▲ Y/hook hyperedge", fontsize=7.6, fontweight="bold", color="#fb7185", zorder=5)

    # Middle: Dense coupled parity-check matrix H
    mat_b = np.zeros((10, 14), dtype=int)
    coupled_cols = [
        [0, 2, 5, 7], [1, 3, 5, 8], [0, 3, 6, 8], [2, 4, 6, 9],
        [1, 4, 7, 9], [0, 2, 6, 9], [1, 3, 7, 8], [2, 3, 5, 9],
        [0, 4, 5, 8], [1, 2, 6, 7], [0, 1, 5, 6, 8], [2, 4, 7, 8, 9],
        [1, 3, 4, 6, 9], [0, 3, 5, 7, 9],
    ]
    for c_idx, rows_on in enumerate(coupled_cols):
        for r_idx in rows_on:
            mat_b[r_idx, c_idx] = 1 if r_idx < 5 else 2
    draw_matrix_grid(
        ax_a, 8.0, 26.5, 84.0, 22.5, mat_b,
        {0: "#f1f5f9", 1: "#0284c7", 2: "#d97706"},
        "Monolithic Parity-Check Matrix H (Weight 4–6 Columns Across X & Z)",
    )

    # Lower box: A* search & empirical metrics
    ax_a.add_patch(
        FancyBboxPatch(
            (5.0, 3.0),
            90.0,
            20.0,
            boxstyle="round,pad=0.35,rounding_size=1.0",
            facecolor="white",
            edgecolor="#94a3b8",
            linewidth=1.3,
            zorder=2,
        )
    )
    b_lines = [
        "• Search complexity: Wide beam (b = 20); every fired detector branches",
        "  over 4–6 coupled fault columns -> exponential A* queue growth.",
        f"• CC d=7: {fmt_time(b_cc7['time_per_round'])}/round, {b_cc7['num_errors']}/{b_cc7['num_shots']:,} errors (LER = {b_cc7['ler']:.2e})",
        f"• BB [[72,12,6]]: {fmt_time(b_bb6['time_per_round'])}/round, {b_bb6['num_errors']}/{b_bb6['num_shots']:,} errors",
        f"• Mitten [[150,30,10]]: {fmt_time(b_mit['time_per_round'])}/round, {b_mit['num_errors']}/{b_mit['num_shots']:,} errors",
    ]
    ax_a.text(7.5, 13.0, "\n".join(b_lines), ha="left", va="center", fontsize=8.35, color="#1e293b",
              linespacing=1.36, zorder=4)

    # -------------------------------------------------------------------------
    # PANEL (b): [G] Gari Gauge Elimination
    # -------------------------------------------------------------------------
    ax_b = axes[1]
    setup_panel(
        ax_b,
        "(b) [G] Gari Gauge Elimination (R H P = [I_r | H_fill-in])",
        "Offline F2 row XORs create unit-weight pivots (w_H=1) + non-local column fill-in",
        "#0284c7",
        "#f0f9ff",
    )

    # Upper schematic: Green unit-weight pivot pins + crimson/purple non-local fill-in arcs
    ax_b.add_patch(
        FancyBboxPatch(
            (5.0, 56.5),
            90.0,
            31.0,
            boxstyle="round,pad=0.3,rounding_size=1.0",
            facecolor="#0f172a",
            edgecolor="#0284c7",
            linewidth=1.3,
            zorder=2,
        )
    )
    ax_b.text(50.0, 85.2, "Eliminated Basis (d_i <- d_i XOR d_k): Unit Pins + Non-Local Fill-In",
              ha="center", va="center", fontsize=8.2, fontweight="bold", color="#f8fafc", zorder=4)

    piv_nodes = [(14, 75), (29, 76), (44, 75), (19, 63.5), (36, 63.5)]
    other_nodes = [(56, 76), (72, 77), (87, 75), (56, 63.5), (72, 63), (87, 64)]

    # Draw crimson/purple non-local fill-in arcs stretching across non-pivot right sublattice
    fill_pairs = [
        ((56, 76), (87, 75), 0.26),
        ((56, 63.5), (87, 64), -0.26),
        ((56, 76), (87, 64), 0.18),
        ((56, 63.5), (87, 75), -0.18),
        ((72, 77), (87, 64), 0.22),
    ]
    for p0, p1, rad in fill_pairs:
        ax_b.add_patch(
            FancyArrowPatch(
                p0, p1, connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-", linewidth=2.0, linestyle="--", color="#f43f5e", alpha=0.88, zorder=3
            )
        )

    # Draw green unit-weight pivot pins (w_H = 1)
    for idx_p, (nx, ny) in enumerate(piv_nodes, 1):
        ax_b.scatter([nx], [ny], s=180, facecolors="none", edgecolors="#22c55e", linewidths=2.2, zorder=5)
        ax_b.scatter([nx], [ny], s=85, color="#4ade80", edgecolors="white", linewidths=1.0, zorder=6)
        ax_b.text(
            nx, ny + 4.3, f"pin#{idx_p}\n(w_H=1)", ha="center", va="center",
            fontsize=6.9, fontweight="bold", color="#4ade80",
            bbox=dict(boxstyle="round,pad=0.12", fc="#0f172a", ec="none"), zorder=7
        )

    for nx, ny in other_nodes:
        ax_b.scatter([nx], [ny], s=90, color="#c084fc", edgecolors="white", linewidths=1.0, zorder=5)

    ax_b.text(71.5, 69.8, "Non-local fill-in arcs\n(w_H grows to 5–9+;\nstatic XOR priors)",
              ha="center", va="center", fontsize=7.4, fontweight="bold", color="#fda4af",
              bbox=dict(boxstyle="round,pad=0.2", fc="#1e293b", ec="#f43f5e", lw=0.9), zorder=6)

    ax_b.text(8.0, 58.5, "● Unit-weight pivot (w_H=1)", fontsize=7.5, fontweight="bold", color="#4ade80", zorder=5)
    ax_b.text(55.0, 58.5, "--- Non-local fill-in arcs", fontsize=7.5, fontweight="bold", color="#fb7185", zorder=5)

    # Middle: R H P = [I_r | H_fill-in]
    mat_g = np.zeros((10, 14), dtype=int)
    for i in range(6):
        mat_g[i, i] = 3  # Green diagonal pivot I_r
    fill_cols = [
        [0, 1, 3, 5, 6, 8],
        [0, 2, 4, 5, 7, 9],
        [1, 2, 3, 6, 7, 8],
        [0, 3, 4, 6, 8, 9],
        [1, 2, 4, 5, 7, 8, 9],
        [0, 1, 3, 5, 6, 7, 9],
        [2, 3, 4, 5, 6, 8, 9],
        [0, 1, 2, 4, 6, 7, 8],
    ]
    for offset_c, rows_on in enumerate(fill_cols):
        c_idx = 6 + offset_c
        for r_idx in rows_on:
            mat_g[r_idx, c_idx] = 4 if r_idx < 6 else 5
    draw_matrix_grid(
        ax_b, 8.0, 26.5, 84.0, 22.5, mat_g,
        {0: "#f1f5f9", 3: "#16a34a", 4: "#e11d48", 5: "#9333ea"},
        "Eliminated Matrix R H P = [I_r (Unit Pivots) | H_fill-in (Non-Local)]",
    )

    # Divider line between I_r and H_fill-in
    div_x = 8.0 + 84.0 * (6.0 / 14.0)
    ax_b.plot([div_x, div_x], [26.0, 49.5], color="#0f172a", linewidth=2.0, linestyle="--", zorder=5)

    # Lower box: Speedup vs Color-Code fill-in trade-offs
    ax_b.add_patch(
        FancyBboxPatch(
            (5.0, 3.0),
            90.0,
            20.0,
            boxstyle="round,pad=0.35,rounding_size=1.0",
            facecolor="white",
            edgecolor="#0284c7",
            linewidth=1.3,
            zorder=2,
        )
    )
    cc5_pen = g_cc5["ler"] / b_cc5["ler"]
    g_lines = [
        "• Why [G] speeds up A* (8x–166x): Unit-weight pins (w_H=1) ordered first",
        "  resolve greedily at narrow beam (b=5) with zero A* branching:",
        f"  CC d=5: {g_cc5['speedup_vs_B']:.1f}x; CC d=7: {g_cc7['speedup_vs_B']:.1f}x ({fmt_time(g_cc7['time_per_round'])}); "
        f"BB d=6: {g_bb6['speedup_vs_B']:.1f}x; Mitten: {g_mit['speedup_vs_B']:.1f}x.",
        "• Structural limitation: Row XORs create non-local fill-in (w_H -> 5–9+) &",
        f"  static XOR priors (CC d=5 LER {cc5_pen:.2f}x [B]; dense 4-cycles block BP).",
    ]
    ax_b.text(7.5, 13.0, "\n".join(g_lines), ha="left", va="center", fontsize=8.1, color="#1e293b",
              linespacing=1.36, zorder=4)

    # -------------------------------------------------------------------------
    # PANEL (c): [M] / [O] Multi-Pass DEM Shattering
    # -------------------------------------------------------------------------
    ax_c = axes[2]
    setup_panel(
        ax_c,
        "(c) [M] / [O] Multi-Pass DEM Shattering (D0 -> D1)",
        "Zero row elimination, zero fill-in, preserved locality + dynamic per-shot LLR update",
        "#059669",
        "#ecfdf5",
    )

    # Upper schematic: Separated D0 (X) and D1 (Z) planes connected by vertical LLR fibers
    ax_c.add_patch(
        FancyBboxPatch(
            (5.0, 56.5),
            90.0,
            31.0,
            boxstyle="round,pad=0.3,rounding_size=1.0",
            facecolor="#0f172a",
            edgecolor="#059669",
            linewidth=1.3,
            zorder=2,
        )
    )
    ax_c.text(50.0, 85.2, "Shattered Sector Planes: Pass 0 (D0: X) -> Pass 1 (D1: Z), Zero Fill-In",
              ha="center", va="center", fontsize=8.1, fontweight="bold", color="#f8fafc", zorder=4)

    # Pass 0 plane (top) and Pass 1 plane (bottom)
    ax_c.add_patch(mpatches.Rectangle((9.0, 74.0), 82.0, 8.2, facecolor="#0284c7", alpha=0.18,
                                      edgecolor="#38bdf8", linewidth=1.1, zorder=3))
    ax_c.add_patch(mpatches.Rectangle((9.0, 59.5), 82.0, 8.2, facecolor="#d97706", alpha=0.18,
                                      edgecolor="#f59e0b", linewidth=1.1, zorder=3))
    ax_c.text(11.0, 80.3, "Pass 0: D0 (X-sector, w_H <= 3)", fontsize=7.4, fontweight="bold", color="#38bdf8", zorder=5)
    ax_c.text(11.0, 61.0, "Pass 1: D1 (Z-sector, w_H <= 3)", fontsize=7.4, fontweight="bold", color="#fbbf24", zorder=5)

    d0_pts = [(22, 76.5), (42, 76.5), (62, 76.5), (80, 76.5)]
    d1_pts = [(22, 65.2), (42, 65.2), (62, 65.2), (80, 65.2)]
    for (x0_p, y0_p), (x1_p, y1_p) in zip(d0_pts, d1_pts):
        ax_c.add_patch(
            FancyArrowPatch(
                (x0_p, y0_p - 1.4), (x1_p, y1_p + 1.4),
                arrowstyle="-|>", mutation_scale=11, linewidth=1.8, linestyle="--", color="#34d399", zorder=4
            )
        )
        ax_c.scatter([x0_p], [y0_p], s=85, color="#38bdf8", edgecolors="white", linewidths=1.0, zorder=5)
        ax_c.scatter([x1_p], [y1_p], s=85, color="#f59e0b", edgecolors="white", linewidths=1.0, zorder=5)

    ax_c.text(50.0, 70.85, "Dynamic Per-Shot Soft LLR Update (75% boost / 25% attenuation)",
              ha="center", va="center", fontsize=7.2, fontweight="bold", color="#6ee7b7",
              bbox=dict(boxstyle="round,pad=0.18", fc="#064e3b", ec="#10b981", lw=0.8), zorder=6)

    # Middle: Block-split sparse matrices H^(0) and H^(1)
    mat_m = np.zeros((10, 14), dtype=int)
    pass0_cols = [[0, 1], [1, 2], [2, 3], [0, 3], [1, 4], [2, 4], [0, 2, 4]]
    pass1_cols = [[5, 6], [6, 7], [7, 8], [5, 8], [6, 9], [7, 9], [5, 7, 9]]
    for c_idx, rows_on in enumerate(pass0_cols):
        for r_idx in rows_on:
            mat_m[r_idx, c_idx] = 1
    for offset_c, rows_on in enumerate(pass1_cols):
        c_idx = 7 + offset_c
        for r_idx in rows_on:
            mat_m[r_idx, c_idx] = 2
    draw_matrix_grid(
        ax_c, 8.0, 26.5, 84.0, 22.5, mat_m,
        {0: "#f1f5f9", 1: "#0284c7", 2: "#d97706"},
        "Shattered H^(0) = H[D0,:] & H^(1) = H[D1,:] (Sparse w_H<=3, Zero Fill-In)",
    )
    # Horizontal & vertical sector divider
    mid_x = 8.0 + 84.0 * 0.5
    mid_y = 26.5 + 22.5 * 0.5
    ax_c.plot([mid_x, mid_x], [26.5, 49.0], color="#059669", linewidth=1.8, linestyle="--", zorder=5)
    ax_c.plot([8.0, 92.0], [mid_y, mid_y], color="#059669", linewidth=1.8, linestyle="--", zorder=5)

    # Lower box: Speed + Accuracy Pareto dominance
    ax_c.add_patch(
        FancyBboxPatch(
            (5.0, 3.0),
            90.0,
            20.0,
            boxstyle="round,pad=0.35,rounding_size=1.0",
            facecolor="white",
            edgecolor="#059669",
            linewidth=1.3,
            zorder=2,
        )
    )
    m_lines = [
        "• Zero Fill-In + Dynamic Conditioning: Preserves spatial locality & code",
        "  distance while eliminating cross-basis 4-cycles (C4) for both A* and BP-OSD.",
        f"• [M] Multi-Pass A*: {m_cc7['speedup_vs_B']:.0f}x faster than [B] on CC d=7 ({fmt_time(m_cc7['time_per_round'])},",
        f"  {m_cc7['speedup_vs_G']:.1f}x faster than [G]); {m_bb6['speedup_vs_B']:.0f}x faster on BB d=6 ({m_bb6['speedup_vs_G']:.1f}x vs [G]).",
        f"• [O] Multi-Pass BP-OSD: {o_cc7['speedup_vs_B']:,.0f}x faster on CC d=7 ({fmt_time(o_cc7['time_per_round'])}); "
        f"{o_mit['speedup_vs_B']:,.0f}x on Mitten.",
    ]
    ax_c.text(7.5, 13.0, "\n".join(m_lines), ha="left", va="center", fontsize=8.1, color="#1e293b",
              linespacing=1.36, zorder=4)

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.955])
    save_figure_everywhere(fig, canonical_stem, out_dirs)
    used = [b_cc7, g_cc7, m_cc7, o_cc7, b_cc5, g_cc5, b_bb6, g_bb6, m_bb6, b_mit, g_mit, o_mit]
    return {
        "stem": canonical_stem,
        "tags": ["cc_d5", "cc_d7", "bb_d6", "mitten_d10"],
        "decoders": ["B", "G", "M", "O"],
        "json_sources": sorted({r["json_source"] for r in used}),
        "commands": [r["command_line"] for r in used],
        "export_cmd": f"python3 benchmarking/viz/export_static.py --figure {canonical_stem}",
    }


def _find_ablation_json() -> Optional[pathlib.Path]:
    candidates = [
        DATA_DIR / "ablation_4circuits_results.json",
        pathlib.Path(
            "/usr/local/google/home/ariash/.gemini/jetski/brain/67c42a3b-49c1-44cb-a2ce-ec1851dd8f89/scratch/ablation_audit/ablation_4circuits_results.json"
        ),
        pathlib.Path("/tmp/ablation_audit/ablation_4circuits_results.json"),
    ]
    existing = []
    for p in candidates:
        try:
            if p.exists():
                existing.append((p.stat().st_mtime, p))
        except Exception:
            continue
    if not existing:
        return None
    existing.sort(key=lambda x: x[0], reverse=True)
    return existing[0][1]


def _fmt_ms_short(ms: float) -> str:
    if ms >= 100.0:
        return f"{ms:.0f}ms"
    if ms >= 10.0:
        return f"{ms:.1f}ms"
    return f"{ms:.2f}ms"


def _fmt_ler_short(ler: float, errors: int, shots: int, rounds: int) -> str:
    if errors == 0 or ler <= 0.0:
        return f"0/{shots}"
    exp = int(math.floor(math.log10(ler)))
    mant = ler / (10.0 ** exp)
    return f"{mant:.2f}e{exp}"


def plot_ablation_4circuits_suite(
    out_dirs: List[pathlib.Path],
    target_figure: Optional[str] = None,
) -> List[Dict[str, Any]]:
    jpath = _find_ablation_json()
    if jpath is None:
        return []
    raw = json.loads(jpath.read_text())
    circuits_dict = raw.get("circuits", {})
    c_order = [
        ("cc_d11", "Color Code\nd=11 (r=11)"),
        ("bb_d6", "Bivariate Bicycle\n[[72, 12, 6]]"),
        ("mitten_d10", "Mitten Code\n[[150, 30, 10]]"),
        ("hyp_iter40", "Hyperbolic Oct.\niter=40 [[288,12,8]]"),
    ]
    entries: List[Dict[str, Any]] = []

    def should_run(stem: str) -> bool:
        return target_figure is None or target_figure == stem

    # -------------------------------------------------------------------------
    # 1. fig_ablation_bp_comp (Progression A — 3 bars per circuit)
    # -------------------------------------------------------------------------
    stem_a = "fig_ablation_bp_comp"
    if should_run(stem_a):
        a_labels = [
            ("A1", "A1: Scalar Monolithic BP (1.00×)", "#64748B"),
            ("A2", "A2: +16-Lane AVX2 SIMD Batching (--bp-simd-batch) [0.0% LER Δ]", "#1967D2"),
            ("A3", "A3: +Multi-Pass DEM/Tanner Compression (--multipass, default)", "#1E8E3E"),
        ]
        fig, ax = plt.subplots(figsize=(12.2, 6.2), dpi=220)
        fig.suptitle(
            "BP Computational Ablation Across 4 Quantum Code Families (16-Lane SIMD: 0.0% LER Change)",
            fontsize=13.5,
            fontweight="bold",
            color="#0f172a",
            y=0.98,
        )
        x = np.arange(len(c_order))
        width = 0.24
        max_sp = 10.0
        cmds_a: List[str] = []

        for s_idx, (s_key, s_label, s_color) in enumerate(a_labels):
            vals = []
            ms_vals = []
            for cid, _ in c_order:
                steps = circuits_dict[cid]["progressions"]["A_bp_computational"]
                if len(steps) > 3:
                    steps = [st for st in steps if "compaction" not in st["step_id"]]
                st0 = steps[0]
                st = steps[min(s_idx, len(steps) - 1)]
                cmds_a.append(st.get("full_command", ""))
                sp = st0["bp_ms_per_shot"] / max(1e-9, st["bp_ms_per_shot"])
                vals.append(sp)
                ms_vals.append(st["bp_ms_per_shot"])
                max_sp = max(max_sp, sp)

            offset = (s_idx - 1.0) * (width + 0.02)
            bars = ax.bar(
                x + offset,
                vals,
                width=width,
                color=s_color,
                edgecolor="#1e293b",
                linewidth=1.1 if s_idx == 2 else 0.8,
                label=s_label,
                zorder=3,
            )
            for b, sp, ms in zip(bars, vals, ms_vals):
                sp_str = f"{sp:.1f}×" if sp >= 10.0 else f"{sp:.2f}×"
                lbl = f"{sp_str}\n({_fmt_ms_short(ms)})"
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    sp * 1.08,
                    lbl,
                    ha="center",
                    va="bottom",
                    fontsize=8.6,
                    fontweight="bold" if s_idx == 2 else "normal",
                    color="#14532D" if s_idx == 2 else "#1e293b",
                )

        ax.set_yscale("log")
        ax.set_ylim(0.7, max_sp * 2.3)
        ax.set_ylabel("Cumulative BP Speedup vs. A1 (×, log)", fontsize=11.5, fontweight="bold", color="#0f172a")
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in c_order], fontsize=11, fontweight="bold", color="#0f172a")
        ax.grid(True, which="both", axis="y", linestyle="--", color="#cbd5e1", alpha=0.65, zorder=0)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cbd5e1")
            spine.set_linewidth(1.1)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=3,
            fontsize=8.6,
            frameon=True,
            facecolor="#f8fafc",
            edgecolor="#cbd5e1",
        )
        plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
        save_figure_everywhere(fig, stem_a, out_dirs)
        entries.append(
            {
                "stem": stem_a,
                "tags": [c for c, _ in c_order],
                "decoders": ["O"],
                "json_sources": ["benchmarking/data/ablation_4circuits_results.json"],
                "commands": [c for c in cmds_a if c][:6],
                "export_cmd": f"python3 benchmarking/viz/export_static.py --figure {stem_a}",
            }
        )

    # -------------------------------------------------------------------------
    # 2. fig_ablation_bp_algo (Progression B — 3 bars per circuit, dual plot)
    # -------------------------------------------------------------------------
    stem_b = "fig_ablation_bp_algo"
    if should_run(stem_b):
        b_labels = [
            ("B1", "B1: Flooding + Unweighted (--no-bp-serial --bp-reweight=none)", "#64748B"),
            ("B2", "B2: +Serial Check Schedule (--bp-serial --bp-reweight=none)", "#1967D2"),
            ("B3", "B3: +Soft-Power Edge Reweighting (--bp-serial --bp-reweight=soft, default)", "#1E8E3E"),
        ]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.4, 6.0), dpi=220)
        x = np.arange(len(c_order))
        width = 0.25
        max_gain = 15.0
        cmds_b: List[str] = []

        for s_idx, (s_key, s_label, s_color) in enumerate(b_labels):
            convs = []
            gains = []
            gain_labels = []
            for cid, _ in c_order:
                steps = circuits_dict[cid]["progressions"]["B_bp_algorithmic"]
                if len(steps) > 3:
                    steps = [st for st in steps if "random_noreweight" not in st["step_id"]]
                st0 = steps[0]
                st = steps[min(s_idx, len(steps) - 1)]
                cmds_b.append(st.get("full_command", ""))
                conv = float(st["primary_bp_convergence_rate_pct"])
                convs.append(conv)
                ler0 = float(st0["ler_per_round"])
                ler_cur = float(st["ler_per_round"])
                errs = int(st["errors"])
                shots = int(st["shots"])
                rounds = int(st["rounds"])
                if errs == 0 or ler_cur <= 0.0:
                    ub_ler = shot_to_round_ler(wilson_interval(0, shots)[1], rounds)
                    g = max(2.0, ler0 / max(1e-9, ub_ler))
                    gains.append(g)
                    exp_u = int(math.floor(math.log10(ub_ler)))
                    mant_u = ub_ler / (10.0 ** exp_u)
                    gain_labels.append(f"0 errs\n(<{mant_u:.1f}e{exp_u})")
                else:
                    g = ler0 / ler_cur if ler0 > 0 else 1.0
                    gains.append(g)
                    gain_labels.append(f"{g:.2f}×\n{_fmt_ler_short(ler_cur, errs, shots, rounds)}")
                max_gain = max(max_gain, gains[-1])

            offset = (s_idx - 1.0) * (width + 0.03)
            b1 = ax1.bar(
                x + offset,
                convs,
                width=width,
                color=s_color,
                edgecolor="#1e293b",
                linewidth=1.1 if s_idx == 2 else 0.8,
                label=s_label,
                zorder=3,
            )
            for bar, cv in zip(b1, convs):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    cv + 1.8,
                    f"{cv:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8.2,
                    fontweight="bold" if s_idx == 2 else "normal",
                    color="#14532D" if s_idx == 2 else "#1e293b",
                )

            b2 = ax2.bar(
                x + offset,
                gains,
                width=width,
                color=s_color,
                edgecolor="#1e293b",
                linewidth=1.1 if s_idx == 2 else 0.8,
                label=s_label,
                zorder=3,
            )
            for bar, gn, glbl in zip(b2, gains, gain_labels):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    gn * 1.08,
                    glbl,
                    ha="center",
                    va="bottom",
                    fontsize=7.3,
                    fontweight="bold" if s_idx == 2 else "normal",
                    color="#14532D" if s_idx == 2 else "#1e293b",
                )

        ax1.set_ylim(0, 114)
        ax1.set_ylabel("Primary Pure-BP Convergence Rate (%)", fontsize=11, fontweight="bold", color="#0f172a")
        ax1.set_title("(a) Primary Pure-BP Convergence Rate (Cuts OSD Load)", fontsize=11.5, fontweight="bold", pad=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels([lbl for _, lbl in c_order], fontsize=10, fontweight="bold", color="#0f172a")
        ax1.grid(True, axis="y", linestyle="--", color="#cbd5e1", alpha=0.65, zorder=0)

        ax2.set_yscale("log")
        ax2.set_ylim(0.45, max_gain * 2.4)
        ax2.set_ylabel("End-to-End LER Reduction vs. B1 (×, log)", fontsize=11, fontweight="bold", color="#0f172a")
        ax2.set_title("(b) End-to-End LER Reduction Factor (with LER/round)", fontsize=11.5, fontweight="bold", pad=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels([lbl for _, lbl in c_order], fontsize=10, fontweight="bold", color="#0f172a")
        ax2.grid(True, which="both", axis="y", linestyle="--", color="#cbd5e1", alpha=0.65, zorder=0)

        for ax in (ax1, ax2):
            for spine in ax.spines.values():
                spine.set_edgecolor("#cbd5e1")
                spine.set_linewidth(1.1)

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=3,
            fontsize=8.8,
            frameon=True,
            facecolor="#f8fafc",
            edgecolor="#cbd5e1",
        )
        plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
        save_figure_everywhere(fig, stem_b, out_dirs)
        entries.append(
            {
                "stem": stem_b,
                "tags": [c for c, _ in c_order],
                "decoders": ["O"],
                "json_sources": ["benchmarking/data/ablation_4circuits_results.json"],
                "commands": [c for c in cmds_b if c][:6],
                "export_cmd": f"python3 benchmarking/viz/export_static.py --figure {stem_b}",
            }
        )

    # -------------------------------------------------------------------------
    # 3. fig_ablation_osd_comp (Progression C — 3 bars per circuit)
    # -------------------------------------------------------------------------
    stem_c = "fig_ablation_osd_comp"
    if should_run(stem_c):
        c_labels = [
            ("C1", "C1: Full RREF + Full Sort (1.00×)", "#64748B"),
            ("C2", "C2: +Dynamic Zero-Remainder Early Exit (--osd-early-exit) [0.0% LER Δ]", "#1967D2"),
            ("C3", "C3: +Horizon-Bounded Partial Sort (K=1.2M, nth_element) (--osd-col-truncation --osd-partial-sort)", "#1E8E3E"),
        ]
        fig, ax = plt.subplots(figsize=(12.2, 6.2), dpi=220)
        fig.suptitle(
            "OSD Computational Ablation Across 4 Quantum Code Families (0.0% LER Change — Bit-for-Bit Identical Codewords)",
            fontsize=13.0,
            fontweight="bold",
            color="#0f172a",
            y=0.98,
        )
        x = np.arange(len(c_order))
        width = 0.24
        max_sp = 10.0
        cmds_c: List[str] = []

        for s_idx, (s_key, s_label, s_color) in enumerate(c_labels):
            vals = []
            ms_vals = []
            for cid, _ in c_order:
                steps = circuits_dict[cid]["progressions"]["C_osd_computational"]
                if len(steps) > 3:
                    steps = [st for st in steps if st["step_id"] != "C3_col_truncation"]
                st0 = steps[0]
                st = steps[min(s_idx, len(steps) - 1)]
                cmds_c.append(st.get("full_command", ""))
                sp = st0["osd_ms_per_osd_shot"] / max(1e-9, st["osd_ms_per_osd_shot"])
                vals.append(sp)
                ms_vals.append(st["osd_ms_per_osd_shot"])
                max_sp = max(max_sp, sp)

            offset = (s_idx - 1.0) * (width + 0.02)
            bars = ax.bar(
                x + offset,
                vals,
                width=width,
                color=s_color,
                edgecolor="#1e293b",
                linewidth=1.1 if s_idx == 2 else 0.8,
                label=s_label,
                zorder=3,
            )
            for b, sp, ms in zip(bars, vals, ms_vals):
                lbl = f"{sp:.2f}×\n({_fmt_ms_short(ms)})"
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    sp * 1.07,
                    lbl,
                    ha="center",
                    va="bottom",
                    fontsize=8.6,
                    fontweight="bold" if s_idx == 2 else "normal",
                    color="#14532D" if s_idx == 2 else "#1e293b",
                )

        ax.set_yscale("log")
        ax.set_ylim(0.75, max_sp * 2.15)
        ax.set_ylabel("Cumulative Isolated OSD Speedup vs. C1 (×, log)", fontsize=11.5, fontweight="bold", color="#0f172a")
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in c_order], fontsize=11, fontweight="bold", color="#0f172a")
        ax.grid(True, which="both", axis="y", linestyle="--", color="#cbd5e1", alpha=0.65, zorder=0)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cbd5e1")
            spine.set_linewidth(1.1)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=2,
            fontsize=8.4,
            frameon=True,
            facecolor="#f8fafc",
            edgecolor="#cbd5e1",
        )
        plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.89])
        save_figure_everywhere(fig, stem_c, out_dirs)
        entries.append(
            {
                "stem": stem_c,
                "tags": [c for c, _ in c_order],
                "decoders": ["O"],
                "json_sources": ["benchmarking/data/ablation_4circuits_results.json"],
                "commands": [c for c in cmds_c if c][:6],
                "export_cmd": f"python3 benchmarking/viz/export_static.py --figure {stem_c}",
            }
        )

    # -------------------------------------------------------------------------
    # 4. fig_ablation_osd_algo (Progression D — 4 bars per circuit D0..D3)
    # -------------------------------------------------------------------------
    stem_d = "fig_ablation_osd_algo"
    if should_run(stem_d):
        d_labels = [
            ("D0", "D0: Pure BP (--osd-order -1, no OSD)", "#64748B"),
            ("D1", "D1: +Naive OSD-0 (Final Iteration L^(T))", "#1967D2"),
            ("D2", "D2: +Trajectory Seed Diversification (Min-Syndrome L^(t*) + 2 Damped Relay Legs)", "#00796B"),
            ("D3", "D3: +Higher-Order OSD(O10/W2) & α-ISD (Default)", "#1E8E3E"),
        ]
        fig, ax = plt.subplots(figsize=(12.6, 6.3), dpi=220)
        fig.suptitle(
            "OSD Algorithmic Ablation Across 4 Quantum Code Families (LER Reduction vs. Pure BP D0, LER/round & Total ms/shot)",
            fontsize=12.6,
            fontweight="bold",
            color="#0f172a",
            y=0.98,
        )
        x = np.arange(len(c_order))
        width = 0.195
        max_gain = 20.0
        cmds_d: List[str] = []

        for s_idx, (s_key, s_label, s_color) in enumerate(d_labels):
            vals = []
            top_lbls = []
            for cid, _ in c_order:
                steps = circuits_dict[cid]["progressions"]["D_osd_algorithmic"]
                if len(steps) > 4:
                    steps = [st for st in steps if st["step_id"] != "D2_min_syndrome_osd0"]
                st0 = steps[0]
                st = steps[min(s_idx, len(steps) - 1)]
                cmds_d.append(st.get("full_command", ""))
                ler0 = float(st0["ler_per_round"])
                ler_cur = float(st["ler_per_round"])
                errs = int(st["errors"])
                shots = int(st["shots"])
                rounds = int(st["rounds"])
                ms = float(st["total_decode_ms_per_shot"])
                if errs == 0 or ler_cur <= 0.0:
                    ub_ler = shot_to_round_ler(wilson_interval(0, shots)[1], rounds)
                    gain = max(2.0, ler0 / max(1e-9, ub_ler))
                else:
                    gain = ler0 / ler_cur if ler0 > 0 else 1.0
                vals.append(gain)
                max_gain = max(max_gain, gain)
                g_str = f"{gain:.0f}×" if gain >= 100.0 else (f"{gain:.1f}×" if gain >= 10.0 else f"{gain:.2f}×")
                ler_s = _fmt_ler_short(ler_cur, errs, shots, rounds)
                top_lbls.append(f"{g_str}\n{ler_s}\n[{_fmt_ms_short(ms)}]")

            offset = (s_idx - 1.5) * (width + 0.022)
            bars = ax.bar(
                x + offset,
                vals,
                width=width,
                color=s_color,
                edgecolor="#1e293b",
                linewidth=1.1 if s_idx == 3 else 0.8,
                label=s_label,
                zorder=3,
            )
            for b, gn, lbl in zip(bars, vals, top_lbls):
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    gn * 1.09,
                    lbl,
                    ha="center",
                    va="bottom",
                    fontsize=7.1,
                    fontweight="bold" if s_idx == 3 else "normal",
                    color="#14532D" if s_idx == 3 else "#1e293b",
                )

        ax.set_yscale("log")
        ax.set_ylim(0.55, max_gain * 3.2)
        ax.set_ylabel("Cumulative LER Reduction vs. D0 Pure BP (×, log)", fontsize=11.5, fontweight="bold", color="#0f172a")
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in c_order], fontsize=11, fontweight="bold", color="#0f172a")
        ax.grid(True, which="both", axis="y", linestyle="--", color="#cbd5e1", alpha=0.65, zorder=0)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cbd5e1")
            spine.set_linewidth(1.1)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=2,
            fontsize=8.3,
            frameon=True,
            facecolor="#f8fafc",
            edgecolor="#cbd5e1",
        )
        plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.89])
        save_figure_everywhere(fig, stem_d, out_dirs)
        entries.append(
            {
                "stem": stem_d,
                "tags": [c for c, _ in c_order],
                "decoders": ["O"],
                "json_sources": ["benchmarking/data/ablation_4circuits_results.json"],
                "commands": [c for c in cmds_d if c][:6],
                "export_cmd": f"python3 benchmarking/viz/export_static.py --figure {stem_d}",
            }
        )

    return entries


def write_provenance_tex(provenance_entries: List[Dict[str, Any]], out_dirs: List[pathlib.Path]) -> str:
    lines = [
        "% =============================================================================",
        "% AUTO-GENERATED FIGURE PROVENANCE MANIFEST (benchmarking/viz/export_static.py)",
        f"% Git Commit: {GIT_COMMIT}",
        "% =============================================================================",
        "",
    ]
    for entry in provenance_entries:
        stem = entry["stem"]
        lines.append(f"% === PROVENANCE: figures/{stem}.pdf ===")
        lines.append(f"% Figure Stem     : {stem}")
        lines.append(f"% Git Commit      : {GIT_COMMIT}")
        lines.append(f"% Source JSON(s)  : {', '.join(entry['json_sources'])}")
        lines.append(f"% Circuits / Tags : {', '.join(entry['tags'])}")
        lines.append(f"% Decoders        : {', '.join(entry['decoders'])}")
        lines.append(f"% Export Command  : {entry['export_cmd']}")
        for idx, cmd in enumerate(entry["commands"], 1):
            lines.append(f"% CLI Command [{idx}] : {cmd}")
        lines.append("% =============================================================================")
        lines.append("")

    content = "\n".join(lines)
    for d in out_dirs + [VIZ_DIR]:
        try:
            d.mkdir(parents=True, exist_ok=True)
            (d / "figures_provenance.tex").write_text(content)
        except Exception:
            pass
    return content


def write_datasets_js(
    by_key: Dict[Tuple[str, str], Dict[str, Any]],
    psweep_records: List[Dict[str, Any]],
    provenance_entries: List[Dict[str, Any]],
) -> str:
    records_list = [by_key[k] for k in sorted(by_key.keys())]
    # Append non-(p=0.001) psweep records to records_list as well (tagged e.g. bb_d6_p0.0005)
    for pr in psweep_records:
        if abs(float(pr.get("p") or 0.001) - 0.001) > 1e-7:
            records_list.append(pr)
    recs_str = ",\n    ".join(json.dumps(r) for r in records_list)
    psweep_str = ",\n    ".join(json.dumps(r) for r in psweep_records)
    prov_map = {e["stem"]: e for e in provenance_entries}
    decoders_export = dict(DECODER_META)
    js_content = (
        "/* Auto-generated by benchmarking/viz/export_static.py — DO NOT EDIT MANUALLY */\n"
        "window.TESSERACT_VIZ_DATASETS = {\n"
        f'  "git_commit": {json.dumps(GIT_COMMIT)},\n'
        '  "generated_by": "benchmarking/viz/export_static.py",\n'
        f'  "decoders": {json.dumps(decoders_export)},\n'
        f'  "circuits": {json.dumps(CIRCUIT_META)},\n'
        f'  "records": [\n    {recs_str}\n  ],\n'
        f'  "psweep_records": [\n    {psweep_str}\n  ],\n'
        f'  "figures_provenance": {json.dumps(prov_map)}\n'
        "};\n"
        "window.TESSERACT_PLOTS_DATA = window.TESSERACT_VIZ_DATASETS;\n"
    )
    (VIZ_DIR / "datasets.js").write_text(js_content)
    (VIZ_DIR / "plots_data.js").write_text(js_content)
    return js_content


def generate_all(target_figure: Optional[str] = None) -> List[Dict[str, Any]]:
    by_key = load_all_enriched_data()
    psweep_records = load_psweep_records(by_key)
    out_dirs = [MULTIPASS_PLOTS_DIR, TMP_FIGURES_DIR, TMP_PAPER_BUILD_FIGURES_DIR, PAPER_FIGURES_DIR]
    provenance_entries: List[Dict[str, Any]] = []

    def should_run(stem: str) -> bool:
        return target_figure is None or target_figure == stem

    # 1. Color Codes d=7, 9 tradeoff (plus d=5,7,9,11)
    stem1 = "tradeoff__code=cc__d=7_9__dec=B_G_M_O__x=log"
    if should_run(stem1):
        provenance_entries.append(
            plot_tradeoff_family(
                by_key,
                ["cc_d7", "cc_d9"],
                "Superdense Color Code Speed–Accuracy Tradeoffs (d=7, 9)\n"
                "(r=d, p=0.001, noise=si1000, c=superdense_color_code_Z, gates=cz.stim)",
                stem1,
                out_dirs,
                legacy_names=[
                    "tradeoff__code=cc__d=7_9__dec=B_G_M_C_O__x=log.png",
                    "tradeoff__code=cc__d=7_9__dec=B_G_M_C_O__x=log.pdf",
                ],
            )
        )

    stem1b = "tradeoff__code=cc__d=5_7_9_11__dec=B_G_M_O__x=log"
    if should_run(stem1b):
        provenance_entries.append(
            plot_tradeoff_family(
                by_key,
                ["cc_d5", "cc_d7", "cc_d9", "cc_d11"],
                "Superdense Color Code Speed–Accuracy Tradeoffs (d=5, 7, 9, 11)\n"
                "(r=d, p=0.001, noise=si1000, c=superdense_color_code_Z, gates=cz.stim)",
                stem1b,
                out_dirs,
                legacy_names=[
                    "cc_baseline_gari_multipass_tradeoffs.png",
                    "tradeoff__code=cc__d=5_7_9_11__dec=B_G_M_C_O__x=log.png",
                    "tradeoff__code=cc__d=5_7_9_11__dec=B_G_M_C_O__x=log.pdf",
                ],
            )
        )

    # 2. Bivariate Bicycle [[72, 12, 6]] (d=6) and Gross [[144, 12, 12]] (d=12) combined & individual
    stem2 = "tradeoff__code=bb__d=6_12_gross__dec=B_G_M_O__x=log"
    if should_run(stem2):
        provenance_entries.append(
            plot_tradeoff_family(
                by_key,
                ["bb_d6", "bb_d12_gross"],
                "Bivariate Bicycle Speed–Accuracy Tradeoffs: [[72, 12, 6]] (d=6) & Gross [[144, 12, 12]] (d=12)\n"
                "(r=d, p=0.001, noise=si1000, c=bivariate_bicycle_Z, A=x^3+y+y^2, B=y^3+x+x^2)",
                stem2,
                out_dirs,
                legacy_names=[
                    "tradeoff__code=bb__d=6_12_gross__dec=B_G_M_C_O__x=log.png",
                    "tradeoff__code=bb__d=6_12_gross__dec=B_G_M_C_O__x=log.pdf",
                ],
            )
        )

    stem2b = "tradeoff__code=bb_d6__dec=B_G_M_O__x=log"
    if should_run(stem2b):
        provenance_entries.append(
            plot_tradeoff_family(
                by_key,
                ["bb_d6"],
                "Bivariate Bicycle Speed–Accuracy Tradeoffs: [[72, 12, 6]] (d=6, q=144)\n"
                "(r=6, d=6, p=0.001, noise=si1000, c=bivariate_bicycle_Z, [[72, 12, 6]])",
                stem2b,
                out_dirs,
                legacy_names=[
                    "bb_baseline_gari_multipass_tradeoffs.png",
                    "tradeoff__code=bb_d6__dec=B_G_M_C_O__x=log.png",
                    "tradeoff__code=bb_d6__dec=B_G_M_C_O__x=log.pdf",
                ],
            )
        )

    stem2c = "tradeoff__code=gross_d12__dec=B_G_M_O__x=log"
    if should_run(stem2c):
        provenance_entries.append(
            plot_tradeoff_family(
                by_key,
                ["bb_d12_gross"],
                "Gross Code Speed–Accuracy Tradeoffs: [[144, 12, 12]] (d=12, q=288)\n"
                "(r=12, d=12, p=0.001, noise=si1000, c=bivariate_bicycle_Z, [[144, 12, 12]])",
                stem2c,
                out_dirs,
                legacy_names=[
                    "tradeoff__code=gross_d12__dec=B_G_M_C_O__x=log.png",
                    "tradeoff__code=gross_d12__dec=B_G_M_C_O__x=log.pdf",
                ],
            )
        )

    # 3. Mitten [[150, 30, 10]] (d=10)
    stem3 = "tradeoff__code=mitten__d=10__dec=B_G_M_O__x=log"
    if should_run(stem3):
        provenance_entries.append(
            plot_tradeoff_family(
                by_key,
                ["mitten_d10"],
                "Mitten Code Speed–Accuracy Tradeoffs: [[150, 30, 10]] (d=10, q=270)\n"
                "(r=10, d=10, p=0.001, noise=xyz_hookfree, c=mitten_X, [[150, 30, 10]])",
                stem3,
                out_dirs,
                include_external=True,
                legacy_names=[
                    "mitten_baseline_gari_multipass_tradeoffs.png",
                    "tradeoff__code=mitten__d=10__dec=B_G_M_C_O__x=log.png",
                    "tradeoff__code=mitten__d=10__dec=B_G_M_C_O__x=log.pdf",
                ],
            )
        )

    # 4. Hyperbolic Octagon Block LER
    stem4 = "hyperbolic_ler__iters=32_40_58_109__dec=AQ_BPOSD__norm=block"
    if should_run(stem4):
        provenance_entries.append(
            plot_hyperbolic_ler(
                by_key,
                "block",
                stem4,
                out_dirs,
                legacy_names=[
                    "hyperbolic_multipass_slow_comparison.png",
                    "hyperbolic_ler__iters=32_40_58_109__dec=AQ_BPOSD_" + "Cascade__norm=block.png",
                    "hyperbolic_ler__iters=32_40_58_109__dec=AQ_BPOSD_" + "Cascade__norm=block.pdf",
                ],
            )
        )

    # 5. Hyperbolic Octagon Per-Logical LER
    stem5 = "hyperbolic_ler__iters=32_40_58_109__dec=AQ_BPOSD__norm=per_logical"
    if should_run(stem5):
        provenance_entries.append(
            plot_hyperbolic_ler(
                by_key,
                "per_logical",
                stem5,
                out_dirs,
                legacy_names=[
                    "hyperbolic_multipass_per_logical.png",
                    "hyperbolic_ler__iters=32_40_58_109__dec=AQ_BPOSD_" + "Cascade__norm=per_logical.png",
                    "hyperbolic_ler__iters=32_40_58_109__dec=AQ_BPOSD_" + "Cascade__norm=per_logical.pdf",
                ],
            )
        )

    # 6. Stage Breakdown across Color Code distances d=5..23
    stem6 = "stage_breakdown__code=cc__d=5_7_9_11__dec=BPOSD"
    if should_run(stem6):
        provenance_entries.append(
            plot_stage_breakdown(
                by_key,
                ["cc_d5", "cc_d7", "cc_d9", "cc_d11", "cc_d13", "cc_d15", "cc_d17", "cc_d19", "cc_d21", "cc_d23"],
                "Tesseract-BP-OSD Shot Resolution & CPU Compute Share Across Color Code Distances (d=5..23)",
                stem6,
                out_dirs,
            )
        )

    # 7. Stage Breakdown across all code families
    stem7 = "stage_breakdown__codes=all_families__dec=BPOSD"
    if should_run(stem7):
        provenance_entries.append(
            plot_stage_breakdown(
                by_key,
                [
                    "cc_d5",
                    "cc_d7",
                    "cc_d9",
                    "cc_d11",
                    "bb_d6",
                    "bb_d12_gross",
                    "mitten_d10",
                    "hyp_iter32",
                    "hyp_iter40",
                    "hyp_iter58",
                    "hyp_iter109",
                ],
                "Tesseract-BP-OSD Cross-Family Shot & CPU Compute Breakdown (Color, BB, Gross, Mitten & Hyperbolic)",
                stem7,
                out_dirs,
            )
        )

    # 8. Architecture & Shot-Flow Diagram
    stem8 = "arch_flow__decoder=Tesseract_BPOSD"
    if should_run(stem8):
        provenance_entries.append(plot_architecture_flow(by_key, stem8, out_dirs))

    # 9. DEM Structural Transformations Diagram ([B] vs [G] vs [M]/[O])
    stem9 = "dem_transforms__dec=B_G_M"
    if should_run(stem9):
        provenance_entries.append(plot_dem_transforms(by_key, stem9, out_dirs))

    # 10. Color Code Distance Scaling (d=3..23)
    stem10 = "cc_scaling__code=cc__d=3_to_23__dec=all"
    if should_run(stem10):
        provenance_entries.append(
            plot_cc_scaling(
                by_key,
                stem10,
                out_dirs,
                legacy_names=["scaling__code=cc__p=0.001__dec=B_G_M_C_O_T.png"],
            )
        )

    # 11. Bivariate Bicycle [[72, 12, 6]] p-Sweep
    stem11 = "bb_psweep__code=bb_d6__dec=all"
    if should_run(stem11):
        provenance_entries.append(
            plot_bb_psweep(
                psweep_records,
                "bb_d6",
                "Bivariate Bicycle [[72, 12, 6]] (d = 6, r = 6): LER & CPU Time vs. Physical Noise p",
                stem11,
                out_dirs,
            )
        )

    # 12. Gross Code [[144, 12, 12]] p-Sweep
    stem12 = "bb_psweep__code=bb_d12_gross__dec=all"
    if should_run(stem12):
        provenance_entries.append(
            plot_bb_psweep(
                psweep_records,
                "bb_d12_gross",
                "Gross Code [[144, 12, 12]] (d = 12, r = 12): LER & CPU Time vs. Physical Noise p",
                stem12,
                out_dirs,
            )
        )

    # 13. 4-Circuit Cumulative Ablation Suite (fig_ablation_{bp_comp,bp_algo,osd_comp,osd_algo})
    ablation_dirs = out_dirs + [
        pathlib.Path("/usr/local/google/home/ariash/.gemini/jetski/brain/67c42a3b-49c1-44cb-a2ce-ec1851dd8f89"),
    ]
    provenance_entries.extend(plot_ablation_4circuits_suite(ablation_dirs, target_figure))

    if target_figure is None:
        write_provenance_tex(provenance_entries, out_dirs)
        write_datasets_js(by_key, psweep_records, provenance_entries)
    print(f"Exported {len(provenance_entries)} figure suites and benchmarking/viz/datasets.js (commit {GIT_COMMIT}).")
    return provenance_entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic static figure & dataset exporter.")
    parser.add_argument("--figure", type=str, default=None, help="Optional single figure stem to generate.")
    args = parser.parse_args()
    generate_all(target_figure=args.figure)


if __name__ == "__main__":
    main()
