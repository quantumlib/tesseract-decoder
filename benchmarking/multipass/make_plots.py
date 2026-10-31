#!/usr/bin/env python3
"""Generate tradeoff plots for Baseline -> GARI -> Multi-Pass decoding."""

import glob
import json
import math
import pathlib
import sys

RF = "bazel-bin/benchmarking/sparsify_errors/plot.runfiles"
sys.path[:0] = (
    [f"{RF}/_main/src", f"{RF}/_main/src/py", "bazel-bin/src", "src/py"]
    + glob.glob(f"{RF}/*/site-packages")
    + glob.glob("venv/lib/python*/site-packages")
)

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from scipy.stats import norm

SCRATCH = pathlib.Path(__file__).resolve().parent
ARTIFACT_DIR = SCRATCH / "plots"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def wilson_interval(failures: int, shots: int, confidence: float = 0.95):
    z = float(norm.ppf(0.5 + confidence / 2.0))
    n = float(shots)
    p = failures / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def shot_to_round_ler(p_shot: float, rounds: int) -> float:
    if p_shot <= 0.0:
        return 0.0
    if p_shot >= 0.5:
        return 0.5
    return 0.5 * (1.0 - (1.0 - 2.0 * p_shot) ** (1.0 / rounds))


def relative_risk_ci(f0: int, n0: int, f1: int, n1: int, r: int):
    """95% CI on per-round LER ratio (after / before) with Haldane correction."""
    a0, b0 = f0 + 0.5, n0 + 1.0
    a1, b1 = f1 + 0.5, n1 + 1.0
    p0 = a0 / b0
    p1 = a1 / b1
    ler0 = shot_to_round_ler(p0, r)
    ler1 = shot_to_round_ler(p1, r)
    ratio = ler1 / ler0
    se_log = math.sqrt((1.0 / a1 - 1.0 / b1) + (1.0 / a0 - 1.0 / b0))
    lo = ratio * math.exp(-1.95996 * se_log)
    hi = ratio * math.exp(+1.95996 * se_log)
    return lo, hi


def load_enriched_results():
    raw = json.loads((SCRATCH / "results.json").read_text())
    by_key = {}
    for row in raw:
        key = (row["tag"], row["mode"])
        f_stat = row["num_errors"] + row["num_low_confidence"]
        n_stat = row["num_shots"]
        r = row["r"]
        p_shot = f_stat / n_stat if f_stat > 0 else 0.5 / (n_stat + 1.0)
        p_lo, p_hi = wilson_interval(f_stat, n_stat)
        ler = shot_to_round_ler(p_shot, r)
        ler_lo = shot_to_round_ler(p_lo, r)
        ler_hi = shot_to_round_ler(p_hi, r)
        enriched = dict(row)
        enriched.update(
            {
                "eff_failures": f_stat,
                "eff_shots": n_stat,
                "ler": ler,
                "ler_lo": ler_lo,
                "ler_hi": ler_hi,
                "ler_err_minus": max(ler - ler_lo, ler * 0.02),
                "ler_err_plus": max(ler_hi - ler, ler * 0.02),
            }
        )
        by_key[key] = enriched
    return by_key


def style_axes(ax, title: str):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(
        "Time per round (seconds)", fontsize=12, fontstyle="italic", color="#333333"
    )
    ax.set_ylabel(
        "Logical Error Rate per round",
        fontsize=12,
        fontstyle="italic",
        color="#333333",
    )
    ax.set_title(
        title, loc="left", fontsize=12.5, fontweight="bold", color="#333333", pad=10
    )
    ax.grid(True, which="major", color="#dcdcdc", linewidth=0.9)
    ax.grid(True, which="minor", color="#f0f0f0", linewidth=0.5, linestyle=":")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_color("#dcdcdc")
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#222222")
        ax.spines[spine].set_linewidth(1.1)


def draw_point(
    ax,
    pt,
    marker,
    fill_color,
    ring_color,
    letter: str,
    size=110,
    ring_size=260,
    badge_offset=(8, 8),
    zorder=5,
):
    x = pt["time_per_round"]
    y = pt["ler"]
    ax.errorbar(
        [x],
        [y],
        yerr=[[pt["ler_err_minus"]], [pt["ler_err_plus"]]],
        fmt="none",
        ecolor=fill_color if fill_color != "#f5c242" else "#d99b00",
        elinewidth=1.4,
        alpha=0.85,
        zorder=zorder - 1,
    )
    ax.scatter(
        [x],
        [y],
        marker=marker,
        s=ring_size,
        facecolors="none",
        edgecolors=ring_color,
        linewidths=2.6,
        zorder=zorder + 1,
    )
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
    ax.annotate(
        letter,
        xy=(x, y),
        xytext=badge_offset,
        textcoords="offset points",
        fontsize=7.8,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="circle,pad=0.16",
            fc=ring_color,
            ec="white",
            lw=0.9,
        ),
        zorder=zorder + 3,
    )


def draw_arrow_segment(
    ax, pt_from, pt_to, color="#555555", lw=1.8, linestyle="-", alpha=0.88
):
    x0, y0 = pt_from["time_per_round"], pt_from["ler"]
    x1, y1 = pt_to["time_per_round"], pt_to["ler"]
    ax.plot(
        [x0, x1],
        [y0, y1],
        linestyle=linestyle,
        linewidth=lw,
        color=color,
        alpha=alpha,
        zorder=3,
    )
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=max(1.0, lw - 0.1),
            ls=linestyle,
            alpha=alpha,
            shrinkA=10,
            shrinkB=10,
            mutation_scale=11,
        ),
        zorder=3,
    )


def plot_color_codes(by_key):
    fig, ax = plt.subplots(figsize=(12.5, 8.5), dpi=200)
    style_axes(
        ax,
        "Logical Error Rate per Round vs Time per Round\n"
        "(r=d, p=0.001, noise=si1000, c=superdense_color_code_Z, gates=cz.stim)",
    )

    cc_configs = [
        ("cc_d3", "cc, d=3", "o", "#f5c242"),
        ("cc_d5", "cc, d=5", "^", "#f5c242"),
        ("cc_d7", "cc, d=7", "*", "#f5c242"),
        ("cc_d9", "cc, d=9", "D", "#f5c242"),
    ]

    for tag, label, marker, fill_color in cc_configs:
        b_pt = by_key[(tag, "baseline_b20")]
        g_pt = by_key[(tag, "gari_b5")]
        m20_pt = by_key[(tag, "mp_2p_b20")]
        bp_pt = by_key[(tag, "mp_bp_osd")]
        cas_pt = by_key.get((tag, "mp_bp_osd_cascade"))
        rel_pt = by_key.get((tag, "relay_bp"))
        tel_pt = by_key.get((tag, "telescoping_bp"))

        # Draw points with uniform single ring + short letter badge
        g_offset = (-8, 8) if tag == "cc_d3" else (8, 8)
        draw_point(ax, b_pt, marker, fill_color, "#111111", "B", size=110, ring_size=260)
        draw_point(
            ax, g_pt, marker, fill_color, "#108000", "G", size=110, ring_size=260, badge_offset=g_offset
        )
        draw_point(ax, m20_pt, marker, fill_color, "#0055cc", "M", size=110, ring_size=260)
        draw_point(ax, bp_pt, marker, fill_color, "#008080", "O", size=110, ring_size=260)
        if cas_pt is not None:
            draw_point(ax, cas_pt, marker, fill_color, "#c9184a", "C", size=110, ring_size=260, badge_offset=(-8, -8))
        if rel_pt is not None:
            draw_point(ax, rel_pt, marker, fill_color, "#e66100", "R", size=110, ring_size=260)
        if tel_pt is not None:
            draw_point(ax, tel_pt, marker, fill_color, "#7b3f00", "T", size=110, ring_size=260)

        # Solid main progression: Baseline -> GARI -> MP(b=20) -> 3-Stage Cascade -> MP-BP-OSD
        draw_arrow_segment(ax, b_pt, g_pt, color="#444444", lw=1.8, linestyle="-")
        draw_arrow_segment(ax, g_pt, m20_pt, color="#0055cc", lw=1.8, linestyle="-")
        if cas_pt is not None:
            draw_arrow_segment(ax, m20_pt, cas_pt, color="#c9184a", lw=1.8, linestyle="-")
            draw_arrow_segment(ax, cas_pt, bp_pt, color="#008080", lw=1.8, linestyle="-")
        else:
            draw_arrow_segment(ax, m20_pt, bp_pt, color="#008080", lw=1.8, linestyle="-")

        # Secondary dashed progression: Baseline -> Relay-BP -> Telescoping-BP
        if rel_pt is not None and tel_pt is not None:
            draw_arrow_segment(
                ax, b_pt, rel_pt, color="#e66100", lw=1.35, linestyle="--", alpha=0.75
            )
            draw_arrow_segment(
                ax, rel_pt, tel_pt, color="#7b3f00", lw=1.35, linestyle="--", alpha=0.75
            )

    ax.set_xlim(1.0e-6, 4.0e-1)
    ax.set_ylim(1.3e-6, 4.8e-2)

    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#f5c242",
            markeredgecolor="#b58900",
            markersize=9,
            label="cc, d=3",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="^",
            markerfacecolor="#f5c242",
            markeredgecolor="#b58900",
            markersize=9,
            label="cc, d=5",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="*",
            markerfacecolor="#f5c242",
            markeredgecolor="#b58900",
            markersize=11,
            label="cc, d=7",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="D",
            markerfacecolor="#f5c242",
            markeredgecolor="#b58900",
            markersize=8.5,
            label="cc, d=9",
        ),
        mlines.Line2D([], [], color="none", label=""),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#cccccc",
            markeredgecolor="#111111",
            markeredgewidth=2.6,
            markersize=10,
            label="[B] Baseline (beam=20, climb=True, no_revisit=True, orders=1)",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#cccccc",
            markeredgecolor="#108000",
            markeredgewidth=2.6,
            markersize=10,
            label="[G] GARI (prior=xor, beam=5, climb=True, orders=1)",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#cccccc",
            markeredgecolor="#0055cc",
            markeredgewidth=2.6,
            markersize=10,
            label="[M] Multi-Pass Tesseract (2-pass causal, beam=20)",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#cccccc",
            markeredgecolor="#c9184a",
            markeredgewidth=2.6,
            markersize=10,
            label="[C] 3-Stage Cascade (MP-BP-OSD -> Monolithic BP-OSD -> Tesseract)",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#cccccc",
            markeredgecolor="#008080",
            markeredgewidth=2.6,
            markersize=10,
            label="[O] Multi-Pass BP-OSD (2-pass causal, max_iter=20, OSD-2)",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#cccccc",
            markeredgecolor="#e66100",
            markeredgewidth=2.6,
            markersize=10,
            label="[R] Relay-BP (External: trmue/relay, S=5)",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#cccccc",
            markeredgecolor="#7b3f00",
            markeredgewidth=2.6,
            markersize=10,
            label="[T] Telescoping-BP (External: a7b/yarn, GARI S3)",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8.8,
        frameon=True,
        facecolor="white",
        edgecolor="#dddddd",
        borderpad=0.8,
    )

    plt.tight_layout()
    out_path = ARTIFACT_DIR / "cc_baseline_gari_multipass_tradeoffs.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_bb_code(by_key):
    fig, ax = plt.subplots(figsize=(11.5, 7.8), dpi=200)
    style_axes(
        ax,
        "Logical Error Rate per Round vs Time per Round\n"
        "(r=6, d=6, p=0.001, noise=si1000, c=bivariate_bicycle_Z, [[144, 12, 6]])",
    )

    tag = "bb_d6"
    marker = "o"
    fill_color = "#ff00ff"
    b_pt = by_key[(tag, "baseline_b20")]
    g_pt = by_key[(tag, "gari_b5")]
    m20_pt = by_key[(tag, "mp_2p_b20")]
    bp_pt = by_key[(tag, "mp_bp_osd")]
    cas_pt = by_key.get((tag, "mp_bp_osd_cascade"))
    rel_pt = by_key.get((tag, "relay_bp"))
    tel_pt = by_key.get((tag, "telescoping_bp"))

    draw_point(ax, b_pt, marker, fill_color, "#111111", "B", size=140, ring_size=320)
    draw_point(ax, g_pt, marker, fill_color, "#108000", "G", size=140, ring_size=320)
    draw_point(ax, m20_pt, marker, fill_color, "#0055cc", "M", size=140, ring_size=320)
    draw_point(ax, bp_pt, marker, fill_color, "#008080", "O", size=140, ring_size=320)
    if cas_pt is not None:
        draw_point(ax, cas_pt, marker, fill_color, "#c9184a", "C", size=140, ring_size=320, badge_offset=(-8, -8))
    if rel_pt is not None:
        draw_point(ax, rel_pt, marker, fill_color, "#e66100", "R", size=140, ring_size=320)
    if tel_pt is not None:
        draw_point(ax, tel_pt, marker, fill_color, "#7b3f00", "T", size=140, ring_size=320)

    # Solid main progression: Baseline -> GARI -> MP(b=20) -> 3-Stage Cascade -> MP-BP-OSD
    draw_arrow_segment(ax, b_pt, g_pt, color="#444444", lw=1.8, linestyle="-")
    draw_arrow_segment(ax, g_pt, m20_pt, color="#0055cc", lw=1.8, linestyle="-")
    if cas_pt is not None:
        draw_arrow_segment(ax, m20_pt, cas_pt, color="#c9184a", lw=1.8, linestyle="-")
        draw_arrow_segment(ax, cas_pt, bp_pt, color="#008080", lw=1.8, linestyle="-")
    else:
        draw_arrow_segment(ax, m20_pt, bp_pt, color="#008080", lw=1.8, linestyle="-")

    # Secondary dashed progression: Baseline -> Relay-BP -> Telescoping-BP
    if rel_pt is not None and tel_pt is not None:
        draw_arrow_segment(
            ax, b_pt, rel_pt, color="#e66100", lw=1.35, linestyle="--", alpha=0.75
        )
        draw_arrow_segment(
            ax, rel_pt, tel_pt, color="#7b3f00", lw=1.35, linestyle="--", alpha=0.75
        )

    ax.set_xlim(2.0e-5, 1.0e0)
    ax.set_ylim(1.0e-5, 1.5e-3)

    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#ff00ff",
            markeredgecolor="#aa00aa",
            markersize=10,
            label="bb, d=6, q=144",
        ),
        mlines.Line2D([], [], color="none", label=""),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#cccccc",
            markeredgecolor="#111111",
            markeredgewidth=2.6,
            markersize=10,
            label="[B] Baseline (beam=20, climb=True, no_revisit=True, orders=1)",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#cccccc",
            markeredgecolor="#108000",
            markeredgewidth=2.6,
            markersize=10,
            label="[G] GARI (prior=xor, beam=5, climb=True, orders=1)",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#cccccc",
            markeredgecolor="#0055cc",
            markeredgewidth=2.6,
            markersize=10,
            label="[M] Multi-Pass Tesseract (2-pass causal, beam=20)",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#cccccc",
            markeredgecolor="#c9184a",
            markeredgewidth=2.6,
            markersize=10,
            label="[C] 3-Stage Cascade (MP-BP-OSD -> Monolithic BP-OSD -> Tesseract)",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#cccccc",
            markeredgecolor="#008080",
            markeredgewidth=2.6,
            markersize=10,
            label="[O] Multi-Pass BP-OSD (2-pass causal, max_iter=20, OSD-2)",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#cccccc",
            markeredgecolor="#e66100",
            markeredgewidth=2.6,
            markersize=10,
            label="[R] Relay-BP (External: trmue/relay, S=5)",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#cccccc",
            markeredgecolor="#7b3f00",
            markeredgewidth=2.6,
            markersize=10,
            label="[T] Telescoping-BP (External: a7b/yarn, GARI S3)",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8.8,
        frameon=True,
        facecolor="white",
        edgecolor="#dddddd",
        borderpad=0.8,
    )

    plt.tight_layout()
    out_path = ARTIFACT_DIR / "bb_baseline_gari_multipass_tradeoffs.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_mitten_code(by_key):
    tag = "mitten_d10"
    if (tag, "baseline_b20") not in by_key:
        return

    fig, ax = plt.subplots(figsize=(11.5, 7.8), dpi=200)
    style_axes(
        ax,
        "Logical Error Rate per Round vs Time per Round\n"
        "(r=10, d=10, p=0.001, noise=xyz_hookfree, c=mitten_X, [[150, 30, 10]], q=270)",
    )

    marker = "o"
    fill_color = "#38bdf8"
    b_pt = by_key[(tag, "baseline_b20")]
    g_pt = by_key[(tag, "gari_b5")]
    m20_pt = by_key[(tag, "mp_2p_b20")]
    bp_pt = by_key[(tag, "mp_bp_osd")]
    cas_pt = by_key.get((tag, "mp_bp_osd_cascade"))
    rel_pt = by_key.get((tag, "relay_bp"))
    tel_pt = by_key.get((tag, "telescoping_bp"))

    draw_point(ax, b_pt, marker, fill_color, "#111111", "B", size=125, ring_size=300)
    draw_point(ax, g_pt, marker, fill_color, "#108000", "G", size=125, ring_size=300)
    draw_point(ax, m20_pt, marker, fill_color, "#0055cc", "M", size=125, ring_size=300)
    draw_point(ax, bp_pt, marker, fill_color, "#008080", "O", size=125, ring_size=300)
    if cas_pt is not None:
        draw_point(ax, cas_pt, marker, fill_color, "#c9184a", "C", size=125, ring_size=300, badge_offset=(-8, -8))
    if rel_pt is not None:
        draw_point(ax, rel_pt, marker, fill_color, "#e66100", "R", size=125, ring_size=300)
    if tel_pt is not None:
        draw_point(ax, tel_pt, marker, fill_color, "#7b3f00", "T", size=125, ring_size=300)

    # Solid main progression: Baseline -> GARI -> MP(b=20) -> 3-Stage Cascade -> MP-BP-OSD
    draw_arrow_segment(ax, b_pt, g_pt, color="#444444", lw=1.8, linestyle="-")
    draw_arrow_segment(ax, g_pt, m20_pt, color="#0055cc", lw=1.8, linestyle="-")
    if cas_pt is not None:
        draw_arrow_segment(ax, m20_pt, cas_pt, color="#c9184a", lw=1.8, linestyle="-")
        draw_arrow_segment(ax, cas_pt, bp_pt, color="#008080", lw=1.8, linestyle="-")
    else:
        draw_arrow_segment(ax, m20_pt, bp_pt, color="#008080", lw=1.8, linestyle="-")

    # Secondary dashed progression: Baseline -> Relay-BP -> Telescoping-BP
    if rel_pt is not None and tel_pt is not None:
        draw_arrow_segment(
            ax, b_pt, rel_pt, color="#e66100", lw=1.35, linestyle="--", alpha=0.75
        )
        draw_arrow_segment(
            ax, rel_pt, tel_pt, color="#7b3f00", lw=1.35, linestyle="--", alpha=0.75
        )

    ax.set_xlim(1.0e-4, 2.0e1)
    ax.set_ylim(1.2e-7, 2.2e-2)

    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#38bdf8",
            markeredgecolor="#0284c7",
            markersize=9.5,
            label="mitten, d=10, q=270, [[150, 30, 10]]",
        ),
        mlines.Line2D([], [], color="none", label=""),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#cccccc",
            markeredgecolor="#111111",
            markeredgewidth=2.6,
            markersize=10,
            label="[B] Baseline (beam=20, climb=True, no_revisit=True, orders=1)",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#cccccc",
            markeredgecolor="#108000",
            markeredgewidth=2.6,
            markersize=10,
            label="[G] GARI (prior=xor, beam=5, climb=True, orders=1)",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#cccccc",
            markeredgecolor="#0055cc",
            markeredgewidth=2.6,
            markersize=10,
            label="[M] Multi-Pass Tesseract (2-pass causal, beam=20)",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#cccccc",
            markeredgecolor="#c9184a",
            markeredgewidth=2.6,
            markersize=10,
            label="[C] 3-Stage Cascade (MP-BP-OSD -> Random-Sched BP-OSD -> MP-Tesseract)",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#cccccc",
            markeredgecolor="#008080",
            markeredgewidth=2.6,
            markersize=10,
            label="[O] Multi-Pass BP-OSD (2-pass causal, max_iter=20, OSD-2)",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#cccccc",
            markeredgecolor="#e66100",
            markeredgewidth=2.6,
            markersize=10,
            label="[R] Relay-BP (External: trmue/relay, S=5)",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#cccccc",
            markeredgecolor="#7b3f00",
            markeredgewidth=2.6,
            markersize=10,
            label="[T] Telescoping-BP (External: a7b/yarn, GARI S3)",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=8.6,
        frameon=True,
        facecolor="white",
        edgecolor="#dddddd",
        borderpad=0.75,
    )

    plt.tight_layout()
    out_path = ARTIFACT_DIR / "mitten_baseline_gari_multipass_tradeoffs.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    by_key = load_enriched_results()
    plot_color_codes(by_key)
    plot_bb_code(by_key)
    plot_mitten_code(by_key)


if __name__ == "__main__":
    main()
