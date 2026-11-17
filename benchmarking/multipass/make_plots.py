#!/usr/bin/env python3
"""Generate tradeoff plots for Baseline -> GARI -> Multi-Pass decoding."""

import glob
import json
import math
import pathlib
import sys

RF = "bazel-bin/benchmarking/sparsify_errors/plot.runfiles"
sys.path[:0] = [f"{RF}/_main/src", f"{RF}/_main/src/py"] + glob.glob(
    f"{RF}/*/site-packages"
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
    ax, pt, marker, fill_color, ring_color, size=110, ring_size=260, zorder=5
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


def draw_arrow_segment(ax, pt_from, pt_to, color="#555555", lw=1.4):
    x0, y0 = pt_from["time_per_round"], pt_from["ler"]
    x1, y1 = pt_to["time_per_round"], pt_to["ler"]
    ax.plot(
        [x0, x1],
        [y0, y1],
        linestyle="--",
        linewidth=lw,
        color=color,
        alpha=0.85,
        zorder=3,
    )
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw - 0.1,
            ls="--",
            shrinkA=10,
            shrinkB=10,
            mutation_scale=11,
        ),
        zorder=3,
    )


def annotate_segment_box(
    ax,
    pt_from,
    pt_to,
    x_factor=1.0,
    y_offset_factor=1.18,
    color="#111111",
    label_prefix="",
):
    x0, y0 = pt_from["time_per_round"], pt_from["ler"]
    x1, y1 = pt_to["time_per_round"], pt_to["ler"]
    mid_x = math.exp(0.5 * (math.log(x0) + math.log(x1))) * x_factor
    mid_y = math.exp(0.5 * (math.log(y0) + math.log(y1)))
    spd = x0 / x1
    lo, hi = relative_risk_ci(
        pt_from["eff_failures"],
        pt_from["eff_shots"],
        pt_to["eff_failures"],
        pt_to["eff_shots"],
        pt_from["r"],
    )
    prefix = f"{label_prefix}\n" if label_prefix else ""
    text = f"{prefix}{spd:.1f}x spd\n{lo:.2f}-{hi:.2f}x err"
    ax.text(
        mid_x,
        mid_y * y_offset_factor,
        text,
        fontsize=8.2,
        color=color,
        ha="center",
        va="bottom" if y_offset_factor >= 1.0 else "top",
        bbox=dict(
            boxstyle="round,pad=0.20", fc="white", ec="#cccccc", lw=0.6, alpha=0.90
        ),
        zorder=8,
    )


def plot_color_codes(by_key):
    fig, ax = plt.subplots(figsize=(13.0, 9.0), dpi=200)
    style_axes(
        ax,
        "Logical Error Rate per Round vs Time per Round\n"
        "(r=d, p=0.001, noise=si1000, c=superdense_color_code_Z, gates=cz.stim)",
    )

    # Per-distance customized label offsets to ensure zero overlap
    cc_configs = [
        # (tag, label, marker, fill, gari_x, gari_y, mp_x, mp_y, badge_x, badge_y)
        ("cc_d3", "cc, d=3", "o", "#f5c242", 1.0, 1.18, 0.85, 0.72, 0.68, 1.0),
        ("cc_d5", "cc, d=5", "^", "#f5c242", 1.0, 1.18, 1.0, 0.68, 0.68, 1.05),
        ("cc_d7", "cc, d=7", "*", "#f5c242", 1.0, 1.18, 1.45, 0.56, 0.68, 1.12),
        ("cc_d9", "cc, d=9", "D", "#f5c242", 1.35, 0.70, 1.95, 0.48, 0.68, 0.95),
    ]

    for (
        tag,
        label,
        marker,
        fill_color,
        gx,
        gy,
        mx,
        my,
        bx,
        by,
    ) in cc_configs:
        b_pt = by_key[(tag, "baseline_b20")]
        g_pt = by_key[(tag, "gari_b5")]
        m20_pt = by_key[(tag, "mp_2p_b20")]
        m5_pt = by_key[(tag, "mp_2p_b5")]

        # Draw points: Baseline -> GARI -> Multi-Pass (b=20) -> Multi-Pass (b=5)
        draw_point(ax, b_pt, marker, fill_color, "#111111", size=110, ring_size=240)
        draw_point(ax, g_pt, marker, fill_color, "#108000", size=110, ring_size=260)
        draw_point(ax, m20_pt, marker, fill_color, "#0055cc", size=95, ring_size=220)
        draw_point(ax, m5_pt, marker, fill_color, "#d62728", size=110, ring_size=260)

        # Connect progression: Baseline -> GARI -> MP(b=20) -> MP(b=5)
        draw_arrow_segment(ax, b_pt, g_pt, color="#444444")
        draw_arrow_segment(ax, g_pt, m20_pt, color="#0055cc")
        draw_arrow_segment(ax, m20_pt, m5_pt, color="#d62728")

        # Annotate Base -> GARI and GARI -> MP(b=5)
        annotate_segment_box(
            ax,
            b_pt,
            g_pt,
            x_factor=gx,
            y_offset_factor=gy,
            color="#106000",
            label_prefix="Base→GARI",
        )
        annotate_segment_box(
            ax,
            g_pt,
            m5_pt,
            x_factor=mx,
            y_offset_factor=my,
            color="#b01518",
            label_prefix="GARI→MP(b=5)",
        )

        # Total speedup badge near Multi-Pass (b=5)
        tot_spd = b_pt["time_per_round"] / m5_pt["time_per_round"]
        tot_lo, tot_hi = relative_risk_ci(
            b_pt["eff_failures"],
            b_pt["eff_shots"],
            m5_pt["eff_failures"],
            m5_pt["eff_shots"],
            b_pt["r"],
        )
        spd_b20 = b_pt["time_per_round"] / m20_pt["time_per_round"]
        ax.text(
            m5_pt["time_per_round"] * bx,
            m5_pt["ler"] * by,
            f"[{label}]\nBase→MP(b=5): {tot_spd:.0f}x spd ({tot_lo:.2f}-{tot_hi:.2f}x err)\nBase→MP(b=20): {spd_b20:.0f}x spd",
            fontsize=7.8,
            fontweight="bold",
            color="#8c1014",
            ha="right",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.24",
                fc="#fff8f8",
                ec="#e0b0b0",
                lw=0.8,
                alpha=0.93,
            ),
            zorder=9,
        )

    ax.set_xlim(1.2e-7, 2.5e-1)
    ax.set_ylim(1.8e-6, 1.5e-2)

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
            markeredgewidth=2.4,
            markersize=10,
            label="Baseline (beam=20, climb=True, no_revisit=True, orders=1)",
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
            label="GARI (prior=xor, beam=5, climb=True, orders=1)",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#cccccc",
            markeredgecolor="#0055cc",
            markeredgewidth=2.4,
            markersize=10,
            label="Multi-Pass (2-pass causal, beam=20, climb=True, orders=1)",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#cccccc",
            markeredgecolor="#d62728",
            markeredgewidth=2.6,
            markersize=10,
            label="Multi-Pass (2-pass causal, beam=5, climb=True, orders=1)",
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
    fig, ax = plt.subplots(figsize=(11.5, 8.0), dpi=200)
    style_axes(
        ax,
        "Logical Error Rate per Round vs Time per Round\n"
        "(r=6, d=6, p=0.001, noise=si1000, c=bivariate_bicycle_Z, [[144, 12, 6]])",
    )

    tag = "bb_d6"
    marker = "H"
    fill_color = "#ff00ff"
    b_pt = by_key[(tag, "baseline_b20")]
    g_pt = by_key[(tag, "gari_b5")]
    m20_pt = by_key[(tag, "mp_2p_b20")]
    m5_pt = by_key[(tag, "mp_2p_b5")]

    draw_point(ax, b_pt, marker, fill_color, "#111111", size=140, ring_size=300)
    draw_point(ax, g_pt, marker, fill_color, "#108000", size=140, ring_size=320)
    draw_point(ax, m20_pt, marker, fill_color, "#0055cc", size=130, ring_size=290)
    draw_point(ax, m5_pt, marker, fill_color, "#d62728", size=140, ring_size=320)

    draw_arrow_segment(ax, b_pt, g_pt, color="#444444")
    draw_arrow_segment(ax, g_pt, m20_pt, color="#0055cc")
    draw_arrow_segment(ax, m20_pt, m5_pt, color="#d62728")

    annotate_segment_box(
        ax,
        b_pt,
        g_pt,
        x_factor=1.0,
        y_offset_factor=1.22,
        color="#106000",
        label_prefix="Baseline → GARI",
    )
    annotate_segment_box(
        ax,
        g_pt,
        m20_pt,
        x_factor=1.05,
        y_offset_factor=0.64,
        color="#0044aa",
        label_prefix="GARI → MP (b=20)",
    )
    annotate_segment_box(
        ax,
        m20_pt,
        m5_pt,
        x_factor=0.82,
        y_offset_factor=1.32,
        color="#b01518",
        label_prefix="MP(b=20) → MP(b=5)",
    )

    tot_spd_b5 = b_pt["time_per_round"] / m5_pt["time_per_round"]
    gari_spd_b5 = g_pt["time_per_round"] / m5_pt["time_per_round"]
    tot_lo, tot_hi = relative_risk_ci(
        b_pt["eff_failures"],
        b_pt["eff_shots"],
        m5_pt["eff_failures"],
        m5_pt["eff_shots"],
        b_pt["r"],
    )
    tot_spd_b20 = b_pt["time_per_round"] / m20_pt["time_per_round"]
    gari_spd_b20 = g_pt["time_per_round"] / m20_pt["time_per_round"]
    b20_lo, b20_hi = relative_risk_ci(
        b_pt["eff_failures"],
        b_pt["eff_shots"],
        m20_pt["eff_failures"],
        m20_pt["eff_shots"],
        b_pt["r"],
    )

    info_box = (
        f"Multi-Pass (2-pass causal) [bb, d=6, q=144]\n"
        f"• beam=20: t/r={m20_pt['time_per_round']:.3e}s, LER/r={m20_pt['ler']:.3e}\n"
        f"  {tot_spd_b20:.1f}x vs Base ({b20_lo:.2f}-{b20_hi:.2f}x err), {gari_spd_b20:.1f}x vs GARI\n"
        f"• beam=5:  t/r={m5_pt['time_per_round']:.3e}s, LER/r={m5_pt['ler']:.3e}\n"
        f"  {tot_spd_b5:.1f}x vs Base ({tot_lo:.2f}-{tot_hi:.2f}x err), {gari_spd_b5:.1f}x vs GARI"
    )
    ax.annotate(
        info_box,
        xy=(m5_pt["time_per_round"], m5_pt["ler"]),
        xytext=(2.2e-5, 3.2e-4),
        fontsize=8.8,
        family="monospace",
        color="#222222",
        bbox=dict(
            boxstyle="round,pad=0.5", fc="#fafafa", ec="#bbbbbb", lw=1.0, alpha=0.95
        ),
        arrowprops=dict(arrowstyle="->", color="#888888", lw=1.0),
        zorder=10,
    )

    ax.set_xlim(1.0e-5, 2.0e0)
    ax.set_ylim(5.0e-6, 2.0e-3)

    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="H",
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
            markeredgewidth=2.4,
            markersize=10,
            label="Baseline (beam=20, climb=True, no_revisit=True, orders=1)",
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
            label="GARI (prior=xor, beam=5, climb=True, orders=1)",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#cccccc",
            markeredgecolor="#0055cc",
            markeredgewidth=2.4,
            markersize=10,
            label="Multi-Pass (2-pass causal, beam=20, climb=True, orders=1)",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#cccccc",
            markeredgecolor="#d62728",
            markeredgewidth=2.6,
            markersize=10,
            label="Multi-Pass (2-pass causal, beam=5, climb=True, orders=1)",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=9,
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


def plot_combined_summary(by_key):
    """Clean progression plot (Baseline -> GARI -> Multi-Pass) across all codes."""
    fig, ax = plt.subplots(figsize=(13.0, 9.0), dpi=200)
    style_axes(
        ax,
        "Tesseract Correlated Decoding Progression: Baseline → GARI → Multi-Pass\n"
        "(SI1000 p=0.001, Memory-Z, Single-Core Decode Time per Round)",
    )

    # Carefully separated offsets so bb_d6 (above) and cc_d7 (below) do not collide
    all_configs = [
        ("cc_d3", "cc d=3", "o", "#f5c242", 1.0, 1.18, 0.85, 0.72, 0.68, 1.0),
        ("cc_d5", "cc d=5", "^", "#f5c242", 1.0, 1.18, 1.0, 0.68, 0.68, 1.0),
        ("cc_d7", "cc d=7", "*", "#f5c242", 0.90, 0.66, 1.45, 0.45, 0.68, 1.05),
        ("cc_d9", "cc d=9", "D", "#f5c242", 1.35, 0.68, 1.95, 0.48, 0.68, 0.95),
        ("bb_d6", "bb d=6", "H", "#ff00ff", 1.15, 1.25, 1.55, 1.35, 0.72, 1.68),
    ]

    for (
        tag,
        short_lbl,
        marker,
        fill_color,
        gx,
        gy,
        mx,
        my,
        bx,
        by,
    ) in all_configs:
        b_pt = by_key[(tag, "baseline_b20")]
        g_pt = by_key[(tag, "gari_b5")]
        m20_pt = by_key[(tag, "mp_2p_b20")]
        m5_pt = by_key[(tag, "mp_2p_b5")]

        draw_point(ax, b_pt, marker, fill_color, "#111111", size=110, ring_size=245)
        draw_point(ax, g_pt, marker, fill_color, "#108000", size=110, ring_size=265)
        draw_point(ax, m20_pt, marker, fill_color, "#0055cc", size=95, ring_size=220)
        draw_point(ax, m5_pt, marker, fill_color, "#d62728", size=110, ring_size=265)

        draw_arrow_segment(ax, b_pt, g_pt, color="#444444")
        draw_arrow_segment(ax, g_pt, m20_pt, color="#0055cc")
        draw_arrow_segment(ax, m20_pt, m5_pt, color="#d62728")

        annotate_segment_box(
            ax,
            b_pt,
            g_pt,
            x_factor=gx,
            y_offset_factor=gy,
            color="#106000",
            label_prefix=f"[{short_lbl}] Base→GARI",
        )
        annotate_segment_box(
            ax,
            g_pt,
            m5_pt,
            x_factor=mx,
            y_offset_factor=my,
            color="#b01518",
            label_prefix=f"[{short_lbl}] GARI→MP(b=5)",
        )

        tot_spd = b_pt["time_per_round"] / m5_pt["time_per_round"]
        gari_spd = g_pt["time_per_round"] / m5_pt["time_per_round"]
        ax.text(
            m5_pt["time_per_round"] * bx,
            m5_pt["ler"] * by,
            f"{short_lbl}: {tot_spd:.0f}x vs Base\n({gari_spd:.1f}x vs GARI)",
            fontsize=8.0,
            fontweight="bold",
            color="#8c1014" if tag != "bb_d6" else "#660066",
            ha="right" if tag != "bb_d6" else "center",
            va="center" if tag != "bb_d6" else "bottom",
            bbox=dict(
                boxstyle="round,pad=0.22",
                fc="#fff8f8" if tag != "bb_d6" else "#fdf2ff",
                ec="#e0b0b0" if tag != "bb_d6" else "#d8a0e0",
                lw=0.8,
                alpha=0.93,
            ),
            zorder=9,
        )

    ax.set_xlim(1.5e-7, 3.0e-1)
    ax.set_ylim(1.8e-6, 1.5e-2)

    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#f5c242",
            markeredgecolor="#b58900",
            markersize=9,
            label="cc, d=3 (q=13)",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="^",
            markerfacecolor="#f5c242",
            markeredgecolor="#b58900",
            markersize=9,
            label="cc, d=5 (q=37)",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="*",
            markerfacecolor="#f5c242",
            markeredgecolor="#b58900",
            markersize=11,
            label="cc, d=7 (q=73)",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="D",
            markerfacecolor="#f5c242",
            markeredgecolor="#b58900",
            markersize=8.5,
            label="cc, d=9 (q=121)",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="H",
            markerfacecolor="#ff00ff",
            markeredgecolor="#aa00aa",
            markersize=10,
            label="bb, d=6 (q=144)",
        ),
        mlines.Line2D([], [], color="none", label=""),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#cccccc",
            markeredgecolor="#111111",
            markeredgewidth=2.4,
            markersize=10,
            label="Baseline (Monolithic Tesseract, beam=20)",
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
            label="GARI (XOR prior, beam=5)",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#cccccc",
            markeredgecolor="#0055cc",
            markeredgewidth=2.4,
            markersize=10,
            label="Multi-Pass (2-pass causal, beam=20)",
        ),
        mlines.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#cccccc",
            markeredgecolor="#d62728",
            markeredgewidth=2.6,
            markersize=10,
            label="Multi-Pass (2-pass causal, beam=5)",
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
    out_path = ARTIFACT_DIR / "combined_baseline_gari_multipass_tradeoffs.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    by_key = load_enriched_results()
    plot_color_codes(by_key)
    plot_bb_code(by_key)
    plot_combined_summary(by_key)


if __name__ == "__main__":
    main()
