# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm
import numpy as np
from scipy.stats import binomtest
from benchmarking.sparsify_errors.benchmark_data import BenchmarkDataError, read_jsonl


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "aggregated_results.jsonl"
DEFAULT_OUTPUT_DIR = HERE / "plots"
FAILURE_CONFIDENCE = 0.95
PLOT_GLOBAL_INVARIANTS = (
    "dem_path",
    "det_beam",
    "det_order_method",
    "det_order_seed",
    "det_penalty",
    "merge_errors",
    "no_revisit_dets",
    "num_det_orders",
    "pqlimit",
)
CIRCUIT_INVARIANTS = (
    "basis",
    "circuit_sha256",
    "code_family",
    "distance",
    "num_compiled_errors",
    "num_detectors",
    "num_qubits",
    "physical_error_rate",
    "rounds",
)


def process_data(filepath):
    """Loads strict, self-contained aggregate rows without pooling X and Z."""

    return read_jsonl(Path(filepath))


def _zero_failure_upper_bound(num_shots, confidence=FAILURE_CONFIDENCE):
    """One-sided exact binomial upper bound for zero observed failures."""

    return 1 - (1 - confidence) ** (1 / num_shots)


def _save_figure(filename):
    path = Path(filename)
    if path.suffix.lower() != ".pdf":
        path = path.with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def _group_exact_circuits(metrics):
    circuits = {}
    for metric in metrics:
        circuits.setdefault(metric["circuit_path"], []).append(metric)
    return circuits


def _marker_map(metrics):
    markers = ["o", "s", "^", "D", "v", "p", "*", "h", "X", "<", ">", "P"]
    result = {}
    for code_family in sorted({metric["type"] for metric in metrics}):
        identities = sorted(
            {
                (metric["type"], metric["d"], metric["q"], metric["basis"])
                for metric in metrics
                if metric["type"] == code_family
            }
        )
        if len(identities) > len(markers):
            raise BenchmarkDataError(
                f"{code_family} needs {len(identities)} distinct markers; "
                f"only {len(markers)} are configured"
            )
        result.update(zip(identities, markers))
    return result


def _observed_segments(points):
    """Yields adjacent curve segments whose endpoints are both observations."""

    for first, second in zip(points, points[1:]):
        if not first["is_upper_limit"] and not second["is_upper_limit"]:
            yield first, second


def _validate_plot_sweep(rows):
    if any(
        row["sparsify_errors"] and row["sparsify_reactivate_limit"] == -1
        for row in rows
    ):
        raise BenchmarkDataError(
            "plot input contains automatic sparsify_reactivate_limit=-1; "
            "plots require explicit resolved limits"
        )

    for field in PLOT_GLOBAL_INVARIANTS:
        values = {row[field] for row in rows}
        if len(values) != 1:
            raise BenchmarkDataError(
                f"plot input mixes values for fixed field {field!r}: {sorted(values)!r}"
            )

    circuits = {}
    for row in rows:
        circuits.setdefault(row["circuit_path"], []).append(row)
    for circuit_path, circuit_rows in circuits.items():
        for field in CIRCUIT_INVARIANTS:
            values = {row[field] for row in circuit_rows}
            if len(values) != 1:
                raise BenchmarkDataError(
                    f"{circuit_path}: circuit field {field!r} changes within the sweep"
                )
        sparsify_settings = {
            (row["sparsify_base_degree"], row["sparsify_max_degree"])
            for row in circuit_rows
            if row["sparsify_errors"]
        }
        if len(sparsify_settings) > 1:
            raise BenchmarkDataError(
                f"{circuit_path}: multiple base/max-degree settings cannot share one curve"
            )


def _relative_risk_interval(before_failures, before_shots, after_failures, after_shots):
    """Returns a 95% log-relative-risk interval with Haldane corrections."""

    if before_shots <= 0 or after_shots <= 0:
        raise BenchmarkDataError("relative-risk sample sizes must be positive")
    before_adjusted = before_failures + 0.5
    after_adjusted = after_failures + 0.5
    before_total_adjusted = before_shots + 1.0
    after_total_adjusted = after_shots + 1.0
    relative_risk = (after_adjusted / after_total_adjusted) / (
        before_adjusted / before_total_adjusted
    )
    standard_error = math.sqrt(
        max(
            0,
            1 / after_adjusted
            - 1 / after_total_adjusted
            + 1 / before_adjusted
            - 1 / before_total_adjusted,
        )
    )
    return (
        relative_risk * math.exp(-1.96 * standard_error),
        relative_risk * math.exp(1.96 * standard_error),
    )


def get_optimal_reactivate_limit(num_detectors, base_degree, c_type):
    """
    The robust M-scaling heuristic.
    Scales exponentially with base degree k, linearly with num_detectors.
    M = round( (4.5^(k-2) / 3) * num_detectors )
    """
    k = base_degree
    if k == -1:
        # Fallback to logical defaults if run with sparsify_errors=False
        k = 2 if c_type == "surfacecodes" else 3

    target_m = (4.5 ** (max(2, k) - 2) / 3.0) * num_detectors
    return max(8, round(target_m))


def compute_metrics(rows):
    """Converts each exact-circuit aggregate row into one plotted metric."""

    _validate_plot_sweep(rows)
    metrics = []
    keys = set()
    for row in rows:
        key = (
            row["circuit_path"],
            row["sparsify_errors"],
            row["sparsify_base_degree"],
            row["sparsify_max_degree"],
            row["sparsify_reactivate_limit"],
        )
        if key in keys:
            raise BenchmarkDataError(f"duplicate aggregate configuration: {key!r}")
        keys.add(key)

        shots = row["num_shots"]
        failures = row["num_errors"] + row["num_low_confidence"]
        observed_probability = failures / shots
        is_upper_limit = failures == 0
        if is_upper_limit:
            plotted_probability = _zero_failure_upper_bound(shots)
            error_low = error_high = 0.0
        else:
            ci = binomtest(k=failures, n=shots).proportion_ci(
                confidence_level=FAILURE_CONFIDENCE
            )
            plotted_probability = observed_probability
            error_low = observed_probability - ci.low
            error_high = ci.high - observed_probability

        rounds = row["rounds"]
        metrics.append(
            {
                "basis": row["basis"],
                "circuit_path": row["circuit_path"],
                "p": row["physical_error_rate"],
                "type": row["code_family"],
                "d": row["distance"],
                "q": row["num_qubits"],
                "r": rounds,
                "M": float("inf")
                if not row["sparsify_errors"]
                else row["sparsify_reactivate_limit"],
                "k": row["sparsify_base_degree"],
                "E": row["num_compiled_errors"],
                "D": row["num_detectors"],
                "num_optional_errors": row["num_optional_errors"],
                "failures": failures,
                "is_upper_limit": is_upper_limit,
                "observed_ler": observed_probability / rounds,
                "ler": plotted_probability / rounds,
                "ler_err_low": error_low / rounds,
                "ler_err_high": error_high / rounds,
                "time_per_round": row["total_time_seconds"] / shots / rounds,
                "shots": shots,
            }
        )
    return sorted(metrics, key=lambda metric: (metric["circuit_path"], metric["M"]))


def get_M_alpha(M):
    if M == float("inf"):
        return 1.0
    logM = math.log2(M) if M > 0 else 0
    return min(max((logM - 4) / 8.0, 0.2), 1.0)


def interpolate_required_M(pareto, target_ler):
    valid_pts = [p for p in pareto if p["M"] > 0 and p["M"] != float("inf")]
    if len(valid_pts) == 0:
        return (
            float("inf")
            if (len(pareto) > 0 and pareto[-1]["ler"] <= target_ler)
            else float("nan")
        )

    for i in range(len(valid_pts) - 1):
        p1, p2 = valid_pts[i], valid_pts[i + 1]
        if p1["ler"] >= target_ler >= p2["ler"]:
            if p1["ler"] == p2["ler"]:
                return p2["M"]
            log_m1, log_m2 = math.log2(p1["M"]), math.log2(p2["M"])
            ratio = (target_ler - p2["ler"]) / (p1["ler"] - p2["ler"])
            return 2 ** (log_m2 + ratio * (log_m1 - log_m2))

    if valid_pts[0]["ler"] <= target_ler:
        return valid_pts[0]["M"]
    return float("inf")


def fit_power_law(x_vals, y_vals):
    valid_pairs = [
        (x, y)
        for x, y in zip(x_vals, y_vals)
        if x > 0 and y > 0 and y != float("inf") and not math.isnan(y)
    ]
    if len(valid_pairs) < 2:
        return float("nan"), float("nan"), float("nan")

    log_x = [math.log2(x) for x, y in valid_pairs]
    log_y = [math.log2(y) for x, y in valid_pairs]
    mean_lx, mean_ly = sum(log_x) / len(log_x), sum(log_y) / len(log_y)

    num = sum((lx - mean_lx) * (ly - mean_ly) for lx, ly in zip(log_x, log_y))
    den = sum((lx - mean_lx) ** 2 for lx in log_x)
    if den == 0:
        return float("nan"), float("nan"), float("nan")

    k = num / den
    log_c = mean_ly - k * mean_lx
    c = 2**log_c

    ss_tot = sum((ly - mean_ly) ** 2 for ly in log_y)
    ss_res = sum((ly - (k * lx + log_c)) ** 2 for lx, ly in zip(log_x, log_y))
    return k, c, 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0


def extract_fit_data(metrics, p_filter):
    filtered = [m for m in metrics if m["p"] == p_filter]
    code_basis_pairs = sorted(set((m["type"], m["basis"]) for m in filtered))
    fit_data = {}
    for c_type, basis in code_basis_pairs:
        c_metrics = [m for m in filtered if m["type"] == c_type and m["basis"] == basis]
        circuits = {}
        for m in c_metrics:
            ckey = m["circuit_path"]
            if ckey not in circuits:
                circuits[ckey] = []
            circuits[ckey].append(m)

        data = {"E": [], "M0": [], "M5": [], "M10": [], "min_ler": [], "ckey": []}
        for circuit_path in sorted(circuits):
            all_pts = circuits[circuit_path]
            pts = [point for point in all_pts if not point["is_upper_limit"]]
            if not pts:
                continue
            pts_sorted = sorted(pts, key=lambda x: x["time_per_round"])
            pareto, best_ler = [], float("inf")
            error_counts = {point["E"] for point in all_pts}
            if len(error_counts) != 1:
                raise BenchmarkDataError(
                    f"compiled error count changes within {circuit_path}"
                )
            error_count = error_counts.pop()

            for pt in pts_sorted:
                if pt["ler"] < best_ler:
                    pareto.append(pt)
                    best_ler = pt["ler"]

            if len(pareto) == 0:
                continue
            data["E"].append(error_count)
            data["M0"].append(interpolate_required_M(pareto, best_ler * 1.0001))
            data["M5"].append(interpolate_required_M(pareto, best_ler * 1.05))
            data["M10"].append(interpolate_required_M(pareto, best_ler * 1.10))
            data["min_ler"].append(best_ler)
            data["ckey"].append((pts[0]["d"], pts[0]["q"], basis))
        fit_data[(c_type, basis)] = data
    return fit_data


def evaluate_scaling_ansatz(metrics, p_filter):
    fit_data = extract_fit_data(metrics, p_filter)
    if not fit_data:
        return

    print(
        f"\n{'=' * 80}\n ERROR COUNT SCALING ANALYSIS: M vs DEM Errors (E) [p = {p_filter}]\n{'=' * 80}"
    )
    for (c_type, basis), data in fit_data.items():
        print(f"\n--- {c_type.upper()} {basis} ---")
        print(
            f"{'d':<4} | {'q':<5} | {'basis':<5} | {'Compiled E':<14} | {'Min rate':<9} | {'+0% M':<10} | {'+5% M':<10} | {'+10% M':<10}"
        )
        print("-" * 79)
        for i in range(len(data["E"])):
            c_d, c_q, c_basis = data["ckey"][i]
            m0, m5, m10 = data["M0"][i], data["M5"][i], data["M10"][i]
            sm0 = f"{m0:.1f}" if m0 != float("inf") and not math.isnan(m0) else "inf"
            sm5 = f"{m5:.1f}" if m5 != float("inf") and not math.isnan(m5) else "inf"
            sm10 = (
                f"{m10:.1f}" if m10 != float("inf") and not math.isnan(m10) else "inf"
            )
            print(
                f"{c_d:<4} | {c_q:<5} | {c_basis:<5} | {data['E'][i]:<14.1f} | {data['min_ler'][i]:.2e} | {sm0:<10} | {sm5:<10} | {sm10:<10}"
            )

        print("-" * 79)
        k0, c0, r0 = fit_power_law(data["E"], data["M0"])
        k5, c5, r5 = fit_power_law(data["E"], data["M5"])
        k10, c10, r10 = fit_power_law(data["E"], data["M10"])

        print("POWER LAW FIT (M = c * E^k)")
        print(f"{'+0% penalty':<15} | {k0:<15.4f} | {c0:<15.4e} | {r0:<10.4f}")
        print(f"{'+5% penalty':<15} | {k5:<15.4f} | {c5:<15.4e} | {r5:<10.4f}")
        print(f"{'+10% penalty':<15} | {k10:<15.4f} | {c10:<15.4e} | {r10:<10.4f}\n")


def plot_power_law_fits(metrics, p_filter, filename, title):
    fit_data = extract_fit_data(metrics, p_filter)
    if not fit_data:
        return
    code_types = ["surfacecodes", "colorcodes", "bivariatebicyclecodes"]
    present_types = [ct for ct in code_types if any(key[0] == ct for key in fit_data)]
    if len(present_types) == 0:
        return

    display_names = {
        "surfacecodes": "Surface Codes",
        "colorcodes": "Color Codes",
        "bivariatebicyclecodes": "Bicycle Codes",
    }
    fig, axes = plt.subplots(
        nrows=1, ncols=len(present_types), figsize=(6 * len(present_types), 6)
    )
    if len(present_types) == 1:
        axes = [axes]

    colors, markers = (
        {"+0%": "black", "+5%": "red", "+10%": "orange"},
        {"+0%": "o", "+5%": "s", "+10%": "^"},
    )
    for idx, c_type in enumerate(present_types):
        ax = axes[idx]
        for basis in ("X", "Z"):
            data = fit_data.get((c_type, basis))
            if data is None:
                continue
            for penalty, key in [("+0%", "M0"), ("+5%", "M5"), ("+10%", "M10")]:
                valid_pairs = [
                    (x, y)
                    for x, y in zip(data["E"], data[key])
                    if y > 0 and y != float("inf") and not math.isnan(y)
                ]
                valid_x = [x for x, _ in valid_pairs]
                valid_y = [y for _, y in valid_pairs]
                if valid_x:
                    ax.scatter(
                        valid_x,
                        valid_y,
                        color=colors[penalty],
                        marker=markers[penalty],
                        alpha=0.85 if basis == "X" else 0.45,
                        label=f"{penalty} {basis} data",
                    )
                    if len(valid_x) > 1:
                        k, c, _ = fit_power_law(valid_x, valid_y)
                        if not math.isnan(k):
                            fit_x = np.linspace(min(valid_x), max(valid_x), 100)
                            ax.plot(
                                fit_x,
                                c * (fit_x**k),
                                color=colors[penalty],
                                linestyle="--" if basis == "X" else ":",
                                alpha=0.8,
                                label=f"{penalty} {basis} fit (k={k:.2f})",
                            )

        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.set_xlabel("Compiled Tesseract errors (E)")
        if idx == 0:
            ax.set_ylabel("Required M limit")
        ax.set_title(f"{display_names[c_type]}")
        ax.legend(fontsize=8)

    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    _save_figure(filename)
    plt.close()


def plot_tradeoff_arrows(metrics, p_filter, filename, title):
    plt.figure(figsize=(10, 8))
    filtered = [m for m in metrics if m["p"] == p_filter or p_filter == "both"]
    circuits = _group_exact_circuits(filtered)

    color_map = {
        "surfacecodes": "#5D95E8",
        "colorcodes": "#F6C644",
        "bivariatebicyclecodes": "fuchsia",
    }
    marker_map = _marker_map(filtered)

    for _, points in circuits.items():
        c_type = points[0]["type"]
        c_d = points[0]["d"]
        c_q = points[0]["q"]
        c_p = points[0]["p"]
        basis = points[0]["basis"]
        base_color = color_map.get(c_type, "black")
        marker = marker_map[(c_type, c_d, c_q, basis)]

        before_pts = [p for p in points if p["M"] == float("inf")]
        if not before_pts:
            continue
        before_pt = before_pts[0]

        k_vals = [p["k"] for p in points if p["k"] != -1]
        k_val = k_vals[0] if len(k_vals) > 0 else -1
        detector_counts = {p["D"] for p in points}
        if len(detector_counts) != 1:
            raise BenchmarkDataError(
                f"detector count changes within {points[0]['circuit_path']}"
            )
        opt_M = get_optimal_reactivate_limit(detector_counts.pop(), k_val, c_type)

        valid_pts = [p for p in points if p["M"] > 0 and p["M"] != float("inf")]
        if not valid_pts:
            continue

        after_pt = min(valid_pts, key=lambda x: abs(x["M"] - opt_M))

        x0, y0 = before_pt["time_per_round"], before_pt["ler"]
        x1, y1 = after_pt["time_per_round"], after_pt["ler"]

        if x0 <= 0 or x1 <= 0 or y0 <= 0 or y1 <= 0:
            continue

        is_p002 = c_p == 0.002
        fc = "white" if is_p002 else base_color
        ls = "--" if is_p002 else "-"
        ec_before = "black"
        ec_after = base_color if is_p002 else "none"
        lw_after = 1.5 if is_p002 else 0

        y0_err = [[before_pt["ler_err_low"]], [before_pt["ler_err_high"]]]
        y1_err = [[after_pt["ler_err_low"]], [after_pt["ler_err_high"]]]

        if not before_pt["is_upper_limit"]:
            plt.errorbar(
                [x0],
                [y0],
                yerr=y0_err,
                fmt="none",
                ecolor=ec_before,
                alpha=0.3,
                zorder=1,
            )
        if not after_pt["is_upper_limit"]:
            plt.errorbar(
                [x1],
                [y1],
                yerr=y1_err,
                fmt="none",
                ecolor=base_color,
                alpha=0.7,
                zorder=2,
            )

        plt.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->",
                color=base_color,
                lw=1.5,
                ls=":"
                if before_pt["is_upper_limit"] or after_pt["is_upper_limit"]
                else ls,
                shrinkA=8,
                shrinkB=8,
            ),
            zorder=3,
        )

        plt.scatter(
            [x0],
            [y0],
            facecolors=fc,
            edgecolors=ec_before,
            marker="v" if before_pt["is_upper_limit"] else marker,
            s=80,
            linewidths=1.5,
            alpha=0.4,
            zorder=4,
        )
        plt.scatter(
            [x1],
            [y1],
            facecolors=fc,
            edgecolors=ec_after,
            marker="v" if after_pt["is_upper_limit"] else marker,
            s=80,
            linewidths=lw_after,
            alpha=1.0,
            zorder=5,
        )

        speedup = x0 / x1 if x1 > 0 else 1

        if before_pt["is_upper_limit"] or after_pt["is_upper_limit"]:
            ler_str = "censored (95% bound)"
        else:
            r_low, r_high = _relative_risk_interval(
                before_pt["failures"],
                before_pt["shots"],
                after_pt["failures"],
                after_pt["shots"],
            )
            if round(r_low, 2) == round(r_high, 2):
                ler_str = f"{r_low:.2f}x err"
            else:
                ler_str = f"{r_low:.2f}-{r_high:.2f}x err"

        mid_x = math.exp((math.log(x0) + math.log(x1)) / 2)
        mid_y = math.exp((math.log(y0) + math.log(y1)) / 2)

        plt.text(
            mid_x,
            mid_y * 1.05,
            f"{speedup:.1f}x spd\n{ler_str}",
            fontsize=6,
            color="black",
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            zorder=6,
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)

    valid_lers = [p["ler"] for p in filtered if p["ler"] > 0]
    if valid_lers:
        plt.ylim(bottom=min(valid_lers) / 5.0)

    plt.xlabel("Time per round (seconds)")
    plt.ylabel("Shot failure probability ÷ rounds")
    plt.title(title)

    legend_elements = []
    display_names = {
        "surfacecodes": "Surface Codes",
        "colorcodes": "Color Codes",
        "bivariatebicyclecodes": "Bicycle Codes",
    }

    for c_type in ["surfacecodes", "colorcodes", "bivariatebicyclecodes"]:
        type_qdbs = sorted(
            set((m["d"], m["q"], m["basis"]) for m in filtered if m["type"] == c_type)
        )
        if type_qdbs:
            c_color = color_map.get(c_type, "black")
            k_set = set(
                [m["k"] for m in filtered if m["type"] == c_type and m["k"] != -1]
            )
            k_str = f" (k={list(k_set)[0]})" if len(k_set) == 1 else ""
            legend_elements.append(
                mlines.Line2D(
                    [0], [0], color="none", label=f"  {display_names[c_type]}{k_str}"
                )
            )
            for qdb in type_qdbs:
                legend_elements.append(
                    mlines.Line2D(
                        [0],
                        [0],
                        color="none",
                        marker=marker_map[(c_type, *qdb)],
                        markerfacecolor=c_color,
                        markeredgecolor="none",
                        markersize=8,
                        label=f"d={qdb[0]}, q={qdb[1]}, {qdb[2]}",
                    )
                )

    legend_elements.append(mlines.Line2D([0], [0], color="none", label=""))
    if p_filter == "both" or p_filter == 0.001:
        legend_elements.append(
            mlines.Line2D(
                [0],
                [0],
                color="gray",
                linestyle="-",
                lw=2,
                marker="o",
                markerfacecolor="gray",
                markeredgecolor="none",
                label="p=0.001 (Solid Line, Filled)",
            )
        )
    if p_filter == "both" or p_filter == 0.002:
        legend_elements.append(
            mlines.Line2D(
                [0],
                [0],
                color="gray",
                linestyle="--",
                lw=2,
                marker="o",
                markerfacecolor="white",
                markeredgecolor="gray",
                markeredgewidth=1.5,
                label="p=0.002 (Dashed Line, Hollow)",
            )
        )

    legend_elements.extend(
        [
            mlines.Line2D(
                [0],
                [0],
                color="gray",
                marker="o",
                linestyle="None",
                markerfacecolor="gray",
                markeredgecolor="black",
                markersize=8,
                markeredgewidth=1.5,
                alpha=0.4,
                label="Before (sparsify_errors=False)",
            ),
            mlines.Line2D(
                [0],
                [0],
                color="gray",
                marker="o",
                linestyle="None",
                markerfacecolor="gray",
                markeredgecolor="none",
                markersize=8,
                alpha=1.0,
                label="After (Heuristic M applied)",
            ),
            mlines.Line2D(
                [0],
                [0],
                color="gray",
                marker="v",
                linestyle="None",
                markersize=8,
                label="95% upper limit (0 failures)",
            ),
        ]
    )

    plt.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=8,
        labelspacing=0.6,
    )
    plt.tight_layout()
    _save_figure(filename)


def plot_ler_vs_time(metrics, p_filter, filename, title, highlight_heuristic=False):
    plt.figure(figsize=(10, 8))
    filtered = [m for m in metrics if m["p"] == p_filter or p_filter == "both"]
    circuits = _group_exact_circuits(filtered)

    color_map = {
        "surfacecodes": "#5D95E8",
        "colorcodes": "#F6C644",
        "bivariatebicyclecodes": "fuchsia",
    }
    marker_map = _marker_map(filtered)

    for _, points in circuits.items():
        points.sort(key=lambda x: x["M"])
        c_type = points[0]["type"]
        c_d = points[0]["d"]
        c_q = points[0]["q"]
        c_p = points[0]["p"]
        basis = points[0]["basis"]
        base_color = color_map.get(c_type, "black")
        marker = marker_map[(c_type, c_d, c_q, basis)]

        is_p002 = c_p == 0.002
        line_style = "--" if is_p002 else "-"

        for p1, p2 in _observed_segments(points):
            seg_alpha = (get_M_alpha(p1["M"]) + get_M_alpha(p2["M"])) / 2.0
            plt.plot(
                [p1["time_per_round"], p2["time_per_round"]],
                [p1["ler"], p2["ler"]],
                color=base_color,
                linestyle=line_style,
                alpha=seg_alpha,
                linewidth=1.5,
                zorder=1,
            )

        for p in points:
            M, alpha = p["M"], get_M_alpha(p["M"])
            sz = 80 if M == float("inf") else 50

            fc = "white" if is_p002 else base_color
            ec = "black" if M == float("inf") else (base_color if is_p002 else "none")
            lw = 1.5 if (M == float("inf") or is_p002) else 0

            if not p["is_upper_limit"]:
                y_err_asym = [[p["ler_err_low"]], [p["ler_err_high"]]]
                plt.errorbar(
                    p["time_per_round"],
                    p["ler"],
                    yerr=y_err_asym,
                    fmt="none",
                    ecolor=base_color,
                    alpha=alpha,
                    zorder=2,
                )
            plt.scatter(
                [p["time_per_round"]],
                [p["ler"]],
                facecolors=fc,
                edgecolors=ec,
                linewidths=lw,
                marker="v" if p["is_upper_limit"] else marker,
                alpha=alpha,
                s=sz,
                zorder=3,
            )

        if highlight_heuristic and len(points) > 0:
            detector_counts = {p["D"] for p in points}
            if len(detector_counts) != 1:
                raise BenchmarkDataError(
                    f"detector count changes within {points[0]['circuit_path']}"
                )
            k_val = [p["k"] for p in points if p["k"] != -1]
            opt_M = get_optimal_reactivate_limit(
                detector_counts.pop(), k_val[0] if k_val else -1, c_type
            )

            valid_points = [p for p in points if p["M"] > 0 and p["M"] != float("inf")]
            if valid_points:
                best_p = min(valid_points, key=lambda x: abs(x["M"] - opt_M))
                plt.scatter(
                    [best_p["time_per_round"]],
                    [best_p["ler"]],
                    facecolors="none",
                    edgecolors="red",
                    linewidths=2.5,
                    marker="o",
                    s=250,
                    zorder=4,
                )

    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)

    valid_lers = [p["ler"] for p in filtered if p["ler"] > 0]
    if valid_lers:
        plt.ylim(bottom=min(valid_lers) / 5.0)

    plt.xlabel("Time per round (seconds)")
    plt.ylabel("Shot failure probability ÷ rounds")
    plt.title(title)

    legend_elements = []
    display_names = {
        "surfacecodes": "Surface Codes",
        "colorcodes": "Color Codes",
        "bivariatebicyclecodes": "Bicycle Codes",
    }

    for c_type in ["surfacecodes", "colorcodes", "bivariatebicyclecodes"]:
        type_qdbs = sorted(
            set((m["d"], m["q"], m["basis"]) for m in filtered if m["type"] == c_type)
        )
        if type_qdbs:
            c_color = color_map.get(c_type, "black")
            k_set = set(
                [m["k"] for m in filtered if m["type"] == c_type and m["k"] != -1]
            )
            k_str = f" (k={list(k_set)[0]})" if len(k_set) == 1 else ""
            legend_elements.append(
                mlines.Line2D(
                    [0], [0], color="none", label=f"  {display_names[c_type]}{k_str}"
                )
            )
            for qdb in type_qdbs:
                legend_elements.append(
                    mlines.Line2D(
                        [0],
                        [0],
                        color="none",
                        marker=marker_map[(c_type, *qdb)],
                        markerfacecolor=c_color,
                        markeredgecolor="none",
                        markersize=8,
                        label=f"d={qdb[0]}, q={qdb[1]}, {qdb[2]}",
                    )
                )

    legend_elements.append(mlines.Line2D([0], [0], color="none", label=""))
    if p_filter == "both" or p_filter == 0.001:
        legend_elements.append(
            mlines.Line2D(
                [0],
                [0],
                color="gray",
                linestyle="-",
                lw=2,
                marker="o",
                markerfacecolor="gray",
                markeredgecolor="none",
                label="p=0.001 (Solid Line, Filled)",
            )
        )
    if p_filter == "both" or p_filter == 0.002:
        legend_elements.append(
            mlines.Line2D(
                [0],
                [0],
                color="gray",
                linestyle="--",
                lw=2,
                marker="o",
                markerfacecolor="white",
                markeredgecolor="gray",
                markeredgewidth=1.5,
                label="p=0.002 (Dashed Line, Hollow)",
            )
        )

    legend_elements.extend(
        [
            mlines.Line2D(
                [0],
                [0],
                color="gray",
                marker="o",
                linestyle="None",
                markersize=8,
                alpha=0.3,
                label="Lower M",
            ),
            mlines.Line2D(
                [0],
                [0],
                color="gray",
                marker="o",
                linestyle="None",
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=1.5,
                label="M=inf (sparsify_errors=false)",
            ),
            mlines.Line2D(
                [0],
                [0],
                color="gray",
                marker="v",
                linestyle="None",
                markersize=8,
                label="95% upper limit (0 failures)",
            ),
        ]
    )

    if highlight_heuristic:
        legend_elements.append(mlines.Line2D([0], [0], color="none", label=""))
        legend_elements.append(
            mlines.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markeredgecolor="red",
                markerfacecolor="none",
                markersize=12,
                markeredgewidth=2.5,
                label="Heuristic Optimal M",
            )
        )

    plt.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=8,
        labelspacing=0.6,
    )
    plt.tight_layout()
    _save_figure(filename)


def plot_stacked_ler_vs_M(metrics, p_filter, filename, title, log_y=False):
    filtered = [m for m in metrics if m["p"] == p_filter or p_filter == "both"]
    circuits = {}
    for m in filtered:
        ckey = (m["type"], m["d"], m["q"])
        circuits.setdefault(ckey, []).append(m)

    sorted_keys = sorted(list(circuits.keys()), key=lambda x: (x[0], x[2]))
    num_subplots = len(sorted_keys)
    if num_subplots == 0:
        return

    fig, axes = plt.subplots(
        nrows=num_subplots, ncols=1, sharex=True, figsize=(8, 2.5 * num_subplots)
    )
    if num_subplots == 1:
        axes = [axes]

    color_map = {
        "surfacecodes": "#5D95E8",
        "colorcodes": "#F6C644",
        "bivariatebicyclecodes": "fuchsia",
    }
    display_names = {
        "surfacecodes": "Surface Codes",
        "colorcodes": "Color Codes",
        "bivariatebicyclecodes": "Bicycle Codes",
    }

    for i in range(num_subplots):
        ckey = sorted_keys[i]
        c_type, c_d, c_q = ckey
        all_points = circuits[ckey]
        ax = axes[i]

        exact_groups = {}
        for point in all_points:
            group_key = (point["basis"], point["p"], point["circuit_path"])
            exact_groups.setdefault(group_key, []).append(point)

        finite_Ms = [
            p["M"] for p in all_points if p["M"] != float("inf") and p["M"] > 0
        ]
        max_M = max(finite_Ms) if finite_Ms else 1
        min_M = min(finite_Ms) if finite_Ms else 1
        color = color_map.get(c_type, "black")

        for (basis, p_val, _), pts in sorted(exact_groups.items()):
            pts.sort(key=lambda x: x["M"])
            plotted_points = []
            upper_limits = []
            for pt in pts:
                if pt["M"] == float("inf"):
                    x_value = max_M * 4
                elif pt["M"] == 0:
                    x_value = min_M / 4
                else:
                    x_value = pt["M"]
                plotted_point = {**pt, "plot_x": x_value}
                plotted_points.append(plotted_point)
                if pt["is_upper_limit"]:
                    upper_limits.append(plotted_point)

            observed = [
                point for point in plotted_points if not point["is_upper_limit"]
            ]

            is_p002 = p_val == 0.002
            ls = "--" if is_p002 else "-"
            marker = "o" if basis == "X" else "s"
            mfc = "white" if is_p002 else color

            k_set = {point["k"] for point in pts if point["k"] != -1}
            k_str = f" (k={list(k_set)[0]})" if len(k_set) == 1 else ""
            label = f"{basis}, p={p_val}{k_str}"

            if observed:
                for first, second in _observed_segments(plotted_points):
                    ax.plot(
                        [first["plot_x"], second["plot_x"]],
                        [first["ler"], second["ler"]],
                        linestyle=ls,
                        color=color,
                    )
                ax.errorbar(
                    [point["plot_x"] for point in observed],
                    [point["ler"] for point in observed],
                    yerr=[
                        [point["ler_err_low"] for point in observed],
                        [point["ler_err_high"] for point in observed],
                    ],
                    fmt=marker,
                    linestyle="none",
                    color=color,
                    markerfacecolor=mfc,
                    capsize=3,
                    label=label,
                )
                for point in observed:
                    if point["M"] in (0, float("inf")):
                        special_marker = "X" if point["M"] == 0 else "*"
                        size = 8 if point["M"] == 0 else 10
                        ax.scatter(
                            [point["plot_x"]],
                            [point["ler"]],
                            marker=special_marker,
                            facecolor=mfc,
                            edgecolor="black",
                            s=size**2,
                            zorder=5,
                        )
            for upper_limit_index, point in enumerate(upper_limits):
                ax.scatter(
                    [point["plot_x"]],
                    [point["ler"]],
                    marker="v",
                    facecolor=mfc,
                    edgecolor=color,
                    s=64,
                    zorder=6,
                    label=(
                        f"{label}, 95% upper limit"
                        if upper_limit_index == 0
                        else "_nolegend_"
                    ),
                )

        if log_y:
            ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.axvline(x=max_M * 2, color="gray", linestyle=":")
        ax.axvline(x=min_M / 2, color="gray", linestyle=":")
        ax.set_ylabel("Shot failure probability ÷ rounds")
        ax.set_title(f"{display_names.get(c_type, c_type)} (d={c_d}, q={c_q})")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)

    axes[-1].set_xlabel("M (log2 scale; M=0 uses ×, baseline uses *)")
    fig.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    _save_figure(filename)


def plot_mq_scaling_meta_analysis(metrics, p_filter, filename, title):
    filtered = [m for m in metrics if m["p"] == p_filter or p_filter == "both"]
    if len(filtered) == 0:
        return

    code_types = ["surfacecodes", "colorcodes", "bivariatebicyclecodes"]
    display_names = {
        "surfacecodes": "Surface Codes",
        "colorcodes": "Color Codes",
        "bivariatebicyclecodes": "Bicycle Codes",
    }
    present_types = [ct for ct in code_types if any(m["type"] == ct for m in filtered)]
    if len(present_types) == 0:
        return

    fig, axes = plt.subplots(
        nrows=len(present_types),
        ncols=1,
        figsize=(10, 6 * len(present_types)),
        sharex=False,
    )
    if len(present_types) == 1:
        axes = [axes]

    for idx, c_type in enumerate(present_types):
        ax1 = axes[idx]
        ax2 = ax1.twinx()

        c_metrics = [m for m in filtered if m["type"] == c_type]
        circuits = _group_exact_circuits(c_metrics)

        sorted_keys = sorted(list(circuits.keys()))
        all_finite_fractions = [
            point["M"] / point["E"]
            for points in circuits.values()
            for point in points
            if point["M"] > 0
            and point["M"] != float("inf")
            and not point["is_upper_limit"]
        ]

        if all_finite_fractions:
            min_frac, max_frac = min(all_finite_fractions), max(all_finite_fractions)
            if min_frac >= max_frac:
                min_frac *= 0.5
                max_frac *= 2.0
        else:
            min_frac, max_frac = 0.5, 2.0

        x_zero, x_inf = min_frac / 4.0, max_frac * 4.0
        colors = cm.viridis(
            [i / max(1, len(sorted_keys) - 1) for i in range(len(sorted_keys))]
        )

        for c_idx, circuit_path in enumerate(sorted_keys):
            pts = circuits[circuit_path]
            c_d, c_q, c_p = pts[0]["d"], pts[0]["q"], pts[0]["p"]
            basis = pts[0]["basis"]

            pts_sorted = sorted(
                (point for point in pts if not point["is_upper_limit"]),
                key=lambda x: x["time_per_round"],
            )
            pareto, best_ler = [], float("inf")
            for pt in pts_sorted:
                if pt["ler"] < best_ler:
                    pareto.append(pt)
                    best_ler = pt["ler"]

            if not pareto:
                continue

            min_ler_val = max(1e-12, best_ler)
            ref_time_val = max(1e-12, pareto[-1]["time_per_round"])
            x_vals, y1_vals, y2_vals, y1_err_low, y1_err_high = [], [], [], [], []

            for pt in pareto:
                if pt["M"] == 0:
                    x = x_zero
                elif pt["M"] == float("inf"):
                    x = x_inf
                else:
                    x = pt["M"] / pt["E"]

                x_vals.append(x)
                y1_vals.append(pt["ler"] / min_ler_val)
                y2_vals.append(pt["time_per_round"] / ref_time_val)
                y1_err_low.append(pt["ler_err_low"] / min_ler_val)
                y1_err_high.append(pt["ler_err_high"] / min_ler_val)

            color = colors[c_idx]
            is_p002 = c_p == 0.002
            mfc = "white" if is_p002 else color
            ls = "--" if is_p002 else "-"
            label = f"d={c_d}, q={c_q}, {basis}"
            if p_filter == "both":
                label += f", p={c_p}"
            error_marker = "o" if basis == "X" else "^"
            time_marker = "s" if basis == "X" else "D"

            ax1.plot(
                x_vals, y1_vals, linestyle=ls, color=color, label=label, markersize=0
            )
            ax1.errorbar(
                x_vals,
                y1_vals,
                yerr=[y1_err_low, y1_err_high],
                fmt=error_marker,
                color=color,
                markerfacecolor=mfc,
                capsize=4,
                markersize=6,
            )
            ax2.plot(
                x_vals,
                y2_vals,
                marker=time_marker,
                linestyle=":",
                color=color,
                markerfacecolor=mfc,
                alpha=0.7,
                markersize=5,
            )

            for i, pt in enumerate(pareto):
                if pt["M"] == 0:
                    ax1.scatter(
                        x_vals[i],
                        y1_vals[i],
                        marker="X",
                        facecolor=mfc,
                        edgecolor="black",
                        s=80,
                        zorder=5,
                    )
                    ax2.scatter(
                        x_vals[i],
                        y2_vals[i],
                        marker="X",
                        facecolor=mfc,
                        edgecolor="black",
                        s=70,
                        alpha=0.7,
                        zorder=5,
                    )
                elif pt["M"] == float("inf"):
                    ax1.scatter(
                        x_vals[i],
                        y1_vals[i],
                        marker="*",
                        facecolor=mfc,
                        edgecolor="black",
                        s=120,
                        zorder=5,
                    )
                    ax2.scatter(
                        x_vals[i],
                        y2_vals[i],
                        marker="*",
                        facecolor=mfc,
                        edgecolor="black",
                        s=100,
                        alpha=0.7,
                        zorder=5,
                    )

        ax1.set_xscale("log", base=2)
        ax1.set_yscale("log")
        ax1.set_ylim(bottom=0.9)
        ax1.axhline(1.00, color="black", linestyle="-", alpha=0.4, linewidth=1)
        ax1.axhline(
            1.05,
            color="red",
            linestyle=":",
            alpha=0.8,
            label="+5% failure-rate penalty",
        )
        ax1.axhline(
            1.10,
            color="orange",
            linestyle=":",
            alpha=0.8,
            label="+10% failure-rate penalty",
        )
        ax1.axvline(x_zero * 2, color="gray", linestyle=":")
        ax1.axvline(x_inf / 2, color="gray", linestyle=":")
        ax1.set_ylabel(
            "Normalized failure rate (solid)", color="black", fontweight="bold"
        )
        ax2.set_ylabel(
            "Normalized time (dotted; square/diamond markers)",
            color="gray",
            fontweight="bold",
        )
        ax1.set_title(f"{display_names[c_type]} (p={p_filter})")

        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper left",
            bbox_to_anchor=(1.10, 1),
        )

    axes[-1].set_xlabel("M / compiled errors [log2 scale; × marks M=0; * marks M=∞]")
    fig.suptitle(title, y=1.02, fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_figure(filename)


def _default_input() -> Path:
    repository_path = (
        Path.cwd() / "benchmarking/sparsify_errors/aggregated_results.jsonl"
    )
    if repository_path.exists():
        return repository_path
    working_directory_path = Path.cwd() / "aggregated_results.jsonl"
    if working_directory_path.exists():
        return working_directory_path
    return DEFAULT_INPUT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=_default_input())
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.input.parent / "plots"
    try:
        metrics = compute_metrics(process_data(args.input))
    except (BenchmarkDataError, OSError) as ex:
        print(ex, file=sys.stderr)
        return 1

    print(
        f"Loaded {len(metrics)} exact-circuit aggregate points. "
        "Running analysis and generating plots..."
    )
    for p_val in [0.001, 0.002]:
        evaluate_scaling_ansatz(metrics, p_val)
        suffix = f"p{p_val}.pdf"
        plot_ler_vs_time(
            metrics,
            p_val,
            output_dir / f"ler_vs_time_{suffix}",
            f"Failure Rate vs Time per Round (p={p_val})",
        )
        plot_ler_vs_time(
            metrics,
            p_val,
            output_dir / f"ler_vs_time_highlighted_{suffix}",
            f"Failure Rate vs Time per Round - Heuristic Target (p={p_val})",
            highlight_heuristic=True,
        )
        plot_tradeoff_arrows(
            metrics,
            p_val,
            output_dir / f"tradeoff_arrows_{suffix}",
            f"Before vs After: Sparsification Tradeoffs (p={p_val})",
        )
        plot_stacked_ler_vs_M(
            metrics,
            p_val,
            output_dir / f"ler_vs_M_stacked_{suffix}",
            f"Failure Rate vs M (p={p_val})",
        )
        plot_stacked_ler_vs_M(
            metrics,
            p_val,
            output_dir / f"ler_vs_M_stacked_logy_{suffix}",
            f"Failure Rate vs M [Log Y] (p={p_val})",
            log_y=True,
        )
        plot_mq_scaling_meta_analysis(
            metrics,
            p_val,
            output_dir / f"mq_scaling_meta_{suffix}",
            f"M/Compiled-Error Scaling Meta-Analysis (p={p_val})",
        )
        plot_power_law_fits(
            metrics,
            p_val,
            output_dir / f"power_law_fits_{suffix}",
            f"Power Law Extrapolations (p={p_val})",
        )

    plot_ler_vs_time(
        metrics,
        "both",
        output_dir / "ler_vs_time_combined.pdf",
        "Failure Rate vs Time per Round (Combined)",
    )
    plot_ler_vs_time(
        metrics,
        "both",
        output_dir / "ler_vs_time_highlighted_combined.pdf",
        "Failure Rate vs Time per Round - Heuristic Target (Combined)",
        highlight_heuristic=True,
    )
    plot_tradeoff_arrows(
        metrics,
        "both",
        output_dir / "tradeoff_arrows_combined.pdf",
        "Before vs After: Sparsification Tradeoffs (Combined)",
    )
    plot_stacked_ler_vs_M(
        metrics,
        "both",
        output_dir / "ler_vs_M_stacked_combined.pdf",
        "Failure Rate vs M (Combined)",
    )
    plot_stacked_ler_vs_M(
        metrics,
        "both",
        output_dir / "ler_vs_M_stacked_logy_combined.pdf",
        "Failure Rate vs M [Log Y] (Combined)",
        log_y=True,
    )

    print(f"Done. PDF plots saved in {output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
