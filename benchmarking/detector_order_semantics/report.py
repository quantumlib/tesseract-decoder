#!/usr/bin/env python3
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

"""Validate raw PR #306 stats and generate JSONL, Markdown, and an SVG plot."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Iterable


Z_95 = 1.959963984540054
CASE_ORDER = {
    "surface-d7": 0,
    "color-d7": 1,
    "bb-72-12-6": 2,
}
ORDER_ORDER = {"bfs": 0, "coordinate": 1}
REVISION_NAMES = {"baseline", "candidate"}


def display_order(value: str) -> str:
    return "BFS" if value == "bfs" else value.capitalize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def strict_json_loads(text: str, label: str) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant {value!r} in {label}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r} in {label}")
            result[key] = value
        return result

    return json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )


def validate_int(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def validate_number(value: object, label: str, *, minimum: float = 0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{label} must be finite and >= {minimum}")
    return result


def validate_raw_record(record: object, label: str) -> dict[str, object]:
    if not isinstance(record, dict):
        raise ValueError(f"{label} must contain a JSON object")
    required = {
        "schema",
        "schema_version",
        "case_slug",
        "case_label",
        "circuit",
        "circuit_sha256",
        "order",
        "revision",
        "source_commit",
        "binary_sha256",
        "sample_seed",
        "det_order_seed",
        "requested_shots",
        "threads",
        "wall_time_seconds",
        "native_stats",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"{label} is missing {sorted(missing)}")
    if record["schema"] != "tesseract.detector_order_semantics_raw":
        raise ValueError(f"{label} has the wrong schema")
    if record["schema_version"] != 1:
        raise ValueError(f"{label} has an unsupported schema version")
    if record["case_slug"] not in CASE_ORDER:
        raise ValueError(f"{label} has an unknown case")
    if record["order"] not in ORDER_ORDER:
        raise ValueError(f"{label} has an unknown detector order")
    if record["revision"] not in REVISION_NAMES:
        raise ValueError(f"{label} has an unknown revision")
    for field in ("case_label", "circuit"):
        if not isinstance(record[field], str) or not record[field]:
            raise ValueError(f"{label}.{field} must be a nonempty string")
    for field, length in (
        ("circuit_sha256", 64),
        ("binary_sha256", 64),
        ("source_commit", 40),
    ):
        value = record[field]
        if (
            not isinstance(value, str)
            or len(value) != length
            or any(ch not in "0123456789abcdef" for ch in value)
        ):
            raise ValueError(f"{label}.{field} is not valid hexadecimal")
    for field in ("sample_seed", "det_order_seed"):
        validate_int(record[field], f"{label}.{field}")
    requested = validate_int(
        record["requested_shots"], f"{label}.requested_shots", minimum=1
    )
    validate_int(record["threads"], f"{label}.threads", minimum=1)
    wall = record["wall_time_seconds"]
    if wall is not None:
        validate_number(wall, f"{label}.wall_time_seconds")
    stats = record["native_stats"]
    if not isinstance(stats, dict):
        raise ValueError(f"{label}.native_stats must be an object")
    for field in ("num_shots", "num_errors", "num_low_confidence"):
        if field not in stats:
            raise ValueError(f"{label}.native_stats is missing {field}")
        validate_int(stats[field], f"{label}.native_stats.{field}")
    if "total_time_seconds" not in stats:
        raise ValueError(f"{label}.native_stats is missing total_time_seconds")
    validate_number(
        stats["total_time_seconds"],
        f"{label}.native_stats.total_time_seconds",
    )
    if stats["num_shots"] != requested:
        raise ValueError(f"{label} did not complete all requested shots")
    failures = stats["num_errors"] + stats["num_low_confidence"]
    if failures > stats["num_shots"]:
        raise ValueError(f"{label} has more failures than shots")
    return record


def read_records(input_dir: Path) -> list[dict[str, object]]:
    paths = sorted(input_dir.glob("*.json"))
    records = [
        validate_raw_record(
            strict_json_loads(path.read_text(encoding="utf-8"), str(path)), str(path)
        )
        for path in paths
        if not path.name.endswith(".native.json")
    ]
    if len(records) != 12:
        raise ValueError(f"expected exactly 12 raw records; found {len(records)}")
    keyed = {}
    for record in records:
        key = (record["case_slug"], record["order"], record["revision"])
        if key in keyed:
            raise ValueError(f"duplicate raw record for {key}")
        keyed[key] = record
    expected = {
        (case, order, revision)
        for case in CASE_ORDER
        for order in ORDER_ORDER
        for revision in REVISION_NAMES
    }
    if set(keyed) != expected:
        raise ValueError(f"raw record grid mismatch: missing={sorted(expected - set(keyed))}")
    return records


def wilson_interval(failures: int, shots: int) -> tuple[float, float]:
    p = failures / shots
    z2 = Z_95 * Z_95
    denominator = 1 + z2 / shots
    center = p + z2 / (2 * shots)
    spread = Z_95 * math.sqrt((p * (1 - p) + z2 / (4 * shots)) / shots)
    return (max(0.0, (center - spread) / denominator),
            min(1.0, (center + spread) / denominator))


def relative_reduction(
    baseline_failures: int,
    baseline_shots: int,
    candidate_failures: int,
    candidate_shots: int,
) -> tuple[float | None, float | None, float | None, float | None]:
    if baseline_failures == 0 or candidate_failures == 0:
        return None, None, None, None
    baseline_rate = baseline_failures / baseline_shots
    candidate_rate = candidate_failures / candidate_shots
    risk_ratio = candidate_rate / baseline_rate
    log_se = math.sqrt(
        1 / candidate_failures
        - 1 / candidate_shots
        + 1 / baseline_failures
        - 1 / baseline_shots
    )
    ratio_low = risk_ratio * math.exp(-Z_95 * log_se)
    ratio_high = risk_ratio * math.exp(Z_95 * log_se)
    return (
        1 - risk_ratio,
        risk_ratio * log_se,
        1 - ratio_high,
        1 - ratio_low,
    )


def log_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_exact_two_sided(
    baseline_failures: int,
    baseline_shots: int,
    candidate_failures: int,
    candidate_shots: int,
) -> float:
    """Two-sided Fisher exact p-value for two independent binomial samples."""
    total_failures = baseline_failures + candidate_failures
    total_shots = baseline_shots + candidate_shots
    lower = max(0, total_failures - candidate_shots)
    upper = min(baseline_shots, total_failures)

    def log_probability(baseline_count: int) -> float:
        return (
            log_choose(baseline_shots, baseline_count)
            + log_choose(candidate_shots, total_failures - baseline_count)
            - log_choose(total_shots, total_failures)
        )

    observed_log = log_probability(baseline_failures)
    selected_logs = [
        log_probability(count)
        for count in range(lower, upper + 1)
        if log_probability(count) <= observed_log + 1e-12
    ]
    if not selected_logs:
        return 0.0
    maximum = max(selected_logs)
    return min(1.0, math.exp(maximum) * sum(math.exp(x - maximum) for x in selected_logs))


def make_comparisons(records: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    keyed = {
        (record["case_slug"], record["order"], record["revision"]): record
        for record in records
    }
    rows = []
    for case in CASE_ORDER:
        for order in ORDER_ORDER:
            baseline = keyed[(case, order, "baseline")]
            candidate = keyed[(case, order, "candidate")]
            fields_that_must_match = (
                "case_label",
                "circuit",
                "circuit_sha256",
                "sample_seed",
                "det_order_seed",
                "requested_shots",
                "threads",
            )
            for field in fields_that_must_match:
                if baseline[field] != candidate[field]:
                    raise ValueError(f"{case}/{order} differs in {field}")
            if baseline["binary_sha256"] == candidate["binary_sha256"]:
                raise ValueError(f"{case}/{order} used identical binaries")
            b_stats = baseline["native_stats"]
            c_stats = candidate["native_stats"]
            b_failures = b_stats["num_errors"] + b_stats["num_low_confidence"]
            c_failures = c_stats["num_errors"] + c_stats["num_low_confidence"]
            b_shots = b_stats["num_shots"]
            c_shots = c_stats["num_shots"]
            b_rate = b_failures / b_shots
            c_rate = c_failures / c_shots
            b_low, b_high = wilson_interval(b_failures, b_shots)
            c_low, c_high = wilson_interval(c_failures, c_shots)
            reduction, reduction_se, reduction_low, reduction_high = relative_reduction(
                b_failures, b_shots, c_failures, c_shots
            )
            b_time = float(b_stats["total_time_seconds"])
            c_time = float(c_stats["total_time_seconds"])
            b_wall = baseline["wall_time_seconds"]
            c_wall = candidate["wall_time_seconds"]
            rows.append(
                {
                    "case_slug": case,
                    "case_label": baseline["case_label"],
                    "order": order,
                    "circuit": baseline["circuit"],
                    "circuit_sha256": baseline["circuit_sha256"],
                    "sample_seed": baseline["sample_seed"],
                    "det_order_seed": baseline["det_order_seed"],
                    "threads": baseline["threads"],
                    "baseline_commit": baseline["source_commit"],
                    "candidate_commit": candidate["source_commit"],
                    "baseline_binary_sha256": baseline["binary_sha256"],
                    "candidate_binary_sha256": candidate["binary_sha256"],
                    "baseline_shots": b_shots,
                    "baseline_failures": b_failures,
                    "baseline_failure_rate": b_rate,
                    "baseline_failure_rate_low95": b_low,
                    "baseline_failure_rate_high95": b_high,
                    "candidate_shots": c_shots,
                    "candidate_failures": c_failures,
                    "candidate_failure_rate": c_rate,
                    "candidate_failure_rate_low95": c_low,
                    "candidate_failure_rate_high95": c_high,
                    "relative_error_reduction": reduction,
                    "relative_error_reduction_se": reduction_se,
                    "relative_error_reduction_low95": reduction_low,
                    "relative_error_reduction_high95": reduction_high,
                    "fisher_exact_two_sided_p": fisher_exact_two_sided(
                        b_failures, b_shots, c_failures, c_shots
                    ),
                    "baseline_summed_decode_time_seconds": b_time,
                    "candidate_summed_decode_time_seconds": c_time,
                    "summed_decode_time_speedup": b_time / c_time if c_time else None,
                    "baseline_wall_time_seconds": b_wall,
                    "candidate_wall_time_seconds": c_wall,
                    "wall_time_speedup": (
                        float(b_wall) / float(c_wall)
                        if b_wall is not None and c_wall not in (None, 0)
                        else None
                    ),
                    "baseline_throughput_shots_per_wall_second": (
                        b_shots / float(b_wall) if b_wall not in (None, 0) else None
                    ),
                    "candidate_throughput_shots_per_wall_second": (
                        c_shots / float(c_wall) if c_wall not in (None, 0) else None
                    ),
                }
            )
    return rows


def pct(value: float) -> str:
    return f"{value * 100:.3f}%"


def format_probability(value: float) -> str:
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def format_rate_cell(row: dict[str, object], prefix: str) -> str:
    return (
        f"{row[prefix + '_failures']:,} / {row[prefix + '_shots']:,}<br>"
        f"{pct(row[prefix + '_failure_rate'])} "
        f"[{pct(row[prefix + '_failure_rate_low95'])}, "
        f"{pct(row[prefix + '_failure_rate_high95'])}]"
    )


def format_reduction(row: dict[str, object]) -> str:
    value = row["relative_error_reduction"]
    if value is None:
        return "n/a (zero failures)"
    return (
        f"{value * 100:+.1f}% ± {row['relative_error_reduction_se'] * 100:.1f}%<br>"
        f"95% CI [{row['relative_error_reduction_low95'] * 100:+.1f}%, "
        f"{row['relative_error_reduction_high95'] * 100:+.1f}%]"
    )


def format_timing(row: dict[str, object], prefix: str) -> str:
    decode = row[prefix + "_summed_decode_time_seconds"]
    wall = row[prefix + "_wall_time_seconds"]
    if wall is None:
        return f"{decode:,.1f}s"
    return f"{decode:,.1f}s<br>wall {wall:,.1f}s"


def markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# PR #306 detector-order benchmark",
        "",
        (
            "Failures are `num_errors + num_low_confidence`; intervals are Wilson 95%. "
            "The timing field is summed per-shot decode time, not process CPU time. "
            "Fisher p-values treat the aggregate samples as independent; identical "
            "sample seeds do not provide the per-shot discordance needed for a paired test."
        ),
        "",
        "![Relative error reduction and timing speedup](comparison.svg)",
        "",
        (
            "| Circuit (p=0.002) | Ordering | Baseline failures (95% CI) | "
            "Candidate failures (95% CI) | Relative error reduction | "
            "Fisher p (independent) | Baseline summed decode time | "
            "Candidate summed decode time | Speedup |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    row["case_label"],
                    display_order(row["order"]),
                    format_rate_cell(row, "baseline"),
                    format_rate_cell(row, "candidate"),
                    format_reduction(row),
                    format_probability(row["fisher_exact_two_sided_p"]),
                    format_timing(row, "baseline"),
                    format_timing(row, "candidate"),
                    f"{row['summed_decode_time_speedup']:.2f}×",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Raw inputs and provenance are retained alongside this report.",
            "",
        ]
    )
    return "\n".join(lines)


def svg_plot(rows: list[dict[str, object]]) -> str:
    width, height = 1080, 500
    top = 86
    row_height = 58
    left_label_x = 20
    reduction_x0, reduction_x1 = 300, 690
    speed_x0, speed_x1 = 770, 1050
    reduction_min, reduction_max = -0.30, 0.35
    speed_min, speed_max = 0.90, 1.25

    def map_reduction(value: float) -> float:
        return reduction_x0 + (value - reduction_min) / (reduction_max - reduction_min) * (
            reduction_x1 - reduction_x0
        )

    def map_speed(value: float) -> float:
        clipped = min(speed_max, max(speed_min, value))
        return speed_x0 + (clipped - speed_min) / (speed_max - speed_min) * (
            speed_x1 - speed_x0
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:system-ui,-apple-system,sans-serif;fill:#202124}.title{font-size:17px;font-weight:650}.label{font-size:14px}.tick{font-size:12px;fill:#5f6368}.ci{stroke:#5f6368;stroke-width:2}.point{fill:#1a73e8}.speed{fill:#188038}.guide{stroke:#9aa0a6;stroke-dasharray:5 5}.grid{stroke:#e8eaed}</style>',
        '<text x="20" y="31" class="title">PR #306 detector-order benchmark</text>',
        '<text x="300" y="60" class="title">Relative error reduction (95% CI)</text>',
        '<text x="770" y="60" class="title">Summed decode-time speedup</text>',
    ]
    for tick in (-0.25, 0, 0.25):
        x = map_reduction(tick)
        parts.append(f'<line x1="{x:.1f}" y1="72" x2="{x:.1f}" y2="438" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="465" text-anchor="middle" class="tick">{tick * 100:+.0f}%</text>')
    zero_x = map_reduction(0)
    parts.append(f'<line x1="{zero_x:.1f}" y1="72" x2="{zero_x:.1f}" y2="438" class="guide"/>')
    for tick in (0.9, 1.0, 1.1, 1.2):
        x = map_speed(tick)
        parts.append(f'<line x1="{x:.1f}" y1="72" x2="{x:.1f}" y2="438" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="465" text-anchor="middle" class="tick">{tick:.1f}×</text>')
    one_x = map_speed(1)
    parts.append(f'<line x1="{one_x:.1f}" y1="72" x2="{one_x:.1f}" y2="438" class="guide"/>')

    for index, row in enumerate(rows):
        y = top + index * row_height
        label = f"{row['case_label']} · {display_order(row['order'])}"
        parts.append(f'<text x="{left_label_x}" y="{y + 5}" class="label">{html.escape(label)}</text>')
        reduction = row["relative_error_reduction"]
        if reduction is not None:
            low = map_reduction(row["relative_error_reduction_low95"])
            high = map_reduction(row["relative_error_reduction_high95"])
            point = map_reduction(reduction)
            parts.append(f'<line x1="{low:.1f}" y1="{y}" x2="{high:.1f}" y2="{y}" class="ci"/>')
            parts.append(f'<line x1="{low:.1f}" y1="{y - 5}" x2="{low:.1f}" y2="{y + 5}" class="ci"/>')
            parts.append(f'<line x1="{high:.1f}" y1="{y - 5}" x2="{high:.1f}" y2="{y + 5}" class="ci"/>')
            parts.append(f'<circle cx="{point:.1f}" cy="{y}" r="5" class="point"/>')
        speed = row["summed_decode_time_speedup"]
        speed_x = map_speed(speed)
        parts.append(f'<circle cx="{speed_x:.1f}" cy="{y}" r="5" class="speed"/>')
        parts.append(f'<text x="{min(speed_x + 9, speed_x1 - 2):.1f}" y="{y + 4}" class="tick">{speed:.2f}×</text>')
    parts.append('<text x="20" y="490" class="tick">Intervals use aggregate independent-binomial relative-risk math; timing points have no replicate interval.</text>')
    parts.append("</svg>\n")
    return "\n".join(parts)


def write_outputs(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            )
    (output_dir / "results.md").write_text(markdown(rows), encoding="utf-8")
    (output_dir / "comparison.svg").write_text(svg_plot(rows), encoding="utf-8")


def main() -> int:
    args = parse_args()
    records = read_records(args.input_dir)
    rows = make_comparisons(records)
    write_outputs(rows, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
