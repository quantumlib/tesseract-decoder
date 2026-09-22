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

"""Build pinned revisions, run PR #306 benchmarks, and render their report."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import time


BASELINE_REF = "4fd104d88205b9c5424c94b2da28e684c113ebcf"
CANDIDATE_REF = "9d62d5d317892c706c73d899dae87644911f199f"
STIM_REVISION_PATTERN = 'commit = "'

CASES = (
    {
        "slug": "surface-d7",
        "label": "Surface code d=7",
        "base_degree": 2,
        "circuit": (
            "testdata/surfacecodes/"
            "r=7,d=7,p=0.002,noise=si1000,c=surface_code_X,q=97,gates=cz.stim"
        ),
    },
    {
        "slug": "color-d7",
        "label": "Color code d=7",
        "base_degree": 3,
        "circuit": (
            "testdata/colorcodes/"
            "r=7,d=7,p=0.002,noise=si1000,c=superdense_color_code_X,"
            "q=73,gates=cz.stim"
        ),
    },
    {
        "slug": "bb-72-12-6",
        "label": "BB [[72,12,6]]",
        "base_degree": 3,
        "circuit": (
            "testdata/bivariatebicyclecodes/"
            "r=6,d=6,p=0.002,noise=si1000,c=bivariate_bicycle_X,"
            "nkd=[[72,12,6]],q=144,iscolored=True,A_poly=x^3+y+y^2,"
            "B_poly=y^3+x+x^2.stim"
        ),
    },
)

ORDERS = (
    ("bfs", "--det-order-bfs"),
    ("coordinate", "--det-order-coordinate"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardware-description", required=True)
    parser.add_argument("--bazel", default="bazel")
    parser.add_argument("--baseline-ref", default=BASELINE_REF)
    parser.add_argument("--candidate-ref", default=CANDIDATE_REF)
    parser.add_argument("--baseline-bin", type=Path)
    parser.add_argument("--candidate-bin", type=Path)
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--sample-seed", type=int, default=1_234_567)
    parser.add_argument("--det-order-seed", type=int, default=518_278_944)
    parser.add_argument("--pqlimit", type=int, default=100_000)
    return parser.parse_args()


def run_checked(
    command: list[str],
    *,
    cwd: Path,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=capture_output,
        text=True,
    )


def resolve_commit(repo_root: Path, ref: str) -> str:
    return run_checked(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=repo_root
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_file_bytes(repo_root: Path, commit: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return result.stdout


def extract_source(repo_root: Path, commit: str, destination: Path) -> None:
    destination.mkdir(parents=True)
    process = subprocess.Popen(
        ["git", "archive", "--format=tar", commit],
        cwd=repo_root,
        stdout=subprocess.PIPE,
    )
    assert process.stdout is not None
    with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
        archive.extractall(destination, filter="data")
    if process.wait() != 0:
        raise subprocess.CalledProcessError(process.returncode, process.args)


def stim_revision(source_root: Path) -> str:
    module = (source_root / "MODULE.bazel").read_text(encoding="utf-8")
    marker = 'name = "stim"'
    start = module.find(marker)
    if start < 0:
        raise ValueError("MODULE.bazel has no Stim repository")
    commit_start = module.find(STIM_REVISION_PATTERN, start)
    if commit_start < 0:
        raise ValueError("Stim repository has no pinned commit")
    commit_start += len(STIM_REVISION_PATTERN)
    commit_end = module.find('"', commit_start)
    revision = module[commit_start:commit_end]
    if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
        raise ValueError("Stim revision is not a full hexadecimal commit")
    return revision


def build_binary(source_root: Path, bazel_root: Path, bazel: str) -> Path:
    command = [
        bazel,
        f"--output_user_root={bazel_root}",
        "build",
        "-c",
        "opt",
        "//src:tesseract",
    ]
    subprocess.run(command, cwd=source_root, check=True)
    binary = source_root / "bazel-bin/src/tesseract"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"Bazel did not produce {binary}")
    return binary


def copy_binary(source: Path, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source.resolve(), destination)
    destination.chmod(0o555)
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "size_bytes": destination.stat().st_size,
    }


def command_for(
    binary: Path,
    circuit: Path,
    case: dict[str, object],
    order_flag: str,
    args: argparse.Namespace,
    stats_path: Path,
) -> list[str]:
    return [
        str(binary),
        "--circuit",
        str(circuit),
        "--sample-num-shots",
        str(args.shots),
        "--sample-seed",
        str(args.sample_seed),
        "--det-order-seed",
        str(args.det_order_seed),
        "--threads",
        str(args.threads),
        "--beam",
        "5",
        "--beam-climbing",
        "--num-det-orders",
        "5",
        "--no-revisit-dets",
        "--pqlimit",
        str(args.pqlimit),
        "--sparsify-errors",
        "--sparsify-base-degree",
        str(case["base_degree"]),
        order_flag,
        "--stats-out",
        str(stats_path),
    ]


def read_stats(path: Path) -> dict[str, object]:
    stats = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "num_shots",
        "num_errors",
        "num_low_confidence",
        "total_time_seconds",
    }
    missing = required - set(stats)
    if missing:
        raise ValueError(f"{path} is missing native stats {sorted(missing)}")
    return stats


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    args.repo_root = args.repo_root.resolve(strict=True)
    args.output = args.output.resolve()
    if args.output.exists():
        raise SystemExit(f"refusing to reuse existing output directory: {args.output}")
    if args.shots <= 0 or args.threads <= 0 or args.pqlimit <= 0:
        raise SystemExit("shots, threads, and pqlimit must be positive")
    if (args.baseline_bin is None) != (args.candidate_bin is None):
        raise SystemExit("supply both --baseline-bin and --candidate-bin, or neither")

    baseline_commit = resolve_commit(args.repo_root, args.baseline_ref)
    candidate_commit = resolve_commit(args.repo_root, args.candidate_ref)
    output = args.output
    raw_dir = output / "raw"
    artifacts = output / "artifacts"
    sources = output / "sources"
    bazel_roots = output / "bazel"
    raw_dir.mkdir(parents=True)
    artifacts.mkdir()

    circuit_records = {}
    for case in CASES:
        path = str(case["circuit"])
        baseline_bytes = git_file_bytes(args.repo_root, baseline_commit, path)
        candidate_bytes = git_file_bytes(args.repo_root, candidate_commit, path)
        if baseline_bytes != candidate_bytes:
            raise ValueError(f"circuit differs between revisions: {path}")
        destination = artifacts / "circuits" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(candidate_bytes)
        destination.chmod(0o444)
        circuit_records[path] = {
            "sha256": hashlib.sha256(candidate_bytes).hexdigest(),
            "size_bytes": len(candidate_bytes),
        }

    external_binaries = args.baseline_bin is not None
    if external_binaries:
        baseline_source = candidate_source = None
        baseline_built = args.baseline_bin.resolve(strict=True)
        candidate_built = args.candidate_bin.resolve(strict=True)
        stim_revisions = None
    else:
        baseline_source = sources / "baseline"
        candidate_source = sources / "candidate"
        extract_source(args.repo_root, baseline_commit, baseline_source)
        extract_source(args.repo_root, candidate_commit, candidate_source)
        baseline_built = build_binary(
            baseline_source, bazel_roots / "baseline", args.bazel
        )
        candidate_built = build_binary(
            candidate_source, bazel_roots / "candidate", args.bazel
        )
        stim_revisions = {
            "baseline": stim_revision(baseline_source),
            "candidate": stim_revision(candidate_source),
        }

    binaries = {
        "baseline": copy_binary(baseline_built, artifacts / "baseline-tesseract"),
        "candidate": copy_binary(candidate_built, artifacts / "candidate-tesseract"),
    }

    try:
        bazel_version = run_checked(
            [args.bazel, "version"], cwd=args.repo_root
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        bazel_version = None

    manifest = {
        "schema": "tesseract.detector_order_semantics_benchmark",
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "baseline_commit": baseline_commit,
        "candidate_commit": candidate_commit,
        "binary_source_binding": "external-unverified" if external_binaries else "built",
        "binaries": binaries,
        "circuits": circuit_records,
        "stim_revisions": stim_revisions,
        "hardware_description": args.hardware_description,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "bazel_version": bazel_version,
        "configuration": {
            "shots": args.shots,
            "threads": args.threads,
            "sample_seed": args.sample_seed,
            "det_order_seed": args.det_order_seed,
            "pqlimit": args.pqlimit,
            "beam": 5,
            "beam_climbing": True,
            "num_det_orders": 5,
            "no_revisit_dets": True,
            "sparsify_errors": True,
        },
    }
    write_json(output / "manifest.json", manifest)

    for case in CASES:
        circuit_relative = str(case["circuit"])
        circuit = artifacts / "circuits" / circuit_relative
        for order_name, order_flag in ORDERS:
            for revision in ("baseline", "candidate"):
                stem = f"{case['slug']}-{order_name}-{revision}"
                native_path = raw_dir / f"{stem}.native.json"
                stdout_path = raw_dir / f"{stem}.stdout.txt"
                stderr_path = raw_dir / f"{stem}.stderr.txt"
                command = command_for(
                    artifacts / f"{revision}-tesseract",
                    circuit,
                    case,
                    order_flag,
                    args,
                    native_path,
                )
                started = dt.datetime.now(dt.timezone.utc).isoformat()
                before = time.monotonic()
                completed = subprocess.run(
                    command,
                    cwd=output,
                    capture_output=True,
                    text=True,
                )
                wall_time = time.monotonic() - before
                stdout_path.write_text(completed.stdout, encoding="utf-8")
                stderr_path.write_text(completed.stderr, encoding="utf-8")
                if completed.returncode != 0:
                    raise subprocess.CalledProcessError(
                        completed.returncode,
                        command,
                        completed.stdout,
                        completed.stderr,
                    )
                stats = read_stats(native_path)
                record = {
                    "schema": "tesseract.detector_order_semantics_raw",
                    "schema_version": 1,
                    "case_slug": case["slug"],
                    "case_label": case["label"],
                    "circuit": circuit_relative,
                    "circuit_sha256": circuit_records[circuit_relative]["sha256"],
                    "base_degree": case["base_degree"],
                    "order": order_name,
                    "revision": revision,
                    "source_commit": (
                        baseline_commit if revision == "baseline" else candidate_commit
                    ),
                    "binary_sha256": binaries[revision]["sha256"],
                    "command": command,
                    "started_at_utc": started,
                    "wall_time_seconds": wall_time,
                    "sample_seed": args.sample_seed,
                    "det_order_seed": args.det_order_seed,
                    "requested_shots": args.shots,
                    "threads": args.threads,
                    "native_stats": stats,
                    "native_stats_file": native_path.name,
                    "stdout_file": stdout_path.name,
                    "stderr_file": stderr_path.name,
                }
                write_json(raw_dir / f"{stem}.json", record)

    report_script = Path(__file__).with_name("report.py")
    subprocess.run(
        [
            sys.executable,
            str(report_script),
            "--input-dir",
            str(raw_dir),
            "--output-dir",
            str(output),
        ],
        check=True,
    )
    print(f"Benchmark complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
