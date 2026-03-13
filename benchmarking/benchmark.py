#!/usr/bin/env python3.13
"""
Tesseract Decoder Benchmarker

This script automates the process of benchmarking the Tesseract decoder using hyperfine.
It compares the performance of your current working directory against a baseline revision.

Basic Usage:
    Run the benchmarker with default settings (compares current directory against 'main'):
    $ ./benchmarking/benchmark.py

    Run a quick benchmark (minimal shots and runs, useful for sanity checking before a long run):
    $ ./benchmarking/benchmark.py -q

    Compare against a specific baseline revision (e.g., a specific commit or branch):
    $ ./benchmarking/benchmark.py -b my-feature-branch

    Filter circuits by group name (e.g., only run 'surface_code' circuits) See circuits.json for available groups:
    $ ./benchmarking/benchmark.py -g surface_code

Benchmarking Multiple Changes:
    You can benchmark multiple working directories simultaneously against the baseline.
    This is useful if you have several different implementations across different
    directories that you want to compare side-by-side in a single run.
    
    To set up additional directories for your changes:
    - Using git: Create a new worktree.
      $ git worktree add ../path-to-experiment1 <branch-or-commit>
    - Using jj (jujutsu): Add a new workspace.
      $ jj workspace add ../path-to-experiment1 -r <revision>
    
    Use the -d or --dir flag for each additional directory you want to include:
    $ ./benchmarking/benchmark.py -d ../path-to-experiment1 -d ../path-to-experiment2
    
    You can also provide a label for the plot by using the format label=path:
    $ ./benchmarking/benchmark.py -d "experiment1=../path-to-experiment1"
    
    This will benchmark the baseline, the current working directory, and the two 
    extra directories specified, providing a single cohesive report.

Command Line Flags:
    -b, --baseline <rev> : Specify baseline revision (default: main). Can be a branch or commit.
    -d, --dir <lbl=path> : Add extra working directories to benchmark against. Format: path or label=path. Can be specified multiple times.
    -q, --quick          : Enable quick mode (fewer shots, warmup rounds, and runs). Useful for testing.
    -g, --group <name>   : Filter circuits to benchmark by group name (e.g. 'surface_code').
    --skip-build         : Skip the bazel build step (assuming binaries are already built).
    --loop               : Continuously loop the benchmarks. Take a step away from your computer, and grab a Nuka Cola.
    --shots <num>        : Override the default sample-num-shots (default: 5000). Mutually exclusive with -q.
    --warmup <num>       : Override the default warmup-rounds (default: 15). Mutually exclusive with -q.
    --runs <num>         : Override the default num-runs (default: 50). Mutually exclusive with -q.
"""

import argparse
import contextlib
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import plotting
import workspace

# Configure logging with LA timezone
class Formatter(logging.Formatter):
    def converter(self, timestamp):
        dt = datetime.fromtimestamp(timestamp, tz=ZoneInfo('America/Los_Angeles'))
        return dt.timetuple()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)

def print_batch_summary(json_output_files: list[Path], circuit_names: list[str]) -> None:
    logger.info("===================================================")
    logger.info(">>> BATCH RUN SUMMARY")
    logger.info("===================================================")

    for json_file, c_name in zip(json_output_files, circuit_names):
        if Path(json_file).exists():
            try:
                with open(json_file, 'r') as f:
                    results_data = json.load(f)
                
                results_list = results_data.get('results', [])
                if len(results_list) >= 2:
                    baseline_mean = results_list[0].get('mean')
                    pwd_mean = results_list[1].get('mean')

                    if baseline_mean is not None and pwd_mean is not None and pwd_mean > 0:
                        speedup = baseline_mean / pwd_mean
                        logger.info(f"Circuit: {c_name}")
                        logger.info(f"  Baseline Mean: {baseline_mean:.4f} s")
                        logger.info(f"  PWD Mean:      {pwd_mean:.4f} s")
                        logger.info(f"  Speedup:       {speedup:.4f}x")
                        logger.info("---------------------------------------------------")
            except Exception as e:
                logger.error(f"Failed to parse or summarize {json_file}: {e}")

def run_benchmark_batch(args: argparse.Namespace, workspaces: list[str | Path], workspace_names: list[str]) -> None:
    logger.info("===================================================")
    logger.info(">>> STARTING NEW BATCH RUN SEQUENCE")
    logger.info("===================================================")


    if args.quick:
        logger.info(f">>> Quick mode enabled: Reduced shots ({args.sample_num_shots}), warmup ({args.warmup_rounds}), and runs ({args.num_runs}).")

    la_tz = ZoneInfo('America/Los_Angeles')
    timestamp = datetime.now(la_tz).strftime('%Y-%m-%d_%H_%M')
    result_dir = Path(f"benchmarking/results/{timestamp}_{args.num_runs}")
    
    logger.info(f">>> Output directory: {result_dir}")
    (result_dir / "benchmark_json").mkdir(parents=True, exist_ok=True)
    (result_dir / "benchmark_whiskers").mkdir(parents=True, exist_ok=True)

    try:
        with open("benchmarking/circuits.json", 'r') as f:
            circuits_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load circuits JSON: {e}")
        sys.exit(1)

    if args.group:
        logger.info(f">>> Filtering circuits by group: {args.group}")
        circuits = [c for c in circuits_data if c.get('group') == args.group]
    else:
        circuits = circuits_data

    json_output_files = []
    circuit_names = []

    tesseract_args = [
        "--sample-num-shots", str(args.sample_num_shots),
        "--print-stats", "--threads", "48", "--beam", "5", 
        "--no-revisit-dets", "--num-det-orders", "1", 
        "--pqlimit", "100000", "--sample-seed", "123456"
    ]

    for circuit in circuits:
        c_name = circuit['name']
        c_path = circuit['path']
        
        json_file = result_dir / "benchmark_json" / f"results_{c_name}.json"
        whisker_file = result_dir / "benchmark_whiskers" / f"results_{c_name}.png"

        json_output_files.append(json_file)
        circuit_names.append(c_name)

        logger.info("---------------------------------------------------")
        logger.info(f">>> BENCHMARKING CIRCUIT: {c_name}")
        logger.info(f">>> Path: {c_path}")

        hyperfine_cmd = [
            "hyperfine",
            "--warmup", str(args.warmup_rounds),
            "--runs", str(args.num_runs),
            "--export-json", str(json_file)
        ]

        for name, d in zip(workspace_names, workspaces):
            hyperfine_cmd.extend(["-n", name])
            
            binary_path = Path(d) / "bazel-bin" / "src" / "tesseract"
            if str(d) == ".":
               binary_path = Path("bazel-bin") / "src" / "tesseract"
            
            cmd_for_binary = f"{binary_path} --circuit '{c_path}' " + " ".join(tesseract_args)
            hyperfine_cmd.append(cmd_for_binary)

        workspace.run_cmd(hyperfine_cmd)

        plotting.plot_benchmark_results(json_file=str(json_file), labels=workspace_names, output_file=str(whisker_file))

    print_batch_summary(json_output_files, circuit_names)
    logger.info(f">>> Batch Run Complete! Results saved in: {result_dir}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark tesseract decoder using hyperfine.")
    parser.add_argument("-b", "--baseline", default="main", help="Specify baseline revision (default: main)")
    parser.add_argument("-d", "--dir", action="append", default=[], help="Add extra working directories to benchmark against. Format: path or label=path. Can be specified multiple times.")
    parser.add_argument("--skip-build", action="store_true", help="Skip the bazel build step")
    parser.add_argument("--loop", action="store_true", help="Loop runs rather than running once.")

    parser.add_argument("-q", "--quick", action="store_true", help="Enable quick mode (fewer shots/runs)")
    parser.add_argument("-g", "--group", default="", help="Filter circuits by group name")
    parser.add_argument("--shots", type=int, default=5000, help="Override the default sample-num-shots (mutually exclusive with -q)")
    parser.add_argument("--warmup", type=int, default=15, help="Override the default warmup-rounds (mutually exclusive with -q)")
    parser.add_argument("--runs", type=int, default=50, help="Override the default num-runs (mutually exclusive with -q)")
    
    args = parser.parse_args()

    if args.quick and (args.shots != 5000 or args.warmup != 15 or args.runs != 50):
        parser.error("-q/--quick cannot be used with --shots, --warmup, or --runs")

    args.sample_num_shots = 500 if args.quick else args.shots
    args.warmup_rounds = 1 if args.quick else args.warmup
    args.num_runs = 2 if args.quick else args.runs

    baseline_dir = "../baseline_bench_tmp"
    vcs = workspace.check_vcs()
    if not vcs:
        logger.error("Error: Neither a jj nor git repository detected.")
        sys.exit(1)
    with workspace.managed_baseline(baseline_dir, args.baseline, vcs):
        extra_workspaces = []
        extra_names = []
        for d in args.dir:
            if '=' in d:
                lbl, pth = d.split('=', 1)
                extra_names.append(lbl)
                extra_workspaces.append(pth)
            else:
                extra_names.append(Path(d).name)
                extra_workspaces.append(d)

        workspaces = [baseline_dir, "."] + extra_workspaces
        workspace_names = ["baseline", "pwd"] + extra_names

        workspace.build_all(workspaces, args.skip_build)

        if args.loop:
            while True:
                run_benchmark_batch(args, workspaces, workspace_names)
                logger.info(">>> Restarting in 5 seconds... (Press Ctrl+C to stop)")
                time.sleep(5)
                workspace.build_all(workspaces, args.skip_build)
        else:
            run_benchmark_batch(args, workspaces, workspace_names)

if __name__ == "__main__":
    main()
