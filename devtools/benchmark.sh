#!/usr/bin/env bash
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${WORKSPACE_ROOT}"

# CL / Experiment tag (default: baseline)
CL_TAG="${1:-baseline}"
OUT_DIR="${WORKSPACE_ROOT}/out/${CL_TAG}"
mkdir -p "${OUT_DIR}"

THREADS="${THREADS:-48}"
SAMPLE_SEED="${SAMPLE_SEED:-1234}"

echo "============================================================"
echo " Running Tesseract-BP Benchmarks for [${CL_TAG}]"
echo " Output directory: ${OUT_DIR}"
echo " Threads: ${THREADS}"
echo "============================================================"

# Build binaries using Bazel (single-core for build as per rules)
echo "=== Building C++ Binaries (bazel build -c opt --jobs=1 src:bp src:tesseract) ==="
bazel build -c opt --jobs=1 src:bp src:tesseract

BP_BIN="${WORKSPACE_ROOT}/bazel-bin/src/bp"
TESSERACT_BIN="${WORKSPACE_ROOT}/bazel-bin/src/tesseract"

run_bp_benchmark() {
    local name="$1"
    local circuit="$2"
    local shots="${3:-10000}"
    local norm="${4:-0.75}"
    local osd_order="${5:--1}"
    local osd_weight="${6:-0}"
    local max_iter="${7:-30}"
    local max_errors="${8:-100}"
    local schedule="${9:-serial}"

    if [[ ! -f "${circuit}" ]]; then
        echo "[SKIP] Circuit not found: ${circuit}"
        return 0
    fi

    local out_json="${OUT_DIR}/${name}.json"
    echo ""
    echo "------------------------------------------------------------"
    echo ">> Running: ${name}"
    echo "   Circuit: ${circuit}"
    echo "   Config: schedule=${schedule}, batched=true, osd_order=${osd_order}, osd_weight=${osd_weight}, norm=${norm}"
    echo "------------------------------------------------------------"

    local cmd=(
        "${BP_BIN}"
        --circuit "${circuit}"
        --sample-num-shots "${shots}"
        --sample-seed "${SAMPLE_SEED}"
        --threads "${THREADS}"
        --max-errors "${max_errors}"
        --normalization-factor "${norm}"
        --max-iter "${max_iter}"
        --schedule "${schedule}"
        --batched
        --print-stats
        --stats-out "${out_json}"
    )

    if [[ "${osd_order}" -ge 0 ]]; then
        cmd+=(--osd-order "${osd_order}" --osd-weight "${osd_weight}")
    fi

    "${cmd[@]}"
}

run_tesseract_benchmark() {
    local name="$1"
    local circuit="$2"
    local shots="${3:-10000}"
    local max_errors="${4:-100}"

    if [[ ! -f "${circuit}" ]]; then
        echo "[SKIP] Circuit not found: ${circuit}"
        return 0
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo ">> Running Tesseract: ${name}"
    echo "   Circuit: ${circuit}"
    echo "------------------------------------------------------------"

    local cmd=(
        "${TESSERACT_BIN}"
        --circuit "${circuit}"
        --sample-num-shots "${shots}"
        --sample-seed "${SAMPLE_SEED}"
        --threads "${THREADS}"
        --max-errors "${max_errors}"
        --no-revisit-dets
        --beam 20
        --beam-climbing
        --num-det-orders 21
        --det-order-index
        --pqlimit 1000000
        --print-stats
    )

    "${cmd[@]}"
}

# ==============================================================================
# 1. Surface Codes (d=3, d=5, d=7, d=9)
# ==============================================================================
run_bp_benchmark \
    "surface_code_d3_p001_serial_batched" \
    "testdata/surfacecodes/r=3,d=3,p=0.001,noise=si1000,c=surface_code_Z,q=17,gates=cz.stim" \
    100000 \
    0.625

run_bp_benchmark \
    "surface_code_d3_p001_parallel_batched" \
    "testdata/surfacecodes/r=3,d=3,p=0.001,noise=si1000,c=surface_code_Z,q=17,gates=cz.stim" \
    100000 \
    0.625 \
    -1 0 30 100 "parallel"

run_bp_benchmark \
    "surface_code_d5_p001_serial_batched" \
    "testdata/surfacecodes/r=5,d=5,p=0.001,noise=si1000,c=surface_code_Z,q=49,gates=cz.stim" \
    100000 \
    0.625

run_bp_benchmark \
    "surface_code_d5_p001_parallel_batched" \
    "testdata/surfacecodes/r=5,d=5,p=0.001,noise=si1000,c=surface_code_Z,q=49,gates=cz.stim" \
    100000 \
    0.625 \
    -1 0 30 100 "parallel"

run_bp_benchmark \
    "surface_code_d7_p001_serial_batched" \
    "testdata/surfacecodes/r=7,d=7,p=0.001,noise=si1000,c=surface_code_Z,q=97,gates=cz.stim" \
    50000 \
    0.625

run_bp_benchmark \
    "surface_code_d9_p001_serial_batched" \
    "testdata/surfacecodes/r=9,d=9,p=0.001,noise=si1000,c=surface_code_Z,q=161,gates=cz.stim" \
    20000 \
    0.625

# ==============================================================================
# 2. Color Codes (d=5, d=7)
# ==============================================================================
run_bp_benchmark \
    "color_code_d5_superdense_serial_batched" \
    "testdata/colorcodes/r=5,d=5,p=0.001,noise=si1000,c=superdense_color_code_Z,q=37,gates=cz.stim" \
    50000 \
    0.9063

run_bp_benchmark \
    "color_code_d7_superdense_serial_batched" \
    "testdata/colorcodes/r=7,d=7,p=0.001,noise=si1000,c=superdense_color_code_Z,q=73,gates=cz.stim" \
    20000 \
    0.9063

# ==============================================================================
# 3. Bivariate Bicycle Codes (from testdata)
# ==============================================================================
run_bp_benchmark \
    "bb_72_12_6_serial_batched_osd0" \
    "testdata/bivariatebicyclecodes/r=6,d=6,p=0.001,noise=si1000,c=bivariate_bicycle_Z,nkd=[[72,12,6]],q=144,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim" \
    10000 \
    0.75 \
    10000 \
    0 \
    30

run_bp_benchmark \
    "bb_72_12_6_serial_batched_osd1" \
    "testdata/bivariatebicyclecodes/r=6,d=6,p=0.001,noise=si1000,c=bivariate_bicycle_Z,nkd=[[72,12,6]],q=144,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim" \
    5000 \
    0.675 \
    10000 \
    1 \
    30

run_bp_benchmark \
    "bb_90_8_10_serial_batched_osd0" \
    "testdata/bivariatebicyclecodes/r=10,d=10,p=0.001,noise=si1000,c=bivariate_bicycle_Z,nkd=[[90,8,10]],q=180,iscolored=True,A_poly=x^9+y+y^2,B_poly=x^7+1+x^2.stim" \
    10000 \
    0.75 \
    10000 \
    0 \
    30

run_bp_benchmark \
    "bb_108_8_10_serial_batched_osd0" \
    "testdata/bivariatebicyclecodes/r=10,d=10,p=0.001,noise=si1000,c=bivariate_bicycle_Z,nkd=[[108,8,10]],q=216,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim" \
    10000 \
    0.75 \
    10000 \
    0 \
    30

run_bp_benchmark \
    "bb_144_12_12_serial_batched_osd0" \
    "testdata/bivariatebicyclecodes/r=12,d=12,p=0.001,noise=si1000,c=bivariate_bicycle_Z,nkd=[[144,12,12]],q=288,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim" \
    10000 \
    0.75 \
    10000 \
    0 \
    30

# ==============================================================================
# 4. High-Rate / CPM Codes (if present in benchmarking/hrcodes/)
# ==============================================================================
run_bp_benchmark \
    "bb_z_onebasis_serial_batched_osd0" \
    "benchmarking/hrcodes/traincodes/circuits/r=6,d=6,p=0.001,noise=si1000,c=bivariate_bicycle_Z_onebasis.stim" \
    100000 \
    0.75 \
    10000 \
    0 \
    30

run_bp_benchmark \
    "cpm348_serial_batched_hard" \
    "benchmarking/hrcodes/traincodes/circuits/cpm348_p_1e-3_Z_seed_470.stim" \
    1000 \
    0.9063

run_bp_benchmark \
    "cpm564_serial_batched_osd1" \
    "benchmarking/hrcodes/traincodes/circuits/cpm564_p_1e-3_Z_seed_55_64ops.stim" \
    100000 \
    0.75 \
    1000 \
    1 \
    1000

run_tesseract_benchmark \
    "cpm348_tesseract_beam" \
    "benchmarking/hrcodes/traincodes/circuits/cpm348_p_1e-3_Z_seed_470_64ops.stim" \
    100000

run_bp_benchmark \
    "cpm348_serial_batched_osd1" \
    "benchmarking/hrcodes/traincodes/circuits/cpm348_p_1e-3_Z_seed_470.stim" \
    100000 \
    0.75 \
    1000 \
    1 \
    100

echo ""
echo "============================================================"
echo " Summary of results in ${OUT_DIR}:"
echo "============================================================"

python3 - <<EOF
import os, glob, json

out_dir = "${OUT_DIR}"
cl_tag = "${CL_TAG}"
json_files = sorted(glob.glob(os.path.join(out_dir, "*.json")))

if not json_files:
    print("No JSON files found in " + out_dir)
else:
    print(f"| {'Benchmark':<42} | {'Decoder':<22} | {'Shots':>8} | {'Errors':>8} | {'Wall (s)':>9} | {'CPU (s)':>9} | {'Shots/sec':>12} | {'LER':>10} |")
    print(f"|:{'-'*42}-|-{'-'*22}-|-{'-'*8}:|-{'-'*8}:|-{'-'*9}:|-{'-'*9}:|-{'-'*12}:|-{'-'*10}:|")
    for jf in json_files:
        name = os.path.splitext(os.path.basename(jf))[0]
        try:
            with open(jf, "r") as f:
                d = json.load(f)
            shots = d.get("num_shots", 0)
            errors = d.get("num_errors", 0)
            errors_str = str(errors) if errors is not None else "N/A"
            wall_t = d.get("wall_time_seconds", d.get("total_time_seconds", 0.0))
            cpu_t = d.get("cpu_time_seconds", d.get("total_time_seconds", 0.0))
            th = d.get("shots_per_second", (shots / wall_t) if wall_t > 0 else 0.0)
            ler_str = f"{(errors / shots):.5f}" if (errors is not None and shots > 0) else "N/A"
            dec = d.get("decoder", "N/A")
            print(f"| {name:<42} | {dec:<22} | {shots:>8} | {errors_str:>8} | {wall_t:>9.3f} | {cpu_t:>9.3f} | {th:>12,.1f} | {ler_str:>10} |")
        except Exception as e:
            print(f"| {name:<42} | Error reading JSON: {e}")
print()
EOF
