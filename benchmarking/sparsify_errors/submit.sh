#!/usr/bin/env bash
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

# Exit immediately if a command exits with a non-zero status (-e),
# treat unset variables as an error (-u), and catch errors in pipes (-o pipefail).
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
OUT_DIR="$SCRIPT_DIR/out"
cd "$REPO_ROOT"

TESSERACT_BIN="$REPO_ROOT/bazel-bin/src/tesseract"
: "${BENCHMARK_HARDWARE_DESCRIPTION:?Set this to the CPU and cluster/host description}"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to benchmark a dirty checkout." >&2
  exit 1
fi
TESSERACT_COMMIT=$(git rev-parse HEAD)

# A successful Bazel build binds the snapshotted executable below to the clean
# source checkout instead of trusting a possibly stale bazel-bin artifact.
bazel build --jobs=1 src:tesseract
if [[ ! -x "$TESSERACT_BIN" ]]; then
  echo "Bazel did not produce $TESSERACT_BIN." >&2
  exit 1
fi

REPETITIONS_PER_CONFIGURATION=1000
EXPECTED_CIRCUIT_COUNT=56
SPARSIFY_REACTIVATE_LIMITS=(0 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192)
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
SAMPLE_SEED_SCHEME="run_namespace_times_stride_plus_job_index"
SAMPLE_SEED_STRIDE=1048576
SAMPLE_SEED_NAMESPACE=$(python3 -c \
  'import hashlib, sys; print(int.from_bytes(hashlib.sha256(sys.argv[1].encode()).digest()[:5], "big"))' \
  "$RUN_ID")
RUN_DIR="$OUT_DIR/$RUN_ID"
JOBS_DIR="$RUN_DIR/jobs"
CIRCUIT_LIST="$RUN_DIR/circuits.txt"
SNAPSHOT_DIR="$RUN_DIR/artifacts"
SNAPSHOT_REPO="$SNAPSHOT_DIR/repo"
SNAPSHOT_TESSERACT_BIN="$SNAPSHOT_DIR/tesseract"
mkdir -p "$JOBS_DIR" "$SNAPSHOT_REPO"

shopt -s nullglob
for p_err in 0.001 0.002; do
  for circuit in testdata/bivariatebicyclecodes/r={6,10,12}*p=$p_err,noise=si1000,c=*.stim; do
    echo "$circuit"
  done
  for circuit in testdata/colorcodes/r={3,5,7,9,11}*p=$p_err,noise=si1000,c=superdense_color_code_*.stim; do
    echo "$circuit"
  done
  for circuit in testdata/surfacecodes/r={3,5,7,9,11}*p=$p_err,noise=si1000,c=surface_code_*.stim; do
    echo "$circuit"
  done
done | sort -u > "$CIRCUIT_LIST"

if [[ ! -s "$CIRCUIT_LIST" ]]; then
  echo "No benchmark circuits matched." >&2
  exit 1
fi
NUM_CIRCUITS=$(wc -l < "$CIRCUIT_LIST" | tr -d ' ')
if [[ "$NUM_CIRCUITS" -ne "$EXPECTED_CIRCUIT_COUNT" ]]; then
  echo "Expected $EXPECTED_CIRCUIT_COUNT benchmark circuits; found $NUM_CIRCUITS." >&2
  exit 1
fi

install -m 0555 "$TESSERACT_BIN" "$SNAPSHOT_TESSERACT_BIN"
while IFS= read -r circuit; do
  snapshot_circuit="$SNAPSHOT_REPO/$circuit"
  mkdir -p "$(dirname "$snapshot_circuit")"
  install -m 0444 "$circuit" "$snapshot_circuit"
done < "$CIRCUIT_LIST"

EXPECTED_JOB_COUNT=$((
  NUM_CIRCUITS
  * (1 + ${#SPARSIFY_REACTIVATE_LIMITS[@]})
  * REPETITIONS_PER_CONFIGURATION
))
if [[ "$EXPECTED_JOB_COUNT" -gt "$SAMPLE_SEED_STRIDE" ]]; then
  echo "Expected job count exceeds the configured sample-seed stride." >&2
  exit 1
fi
printf -v REACTIVATE_LIMITS_CSV '%s,' "${SPARSIFY_REACTIVATE_LIMITS[@]}"
REACTIVATE_LIMITS_CSV=${REACTIVATE_LIMITS_CSV%,}

python3 - "$RUN_DIR/manifest.json" "$RUN_ID" "$BENCHMARK_HARDWARE_DESCRIPTION" \
  "$TESSERACT_COMMIT" "$SNAPSHOT_TESSERACT_BIN" "$SNAPSHOT_REPO" "$CIRCUIT_LIST" \
  "$REPETITIONS_PER_CONFIGURATION" "$REACTIVATE_LIMITS_CSV" \
  "$EXPECTED_JOB_COUNT" "$SAMPLE_SEED_NAMESPACE" "$SAMPLE_SEED_SCHEME" \
  "$SAMPLE_SEED_STRIDE" <<'PY'
import datetime
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
run_id = sys.argv[2]
hardware = sys.argv[3]
expected_tesseract_commit = sys.argv[4]
binary_path = Path(sys.argv[5])
snapshot_repo = Path(sys.argv[6])
circuit_list_path = Path(sys.argv[7])
repetitions_per_configuration = int(sys.argv[8])
reactivate_limits = [int(value) for value in sys.argv[9].split(",")]
expected_job_count = int(sys.argv[10])
sample_seed_namespace = int(sys.argv[11])
sample_seed_scheme = sys.argv[12]
sample_seed_stride = int(sys.argv[13])

status = subprocess.run(
    ["git", "status", "--porcelain"],
    check=True,
    capture_output=True,
    text=True,
).stdout
if status:
    raise SystemExit("Refusing to benchmark a dirty checkout.")
tesseract_commit = subprocess.run(
    ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
).stdout.strip()
if tesseract_commit != expected_tesseract_commit:
    raise SystemExit("Checkout changed while benchmark artifacts were being built.")
module_text = Path("MODULE.bazel").read_text(encoding="utf-8")
stim_match = re.search(
    r'git_repository\(\s*name = "stim",\s*commit = "([0-9a-f]{40})"',
    module_text,
    re.DOTALL,
)
if stim_match is None:
    raise SystemExit("Could not determine the pinned Stim revision from MODULE.bazel.")

circuits = {}
for circuit_path in circuit_list_path.read_text(encoding="utf-8").splitlines():
    snapshot_path = snapshot_repo / circuit_path
    circuits[circuit_path] = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()

manifest = {
    "schema_version": 1,
    "run_id": run_id,
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "tesseract_commit": tesseract_commit,
    "stim_revision": stim_match.group(1),
    "hardware_description": hardware,
    "tesseract_binary_sha256": hashlib.sha256(binary_path.read_bytes()).hexdigest(),
    "git_dirty": False,
    "det_order_method": "index",
    "merge_errors": True,
    "circuit_sha256": circuits,
    "expected_job_count": expected_job_count,
    "sample_seed_namespace": sample_seed_namespace,
    "sample_seed_scheme": sample_seed_scheme,
    "sample_seed_stride": sample_seed_stride,
    "sweep": {
        "include_baseline": True,
        "repetitions_per_configuration": repetitions_per_configuration,
        "sparsify_base_degree_by_directory": {
            "bivariatebicyclecodes": 3,
            "colorcodes": 3,
            "surfacecodes": 2,
        },
        "sparsify_max_degree": -1,
        "sparsify_reactivate_limits": reactivate_limits,
    },
}
manifest_path.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

COUNTER=0

for num in $(seq 1 "$REPETITIONS_PER_CONFIGURATION"); do
  cat "$CIRCUIT_LIST"
done | shuf | while IFS= read -r circuit; do

  # Determine base degree based on the folder/filename
  if [[ "$circuit" == *"surfacecodes"* ]]; then
    SPARSIFY_BASE_DEGREE=2
  else
    SPARSIFY_BASE_DEGREE=3
  fi

  # Iterate through the requested reactivate limits
  for SPARSIFY_REACTIVATE_LIMIT in "${SPARSIFY_REACTIVATE_LIMITS[@]}"; do
    SAMPLE_SEED=$((SAMPLE_SEED_NAMESPACE * SAMPLE_SEED_STRIDE + COUNTER))
    echo "Submitting: $circuit | Degree: $SPARSIFY_BASE_DEGREE | Limit: $SPARSIFY_REACTIVATE_LIMIT"

    sbatch --partition=c2 --job-name=None4u \
            --ntasks=1 \
            --mem=120gb \
            --cpus-per-task=60 \
            --time=200:00:00 \
            --wrap="cd \"$SNAPSHOT_REPO\" && \"$SNAPSHOT_TESSERACT_BIN\" --circuit \"$circuit\" --sample-num-shots 10000 --max-errors 10 --sample-seed $SAMPLE_SEED --threads 30 --no-revisit-dets --beam 20 --beam-climbing --num-det-orders 21 --det-order-index --pqlimit 1000000 --sparsify-errors --sparsify-base-degree $SPARSIFY_BASE_DEGREE --sparsify-reactivate-limit $SPARSIFY_REACTIVATE_LIMIT --stats-out \"$JOBS_DIR/${COUNTER}.json\""

    # Increment counter for every single job so JSON files don't get overwritten
    COUNTER=$((COUNTER + 1))
  done

  # Submit also one baseline job
  SAMPLE_SEED=$((SAMPLE_SEED_NAMESPACE * SAMPLE_SEED_STRIDE + COUNTER))
  sbatch --partition=c2 --job-name=None4u \
          --ntasks=1 \
          --mem=120gb \
          --cpus-per-task=60 \
          --time=200:00:00 \
          --wrap="cd \"$SNAPSHOT_REPO\" && \"$SNAPSHOT_TESSERACT_BIN\" --circuit \"$circuit\" --sample-num-shots 10000 --max-errors 10 --sample-seed $SAMPLE_SEED --threads 30 --no-revisit-dets --beam 20 --beam-climbing --num-det-orders 21 --det-order-index --pqlimit 1000000 --stats-out \"$JOBS_DIR/${COUNTER}.json\""
  # Increment counter for every single job so JSON files don't get overwritten
  COUNTER=$((COUNTER + 1))
done

echo "Submitted benchmark run $RUN_ID; manifest and jobs are under $RUN_DIR"
