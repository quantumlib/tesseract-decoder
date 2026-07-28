# Error sparsification benchmarks

This directory preserves the benchmark artifacts associated with
[PR #254](https://github.com/quantumlib/tesseract-decoder/pull/254), which
introduced `--sparsify-errors`.

## Contents

- `submit.sh`: Slurm submission script for the benchmark sweep.
- `aggregate.py`: strict, deterministic aggregation of raw per-job stats.
- `enrich.py`: reconstruction and validation of circuit/DEM metadata for an
  existing aggregate.
- `aggregated_results.jsonl`: self-contained aggregate consumed by the plots.
- `provenance.json`: known provenance and explicit gaps for the historical
  aggregate.
- `plot.py`: statistical analysis and PDF generation.
- `plots/`: PDFs generated from `aggregated_results.jsonl`.

## Re-running jobs

From the repository root:

```bash
export BENCHMARK_HARDWARE_DESCRIPTION="CPU model; host/cluster description"
benchmarking/sparsify_errors/submit.sh
```

The script rebuilds Tesseract with Bazel, reads the exact 56-circuit sweep from
`testdata/`, and submits jobs with `sbatch`. Its Slurm partition, memory, CPU
count, and wall time were tuned for the original cluster and may need
adjustment before reuse. Per-job JSON stats are written under a new
`benchmarking/sparsify_errors/out/<run-id>/jobs/` directory. Submission refuses
a dirty checkout and snapshots the built binary plus every circuit into that
run directory before any jobs are queued. Jobs execute only those immutable
snapshots, rather than live `bazel-bin/` or `testdata/` paths.

The sibling `manifest.json` captures the full Tesseract and Stim revisions,
snapshot hashes, hardware description, start time, expected sweep grid, and
expected job count. The checked-in configuration contains 1,000 repetitions
for each of 15 configurations of each circuit, or 840,000 jobs total. Every
job receives an explicit seed equal to a run-specific namespace times a fixed
stride plus its job index. The manifest records that scheme; aggregation
verifies it and rejects duplicate namespaces when combining runs.

Aggregate one manifest-scoped run, or several runs whose manifests agree:

```bash
bazel run --jobs=1 //benchmarking/sparsify_errors:aggregate -- \
  --output benchmarking/sparsify_errors/aggregated_results.jsonl \
  benchmarking/sparsify_errors/out/<run-id> \
  benchmarking/sparsify_errors/out/<compatible-run-id>
```

The aggregator re-hashes the read-only binary and circuit snapshots, verifies
manifest compatibility, the contiguous job-file range, and exact
per-circuit/configuration repetition counts. It therefore refuses altered
artifacts and any run that is incomplete or still in progress. It also rejects
unknown raw fields and duplicate seeds, validates required types and fixed
sweep settings, groups by exact `circuit_path` plus the recorded decoder
configuration, sums outcomes, records the run IDs that actually contributed
to each row, and writes canonical JSONL. It rejects mixed values for parameters
expected to be fixed across the sweep.

## Data schema

Each aggregate row retains observed outcomes (`num_shots`, `num_errors`,
`num_low_confidence`, and `total_time_seconds`), its available circuit and
decoder configuration, and authoritative model metadata emitted by the same
Tesseract binary that performed the timed decode:

- `num_detectors`: detector count used by the heuristic M calculation.
- `num_raw_dem_errors`: error count in Stim's generated DEM.
- `num_compiled_errors`: error count after Tesseract's indistinguishable-error
  merge and zero-probability removal.
- `num_mandatory_errors`: compiled errors whose detector degree is no greater
  than `sparsify_base_degree`.
- `num_optional_errors`: compiled errors eligible for reactivation after the
  configured base- and maximum-degree filters.
- `basis`, `code_family`, `distance`, `num_qubits`, `physical_error_rate`, and
  `rounds`: explicit circuit identity fields in the aggregate.
- `circuit_sha256`: content identity of the snapshotted circuit used by jobs.
- `merge_errors` and `det_order_method`: model-compilation and detector-order
  modes. New raw jobs record these directly; the historical rows use the
  explicit assumptions documented in `provenance.json`.

New per-job stats record detector/error counts after the runtime binary has
performed its actual DEM compilation. Aggregation validates that those counts
agree across repetitions and marks their source as `benchmark_runtime_stats`;
it does not reconstruct them with the aggregation environment. Mandatory and
optional counts are `null` for non-sparsified baseline rows.
`num_errors` remains the observed decoding-failure count; it is deliberately
not reused for error-model size.

The plot loader requires this schema and reports the JSONL line and circuit on
missing or malformed data. Plotting never opens circuit files or substitutes
fallback metadata.

## Historical provenance limitation

The committed numerical results came from the full-dataset attachment on
PR #254. The original raw per-job files and aggregation program were not
located in the referenced artifacts, so their exact runtime commit, Stim
version, hardware, and job composition cannot be reconstructed. In particular,
178 aggregate rows contain more than the 10,000,000 requested-shot cap per
circuit/configuration in one invocation of the checked-in sweep (1,000
repetitions times 10,000 requested shots); the maximum is 231,100,000 shots.
This shows that additional job sets were combined, but not which ones.

The historical rows did not record merge mode or detector-order method. Their
`merge_errors=true` and `det_order_method="index"` fields are explicit legacy
assumptions based on the checked-in submission command and decoder defaults,
not runtime-recorded facts.

`provenance.json` records this limitation, the imported aggregate's SHA-256,
and the repository revision used to reconstruct circuit metadata. `enrich.py`
does not claim that the current circuits or dependency versions were the ones
used for the historical timing runs.

For an existing aggregate whose model metadata is absent or stale, regenerate
it from the checked-in circuits and verify the canonical result with:

```bash
bazel run --jobs=1 //benchmarking/sparsify_errors:enrich -- \
  --assume-merge-errors \
  --assume-det-order-method index \
  --input benchmarking/sparsify_errors/aggregated_results.jsonl \
  --output benchmarking/sparsify_errors/aggregated_results.jsonl \
  --repo-root .

bazel run --jobs=1 //benchmarking/sparsify_errors:enrich -- \
  --check \
  --input benchmarking/sparsify_errors/aggregated_results.jsonl \
  --output benchmarking/sparsify_errors/aggregated_results.jsonl \
  --repo-root .
```

The assumption flags are needed only when importing legacy rows that lack
those fields. Once enriched, `--check` requires no assumptions and verifies
the committed circuit hashes and compiled counts against the checked-in
circuits.

## Statistical interpretation and plots

X- and Z-basis circuits are separate experiments throughout; stopped-sampling
counts are never pooled between bases. A plotted failure count is
`num_errors + num_low_confidence`. The y-axis reports shot failure probability
divided by `rounds`, not an exact inferred per-round logical error rate.

A zero-failure observation is plotted at its one-sided 95% exact binomial
upper bound with a downward-limit marker. Censored points are excluded from
power-law fits and Pareto interpolation. Relative-risk annotations use a 0.5
correction to both outcomes, hence an adjusted total of `num_shots + 1`.

Generate the 19 committed PDFs from the repository root:

```bash
bazel run --jobs=1 //benchmarking/sparsify_errors:plot -- \
  --input benchmarking/sparsify_errors/aggregated_results.jsonl \
  --output-dir benchmarking/sparsify_errors/plots
```

Only PDFs are produced. The plotting dependencies are direct entries in
`src/py/requirements.in` and are pinned by `src/py/requirements_lock.txt`.
