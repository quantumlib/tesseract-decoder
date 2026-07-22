# Error sparsification benchmarks

This directory preserves the benchmarking artifacts used for
[PR #254](https://github.com/quantumlib/tesseract-decoder/pull/254), which
introduced `--sparsify-errors`.

## Contents

- `submit.sh`: Slurm submission script for the benchmark sweep.
- `aggregated_results.jsonl`: Aggregated benchmark results used by the plots.
- `plot.py`: Plotting and summary-analysis script.
- `plots/`: PDF plots generated from `aggregated_results.jsonl`.

## Re-running jobs

From the repository root:

```bash
bazel build src:tesseract
benchmarking/sparsify_errors/submit.sh
```

The script assumes the Tesseract binary is available at
`./bazel-bin/src/tesseract`, reads circuits from `testdata/`, and submits jobs
with `sbatch`. The Slurm partition, memory, CPU count, and walltime are tuned
for the cluster used for the original PR benchmark and may need adjustment
before reuse.

Per-job stats are written under `out/`. Those raw per-job JSON files are not
included here; `aggregated_results.jsonl` is the aggregated dataset used for the
published plots.

## Re-making plots

From this directory:

```bash
python3 plot.py
```

The plotting script expects `aggregated_results.jsonl` in the current working
directory and writes outputs into `plots/`. It requires `matplotlib`, `numpy`,
`scipy`, and `stim`.
