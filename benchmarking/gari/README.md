# GARI Benchmarks


## Contents

- `submit.sh`: Slurm submission script for the benchmark sweep.
- `submit_locally.sh`: Local bash script execution equivalent for the benchmark sweep.
- `aggregated_results.jsonl`: Aggregated benchmark results (not checked into source control).
- `interactive_plot.py`: Interactive html dashboard for plotting and visualizing performance tradeoffs dynamically in the browser.

## steps before running jobs

```bash
bazel build src:tesseract
```

Generate all the gari dems with all prior modes and detector ordering for the targeted circuits.
From the repository root:
# Run DEM Generation under Bazel:
```
bazel run //src/py/_tesseract_py_util:gari_dem_utils -- "testdata/bivariatebicyclecodes/"
bazel run //src/py/_tesseract_py_util:gari_dem_utils -- "testdata/colorcodes/"
bazel run //src/py/_tesseract_py_util:gari_dem_utils -- "testdata/surfacecodes/"
```

# Run test simulation under Bazel:
```
bazel run //src/py:gari_simulation_test
```

The gari_dem_utils and simulation scripts require `numpy`, `scipy`, `matplotlib`, and `stim`, which are managed by Bazel when run via `bazel run`.

## Re-running jobs

From the repository root:

```bash
benchmarking/gari/submit.sh
```

The script assumes the Tesseract binary is available at
`./bazel-bin/src/tesseract`, reads circuits from `testdata/`, gari dems for the corresponding circuits, and submits jobs
with `sbatch`. The Slurm partition, memory, CPU count, and walltime are tuned
for the cluster used for the original PR benchmark and may need adjustment
before reuse.

Per-job stats are written under `out/`. Those raw per-job JSON files are not
included here; `aggregated_results.jsonl` is the aggregated dataset used for the
published plots.

## Plotting and Visualization

From this directory:

```bash
python3 interactive_plot.py
```

The `interactive_plot.py` script reads the local `aggregated_results.jsonl` data and outputs an interactive HTML dashboard (`interactive_plot.html`) that can be opened in any web browser. 
It requires `bokeh`, `pandas`, `numpy`, and `jupyter_bokeh`. 

Within the interactive dashboard, you can filter against Code Families, Distances, and other hyperparmeters, dynamically generating visual speedup baselines and Logical Error Rate plots.
