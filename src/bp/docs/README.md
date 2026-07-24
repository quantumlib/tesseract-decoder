# Belief Propagation (BP) Decoder

A high-performance monolithic C++ Belief Propagation (BP) decoder for Quantum Error Correction (QEC) Low-Density Parity-Check (LDPC) and Bivariate Bicycle codes, featuring AVX-512 SIMD shot-batching, flexible message passing schedules, and integrated Ordered Statistics Decoding (OSD) post-processing.

---

## Features

- **Monolithic C++ Engine:** Direct Tanner graph representations with memory-efficient check-to-variable node updates.
- **Schedules:**
  - `serial`: Node-by-node sequential Gallager message updates for fast convergence.
  - `parallel`: Synchronous round-based message updates across check/variable nodes.
- **Update Rules:**
  - `min-sum`: Min-Sum update rule with a customizable normalization scaling factor (default `0.625`).
  - `product-sum`: Standard Sum-Product (log-likelihood ratio) updates.
- **Post-Processing Engines:**
  - **Hard Decision:** Fast threshold decision based on LLR signs.
  - **OSD (Ordered Statistics Decoding):** Post-processing using matrix order selection (`--osd-order >= 0`, e.g., OSD-0 or OSD-10) for higher decoding accuracy.
- **AVX-512 SIMD Batching:** Decodes 64 shots simultaneously per worker thread using 64-bit vector register operations (`--batched`).
- **Arbitrary Observable Scale:** Fully handles detection error models with arbitrary numbers of observables (beyond 64-bit mask limits).
- **Sinter Integration:** Seamless compatibility with `sinter.collect()` via `tesseract_decoder.bp_sinter_compat`.

---

## Building

Build the standalone multithreaded `bp` binary using Bazel:

```bash
bazel build //src:bp
```

The resulting executable will be built at `bazel-bin/src/bp`.

---

## Command Line Usage

### Basic Usage Syntax

```bash
bazel run //src:bp -- <options>
```

### Options Reference

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--circuit <path>` | — | Path to a Stim `.stim` circuit file. |
| `--dem <path>` | — | Path to a Stim `.dem` Detector Error Model file. |
| `--sample-num-shots <N>` | `0` | Number of shots to sample and decode in memory. |
| `--max-errors <E>` | Unlimited | Early stop threshold after `<E>` logical errors occur. |
| `--sample-seed <seed>` | `12345` | Random seed for Stim shot sampling. |
| `--in <file>` | — | File to read detection events from. |
| `--in-format <fmt>` | — | Format of input shots file (e.g., `b8`, `01`, `hits`). |
| `--out <file>` | — | Destination path for predicted observable flips (`-` for stdout). |
| `--out-format <fmt>` | — | Format of output predictions file. |
| `--schedule <sched>` | `serial` | Message passing schedule (`serial` or `parallel`). |
| `--update-rule <rule>` | `min-sum` | BP update rule (`min-sum`). |
| `--max-iter <N>` | `20` | Maximum BP iteration limit per shot. |
| `--normalization-factor <f>`| `0.625` | Min-sum check update normalization factor. |
| `--osd-order <int>` | `-1` | `-1` for Hard Decision; `>= 0` for OSD (e.g., `0`, `10`). |
| `--osd-weight <int>` | `0` | Weight parameter for OSD post-processing. |
| `--batched` | `false` | Enables **AVX-512 SIMD shot batching** (decodes 64 shots in parallel per thread). |
| `--threads <N>` | Core count | Number of parallel worker threads across shots. |
| `--stats-out <file>` | — | Path to write full JSON stats (`-` for stdout). |
| `--sinter-csv-out <file>` | — | Path to append Sinter CSV benchmark summary lines (`-` for stdout). |
| `--print-stats` | `false` | Output periodic progress updates to stdout. |

---

## CLI Examples

### 1. High-Performance AVX-512 Batched BP
Runs 64-shot SIMD batching across all available CPU cores:

```bash
bazel run //src:bp -- \
  --circuit "$PWD/testdata/bivariatebicyclecodes/r=10,d=10,p=0.001,noise=si1000,c=bivariate_bicycle_X,nkd=[[108,8,10]],q=216,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim" \
  --sample-num-shots 10000 \
  --batched \
  --schedule parallel \
  --threads $(nproc) \
  --print-stats
```

### 2. BP + OSD-0 Post-Processing
Decodes sampled shots using serial min-sum BP with OSD-0 post-processing:

```bash
bazel run //src:bp -- \
  --circuit "$PWD/testdata/bivariatebicyclecodes/r=10,d=10,p=0.001,noise=si1000,c=bivariate_bicycle_X,nkd=[[108,8,10]],q=216,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim" \
  --sample-num-shots 1000 \
  --schedule serial \
  --normalization-factor 0.625 \
  --osd-order 0 \
  --threads $(nproc) \
  --print-stats
```

### 3. Sinter CSV Output Line
Outputs benchmark results formatted as a single Sinter CSV line:

```bash
bazel run //src:bp -- \
  --circuit "$PWD/testdata/bivariatebicyclecodes/r=10,d=10,p=0.001,noise=si1000,c=bivariate_bicycle_X,nkd=[[108,8,10]],q=216,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim" \
  --sample-num-shots 50000 \
  --batched \
  --schedule serial \
  --sinter-csv-out -
```

---

## Python Interface

### 1. Direct Python Usage

```python
import numpy as np
import stim
from tesseract_decoder import bp

dem = stim.DetectorErrorModel("""
    error(0.1) D0
    error(0.2) D1 L0
""")

params = bp.BPParams()
params.max_iter = 20
params.update_rule = "min-sum"
params.schedule = "serial"
params.normalization_factor = 0.625

decoder = bp.TesseractBpDecoder(dem, params)

# Post-processor: OSD-0 or Hard Decision
post_processor = decoder.create_osd_post_processor(osd_order=0, osd_weight=0)
# post_processor = bp.HardDecisionPostProcessor()

# Decode a syndrome boolean array
syndrome = np.array([True, False], dtype=bool)
predictions = decoder.decode(syndrome, post_processor)
```

### 2. Sinter Integration (`sinter.collect`)

```python
import sinter
from tesseract_decoder import bp, bp_sinter_compat

params = bp.BPParams()
params.schedule = "serial"
params.max_iter = 20

# Compatible decoder factory for sinter.collect()
custom_decoders = {
    "tesseract_bp_osd0": bp_sinter_compat.TesseractBpSinterDecoder(
        params, osd_order=0, osd_weight=0
    )
}

# Run sinter sampling pipeline
# sinter.collect(..., custom_decoders=custom_decoders)
```

---

## Running Unit Tests

Run the BP C++ and Python test suites:

```bash
# C++ BP Unit Tests
bazel test //src:bp_tests

# Python Sinter Compatibility Tests
bazel test //src/py:tesseract_bp_sinter_test
```
