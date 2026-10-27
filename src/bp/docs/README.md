# Belief Propagation (BP-OSD) Decoder

A high-performance monolithic C++ Belief Propagation with Ordered Statistics Decoding (BP-OSD) and Multi-Pass Shattered BP-OSD decoder for Quantum Error Correction (QEC) Low-Density Parity-Check (LDPC), Bivariate Bicycle (BB), and Color Codes.

---

## Overview & Key Features

- **Serial Normalized Min-Sum BP Engine:** Sequential check-node Gallager schedule with configurable check-to-variable normalization factor (`--normalization-factor`) and optional randomized check permutation (`--random-schedule`) to mitigate trapping sets.
- **Integrated OSD-$w$ Post-Processing:** Supports Hard Decision (`--osd-order -1`), OSD-0 (`--osd-order 0`), and higher-order combination sweep OSD (`--osd-order <k> --osd-weight <w>`), plus one-flag presets (`--fast` and `BPParams::Osd()` / `BPParams::Fast()`).
- **Multi-Pass Shattered BP-OSD (`--multipass`):** Multi-stage decoding across decomposed Tanner graph components with soft posterior LLR reweighting between passes.
- **AVX-512 SIMD Shot Batching & Two-Stage Compaction:** Processes 512-shot macro-batches per worker thread using 16-lane (`int32_t`) AVX-512 vector instructions and cross-batch survivor compaction.
- **Arbitrary Observable Scale:** Supports Detector Error Models (DEMs) with arbitrary numbers of logical observables.
- **Built-in Phase Profiler (`--profile`):** Reports per-phase timing (`construction`, `bp`, `deinterleave`, `post_processing`), convergence iteration quantiles (`p50`, `p90`, `max`), and sub-phase OSD breakdown (`sort_Order`, `build_matrix`, `gauss_elim`, `osd0_solve`, `higher_order_search`).
- **Python & Sinter Integration:** Native `pybind11` bindings (`tesseract_decoder.bp`) and first-class `sinter.collect()` compatibility (`tesseract_decoder.bp_sinter_compat`).

---

## High-Level Architecture & Optimizations

1. **Flat Cache-Coherent CSR Tanner Graph (`TannerGraph`):**
   - Stores check-to-variable and variable-to-check adjacency lists in contiguous Compressed Sparse Row (CSR) arrays rather than nested `std::vector` allocations.
   - Eliminates pointer chasing and improves L1/L2 hardware prefetching during iterative message passing.

2. **16-Lane AVX-512 Fixed-Point Serial Min-Sum Kernel (`bp_serial_min_sum.cc`):**
   - Quantizes log-likelihood ratios (LLRs) into 32-bit signed fixed-point integers (`int32_t`, `BP_BATCH_SIZE = 16` shots per SIMD register).
   - Computes check-node first and second minimums (`min1`, `min2`) and sign parities entirely in vector registers, applying fixed-point normalization scaling without floating-point conversions.
   - Tracks per-lane syndrome satisfaction masks at the end of each iteration and freezes converged lanes in-place so their posterior LLRs cannot oscillate while remaining lanes continue iterating.

3. **Two-Stage Cross-Batch Survivor Compaction (`TesseractBpDecoder::decode_batch`):**
   - At typical physical error rates ($p = 0.001$), 86%–95% of shots converge within 1–4 BP iterations, while a small tail of unconverged shots requires the full `max_iter` budget.
   - **Stage 1 (Fast Screening):** Runs all 16-shot SIMD sub-batches across a 512-shot macro-batch for at most `kStage1MaxIter = 4` iterations. Converged shots are retired immediately to hard-decision observable prediction.
   - **Stage 2 (Dense Survivor Compaction):** Compacts only the unconverged survivor shots from across all sub-batches into dense 16-shot SIMD sub-batches and runs them for `1..max_iter` before dispatching any still-unconverged shots to OSD. This eliminates "tail-lane starvation" (where 15 converged SIMD lanes idle waiting for 1 trapped lane) and speeds up batched BP by **1.46x–1.77x** with bit-for-bit identical logical error rates.

4. **OSD-$w$ Post-Processing Optimizations (`OsdPostProcessor`):**
   - **BP Peeling (Early Termination):** Invokes OSD exclusively on the small fraction of shots that fail to satisfy the syndrome after `max_iter` BP iterations.
   - **64-Bit Bitword Gaussian Elimination:** Packs the reliability-ordered parity-check submatrix into 64-bit column words (`uint64_t`) and uses hardware trailing-zero bit scans (`__builtin_ctzll`) and word-level XORs for pivot elimination.
   - **Dynamic Free-Column Truncation (`--osd-truncation-factor`, default `1.2`):** Bounds the dense Gaussian elimination matrix to $D$ pivot columns plus at most $\lceil \tau \cdot D \rceil$ free columns (where $D$ is the number of detectors), avoiding cubic elimination overhead on redundant high-reliability columns in large circuit-level DEMs.
   - **Residual Syndrome Early Stopping (`osd_early_exit`):** Tracks remaining active syndrome defects during forward elimination and terminates the column scan early once all active defects are spanned by independent pivots.
   - **Reference-Delta Higher-Order Search (`osd_order > 0`):** Evaluates candidate error patterns of Hamming weight $\le w$ in the $k$ most-unreliable free coordinates using incremental delta updates from the OSD-0 base solution rather than recomputing full matrix-vector products.

5. **Multi-Pass Shattered BP-OSD (`MultiPassBpDecoder`):**
   - Partitions detectors into component subgraphs (automatically or via GARI-informed schedules) and runs sequential BP-OSD passes, propagating residual syndromes and soft-reweighted prior LLRs across passes to break symmetric trapping sets on high-degeneracy codes.

---

## Building

Build the standalone multithreaded `bp` binary using Bazel:

```bash
bazel build --jobs=1 //src:bp
```

The resulting executable is placed at `bazel-bin/src/bp`.

---

## Command Line Usage

### Basic Syntax

```bash
bazel run --jobs=1 //src:bp -- <options>
```

### Options Reference

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--circuit <path>` | — | Path to a Stim `.stim` circuit file. |
| `--dem <path>` | — | Path to a Stim `.dem` Detector Error Model file. |
| `--sample-num-shots <N>` | `0` | Number of shots to sample and decode in memory. |
| `--max-errors <E>` | Unlimited | Early stop threshold after `<E>` logical errors occur. |
| `--sample-seed <seed>` | `12345` | Random seed for Stim shot sampling and randomized schedules. |
| `--in <file>` | — | File to read detection events from. |
| `--in-format <fmt>` | — | Format of input shots file (e.g., `b8`, `01`, `hits`). |
| `--obs_in <file>` | — | Optional file to read ground-truth observable flips from. |
| `--obs-in-format <fmt>` | — | Format of observable input file. |
| `--out <file>` | — | Destination path for predicted observable flips (`-` for stdout). |
| `--out-format <fmt>` | — | Format of output predictions file. |
| `--max-iter <N>` | `20` | Maximum BP iteration limit per shot. |
| `--normalization-factor <f>` | `0.625` | Min-sum check-to-variable normalization factor. |
| `--osd-order <int>` | `-1` | `-1` for Hard Decision; `0` for OSD-0; `> 0` for higher-order OSD-$w$ over `<int>` free columns. |
| `--osd-weight <int>` | `0` | Maximum combination Hamming weight $w$ for higher-order OSD search. |
| `--osd-truncation-factor <f>` | `1.2` | Bounds free columns in OSD Gaussian elimination to `<f> * num_detectors` (`0.0` disables truncation). |
| `--disable-osd-early-exit` | `false` | Disables residual-syndrome early stopping during OSD Gaussian elimination. |
| `--fast` | `false` | Fast BP-OSD preset (`max_iter=10`, `osd_order=10`, `osd_weight=0`, batched). |
| `--batched` | `true` | Enables 512-shot macro-batching with 16-lane AVX-512 SIMD + two-stage compaction (on by default). |
| `--random-schedule` | `false` | Uses a randomized check-node permutation in serial BP to mitigate trapping sets. |
| `--multipass` | `false` | Enables Multi-Pass Shattered BP-OSD decoding. |
| `--num-passes <N>` | `2` | Number of passes in Multi-Pass BP. |
| `--multipass-soft-reweight` | `true` | Enables soft LLR reweighting across passes (on by default). |
| `--disable-multipass-soft-reweight` | `false` | Disables soft LLR reweighting across passes. |
| `--threads <N>` | Core count | Number of parallel worker threads across shot batches. |
| `--profile` | `false` | Prints detailed BP and OSD per-phase timing and convergence histograms. |
| `--stats-out <file>` | — | Path to write full JSON benchmark summary (`-` for stdout). |
| `--sinter-csv-out <file>` | — | Path to append Sinter CSV summary lines (`-` for stdout). |
| `--print-stats` | `false` | Outputs periodic progress updates to stdout. |

---

## CLI Examples

### 1. Fast Preset BP-OSD (`--fast`) with Profiling (`--profile`)
Runs the fast BP-OSD preset (`max_iter=10`, `osd_order=10`, `osd_weight=0`, two-stage SIMD batching) and prints per-phase BP and OSD timing breakdowns:

```bash
bazel run --jobs=1 //src:bp -- \
  --circuit "$PWD/testdata/bivariatebicyclecodes/r=6,d=6,p=0.001,noise=si1000,c=bivariate_bicycle_Z,nkd=[[72,12,6]],q=144,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim" \
  --sample-num-shots 10000 \
  --fast \
  --threads 1 \
  --profile
```

### 2. High-Accuracy Higher-Order BP+OSD (`OSD-24, weight 2`)
Decodes using serial normalized min-sum BP (`max_iter=20`) followed by order-24, weight-2 OSD combination sweep:

```bash
bazel run --jobs=1 //src:bp -- \
  --circuit "$PWD/testdata/bivariatebicyclecodes/r=6,d=6,p=0.001,noise=si1000,c=bivariate_bicycle_Z,nkd=[[72,12,6]],q=144,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim" \
  --sample-num-shots 10000 \
  --max-iter 20 \
  --normalization-factor 0.875 \
  --osd-order 24 \
  --osd-weight 2 \
  --threads $(nproc) \
  --print-stats
```

### 3. Multi-Pass Shattered BP-OSD (`--multipass`)
Runs 2-pass shattered BP-OSD with soft LLR reweighting and outputs a Sinter CSV summary line:

```bash
bazel run --jobs=1 //src:bp -- \
  --circuit "$PWD/testdata/colorcodes/r=7,d=7,p=0.001,noise=si1000,c=color_code_666_X,q=73.stim" \
  --sample-num-shots 50000 \
  --multipass \
  --num-passes 2 \
  --osd-order 10 \
  --osd-weight 0 \
  --sinter-csv-out -
```

---

## Python Interface

### 1. Direct Python Usage (`tesseract_decoder.bp`)

```python
import numpy as np
import stim
from tesseract_decoder import bp

dem = stim.DetectorErrorModel("""
    error(0.1) D0
    error(0.2) D1 L0
""")

# Option A: Built-in presets
# params = bp.BPParams.Fast()  # max_iter=10, osd_order=10, osd_weight=0
# params = bp.BPParams.Osd()   # max_iter=20, osd_order=24, osd_weight=2

# Option B: Custom configuration
params = bp.BPParams()
params.max_iter = 20
params.normalization_factor = 0.875
params.osd_truncation_factor = 1.2
params.osd_early_exit = True

decoder = bp.TesseractBpDecoder(dem, params)

# Create an OSD post-processor (or pass None to decode() for Hard Decision)
post_processor = decoder.create_osd_post_processor(osd_order=0, osd_weight=0)

# Decode a single shot from a dense boolean syndrome array
syndrome = np.array([True, False], dtype=bool)
predictions = decoder.decode(syndrome, post_processor)

# Multi-Pass BP-OSD decoder
mp_config = bp.MultiPassBpConfig(params=bp.BPParams.Fast(), num_passes=2, soft_reweight=True)
mp_decoder = bp.MultiPassBpDecoder(dem, mp_config)
mp_predictions = mp_decoder.decode(syndrome)
```

### 2. Sinter Integration (`sinter.collect`)

```python
import sinter
from tesseract_decoder import bp_sinter_compat

# Built-in Sinter factory presets:
custom_decoders = {
    "tesseract_bp_osd": bp_sinter_compat.TesseractBpSinterDecoder.BpOsd(),
    "tesseract_bp_osd_fast": bp_sinter_compat.TesseractBpSinterDecoder.BpOsdFast(),
    "tesseract_multipass_bp_osd": bp_sinter_compat.TesseractBpSinterDecoder.MultiPassBpOsd(),
    "tesseract_multipass_bp_osd_fast": bp_sinter_compat.TesseractBpSinterDecoder.MultiPassBpOsdFast(),
}

# Run sinter sampling pipeline:
# sinter.collect(..., custom_decoders=custom_decoders)
```

---

## Running Unit Tests

Run the C++ and Python BP test suites (using `--jobs=1` per repository guidelines):

```bash
# C++ BP & OSD Unit Tests
bazel test --jobs=1 //src:bp_tests

# C++ Sinter Compatibility Tests
bazel test --jobs=1 //src:sinter_compat_tests

# Python Sinter Compatibility Tests
bazel test --jobs=1 //src/py:tesseract_bp_sinter_test
```
