<div align="center">

# [Tesseract Decoder](https://quantumlib.github.io/tesseract-decoder)

A Search-Based Decoder for Quantum Error Correction.

[![Licensed under the Apache 2.0 open-source license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative\&logoColor=white\&style=flat-square)](https://github.com/quantumlib/tesseract-decoder/blob/main/LICENSE)
![C++](https://img.shields.io/badge/C++-20-fcbc2c?style=flat-square&logo=C%2B%2B&logoColor=white)

[Installation](#installation) &ndash;
[Usage](#usage) &ndash;
[Python Interface](#python-interface) &ndash;
[Paper](https://arxiv.org/pdf/2503.10988) &ndash;
[Help](#help) &ndash;
[Citation](#citation) &ndash;
[Contact](#contact)

</div>

Tesseract is a Most Likely Error decoder designed for Low Density Parity Check (LDPC) quantum
error-correcting codes. It applies pruning heuristics and manifold orientation techniques during a
search over the error subsets to identify the most likely error configuration consistent with the
observed syndrome. Tesseract achieves significant speed improvements over traditional integer
programming-based decoders while maintaining comparable accuracy at moderate physical error rates.

We tested the Tesseract decoder for:

*   Surface codes
*   Color codes
*   Bivariate-bicycle codes
*   Transversal CNOT protocols for surface codes

## Features

*   **A\* search:** deploys [A\* search](https://en.wikipedia.org/wiki/A*_search_algorithm) while
    running a [Dijkstra algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm) with early
    stop for high performance.
*   **Stim and DEM Support:** processes [Stim](https://github.com/quantumlib/stim) circuit files and
    [Detector Error Model
    (DEM)](https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md)
    files with arbitrary error models. Zero-probability error instructions are
    automatically removed when a DEM is loaded.
*   **Parallel Decoding:** uses multithreading to accelerate the decoding process, making it
    suitable for large-scale simulations.
*   **Efficient Beam Search:** implements a [beam search](https://en.wikipedia.org/wiki/Beam_search)
    algorithm to minimize decoding cost and enhance efficiency.
**Sampling and Shot Range Processing:** supports sampling shots from circuits. When a detection
    error model is provided without an accompanying circuit, Tesseract requires detection events from
    files using `--in`. The decoder can also process specific shot ranges for flexible experiment
    setups.
*   **Detailed Statistics:** provides comprehensive statistics output, including shot counts, error
    counts, and processing times.
*   **Heuristics**: includes flexible heuristic options: `--beam`, `--det-penalty`,
    `--beam-climbing`, `--no-revisit-dets`, `--at-most-two-errors-per-detector`, and `--pqlimit` to
    improve performance while maintaining a low logical error rate. To learn more about these
    options, use `./bazel-bin/src/tesseract --help`
*   **Visualization tool:** open the [viz directory](viz/) in your browser to view decoding results. See [viz/README.md](viz/README.md) for instructions on generating the visualization JSON.

## Installation

Tesseract relies on the following external libraries:

*   [argparse](https://github.com/p-ranav/argparse): For command-line argument parsing.
*   [nlohmann/json](https://github.com/nlohmann/json): For JSON handling (used for statistics output).
*   [Stim](https://github.com/quantumlib/stim): For quantum circuit simulation and error model
    handling.

### Build Instructions

Tesseract uses [Bazel](https://bazel.build/) as its build system. To build the decoder:

```bash
bazel build src:all
```

## Running Tests

Unit tests are executed with Bazel. Run the quick test suite using:
```bash
bazel test //src:all
```
By default the tests use reduced parameters and finish in under 30 seconds.
To run a more exhaustive suite with additional shots and larger distances, set:
```bash
TESSERACT_LONG_TESTS=1 bazel test //src:all
```


## Usage

The file `tesseract_main.cc` provides the main entry point for Tesseract Decoder. It can decode
error events from Stim circuits, DEM files, and pre-existing detection event files.

Basic Usage:

```bash
./tesseract --circuit CIRCUIT_FILE.stim --sample-num-shots N --print-stats
```

To decode pre-generated detection events, provide the input file using
`--in SHOTS_FILE --in-format FORMAT`.


Example with Advanced Options:

```bash
./tesseract \
        --pqlimit 1000000 \
        --at-most-two-errors-per-detector \
        --det-order-seed 232852747 \
        --circuit circuit_file.stim \
        --sample-seed 232856747 \
        --sample-num-shots 10000 \
        --threads 32 \
        --print-stats \
        --beam 23 \
        --num-det-orders 1 \
        --shot-range-begin 582 \
        --shot-range-end 583
```

### Example Usage

Sampling Shots from a Circuit:

```bash
./tesseract --circuit surface_code.stim --sample-num-shots 1000 --out predictions.01 --out-format 01
```

Using a Detection Event File:

```bash
./tesseract --in events.01 --in-format 01 --dem surface_code.dem --out decoded.txt
```

Using a Detection Event File and Observable Flips:

```bash
./tesseract --in events.01 --in-format 01 --obs_in obs.01 --obs-in-format 01 --dem surface_code.dem --out decoded.txt
```

Tesseract supports reading and writing from all of Stim's standard [output
formats](https://github.com/quantumlib/Stim/blob/main/doc/result_formats.md).

### Performance Optimization

Here are some tips for improving performance:

*   *Parallelism over shots*: increase `--threads` to leverage multicore processors for faster
    decoding.
*   *Beam Search*: use `--beam` to control the trade-off between accuracy and speed. Smaller beam sizes
    result in faster decoding but potentially lower accuracy.
*   *Beam Climbing*: enable `--beam-climbing` for enhanced cost-based decoding.
*   *At most two errors per detector*: enable `--at-most-two-errors-per-detector` to improve
    performance.
*   *Priority Queue limit*: use `--pqlimit` to limit the size of the priority queue.

### Output Formats

*   *Observable flips output*: predictions of logical errors.
*   *DEM usage frequency output*: if `--dem-out` is specified, outputs estimated error frequencies.
*   *Statistics output*: includes number of shots, errors, low confidence shots, and processing time.

## Python Interface

[Full Python wrapper documentation](src/py/README.md)

This repository contains the C++ implementation of the Tesseract quantum error correction decoder, along with a Python wrapper. The Python wrapper/interface exposes the decoding algorithms and helper utilities, allowing Python users to leverage this high-performance decoding algorithm.

The following example demonstrates how to create and use the Tesseract decoder using the Python interface.

```python
from tesseract_decoder import tesseract
import stim
import numpy as np


# 1. Define a detector error model (DEM)
dem = stim.DetectorErrorModel("""
    error(0.1) D0 D1 L0
    error(0.2) D1 D2 L1
    detector(0, 0, 0) D0
    detector(1, 0, 0) D1
    detector(2, 0, 0) D2
""")

# 2. Create the decoder configuration
config = tesseract.TesseractConfig(dem=dem, det_beam=50)

# 3. Create a decoder instance
decoder = config.compile_decoder()

# 4. Simulate detector outcomes
syndrome = np.array([0, 1, 1], dtype=bool)

# 5a. Decode to observables
flipped_observables = decoder.decode(syndrome)
print(f"Flipped observables: {flipped_observables}")

# 5b. Alternatively, decode to errors
decoder.decode_to_errors(syndrome)
predicted_errors = decoder.predicted_errors_buffer
# Indices of predicted errors
print(f"Predicted errors indices: {predicted_errors}")
# Print properties of predicted errors
for i in predicted_errors:
    print(f"    {i}: {decoder.errors[i]}")
```
## Good Starting Points for Tesseract Configurations:
 The [Tesseract paper](https://arxiv.org/pdf/2503.10988) recommends two setup for starting your exploration with tesseract:


(1) Long-beam setup:
```
tesseract_config = tesseract.TesseractConfig(
    dem=dem,
    pqlimit=1_000_000,
    det_beam=20,
    beam_climbing=True,
    num_det_orders=21,
    det_order=tesseract_decoder.utils.DetOrder.DetIndex,
    no_revisit_dets=True,
)
```
(2) Short-beam setup:

```
tesseract_config = tesseract.TesseractConfig(
    dem=dem,
    pqlimit=200_000,
    det_beam=15,
    beam_climbing=True,
    num_det_orders=16,
    det_order=tesseract_decoder.utils.DetOrder.DetIndex,
    no_revisit_dets=True,
)
```
For `det_order`, you can use two other options of `DetIndex` and `DetCoordinate` as well.
These values balance decoding speed and accuracy across the benchmarks reported in the paper and can be adjusted for specific use cases.
## Help

*   Do you have a feature request or want to report a bug? [Open an issue on
    GitHub] to report it!
*   Do you have a code contribution? Read our [contribution guidelines], then
    open a [pull request]!

[Open an issue on GitHub]: https://github.com/quantumlib/tesseract-decoder/issues/new/choose
[contribution guidelines]: https://github.com/quantumlib/tesseract-decoder/blob/main/CONTRIBUTING.md
[pull request]: https://help.github.com/articles/about-pull-requests

We are committed to providing a friendly, safe, and welcoming environment for
all. Please read and respect our [Code of Conduct](CODE_OF_CONDUCT.md).

## Citation

When publishing articles or otherwise writing about Tesseract Decoder, please
cite the following:

```latex
@misc{beni2025tesseractdecoder,
    title={Tesseract: A Search-Based Decoder for Quantum Error Correction},
    author = {Aghababaie Beni, Laleh and Higgott, Oscar and Shutty, Noah},
    year={2025},
    eprint={2503.10988},
    archivePrefix={arXiv},
    primaryClass={quant-ph},
    doi = {10.48550/arXiv.2503.10988},
    url={https://arxiv.org/abs/2503.10988},
}
```

## Contact

For any questions or concerns not addressed here, please email <tesseract-decoder-dev@google.com>.

## Disclaimer

Tesseract Decoder is not an officially supported Google product. This project is not eligible for
the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).

Copyright 2025 Google LLC.

<div align="center">
  <a href="https://quantumai.google">
    <img width="15%" alt="Google Quantum AI"
         src="./docs/images/quantum-ai-vertical.svg">
  </a>
</div>
