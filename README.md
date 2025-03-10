# Tesseract Decoder

A Search-Based Decoder for Quantum Error Correction

[![Licensed under the Apache 2.0 open-source
license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/tesseract-decoder/blob/main/LICENSE)

## Overview

Tesseract is a Most Likely Error decoder designed for Low Density Parity Check (LDPC) quantum error-correcting codes. It employs the A\* search algorithm to efficiently navigate the exponentially large graph of possible error subsets, identifying the most likely error configuration consistent with the observed syndrome. Tesseract leverages several pruning heuristics and manifold orientation techniques to achieve significant speed improvements over traditional integer programming-based decoders, while maintaining comparable accuracy at moderate physical error rates.

We tested the Tesseract decoder for:

-   Surface codes
-   Color codes
-   Bivariate-bicycle codes
-   Transversal CNOT protocols for surface codes
Stim circuits for these protocols are stored in `testdata/`.
   
## Features

-   **A\* search:** deploys A* search while running a Dijkstra algorithm with early stop for high performance.
-   **Stim and DEM Support:** Processes Stim circuit files and Detector Error Model (DEM) files with arbitrary error models.
-   **Parallel Decoding:** Utilizes multi-threading to accelerate the decoding process, making it suitable for large-scale simulations.
-   **Efficient Beam Search:** Implements a beam search algorithm to minimize decoding cost and enhance efficiency.
-   **Sampling and Shot Range Processing:** Supports sampling shots from circuits and processing specific ranges of shots for flexible experiment setups.
-   **Detailed Statistics:** Provides comprehensive statistics output, including shot counts, error counts, and processing times.
- **Heuristics**: Includes flexible heuristic options: `--beam, --det-penalty, --beam-climbing, --no-revisit-dets, --at-most-two-errors-per-detector` and `--pqlimit` to improve performance while maintaining a low logical error rate. To learn more about these options, use `./bazel-bin/tesseract/tesseract --help`


## Installation

Tesseract relies on the following external libraries:

-   [argparse](https://github.com/p-ranav/argparse): For command-line argument parsing.
-   [nlohmann/json](https://github.com/nlohmann/json): For JSON handling (used for statistics output).
-   [Stim](https://github.com/quantumlib/stim): For quantum circuit simulation and error model handling.

### Build Instructions

Tesseract uses Bazel as its build system. To build the decoder:

```bash
bazel build tesseract:all
```
Usage
Running the Decoder
The tesseract_main.cc file provides the main entry point for the Tesseract decoder. It can decode error events from Stim circuits, DEM files, and pre-existing detection event files.

Basic Usage:

 ```bash 
./tesseract --circuit <circuit_file.stim> --sample-num-shots <N> --print-stats
```
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

Sampling Shots from a Circuit
```bash 
./tesseract --circuit surface_code.stim --sample-num-shots 1000 --out predictions.01 --out-format 01
```
Using a Detection Event File
```bash 
./tesseract --in events.01 --in-format 01 --dem surface_code.dem --out decoded.txt
```
Using a Detection Event File and Observable Flips
```bash 
./tesseract --in events.01 --in-format 01 --obs_in obs.01 --obs-in-format 01 --dem surface_code.dem --out decoded.txt
```
Tesseract supports reading and writing from all of Stim's standard output formats (see https://github.com/quantumlib/Stim/blob/main/doc/result_formats.md).
### Performance Optimization
* Parallelism over shots: Increase ```--threads``` to leverage multi-core processors for faster decoding.
* Beam Search: Use ```--beam``` to control the trade-off between accuracy and speed. Smaller beam sizes result in faster decoding but potentially lower accuracy.
* Beam Climbing: Enable ```--beam-climbing``` for enhanced cost-based decoding.
* At most two errors per detector: Enable ```--at-most-two-errors-per-detector``` to improve performance.
* Priority Queue Limit: Use ```--pqlimit``` to limit the size of the priority queue.

### Output Formats
* Observable flips output: Predictions of logical errors.
* DEM usage frequency output: If ```--dem-out``` is specified, outputs estimated error frequencies.
* Statistics output: Includes number of shots, errors, low confidence shots, and processing time.
  


<!-- ## Citing Tesseract Decoder<a name="how-to-cite-tesseract"> -->

## Contact

For any questions or concerns not addressed here, please email
<quantum-oss-maintainers@google.com>.

## Disclaimer

Tesseract Decoder is not an official Google product.
Copyright 2025 Google LLC.
