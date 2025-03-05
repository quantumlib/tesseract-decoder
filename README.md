# Tesseract Decoder

A most-likely-error decoder for quantum error correction

[![Licensed under the Apache 2.0 open-source
license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/tesseract-decoder/blob/main/LICENSE)

## Introduction

The implementation of [quantum error
correction](https://en.wikipedia.org/wiki/Quantum_error_correction) (QEC)
requires fast and accurate decoders to achieve low logical error rates. Decoding
is an NP-hard optimization problem in the worst case, but there exists a variety
of partial solutions for specific error-correcting codes. The Tesseract Decoder
takes a novel approach: rather than building an algorithm with a polynomial
runtime and using heuristics to make it more accurate, we begin with an
exponential-time algorithm that always identifies the most likely error and use
heuristics to make it faster. The decoder uses [A*
search](https://en.wikipedia.org/wiki/A*_search_algorithm) along with a variety
of pruning heuristics.
We tested the Tesseract decoder for:

-   Surface codes
-   Superdense Color codes
-   Bivariate-bicycle codes
-   Transversal CNOT protocols for surface codes
   
## Features


-   **A\* search:** deploys A* search while running a semi Dijkstra algorithm with early stop for high performance.
-   **Stim and DEM Support:** Processes Stim circuit files and Detector Error Model (DEM) files for comprehensive error decoding.
-   **Parallel Decoding:** Utilizes multi-threading to accelerate the decoding process, making it suitable for large-scale simulations.
-   **Efficient Beam Search:** Implements a beam search algorithm to minimize decoding cost and enhance efficiency.
-   **Sampling and Shot Range Processing:** Supports sampling shots from circuits and processing specific ranges of shots for flexible experiment setups.
-   **Detailed Statistics:** Provides comprehensive statistics output, including shot counts, error counts, and processing times.
- **Heuristics**: Includes heuristics such as beam climbing and at most two errors per detector to improve performance.

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
./tesseract --circuit <circuit_file> --out <output_file>
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
./tesseract --circuit surface_code.stim --sample-num-shots 1000 --out sampled_results.txt
```
Using a Detection Event File
```bash 
./tesseract --in events.01 --in-format 01 --dem surface_code.dem --out decoded.txt
```
Using a Detection Event File and Observable Flips
```bash 
./tesseract --in events.01 --in-format 01 --obs_in obs.01 --obs-in-format 01 --dem surface_code.dem --out decoded.txt
```

### Performance Optimization
* Parallelism: Increase ```--threads``` to leverage multi-core processors for faster decoding.
* Beam Search: Use ```--beam``` to control the trade-off between accuracy and speed. Smaller beam sizes result in faster decoding but potentially lower accuracy.
* Beam Climbing: Enable ```--beam-climbing``` for enhanced cost-based decoding.
* At most two errors per detector: Enable ```--at-most-two-errors-per-detector``` to improve performance.
* Priority Queue Limit: Use ```--pqlimit``` to limit the size of the priority queue.

### Output Formats
* Observable flips output: Predictions of logical errors.
* DEM usage frequency output: If ```--dem-out``` is specified, outputs estimated error frequencies.
* Statistics output: Includes number of shots, errors, low confidence shots, and processing time.
  

<!-- ## Installation -->

<!-- ## Usage -->

<!-- ## Citing Tesseract Decoder<a name="how-to-cite-tesseract"> -->

## Contact

For any questions or concerns not addressed here, please email
<quantum-oss-maintainers@google.com>.

## Disclaimer

Tesseract Decoder is not an official Google product.
Copyright 2025 Google LLC.
