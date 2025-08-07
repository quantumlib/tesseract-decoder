<div align="center">

# [Tesseract Decoder](https://quantumlib.github.io/tesseract-decoder)

A Search-Based Decoder for Quantum Error Correction.

[![Licensed under the Apache 2.0 open-source license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative\&logoColor=white\&style=flat-square)](https://github.com/quantumlib/tesseract-decoder/blob/main/LICENSE)
![C++](https://img.shields.io/badge/C++-20-fcbc2c?style=flat-square&logo=C%2B%2B&logoColor=white)

[Installation](#installation) &ndash;
[Usage](#usage) &ndash;
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

### `tesseract_decoder.utils` Module
The `tesseract_decoder.utils` module provides various helper functions used throughout the entire project.

#### Functions
* `utils.get_detector_coords(dem: stim.DetectorErrorModel) -> list[list[float]]`
  * Extracts 3D coordinates for each detector from a `stim.DetectorErrorModel`.

**Example Usage**:

```python
import tesseract_decoder.utils as utils
import stim

dem = stim.DetectorErrorModel("""
    detector(0, 0, 0) D0
    detector(1, 0, 0) D1
    detector(0, 1, 0) D2
""")
coords = utils.get_detector_coords(dem)
print("Detector Coordinates:")
for i, coord in enumerate(coords):
    print(f"D{i}: ({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f})")
```

* `utils.build_detector_graph(dem: stim.DetectorErrorModel) -> list[list[int]]`
  * Builds an adjacency list representation of a graph where detectors are nodes. An edge exists between two detectors if a single error mechanism in the DEM can activate both of them.

**Example Usage**:

```python
import tesseract_decoder.utils as utils
import stim

dem = stim.DetectorErrorModel("""
    error(0.1) D0 D1
    error(0.2) D1 D2
    error(0.3) D0 D2
    detector(0, 0, 0) D0
    detector(1, 0, 0) D1
    detector(0, 1, 0) D2
""")
graph = utils.build_detector_graph(dem)
print("Detector Graph Adjacency List:")
for i, neighbors in enumerate(graph):
    print(f"D{i}: {neighbors}")
```

* `utils.get_errors_from_dem(dem: stim.DetectorErrorModel) -> list[common.Error]`
  * Converts the error mechanisms within a `stim.DetectorErrorModel` into the internal `tesseract_decoder.common.Error` data structure used by the Tesseract decoder.

**Example Usage**:

```python
import tesseract_decoder.utils as utils
import tesseract_decoder.common as common
import stim

dem = stim.DetectorErrorModel("""
    error(0.1) D0 L0
    error(0.05) D1
    error(0) D2
""")
errors = utils.get_errors_from_dem(dem)
print("Errors extracted from DEM:")
for error in errors:
    print(f"Error probability: {error.probability}")
    print(f"Error likelihood cost: {error.likelihood_cost}")
    print(f"Error symptom detectors: {error.symptom.detectors}")
```

### `tesseract_decoder.common` Module
The `tesseract_decoder.common` module provides fundamental data structures and utility functions used for decoding quantum circuit shots inside Tessereact. It exposes classes **`Symptom`** and **`Error`**, which represent error effects and complete error mechanisms, respectively. Additionally, it includes functions for manipulating `stim.DetectorErrorModel` objects, such as merging identical errors, removing zero-probability errors, and estimating error probabilities from shot counts.

#### Class `common.Symptom`
A Python class representing the effect of an error mechanism.

* `Symptom(detectors: list[int] = [], observables: list[int] = [])`
* `__str__()`
* `__eq__(other)`
* `__ne__(other)`
* `as_dem_instruction_targets() -> list[stim.DemTarget]`

**Example Usage**:

```python
import tesseract_decoder.common as common
import stim

# Create a symptom with two detectors and one observable
s = common.Symptom(detectors=[0, 1], observables=[2])

# Access detectors
print(f"Detectors: {s.detectors}")
# Access observables
print(f"Observables: {s.observables}")

# Use as_dem_instruction_targets
targets = s.as_dem_instruction_targets()
print(f"DEM targets: {targets}")

# Demonstrate equality and inequality
s2 = common.Symptom(detectors=[0, 1], observables=[2])
s3 = common.Symptom(detectors=[0, 1, 3], observables=[2])
print(f"s == s2: {s == s2}")
print(f"s != s3: {s != s3}")
```
#### Class `common.Error`
A Python class representing a complete error mechanism.

* `Error()`
* `Error(likelihood_cost: float, detectors: list[int], observables: list[int], dets_array: list[bool])`
* `Error(likelihood_cost: float, probability: float, detectors: list[int], observables: list[int], dets_array: list[bool])`
* `Error(error: stim.DemInstruction)`
* `__str__()`

**Example Usage**:

```python
import tesseract_decoder.common as common
import stim
import math

# Create an empty Error
error = common.Error()
print(f"Error probability: {error.probability}")
print(f"Error likelihood cost: {error.likelihood_cost}")
print(f"Error symptom detectors: {error.symptom.detectors}")

# Create an Error from a stim.DemInstruction
dem_instruction = stim.DemInstruction(type='error', arg_data=[0.1], target_data=[stim.DemTarget(is_relative_detector_id=True, val=1)])
error2 = common.Error(error=dem_instruction)
print(f"Error probability: {error2.probability}")
print(f"Error likelihood cost: {error2.likelihood_cost}")
print(f"Error symptom detectors: {error2.symptom.detectors}")

# Create an Error with explicit parameters
error3 = common.Error(
    likelihood_cost=-math.log(0.2 / (1 - 0.2)),
    probability=0.2,
    detectors=[1, 2],
    observables=[0],
    dets_array=[False, True, True, False]
)
print(f"Error probability: {error3.probability}")
print(f"Error likelihood cost: {error3.likelihood_cost}")
print(f"Error symptom detectors: {error3.symptom.detectors}")
```

#### Functions
* `common.merge_identical_errors(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel`
  * Takes a `stim.DetectorErrorModel` and returns a new model where error mechanisms with identical symptoms are combined.

**Example Usage**:

```python
import tesseract_decoder.common as common
import stim

original_dem = stim.DetectorErrorModel("""
    error(0.1) D0 D1
    error(0.05) D0 D1
    error(0.2) D2
""")
print("Original DEM:")
print(original_dem)
# This DEM has 3 error instructions.
# Two have the same symptom (D0 D1) with probabilities 0.1 and 0.05.
# The third has a different symptom (D2) with probability 0.2.

merged_dem = common.merge_identical_errors(original_dem)
print("\nMerged DEM:")
print(merged_dem)
# This merged DEM has 2 error instructions.
# The two errors with the same symptom D0 D1 have been combined into a single instruction.
# The new probability for the D0 D1 error is 0.1+0.05−(0.1×0.05)=0.145.
# The error with symptom D2 remains unchanged.
```

* `common.remove_zero_probability_errors(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel`
  * Filters a `stim.DetectorErrorModel` to create a new model that excludes errors with a probability of zero.

**Example Usage**:

```python
import tesseract_decoder.common as common
import stim

original_dem = stim.DetectorErrorModel("""
    error(0.1) D0
    error(0) D1
    error(0.2) D2
""")
print("Original DEM:")
print(original_dem)
# This DEM has 3 error instructions, one of them has probability of zero.

cleaned_dem = common.remove_zero_probability_errors(original_dem)
print("\Cleaned DEM:")
print(cleaned_dem)
# This DEM has 2 error instructions, none of them have probability of zero.
```

* `common.dem_from_counts(orig_dem: stim.DetectorErrorModel, error_counts: list[int], num_shots: int) -> stim.DetectorErrorModel`
  * Creates a new `stim.DetectorErrorModel` by re-evaluating the probabilities of the original error mechanisms based on experimental counts. The probability for each error is calculated as `error_counts[i] / num_shots`.

**Example Usage**:

```python
import tesseract_decoder.common as common
import stim

# Original DEM
original_dem = stim.DetectorErrorModel("""
    error(0.1) D0
    error(0.2) D1
    error(0.05) D2
""")
print("Original DEM:")
print(original_dem)

# Simulate some error counts from 1000 shots
# Error at D0 occurred 100 times, D1 occurred 250 times, D2 occurred 40 times
error_counts = [100, 250, 40]
num_shots = 1000

estimated_dem = common.dem_from_counts(original_dem, error_counts, num_shots)
print("\nEstimated DEM:")
print(estimated_dem)
# Expected probabilities: D0 -> 100/1000 = 0.1, D1 -> 250/1000 = 0.25, D2 -> 40/1000 = 0.04
```

### `tesseract_decoder.tesseract` Module
The `tesseract_decoder.tesseract` module provides the Tesseract decoder, which employs the A* search to decode a most-likely error configuration from the measured syndrome.

#### Class `tesseract.TesseractConfig`
This class holds the configuration parameters that control the behavior of the Tesseract decoder.
* `TesseractConfig(dem: stim.DetectorErrorModel, det_beam: int = INF_DET_BEAM, beam_climbing: bool = False, no_revisit_dets: bool = False, at_most_two_errors_per_detector: bool = False, verbose: bool = False, pqlimit: int = sys.maxsize, det_orders: list[list[int]] = [], det_penalty: float = 0.0)`
* `__str__()`

**Example Usage**:

```python
import tesseract_decoder.tesseract as tesseract
import stim
import sys

dem = stim.DetectorErrorModel("""
    error(0.1) D0 D1
    error(0.2) D1 D2 L0
    detector(0, 0, 0) D0
    detector(1, 0, 0) D1
    detector(2, 0, 0) D2
""")

# Basic configuration
config1 = tesseract.TesseractConfig(dem=dem)
print(f"Basic configuration detection beam: {config1.det_beam}")
print(f"Basic configuration beam climbing: {config1.det_beam}")
print(f"Basic configuration no-revisit detection events: {config1.det_beam}")
print(f"Basic configuration pqlimit: {config1.det_beam}")
print(f"Basic configuration verbose: {config1.det_beam}")
print(f"Basic configuration detection penalty: {config1.det_beam}")

# Configuration with custom parameters
config2 = tesseract.TesseractConfig(
    dem=dem,
    det_beam=50,
    beam_climbing=True,
    no_revisit_dets=True,
    pqlimit=10000,
    verbose=True,
    det_penalty=0.1
)
print(f"Custom configuration detection beam: {config2.det_beam}")
print(f"Custom configuration beam climbing: {config2.det_beam}")
print(f"Custom configuration no-revisit detection events: {config2.det_beam}")
print(f"Custom configuration pqlimit: {config2.det_beam}")
print(f"Custom configuration verbose: {config2.det_beam}")
print(f"Custom configuration detection penalty: {config2.det_beam}")
```

#### Class `tesseract.TesseractDecoder`
This is the main class that implements the Tesseract decoding logic.
* `TesseractDecoder(config: tesseract.TesseractConfig)`
* `decode_to_errors(detections: list[int])`
* `decode_to_errors(detections: list[int], detector_order: int, detector_beam: int)`
* `get_observables_from_errors(predicted_errors: list[int]) -> list[bool]`
* `cost_from_errors(predicted_errors: list[int]) -> float`
* `decode(detections: list[int]) -> list[bool]`

**Example Usage**:

```python
import tesseract_decoder.tesseract as tesseract
import stim
import tesseract_decoder.common as common

# Create a DEM and a configuration
dem = stim.DetectorErrorModel("""
    error(0.1) D0
    error(0.2) D1 L0
    detector(0, 0, 0) D0
    detector(1, 0, 0) D1
""")
config = tesseract.TesseractConfig(dem=dem)

# Create the decoder
decoder = tesseract.TesseractDecoder(config)

# Decode the detections and get flipped observables
detections = [1]
flipped_observables = decoder.decode(detections)
print(f"Flipped observables for detections {detections}: {flipped_observables}")

# Access predicted errors after decoding
predicted_errors = decoder.predicted_errors_buffer
print(f"Predicted errors: {predicted_errors}")

# Calculate cost for predicted errors
cost = decoder.cost_from_errors(predicted_errors)
print(f"Cost of predicted errors: {cost}")

# Check the low confidence flag
print(f"Decoder low confidence: {decoder.low_confidence_flag}")
```

### `tesseract_decoder.simplex` Module
The `tesseract_decoder.simplex` module provides the Simplex-based decoder, which solves the decoding problem using an integer linear program.

#### Class `simplex.SimplexConfig`
This class holds the configuration parameters that control the behavior of the Simplex decoder.
* `SimplexConfig(dem: stim.DetectorErrorModel, parallelize: bool = False, window_length: int = 0, window_slide_length: int = 0, verbose: bool = False)`
* `__str__()`
* `windowing_enabled() -> bool`

**Example Usage**:

```python
import tesseract_decoder.simplex as simplex
import stim

dem = stim.DetectorErrorModel("""
    # Example DEM
    error(0.1) D0
    error(0.2) D1 L0
""")

config = simplex.SimplexConfig(
    dem=dem,
    parallelize=False,
    window_length=10,
    window_slide_length=5,
    verbose=True
)

print(f"Configuration parallelize enabled: {config.parallelize}");
print(f"Configuration window length: {config.window_length}")
print(f"Configuration window slide length: {config.window_length}")
print(f"Configuration windowing enabled: {config.windowing_enabled()}")
print(f"Configuration verbose enabled: {config.verbose}")
```

#### Class `simplex.SimplexDecoder`
This is the main class for performing decoding using the Simplex algorithm.
* `SimplexDecoder(config: simplex.SimplexConfig)`
* `init_ilp()`
* `decode_to_errors(detections: list[int])`
* `get_observables_from_errors(predicted_errors: list[int]) -> list[bool]`
* `cost_from_errors(predicted_errors: list[int]) -> float`
* `decode(detections: list[int]) -> list[bool]`

**Example Usage**:

```python
import tesseract_decoder.simplex as simplex
import stim
import tesseract_decoder.common as common

# Create a DEM and a configuration
dem = stim.DetectorErrorModel("""
    error(0.1) D0
    error(0.2) D1 L0
    detector(0, 0, 0) D0
    detector(1, 0, 0) D1
""")
config = simplex.SimplexConfig(dem=dem)

# Create and initialize the decoder
decoder = simplex.SimplexDecoder(config)
decoder.init_ilp()

# Decode a shot where detector D1 fired
detections = [1]
flipped_observables = decoder.decode(detections)
print(f"Flipped observables for detections {detections}: {flipped_observables}")

# Access predicted errors
predicted_error_indices = decoder.predicted_errors_buffer
print(f"Predicted error indices: {predicted_error_indices}")

# Calculate cost from the predicted errors
cost = decoder.cost_from_errors(predicted_error_indices)
print(f"Cost of predicted errors: {cost}")
```


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

For any questions or concerns not addressed here, please email <quantum-oss-maintainers@google.com>.

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
