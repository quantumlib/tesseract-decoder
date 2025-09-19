## Python Interface

### `tesseract_decoder.tesseract` Module
The `tesseract_decoder.tesseract` module provides the Tesseract decoder, which employs the A* search to decode a most-likely error configuration from the measured syndrome.

#### Class `tesseract.TesseractConfig`
This class holds the configuration parameters that control the behavior of the Tesseract decoder.
* `TesseractConfig(dem: stim.DetectorErrorModel, det_beam: int = INF_DET_BEAM, beam_climbing: bool = False, no_revisit_dets: bool = False, verbose: bool = False, pqlimit: int = sys.maxsize, det_orders: list[list[int]] = [], det_penalty: float = 0.0)`
* `__str__()`

Explanation of configuration arguments:
* `dem`: This is a required argument that takes a `stim.DetectorErrorModel`. It provides the logical structure of the quantum error-correcting code, including the detectors and the relationships between them. This model is essential for the decoder to understand the syndrome and potential error locations.
* `det_beam` - This integer value represents the beam search cutoff. It specifies a threshold for the number of "residual detection events" a node can have before it is pruned from the search. A lower `det_beam` value makes the search more aggressive, potentially sacrificing accuracy for speed. The default value `INF_DET_BEAM` means no beam cutoff is applied.
* `beam_climbing` - A boolean flag that, when set to `True`, enables a heuristic called "beam climbing." This optimization causes the decoder to try different `det_beam` values (up to a maximum) to find a good decoding path. This can improve the decoder's chance of finding the most likely error, even with an initial narrow beam search.
* `no_revisit_dets` - A boolean flag that, when `True`, activates a heuristic to prevent the decoder from revisiting nodes that have the same set of leftover detection events as a node it has already visited. This can help to reduce search redundancy and improve decoding speed.

* `verbose` - A boolean flag that, when `True`, enables verbose logging. This is useful for debugging and understanding the decoder's internal behavior, as it will print information about the search process.
* `pqlimit` - An integer that sets a limit on the number of nodes in the priority queue. This can be used to constrain the memory usage of the decoder. The default value is `sys.maxsize`, which means the size is effectively unbounded.
* `det_orders` - A list of lists of integers, where each inner list represents an ordering of the detectors. This is used for "ensemble reordering," an optimization that tries different detector orderings to improve the search's convergence. The default is an empty list, meaning a single, fixed ordering is used.
* `det_penalty` - A floating-point value that adds a cost for each residual detection event. This encourages the decoder to prioritize paths that resolve more detection events, steering the search towards more complete solutions. The default value is `0.0`, meaning no penalty is applied.

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
* `decode_to_errors(syndrome: np.ndarray)`
* `decode_to_errors(syndrome: np.ndarray, det_order: int, det_beam: int)`
* `get_observables_from_errors(predicted_errors: list[int]) -> list[bool]`
* `cost_from_errors(predicted_errors: list[int]) -> float`
* `decode(syndrome: np.ndarray) -> np.ndarray`

Explanation of each method:
#### `decode_to_errors(syndrome: np.ndarray)`

Decodes a single measurement shot to predict a list of errors.

* **Parameters:** `syndrome` is a 1D NumPy array of booleans representing the detector outcomes for a single shot.

* **Returns:** A list of integers, where each integer is the index of a predicted error.

#### `decode_to_errors(syndrome: np.ndarray, det_order: int, det_beam: int)`

An overloaded version of the `decode_to_errors` method that allows for a different decoding strategy.

* **Parameters:**

  * `syndrome` is a 1D NumPy array of booleans representing the detector outcomes for a single shot.

  * `det_order` is an integer that specifies a different ordering of detectors to use for the decoding.

  * `det_beam` is an integer that specifies the beam size to use for the decoding.

* **Returns:** A list of integers, where each integer is the index of a predicted error.

#### `get_observables_from_errors(predicted_errors: list[int]) -> list[bool]`

Converts a list of predicted error indices into a list of flipped logical observables.

* **Parameters:** `predicted_errors` is a list of integers representing the predicted error indices.

* **Returns:** A list of booleans. Each boolean corresponds to a logical observable and is `True` if the observable was flipped, and `False` otherwise.

#### `cost_from_errors(predicted_errors: list[int]) -> float`

Calculates the total logarithmic probability cost for a given set of predicted errors. The cost is a measure of how likely a set of errors is.

* **Parameters:** `predicted_errors` is a list of integers representing the predicted error indices.

* **Returns:** A float representing the total logarithmic probability cost.

#### `decode_from_detection_events(detections: list[int]) -> numpy.ndarray`

This method decodes a single shot from a list of detection events. This is an alternative to the `decode` method that takes a NumPy array.

* **Parameters:** `detections` is a list of integers representing the indices of the detectors that were fired.

* **Returns:** A 1D NumPy array of booleans. Each boolean indicates whether the corresponding logical observable has been flipped by the decoded error.

#### `decode(syndrome: numpy.ndarray) -> numpy.ndarray`

A convenience function that decodes a single shot and returns the flipped logical observables directly. It combines the functionality of `decode_to_errors` and `get_observables_from_errors`.

* **Parameters:** `syndrome` is a 1D NumPy array of booleans representing the detector outcomes for a single shot. The length of the array should match the number of detectors in the DEM.

* **Returns:** A 1D NumPy array of booleans, where each boolean corresponds to a logical observable and is `True` if the observable was flipped.

#### `decode_batch(syndromes: numpy.ndarray) -> numpy.ndarray`

Decodes a batch of shots at once.

* **Parameters:** `syndromes` is a 2D NumPy array of booleans where each row represents a single shot's detector outcomes. The shape should be `(num_shots, num_detectors)`.

* **Returns:** A 2D NumPy array of booleans with the shape `(num_shots, num_observables)`. Each row is the decoder's prediction of which observables were flipped in the shot.

**Example Usage**:

```python
import tesseract_decoder.tesseract as tesseract
import stim
import numpy as np

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

# --- Decode a single shot using detection events (list of integers) ---
detections = [1]
flipped_observables_events = decoder.decode_from_detection_events(detections)
print(f"Decoded (from events) flipped observables for detections {detections}: {flipped_observables_events}")

# Access predicted errors
predicted_errors = decoder.predicted_errors_buffer
print(f"\nPredicted errors after single-shot decode: {predicted_errors}")

# Calculate cost for predicted errors
cost = decoder.cost_from_errors(predicted_errors)
print(f"Cost of predicted errors: {cost}")

# Check the low confidence flag
print(f"Decoder low confidence: {decoder.low_confidence_flag}")

# --- Decode a single shot using a syndrome array (NumPy array of booleans) ---
syndrome_array = np.array([False, True])
flipped_observables_syndrome = decoder.decode(syndrome_array)
print(f"Decoded (from syndrome) flipped observables for syndrome {syndrome_array}: {flipped_observables_syndrome}")

# --- Decode a batch of shots using a syndrome array (2D NumPy array of booleans) ---
syndromes_batch = np.array([[False, True], [True, False]])
flipped_observables_batch = decoder.decode_batch(syndromes_batch)
print(f"Decoded (batch) flipped observables for syndromes:\n{syndromes_batch}\nResult:\n{flipped_observables_batch}")
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
* `decode_to_errors(syndrome: np.ndarray)`
* `get_observables_from_errors(predicted_errors: list[int]) -> list[bool]`
* `cost_from_errors(predicted_errors: list[int]) -> float`
* `decode(syndrome: np.ndarray) -> np.ndarray`

**Example Usage**:

```python
import tesseract_decoder.simplex as simplex
import stim
import tesseract_decoder.common as common
import numpy as np

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
syndrome = np.array([0, 1], dtype=bool)
flipped_observables = decoder.decode(syndrome)
print(f"Flipped observables for syndrome {syndrome.tolist()}: {flipped_observables}")

# Access predicted errors
predicted_error_indices = decoder.predicted_errors_buffer
print(f"Predicted error indices: {predicted_error_indices}")

# Calculate cost from the predicted errors
cost = decoder.cost_from_errors(predicted_error_indices)
print(f"Cost of predicted errors: {cost}")
```

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
* `Error(likelihood_cost: float, detectors: list[int], observables: list[int])`
* `Error(error: stim.DemInstruction)`
* `__str__()`

**Example Usage**:

```python
import tesseract_decoder.common as common
import stim
import math

# Create an empty Error
error = common.Error()
print(f"Error likelihood cost: {error.likelihood_cost}")
print(f"Error symptom detectors: {error.symptom.detectors}")

# Create an Error from a stim.DemInstruction
dem_instruction = stim.DemInstruction(type='error', arg_data=[0.1], target_data=[stim.DemTarget(is_relative_detector_id=True, val=1)])
error2 = common.Error(error=dem_instruction)
print(f"Error likelihood cost: {error2.likelihood_cost}")
print(f"Error symptom detectors: {error2.symptom.detectors}")

# Create an Error with explicit parameters
error3 = common.Error(
    likelihood_cost=-math.log(0.2 / (1 - 0.2)),
    detectors=[1, 2],
    observables=[0],
)
print(f"Error likelihood cost: {error3.likelihood_cost}")
print(f"Error symptom detectors: {error3.symptom.detectors}")
```

#### Functions
* `common.merge_indistinguishable_errors(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel`
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

merged_dem = common.merge_indistinguishable_errors(original_dem)
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

### Sinter Integration
The Tesseract Python interface is compatible with the Sinter framework, which is a powerful tool for large-scale decoding, benchmarking, and error-rate estimation.

#### The TesseractSinterDecoder Object
All Sinter examples rely on this utility function to provide the Sinter-compatible Tesseract decoder.

```python
import sinter
import stim
from sinter._decoding._decoding import sample_decode

from src.tesseract_decoder import tesseract_sinter_compat as tesseract_module
from src import tesseract_decoder

# Define a function that returns a dictionary mapping a decoder name to its
# Sinter-compatible decoder object.
def get_tesseract_decoder_for_sinter():
    return tesseract_module.make_tesseract_sinter_decoders_dict()
```

#### Decoding with `sinter.collect`
`sinter.collect` is a powerful function for running many decoding jobs in parallel and collecting the results for large-scale benchmarking.

```python
# Create a repetition code circuit to test the decoder.
circuit = stim.Circuit.generated(
    'repetition_code:memory',
    distance=3,
    rounds=3,
    after_clifford_depolarization=0.01
)

# Use sinter.collect to run the decoding task.
results, = sinter.collect(
    num_workers=1,
    tasks=[sinter.Task(circuit=circuit)],
    decoders=["tesseract"],
    max_shots=1000,
    custom_decoders=get_tesseract_decoder_for_sinter(),
)

# Print a summary of the decoding results.
print("Basic Repetition Code Decoding Results:")
print(f"Shots run: {results.shots}")
print(f"Observed errors: {results.errors}")
print(f"Logical error rate: {results.errors / results.shots}")
```

#### Running with multiple workers
This example demonstrates how to use multiple worker threads to speed up the simulation.
```python
# Use sinter.collect with multiple workers for faster decoding.
results, = sinter.collect(
    num_workers=4,
    tasks=[sinter.Task(circuit=circuit)],
    decoders=["tesseract"],
    max_shots=10000,
    custom_decoders=get_tesseract_decoder_for_sinter(),
)

print("\nDecoding with 4 worker threads:")
print(f"Shots run: {results.shots}")
print(f"Observed errors: {results.errors}")
print(f"Logical error rate: {results.errors / results.shots}")
```

#### Decoding with `sinter.sample_decode`
`sinter.sample_decode` is a simpler, non-parallel function for directly decoding a single circuit. It's useful for quick tests and debugging without the overhead of the `sinter.collect` framework.

```python
# Create a repetition code circuit.
circuit = stim.Circuit.generated('repetition_code:memory', distance=5, rounds=5)

# Use sinter.sample_decode for a direct decoding run.
result = sample_decode(
    circuit_obj=circuit,
    dem_obj=circuit.detector_error_model(),
    num_shots=1000,
    decoder="tesseract",
    custom_decoders=get_tesseract_decoder_for_sinter(),
)

print("Basic sample_decode Results:")
print(f"Shots run: {result.shots}")
print(f"Observed errors: {result.errors}")
print(f"Logical error rate: {result.errors / result.shots}")
```