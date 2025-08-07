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

