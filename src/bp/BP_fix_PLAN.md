# Tesseract-BP: Deep Analysis of Belief Propagation and Ordered Statistic Decoding

## 1. Introduction

This document provides a comprehensive technical analysis of the advanced Belief Propagation (BP) features and Ordered Statistic Decoding (OSD) integration within the Tesseract-BP codebase. Tesseract-BP is a sophisticated Quantum Error Correction (QEC) framework. It fundamentally operates as an A*-based Search Decoder augmented with highly optimized heuristic and deterministic sub-routines. Crucially, this is a dedicated QEC engine operating on Tanner graphs derived from physical quantum circuits and is fundamentally distinct from any Optical Character Recognition (OCR) systems.

The core decoding mechanism relies on the `TesseractBpDecoder`, which employs a Min-Sum Belief Propagation algorithm iteratively operating over a bipartite Tanner Graph. The structure of this Tanner Graph explicitly maps check nodes (corresponding to syndrome detectors) to variable nodes (corresponding to discrete physical errors). The decoder architecture is instantiated and parameterized dynamically from a `stim::DetectorErrorModel` (DEM), natively coupling it to standard quantum circuit simulation output.

## 2. High-Level Architecture and Data Flow

1. **Graph Construction**: The decoder ingests a `stim::DetectorErrorModel` and converts it into a Bipartite **Tanner Graph** (`src/bp/tesseract_bp_decoder.cc`), mapping detection events (check nodes) to errors (variable nodes).
2. **Batched Execution (SIMD)**: The entry point (`src/bp/bp_main.cc`) chunks execution into batches of 64 shots. `TesseractBpDecoder::decode_batch` further subdivides this into micro-batches defined by `BP_BATCH_SIZE = 16`. 
3. **Belief Propagation (BP)**: Inside `batched_bp_parallel_min_sum.inl`, the algorithm computes variable-to-check and check-to-variable messages. The batch size of 16 aligns perfectly with 512-bit vector registers (AVX-512) for 32-bit `LLR_INT` types, enabling simultaneous computation of 16 independent decoding problems.
4. **Integration with OSD**: After the BP iterations complete or converge, the execution is handed over to a `PostProcessor` interface.

## 3. Belief Propagation Core Architecture

The `TesseractBpDecoder` provides the foundation for the heuristic soft-decision decoding.

### 3.1 Min-Sum Algorithm and Tanner Graph Representation
The foundational Belief Propagation algorithm utilizes the Min-Sum update rule, an approximation of the Sum-Product algorithm that significantly reduces computational complexity while maintaining comparable decoding thresholds. The Tanner Graph representation is critical:
*   **Variable Nodes ($V$):** Represent the potential physical errors. Each variable node $v_i \in V$ is initialized with a prior probability derived from the DEM error probabilities.
*   **Check Nodes ($C$):** Represent the syndrome detectors. Each check node $c_j \in C$ enforces a parity constraint dictated by the measurement outcomes.

Messages are iteratively passed along the edges of the Tanner Graph. The logarithmic likelihood ratio (LLR) is used to represent message values, facilitating numerical stability. The prior LLR for an error with probability $p$ is $\ln((1-p)/p)$.

### 3.2 BP Scheduling Modalities
The scheduling of message passing critically impacts convergence speed and decoding dynamics. Tesseract-BP implements two distinct scheduling algorithms, configurable via the `--schedule` CLI argument:

1.  **Synchronous (Parallel) Schedule (`--schedule parallel`):**
    This schedule executes a purely parallel update sequence in two distinct phases per iteration:
    *   **Phase 1 (Variable-to-Check):** All messages from variable nodes to check nodes ($V \rightarrow C$) are computed concurrently using the posterior values from the previous iteration.
    *   **Phase 2 (Check-to-Variable):** All messages from check nodes to variable nodes ($C \rightarrow V$) are updated based on the newly computed $V \rightarrow C$ messages.
    This method allows for massive parallelism but may suffer from slower convergence rates due to delayed propagation of high-confidence beliefs.

2.  **Asynchronous (Serial) Schedule (`--schedule serial` - Default):**
    This schedule employs an asynchronous sequential update over the variable nodes. It iterates through the variable nodes $v_i \in V$ one by one. For each node $v_i$:
    *   The posterior belief of $v_i$ is immediately updated based on its currently available $C \rightarrow V$ messages.
    *   The outgoing $V \rightarrow C$ messages from $v_i$ to its neighboring check nodes are recalculated.
    This sequential propagation ensures that subsequent variable node updates within the same iteration benefit from the most recent information, typically leading to significantly faster convergence compared to the parallel schedule.

### 3.3 AVX-512 Vectorized Batching
To achieve maximum throughput, the BP decoder features highly optimized AVX-512 Vectorized Batching, enabled via the `--batched` flag.
*   **Outer Loop Dynamics:** The outer loop of the batched execution processes large blocks of 64 independent quantum shots concurrently.
*   **Inner SIMD Processing (`decode_batch`):** Internally, the `decode_batch` function slices the 64-shot block into highly dense, SIMD-friendly micro-batches of 16 (defined by `BP_BATCH_SIZE=16`). Given 32-bit floating-point representations for LLRs, this exactly maps to the 512-bit width of AVX-512 registers ($16 \times 32 = 512$ bits).
*   **Compiler Optimization:** The codebase utilizes tight loops heavily annotated with `#pragma GCC ivdep` (ignore vector dependencies) to aggressively force the compiler to generate contiguous packed AVX-512 instructions, overriding conservative dependency analysis and maximizing instruction-level parallelism.

### 3.4 Per-Shot Early Termination
Optimizing execution time for batches where individual shots exhibit varying convergence rates, Tesseract-BP implements an advanced early termination mechanism.
*   **Active Shots Bitmask:** A specialized bitmask, `active_shots`, tracks the convergence status of each shot within the SIMD batch.
*   **Convergence Criterion:** A shot is marked as converged when all its corresponding parity check constraints are satisfied (i.e., the hard-decision assignment yields a zero residual syndrome).
*   **Computation Bypass:** Converged shots dynamically bypass all subsequent message-passing calculations within the iteration loop. This minimizes redundant computation and drastically reduces the average decoding time per batch, especially in low-error regimes where a subset of shots converges rapidly.

## 4. The BP to OSD Interface

The critical integration point is in `TesseractBpDecoder::decode_batch`. If the user specifies an `--osd-order`, the `post_processor` injected is the `OsdPostProcessor` (`src/bp/osd_post_processor.cc`). When BP fails to converge perfectly, the OSD logic operates as a powerful algebraic fallback mechanism:

1. **Residual Syndrome Calculation**: Computes the residual syndrome after taking hard decisions based on the BP posteriors.
2. **Reliability Sorting**: Sorts the variable nodes (errors) based on the absolute magnitude of their BP posteriors (reliabilities). Higher magnitude indicates higher confidence.
3. **Information Set Generation (Gaussian Elimination)**: Utilizes `stim::simd_bits` to perform column-pivoted Gaussian elimination over the parity check matrix, yielding a full-rank parity basis.
4. **Perturbation Search**: Searches perturbation patterns (up to the defined `osd_weight_` parameter) among the least reliable free columns outside the information set to find a correction with the minimum weight/cost.
5. **Output Merge**: Merges the initial BP hard decision with the best OSD correction.

## 5. Ordered Statistic Decoding (OSD) Post-Processing Details

This section elaborates on the underlying mathematics and operations briefly mentioned in the interface.

### 5.1 Reliability Ordering
The fundamental principle of OSD is to leverage the soft information (posteriors) derived from the failed BP convergence.
*   **Sorting by LLR Magnitude:** The variable nodes (errors) are ordered by their reliability. The reliability is quantified as the absolute value of their BP posterior LLRs ($|LLR|$). 

### 5.2 Information Set Generation
OSD constructs an *information set*—a basis spanning the required parity check space.
*   **Gaussian Elimination with Column Pivoting:** The algorithm performs Gaussian elimination on the parity check matrix (H-matrix) columns, ordered by reliability (most reliable first). Column pivoting ensures that the most reliable linearly independent columns are selected to form a full-rank submatrix.
*   **`stim::simd_bits` Utilization:** The heavy linear algebraic operations, specifically the row operations during Gaussian elimination, are highly optimized using the `stim::simd_bits` data structures, exploiting wide bitwise SIMD operations for maximum efficiency.

### 5.3 Perturbation and Re-encoding
Once the information set (basis) is formed, the algorithm generates correction candidates.
*   **OSD Order (`--osd-order`):** Specifies the maximum Hamming weight of perturbation patterns applied to the bits *outside* the information set (the less reliable bits).
*   **Perturbation Weights (`--osd-weight`):** Tesseract-BP supports OSD-0 (no perturbations), OSD-1 (single-bit perturbations), and OSD-2 (two-bit perturbations).
*   **Candidate Evaluation:** For each perturbation pattern, the bits within the information set are deterministically calculated (re-encoded) to satisfy the residual syndrome. The overall candidate correction is formed, and the correction with the minimum total weight (or highest total probability) is selected as the final decoding solution.

## 6. Command Line Interface (CLI) Configuration

Tesseract-BP provides an extensive CLI to parameterize the execution environment, input/output data flows, and internal algorithmic behaviors.

### 6.1 Data Input and Output Options
*   `--circuit <path>`: Specifies the path to the physical Stim circuit file.
*   `--dem <path>`: Specifies the path to the Stim Detector Error Model file, fundamentally defining the Tanner Graph structure.
*   `--sample-num-shots`: Determines the number of Monte Carlo shots to execute during sampling modes.
*   `--max-errors`: Defines an early exit threshold for sampling based on accumulated logical errors.
*   `--sample-seed`: Seeds the pseudo-random number generator for deterministic execution.
*   `--in`, `--in-format`: Defines the path and format (e.g., `b8`) of the input syndrome data.
*   `--obs_in`, `--obs-in-format`: Defines the path and format of the input observable data (used for validation).
*   `--out`, `--out-format`: Defines the path and format of the decoding results output.

### 6.2 Belief Propagation and OSD Core Options
*   `--max-iter <int>`: The maximum number of BP message passing iterations allowed before declaring non-convergence. Default is `20`.
*   `--update-rule <string>`: Specifies the message update algebra. Default is `"min-sum"`.
*   `--schedule <string>`: Selects the scheduling modality. Options are `"serial"` or `"parallel"`. Default is `"serial"`.
*   `--normalization-factor <float>`: A damping factor applied to messages to prevent oscillation and aid convergence in graphs with short cycles. Default is `0.625`.
*   `--batched`: A boolean flag to enable the high-performance AVX-512 vectorized batching execution path.
*   `--osd-order <int>`: Determines the post-processing behavior. `-1` specifies purely hard decision (no OSD). Values `>= 0` enable OSD with the corresponding perturbation search depth.
*   `--osd-weight <int>`: Explicitly limits the OSD perturbation weight. Supported values are `0`, `1`, and `2`, corresponding to OSD-0, OSD-1, and OSD-2 respectively.
*   `--threads <int>`: Sets the number of concurrent execution threads, controlling hardware-level concurrency for batch processing.

### 6.3 Logging and Telemetry Options
*   `--stats-out`: Specifies the output path for generic decoding statistics.
*   `--sinter-csv-out`: Specifies the output path for statistics formatted as a CSV compatible with the `sinter` analysis framework.
*   `--verbose`: Enables highly detailed standard error logging of internal decoder states.
*   `--print-stats`: A boolean flag to print summary statistics to standard output upon completion.

### Follow-up Architectural Clarifications

1. **Tesseract as BP Post-Processor:** Tesseract Search Decoder is purely a standalone entity. There is no option to use it as a post-processor for Belief Propagation. The `PostProcessor` interface in `bp_main.cc` solely branches into `OsdPostProcessor` (if `--osd-order >= 0`) or `HardDecisionPostProcessor` otherwise.
2. **`--osd-order` vs `--osd-weight`:** Both parameterize OSD, but differ in dimension. `--osd-order` restricts the size of the candidate pool by picking the N least reliable "free columns" from the Gaussian Elimination basis. `--osd-weight` configures the Hamming weight—the maximum number of simultaneous bit flips applied to those chosen candidates (OSD-0, OSD-1, OSD-2).
3. **`--batched` vs `--threads`:** `--threads` handles high-level multi-core concurrency using standard `std::thread`, dispatching independent shots to separate CPU cores. `--batched` handles low-level SIMD pipeline vectorization, operating on batches of 16 using `#pragma GCC ivdep` so a single thread processes 16 shots concurrently down the AVX-512 pipeline.
4. **Serial Scheduling Behavior:** In both `bp_serial_min_sum.inl` and `batched_bp_serial_min_sum.inl`, the serial schedule is strictly fixed and linear. The variables are visited sequentially from `0` to `num_variables - 1` across every iteration. It does not randomize or shuffle the graph traversal order per iteration.

### Proposed Performance Optimization Plan

1. **Memory Foundation Rewrite**: Transition `posteriors_batch` into an interleaved 1-Dimensional flat array to instantly resolve the L1 cache flushing. This is a strict prerequisite for any stochastic scheduling.
2. **Horizontal (Layered) Scheduling implementation**: Process the graph iteratively over check nodes. Compute V->C messages dynamically on the fly (`L_v - R_{c,v}`), deleting the `var_to_check_messages` array altogether. This saves 50% CPU memory bandwidth.
3. **Explicit AVX-512 Intrinsics**: Move away from `#pragma GCC ivdep` for tracking minimums (`min1` and `min2`). Subbing in explicit `_mm512_min_epi32` instructions will strip conditional logic latency in the core math loops.
4. **Viability of Randomized Horizontal Serial Schedule**: With the flat memory array and 50% bandwidth reduction accomplished above, applying a fast PRNG (like `xorshift`) to perturb the check node traversal sequence is **highly viable**. The random jump penalty drops drastically (since the footprint is halved), and the ability to break QEC trapping sets allows it to converge much faster.
5. **Expected Speedups**: 
    - The Layered schedule natively halves the iterations required to converge (**2x speedup**).
    - Reduced memory payload translates directly to raw throughput since BP is memory-bound (**~1.5x - 2x speedup**). 
    - Combined, these architectures project an overall **3x to 5x end-to-end simulation acceleration** on the decoder.

#### The 10x Speedup Mathematical Myth

* **The 10x Trap:** Although QEC matrices are highly rectangular (10x more Variables than Checks), swapping from a Vertical (Variable) schedule to a Horizontal (Check) schedule does **not** grant a 10x algorithmic speedup.
* **The Mathematics:** The graph is mathematically bounded by its Edges ($E$). The vertical schedule executes a large outer loop ($N \approx 10M$) with a very small inner loop (degree $\approx 3$). The horizontal schedule executes a small outer loop ($M$) with a very large inner loop (degree $\approx 30$). Total math operations equal exactly $O(E)$ in both geometries.
* **The True Root of Hardware Gain:** Reiterate that the true speedup achieved by Horizontal Layered schedules derives strictly from algorithmic convergence (halving iteration counts) and wiping out tracking states for `min1`/`min2`, rather than loop compression.

### Visualizing the Memory Foundation Rewrite (Interleaving)
To understand why SIMD random scheduling currently destroys cache locality, we must observe how the variable arrays (like `posteriors_batch`) are laid out physically in RAM.

#### 1. The Current Layout: Non-Interleaved (Vector of Vectors)
Currently, `tesseract-bp` stores batches as 16 entirely separate, independent arrays in the system heap memory (`std::vector<std::vector<LLR_INT>>`).

```text
[RAM Address Space]
Shot 0:  [ Var0, Var1, Var2, Var3, ..., VarN ]  <-- Array 1
...  (Random RAM gap)
Shot 1:  [ Var0, Var1, Var2, Var3, ..., VarN ]  <-- Array 2
...  (Random RAM gap)
Shot 2:  [ Var0, Var1, Var2, Var3, ..., VarN ]  <-- Array 3
...
Shot 15: [ Var0, Var1, Var2, Var3, ..., VarN ]  <-- Array 16
```
**SIMD Penalty:** When a random schedule requests **Variable 3**, the CPU must load Variable 3 for all 16 shots simultaneously into its 512-bit register. The CPU physically jumps to 16 completely disconnected locations in RAM, loading 16 full cache lines (1,024 bytes transferred) just to extract 64 bytes of data. The L1 Cache is flushed instantly.

#### 2. The Solution: Interleaved 1D Array
The proposed rewrite allocates one massive, contiguous block of RAM. The variables are interleaved (packed side-by-side) so that all 16 shots for a specific variable always sit physically touching each other.

```text
[RAM Address Space - Single Mass Allocation]

      [---- Variable 0 Block ----]   [---- Variable 1 Block ----]   [---- Variable 2 Block ----] 
RAM:  [ S0, S1, S2, ..., S14, S15 |  S0, S1, S2, ..., S14, S15  |  S0, S1, S2, ..., S14, S15 | ... ]
```
**SIMD Rescue:** When the random schedule requests **Variable 3**, the CPU jumps to the Variable 3 Block. Because all 16 shots (`S0` through `S15`) are perfectly packed, they equate to exactly 64-bytes of consecutive memory. The CPU issues a single SIMD aligned load instruction, fetching exactly one cache line (64 bytes) from RAM. This shields the L1 Cache from random check node hopping penalties!
