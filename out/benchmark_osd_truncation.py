import time
import stim
import numpy as np
import sinter
import tesseract_decoder.bp_sinter_compat as bp_sinter_compat
import tesseract_decoder.bp as bp

# A custom Sinter Decoder wrapper to pass dynamic OSD truncation factor
class TruncationDecoder(sinter.Decoder):
    def __init__(self, truncation_factor: float):
        self.truncation_factor = truncation_factor

    def decode_via_files(self, **kwargs):
        raise NotImplementedError()
        
    def compile_decoder_for_dem(self, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        params = bp.BPParams()
        params.max_iter = 30
        params.update_rule = "min-sum"
        params.schedule = "parallel"
        params.osd_truncation_factor = self.truncation_factor
        return bp_sinter_compat.TesseractBpSinterDecoder(params, 10, 0, True).compile_decoder_for_dem(dem=dem)

def run_benchmark():
    distance = 11
    rounds = 11
    
    # We use a custom noise model with heterogeneous noise (from 0.001 to 0.01)
    # This prevents artificial ties in LLR values, allowing OSD to pick bases practically.
    print(f"Generating Heterogeneous Surface Code (d={distance}, r={rounds})...")
    
    # Generate baseline circuit WITH noise so it actually creates DEPOLARIZE instructions
    circuit = stim.Circuit.generated(
        "surface_code:unrotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=0.005
    )
    
    # Inject heterogeneous noise to operations
    np.random.seed(42)
    noisy_circuit = stim.Circuit()
    for inst in circuit:
        if inst.name in ("DEPOLARIZE1", "X_ERROR", "Y_ERROR", "Z_ERROR"):
            noisy_circuit.append(inst.name, inst.targets_copy(), np.random.uniform(0.01, 0.07))
        elif inst.name == "DEPOLARIZE2":
            noisy_circuit.append("DEPOLARIZE2", inst.targets_copy(), np.random.uniform(0.01, 0.07))
        else:
            noisy_circuit.append(inst)

    task = sinter.Task(
        circuit=noisy_circuit,
        json_metadata={'d': distance, 'r': rounds}
    )
    
    # We will run 1000 shots across different factors
    num_shots = 1000
    
    # Test factors: 0.0 (no truncation, uses all 53,782 columns), 1.05 (2420 * 1.05 = 2541), 1.1, 1.2
    factors = [0.0, 1.05, 1.10, 1.20]
    
    print(f"\n--- Running OSD Truncation Benchmark (Shots={num_shots}) ---")
    dem = noisy_circuit.detector_error_model()
    print(f"Circuit DEM Errors (Columns): {sum(1 for inst in dem.flattened() if inst.type == 'error')}")
    print(f"Circuit DEM Detectors (Rows): {dem.num_detectors}")
    print("-" * 60)
    
    for factor in factors:
        factor_str = f"factor_{factor:.2f}" if factor > 0 else "all_columns"
        
        t0 = time.time()
        stats = sinter.collect(
            num_workers=1,
            max_shots=num_shots,
            max_errors=num_shots,
            tasks=[task],
            decoders=[factor_str],
            custom_decoders={
                factor_str: TruncationDecoder(factor)
            },
            print_progress=False
        )
        t1 = time.time()
        
        stat = stats[0]
        ler = stat.errors / stat.shots
        
        num_errors = sum(1 for inst in dem.flattened() if inst.type == "error")
        num_cols = int(dem.num_detectors * factor) if factor > 0 else num_errors
        print(f"[OSD Columns: {num_cols:5d} | Factor: {factor:.2f}] LER: {ler:.5f} ({stat.errors} errors) | Time: {t1 - t0:.1f} s")

if __name__ == "__main__":
    run_benchmark()
