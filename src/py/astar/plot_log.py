import sys
import os
import matplotlib.pyplot as plt
import numpy as np

def analyze_log(filename):
    # Preemptively check if the file exists
    if not os.path.isfile(filename):
        print(f"File not found: {filename}")
        return

    min_masses = []
    errors = []
    
    current_errors = 0
    current_low_conf = 0
    
    pending_error_diff = None
    pending_low_conf_diff = None
    
    # Parse the log file line by line
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
                
            try:
                if parts[0] == "num_shots":
                    # Find 'num_errors' and 'num_low_confidence' and grab the values
                    idx_err = parts.index("num_errors")
                    errs = int(parts[idx_err + 2])
                    
                    idx_lc = parts.index("num_low_confidence")
                    lc = int(parts[idx_lc + 2])
                    
                    # Calculate diffs for this specific shot
                    pending_error_diff = errs - current_errors
                    pending_low_conf_diff = lc - current_low_conf
                    
                    current_errors = errs
                    current_low_conf = lc
                    
                elif parts[0] == "branch_masses":
                    obs0 = float(parts[1].split("=")[1])
                    obs1 = float(parts[2].split("=")[1])
                    
                    # Override if it was flagged as a low confidence shot
                    if pending_low_conf_diff is not None and pending_low_conf_diff > 0:
                        obs0 = 0.5
                        obs1 = 0.5
                        # Count the low confidence increment as additional logical errors
                        pending_error_diff += pending_low_conf_diff
                    else:
                        norm = obs0 + obs1
                        if norm == 0:
                            obs0 = 0.5
                            obs1 = 0.5
                        else:
                            obs0 /= norm
                            obs1 /= norm
                    
                    # Only append if we just successfully parsed a num_shots line
                    if pending_error_diff is not None:
                        min_masses.append(min(obs0, obs1))
                        errors.append(pending_error_diff)
                        
                        # Reset pending diffs to ensure we don't double-count
                        pending_error_diff = None
                        pending_low_conf_diff = None
                        
            except Exception:
                # If anything fails (IndexError, ValueError, etc.) due to a malformed line, 
                # immediately break and assume it's the end of the file.
                break

    min_masses = np.array(min_masses)
    errors = np.array(errors)
    
    if len(min_masses) == 0:
        print("No valid shot data found in the file.")
        return

    # To calculate how error rates change based on our cutoff, 
    # we sort the shots from most certain (lowest min_mass) to least certain.
    sorted_idx = np.argsort(min_masses)
    sorted_masses = min_masses[sorted_idx]
    sorted_errors = errors[sorted_idx]

    N = len(sorted_masses)
    
    # K represents the number of shots we *accept* (1 to N)
    K_arr = np.arange(1, N + 1)
    
    # Cumulative errors in the accepted subset of shots
    accepted_errors = np.cumsum(sorted_errors)
    
    # Error rate = (errors in accepted subset) / (number of accepted shots)
    error_rates = accepted_errors / K_arr
    
    # Rejection rate = (number of rejected shots) / (total shots)
    rejection_rates = (N - K_arr) / N

    # ------------------
    # Pre-process for Log Scale Histogram
    # ------------------
    # Find the smallest non-zero mass. If everything is 0, default to 1e-10
    if np.any(min_masses > 0):
        min_nonzero = np.min(min_masses[min_masses > 0])
        # Set exact 0s to half the minimum non-zero value so they fall in the leftmost bin
        epsilon = min_nonzero / 2.0 
    else:
        epsilon = 1e-10

    # Replace 0s with epsilon
    masses_for_hist = np.where(min_masses == 0, epsilon, min_masses)
    
    # Safely get max mass to define bin edges
    max_mass = np.max(masses_for_hist)
    if max_mass == epsilon:
        max_mass = epsilon * 10 # Fallback in case all values were 0
        
    # Generate 50 logarithmically spaced bins
    log_bins = np.logspace(np.log10(epsilon), np.log10(max_mass), 50)

    # ------------------
    # Create the Figures
    # ------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Distribution of min masses (Log Scale X)
    axes[0].hist(masses_for_hist, bins=log_bins, color='skyblue', edgecolor='black')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Min Mass (Log Scale, 0s in leftmost bin)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Min Masses')

    # Plot 2: Logical error rate vs Min Mass Cutoff
    axes[1].plot(sorted_masses, error_rates, color='purple', lw=2)
    axes[1].set_xlabel('Min Mass Cutoff (Threshold)')
    axes[1].set_ylabel('Logical Error Rate (Accepted Shots)')
    axes[1].set_title('Error Rate vs Min Mass Cutoff')
    axes[1].grid(True, linestyle='--', alpha=0.7)

    # Plot 3: Logical error rate vs Rejection rate
    axes[2].plot(rejection_rates, error_rates, color='red', lw=2)
    axes[2].set_xlabel('Rejection Rate')
    axes[2].set_ylabel('Logical Error Rate (Accepted Shots)')
    axes[2].set_title('Error Rate vs Rejection Rate')
    axes[2].grid(True, linestyle='--', alpha=0.7)
    axes[2].set_xlim(0, 1)

    plt.tight_layout()
    
    # Generate output filename based on input filename
    base_name = os.path.splitext(os.path.basename(filename))[0]
    out_filename = f"{base_name}_analysis.png"
    
    # Save to disk instead of displaying
    plt.savefig(out_filename, dpi=300, bbox_inches='tight')
    print(f"Success! Plot saved to disk as: {out_filename}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <log_file.txt>")
    else:
        analyze_log(sys.argv[1])
