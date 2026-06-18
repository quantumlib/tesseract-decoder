#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status (-e),
# treat unset variables as an error (-u), and catch errors in pipes (-o pipefail).
set -euo pipefail

mkdir -p out

TESSERACT_BIN=./bazel-bin/src/tesseract
# Create a timestamp in nanoseconds for when the script starts
STARTTIME=$(($(date +%s%N)))

COUNTER=0

for num in $(seq 0 999); do
  for p_err in 0.001 0.002; do
    for circuit in testdata/bivariatebicyclecodes/r={6,10,12}*p=$p_err,noise=si1000,c=*.stim; do
      echo "$circuit"
    done
    for circuit in testdata/colorcodes/r={3,5,7,9,11}*p=$p_err,noise=si1000,c=superdense_color_code_*.stim; do
      echo "$circuit"
    done
    for circuit in testdata/surfacecodes/r={3,5,7,9,11}*p=$p_err,noise=si1000,c=surface_code_*.stim; do
      echo "$circuit"
    done
  done
done | shuf | while read circuit; do
  
  # Determine base degree based on the folder/filename
  if [[ "$circuit" == *"surfacecodes"* ]]; then
    SPARSIFY_BASE_DEGREE=2
  else
    SPARSIFY_BASE_DEGREE=3
  fi

  # Iterate through the requested reactivate limits
  for SPARSIFY_REACTIVATE_LIMIT in 0 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192; do
    
    echo "Submitting: $circuit | Degree: $SPARSIFY_BASE_DEGREE | Limit: $SPARSIFY_REACTIVATE_LIMIT"

    sbatch --partition=c2 --job-name=None4u \
            --ntasks=1 \
            --mem=120gb \
            --cpus-per-task=60 \
            --time=200:00:00 \
            --wrap="$TESSERACT_BIN --circuit \"$circuit\" --sample-num-shots 10000 --max-errors 10 --threads 30 --no-revisit-dets --beam 20 --beam-climbing --num-det-orders 21 --det-order-index --pqlimit 1000000 --sparsify-errors --sparsify-base-degree $SPARSIFY_BASE_DEGREE --sparsify-reactivate-limit $SPARSIFY_REACTIVATE_LIMIT --stats-out out/${STARTTIME}-${COUNTER}.json"
    
    # Increment counter for every single job so JSON files don't get overwritten
    COUNTER=$((COUNTER + 1))
  done

  # Submit also one baseline job
  sbatch --partition=c2 --job-name=None4u \
          --ntasks=1 \
          --mem=120gb \
          --cpus-per-task=60 \
          --time=200:00:00 \
          --wrap="$TESSERACT_BIN --circuit \"$circuit\" --sample-num-shots 10000 --max-errors 10 --threads 30 --no-revisit-dets --beam 20 --beam-climbing --num-det-orders 21 --det-order-index --pqlimit 1000000 --stats-out out/${STARTTIME}-${COUNTER}.json"
  # Increment counter for every single job so JSON files don't get overwritten
  COUNTER=$((COUNTER + 1))
done
