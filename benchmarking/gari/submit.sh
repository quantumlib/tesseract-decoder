#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status (-e),
# treat unset variables as an error (-u), and catch errors in pipes (-o pipefail).
set -euo pipefail

mkdir -p out

TESSERACT_BIN=./bazel-bin/src/tesseract
SIMPLEX_BIN=./bazel-bin/src/simplex
# Create a timestamp in nanoseconds for when the script starts
STARTTIME=$(($(date +%s%N)))

COUNTER=0

for num in $(seq 0 20); do
  for p_err in 0.001; do
    # for circuit in testdata/bivariatebicyclecodes/r=6,*p=$p_err,noise=si1000,c=bivariate_bicycle_Z*.stim; do
    #   echo "$circuit"
    # done
    for circuit in testdata/colorcodes/r=7,*p=$p_err,noise=si1000,c=superdense_color_code_Z,*cz.stim; do
      echo "$circuit"
    done
    # for circuit in testdata/surfacecodes/r=9,*p=$p_err,noise=si1000,c=surface_code_Z,*cz.stim; do
    #   echo "$circuit"
    # done
  done
done | shuf | while read circuit; do
  circuit_dir=$(dirname "$circuit")
  circuit_name=$(basename "$circuit" .stim)
  mapping_file="$circuit_dir/gari/${circuit_name}_mapping.json"

  echo "========================================="
  echo "Running benchmark for circuit: $circuit_name"

# Determine base degree based on the folder/filename
  if [[ "$circuit" == *"surfacecodes"* ]]; then
    SPARSIFY_BASE_DEGREE=3
  else
    SPARSIFY_BASE_DEGREE=4
  fi
  # Simplex Baseline
  # sbatch --partition=c2 --job-name=gari \
  #         --ntasks=1 \
  #         --mem=120gb \
  #         --cpus-per-task=30 \
  #         --time=20:00:00 \
  #         --wrap="$SIMPLEX_BIN --circuit \"$circuit\" --sample-num-shots 10000 --max-errors 10 --threads 30 --stats-out out/${STARTTIME}-${COUNTER}-simplex-baseline.json"
  # COUNTER=$((COUNTER + 1))

  # Baseline 1
  sbatch --partition=c2 --job-name=gari \
          --ntasks=1 \
          --mem=120gb \
          --cpus-per-task=30 \
          --time=20:00:00 \
          --wrap="$TESSERACT_BIN --circuit \"$circuit\" --sample-num-shots 10000 --max-errors 10 --threads 30 --no-revisit-dets --beam 20 --beam-climbing --num-det-orders 1 --det-order-index --pqlimit 1000000 --stats-out out/${STARTTIME}-${COUNTER}-baseline-1det.json"
  COUNTER=$((COUNTER + 1))

  # # Baseline 2
  # sbatch --partition=c2 --job-name=gari \
  #         --ntasks=1 \
  #         --mem=120gb \
  #         --cpus-per-task=30 \
  #         --time=20:00:00 \
  #         --wrap="$TESSERACT_BIN --circuit \"$circuit\" --sample-num-shots 10000 --max-errors 10 --threads 30 --no-revisit-dets --beam 20 --beam-climbing --num-det-orders 21 --det-order-index --pqlimit 1000000 --stats-out out/${STARTTIME}-${COUNTER}-baseline-21det.json"
  # COUNTER=$((COUNTER + 1))

  # # Baseline 3
  # sbatch --partition=c2 --job-name=gari \
  #         --ntasks=1 \
  #         --mem=120gb \
  #         --cpus-per-task=30 \
  #         --time=20:00:00 \
  #         --wrap="$TESSERACT_BIN --circuit \"$circuit\" --sample-num-shots 10000 --max-errors 10 --threads 30 --no-revisit-dets --beam 5 --beam-climbing --num-det-orders 1 --det-order-index --pqlimit 1000000 --stats-out out/${STARTTIME}-${COUNTER}-baseline-5beam-1det.json"
  # COUNTER=$((COUNTER + 1))

  # # Baseline 4
  # sbatch --partition=c2 --job-name=gari \
  #         --ntasks=1 \
  #         --mem=120gb \
  #         --cpus-per-task=30 \
  #         --time=20:00:00 \
  #         --wrap="$TESSERACT_BIN --circuit \"$circuit\" --sample-num-shots 10000 --max-errors 10 --threads 30 --no-revisit-dets --beam 5 --beam-climbing --num-det-orders 21 --det-order-index --pqlimit 1000000 --stats-out out/${STARTTIME}-${COUNTER}-baseline-5beam-21det.json"
  # COUNTER=$((COUNTER + 1))

  # GARI Runs
  for L_type in "ogL_"; do
    for mode in modeN modeQ modeR modeS modeS2 modeSO modeSO2; do
      dem_file="$circuit_dir/gari/${circuit_name}_${L_type}${mode}.dem"
      echo "Running GARI mode: $dem_file"

      # sbatch --partition=c2 --job-name=gari \
      #         --ntasks=1 \
      #         --mem=120gb \
      #         --cpus-per-task=30 \
      #         --time=20:00:00 \
      #         --wrap="$SIMPLEX_BIN --circuit \"$circuit\" --sample-num-shots 10000 --max-errors 10 --threads 30 --dem \"$dem_file\" --det-mapping-file \"$mapping_file\" --stats-out out/${STARTTIME}-${COUNTER}-simplex-gari-${L_type}${mode}.json"
      # COUNTER=$((COUNTER + 1))

      for order in order10 all; do
        echo "  Running order: $order"
        
        sbatch --partition=c2 --job-name=gari \
                --ntasks=1 \
                --mem=120gb \
                --cpus-per-task=30 \
                --time=20:00:00 \
                --wrap="$TESSERACT_BIN --circuit \"$circuit\" --sample-num-shots 10000 --max-errors 10 --threads 30 --beam 5 --beam-climbing --num-det-orders 1 --pqlimit 1000000 --dem \"$dem_file\" --det-mapping-file \"$mapping_file\" --custom-order \"$order\" --stats-out out/${STARTTIME}-${COUNTER}-gari-${L_type}${mode}-${order}-revisit_beam5.json"
        COUNTER=$((COUNTER + 1))

        # sbatch --partition=c2 --job-name=gari \
        #         --ntasks=1 \
        #         --mem=120gb \
        #         --cpus-per-task=30 \
        #         --time=20:00:00 \
        #         --wrap="$TESSERACT_BIN --circuit \"$circuit\" --sample-num-shots 10000 --max-errors 10 --threads 30 --beam 20 --beam-climbing --num-det-orders 1 --pqlimit 1000000 --dem \"$dem_file\" --det-mapping-file \"$mapping_file\" --custom-order \"$order\" --stats-out out/${STARTTIME}-${COUNTER}-gari-${L_type}${mode}-${order}-revisit_beam20.json"
        # COUNTER=$((COUNTER + 1))

      #   sbatch --partition=c2 --job-name=gari \
      #           --ntasks=1 \
      #           --mem=120gb \
      #           --cpus-per-task=30 \
      #           --time=20:00:00 \
      #           --wrap="$TESSERACT_BIN --circuit \"$circuit\" --sample-num-shots 10000 --max-errors 10 --threads 30 --beam 5 --num-det-orders 1 --pqlimit 1000000 --dem \"$dem_file\" --det-mapping-file \"$mapping_file\" --custom-order \"$order\" --stats-out out/${STARTTIME}-${COUNTER}-gari-${L_type}${mode}-${order}-revisit_beam5_nc.json"
      #   COUNTER=$((COUNTER + 1))

      #   sbatch --partition=c2 --job-name=gari \
      #           --ntasks=1 \
      #           --mem=120gb \
      #           --cpus-per-task=30 \
      #           --time=20:00:00 \
      #           --wrap="$TESSERACT_BIN --circuit \"$circuit\" --sample-num-shots 10000 --max-errors 10 --threads 30 --beam 10 --beam-climbing --num-det-orders 1 --pqlimit 1000000 --dem \"$dem_file\" --det-mapping-file \"$mapping_file\" --custom-order \"$order\" --stats-out out/${STARTTIME}-${COUNTER}-gari-${L_type}${mode}-${order}-revisit_beam10.json"
      #   COUNTER=$((COUNTER + 1))

      #   sbatch --partition=c2 --job-name=gari \
      #           --ntasks=1 \
      #           --mem=120gb \
      #           --cpus-per-task=30 \
      #           --time=20:00:00 \
      #           --wrap="$TESSERACT_BIN --circuit \"$circuit\" --sample-num-shots 10000 --max-errors 10 --threads 30 --beam 10 --num-det-orders 1 --pqlimit 1000000 --dem \"$dem_file\" --det-mapping-file \"$mapping_file\" --custom-order \"$order\" --stats-out out/${STARTTIME}-${COUNTER}-gari-${L_type}${mode}-${order}-revisit_beam10_nc.json"
      #   COUNTER=$((COUNTER + 1))

        # same with sparcifacation 
        # for SPARSIFY_REACTIVATE_LIMIT in 0 2 4 8 16 32 64 128 256; do 
        #   sbatch --partition=c2 --job-name=gari \
        #           --ntasks=1 \
        #           --mem=120gb \
        #           --cpus-per-task=30 \
        #           --time=20:00:00 \
        #           --wrap="$TESSERACT_BIN --circuit \"$circuit\" --sample-num-shots 10000 --max-errors 10 --threads 30 --beam 5 --beam-climbing --num-det-orders 1 --pqlimit 1000000 --sparsify-errors --sparsify-base-degree $SPARSIFY_BASE_DEGREE --sparsify-reactivate-limit $SPARSIFY_REACTIVATE_LIMIT --dem \"$dem_file\" --det-mapping-file \"$mapping_file\" --custom-order \"$order\" --stats-out out/${STARTTIME}-${COUNTER}-gari-${L_type}${mode}-${order}-revisit_beam5_sparsify.json"
        #   COUNTER=$((COUNTER + 1))

        #   sbatch --partition=c2 --job-name=gari \
        #           --ntasks=1 \
        #           --mem=120gb \
        #           --cpus-per-task=30 \
        #           --time=20:00:00 \
        #           --wrap="$TESSERACT_BIN --circuit \"$circuit\" --sample-num-shots 10000 --max-errors 10 --threads 30 --beam 5 --num-det-orders 1 --pqlimit 1000000 --sparsify-errors --sparsify-base-degree $SPARSIFY_BASE_DEGREE --sparsify-reactivate-limit $SPARSIFY_REACTIVATE_LIMIT --dem \"$dem_file\" --det-mapping-file \"$mapping_file\" --custom-order \"$order\" --stats-out out/${STARTTIME}-${COUNTER}-gari-${L_type}${mode}-${order}-revisit_beam5_nc_sparsify.json"
        #   COUNTER=$((COUNTER + 1))
        # done
      
      done
    done
  done
done
