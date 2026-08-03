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
SHOTS=100
SEED=2

for num in 0; do
  for p_err in 0.001; do
  #   for circuit in testdata/bivariatebicyclecodes/r=6,*p=$p_err,noise=si1000,c=bivariate_bicycle_Z,*.stim; do
  #     echo "$circuit"
  #   done
    for circuit in testdata/colorcodes/r=7,*p=$p_err,noise=si1000,c=superdense_color_code_Z,*.stim; do
      echo "$circuit"
    done
    # for circuit in testdata/surfacecodes/r=7,*p=$p_err,noise=si1000,c=surface_code_Z,*.stim; do
    #   echo "$circuit"
    # done
  done
done | shuf | while read circuit; do
  circuit_dir=$(dirname "$circuit")
  circuit_name=$(basename "$circuit" .stim)
  mapping_file="$circuit_dir/gari/${circuit_name}_mapping.json"

  echo "========================================="
  echo "Running benchmark for circuit: $circuit_name"
  # Submit also one baseline job
   $TESSERACT_BIN --circuit "$circuit" --sample-num-shots $SHOTS --sample-seed $SEED --max-errors 100 --threads 32 --no-revisit-dets --beam 20 --beam-climbing --num-det-orders 1 --det-order-index --pqlimit 1000000 --stats-out out/${STARTTIME}-${COUNTER}-baseline-1det.json
   $SIMPLEX_BIN --circuit "$circuit" --sample-num-shots $SHOTS --sample-seed $SEED --max-errors 100 --threads 32 --stats-out out/${STARTTIME}-${COUNTER}-simplex-baseline.json
   COUNTER=$((COUNTER + 1))
  #  $TESSERACT_BIN --circuit "$circuit" --sample-num-shots $SHOTS --sample-seed $SEED --max-errors 100 --threads 32 --no-revisit-dets --beam 20 --beam-climbing --num-det-orders 21 --det-order-index --pqlimit 1000000 --stats-out out/${STARTTIME}-${COUNTER}-baseline-21det.json
  #  $TESSERACT_BIN --circuit "$circuit" --sample-num-shots $SHOTS --sample-seed $SEED --max-errors 100 --threads 32 --no-revisit-dets --beam 5 --beam-climbing --num-det-orders 1 --det-order-index --pqlimit 1000000 --stats-out out/${STARTTIME}-${COUNTER}-baseline-5beam.json
  #  $TESSERACT_BIN --circuit "$circuit" --sample-num-shots $SHOTS --sample-seed $SEED --max-errors 100 --threads 32 --no-revisit-dets --beam 5 --beam-climbing --num-det-orders 21 --det-order-index --pqlimit 1000000 --stats-out out/${STARTTIME}-${COUNTER}-baseline-5beam.json

   #run with gari
   for L_type in ""; do
     for mode in modeN modeQ modeR modeS modeS2 modeSO modeSO2; do
       dem_file="$circuit_dir/gari/${circuit_name}_${L_type}${mode}.dem"
       echo "Running GARI mode: $dem_file"
       $SIMPLEX_BIN --circuit "$circuit" --sample-num-shots $SHOTS --sample-seed $SEED --max-errors 100 --threads 32 --dem "$dem_file" --det-mapping-file "$mapping_file" --stats-out out/${STARTTIME}-${COUNTER}-simplex-gari-${L_type}${mode}.json
       COUNTER=$((COUNTER + 1))
       for order in order10 all; do
         echo "  Running order: $order"
        #  $TESSERACT_BIN --circuit "$circuit" --sample-num-shots $SHOTS --sample-seed $SEED --max-errors 100 --threads 32 --beam 5 --beam-climbing --num-det-orders 1 --pqlimit 1000000 --dem "$dem_file" --det-mapping-file "$mapping_file" --custom-order "$order" --stats-out out/${STARTTIME}-${COUNTER}-gari-${L_type}${mode}-${order}-revisit_beam5.json
      #   #  $TESSERACT_BIN --circuit "$circuit" --sample-num-shots $SHOTS --sample-seed $SEED --max-errors 100 --threads 32 --beam 20 --beam-climbing --num-det-orders 1 --pqlimit 1000000 --dem "$dem_file" --det-mapping-file "$mapping_file" --custom-order "$order" --stats-out out/${STARTTIME}-${COUNTER}-gari-${L_type}${mode}-${order}-revisit_beam20.json
      #   #  $TESSERACT_BIN --circuit "$circuit" --sample-num-shots $SHOTS --sample-seed $SEED --max-errors 100 --threads 32 --beam 5 --num-det-orders 1 --pqlimit 1000000 --dem "$dem_file" --det-mapping-file "$mapping_file" --custom-order "$order" --stats-out out/${STARTTIME}-${COUNTER}-gari-${L_type}${mode}-${order}-revisit_beam5_nc.json
      #    $TESSERACT_BIN --circuit "$circuit" --sample-num-shots $SHOTS --sample-seed $SEED --max-errors 100 --threads 32 --beam 2 --beam-climbing --num-det-orders 1 --pqlimit 1000000 --dem "$dem_file" --det-mapping-file "$mapping_file" --custom-order "$order" --stats-out out/${STARTTIME}-${COUNTER}-gari-${L_type}${mode}-${order}-revisit_beam2.json
      #    $TESSERACT_BIN --circuit "$circuit" --sample-num-shots $SHOTS --sample-seed $SEED --max-errors 100 --threads 32 --beam 3 --beam-climbing --num-det-orders 1 --pqlimit 1000000 --dem "$dem_file" --det-mapping-file "$mapping_file" --custom-order "$order" --stats-out out/${STARTTIME}-${COUNTER}-gari-${L_type}${mode}-${order}-revisit_beam3.json
      #    $TESSERACT_BIN --circuit "$circuit" --sample-num-shots $SHOTS --sample-seed $SEED --max-errors 100 --threads 32 --beam 4 --beam-climbing --num-det-orders 1 --pqlimit 1000000 --dem "$dem_file" --det-mapping-file "$mapping_file" --custom-order "$order" --stats-out out/${STARTTIME}-${COUNTER}-gari-${L_type}${mode}-${order}-revisit_beam4.json
      #    $TESSERACT_BIN --circuit "$circuit" --sample-num-shots $SHOTS --sample-seed $SEED --max-errors 100 --threads 32 --beam 6 --beam-climbing --num-det-orders 1 --pqlimit 1000000 --dem "$dem_file" --det-mapping-file "$mapping_file" --custom-order "$order" --stats-out out/${STARTTIME}-${COUNTER}-gari-${L_type}${mode}-${order}-revisit_beam6.json

        #  $TESSERACT_BIN --circuit "$circuit" --sample-num-shots $SHOTS --sample-seed $SEED --max-errors 100 --threads 32 --beam 20 --beam-climbing --num-det-orders 1 --pqlimit 1000000 --dem "$dem_file" --det-mapping-file "$mapping_file" --custom-order "$order" --stats-out out/${STARTTIME}-${COUNTER}-gari-${L_type}${mode}-${order}-revisit.json
        #  $TESSERACT_BIN --circuit "$circuit" --sample-num-shots $SHOTS --sample-seed $SEED --max-errors 100 --threads 32 --beam 20 --num-det-orders 1 --pqlimit 1000000 --dem "$dem_file" --det-mapping-file "$mapping_file" --custom-order "$order" --stats-out out/${STARTTIME}-${COUNTER}-gari-${L_type}${mode}-${order}-revisit.json
        #  $TESSERACT_BIN --circuit "$circuit" --sample-num-shots $SHOTS --sample-seed $SEED --max-errors 100 --threads 32 --beam 20 --num-det-orders 1 --pqlimit 10000000 --dem "$dem_file" --det-mapping-file "$mapping_file" --custom-order "$order" --stats-out out/${STARTTIME}-${COUNTER}-gari-${L_type}${mode}-${order}-revisit.json
        #  $TESSERACT_BIN --circuit "$circuit" --sample-num-shots $SHOTS --sample-seed $SEED --max-errors 100 --threads 32 --beam 5 --num-det-orders 1 --pqlimit 10000 --dem "$dem_file" --det-mapping-file "$mapping_file" --custom-order "$order" --stats-out out/${STARTTIME}-${COUNTER}-gari-${L_type}${mode}-${order}-revisit.json
        # for SPARSIFY_REACTIVATE_LIMIT in 0 2 4 8 16 32 64 128; do 
        # echo "Limit: $SPARSIFY_REACTIVATE_LIMIT"
        #  $TESSERACT_BIN --circuit "$circuit" --sample-num-shots $SHOTS --sample-seed $SEED --max-errors 100 --threads 32 --beam 5 --beam-climbing --num-det-orders 1 --pqlimit 1000000 --sparsify-errors --sparsify-base-degree 4 --sparsify-reactivate-limit $SPARSIFY_REACTIVATE_LIMIT --dem "$dem_file" --det-mapping-file "$mapping_file" --custom-order "$order" --stats-out out/${STARTTIME}-${COUNTER}-gari-${L_type}${mode}-${order}-revisit.json
        # #  $TESSERACT_BIN --circuit "$circuit" --sample-num-shots $SHOTS --sample-seed $SEED --max-errors 100 --threads 32 --beam 10 --beam-climbing --num-det-orders 1 --pqlimit 1000000 --sparsify-errors --sparsify-base-degree 4 --sparsify-reactivate-limit $SPARSIFY_REACTIVATE_LIMIT --dem "$dem_file" --det-mapping-file "$mapping_file" --custom-order "$order" --stats-out out/${STARTTIME}-${COUNTER}-gari-${L_type}${mode}-${order}-revisit.json
        #  $TESSERACT_BIN --circuit "$circuit" --sample-num-shots $SHOTS --sample-seed $SEED --max-errors 100 --threads 32 --beam 5 --num-det-orders 1 --pqlimit 1000000 --sparsify-errors --sparsify-base-degree 4 --sparsify-reactivate-limit $SPARSIFY_REACTIVATE_LIMIT --dem "$dem_file" --det-mapping-file "$mapping_file" --custom-order "$order" --stats-out out/${STARTTIME}-${COUNTER}-gari-${L_type}${mode}-${order}-revisit.json

        done
       done
     done
   done
  #  $TESSERACT_BIN --circuit "$circuit" --sample-num-shots $SHOTS --sample-seed $SEED --max-errors 100 --threads 32 --beam 20 --beam-climbing --num-det-orders 21 --det-order-index --pqlimit 1000000 --dem "$dem_file" --det-mapping-file "$mapping_file" --stats-out out/${STARTTIME}-${COUNTER}-gari-revisit-21det.json

#    $TESSERACT_BIN --circuit "$circuit" --sample-num-shots 1000 --sample_seed $SEED --max-errors 10 --threads 32 --no-revisit-dets --beam 20 --beam-climbing --num-det-orders 1 --det-order-index --pqlimit 1000000 --dem "$dem_file" —det-mapping-file "$mapping_file" —stats-out out/${STARTTIME}-${COUNTER}.json

  # Increment counter for every single job so JSON files don't get overwritten
  COUNTER=$((COUNTER + 1))
done
