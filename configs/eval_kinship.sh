#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 0 ]; then
  SEEDS=("$@")
else
  SEEDS=(0 42)
fi
PROCESSED_FILE="${PROCESSED_FILE:-processed/maxhop/kinship/test.json}"
DATA_DIR="${DATA_DIR:-data/kinship}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-with_seed}"

for SEED in "${SEEDS[@]}"; do
  conda run -n llms python evaluate_result_info.py \
    --info_file "checkpoint/rearev/kinship_seed${SEED}/kinship_seed${SEED}_test.info" \
    --processed_file "${PROCESSED_FILE}" \
    --data_dir "${DATA_DIR}" \
    --dataset kinship \
    --output_dir "outputs/rearev_external_eval/kinship_seed${SEED}_${OUTPUT_SUFFIX}" \
    --include_seed_entities
done
