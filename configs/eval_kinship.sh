#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 0 ]; then
  SEEDS=("$@")
else
  SEEDS=(0 42 100)
fi
PROCESSED_FILE="${PROCESSED_FILE:-processed/maxhop/kinship/test.json}"

for SEED in "${SEEDS[@]}"; do
  conda run -n llms python evaluate_result_info.py \
    --info_file "checkpoint/rearev/kinship_seed${SEED}/kinship_seed${SEED}_test.info" \
    --processed_file "${PROCESSED_FILE}" \
    --dataset kinship \
    --experiment_name "kinship_seed${SEED}"
done
