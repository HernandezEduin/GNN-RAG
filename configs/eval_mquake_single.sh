#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 0 ]; then
  SEEDS=("$@")
else
  SEEDS=(0 42 100)
fi
PROCESSED_FILE="${PROCESSED_FILE:-processed/maxhop/mquake_single/test.json}"

for SEED in "${SEEDS[@]}"; do
  conda run -n llms python evaluate_result_info.py \
    --info_file "checkpoint/rearev/mquake_single_seed${SEED}/mquake_single_seed${SEED}_test.info" \
    --processed_file "${PROCESSED_FILE}" \
    --dataset mquake_single \
    --experiment_name "mquake_single_seed${SEED}"
done
