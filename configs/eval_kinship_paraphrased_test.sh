#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 0 ]; then
  SEEDS=("$@")
else
  SEEDS=(0 42 100)
fi

for SEED in "${SEEDS[@]}"; do
  conda run -n llms python evaluate_result_info.py \
    --info_file "checkpoint/rearev/kinship_seed${SEED}/kinship_seed${SEED}_paraphrased_test.info" \
    --processed_file processed/maxhop/kinship_paraphrased_test/test.json \
    --dataset kinship \
    --experiment_name "kinship_seed${SEED}_paraphrased"
done
