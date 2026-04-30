#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 0 ]; then
  SEEDS=("$@")
else
  SEEDS=(0 42 100)
fi

for SEED in "${SEEDS[@]}"; do
  conda run -n llms python evaluate_result_info.py \
    --info_file "checkpoint/rearev/mquake_single_seed${SEED}/mquake_single_seed${SEED}_paraphrased_test.info" \
    --processed_file processed/maxhop/mquake_single_paraphrased_test/test.json \
    --dataset mquake_single \
    --experiment_name "mquake_single_seed${SEED}_paraphrased"
done
