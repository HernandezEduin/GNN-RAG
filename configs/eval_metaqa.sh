#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 0 ]; then
  SEEDS=("$@")
else
  SEEDS=(0 42 100)
fi
PROCESSED_FILE="${PROCESSED_FILE:-./data_authors/metaqa-nhop/test.json}"


for SEED in "${SEEDS[@]}"; do
  conda run -n llms python evaluate_metaqa_nhop_info.py \
    --info_file "checkpoint/rearev/metaqa_author_nhop_seed${SEED}/metaqa_author_nhop_seed${SEED}_test.info" \
    --dataset metaqa \
    --processed_file "${PROCESSED_FILE}" \
    --experiment_name "metaqa_author_nhop_seed${SEED}"
done
