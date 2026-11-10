#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-0}"
CKPT_KIND="${CKPT_KIND:-h1}"

DATA_FOLDER="../processed/maxhop/mquake_multi_paraphrased_test/" \
CHECKPOINT_DIR="../checkpoint/rearev/mquake_multi_seed${SEED}" \
TRAIN_EXPERIMENT_NAME="mquake_multi_seed${SEED}" \
EXPERIMENT_NAME="mquake_multi_seed${SEED}_paraphrased" \
CKPT_KIND="${CKPT_KIND}" \
bash configs/eval_rearev_mquake_multi.sh "${SEED}"
