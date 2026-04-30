#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-0}"
CKPT_KIND="${CKPT_KIND:-h1}"

DATA_FOLDER="../processed/maxhop/kinship_paraphrased_test/" \
CHECKPOINT_DIR="../checkpoint/rearev/kinship_seed${SEED}" \
TRAIN_EXPERIMENT_NAME="kinship_seed${SEED}" \
EXPERIMENT_NAME="kinship_seed${SEED}_paraphrased" \
CKPT_KIND="${CKPT_KIND}" \
bash configs/eval_rearev_kinship.sh "${SEED}"
