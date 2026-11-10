#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-0}"
DATA_FOLDER="${DATA_FOLDER:-../processed/maxhop/mquake_single/}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-../checkpoint/rearev/mquake_single_seed${SEED}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mquake_single_seed${SEED}}"
LOAD_EXPERIMENT="${LOAD_EXPERIMENT:-}"

LOAD_ARGS=()
if [ -n "${LOAD_EXPERIMENT}" ]; then
  LOAD_ARGS=(--load_experiment "${LOAD_EXPERIMENT}")
fi

cd gnn

conda run --no-capture-output -n llms python count_parameters.py ReaRev \
  --name mquake_single \
  --data_folder "${DATA_FOLDER}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --seed "${SEED}" \
  --entity_dim 100 \
  --kg_dim 100 \
  --word_dim 64 \
  --batch_size 8 \
  --test_batch_size 8 \
  --lm lstm \
  --num_iter 2 \
  --num_ins 2 \
  --num_gnn 2 \
  --relation_word_emb False \
  --word_emb_file none \
  --data_eff \
  "${LOAD_ARGS[@]}"
