#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-0}"
EPOCHS="${EPOCHS:-10}"
DATA_FOLDER="${DATA_FOLDER:-../processed/maxhop/mquake_multi/}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-../checkpoint/rearev/mquake_multi_seed${SEED}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mquake_multi_seed${SEED}}"

cd gnn

conda run --no-capture-output -n llms python main.py ReaRev \
  --name mquake_multi \
  --data_folder "${DATA_FOLDER}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --seed "${SEED}" \
  --num_epoch "${EPOCHS}" \
  --warmup_epoch -1 \
  --entity_dim 100 \
  --kg_dim 100 \
  --word_dim 64 \
  --batch_size 8 \
  --test_batch_size 8 \
  --eval_every 1 \
  --lm lstm \
  --num_iter 2 \
  --num_ins 2 \
  --num_gnn 2 \
  --relation_word_emb False \
  --word_emb_file none \
  --data_eff 
