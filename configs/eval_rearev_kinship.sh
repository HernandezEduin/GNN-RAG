#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-0}"
CKPT_KIND="${CKPT_KIND:-h1}"
DATA_FOLDER="${DATA_FOLDER:-../processed/maxhop/kinship/}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-../checkpoint/rearev/kinship_seed${SEED}}"
TRAIN_EXPERIMENT_NAME="${TRAIN_EXPERIMENT_NAME:-kinship_seed${SEED}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-kinship_seed${SEED}}"

cd gnn

conda run -n llms python main.py ReaRev \
  --name kinship \
  --data_folder "${DATA_FOLDER}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --seed "${SEED}" \
  --entity_dim 12 \
  --kg_dim 12 \
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
  --is_eval \
  --load_experiment "${TRAIN_EXPERIMENT_NAME}-${CKPT_KIND}.ckpt"
