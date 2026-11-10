#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-0}"
CKPT_KIND="${CKPT_KIND:-h1}"
DATA_FOLDER="${DATA_FOLDER:-../data_authors/metaqa-nhop/}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-../checkpoint/rearev/metaqa_author_nhop_seed${SEED}}"
TRAIN_EXPERIMENT_NAME="${TRAIN_EXPERIMENT_NAME:-metaqa_author_nhop_seed${SEED}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-metaqa_author_nhop_seed${SEED}}"

cd gnn

conda run -n llms python main.py ReaRev \
  --name metaqa \
  --data_folder "${DATA_FOLDER}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --seed "${SEED}" \
  --entity_dim 50 \
  --kg_dim 100 \
  --batch_size 8 \
  --test_batch_size 20 \
  --lm lstm \
  --num_iter 2 \
  --num_ins 3 \
  --num_gnn 3 \
  --relation_word_emb False \
  --word_emb_file none \
  --data_eff \
  --is_eval \
  --load_experiment "${TRAIN_EXPERIMENT_NAME}-${CKPT_KIND}.ckpt"
