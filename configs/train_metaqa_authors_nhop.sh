#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-0}"
EPOCHS="${EPOCHS:-10}"
# MAX_TRAIN="${MAX_TRAIN:-100000}"
  # --max_train "${MAX_TRAIN}" \
DATA_FOLDER="${DATA_FOLDER:-../data_authors/metaqa-nhop/}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-../checkpoint/rearev/metaqa_author_nhop_seed${SEED}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-metaqa_author_nhop_seed${SEED}}"

cd gnn

conda run --no-capture-output -n llms python main.py ReaRev \
  --name metaqa \
  --data_folder "${DATA_FOLDER}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --seed "${SEED}" \
  --num_epoch "${EPOCHS}" \
  --entity_dim 50 \
  --kg_dim 100 \
  --batch_size 8 \
  --test_batch_size 20 \
  --eval_every 2 \
  --lm lstm \
  --num_iter 2 \
  --num_ins 3 \
  --num_gnn 3 \
  --relation_word_emb False \
  --word_emb_file none \
  --data_eff 
