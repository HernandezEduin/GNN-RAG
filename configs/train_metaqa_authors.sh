#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-0}"
EPOCHS="${EPOCHS:-10}"
DATA_FOLDER="${DATA_FOLDER:-../data_authors/metaqa-3hop/}"
# CHECKPOINT_DIR="${CHECKPOINT_DIR:-../checkpoint/rearev/metaqa_author_with_wordemb}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-metaqa}"

cd gnn

conda run --no-capture-output -n llms python main.py ReaRev \
  --name metaqa \
  --data_folder "${DATA_FOLDER}" \
  --checkpoint_dir "../checkpoint/rearev/metaqa_author_w_wordemb" \
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
  --word_emb_file word_emb_300d.npy 

conda run --no-capture-output -n llms python main.py ReaRev \
  --name metaqa \
  --data_folder "${DATA_FOLDER}" \
  --checkpoint_dir "../checkpoint/rearev/metaqa_author_wo_wordemb" \
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
  --word_emb_file none 