#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-0}"
EPOCHS="${EPOCHS:-10}"
DATA_FOLDER="${DATA_FOLDER:-../processed/maxhop/metaqa/}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-../checkpoint/rearev/metaqa_seed${SEED}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-metaqa_seed${SEED}}"

cd gnn

conda run --no-capture-output -n llms python main.py ReaRev \
  --name metaqa \
  --data_folder "${DATA_FOLDER}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --seed "${SEED}" \
  --num_epoch "${EPOCHS}" \
  --max_train 100000 \
  --warmup_epoch -1 \
  --entity_dim 100 \
  --kg_dim 100 \
  --word_dim 300 \
  --batch_size 16 \
  --test_batch_size 16 \
  --eval_every 2 \
  --lm lstm \
  --lm_dropout 0.35 \
  --linear_dropout 0.35 \
  --num_iter 2 \
  --num_ins 2 \
  --num_gnn 2 \
  --relation_word_emb False \
  --word_emb_file none \
  --data_eff 
