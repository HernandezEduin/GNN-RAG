#!/usr/bin/env bash
set -euo pipefail

MAX_EDGES_ARGS=()
if [ -n "${MAX_EDGES_PER_EXAMPLE:-}" ]; then
  MAX_EDGES_ARGS=(--max_edges_per_example "${MAX_EDGES_PER_EXAMPLE}")
fi

# Builds an evaluation copy whose test split has one row per
# Question-Paraphrased entry. The subgraph, answers, and topic entity are
# unchanged. vocab_from keeps LSTM checkpoints compatible with the original
# trained processed/maxhop/kinship vocabulary.
conda run -n gnn_rag_custom python preprocess_custom_kgqa.py \
  --dataset kinship \
  --data_dir data/kinship \
  --output_dir processed/maxhop/kinship_paraphrased_test \
  "${MAX_EDGES_ARGS[@]}" \
  --question_variants paraphrased \
  --variant_splits test \
  --vocab_from processed/maxhop/kinship/vocab.txt \
  "$@"
