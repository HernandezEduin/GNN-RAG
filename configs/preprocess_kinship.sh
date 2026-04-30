#!/usr/bin/env bash
set -euo pipefail

conda run -n gnn_rag_custom python preprocess_custom_kgqa.py \
  --dataset kinship \
  --data_dir data/kinship \
  --output_dir processed/maxhop/kinship \
  --max_edges_per_example 1000 \
  "$@"
