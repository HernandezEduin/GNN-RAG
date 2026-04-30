#!/usr/bin/env bash
set -euo pipefail

MAX_EDGES_ARGS=()
if [ -n "${MAX_EDGES_PER_EXAMPLE:-}" ]; then
  MAX_EDGES_ARGS=(--max_edges_per_example "${MAX_EDGES_PER_EXAMPLE}")
fi

conda run -n gnn_rag_custom python preprocess_custom_kgqa.py \
  --dataset mquake_multi \
  --data_dir data/mquake_multi \
  --output_dir processed/maxhop/mquake_multi \
  "${MAX_EDGES_ARGS[@]}" \
  "$@"
