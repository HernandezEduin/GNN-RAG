#!/usr/bin/env bash
set -euo pipefail

# MetaQA needs reverse traversal because many questions start from an attribute
# node and ask for movies that point to it. Uncapped reverse BFS can explode, so
# this config uses a conservative cap by default. Increase
# MAX_EDGES_PER_EXAMPLE for larger subgraphs, or set NO_MAX_EDGES=1 to omit it.
MAX_EDGES_PER_EXAMPLE="${MAX_EDGES_PER_EXAMPLE:-100}"
MAX_EDGES_ARGS=()
if [ "${NO_MAX_EDGES:-0}" != "1" ] && [ -n "${MAX_EDGES_PER_EXAMPLE}" ]; then
  MAX_EDGES_ARGS=(--max_edges_per_example "${MAX_EDGES_PER_EXAMPLE}")
fi

conda run --no-capture-output -n gnn_rag_custom python preprocess_custom_kgqa.py \
  --dataset metaqa \
  --data_dir data/metaqa \
  --output_dir processed/maxhop/metaqa \
  --include_reverse_traversal \
  "${MAX_EDGES_ARGS[@]}" \
  "$@"
