#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-data_authors/metaqa-nhop}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.3333333333333333}"

conda run --no-capture-output -n gnn_rag_custom python merge_authors_metaqa_hops.py \
  --input_root data_authors \
  --output_dir "${OUTPUT_DIR}" \
  --train_fraction "${TRAIN_FRACTION}"
