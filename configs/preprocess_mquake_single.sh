conda run -n gnn_rag_custom python preprocess_custom_kgqa.py \
  --dataset mquake_single \
  --data_dir data/mquake_single \
  --output_dir processed/max_hop/mquake_single \
  --max_edges_per_example 1000