conda run -n llms python evaluate_result_info.py \
  --info_file checkpoint/rearev/mquake_single_seed0/mquake_single_seed0_test.info \
  --processed_file processed/maxhop/mquake_single/test.json \
  --data_dir data/mquake_single \
  --dataset mquake_single \
  --output_dir outputs/rearev_external_eval/mquake_single_seed0_with_seed \
  --include_seed_entities

conda run -n llms python evaluate_result_info.py \
  --info_file checkpoint/rearev/mquake_single_seed42/mquake_single_seed42_test.info \
  --processed_file processed/maxhop/mquake_single/test.json \
  --data_dir data/mquake_single \
  --dataset mquake_single \
  --output_dir outputs/rearev_external_eval/mquake_single_seed42_with_seed \
  --include_seed_entities

conda run -n llms python evaluate_result_info.py \
  --info_file checkpoint/rearev/mquake_single_seed100/mquake_single_seed100_test.info \
  --processed_file processed/maxhop/mquake_single/test.json \
  --data_dir data/mquake_single \
  --dataset mquake_single \
  --output_dir outputs/rearev_external_eval/mquake_single_seed100_with_seed \
  --include_seed_entities