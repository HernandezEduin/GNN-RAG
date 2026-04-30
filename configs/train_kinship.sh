cd gnn

conda run -n llms python main.py ReaRev \
  --name kinship \
  --data_folder ../processed/maxhop/kinship/ \
  --checkpoint_dir ../checkpoint/rearev/kinship_seed0 \
  --experiment_name kinship_seed0 \
  --seed 0 \
  --num_epoch 10 \
  --warmup_epoch -1 \
  --entity_dim 12 \
  --kg_dim 12 \
  --word_dim 64 \
  --batch_size 8 \
  --test_batch_size 8 \
  --eval_every 1 \
  --lm lstm \
  --num_iter 2 \
  --num_ins 2 \
  --num_gnn 2 \
  --relation_word_emb False \
  --word_emb_file none \
  --data_eff