cd gnn

# command to train
conda run -n llms python main.py ReaRev \
  --name mquake_single \
  --data_folder ../processed/maxhop/mquake_single/ \
  --checkpoint_dir ../checkpoint/rearev/mquake_single_seed0 \
  --experiment_name mquake_single_seed0 \
  --seed 0 \
  --num_epoch 10 \
  --warmup_epoch -1 \
  --entity_dim 100 \
  --kg_dim 100 \
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

# command to evaluate
conda run -n llms python main.py ReaRev \
  --name mquake_single \
  --data_folder ../processed/maxhop/mquake_single/ \
  --checkpoint_dir ../checkpoint/rearev/mquake_single_seed0 \
  --experiment_name mquake_single_seed0 \
  --seed 0 \
  --num_epoch 10 \
  --warmup_epoch -1 \
  --entity_dim 100 \
  --kg_dim 100 \
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
  --data_eff \
  --is_eval \
  --load_experiment mquake_single_seed0-h1.ckpt