# Custom ReaRev KGQA Runs

Preprocess the custom datasets:

```bash
bash configs/preprocess_kinship.sh
bash configs/preprocess_mquake_single.sh
```

The processed split files are pretty JSON arrays with `indent=4`. Leaving
`MAX_EDGES_PER_EXAMPLE` unset includes all unique triples reachable up to the
effective max hop. To cap subgraphs, run for example:

```bash
MAX_EDGES_PER_EXAMPLE=1000 bash configs/preprocess_mquake_single.sh
```

Train one seed:

```bash
bash configs/train_kinship.sh 0
bash configs/train_mquake_single.sh 0
```

Train the common three-seed set:

```bash
for s in 0 42 100; do bash configs/train_kinship.sh "$s"; done
for s in 0 42 100; do bash configs/train_mquake_single.sh "$s"; done
```

Regenerate `.info` outputs from already trained checkpoints:

```bash
bash configs/eval_rearev_kinship.sh 0
bash configs/eval_rearev_mquake_single.sh 0
```

Calculate external retrieval metrics from `.info`:

```bash
bash configs/eval_kinship.sh 0 42 100
bash configs/eval_mquake_single.sh 0 42 100
```

Metric files are written next to the checkpoint `.info` files, e.g.
`checkpoint/rearev/kinship_seed0/kinship_seed0_metrics.json`.

Paraphrased MQuAKE test evaluation:

```bash
bash configs/preprocess_mquake_single_paraphrased_test.sh
bash configs/eval_rearev_mquake_single_paraphrased_test.sh 0
bash configs/eval_mquake_single_paraphrased_test.sh 0
```

The paraphrased processed copy expands only the test split. Each
`Question-Paraphrased` entry becomes its own row with the same topic entity,
answers, and subgraph. It reuses `processed/maxhop/mquake_single/vocab.txt`
so existing LSTM checkpoints can be loaded without changing embedding shapes.
