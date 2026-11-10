# Custom KGQA Retrieval With ReaRev

This repository is a clone/fork of **GNN-RAG: Graph Neural Retrieval for Large Language Model Reasoning on Knowledge Graphs**. The original GNN-RAG repository contains two main parts:

- `gnn/`: KGQA GNN retrievers.
- `llm/`: downstream RAG/LLM answer generation.

For this project, we mainly use the **GNN retriever**, specifically the **ReaRev** model exposed through `gnn/main.py ReaRev`. In other words, our experiments are retrieval/model experiments over custom KGQA datasets, not full LLM generation experiments unless explicitly stated.

When writing about this code, cite GNN-RAG for the repository/pipeline context and cite ReaRev for the model architecture.

## What We Added

The custom integration lives outside the original repo code where possible:

- `utils/custom_kgqa.py`: robust loader for custom KGQA CSV/KG data.
- `preprocess_custom_kgqa.py`: converts custom datasets to the JSON format consumed by `gnn/dataset_load.py`.
- `evaluate_result_info.py`: external evaluator for ReaRev `.info` outputs.
- `configs/`: reproducible preprocessing, training, checkpoint evaluation, and external metric scripts.

We also patch a few original `gnn/` files so local ReaRev training works with custom datasets and no pretrained word embeddings:

- `--word_emb_file none` is accepted.
- string topic entities are handled correctly.
- LSTM/ReaRev shape issues are fixed.
- seed handling is strengthened for repeatability.
- training now shows epoch and batch `tqdm` progress bars.

## Dataset Format

Each custom dataset directory should contain:

```text
qa_nhop.csv
triplets.txt
node_data.csv          # optional for datasets like Kinship
relation_data.csv      # optional for datasets like Kinship
```

`qa_nhop.csv` should include the question, topic/start entity, answer entity, split, hop count, and preferably gold paths:

```text
Question
Source-Entity
Answer-Entity
SplitLabel
Hops
Paths
Paths-Label
Path-Key              # optional
Question-Paraphrased  # optional
```

The preprocessing uses the provided `Source-Entity` as the retrieval anchor and does not perform entity linking. By default, it builds each question subgraph using the dataset-wide maximum `Hops` value.

Processed `train.json`, `dev.json`, and `test.json` are written as human-readable JSON arrays with `indent=4`. The loader still accepts older JSONL files for backward compatibility.

## Environments

Dataset preprocessing:

```bash
conda run -n gnn_rag_custom python preprocess_custom_kgqa.py --help
```

ReaRev training/evaluation:

```bash
conda run -n llms python gnn/main.py ReaRev --help
```

## Preprocess

```bash
bash configs/preprocess_kinship.sh
bash configs/preprocess_mquake_single.sh
bash configs/preprocess_metaqa.sh
```

By default, preprocessing does **not** cap the number of triples per question subgraph; it includes all unique triples reachable within the effective hop limit. To cap the subgraph size, set `MAX_EDGES_PER_EXAMPLE`:

```bash
MAX_EDGES_PER_EXAMPLE=1000 bash configs/preprocess_mquake_single.sh
```

MetaQA is the exception in the configs: because it uses reverse traversal and has many more QA rows, `configs/preprocess_metaqa.sh` uses `MAX_EDGES_PER_EXAMPLE=100` by default. You can raise it explicitly:

```bash
MAX_EDGES_PER_EXAMPLE=1000 bash configs/preprocess_metaqa.sh
```

MetaQA does not include `Paths` or `Path-Key` columns. When gold paths are absent, preprocessing builds best-effort support paths by running bounded BFS from the topic entity to the gold answer entities using the same traversal setting as subgraph collection. With reverse traversal enabled, the stored triples still keep their original KG direction.

Outputs are written to:

```text
processed/maxhop/kinship/
processed/maxhop/mquake_single/
processed/maxhop/metaqa/
```

Each processed folder contains:

```text
train.json
dev.json
test.json
entities.txt
relations.txt
vocab.txt
preprocess_summary.json
```

## Train ReaRev

Train one seed:

```bash
bash configs/train_kinship.sh 0
bash configs/train_mquake_single.sh 0
```

Train three seeds for mean/std reporting:

```bash
for s in 0 42 100; do bash configs/train_kinship.sh "$s"; done
for s in 0 42 100; do bash configs/train_mquake_single.sh "$s"; done
for s in 0 42 100; do bash configs/train_mquake_multi.sh "$s"; done
for s in 0 42 100; do bash configs/train_metaqa_authors_nhop.sh "$s"; done
```

Checkpoints and logs are written under:

```text
checkpoint/rearev/kinship_seed*/
checkpoint/rearev/mquake_single_seed*/
checkpoint/rearev/metaqa_authors_nhop_seed*/
```

## Authors' MetaQA N-Hop

The ReaRev authors' processed MetaQA folders can be placed under:

```text
data_authors/metaqa-1hop/
data_authors/metaqa-2hop/
data_authors/metaqa-3hop/
```

Their `entities.txt` and `relations.txt` files are identical across hop settings, but the vocabularies differ. To build a single n-hop dataset, concatenate the split files and rebuild one `vocab.txt` from all merged questions:

```bash
bash configs/merge_metaqa_authors_nhop.sh
```

By default, the merge writes one third of each hop's train split to reduce ReaRev loader memory pressure, while dev/test and `vocab.txt` remain built from the full authors' data. Override the train fraction when needed:

```bash
TRAIN_FRACTION=0.25 bash configs/merge_metaqa_authors_nhop.sh
TRAIN_FRACTION=1.0 bash configs/merge_metaqa_authors_nhop.sh
```

This writes:

```text
data_authors/metaqa-nhop/
```

The merge keeps entity/relation ids unchanged, rewrites duplicate question ids as `1hop_*`, `2hop_*`, and `3hop_*`, adds `source_hop` metadata, and ignores `word_emb_300d.npy`.

Train the merged authors' MetaQA n-hop setting without word embeddings:

```bash
bash configs/train_metaqa_authors_nhop.sh 0
```

## Evaluate Checkpoints

Regenerate ReaRev `.info` files from trained checkpoints:

```bash
bash configs/eval_rearev_kinship.sh 0
bash configs/eval_rearev_mquake_single.sh 0
```

Then compute external retrieval metrics:

```bash
bash configs/eval_kinship.sh 0 42 100
bash configs/eval_mquake_single.sh 0 42 100
```

The external evaluator reports:

- `h1`: top retained candidate is a gold answer.
- `mrr`: reciprocal rank of the first gold answer.
- `per_hop`: H1/MRR grouped by hop count.

The `.info` file already contains the model-retained candidates after the repo's `eps` cutoff. Leave `--top_k` unset for faithful ReaRev-style evaluation. Metrics are written next to the checkpoint `.info` file as `<experiment_name>_metrics.json`.

## Paraphrased Evaluation

To evaluate paraphrased MQuAKE questions independently while keeping the same subgraph, answers, and topic entity:

```bash
bash configs/preprocess_mquake_single_paraphrased_test.sh
bash configs/eval_rearev_mquake_single_paraphrased_test.sh 0
bash configs/eval_mquake_single_paraphrased_test.sh 0
```

This creates:

```text
processed/maxhop/mquake_single_paraphrased_test/
```

Only the test split is expanded. Each `Question-Paraphrased` entry becomes a separate row, but it shares the original row's KG evidence. The script copies the original `vocab.txt` so existing LSTM checkpoints remain compatible.

## Notes

The folder `delete/` contains old scratch code and is not part of the active pipeline.
