# Custom KGQA Retrieval With ReaRev

This repository is a clone/fork of **GNN-RAG: Graph Neural Retrieval for Large Language Model Reasoning on Knowledge Graphs**. The original GNN-RAG repository contains two main parts:

- `gnn/`: KGQA GNN retrievers.
- `llm/`: downstream RAG/LLM answer generation.

For this project, we mainly use the **GNN retriever**, specifically the **ReaRev** model exposed through `gnn/main.py ReaRev`. In other words, our experiments are retrieval/model experiments over custom KGQA datasets, not full LLM generation experiments unless explicitly stated.

When writing about this code, cite GNN-RAG for the repository/pipeline context and cite ReaRev for the model architecture.

## What We Added

The custom integration lives outside the original repo code where possible:

- `utils/custom_kgqa.py`: robust loader for custom KGQA CSV/KG data.
- `preprocess_custom_kgqa.py`: converts custom datasets to the JSONL format expected by `gnn/dataset_load.py`.
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
```

Outputs are written to:

```text
processed/maxhop/kinship/
processed/maxhop/mquake_single/
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
```

Checkpoints and logs are written under:

```text
checkpoint/rearev/kinship_seed*/
checkpoint/rearev/mquake_single_seed*/
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
- `subgraph_overlap`: overlap between gold path triples and the candidate-induced subgraph.
- `relation_overlap`: overlap between gold path relations and relations in the candidate-induced subgraph.

The `.info` file already contains the model-retained candidates after the repo's `eps` cutoff. Leave `--top_k` unset for faithful ReaRev-style evaluation.

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
