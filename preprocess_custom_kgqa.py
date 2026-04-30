import argparse
import json
import re
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from utils.custom_kgqa import CustomKGQADataset, QAExample

Triple = Tuple[str, str, str]


# ---------------------------------------------------------------------------
# Text and graph helpers
# ---------------------------------------------------------------------------

def tokenize_for_vocab(text: str) -> List[str]:
    text = str(text).strip().lower()
    text = re.sub(r"\'s", " s", text)
    words = []
    for token in text.split(" "):
        token = re.sub(r"^[^a-z0-9]|[^a-z0-9]$", "", token)
        if token:
            words.append(token)
    return words

def build_adjacency(triples: Sequence[Triple], include_reverse_traversal: bool = False) -> Dict[str, List[Tuple[str, Triple]]]:
    adjacency: Dict[str, List[Tuple[str, Triple]]] = {}
    for h, r, t in triples:
        adjacency.setdefault(h, []).append((t, (h, r, t)))
        if include_reverse_traversal:
            adjacency.setdefault(t, []).append((h, (h, r, t)))
    return adjacency


def unique_triples(triples: Iterable[Triple]) -> List[Triple]:
    seen = set()
    result = []
    for triple in triples:
        if triple in seen:
            continue
        seen.add(triple)
        result.append(triple)
    return result


def collect_subgraph(
    example: QAExample,
    adjacency: Dict[str, List[Tuple[str, Triple]]],
    max_depth: int,
    max_edges: Optional[int],
) -> Tuple[List[str], List[Triple]]:
    """Build the per-question dense subgraph expected by gnn/dataset_load.py.

    Custom datasets already provide the topic entity, so this never performs
    entity linking. Gold path triples are forced into the subgraph before BFS
    because the GNN trainer cannot learn an answer that is absent locally.
    """
    forced_triples = unique_triples(triple for path in example.gold_paths for triple in path)
    triples: List[Triple] = list(forced_triples)
    entities: Set[str] = set(example.start_ids) | set(example.gold_answer_ids)
    for h, _, t in forced_triples:
        entities.add(h)
        entities.add(t)

    queue = deque()
    seen_depth: Dict[str, int] = {}
    for start_id in example.start_ids:
        if start_id in adjacency and start_id not in seen_depth:
            seen_depth[start_id] = 0
            queue.append(start_id)

    seen_triples = set(triples)
    while queue and (max_edges is None or len(triples) < max_edges):
        node = queue.popleft()
        depth = seen_depth[node]
        if depth >= max_depth:
            continue
        for neighbor, triple in adjacency.get(node, []):
            if triple not in seen_triples:
                triples.append(triple)
                seen_triples.add(triple)
                h, _, t = triple
                entities.add(h)
                entities.add(t)
                if max_edges is not None and len(triples) >= max_edges:
                    break
            if neighbor not in seen_depth:
                seen_depth[neighbor] = depth + 1
                queue.append(neighbor)

    return sorted(entities), triples


def answer_objects(example: QAExample, entity_labels: Dict[str, str]) -> List[dict]:
    answers = []
    for entity_id in example.gold_answer_ids:
        answers.append({"kb_id": entity_id, "text": entity_labels.get(entity_id, entity_id)})
    return answers


def example_to_gnn_json(
    example: QAExample,
    entity_labels: Dict[str, str],
    adjacency: Dict[str, List[Tuple[str, Triple]]],
    max_edges_per_example: Optional[int],
    max_hops: Optional[int],
    extra_hops: int,
) -> dict:
    depth = max_hops if max_hops is not None else max(0, (example.hops or 1) + extra_hops)
    subgraph_entities, subgraph_triples = collect_subgraph(
        example,
        adjacency=adjacency,
        max_depth=depth,
        max_edges=max_edges_per_example,
    )
    return {
        "id": example.question_id,
        "question": example.question,
        "entities": example.start_ids,
        "answers": answer_objects(example, entity_labels),
        "subgraph": {
            "entities": subgraph_entities,
            "tuples": [list(triple) for triple in subgraph_triples],
        },
        "metadata": {
            "row_index": example.row_index,
            "split": example.split,
            "hops": example.hops,
            "start_entity_labels": example.start_labels,
            "gold_answer_labels": example.gold_answer_labels,
        },
    }


def write_lines(path: Path, values: Iterable[str]) -> None:
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


class JsonArrayWriter:
    def __init__(self, path: Path):
        self.path = path
        self.handle = path.open("w", encoding="utf-8")
        self.count = 0
        self.handle.write("[\n")

    def write(self, obj: Dict) -> None:
        if self.count:
            self.handle.write(",\n")
        rendered = json.dumps(obj, indent=4, ensure_ascii=False)
        self.handle.write("\n".join(f"    {line}" for line in rendered.splitlines()))
        self.count += 1

    def close(self) -> None:
        if self.count:
            self.handle.write("\n")
        self.handle.write("]\n")
        self.handle.close()


def split_name(split: Optional[str]) -> str:
    value = (split or "train").lower()
    if value in {"valid", "validation"}:
        return "dev"
    return value


# ---------------------------------------------------------------------------
# Question variant helpers
# ---------------------------------------------------------------------------

def parse_split_filter(value: str) -> Optional[Set[str]]:
    normalized = {split_name(part.strip()) for part in value.split(",") if part.strip()}
    if not normalized or "all" in normalized:
        return None
    return normalized


def unique_texts(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def question_variants_for_example(
    example: QAExample,
    mode: str,
    split: str,
    variant_splits: Optional[Set[str]],
) -> List[Tuple[str, str, str, int]]:
    """Return (question_id, question_text, variant_name, variant_index).

    Paraphrase expansion is intentionally done at preprocessing time so each
    paraphrased question can be evaluated independently while sharing the same
    topic entity, answers, and local subgraph.
    """
    if variant_splits is not None and split not in variant_splits:
        return [(example.question_id, example.question, "original", 0)]

    paraphrases = unique_texts(example.question_paraphrases)
    if mode == "original":
        return [(example.question_id, example.question, "original", 0)]
    if mode == "paraphrased":
        if not paraphrases:
            return [(example.question_id, example.question, "original_no_paraphrases", 0)]
        return [
            (f"{example.question_id}:paraphrase:{idx}", question, "paraphrase", idx)
            for idx, question in enumerate(paraphrases, start=1)
        ]
    if mode == "original_and_paraphrased":
        variants = [(example.question_id, example.question, "original", 0)]
        for idx, question in enumerate(unique_texts(paraphrases), start=1):
            if question == example.question:
                continue
            variants.append((f"{example.question_id}:paraphrase:{idx}", question, "paraphrase", idx))
        return variants
    raise ValueError(f"Unsupported question variant mode: {mode}")


def load_vocab(path: Path) -> List[str]:
    with path.open(encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.rstrip("\n")]


# ---------------------------------------------------------------------------
# GNN-RAG/ReaRev JSON conversion
# ---------------------------------------------------------------------------

def preprocess_dataset(
    dataset_name: str,
    data_dir: Path,
    output_dir: Path,
    max_edges_per_example: Optional[int],
    max_hops: Optional[int],
    extra_hops: int,
    include_reverse_traversal: bool,
    limit_per_split: Optional[int],
    question_variants: str,
    variant_splits: str,
    vocab_from: Optional[Path],
) -> dict:
    dataset = CustomKGQADataset(dataset_name, data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_max_hops = None
    if "Hops" in dataset.qa_df.columns:
        hops = dataset.qa_df["Hops"].dropna()
        if not hops.empty:
            dataset_max_hops = int(hops.astype(int).max())
    effective_max_hops = max_hops if max_hops is not None else dataset_max_hops

    adjacency = build_adjacency(dataset.triples, include_reverse_traversal=include_reverse_traversal)
    entities = set(dataset.node_labels.keys())
    relations = set(dataset.relation_labels.keys())
    for h, r, t in dataset.triples:
        entities.add(h)
        entities.add(t)
        relations.add(r)

    fixed_vocab = load_vocab(vocab_from) if vocab_from is not None else None
    vocab = set(fixed_vocab or ["UNK"])
    variant_split_filter = parse_split_filter(variant_splits)

    def add_vocab_text(text: str) -> None:
        if fixed_vocab is not None:
            return
        for token in tokenize_for_vocab(text):
            vocab.add(token)

    split_counts = {"train": 0, "dev": 0, "test": 0}
    output_files = {
        "train": JsonArrayWriter(output_dir / "train.json"),
        "dev": JsonArrayWriter(output_dir / "dev.json"),
        "test": JsonArrayWriter(output_dir / "test.json"),
    }
    try:
        for example in dataset.examples():
            split = split_name(example.split)
            if split not in output_files:
                continue
            if limit_per_split is not None and split_counts[split] >= limit_per_split:
                continue

            for label in example.start_labels + example.gold_answer_labels:
                add_vocab_text(label)

            base_obj = example_to_gnn_json(
                example=example,
                entity_labels=dataset.node_labels,
                adjacency=adjacency,
                max_edges_per_example=max_edges_per_example,
                max_hops=effective_max_hops,
                extra_hops=extra_hops,
            )
            variants = question_variants_for_example(
                example=example,
                mode=question_variants,
                split=split,
                variant_splits=variant_split_filter,
            )
            for variant_id, question_text, variant_name, variant_index in variants:
                if limit_per_split is not None and split_counts[split] >= limit_per_split:
                    break
                add_vocab_text(question_text)
                obj = dict(base_obj)
                obj["id"] = variant_id
                obj["question"] = question_text
                obj["metadata"] = dict(base_obj["metadata"])
                obj["metadata"].update(
                    {
                        "base_question_id": example.question_id,
                        "original_question": example.question,
                        "question_variant": variant_name,
                        "question_variant_index": variant_index,
                    }
                )
                output_files[split].write(obj)
                split_counts[split] += 1
    finally:
        for handle in output_files.values():
            handle.close()

    for relation_id, relation_label in dataset.relation_labels.items():
        for value in (relation_id, relation_label):
            add_vocab_text(value.replace("_", " "))

    write_lines(output_dir / "entities.txt", sorted(entities))
    write_lines(output_dir / "relations.txt", sorted(relations))
    vocab_values = fixed_vocab if fixed_vocab is not None else sorted(vocab)
    write_lines(output_dir / "vocab.txt", vocab_values)

    summary = {
        "dataset": dataset_name,
        "source_data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "num_entities": len(entities),
        "num_relations": len(relations),
        "num_vocab": len(vocab_values),
        "split_counts": split_counts,
        "max_edges_per_example": max_edges_per_example,
        "dataset_max_hops": dataset_max_hops,
        "max_hops": effective_max_hops,
        "extra_hops": extra_hops,
        "include_reverse_traversal": include_reverse_traversal,
        "question_variants": question_variants,
        "variant_splits": "all" if variant_split_filter is None else sorted(variant_split_filter),
        "vocab_from": str(vocab_from) if vocab_from is not None else None,
        "files": {
            "train": str(output_dir / "train.json"),
            "dev": str(output_dir / "dev.json"),
            "test": str(output_dir / "test.json"),
            "entities": str(output_dir / "entities.txt"),
            "relations": str(output_dir / "relations.txt"),
            "vocab": str(output_dir / "vocab.txt"),
        },
    }
    (output_dir / "preprocess_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert custom KGQA CSV/KG data to the original GNN-RAG JSONL format.")
    parser.add_argument("--dataset", required=True, help="Dataset name for metadata, e.g. kinship or mquake_single.")
    parser.add_argument("--data_dir", required=True, type=Path, help="Directory containing qa_nhop.csv and triplets.txt.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Processed dataset folder to create.")
    parser.add_argument(
        "--max_edges_per_example",
        default=None,
        type=int,
        help="Optional cap on local subgraph triples per QA row. If omitted, include all unique triples up to max_hops.",
    )
    parser.add_argument("--max_hops", default=None, type=int, help="Override subgraph BFS depth. Defaults to the dataset-wide maximum Hops value.")
    parser.add_argument("--extra_hops", default=0, type=int, help="Add extra BFS hops beyond each row's Hops.")
    parser.add_argument("--include_reverse_traversal", action="store_true", help="Use reverse edges only for collecting subgraphs; stored triples keep original direction.")
    parser.add_argument("--limit_per_split", default=None, type=int, help="Optional debugging limit per train/dev/test split.")
    parser.add_argument(
        "--question_variants",
        default="original",
        choices=("original", "paraphrased", "original_and_paraphrased"),
        help="Write original questions, one row per Question-Paraphrased entry, or both.",
    )
    parser.add_argument(
        "--variant_splits",
        default="all",
        help="Comma-separated splits where question_variants applies, e.g. test. Other splits keep original questions.",
    )
    parser.add_argument(
        "--vocab_from",
        default=None,
        type=Path,
        help="Copy an existing vocab.txt order, useful for paraphrased eval with an already trained LSTM checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = preprocess_dataset(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_edges_per_example=args.max_edges_per_example,
        max_hops=args.max_hops,
        extra_hops=args.extra_hops,
        include_reverse_traversal=args.include_reverse_traversal,
        limit_per_split=args.limit_per_split,
        question_variants=args.question_variants,
        variant_splits=args.variant_splits,
        vocab_from=args.vocab_from,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
