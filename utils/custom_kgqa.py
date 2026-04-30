import ast
import csv
import json
import math
import re
import string
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import pandas as pd


TEXT_ID_COLUMNS = ("EID", "QID", "RID", "Property", "ID", "id", "entity_id", "relation_id")
TEXT_LABEL_COLUMNS = ("Title", "Label", "Name", "label", "name", "title")


def extract_literals(column, flatten: bool = False):
    """
    Safely parse CSV cells that contain serialized Python list literals.

    Custom KGQA CSVs mix plain strings, list-like strings, and missing values.
    This helper keeps plain strings intact at the per-cell parser layer while
    still using ast.literal_eval for actual serialized lists.
    """
    if isinstance(column, str):
        column = pd.Series([column])

    def safe_parse(x):
        if pd.isna(x):
            return []
        if isinstance(x, list):
            return x
        if not isinstance(x, str):
            return [x]
        value = x.strip()
        if value == "":
            return []
        if value.startswith("[") or value.startswith("(") or value.startswith("{"):
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else [parsed]
        return [value]

    parsed = column.apply(safe_parse)
    if flatten:
        return [item for sublist in parsed for item in sublist]
    return parsed


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    value = str(value).lower()
    value = "".join(ch for ch in value if ch not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def token_set(value: Any) -> Set[str]:
    return {tok for tok in normalize_text(value).split() if tok}


def cell_to_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if pd.isna(value):
        return []
    if not isinstance(value, str):
        return [str(value)]
    value = value.strip()
    if value == "":
        return []
    if value.startswith("[") or value.startswith("("):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple)):
            return [str(v) for v in parsed]
        return [str(parsed)]
    return [value]


def first_existing(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lookup = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def read_mapping_csv(path: Path, default_id_col: Optional[str] = None) -> Dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    id_col = default_id_col or first_existing(df.columns, TEXT_ID_COLUMNS) or df.columns[0]
    label_col = first_existing(df.columns, TEXT_LABEL_COLUMNS)
    if label_col is None:
        label_col = df.columns[1] if len(df.columns) > 1 else id_col
    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        entity_id = str(row[id_col]).strip()
        if not entity_id or entity_id == "nan":
            continue
        label = str(row[label_col]).strip()
        mapping[entity_id] = label if label and label != "nan" else entity_id
    return mapping


def read_triplets(path: Path) -> List[Tuple[str, str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required KG triples file: {path}")
    triples: List[Tuple[str, str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = "\t" if "\t" in sample else ("|" if "|" in sample else ",")
        reader = csv.reader(handle, delimiter=delimiter)
        for row in reader:
            if len(row) < 3:
                continue
            h, r, t = (str(row[0]).strip(), str(row[1]).strip(), str(row[2]).strip())
            if not h or not r or not t:
                continue
            if (h.lower(), r.lower(), t.lower()) in {
                ("head", "relation", "tail"),
                ("source", "relation", "target"),
                ("subject", "predicate", "object"),
            }:
                continue
            triples.append((h, r, t))
    if not triples:
        raise ValueError(f"No triples could be read from {path}")
    return triples


@dataclass
class QAExample:
    dataset: str
    row_index: int
    question_id: str
    question: str
    start_ids: List[str]
    start_labels: List[str]
    gold_answer_ids: List[str]
    gold_answer_labels: List[str]
    hops: Optional[int]
    split: Optional[str]
    gold_paths: List[List[Tuple[str, str, str]]]
    gold_path_labels: List[List[Tuple[str, str, str]]]
    gold_relation_paths: List[List[str]]


class CustomKGQADataset:
    def __init__(
        self,
        dataset: str,
        data_dir: Path,
        metaqa_hop: Optional[int] = None,
        metaqa_variant: str = "vanilla",
        metaqa_split: Optional[str] = None,
    ):
        self.dataset = dataset
        self.data_dir = Path(data_dir)
        self.qa_path = self.data_dir / "qa_nhop.csv"
        self.original_metaqa = False
        if not self.qa_path.exists():
            if (self.data_dir / "kb.txt").exists():
                self.original_metaqa = True
                hop = metaqa_hop or 1
                split = metaqa_split or "test"
                self.qa_path = self.data_dir / f"{hop}-hop" / metaqa_variant / f"qa_{split}.txt"
                if not self.qa_path.exists():
                    raise FileNotFoundError(f"Missing original MetaQA QA file: {self.qa_path}")
            else:
                raise FileNotFoundError(f"Missing required QA file: {self.qa_path}")

        self.node_labels = read_mapping_csv(self.data_dir / "node_data.csv")
        self.relation_labels = read_mapping_csv(self.data_dir / "relation_data.csv")
        triples_path = self.data_dir / ("kb.txt" if self.original_metaqa else "triplets.txt")
        self.triples = read_triplets(triples_path)
        self._infer_missing_labels()
        self.graph = self._build_graph()
        if self.original_metaqa:
            self.qa_df = load_original_metaqa_qa(self.qa_path, metaqa_hop or 1, metaqa_split or "test")
        else:
            self.qa_df = pd.read_csv(self.qa_path)
        self.columns = list(self.qa_df.columns)

    def _infer_missing_labels(self) -> None:
        for h, r, t in self.triples:
            self.node_labels.setdefault(h, h)
            self.node_labels.setdefault(t, t)
            self.relation_labels.setdefault(r, r)

    def _build_graph(self) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        for h, r, t in self.triples:
            graph.add_node(h, label=self.node_labels.get(h, h))
            graph.add_node(t, label=self.node_labels.get(t, t))
            graph.add_edge(h, t, key=r, relation_id=r, relation_label=self.relation_labels.get(r, r))
        return graph

    def examples(self, split: Optional[str] = None, limit: Optional[int] = None) -> Iterable[QAExample]:
        split_col = first_existing(self.columns, ("SplitLabel", "split", "Split"))
        df = self.qa_df
        if split and split_col:
            df = df[df[split_col].astype(str).str.lower() == split.lower()]
        if limit is not None:
            df = df.head(limit)
        for row_index, row in df.iterrows():
            yield self._row_to_example(int(row_index), row, split_col)

    def _row_to_example(self, row_index: int, row: pd.Series, split_col: Optional[str]) -> QAExample:
        qid_col = first_existing(self.columns, ("Question-Number", "Question-ID", "id", "ID"))
        question_col = first_existing(self.columns, ("Question", "question"))
        if question_col is None:
            raise ValueError(f"{self.qa_path} must contain a Question column")

        start_id_col = first_existing(self.columns, ("Source-Entity", "Start-Entity", "Topic-Entity", "start_entity"))
        start_label_col = first_existing(self.columns, ("Source", "Start", "Topic", "source"))
        answer_id_col = first_existing(self.columns, ("Answer-Entity", "Answer-ID", "Answers-Entity", "answer_entity"))
        answer_label_col = first_existing(self.columns, ("Answer", "Answers", "answer"))
        hop_col = first_existing(self.columns, ("Hops", "Hop", "hop"))

        start_ids = cell_to_list(row[start_id_col]) if start_id_col else []
        start_labels = cell_to_list(row[start_label_col]) if start_label_col else []
        if not start_ids:
            start_ids = [self.resolve_entity(label) for label in start_labels]
        if not start_labels:
            start_labels = [self.node_labels.get(entity_id, entity_id) for entity_id in start_ids]

        gold_answer_ids = cell_to_list(row[answer_id_col]) if answer_id_col else []
        gold_answer_labels = cell_to_list(row[answer_label_col]) if answer_label_col else []
        if not gold_answer_ids:
            gold_answer_ids = [self.resolve_entity(label) for label in gold_answer_labels]
        if not gold_answer_labels:
            gold_answer_labels = [self.node_labels.get(entity_id, entity_id) for entity_id in gold_answer_ids]

        hops = None
        if hop_col and not pd.isna(row[hop_col]):
            try:
                hops = int(row[hop_col])
            except (TypeError, ValueError):
                hops = None

        gold_paths = parse_paths(row.get("Paths", None))
        gold_path_labels = parse_paths(row.get("Paths-Label", None))
        relation_paths = relation_paths_from_columns(row, gold_paths)

        return QAExample(
            dataset=self.dataset,
            row_index=row_index,
            question_id=str(row[qid_col]) if qid_col else str(row_index),
            question=str(row[question_col]),
            start_ids=[str(v) for v in start_ids],
            start_labels=[str(v) for v in start_labels],
            gold_answer_ids=[str(v) for v in gold_answer_ids],
            gold_answer_labels=[str(v) for v in gold_answer_labels],
            hops=hops,
            split=str(row[split_col]) if split_col else None,
            gold_paths=gold_paths,
            gold_path_labels=gold_path_labels,
            gold_relation_paths=relation_paths,
        )

    def resolve_entity(self, value: str) -> str:
        value = str(value)
        if value in self.node_labels:
            return value
        norm = normalize_text(value)
        for entity_id, label in self.node_labels.items():
            if normalize_text(label) == norm:
                return entity_id
        return value

    def entity_label(self, entity_id: str) -> str:
        return self.node_labels.get(str(entity_id), str(entity_id))

    def relation_label(self, relation_id: str) -> str:
        return self.relation_labels.get(str(relation_id), str(relation_id))


def parse_paths(value: Any) -> List[List[Tuple[str, str, str]]]:
    if value is None or (not isinstance(value, list) and pd.isna(value)):
        return []
    parsed = value if isinstance(value, list) else ast.literal_eval(str(value))
    if not parsed:
        return []
    if isinstance(parsed, tuple):
        parsed = list(parsed)
    if parsed and len(parsed) == 3 and not isinstance(parsed[0], (list, tuple)):
        parsed = [parsed]
    paths: List[List[Tuple[str, str, str]]] = []
    for item in parsed:
        if isinstance(item, tuple):
            item = list(item)
        if item and len(item) == 3 and not isinstance(item[0], (list, tuple)):
            paths.append([(str(item[0]), str(item[1]), str(item[2]))])
        else:
            triples = []
            for triple in item:
                if len(triple) >= 3:
                    triples.append((str(triple[0]), str(triple[1]), str(triple[2])))
            if triples:
                paths.append(triples)
    return paths


def relation_paths_from_columns(row: pd.Series, gold_paths: List[List[Tuple[str, str, str]]]) -> List[List[str]]:
    if "Path-Key" in row.index and not pd.isna(row["Path-Key"]):
        values = cell_to_list(row["Path-Key"])
        relation_paths = []
        for value in values:
            if "->" in value:
                relation_paths.append([part.strip() for part in value.split("->") if part.strip()])
            else:
                relation_paths.append([value])
        if relation_paths:
            return relation_paths
    return [[triple[1] for triple in path] for path in gold_paths]


def load_original_metaqa_qa(path: Path, hop: int, split: str) -> pd.DataFrame:
    rows = []
    bracket_re = re.compile(r"\[([^\]]+)\]")
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                question, answers = line.split("\t", 1)
            except ValueError:
                continue
            match = bracket_re.search(question)
            source = match.group(1) if match else ""
            clean_question = question.replace("[", "").replace("]", "")
            answer_list = [answer for answer in answers.split("|") if answer]
            rows.append(
                {
                    "Question-Number": idx,
                    "Question": clean_question,
                    "Source": source,
                    "Source-Entity": source,
                    "Answer": answer_list,
                    "Answer-Entity": answer_list,
                    "Hops": hop,
                    "SplitLabel": split,
                }
            )
    return pd.DataFrame(rows)


def shortest_path_triples(graph: nx.MultiDiGraph, start: str, target: str, undirected: bool) -> List[Tuple[str, str, str]]:
    if start == target:
        return []
    path_graph = graph.to_undirected(as_view=True) if undirected else graph
    try:
        nodes = nx.shortest_path(path_graph, start, target)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
    triples: List[Tuple[str, str, str]] = []
    for h, t in zip(nodes[:-1], nodes[1:]):
        edge_data = graph.get_edge_data(h, t)
        if edge_data is None and undirected:
            edge_data = graph.get_edge_data(t, h)
            if edge_data:
                relation = next(iter(edge_data.values()))["relation_id"]
                triples.append((t, relation, h))
                continue
        if edge_data:
            relation = next(iter(edge_data.values()))["relation_id"]
            triples.append((h, relation, t))
    return triples


def traversal_edge_triple(graph: nx.MultiDiGraph, current: str, neighbor: str, undirected: bool) -> Optional[Tuple[str, str, str]]:
    edge_data = graph.get_edge_data(current, neighbor)
    if edge_data:
        relation = next(iter(edge_data.values()))["relation_id"]
        return current, relation, neighbor
    if undirected:
        edge_data = graph.get_edge_data(neighbor, current)
        if edge_data:
            relation = next(iter(edge_data.values()))["relation_id"]
            return neighbor, relation, current
    return None


def retrieve_candidates(
    dataset: CustomKGQADataset,
    example: QAExample,
    top_k: int,
    max_hops: int,
    undirected: bool = True,
) -> List[Dict[str, Any]]:
    graph = dataset.graph
    adjacency_graph = graph.to_undirected(as_view=True) if undirected else graph
    seen: Dict[str, int] = {}
    paths: Dict[str, List[Tuple[str, str, str]]] = {}
    queue = deque()

    for start in example.start_ids:
        if start in adjacency_graph:
            seen[start] = 0
            paths[start] = []
            queue.append(start)

    question_tokens = token_set(example.question)
    candidates: List[Dict[str, Any]] = []
    while queue:
        node = queue.popleft()
        distance = seen[node]
        if distance >= max_hops:
            continue
        neighbors = adjacency_graph.neighbors(node) if node in adjacency_graph else []
        for neighbor in neighbors:
            if neighbor in seen:
                continue
            edge = traversal_edge_triple(graph, node, neighbor, undirected)
            path = paths[node] + ([edge] if edge else [])
            seen[neighbor] = distance + 1
            paths[neighbor] = path
            queue.append(neighbor)
            relation_tokens = set()
            for _, rel_id, _ in path:
                relation_tokens |= token_set(dataset.relation_label(rel_id))
                relation_tokens |= token_set(rel_id)
            lexical_score = len(question_tokens & relation_tokens)
            score = (1.0 / (distance + 1)) + (0.05 * lexical_score)
            candidates.append(
                {
                    "entity_id": neighbor,
                    "entity_label": dataset.entity_label(neighbor),
                    "score": score,
                    "distance": distance + 1,
                    "path": path,
                    "path_labels": [
                        (dataset.entity_label(h), dataset.relation_label(r), dataset.entity_label(t))
                        for h, r, t in path
                    ],
                    "relation_path": [r for _, r, _ in path],
                    "relation_path_labels": [dataset.relation_label(r) for _, r, _ in path],
                }
            )

    candidates.sort(key=lambda item: (-item["score"], item["distance"], item["entity_id"]))
    return candidates[:top_k]


def match_candidate(candidate: Dict[str, Any], example: QAExample) -> bool:
    gold_ids = {str(v) for v in example.gold_answer_ids}
    gold_labels = {normalize_text(v) for v in example.gold_answer_labels}
    return candidate["entity_id"] in gold_ids or normalize_text(candidate["entity_label"]) in gold_labels


def prf(predicted: Set[Any], gold: Set[Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not gold:
        return None, None, None
    if not predicted:
        return 0.0, 0.0, 0.0
    correct = len(predicted & gold)
    precision = correct / len(predicted)
    recall = correct / len(gold)
    f1 = 0.0 if precision == 0.0 or recall == 0.0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate_example(example: QAExample, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    reciprocal_rank = 0.0
    for idx, candidate in enumerate(candidates, start=1):
        if match_candidate(candidate, example):
            reciprocal_rank = 1.0 / idx
            break
    hits_at_1 = 1.0 if candidates and match_candidate(candidates[0], example) else 0.0

    predicted_triples = {tuple(triple) for cand in candidates for triple in cand["path"]}
    gold_triples = {tuple(triple) for path in example.gold_paths for triple in path}
    sub_p, sub_r, sub_f1 = prf(predicted_triples, gold_triples)

    predicted_relations = {rel for cand in candidates for rel in cand["relation_path"]}
    gold_relations = {rel for rel_path in example.gold_relation_paths for rel in rel_path}
    rel_p, rel_r, rel_f1 = prf(predicted_relations, gold_relations)

    return {
        "hits_at_1": hits_at_1,
        "mrr": reciprocal_rank,
        "answer_found": 1.0 if reciprocal_rank > 0 else 0.0,
        "subgraph_precision": sub_p,
        "subgraph_recall": sub_r,
        "subgraph_f1": sub_f1,
        "relation_precision": rel_p,
        "relation_recall": rel_r,
        "relation_f1": rel_f1,
    }


def mean_defined(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def run_retrieval_evaluation(
    dataset_name: str,
    data_dir: Path,
    output_dir: Path,
    top_k: int = 10,
    limit: Optional[int] = None,
    split: Optional[str] = None,
    max_hops: Optional[int] = None,
    undirected: bool = True,
    metaqa_hop: Optional[int] = None,
    metaqa_variant: str = "vanilla",
) -> Dict[str, Any]:
    dataset = CustomKGQADataset(
        dataset_name,
        Path(data_dir),
        metaqa_hop=metaqa_hop,
        metaqa_variant=metaqa_variant,
        metaqa_split=split,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "retrieval_results.jsonl"
    info_path = output_dir / "retrieval.info"
    records: List[Dict[str, Any]] = []

    with results_path.open("w", encoding="utf-8") as results_file, info_path.open("w", encoding="utf-8") as info_file:
        for example in dataset.examples(split=split, limit=limit):
            depth = max_hops if max_hops is not None else (example.hops or 3)
            candidates = retrieve_candidates(dataset, example, top_k=top_k, max_hops=depth, undirected=undirected)
            metrics = evaluate_example(example, candidates)
            record = {
                "dataset": dataset_name,
                "question_id": example.question_id,
                "row_index": example.row_index,
                "split": example.split,
                "question": example.question,
                "start_entity_ids": example.start_ids,
                "start_entity_labels": example.start_labels,
                "start_entities_found": [entity_id in dataset.graph for entity_id in example.start_ids],
                "gold_answer_ids": example.gold_answer_ids,
                "gold_answer_labels": example.gold_answer_labels,
                "gold_answers_found_in_graph": [entity_id in dataset.graph for entity_id in example.gold_answer_ids],
                "hops": example.hops,
                "retrieved_candidate_answer_ids": [cand["entity_id"] for cand in candidates],
                "retrieved_candidate_answer_labels": [cand["entity_label"] for cand in candidates],
                "retrieved_candidates": candidates,
                "retrieved_subgraph_triples": sorted({tuple(triple) for cand in candidates for triple in cand["path"]}),
                "gold_paths": example.gold_paths,
                "gold_path_labels": example.gold_path_labels,
                "gold_relation_paths": example.gold_relation_paths,
                "metrics": metrics,
            }
            records.append(record)
            results_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            info_file.write(
                json.dumps(
                    {
                        "question": example.question,
                        "answers": example.gold_answer_ids or example.gold_answer_labels,
                        "precison": metrics.get("subgraph_precision"),
                        "recall": metrics.get("subgraph_recall"),
                        "f1": metrics.get("subgraph_f1"),
                        "hit": metrics["hits_at_1"],
                        "em": int(metrics["answer_found"]),
                        "cand": [(cand["entity_id"], cand["score"]) for cand in candidates],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics_path = output_dir / "metrics.json"
    summary = {
        "dataset": dataset_name,
        "data_dir": str(data_dir),
        "num_examples": len(records),
        "top_k": top_k,
        "limit": limit,
        "split": split,
        "max_hops": max_hops,
        "undirected": undirected,
        "metaqa_hop": metaqa_hop,
        "metaqa_variant": metaqa_variant,
        "num_nodes": dataset.graph.number_of_nodes(),
        "num_edges": dataset.graph.number_of_edges(),
        "metrics": {
            "hits_at_1": mean_defined(r["metrics"]["hits_at_1"] for r in records),
            "mrr": mean_defined(r["metrics"]["mrr"] for r in records),
            "answer_found": mean_defined(r["metrics"]["answer_found"] for r in records),
            "subgraph_precision": mean_defined(r["metrics"]["subgraph_precision"] for r in records),
            "subgraph_recall": mean_defined(r["metrics"]["subgraph_recall"] for r in records),
            "subgraph_f1": mean_defined(r["metrics"]["subgraph_f1"] for r in records),
            "relation_precision": mean_defined(r["metrics"]["relation_precision"] for r in records),
            "relation_recall": mean_defined(r["metrics"]["relation_recall"] for r in records),
            "relation_f1": mean_defined(r["metrics"]["relation_f1"] for r in records),
        },
        "outputs": {
            "results_jsonl": str(results_path),
            "gnn_style_info": str(info_path),
            "metrics_json": str(metrics_path),
        },
    }
    metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
