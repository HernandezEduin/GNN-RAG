import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from utils.custom_kgqa import parse_paths, read_mapping_csv, relation_paths_from_columns

Triple = Tuple[str, str, str]

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise ValueError(f"{path} must contain a JSON array or JSONL records")
        return rows
    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse JSON on {path}:{line_no}: {exc}") from exc
    return rows


def as_triple_set(triples: Iterable[Sequence[Any]]) -> Set[Triple]:
    result: Set[Triple] = set()
    for triple in triples:
        if len(triple) < 3:
            continue
        result.add((str(triple[0]), str(triple[1]), str(triple[2])))
    return result


def prf(predicted: Set[Any], gold: Set[Any]) -> Dict[str, Optional[float]]:
    if not gold:
        return {"precision": None, "recall": None, "f1": None}
    if not predicted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    correct = len(predicted & gold)
    precision = correct / len(predicted)
    recall = correct / len(gold)
    f1 = 0.0 if precision == 0.0 or recall == 0.0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    materialized = [value for value in values if value is not None]
    if not materialized:
        return None
    return sum(materialized) / len(materialized)


def hop_key(value: Any) -> str:
    if value is None:
        return "unknown"
    try:
        if pd.isna(value):
            return "unknown"
    except TypeError:
        pass
    text = str(value).strip()
    return text if text else "unknown"


def candidate_ids(info_row: Dict[str, Any], top_k: Optional[int]) -> List[str]:
    candidates = info_row.get("cand", [])
    if top_k is not None:
        candidates = candidates[:top_k]
    return [str(item[0]) for item in candidates if item]


def candidate_objects(info_row: Dict[str, Any], labels: Dict[str, str], top_k: Optional[int]) -> List[Dict[str, Any]]:
    candidates = info_row.get("cand", [])
    if top_k is not None:
        candidates = candidates[:top_k]
    objects = []
    for rank, item in enumerate(candidates, start=1):
        if not item:
            continue
        entity_id = str(item[0])
        score = float(item[1]) if len(item) > 1 else None
        objects.append(
            {
                "rank": rank,
                "entity_id": entity_id,
                "label": labels.get(entity_id, entity_id),
                "score": score,
            }
        )
    return objects


def h1_and_mrr(candidates: Sequence[str], gold_answers: Set[str]) -> Tuple[float, float, Optional[int]]:
    if not candidates:
        return 0.0, 0.0, None
    h1 = 1.0 if candidates[0] in gold_answers else 0.0
    for index, candidate in enumerate(candidates, start=1):
        if candidate in gold_answers:
            return h1, 1.0 / index, index
    return h1, 0.0, None


def load_gold_paths_by_row(data_dir: Optional[Path]) -> Dict[int, Dict[str, Set[Any]]]:
    if data_dir is None:
        return {}
    qa_path = data_dir / "qa_nhop.csv"
    if not qa_path.exists():
        raise FileNotFoundError(f"Missing QA file for gold paths: {qa_path}")
    qa_df = pd.read_csv(qa_path)
    by_row: Dict[int, Dict[str, Set[Any]]] = {}
    for row_index, row in qa_df.iterrows():
        gold_paths = parse_paths(row.get("Paths", None))
        relation_paths = relation_paths_from_columns(row, gold_paths)
        gold_triples = {triple for path in gold_paths for triple in path}
        gold_relations = {str(relation) for path in relation_paths for relation in path}
        by_row[int(row_index)] = {
            "triples": gold_triples,
            "relations": gold_relations,
        }

    return by_row


def induced_subgraph(
    processed_row: Dict[str, Any],
    candidate_set: Set[str],
    include_seed_entities: bool,
) -> Set[Triple]:
    entity_set = set(candidate_set)
    if include_seed_entities:
        entity_set.update(str(entity) for entity in processed_row.get("entities", []))
    tuples = processed_row.get("subgraph", {}).get("tuples", [])
    return {
        (str(h), str(r), str(t))
        for h, r, t in as_triple_set(tuples)
        if h in entity_set and t in entity_set
    }


def evaluate(
    info_file: Path,
    processed_file: Path,
    output_dir: Path,
    data_dir: Optional[Path],
    dataset: Optional[str],
    top_k: Optional[int],
    include_seed_entities: bool,
) -> Dict[str, Any]:
    info_rows = load_jsonl(info_file)
    processed_rows = load_jsonl(processed_file)
    if len(info_rows) != len(processed_rows):
        raise ValueError(
            f"Mismatched row counts: {info_file} has {len(info_rows)} rows, "
            f"but {processed_file} has {len(processed_rows)} rows."
        )

    labels = read_mapping_csv(data_dir / "node_data.csv") if data_dir else {}
    gold_by_row = load_gold_paths_by_row(data_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "details.jsonl"

    h1_values = []
    mrr_values = []
    subgraph_f1_values = []
    subgraph_precision_values = []
    subgraph_recall_values = []
    relation_f1_values = []
    relation_precision_values = []
    relation_recall_values = []
    per_hop: Dict[str, Dict[str, List[float]]] = {}

    with details_path.open("w", encoding="utf-8") as details:
        for index, (info_row, processed_row) in enumerate(zip(info_rows, processed_rows)):
            candidates = candidate_ids(info_row, top_k)
            candidate_set = set(candidates)
            gold_answers = {str(answer["kb_id"]) for answer in processed_row.get("answers", [])}
            h1, mrr, gold_rank = h1_and_mrr(candidates, gold_answers)

            pred_triples = induced_subgraph(processed_row, candidate_set, include_seed_entities)
            pred_relations = {relation for _, relation, _ in pred_triples}

            row_index = int(processed_row.get("metadata", {}).get("row_index", index))
            hop = hop_key(processed_row.get("metadata", {}).get("hops"))
            gold_path_info = gold_by_row.get(row_index, {"triples": set(), "relations": set()})
            gold_triples = set(gold_path_info["triples"])
            gold_relations = set(gold_path_info["relations"])

            subgraph_metrics = prf(pred_triples, gold_triples)
            relation_metrics = prf(pred_relations, gold_relations)

            h1_values.append(h1)
            mrr_values.append(mrr)
            subgraph_precision_values.append(subgraph_metrics["precision"])
            subgraph_recall_values.append(subgraph_metrics["recall"])
            subgraph_f1_values.append(subgraph_metrics["f1"])
            relation_precision_values.append(relation_metrics["precision"])
            relation_recall_values.append(relation_metrics["recall"])
            relation_f1_values.append(relation_metrics["f1"])
            per_hop.setdefault(hop, {"h1": [], "mrr": []})
            per_hop[hop]["h1"].append(h1)
            per_hop[hop]["mrr"].append(mrr)

            detail = {
                "dataset": dataset,
                "index": index,
                "question_id": processed_row.get("id"),
                "base_question_id": processed_row.get("metadata", {}).get("base_question_id"),
                "question_variant": processed_row.get("metadata", {}).get("question_variant"),
                "question_variant_index": processed_row.get("metadata", {}).get("question_variant_index"),
                "row_index": row_index,
                "question": processed_row.get("question"),
                "hops": hop,
                "start_entity_ids": processed_row.get("entities", []),
                "gold_answer_ids": sorted(gold_answers),
                "gold_answer_labels": processed_row.get("metadata", {}).get("gold_answer_labels", []),
                "candidates": candidate_objects(info_row, labels, top_k),
                "gold_rank": gold_rank,
                "candidate_induced_subgraph": [list(triple) for triple in sorted(pred_triples)],
                "gold_path_triples": [list(triple) for triple in sorted(gold_triples)],
                "metrics": {
                    "h1": h1,
                    "mrr": mrr,
                    "subgraph_overlap": subgraph_metrics,
                    "relation_overlap": relation_metrics,
                },
            }
            details.write(json.dumps(detail, ensure_ascii=False) + "\n")

    metrics = {
        "dataset": dataset,
        "num_examples": len(info_rows),
        "info_file": str(info_file),
        "processed_file": str(processed_file),
        "data_dir": str(data_dir) if data_dir else None,
        "top_k": top_k,
        "include_seed_entities": include_seed_entities,
        "h1": mean(h1_values),
        "mrr": mean(mrr_values),
        "per_hop": {
            hop: {
                "num_examples": len(values["h1"]),
                "h1": mean(values["h1"]),
                "mrr": mean(values["mrr"]),
            }
            for hop, values in sorted(per_hop.items(), key=lambda item: (item[0] == "unknown", item[0]))
        },
        "subgraph_overlap": {
            "precision": mean(subgraph_precision_values),
            "recall": mean(subgraph_recall_values),
            "f1": mean(subgraph_f1_values),
        },
        "relation_overlap": {
            "precision": mean(relation_precision_values),
            "recall": mean(relation_recall_values),
            "f1": mean(relation_f1_values),
        },
        "outputs": {
            "details": str(details_path),
        },
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Externally evaluate GNN retriever .info outputs with H1, MRR, and candidate-induced evidence overlap."
    )
    parser.add_argument("--info_file", required=True, type=Path, help="Model output .info JSONL file.")
    parser.add_argument("--processed_file", required=True, type=Path, help="Processed split file, usually test.json. JSON array and JSONL are both supported.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory for metrics.json and details.jsonl.")
    parser.add_argument("--data_dir", type=Path, default=None, help="Original custom dataset dir with qa_nhop.csv for gold Paths.")
    parser.add_argument("--dataset", default=None, help="Dataset name to copy into outputs.")
    parser.add_argument("--top_k", type=int, default=None, help="Only use the top K candidates from .info. Default: all retained candidates.")
    parser.add_argument(
        "--include_seed_entities",
        action="store_true",
        help="Add topic/start entities to the candidate set before inducing the evidence subgraph.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(
        info_file=args.info_file,
        processed_file=args.processed_file,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        dataset=args.dataset,
        top_k=args.top_k,
        include_seed_entities=args.include_seed_entities,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
