import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd


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


def h1_and_mrr(candidates: Sequence[str], gold_answers: Set[str]) -> Tuple[float, float, Optional[int]]:
    if not candidates:
        return 0.0, 0.0, None
    h1 = 1.0 if candidates[0] in gold_answers else 0.0
    for index, candidate in enumerate(candidates, start=1):
        if candidate in gold_answers:
            return h1, 1.0 / index, index
    return h1, 0.0, None


def infer_experiment_name(info_file: Path, experiment_name: Optional[str]) -> str:
    if experiment_name:
        return experiment_name
    name = info_file.name
    if name.endswith("_test.info"):
        return name[: -len("_test.info")]
    if name.endswith(".info"):
        return name[: -len(".info")]
    return info_file.stem


def evaluate(
    info_file: Path,
    processed_file: Path,
    dataset: Optional[str],
    top_k: Optional[int],
    experiment_name: Optional[str],
) -> Dict[str, Any]:
    info_rows = load_jsonl(info_file)
    processed_rows = load_jsonl(processed_file)
    if len(info_rows) != len(processed_rows):
        raise ValueError(
            f"Mismatched row counts: {info_file} has {len(info_rows)} rows, "
            f"but {processed_file} has {len(processed_rows)} rows."
        )

    h1_values = []
    mrr_values = []
    per_hop: Dict[str, Dict[str, List[float]]] = {}

    for info_row, processed_row in zip(info_rows, processed_rows):
        candidates = candidate_ids(info_row, top_k)
        gold_answers = {str(answer["kb_id"]) for answer in processed_row.get("answers", [])}
        h1, mrr, _ = h1_and_mrr(candidates, gold_answers)

        hop = hop_key(processed_row.get("metadata", {}).get("hops"))
        h1_values.append(h1)
        mrr_values.append(mrr)
        per_hop.setdefault(hop, {"h1": [], "mrr": []})
        per_hop[hop]["h1"].append(h1)
        per_hop[hop]["mrr"].append(mrr)

    resolved_experiment_name = infer_experiment_name(info_file, experiment_name)
    metrics_path = info_file.parent / f"{resolved_experiment_name}_metrics.json"
    metrics = {
        "dataset": dataset,
        "experiment_name": resolved_experiment_name,
        "num_examples": len(info_rows),
        "info_file": str(info_file),
        "processed_file": str(processed_file),
        "top_k": top_k,
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
        "outputs": {
            "metrics": str(metrics_path),
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Externally evaluate ReaRev .info outputs with H1, MRR, and per-hop H1/MRR."
    )
    parser.add_argument("--info_file", required=True, type=Path, help="Model output .info JSONL file.")
    parser.add_argument("--processed_file", required=True, type=Path, help="Processed split file, usually test.json. JSON array and JSONL are both supported.")
    parser.add_argument("--experiment_name", default=None, help="Name used for <experiment_name>_metrics.json. Defaults to the .info filename before _test.info.")
    parser.add_argument("--dataset", default=None, help="Dataset name to copy into outputs.")
    parser.add_argument("--top_k", type=int, default=None, help="Only use the top K candidates from .info. Default: all retained candidates.")
    parser.add_argument("--output_dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--data_dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--include_seed_entities", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(
        info_file=args.info_file,
        processed_file=args.processed_file,
        dataset=args.dataset,
        top_k=args.top_k,
        experiment_name=args.experiment_name,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
