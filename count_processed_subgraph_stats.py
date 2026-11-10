#!/usr/bin/env python
"""Count subgraph tuple/entity statistics for processed KGQA files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional


def iter_json_records(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fin:
        first_char = ""
        while True:
            char = fin.read(1)
            if not char:
                return
            if not char.isspace():
                first_char = char
                break

        fin.seek(0)
        if first_char == "[":
            rows = json.load(fin)
            if not isinstance(rows, list):
                raise ValueError(f"{path} must contain a JSON array or JSONL records")
            yield from rows
            return

        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Could not parse JSON on {path}:{line_no}: {exc}") from exc


def safe_mean(total: int, count: int) -> Optional[float]:
    if count == 0:
        return None
    return total / count


def summarize_file(path: Path) -> Dict[str, Any]:
    num_examples = 0
    tuple_total = 0
    entity_total = 0
    tuple_min: Optional[int] = None
    tuple_max = 0
    entity_min: Optional[int] = None
    entity_max = 0
    by_hop: Dict[str, Dict[str, int]] = {}

    for row in iter_json_records(path):
        subgraph = row.get("subgraph") or {}
        tuples = subgraph.get("tuples") or []
        entities = subgraph.get("entities") or []
        tuple_count = len(tuples)
        entity_count = len(entities)

        num_examples += 1
        tuple_total += tuple_count
        entity_total += entity_count
        tuple_min = tuple_count if tuple_min is None else min(tuple_min, tuple_count)
        tuple_max = max(tuple_max, tuple_count)
        entity_min = entity_count if entity_min is None else min(entity_min, entity_count)
        entity_max = max(entity_max, entity_count)

        hop = str((row.get("metadata") or {}).get("hops", "unknown"))
        hop_stats = by_hop.setdefault(hop, {"num_examples": 0, "tuple_total": 0, "entity_total": 0})
        hop_stats["num_examples"] += 1
        hop_stats["tuple_total"] += tuple_count
        hop_stats["entity_total"] += entity_count

    return {
        "file": str(path),
        "num_examples": num_examples,
        "tuple_total": tuple_total,
        "avg_tuples_per_subgraph": safe_mean(tuple_total, num_examples),
        "min_tuples_per_subgraph": tuple_min,
        "max_tuples_per_subgraph": tuple_max if num_examples else None,
        "entity_total": entity_total,
        "avg_entities_per_subgraph": safe_mean(entity_total, num_examples),
        "min_entities_per_subgraph": entity_min,
        "max_entities_per_subgraph": entity_max if num_examples else None,
        "by_hop": {
            hop: {
                "num_examples": values["num_examples"],
                "avg_tuples_per_subgraph": safe_mean(values["tuple_total"], values["num_examples"]),
                "avg_entities_per_subgraph": safe_mean(values["entity_total"], values["num_examples"]),
            }
            for hop, values in sorted(by_hop.items(), key=lambda item: (item[0] == "unknown", item[0]))
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count processed KGQA subgraph tuple/entity statistics.")
    parser.add_argument("--data_dir", type=Path, default=None, help="Processed folder containing train/dev/test JSON files.")
    parser.add_argument("--files", nargs="*", type=Path, default=None, help="Specific processed JSON/JSONL files.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write the JSON summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.files:
        files = args.files
    elif args.data_dir:
        files = [args.data_dir / f"{split}.json" for split in ("train", "dev", "test")]
    else:
        raise SystemExit("Provide either --data_dir or --files")

    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")

    file_summaries = {path.stem: summarize_file(path) for path in files}
    total_examples = sum(summary["num_examples"] for summary in file_summaries.values())
    total_tuples = sum(summary["tuple_total"] for summary in file_summaries.values())
    total_entities = sum(summary["entity_total"] for summary in file_summaries.values())
    result = {
        "files": file_summaries,
        "overall": {
            "num_examples": total_examples,
            "tuple_total": total_tuples,
            "avg_tuples_per_subgraph": safe_mean(total_tuples, total_examples),
            "entity_total": total_entities,
            "avg_entities_per_subgraph": safe_mean(total_entities, total_examples),
        },
    }

    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
