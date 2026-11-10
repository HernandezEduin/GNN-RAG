#!/usr/bin/env python
"""Report size, gold-answer coverage, and reachability of processed KGQA subgraphs."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Set


def iter_json_records(path: Path) -> Iterator[Dict[str, Any]]:
    """Read a processed JSON array or JSONL file."""
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
    return total / count if count else None


def new_stats() -> Dict[str, Any]:
    return {
        "num_examples": 0,
        "tuple_total": 0,
        "tuple_min": None,
        "tuple_max": None,
        "entity_total": 0,
        "entity_min": None,
        "entity_max": None,
        "num_with_gold_answers": 0,
        "any_answer_present": 0,
        "all_answers_present": 0,
        "any_answer_reachable": 0,
        "all_answers_reachable": 0,
    }


def reachable_entities(starts: Set[str], tuples: list) -> Set[str]:
    """Traverse the stored subgraph in the original KG edge direction."""
    adjacency: Dict[str, Set[str]] = {}
    for triple in tuples:
        if len(triple) >= 3:
            head, _, tail = triple[:3]
            adjacency.setdefault(str(head), set()).add(str(tail))
    visited = set(starts)
    queue = deque(starts)
    while queue:
        for neighbor in adjacency.get(queue.popleft(), ()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited


def update_stats(stats: Dict[str, Any], row: Dict[str, Any]) -> None:
    subgraph = row.get("subgraph") or {}
    tuples = subgraph.get("tuples") or []
    entities = {str(entity_id) for entity_id in (subgraph.get("entities") or [])}
    tuple_count, entity_count = len(tuples), len(entities)

    stats["num_examples"] += 1
    for kind, count in (("tuple", tuple_count), ("entity", entity_count)):
        stats[f"{kind}_total"] += count
        minimum, maximum = f"{kind}_min", f"{kind}_max"
        stats[minimum] = count if stats[minimum] is None else min(stats[minimum], count)
        stats[maximum] = count if stats[maximum] is None else max(stats[maximum], count)

    answers = {
        str(answer["kb_id"] if isinstance(answer, dict) else answer)
        for answer in (row.get("answers") or [])
        if not isinstance(answer, dict) or answer.get("kb_id") is not None
    }
    if not answers:
        return

    stats["num_with_gold_answers"] += 1
    covered = answers & entities
    stats["any_answer_present"] += bool(covered)
    stats["all_answers_present"] += answers <= entities

    starts = {
        str(start["kb_id"] if isinstance(start, dict) else start)
        for start in (row.get("entities") or [])
        if not isinstance(start, dict) or start.get("kb_id") is not None
    }
    reached = reachable_entities(starts & entities, tuples) & entities
    stats["any_answer_reachable"] += bool(answers & reached)
    stats["all_answers_reachable"] += answers <= reached


def render_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "num_examples": stats["num_examples"],
        "tuple_total": stats["tuple_total"],
        "avg_tuples_per_subgraph": safe_mean(stats["tuple_total"], stats["num_examples"]),
        "min_tuples_per_subgraph": stats["tuple_min"],
        "max_tuples_per_subgraph": stats["tuple_max"],
        "entity_total": stats["entity_total"],
        "avg_entities_per_subgraph": safe_mean(stats["entity_total"], stats["num_examples"]),
        "min_entities_per_subgraph": stats["entity_min"],
        "max_entities_per_subgraph": stats["entity_max"],
        "num_with_gold_answers": stats["num_with_gold_answers"],
    }
    for name in (
        "any_answer_present", "all_answers_present",
        "any_answer_reachable", "all_answers_reachable",
    ):
        result[name] = stats[name]
        result[f"{name}_rate"] = safe_mean(
            stats[name], stats["num_with_gold_answers"]
        )
    return result


def hop_key(hop: str) -> tuple:
    try:
        return (0, int(hop))
    except (TypeError, ValueError):
        return (1, hop)


def summarize_file(path: Path) -> Dict[str, Any]:
    stats = new_stats()
    by_hop: Dict[str, Dict[str, Any]] = {}
    for row in iter_json_records(path):
        update_stats(stats, row)
        raw_hop = (row.get("metadata") or {}).get("hops")
        hop = str(raw_hop) if raw_hop is not None else "unknown"
        update_stats(by_hop.setdefault(hop, new_stats()), row)

    return {
        "file": str(path),
        **render_stats(stats),
        "by_hop": {
            hop: render_stats(values)
            for hop, values in sorted(by_hop.items(), key=lambda item: hop_key(item[0]))
        },
    }


def merge_stats(target: Dict[str, Any], summary: Dict[str, Any]) -> None:
    for field in (
        "num_examples", "tuple_total", "entity_total", "num_with_gold_answers",
        "any_answer_present", "all_answers_present",
        "any_answer_reachable", "all_answers_reachable",
    ):
        target[field] += summary[field]
    for kind in ("tuple", "entity"):
        minimum = summary[f"min_{kind}s_per_subgraph"]
        maximum = summary[f"max_{kind}s_per_subgraph"]
        if minimum is not None:
            key = f"{kind}_min"
            target[key] = minimum if target[key] is None else min(target[key], minimum)
        if maximum is not None:
            key = f"{kind}_max"
            target[key] = maximum if target[key] is None else max(target[key], maximum)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count processed KGQA subgraph sizes, gold-answer coverage and directed reachability."
    )
    parser.add_argument("--data_dir", type=Path, default=None,
                        help="Processed folder containing train/dev/test JSON files.")
    parser.add_argument("--files", nargs="*", type=Path, default=None,
                        help="Specific processed JSON/JSONL files.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional path to write the JSON summary.")
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
    overall = new_stats()
    overall_hops: Dict[str, Dict[str, Any]] = {}
    for summary in file_summaries.values():
        merge_stats(overall, summary)
        for hop, values in summary["by_hop"].items():
            merge_stats(overall_hops.setdefault(hop, new_stats()), values)

    result = {
        "files": file_summaries,
        "overall": {
            **render_stats(overall),
            "by_hop": {
                hop: render_stats(values)
                for hop, values in sorted(overall_hops.items(), key=lambda item: hop_key(item[0]))
            },
        },
    }

    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
