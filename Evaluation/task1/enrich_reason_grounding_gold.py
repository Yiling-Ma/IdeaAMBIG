#!/usr/bin/env python3
"""Augment Task 1 human gold with rationale fields for reason-grounding scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_HUMAN_GOLD = ROOT_DIR / "human_label" / "evaluation_gold.jsonl"
DEFAULT_REAL_BENCH = ROOT_DIR / "Real_bench" / "real_benchmark_instance.jsonl"
DEFAULT_OUTPUT = ROOT_DIR / "human_label" / "evaluation_gold_reason_grounding.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_HUMAN_GOLD)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_REAL_BENCH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def synthesize_ready_reason(source_row: dict[str, Any]) -> str:
    reference = (
        ((source_row.get("eval_targets") or {}).get("reference") or {})
        if isinstance(source_row.get("eval_targets"), dict)
        else {}
    )
    if reference.get("expected_ready") is True:
        return (
            "The codification-ready reference resolves the original blocker and leaves "
            "no implementation-critical missing specification."
        )
    return (
        "The ready version contains no gold implementation-critical blocker under the "
        "benchmark annotation."
    )


def reason_grounding_payload(task_gold_row: dict[str, Any], source_row: dict[str, Any]) -> dict[str, Any]:
    codification = source_row.get("codification_readiness") or {}
    variant = str(task_gold_row.get("variant", ""))
    if variant == "incomplete":
        return {
            "gold_label": "NOT_READY",
            "defects": source_row.get("defects") or [],
            "blocking_missing_specs": codification.get("blocking_missing_specs") or [],
            "readiness_reason": codification.get("reason") or "",
            "expected_clarification_actions": source_row.get("expected_clarification_actions") or [],
        }

    return {
        "gold_label": "READY",
        "defects": [],
        "blocking_missing_specs": [],
        "readiness_reason": synthesize_ready_reason(source_row),
        "expected_clarification_actions": [],
        "resolved_from_source_blockers": codification.get("blocking_missing_specs") or [],
        "source_not_ready_reason": codification.get("reason") or "",
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {args.output}")

    benchmark_rows = load_jsonl(args.benchmark)
    benchmark_by_id = {str(row["id"]): row for row in benchmark_rows}
    gold_rows = load_jsonl(args.gold)

    output_rows: list[dict[str, Any]] = []
    for row in gold_rows:
        instance_id = str(row["instance_id"])
        if instance_id not in benchmark_by_id:
            raise KeyError(f"Instance ID absent from real benchmark file: {instance_id}")
        source_row = benchmark_by_id[instance_id]
        enriched = dict(row)
        enriched["reason_grounding_gold"] = reason_grounding_payload(row, source_row)
        output_rows.append(enriched)

    write_jsonl(args.output, output_rows)
    print(
        json.dumps(
            {
                "rows_written": len(output_rows),
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
