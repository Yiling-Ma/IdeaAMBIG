#!/usr/bin/env python3
"""Remove verbose legacy fields from an existing Task 1 predictions JSONL."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from task1_config import option_mapping


SUCCESS_FIELDS = (
    "status",
    "case_id",
    "model",
    "model_group",
    "predicted_ready",
    "selected_option",
    "option_mapping",
)
ERROR_FIELDS = ("status", "case_id", "model", "model_group", "error")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Atomically replace --input after successful compaction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Option-counterbalancing seed used by the runner.",
    )
    return parser.parse_args()


def compact(row: dict[str, Any], seed: int) -> dict[str, Any]:
    # Recover provider outputs such as "Option B" that an older strict parser
    # recorded as errors. The original response is preserved inside the error
    # message, and the option mapping is deterministic from case_id + seed.
    if row.get("status") == "error":
        match = re.search(
            r"Expected exactly A or B, received ['\"]Option\s+([AB])['\"]",
            str(row.get("error") or ""),
            flags=re.IGNORECASE,
        )
        if match and row.get("case_id"):
            selected_option = match.group(1).upper()
            mapping = option_mapping(str(row["case_id"]), seed)
            row = {
                **row,
                "status": "ok",
                "predicted_ready": mapping[selected_option] == "READY",
                "selected_option": selected_option,
                "option_mapping": mapping,
            }
    fields = SUCCESS_FIELDS if row.get("status") == "ok" else ERROR_FIELDS
    return {field: row[field] for field in fields if field in row}


def main() -> None:
    args = parse_args()
    if args.in_place == bool(args.output):
        raise ValueError("Choose exactly one of --output or --in-place")
    input_path = args.input.resolve()
    output_path = input_path if args.in_place else args.output.resolve()
    temporary = output_path.with_name(f".{output_path.name}.compact.tmp")
    rows = 0
    retained: dict[tuple[str, str], dict[str, Any]] = {}
    with input_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc
            rows += 1
            key = (str(row.get("model")), str(row.get("case_id")))
            compacted = compact(row, args.seed)
            previous = retained.get(key)
            # Prefer a successful prediction over any earlier/later error row.
            if previous is None or previous.get("status") != "ok":
                retained[key] = compacted
    with temporary.open("w", encoding="utf-8") as target:
        for row in retained.values():
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, output_path)
    print(f"Compacted {rows} rows to {len(retained)} model/case rows -> {output_path}")


if __name__ == "__main__":
    main()
