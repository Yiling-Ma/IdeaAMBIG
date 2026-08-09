#!/usr/bin/env python3
"""Summarize Task 3 information-bottleneck metrics for the paper table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRICS = (
    ("Macro-CAS", "macro_clarification_action_success"),
    ("Sufficiency", "resolution_sufficiency_rate"),
    ("No-Assumption", "no_assumption_rate"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-to-end", type=Path, required=True)
    parser.add_argument("--defect-guided", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-5.6-sol")
    parser.add_argument("--subset", default="real")
    return parser.parse_args()


def load_row(path: Path, model: str, subset: str) -> dict[str, Any]:
    metrics_path = path / "metrics.json" if path.is_dir() else path
    rows = json.loads(metrics_path.read_text(encoding="utf-8"))
    matches = [
        row
        for row in rows
        if str(row.get("model")) == model and str(row.get("subset")) == subset
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one row for model={model!r}, subset={subset!r} in "
            f"{metrics_path}, found {len(matches)}"
        )
    return matches[0]


def pct(row: dict[str, Any], key: str) -> float:
    return 100.0 * float(row[key])


def fmt(value: float, signed: bool = False) -> str:
    prefix = "+" if signed and value >= 0 else ""
    return f"{prefix}{value:.1f}"


def main() -> None:
    args = parse_args()
    e2e = load_row(args.end_to_end, args.model, args.subset)
    guided = load_row(args.defect_guided, args.model, args.subset)

    print("Paper table values:")
    for label, key in METRICS:
        print(
            f"{label}: End-to-End={fmt(pct(e2e, key))}, "
            f"Defect-Guided={fmt(pct(guided, key))}, "
            f"Delta={fmt(pct(guided, key) - pct(e2e, key), signed=True)}"
        )

    print("\nLaTeX rows:")
    e2e_values = " & ".join(fmt(pct(e2e, key)) for _, key in METRICS)
    guided_values = " & ".join(fmt(pct(guided, key)) for _, key in METRICS)
    delta_values = " & ".join(
        fmt(pct(guided, key) - pct(e2e, key), signed=True) for _, key in METRICS
    )
    print(f"\\textsc{{End-to-End}} & {e2e_values} \\\\")
    print(f"\\textsc{{Defect-Guided}} & {guided_values} \\\\")
    print(f"\\textbf{{$\\Delta$}} & {delta_values} \\\\")


if __name__ == "__main__":
    main()
