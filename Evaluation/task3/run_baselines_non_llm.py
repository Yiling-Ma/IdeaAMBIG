#!/usr/bin/env python3
"""Generate deterministic non-LLM Task 3 baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from task3_config import (
    DEFAULT_GOLD,
    DEFAULT_POOL,
    DEFAULT_PREDICTIONS_DIR,
    load_jsonl,
    model_filename,
)


BASELINES = {
    "generic": "baseline/generic_question",
    "defect_template": "baseline/defect_template",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument(
        "--output",
        type=Path,
        help="Explicit output file; allowed only when exactly one baseline is selected.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PREDICTIONS_DIR,
        help="Root directory for automatic per-baseline prediction files.",
    )
    parser.add_argument("--baselines", default="all")
    parser.add_argument("--subset", choices=("all", "real", "synthetic"), default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def selected_baselines(spec: str) -> list[str]:
    if spec == "all":
        return list(BASELINES.values())
    values = [value.strip() for value in spec.split(",") if value.strip()]
    resolved = [BASELINES.get(value, value) for value in values]
    unknown = [value for value in resolved if value not in BASELINES.values()]
    if unknown:
        raise ValueError("Unknown baselines: " + ", ".join(unknown))
    return resolved


def baseline_prediction(model: str, target_defect: str) -> dict[str, str]:
    if model == "baseline/generic_question":
        return {
            "action_type": "clarification_question",
            "action": "Could you provide more implementation details?",
            "expected_information": "Additional implementation details.",
        }
    if model == "baseline/defect_template":
        return {
            "action_type": "clarification_question",
            "action": f"Please specify the exact intended implementation for this issue: {target_defect}",
            "expected_information": f"The intended implementation detail that resolves: {target_defect}",
        }
    raise ValueError(f"Unsupported baseline: {model}")


def main() -> None:
    args = parse_args()
    cases = load_jsonl(args.pool.resolve())
    gold = {str(row["case_id"]): row for row in load_jsonl(args.gold.resolve())}
    if args.subset != "all":
        cases = [
            case
            for case in cases
            if gold[str(case["case_id"])]["source_category"] == args.subset
        ]
    if args.limit > 0:
        cases = cases[: args.limit]
    models = selected_baselines(args.baselines)
    if args.output is not None:
        if len(models) != 1:
            raise ValueError("--output may be used only when exactly one baseline is selected")
        output_paths = {models[0]: args.output}
    else:
        output_paths = {
            model: args.output_dir / "deterministic" / args.subset / model_filename(model)
            for model in models
        }
    protected_inputs = {args.pool.resolve(), args.gold.resolve()}
    invalid_outputs = [
        path for path in output_paths.values() if path.resolve() in protected_inputs
    ]
    if invalid_outputs:
        raise ValueError(
            "Prediction output must not overwrite the evaluation pool or gold file: "
            + ", ".join(str(path.resolve()) for path in invalid_outputs)
        )
    if args.overwrite:
        for path in output_paths.values():
            if path.exists():
                path.unlink()

    done_by_model: dict[str, set[tuple[str, str]]] = {}
    for model, path in output_paths.items():
        done_by_model[model] = set()
        if path.exists():
            done_by_model[model] = {
                (str(row.get("model")), str(row.get("case_id")))
                for row in load_jsonl(path)
                if row.get("status") == "ok"
            }
        path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    for model in models:
        with output_paths[model].open("a", encoding="utf-8") as handle:
            for case in cases:
                key = (model, str(case["case_id"]))
                if key in done_by_model[model]:
                    continue
                row: dict[str, Any] = {
                    "status": "ok",
                    "case_id": case["case_id"],
                    "model": model,
                    "model_group": "baseline",
                    "prompt_version": "deterministic",
                    "prediction": baseline_prediction(model, str(case["target_defect"])),
                }
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                written += 1
    print(f"Generated {written} baseline predictions.")
    for model, path in output_paths.items():
        print(f"{model} -> {path.resolve()}")


if __name__ == "__main__":
    main()
