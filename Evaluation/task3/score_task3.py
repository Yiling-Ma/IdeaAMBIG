#!/usr/bin/env python3
"""Score Task 3 clarification actions with Macro-CAS as the primary metric."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from task3_config import (
    DEFAULT_GOLD,
    DEFAULT_JUDGMENTS,
    DEFAULT_POOL,
    DEFAULT_PREDICTIONS_DIR,
    DEFAULT_SCORES_DIR,
    MODELS,
    load_jsonl,
    load_jsonl_collection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SCORES_DIR)
    parser.add_argument("--judge-model", default="anthropic/claude-opus-4.8")
    parser.add_argument("--subsets", default="all,real,synthetic")
    parser.add_argument("--models", default="all")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Count missing predictions or judgments as failures; final runs should be complete.",
    )
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def class_macro(rows: list[dict[str, Any]], label_key: str, metric_key: str) -> float:
    by_class: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_class[str(row[label_key])].append(float(row[metric_key]))
    return mean([mean(values) for values in by_class.values()])


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def clustered_bootstrap_ci(
    rows: list[dict[str, Any]],
    metric: Callable[[list[dict[str, Any]]], float],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if not rows or samples <= 0:
        return 0.0, 0.0
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["paper_id"])].append(row)
    group_ids = sorted(groups)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        sampled: list[dict[str, Any]] = []
        for group_id in rng.choices(group_ids, k=len(group_ids)):
            sampled.extend(groups[group_id])
        estimates.append(metric(sampled))
    estimates.sort()
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def metrics_for(rows: list[dict[str, Any]], bootstrap_samples: int, seed: int) -> dict[str, Any]:
    cas = lambda rs: mean([float(row["action_success"]) for row in rs])
    macro_cas = lambda rs: class_macro(rs, "gold_level2", "action_success")
    result: dict[str, Any] = {
        "n_cases": len(rows),
        "n_papers": len({str(row["paper_id"]) for row in rows}),
        "prediction_coverage": mean([float(row["prediction_available"]) for row in rows]),
        "judgment_coverage": mean([float(row["judgment_available"]) for row in rows]),
        "clarification_action_success": cas(rows),
        "macro_clarification_action_success": macro_cas(rows),
        "target_relevance_rate": mean([float(row["target_relevant"]) for row in rows]),
        "resolution_sufficiency_rate": mean(
            [float(row["resolution_sufficient"]) for row in rows]
        ),
        "no_assumption_rate": mean([float(row["no_assumption"]) for row in rows]),
        "atomicity_rate": mean([float(row["atomic"]) for row in rows]),
        "action_type_agreement": mean(
            [float(row["action_type_agreement"]) for row in rows]
        ),
    }
    targeted = [row for row in rows if row["target_relevant"]]
    result["sufficiency_given_target_relevant"] = mean(
        [float(row["resolution_sufficient"]) for row in targeted]
    )
    for offset, (name, metric) in enumerate(
        (
            ("clarification_action_success", cas),
            ("macro_clarification_action_success", macro_cas),
        )
    ):
        low, high = clustered_bootstrap_ci(
            rows, metric, bootstrap_samples, seed + 1009 * offset
        )
        result[f"{name}_ci95_low"] = low
        result[f"{name}_ci95_high"] = high
    return result


def parse_model_filter(spec: str) -> set[str] | None:
    if spec == "all":
        return None
    models = {value.strip() for value in spec.split(",") if value.strip()}
    if not models:
        raise ValueError("--models must be 'all' or comma-separated model IDs")
    return models


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "model",
        "subset",
        "macro_clarification_action_success",
        "clarification_action_success",
        "target_relevance_rate",
        "resolution_sufficiency_rate",
        "no_assumption_rate",
        "prediction_coverage",
    ]
    labels = [
        "Model",
        "Subset",
        "Macro-CAS ↑",
        "CAS ↑",
        "Target ↑",
        "Sufficiency ↑",
        "No Assumption ↑",
        "Coverage ↑",
    ]
    lines = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join(["---"] * len(labels)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    for label, path in (("prediction", args.predictions), ("judgment", args.judgments)):
        if not path.exists():
            raise FileNotFoundError(f"{label.capitalize()} input not found: {path.resolve()}")
    subsets = [value.strip() for value in args.subsets.split(",") if value.strip()]
    invalid = sorted(set(subsets) - {"all", "real", "synthetic"})
    if invalid or not subsets:
        raise ValueError("Invalid --subsets: " + ", ".join(invalid))
    model_filter = parse_model_filter(args.models)
    pool = {str(row["case_id"]): row for row in load_jsonl(args.pool.resolve())}
    gold = {str(row["case_id"]): row for row in load_jsonl(args.gold.resolve())}
    if set(pool) != set(gold):
        raise ValueError("Pool and gold case_id sets do not match")

    raw_prediction_rows = [
        row
        for row in load_jsonl_collection(args.predictions.resolve())
        if model_filter is None or str(row.get("model")) in model_filter
    ]
    all_models = sorted(
        {str(row.get("model")) for row in raw_prediction_rows if row.get("model")}
    )
    if model_filter is not None:
        missing_models = sorted(model_filter - set(all_models))
        if missing_models:
            raise ValueError(f"Selected models absent from predictions: {missing_models}")
    if not all_models:
        raise ValueError("No evaluated models found")

    predictions: dict[tuple[str, str], dict[str, Any]] = {}
    observed_groups: dict[str, str] = {}
    duplicates: set[tuple[str, str]] = set()
    for row in raw_prediction_rows:
        model = str(row.get("model"))
        case_id = str(row.get("case_id"))
        if row.get("model_group"):
            observed_groups[model] = str(row["model_group"])
        if row.get("status") == "ok" and isinstance(row.get("prediction"), dict):
            key = (model, case_id)
            if key in predictions:
                duplicates.add(key)
            predictions[key] = row
    if duplicates:
        raise ValueError(f"Duplicate successful predictions, e.g. {sorted(duplicates)[:3]}")

    judgments: dict[tuple[str, str], dict[str, Any]] = {}
    for row in load_jsonl_collection(args.judgments.resolve()):
        if row.get("judge_model") != args.judge_model or row.get("status") != "ok":
            continue
        model = str(row.get("evaluated_model"))
        if model_filter is not None and model not in model_filter:
            continue
        key = (model, str(row.get("case_id")))
        if key in judgments:
            raise ValueError(f"Duplicate successful judgment: {key}")
        judgments[key] = row

    eligible_case_ids = [
        case_id
        for case_id in pool
        if "all" in subsets or gold[case_id]["source_category"] in subsets
    ]
    if args.limit > 0:
        eligible_case_ids = eligible_case_ids[: args.limit]
    required = [
        (model, case_id) for model in all_models for case_id in eligible_case_ids
    ]
    missing_predictions = [key for key in required if key not in predictions]
    missing_judgments = [
        key for key in required if key in predictions and key not in judgments
    ]
    if not args.allow_incomplete and (missing_predictions or missing_judgments):
        preview = (missing_predictions + missing_judgments)[:5]
        raise ValueError(
            f"Incomplete run: {len(missing_predictions)} missing predictions and "
            f"{len(missing_judgments)} missing judgments (e.g. {preview})."
        )

    details: list[dict[str, Any]] = []
    for model in all_models:
        for case_id in eligible_case_ids:
            gold_row = gold[case_id]
            prediction_row = predictions.get((model, case_id))
            judgment_row = judgments.get((model, case_id))
            prediction = prediction_row.get("prediction", {}) if prediction_row else {}
            target_relevant = bool(judgment_row and judgment_row.get("target_relevant"))
            sufficient = bool(judgment_row and judgment_row.get("resolution_sufficient"))
            unsupported = bool(
                judgment_row and judgment_row.get("unsupported_assumption")
            )
            no_assumption = bool(judgment_row and not unsupported)
            atomic = bool(judgment_row and judgment_row.get("atomic"))
            action_success = bool(target_relevant and sufficient and not unsupported)
            predicted_type = str(prediction.get("action_type") or "")
            details.append(
                {
                    "model": model,
                    "model_group": MODELS.get(model, {}).get(
                        "group", observed_groups.get(model, "unknown")
                    ),
                    "case_id": case_id,
                    "instance_id": gold_row["instance_id"],
                    "paper_id": gold_row["paper_id"],
                    "source_category": gold_row["source_category"],
                    "gold_level1": gold_row["level1"],
                    "gold_level2": gold_row["level2"],
                    "gold_action_type": gold_row["gold_action_type"],
                    "predicted_action_type": predicted_type,
                    "predicted_action": prediction.get("action", ""),
                    "predicted_expected_information": prediction.get(
                        "expected_information", ""
                    ),
                    "prediction_available": prediction_row is not None,
                    "judgment_available": judgment_row is not None,
                    "target_relevant": target_relevant,
                    "resolution_sufficient": sufficient,
                    "unsupported_assumption": unsupported,
                    "no_assumption": no_assumption,
                    "atomic": atomic,
                    "action_success": action_success,
                    "action_type_agreement": predicted_type == gold_row["gold_action_type"],
                    "judge_confidence": judgment_row.get("confidence", "")
                    if judgment_row
                    else "",
                    "judge_reason": judgment_row.get("reason", "")
                    if judgment_row
                    else "",
                }
            )

    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for model in all_models:
        model_rows = [row for row in details if row["model"] == model]
        for subset in subsets:
            subset_rows = (
                model_rows
                if subset == "all"
                else [row for row in model_rows if row["source_category"] == subset]
            )
            if not subset_rows:
                continue
            summary_rows.append(
                {
                    "model": model,
                    "model_group": subset_rows[0]["model_group"],
                    "subset": subset,
                    **metrics_for(
                        subset_rows,
                        args.bootstrap_samples,
                        args.seed + sum(ord(char) for char in model + subset),
                    ),
                }
            )
            for level, label_key in (
                ("level1", "gold_level1"),
                ("level2", "gold_level2"),
            ):
                for label in sorted({str(row[label_key]) for row in subset_rows}):
                    class_rows = [row for row in subset_rows if row[label_key] == label]
                    per_class_rows.append(
                        {
                            "model": model,
                            "subset": subset,
                            "label_level": level,
                            "label": label,
                            "support": len(class_rows),
                            "cas": mean(
                                [float(row["action_success"]) for row in class_rows]
                            ),
                            "target_relevance": mean(
                                [float(row["target_relevant"]) for row in class_rows]
                            ),
                            "resolution_sufficiency": mean(
                                [float(row["resolution_sufficient"]) for row in class_rows]
                            ),
                            "no_assumption": mean(
                                [float(row["no_assumption"]) for row in class_rows]
                            ),
                        }
                    )

    group_order = {"baseline": 0, "frontier_proprietary": 1, "open_weight": 2}
    summary_rows.sort(
        key=lambda row: (
            group_order.get(str(row["model_group"]), 9),
            str(row["model"]),
            str(row["subset"]),
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "metrics.csv", summary_rows)
    write_markdown(args.output_dir / "metrics.md", summary_rows)
    write_csv(args.output_dir / "per_class_metrics.csv", per_class_rows)
    with (args.output_dir / "case_details.jsonl").open("w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Scored {len(all_models)} models -> {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
