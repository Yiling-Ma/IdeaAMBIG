#!/usr/bin/env python3
"""Score Track 1 localization and taxonomy-aware recall."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from track1_config import (
    DEFAULT_GOLD,
    DEFAULT_JUDGMENTS,
    DEFAULT_POOL,
    DEFAULT_PREDICTIONS,
    DEFAULT_SCORES_DIR,
    LEVEL2_TO_LEVEL1,
    MODELS,
    load_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SCORES_DIR)
    parser.add_argument("--judge-model", default="anthropic/claude-opus-4.8")
    parser.add_argument("--subsets", default="all,real,synthetic")
    parser.add_argument("--models", default="all", help="all or comma-separated model IDs to score")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Debug only: score the first N eligible pool cases. Use with one subset, "
            "matching the same --limit used by the prediction runner."
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Count missing predictions/judgments as misses; omit for final benchmark runs.",
    )
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def class_macro(rows: list[dict[str, Any]], label_key: str, hit_key: str) -> float:
    by_class: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_class[str(row[label_key])].append(float(row[hit_key]))
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
    loc = lambda rs: mean([float(row["semantic_match"]) for row in rs])
    label_l1 = lambda rs: mean([float(row["level1_correct"]) for row in rs])
    label_l2 = lambda rs: mean(
        [float(row["level1_correct"] and row["level2_correct"]) for row in rs]
    )
    l1 = lambda rs: mean([float(row["l1_aware_hit"]) for row in rs])
    l2 = lambda rs: mean([float(row["l2_aware_hit"]) for row in rs])
    macro_label_l1 = lambda rs: class_macro(rs, "gold_level1", "level1_correct")
    macro_label_l2 = lambda rs: class_macro(rs, "gold_level2", "level2_correct")
    macro_l1 = lambda rs: class_macro(rs, "gold_level1", "l1_aware_hit")
    macro_l2 = lambda rs: class_macro(rs, "gold_level2", "l2_aware_hit")
    matched = [row for row in rows if row["semantic_match"]]
    result: dict[str, Any] = {
        "n_cases": len(rows),
        "n_papers": len({row["paper_id"] for row in rows}),
        "prediction_coverage": mean([float(row["prediction_available"]) for row in rows]),
        "judgment_coverage": mean([float(row["judgment_available"]) for row in rows]),
        "label_l1_accuracy": label_l1(rows),
        "label_l2_accuracy": label_l2(rows),
        "macro_label_l1_accuracy": macro_label_l1(rows),
        "macro_label_l2_accuracy": macro_label_l2(rows),
        "localization_recall": loc(rows),
        "l1_aware_recall": l1(rows),
        "l2_aware_recall": l2(rows),
        "macro_l1_aware_recall": macro_l1(rows),
        "macro_l2_aware_recall": macro_l2(rows),
        "l1_accuracy_given_localized": mean(
            [float(row["level1_correct"]) for row in matched]
        ),
        "l2_accuracy_given_localized": mean(
            [float(row["level2_correct"] and row["level1_correct"]) for row in matched]
        ),
        "taxonomy_consistency_rate": mean(
            [float(row["taxonomy_consistent"]) for row in rows if row["prediction_available"]]
        ),
    }
    ci_metrics: dict[str, Callable[[list[dict[str, Any]]], float]] = {
        "label_l2_accuracy": label_l2,
        "localization_recall": loc,
        "l1_aware_recall": l1,
        "l2_aware_recall": l2,
        "macro_l2_aware_recall": macro_l2,
    }
    for offset, (name, function) in enumerate(ci_metrics.items()):
        low, high = clustered_bootstrap_ci(
            rows, function, bootstrap_samples, seed + 1009 * offset
        )
        result[f"{name}_ci95_low"] = low
        result[f"{name}_ci95_high"] = high
    return result


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


def parse_model_filter(spec: str) -> set[str] | None:
    if spec == "all":
        return None
    models = {value.strip() for value in spec.split(",") if value.strip()}
    if not models:
        raise ValueError("--models must be 'all' or a comma-separated list of model IDs")
    return models


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "model",
        "subset",
        "label_l1_accuracy",
        "label_l2_accuracy",
        "localization_recall",
        "l1_aware_recall",
        "l2_aware_recall",
        "macro_l2_aware_recall",
        "prediction_coverage",
    ]
    labels = [
        "Model", "Subset", "Label-L1 Acc ↑", "Label-L2 Acc ↑", "Loc-Acc ↑",
        "L1-aware R ↑", "L2-aware R ↑", "Macro DRR ↑", "Coverage ↑",
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
    for label, path in (
        ("prediction", args.predictions),
        ("semantic judgment", args.judgments),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"{label.capitalize()} file not found: {path.resolve()}. "
                "Run the preceding pipeline stage or pass the corresponding path option."
            )
    subsets = [value.strip() for value in args.subsets.split(",") if value.strip()]
    invalid = sorted(set(subsets) - {"all", "real", "synthetic"})
    if invalid or not subsets:
        raise ValueError("Invalid --subsets: " + ", ".join(invalid))
    model_filter = parse_model_filter(args.models)
    pool = {str(row["case_id"]): row for row in load_jsonl(args.pool)}
    gold = {str(row["case_id"]): row for row in load_jsonl(args.gold)}
    if set(pool) != set(gold):
        raise ValueError("Pool and gold case_id sets do not match")

    raw_predictions = load_jsonl(args.predictions)
    version_rows = [
        row
        for row in raw_predictions
        if model_filter is None or str(row.get("model")) in model_filter
    ]
    all_models = sorted({str(row.get("model")) for row in version_rows if row.get("model")})
    if model_filter is not None:
        missing_selected_models = sorted(model_filter - set(all_models))
        if missing_selected_models:
            raise ValueError(
                "Selected models not found in prediction file: "
                f"{missing_selected_models}"
            )
    predictions: dict[tuple[str, str], dict[str, Any]] = {}
    observed_groups: dict[str, str] = {}
    duplicate_predictions: set[tuple[str, str]] = set()
    for row in version_rows:
        model = str(row.get("model"))
        case_id = str(row.get("case_id"))
        if row.get("model_group"):
            observed_groups[model] = str(row["model_group"])
        if row.get("status") == "ok" and isinstance(row.get("prediction"), dict):
            key = (model, case_id)
            if key in predictions:
                duplicate_predictions.add(key)
            predictions[key] = row
    if duplicate_predictions:
        preview = sorted(duplicate_predictions)[:5]
        raise ValueError(
            "Duplicate successful predictions found for the same model/case_id. "
            f"Examples: {preview}. Remove duplicates before scoring."
        )
    if not all_models:
        raise ValueError("No model rows found in predictions")

    judgments: dict[tuple[str, str], dict[str, Any]] = {}
    for row in load_jsonl(args.judgments):
        if row.get("judge_model") != args.judge_model:
            continue
        if model_filter is not None and str(row.get("evaluated_model")) not in model_filter:
            continue
        if row.get("status") == "ok" and isinstance(row.get("semantic_match"), bool):
            judgments[(str(row.get("evaluated_model")), str(row.get("case_id")))] = row

    eligible_case_ids = [
        case_id
        for case_id in pool
        if "all" in subsets or gold[case_id]["source_category"] in subsets
    ]
    if args.limit > 0:
        eligible_case_ids = eligible_case_ids[: args.limit]
    eligible_case_id_set = set(eligible_case_ids)
    required = [
        (model, case_id)
        for model in all_models
        for case_id in eligible_case_ids
    ]
    missing_predictions = [key for key in required if key not in predictions]
    missing_judgments = [
        key for key in required if key in predictions and key not in judgments
    ]
    if not args.allow_incomplete and (missing_predictions or missing_judgments):
        preview = (missing_predictions + missing_judgments)[:5]
        raise ValueError(
            f"Incomplete run: {len(missing_predictions)} missing/invalid predictions and "
            f"{len(missing_judgments)} missing judgments (e.g. {preview}). "
            "Resume runners or use --allow-incomplete only for debugging."
        )

    details: list[dict[str, Any]] = []
    for model in all_models:
        for case_id, gold_row in gold.items():
            if case_id not in eligible_case_id_set:
                continue
            prediction_row = predictions.get((model, case_id))
            judgment_row = judgments.get((model, case_id))
            prediction = prediction_row.get("prediction", {}) if prediction_row else {}
            predicted_level1 = str(prediction.get("level1") or "")
            predicted_level2 = str(prediction.get("level2") or "")
            semantic_match = bool(judgment_row and judgment_row["semantic_match"])
            level1_correct = predicted_level1 == gold_row["level1"]
            level2_correct = predicted_level2 == gold_row["level2"]
            taxonomy_consistent = (
                predicted_level2 in LEVEL2_TO_LEVEL1
                and LEVEL2_TO_LEVEL1[predicted_level2] == predicted_level1
            )
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
                    "predicted_description": prediction.get("description", ""),
                    "predicted_level1": predicted_level1,
                    "predicted_level2": predicted_level2,
                    "prediction_available": prediction_row is not None,
                    "judgment_available": judgment_row is not None,
                    "semantic_match": semantic_match,
                    "level1_correct": level1_correct,
                    "level2_correct": level2_correct,
                    "taxonomy_consistent": taxonomy_consistent,
                    "l1_aware_hit": semantic_match and level1_correct,
                    "l2_aware_hit": semantic_match and level1_correct and level2_correct,
                    "judge_reason": judgment_row.get("reason", "") if judgment_row else "",
                    "judge_relationship": judgment_row.get("relationship", "")
                    if judgment_row
                    else "",
                    "judge_confidence": judgment_row.get("confidence", "")
                    if judgment_row
                    else "",
                }
            )

    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for model in all_models:
        model_rows = [row for row in details if row["model"] == model]
        for subset in subsets:
            subset_rows = model_rows if subset == "all" else [
                row for row in model_rows if row["source_category"] == subset
            ]
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
            for label_level, label_key, hit_key in (
                ("level1", "gold_level1", "l1_aware_hit"),
                ("level2", "gold_level2", "l2_aware_hit"),
            ):
                labels = sorted({str(row[label_key]) for row in subset_rows})
                for label in labels:
                    class_rows = [row for row in subset_rows if row[label_key] == label]
                    per_class_rows.append(
                        {
                            "model": model,
                            "subset": subset,
                            "label_level": label_level,
                            "label": label,
                            "support": len(class_rows),
                            "aware_recall": mean([float(row[hit_key]) for row in class_rows]),
                            "localization_recall": mean(
                                [float(row["semantic_match"]) for row in class_rows]
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
