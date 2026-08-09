#!/usr/bin/env python3
"""Score Task 1 predictions overall and separately on real/synthetic data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from task1_config import (
    DEFAULT_GOLD,
    DEFAULT_POOL,
    DEFAULT_PREDICTIONS_DIR,
    MODELS,
    load_jsonl,
    load_jsonl_collection,
    source_category,
)


RANDOM_BASELINE_PATTERN = re.compile(r"^baseline/random_empirical_run_\d+$")
SUMMARY_METRIC_COLUMNS = [
    "macro_f1",
    "mcc",
    "unsafe_pass_rate",
    "over_flag_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "scores",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Score available valid predictions; final benchmark runs should omit this flag.",
    )
    parser.add_argument(
        "--subsets",
        default="all,real,synthetic",
        help="Comma-separated subset reports chosen from: all,real,synthetic",
    )
    return parser.parse_args()


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def binary_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # NOT_READY is the positive class.
    tp = sum(not row["gold_ready"] and not row["pred_ready"] for row in rows)
    fn = sum(not row["gold_ready"] and row["pred_ready"] for row in rows)
    fp = sum(row["gold_ready"] and not row["pred_ready"] for row in rows)
    tn = sum(row["gold_ready"] and row["pred_ready"] for row in rows)
    f1_not_ready = safe_div(2 * tp, 2 * tp + fp + fn)
    f1_ready = safe_div(2 * tn, 2 * tn + fp + fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = safe_div(tp * tn - fp * fn, denominator)
    return {
        "n_cases": len(rows),
        "n_ready": tn + fp,
        "n_not_ready": tp + fn,
        "macro_f1": (f1_ready + f1_not_ready) / 2,
        "mcc": mcc,
        "unsafe_pass_rate": safe_div(fn, tp + fn),
        "over_flag_rate": safe_div(fp, tn + fp),
        "confusion_not_ready_positive": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
    }


def model_group_for(model: str, observed_groups: dict[str, str]) -> str:
    if model in MODELS:
        return str(MODELS[model].get("group", "unknown"))
    if model.startswith("baseline/"):
        return "baseline"
    return observed_groups.get(model, "unknown")


def flatten_summary(
    model: str,
    subset: str,
    metrics: dict[str, Any],
    observed_groups: dict[str, str],
) -> dict[str, Any]:
    return {
        "model": model,
        "model_group": model_group_for(model, observed_groups),
        "subset": subset,
        **{key: value for key, value in metrics.items() if not isinstance(value, dict)},
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def fmt_summary(value: Any) -> str:
    if isinstance(value, str):
        return value
    return fmt(value)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "model",
        "subset",
        "macro_f1",
        "mcc",
        "unsafe_pass_rate",
        "over_flag_rate",
    ]
    labels = [
        "Model", "Subset", "Macro-F1 ↑", "MCC ↑", "Unsafe Pass ↓",
        "Over-flag ↓",
    ]
    lines = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join(["---"] * len(labels)) + " |",
    ]
    group_labels = {
        "baseline": "Baselines",
        "frontier_proprietary": "Frontier proprietary models",
        "open_weight": "Open-weight models",
        "unknown": "Other models",
    }
    group_order = ["baseline", "frontier_proprietary", "open_weight", "unknown"]
    rows_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_group[str(row.get("model_group", "unknown"))].append(row)
    ordered_groups = group_order + sorted(set(rows_by_group) - set(group_order))
    for group in ordered_groups:
        group_rows = rows_by_group.get(group)
        if not group_rows:
            continue
        lines.append(
            "| "
            + fmt(group_labels.get(group, group))
            + " | "
            + " | ".join([""] * (len(columns) - 1))
            + " |"
        )
        for row in group_rows:
            lines.append("| " + " | ".join(fmt_summary(row.get(column)) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def row_sort_key(row: dict[str, Any]) -> tuple[int, str, str]:
    group_order = {
        "baseline": 0,
        "frontier_proprietary": 1,
        "open_weight": 2,
        "unknown": 3,
    }
    return (
        group_order.get(str(row.get("model_group", "unknown")), 99),
        str(row.get("model", "")),
        str(row.get("subset", "")),
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def aggregate_random_baseline_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    random_rows_by_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    retained: list[dict[str, Any]] = []
    for row in rows:
        model = str(row.get("model", ""))
        if RANDOM_BASELINE_PATTERN.match(model):
            random_rows_by_subset[str(row.get("subset", ""))].append(row)
        elif model != "baseline/random_empirical":
            retained.append(row)

    for subset, subset_rows in random_rows_by_subset.items():
        if not subset_rows:
            continue
        aggregate: dict[str, Any] = {
            "model": f"Random Guess (mean +/- std over {len(subset_rows)} runs)",
            "model_group": "baseline",
            "subset": subset,
            "n_cases": subset_rows[0].get("n_cases"),
            "n_ready": subset_rows[0].get("n_ready"),
            "n_not_ready": subset_rows[0].get("n_not_ready"),
            "prediction_coverage": subset_rows[0].get("prediction_coverage"),
        }
        for column in SUMMARY_METRIC_COLUMNS:
            values = [float(row[column]) for row in subset_rows]
            aggregate[column] = f"{mean(values):.4f} +/- {sample_std(values):.4f}"
            aggregate[f"{column}_mean"] = mean(values)
            aggregate[f"{column}_std"] = sample_std(values)
        retained.append(aggregate)
    retained.sort(key=row_sort_key)
    return retained


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = [
        "model",
        "model_group",
        "subset",
        "n_cases",
        "n_ready",
        "n_not_ready",
        "prediction_coverage",
    ]
    for column in SUMMARY_METRIC_COLUMNS:
        fields.extend([column, f"{column}_mean", f"{column}_std"])
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized = {field: row.get(field, "") for field in fields}
        for column in SUMMARY_METRIC_COLUMNS:
            if not normalized.get(f"{column}_mean") and isinstance(row.get(column), (int, float)):
                normalized[f"{column}_mean"] = row[column]
                normalized[f"{column}_std"] = 0.0
        normalized_rows.append(normalized)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(normalized_rows)


def main() -> None:
    args = parse_args()
    requested_subsets = [value.strip() for value in args.subsets.split(",") if value.strip()]
    invalid_subsets = sorted(set(requested_subsets) - {"all", "real", "synthetic"})
    if invalid_subsets or not requested_subsets:
        raise ValueError(
            "--subsets must contain one or more of all,real,synthetic; invalid: "
            + ", ".join(invalid_subsets)
        )
    pool = {str(row["case_id"]): row for row in load_jsonl(args.pool)}
    gold = {str(row["case_id"]): row for row in load_jsonl(args.gold)}
    if set(pool) != set(gold):
        raise ValueError("Pool and gold case_id sets do not match")

    raw_predictions = load_jsonl_collection(args.predictions)
    predictions: dict[tuple[str, str], dict[str, Any]] = {}
    configured_models = set()
    observed_model_groups: dict[str, str] = {}
    has_random_runs = any(
        RANDOM_BASELINE_PATTERN.match(str(row.get("model", ""))) for row in raw_predictions
    )
    for row in raw_predictions:
        model = str(row.get("model"))
        if has_random_runs and model == "baseline/random_empirical":
            continue
        case_id = str(row.get("case_id"))
        configured_models.add(model)
        if row.get("model_group"):
            observed_model_groups[model] = str(row["model_group"])
        if row.get("status") == "ok" and isinstance(row.get("predicted_ready"), bool):
            predictions[(model, case_id)] = row

    models = sorted(configured_models)
    if not models:
        raise ValueError("No model predictions found")
    required_case_ids = {
        case_id
        for case_id, gold_row in gold.items()
        if "all" in requested_subsets or source_category(gold_row) in requested_subsets
    }
    if not args.allow_incomplete:
        missing = [
            (model, case_id)
            for model in models
            for case_id in required_case_ids
            if (model, case_id) not in predictions
        ]
        if missing:
            preview = ", ".join(f"{model}/{case}" for model, case in missing[:5])
            raise ValueError(
                f"Missing or invalid predictions: {len(missing)} (e.g. {preview}). "
                "Re-run the same model/subset command (resume is automatic), or use "
                "--allow-incomplete only for debugging."
            )

    summaries: dict[str, dict[str, Any]] = {}
    flat_rows: list[dict[str, Any]] = []
    for model in models:
        joined: list[dict[str, Any]] = []
        for case_id, gold_row in gold.items():
            prediction = predictions.get((model, case_id))
            if not prediction:
                continue
            joined.append(
                {
                    "case_id": case_id,
                    "gold_ready": bool(gold_row["expected_ready"]),
                    "pred_ready": bool(prediction["predicted_ready"]),
                    "source_category": source_category(gold_row),
                }
            )
        model_summary: dict[str, Any] = {}
        for subset in requested_subsets:
            subset_rows = (
                joined if subset == "all" else [r for r in joined if r["source_category"] == subset]
            )
            metrics = binary_metrics(subset_rows)
            metrics["prediction_coverage"] = len(subset_rows) / sum(
                1
                for gold_row in gold.values()
                if subset == "all" or source_category(gold_row) == subset
            )
            model_summary[subset] = metrics
            flat_rows.append(flatten_summary(model, subset, metrics, observed_model_groups))
        summaries[model] = model_summary

    flat_rows.sort(key=row_sort_key)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = aggregate_random_baseline_rows(flat_rows)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "metrics.csv", flat_rows)
    write_markdown(args.output_dir / "metrics.md", flat_rows)
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary_csv(args.output_dir / "metrics_summary.csv", summary_rows)
    write_markdown(args.output_dir / "metrics_summary.md", summary_rows)
    print(f"Scored {len(models)} models -> {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
