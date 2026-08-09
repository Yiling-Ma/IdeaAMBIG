#!/usr/bin/env python3
"""Score downstream clarification utility judgments."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

from task3_config import TASK3_DIR, load_jsonl


DEFAULT_DIR = TASK3_DIR / "outputs" / "synth" / "downstream_clarification"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pct(value: float) -> float:
    return 100.0 * value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgments", type=Path, default=DEFAULT_DIR / "downstream_judgments.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR / "scores")
    parser.add_argument("--judge-model", default="anthropic/claude-opus-4.8")
    parser.add_argument("--generator-model", default="")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument(
        "--bootstrap-size",
        type=int,
        default=0,
        help="Paired bootstrap resample size; 0 uses the observed paired sample size.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def pipeline_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "ready_rate": pct(mean([float(row["ready"]) for row in rows])),
        "completeness": pct(
            mean([(float(row["completeness"]) - 1.0) / 4.0 for row in rows])
        ),
        "missing_detail_recovery": pct(
            mean([float(row["missing_detail_recovered"]) for row in rows])
        ),
        "unsupported_assumption": pct(
            mean([float(row["unsupported_assumption"]) for row in rows])
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def metric_value(row: dict[str, Any], metric: str) -> float:
    if metric == "ready_rate":
        return float(row["ready"])
    if metric == "completeness":
        return (float(row["completeness"]) - 1.0) / 4.0
    if metric == "missing_detail_recovery":
        return float(row["missing_detail_recovered"])
    if metric == "unsupported_assumption":
        return float(row["unsupported_assumption"])
    raise ValueError(f"Unknown metric: {metric}")


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def star(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def paired_bootstrap(
    rows: list[dict[str, Any]],
    samples: int,
    resample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), {})[str(row["pipeline"])] = row
    pairs = [
        value
        for value in by_case.values()
        if "direct" in value and "clarification_assisted" in value
    ]
    if not pairs:
        raise ValueError("No paired direct/clarification_assisted judgments found")
    if resample_size <= 0:
        resample_size = len(pairs)
    metrics = [
        ("ready_rate", "increase"),
        ("completeness", "increase"),
        ("missing_detail_recovery", "increase"),
        ("unsupported_assumption", "decrease"),
    ]
    rng = random.Random(seed)
    result: list[dict[str, Any]] = []
    for metric, direction in metrics:
        observed = mean(
            [
                metric_value(pair["clarification_assisted"], metric)
                - metric_value(pair["direct"], metric)
                for pair in pairs
            ]
        )
        deltas: list[float] = []
        for _ in range(samples):
            selected = [pairs[rng.randrange(len(pairs))] for _ in range(resample_size)]
            deltas.append(
                mean(
                    [
                        metric_value(pair["clarification_assisted"], metric)
                        - metric_value(pair["direct"], metric)
                        for pair in selected
                    ]
                )
            )
        deltas.sort()
        if direction == "increase":
            p_value = (sum(delta <= 0 for delta in deltas) + 1.0) / (samples + 1.0)
        else:
            p_value = (sum(delta >= 0 for delta in deltas) + 1.0) / (samples + 1.0)
        result.append(
            {
                "metric": metric,
                "direction": direction,
                "n_pairs": len(pairs),
                "bootstrap_size": resample_size,
                "observed_delta": pct(observed),
                "ci95_low": pct(percentile(deltas, 0.025)),
                "ci95_high": pct(percentile(deltas, 0.975)),
                "p_value": p_value,
                "significance": star(p_value),
            }
        )
    return result


def main() -> None:
    args = parse_args()
    rows = [
        row
        for row in load_jsonl(args.judgments)
        if row.get("status") == "ok"
        and row.get("judge_model") == args.judge_model
        and (not args.generator_model or row.get("generator_model") == args.generator_model)
    ]
    if not rows:
        raise ValueError("No matching downstream judgments found")

    pipelines = ["direct", "clarification_assisted"]
    summary_rows: list[dict[str, Any]] = []
    metrics_by_pipeline: dict[str, dict[str, Any]] = {}
    for pipeline in pipelines:
        subset = [row for row in rows if row.get("pipeline") == pipeline]
        metrics = pipeline_metrics(subset)
        metrics_by_pipeline[pipeline] = metrics
        summary_rows.append({"pipeline": pipeline, **metrics})

    direct = metrics_by_pipeline["direct"]
    assisted = metrics_by_pipeline["clarification_assisted"]
    summary_rows.append(
        {
            "pipeline": "relative_change",
            "n": min(direct["n"], assisted["n"]),
            "ready_rate": assisted["ready_rate"] - direct["ready_rate"],
            "completeness": assisted["completeness"] - direct["completeness"],
            "missing_detail_recovery": assisted["missing_detail_recovery"]
            - direct["missing_detail_recovery"],
            "unsupported_assumption": assisted["unsupported_assumption"]
            - direct["unsupported_assumption"],
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "metrics.csv", summary_rows)
    significance_rows = paired_bootstrap(
        rows,
        args.bootstrap_samples,
        args.bootstrap_size,
        args.seed,
    )
    (args.output_dir / "significance.json").write_text(
        json.dumps(significance_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "significance.csv", significance_rows)

    lines = [
        "| Pipeline | N | READY Rate ↑ | Completeness ↑ | Missing Detail Recovery ↑ | Unsupported Assumption ↓ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    display_name = {
        "direct": "Idea -> Specification",
        "clarification_assisted": "Idea -> Clarification -> Specification",
        "relative_change": "Relative Change",
    }
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    display_name.get(str(row["pipeline"]), str(row["pipeline"])),
                    str(row["n"]),
                    fmt(row["ready_rate"]),
                    fmt(row["completeness"]),
                    fmt(row["missing_detail_recovery"]),
                    fmt(row["unsupported_assumption"]),
                ]
            )
            + " |"
        )
    (args.output_dir / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sig_lines = [
        "| Metric | Direction | Observed Δ | 95% CI | p-value | Sig. |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in significance_rows:
        sig_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["metric"]),
                    str(row["direction"]),
                    f"{row['observed_delta']:.1f}",
                    f"[{row['ci95_low']:.1f}, {row['ci95_high']:.1f}]",
                    f"{row['p_value']:.4g}",
                    str(row["significance"]),
                ]
            )
            + " |"
        )
    (args.output_dir / "significance.md").write_text(
        "\n".join(sig_lines) + "\n",
        encoding="utf-8",
    )

    print(f"Scored downstream judgments -> {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
