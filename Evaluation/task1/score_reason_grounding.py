#!/usr/bin/env python3
"""Score Task 1 reason grounding with an LLM judge."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from task1_config import load_jsonl, load_jsonl_collection


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS = Path(__file__).resolve().parent / "outputs" / "predictions"
DEFAULT_HUMAN_GOLD = ROOT_DIR / "human_label" / "evaluation_gold.jsonl"
DEFAULT_REAL_BENCH = ROOT_DIR / "Real_bench" / "real_benchmark_instance.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "reason_grounding"
WRITE_LOCK = threading.Lock()


SYSTEM_PROMPT = """You are a rigorous benchmark judge for codification-readiness reasoning.
Evaluate whether a model's short reason is grounded in the benchmark gold rationale.
Judge only against the provided gold fields. Do not use external knowledge.
Return valid JSON only."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--gold", type=Path, default=DEFAULT_HUMAN_GOLD)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_REAL_BENCH)
    parser.add_argument("--judge-model", default="anthropic/claude-opus-4.8")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def normalize_model_group(model: str, row: dict[str, Any]) -> str:
    if row.get("model_group"):
        return str(row["model_group"])
    if model.startswith("baseline/"):
        return "baseline"
    return "unknown"


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


def build_reason_grounding_gold(task_gold_row: dict[str, Any], source_row: dict[str, Any]) -> dict[str, Any]:
    existing = task_gold_row.get("reason_grounding_gold")
    if isinstance(existing, dict):
        return existing
    codification = source_row.get("codification_readiness") or {}
    if str(task_gold_row.get("variant")) == "incomplete":
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


def hydrated_gold_map(gold_path: Path, benchmark_path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    gold_rows = load_jsonl(gold_path)
    missing_rows = [
        row
        for row in gold_rows
        if not isinstance(row.get("reason_grounding_gold"), dict)
    ]
    benchmark_by_id: dict[str, dict[str, Any]] = {}
    if missing_rows:
        benchmark_by_id = {str(row["id"]): row for row in load_jsonl(benchmark_path)}
    for row in gold_rows:
        enriched = dict(row)
        if not isinstance(enriched.get("reason_grounding_gold"), dict):
            instance_id = str(row["instance_id"])
            source_row = benchmark_by_id.get(instance_id)
            if source_row is None:
                raise KeyError(f"Instance ID absent from benchmark file: {instance_id}")
            enriched["reason_grounding_gold"] = build_reason_grounding_gold(row, source_row)
        result[str(row["case_id"])] = enriched
    return result


def extract_json_object(text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    if not text:
        raise ValueError("Judge response was empty")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Judge response was not valid JSON: {text[:400]!r}") from None
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Judge response JSON must be an object")
    return value


def validate_subscore(value: Any, field: str) -> float:
    try:
        score = float(value)
    except Exception as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    allowed = {0.0, 0.5, 1.0}
    if score not in allowed:
        raise ValueError(f"{field} must be one of {sorted(allowed)}, received {score}")
    return score


def judge_prompt(record: dict[str, Any]) -> str:
    expected_label = "READY" if bool(record["gold_expected_ready"]) else "NOT_READY"
    predicted_label = "READY" if bool(record["predicted_ready"]) else "NOT_READY"
    gold_payload = record["reason_grounding_gold"]
    prompt = {
        "task": "Evaluate the grounding quality of a model's Task 1 readiness reason.",
        "scoring_rubric": {
            "label_support": {
                "question": (
                    "Does the reason support the model's predicted label in a way that is "
                    "consistent with the gold label and gold readiness rationale?"
                ),
                "allowed_scores": [0, 0.5, 1],
            },
            "blocker_match": {
                "question": (
                    "Does the reason correctly identify the gold implementation-critical "
                    "defect or correctly reflect that no such blocker remains?"
                ),
                "allowed_scores": [0, 0.5, 1],
            },
            "faithfulness": {
                "question": (
                    "Is the reason faithful to the provided gold rationale, without "
                    "hallucinating unsupported blockers or irrelevant details?"
                ),
                "allowed_scores": [0, 0.5, 1],
            },
        },
        "aggregation_rule": (
            "Final RGS = 0.7 * label_support + 0.2 * blocker_match + 0.1 * faithfulness. "
            "Do not compute the final score yourself; only return the three subscores."
        ),
        "important_rule_for_ready_cases": (
            "For READY cases, do not require the model to restate the full reference "
            "specification. The reason is sufficient if it correctly states that no "
            "implementation-critical blocker remains and does not hallucinate one."
        ),
        "gold": {
            "expected_label": expected_label,
            "defects": gold_payload.get("defects", []),
            "blocking_missing_specs": gold_payload.get("blocking_missing_specs", []),
            "readiness_reason": gold_payload.get("readiness_reason", ""),
        },
        "model_output": {
            "predicted_label": predicted_label,
            "reason": record["reason_text"],
        },
        "return_format": {
            "label_support": 0,
            "blocker_match": 0,
            "faithfulness": 0,
            "explanation": "one concise paragraph",
        },
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def judge_one(
    record: dict[str, Any],
    args: argparse.Namespace,
    api_key: str,
    base_url: str,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "IdeaBench-Task1-Reason-Grounding/1.0",
        "X-Title": "IdeaBench Task 1 Reason Grounding",
    }
    payload = {
        "model": args.judge_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": judge_prompt(record)},
        ],
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }

    last_error = "unknown error"
    for attempt in range(1, args.max_retries + 1):
        try:
            response = post_json(
                f"{base_url.rstrip('/')}/chat/completions",
                headers,
                payload,
                args.timeout,
            )
            choice = (response.get("choices") or [])[0]
            message = choice.get("message") or {}
            raw = str(message.get("content") or "")
            judged = extract_json_object(raw)
            label_support = validate_subscore(judged.get("label_support"), "label_support")
            blocker_match = validate_subscore(judged.get("blocker_match"), "blocker_match")
            faithfulness = validate_subscore(judged.get("faithfulness"), "faithfulness")
            reason_grounding_score = (
                0.7 * label_support + 0.2 * blocker_match + 0.1 * faithfulness
            )
            return {
                **record,
                "judge_model": args.judge_model,
                "label_support": label_support,
                "blocker_match": blocker_match,
                "faithfulness": faithfulness,
                "reason_grounding_score": reason_grounding_score,
                "judge_explanation": str(judged.get("explanation") or "").strip(),
                "judge_raw_output": raw,
            }
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:2000]
            last_error = f"HTTP {exc.code}: {body}"
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < args.max_retries:
            delay = min(30.0, (2 ** (attempt - 1)) + random.random())
            time.sleep(delay)
    return {
        **record,
        "judge_model": args.judge_model,
        "judge_error": last_error,
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    with WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()


def load_completed_scores(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Return the latest successful (model, case_id) -> scored row mapping."""
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return completed
    for row in load_jsonl(path):
        if "reason_grounding_score" not in row:
            continue
        model = str(row.get("model") or "")
        case_id = str(row.get("case_id") or "")
        if not model or not case_id:
            continue
        completed[(model, case_id)] = row
    return completed


def summary_rows(scored_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        key = (str(row["model"]), str(row["model_group"]), str(row.get("subset", "all")))
        buckets[key].append(row)
    summary: list[dict[str, Any]] = []
    for (model, model_group, subset), rows in sorted(buckets.items()):
        summary.append(
            {
                "model": model,
                "model_group": model_group,
                "subset": subset,
                "n_cases": len(rows),
                "reason_grounding_score": sum(float(r["reason_grounding_score"]) for r in rows) / len(rows),
                "label_support": sum(float(r["label_support"]) for r in rows) / len(rows),
                "blocker_match": sum(float(r["blocker_match"]) for r in rows) / len(rows),
                "faithfulness": sum(float(r["faithfulness"]) for r in rows) / len(rows),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = [
        "model",
        "subset",
        "reason_grounding_score",
        "label_support",
        "blocker_match",
        "faithfulness",
    ]
    labels = [
        "Model",
        "Subset",
        "RGS ↑",
        "Label ↑",
        "Blocker ↑",
        "Faithful ↑",
    ]
    lines = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join(["---"] * len(labels)) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def infer_subset(row: dict[str, Any]) -> str:
    source = row.get("source") or {}
    values: list[str] = []
    for key in ("source_type", "construction_method"):
        if source.get(key):
            values.append(str(source[key]))
    for key in ("source_types", "construction_methods"):
        if isinstance(source.get(key), list):
            values.extend(str(value) for value in source[key])
    lowered = " ".join(values).lower()
    if "synthetic" in lowered and "real_gap" not in lowered:
        return "synthetic"
    if "real_gap" in lowered and "synthetic" not in lowered:
        return "real"
    return "all"


def main() -> None:
    args = parse_args()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    gold_by_case = hydrated_gold_map(args.gold.resolve(), args.benchmark.resolve())
    predictions = load_jsonl_collection(args.predictions.resolve())

    jobs: list[dict[str, Any]] = []
    for row in predictions:
        if row.get("status") != "ok" or not isinstance(row.get("predicted_ready"), bool):
            continue
        reason_text = str(row.get("reason_text") or "").strip()
        if not reason_text:
            continue
        case_id = str(row["case_id"])
        gold = gold_by_case.get(case_id)
        if gold is None:
            continue
        jobs.append(
            {
                "case_id": case_id,
                "model": str(row["model"]),
                "model_group": normalize_model_group(str(row["model"]), row),
                "subset": infer_subset(gold),
                "predicted_ready": bool(row["predicted_ready"]),
                "gold_expected_ready": bool(gold["expected_ready"]),
                "reason_text": reason_text,
                "reason_grounding_gold": gold["reason_grounding_gold"],
            }
        )

    if not jobs:
        raise ValueError("No prediction rows with non-empty reason_text were found")

    per_case_path = args.output_dir / "per_case_reason_grounding.jsonl"
    if per_case_path.exists() and args.overwrite:
        per_case_path.unlink()

    completed = {} if args.overwrite else load_completed_scores(per_case_path)
    pending_jobs = [
        row for row in jobs if (str(row["model"]), str(row["case_id"])) not in completed
    ]
    scored_rows: list[dict[str, Any]] = list(completed.values())
    failed_rows: list[dict[str, Any]] = []

    print(
        json.dumps(
            {
                "cases_total": len(jobs),
                "cases_already_scored": len(completed),
                "cases_pending": len(pending_jobs),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    if pending_jobs:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(judge_one, row, args, api_key, base_url)
                for row in pending_jobs
            ]
            with tqdm(total=len(futures), desc="Reason grounding", unit="case") as progress:
                for future in concurrent.futures.as_completed(futures):
                    row = future.result()
                    append_jsonl(per_case_path, row)
                    if "reason_grounding_score" in row:
                        scored_rows.append(row)
                    else:
                        failed_rows.append(row)
                        tqdm.write(
                            f"ERROR {row.get('model')}/{row.get('case_id')}: "
                            f"{row.get('judge_error', 'unknown error')}"
                        )
                    progress.update(1)

    summary = summary_rows(scored_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "reason_grounding_summary.csv", summary)
    write_markdown(args.output_dir / "reason_grounding_summary.md", summary)
    (args.output_dir / "reason_grounding_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if failed_rows:
        write_jsonl(args.output_dir / "reason_grounding_failures.jsonl", failed_rows)

    print(
        json.dumps(
            {
                "judge_model": args.judge_model,
                "cases_submitted": len(jobs),
                "cases_resumed": len(completed),
                "cases_scored": len(scored_rows),
                "cases_failed": len(failed_rows),
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
