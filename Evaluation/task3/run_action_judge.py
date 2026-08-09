#!/usr/bin/env python3
"""Judge Task 3 clarification actions against private resolution evidence."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from tqdm import tqdm

from task3_config import (
    DEFAULT_GOLD,
    DEFAULT_JUDGMENTS,
    DEFAULT_POOL,
    DEFAULT_PREDICTIONS_DIR,
    fetch_openrouter_catalog,
    load_jsonl,
    load_jsonl_collection,
    parse_json_object,
)


WRITE_LOCK = threading.Lock()
ALLOWED_CONFIDENCE = {"high", "medium", "low"}

JUDGE_SYSTEM_PROMPT = """
You are a strict evaluator of clarification actions for underspecified
scientific-method descriptions. Evaluate whether the proposed action would
obtain the information required to resolve the supplied annotated defect.

Do not reward lexical similarity to the reference action. A differently worded
question or a different action type may be fully correct. Do not evaluate
whether the target defect itself is valid; it is an adjudicated benchmark input.
Return strict JSON only.
""".strip()

JUDGE_SCHEMA: dict[str, Any] = {
    "name": "ideaambig_task3_action_judgment",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "target_relevant",
            "resolution_sufficient",
            "unsupported_assumption",
            "atomic",
            "confidence",
            "reason",
        ],
        "properties": {
            "target_relevant": {"type": "boolean"},
            "resolution_sufficient": {"type": "boolean"},
            "unsupported_assumption": {"type": "boolean"},
            "atomic": {"type": "boolean"},
            "confidence": {"type": "string", "enum": sorted(ALLOWED_CONFIDENCE)},
            "reason": {"type": "string"},
        },
    },
}


def text(value: Any, default: str = "[not provided]") -> str:
    value = str(value or "").strip()
    return value or default


def build_judge_prompt(
    specification: str,
    target_defect: str,
    gold: dict[str, Any],
    prediction: dict[str, Any],
) -> str:
    solution_evidence = gold.get("solution_evidence")
    if not isinstance(solution_evidence, dict):
        solution_evidence = {}
    evidence_text = json.dumps(
        {
            key: value
            for key, value in solution_evidence.items()
            if str(value or "").strip()
        },
        ensure_ascii=False,
        indent=2,
    ) or "[not provided]"
    return f"""Evaluate the candidate clarification action using the rubric below.

COUNTERFACTUAL SUFFICIENCY TEST
Assume a cooperative, knowledgeable respondent or artifact inspector answers
only what the action explicitly requests. Would the resulting information be
enough to determine the hidden gold resolution? A generic request for more
details is not sufficient merely because a maximally helpful respondent could
volunteer the entire answer. A candidate is also not sufficient if it is not an
actual clarification question or concrete evidence-seeking action, such as a
diagnosis, repair instruction, implementation guess, or vague request to
"investigate" without stating what information should be obtained.

DIMENSIONS
1. target_relevant: The action specifically concerns the annotated target
   defect, not another plausible issue in the specification.
2. resolution_sufficient: The candidate is a real clarification question or
   concrete evidence-seeking action, and a direct answer or inspection result
   would determine every implementation-critical part of the hidden resolution.
   It need not use the same wording or action type as the reference.
3. unsupported_assumption: The candidate asserts, recommends, or presupposes an
   unsupported value for the missing detail. Merely asking whether one of
   several explicit alternatives applies is not an assertion.
4. atomic: The candidate contains one bounded action addressing the one target
   defect. Multiple tightly coupled subquestions needed to resolve the same
   atomic defect may count as atomic; unrelated requests do not.

Important rules:
- The reference action is one valid realization, not the only valid wording.
- Gold action-type disagreement is not itself an error.
- Fluency and verbosity are not evaluation criteria.
- Do not let facts in the hidden resolution excuse an unsupported assertion in
  the candidate; the evaluated model never saw that resolution.

Underspecified specification:
<specification>
{text(specification)}
</specification>

Annotated target defect shown to the evaluated model:
<target_defect>
{text(target_defect)}
</target_defect>

Private hidden resolution:
<hidden_resolution>
{text(gold.get('hidden_resolution'))}
</hidden_resolution>

Private reference action (semantic aid only):
<reference_action>
Action type: {text(gold.get('gold_action_type'))}
Action: {text(gold.get('gold_action'))}
Evidence to seek: {text(gold.get('gold_evidence_to_seek'))}
</reference_action>

Private solution evidence:
<solution_evidence>
{evidence_text}
</solution_evidence>

Candidate:
<candidate>
Action type: {text(prediction.get('action_type'))}
Action: {text(prediction.get('action'))}
Expected information: {text(prediction.get('expected_information'))}
</candidate>

Return only valid JSON:
{{
  "target_relevant": true,
  "resolution_sufficient": true,
  "unsupported_assumption": false,
  "atomic": true,
  "confidence": "high|medium|low",
  "reason": "brief evidence-based explanation"
}}"""


def validate_judgment(value: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "target_relevant",
        "resolution_sufficient",
        "unsupported_assumption",
        "atomic",
    ):
        if not isinstance(value.get(key), bool):
            raise ValueError(f"{key} must be boolean")
        result[key] = value[key]
    confidence = str(value.get("confidence") or "").strip()
    if confidence not in ALLOWED_CONFIDENCE:
        raise ValueError(f"Invalid confidence: {confidence!r}")
    reason = str(value.get("reason") or "").strip()
    if not reason:
        raise ValueError("reason must be non-empty")
    result["confidence"] = confidence
    result["reason"] = reason
    result["action_success"] = bool(
        result["target_relevant"]
        and result["resolution_sufficient"]
        and not result["unsupported_assumption"]
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--judge-model", default="anthropic/claude-opus-4.8")
    parser.add_argument("--models", default="all")
    parser.add_argument("--subset", choices=("all", "real", "synthetic"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--reasoning-effort", choices=("none", "low", "medium", "high"), default="low"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-catalog-check", action="store_true")
    return parser.parse_args()


def parse_model_filter(spec: str) -> set[str] | None:
    if spec == "all":
        return None
    values = {value.strip() for value in spec.split(",") if value.strip()}
    if not values:
        raise ValueError("No evaluated models selected")
    return values


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    with WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()


def post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def processed_keys(path: Path, judge_model: str) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {
        (str(row.get("evaluated_model")), str(row.get("case_id")))
        for row in load_jsonl(path)
        if row.get("status") == "ok"
        and row.get("judge_model") == judge_model
        and isinstance(row.get("action_success"), bool)
    }


def judge_one(
    case: dict[str, Any],
    gold: dict[str, Any],
    prediction_row: dict[str, Any],
    args: argparse.Namespace,
    supported: set[str],
    api_key: str,
    base_url: str,
) -> dict[str, Any]:
    evaluated_model = str(prediction_row["model"])
    prompt = build_judge_prompt(
        str(case["input_text"]),
        str(case["target_defect"]),
        gold,
        prediction_row["prediction"],
    )
    payload: dict[str, Any] = {
        "model": args.judge_model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    if "temperature" in supported:
        payload["temperature"] = 0
    if "max_tokens" in supported:
        payload["max_tokens"] = args.max_tokens
    elif "max_completion_tokens" in supported:
        payload["max_completion_tokens"] = args.max_tokens
    if "seed" in supported:
        payload["seed"] = args.seed
    if "reasoning" in supported and args.reasoning_effort != "none":
        payload["reasoning"] = {"effort": args.reasoning_effort, "exclude": True}
    if "response_format" in supported or "structured_outputs" in supported:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": JUDGE_SCHEMA,
        }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "IdeaAmbig-Task3-Judge/1.0",
        "X-Title": "IdeaAmbig Task 3 Judge",
    }
    last_error = "unknown error"
    last_output = ""
    for attempt in range(1, args.max_retries + 1):
        try:
            response = post_json(
                f"{base_url.rstrip('/')}/chat/completions", headers, payload, args.timeout
            )
            choice = (response.get("choices") or [])[0]
            last_output = str((choice.get("message") or {}).get("content") or "")
            judgment = validate_judgment(parse_json_object(last_output))
            return {
                "status": "ok",
                "case_id": case["case_id"],
                "evaluated_model": evaluated_model,
                "judge_model": args.judge_model,
                **judgment,
            }
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:2000]
            last_error = f"HTTP {exc.code}: {body}"
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < args.max_retries:
            time.sleep(min(30.0, 2 ** (attempt - 1) + random.random()))
    row: dict[str, Any] = {
        "status": "error",
        "case_id": case["case_id"],
        "evaluated_model": evaluated_model,
        "judge_model": args.judge_model,
        "error": last_error,
    }
    if last_output:
        row["raw_output"] = last_output[:4000]
    return row


def main() -> None:
    args = parse_args()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    pool = {str(row["case_id"]): row for row in load_jsonl(args.pool.resolve())}
    gold = {str(row["case_id"]): row for row in load_jsonl(args.gold.resolve())}
    if set(pool) != set(gold):
        raise ValueError("Pool and gold case_id sets do not match")
    model_filter = parse_model_filter(args.models)
    predictions = [
        row
        for row in load_jsonl_collection(args.predictions.resolve())
        if row.get("status") == "ok"
        and isinstance(row.get("prediction"), dict)
        and (model_filter is None or str(row.get("model")) in model_filter)
        and str(row.get("case_id")) in pool
    ]
    if args.subset != "all":
        predictions = [
            row
            for row in predictions
            if gold[str(row["case_id"])]["source_category"] == args.subset
        ]
    eligible_case_ids = [
        case_id
        for case_id in pool
        if args.subset == "all" or gold[case_id]["source_category"] == args.subset
    ]
    if args.limit > 0:
        eligible_case_ids = eligible_case_ids[: args.limit]
    eligible = set(eligible_case_ids)
    predictions = [row for row in predictions if str(row["case_id"]) in eligible]

    duplicate_keys: set[tuple[str, str]] = set()
    seen: set[tuple[str, str]] = set()
    for row in predictions:
        key = (str(row["model"]), str(row["case_id"]))
        if key in seen:
            duplicate_keys.add(key)
        seen.add(key)
    if duplicate_keys:
        raise ValueError(f"Duplicate successful predictions, e.g. {sorted(duplicate_keys)[:3]}")

    if args.overwrite and args.output.exists():
        args.output.unlink()
    done = processed_keys(args.output, args.judge_model)
    predictions = [
        row
        for row in predictions
        if (str(row["model"]), str(row["case_id"])) not in done
    ]
    if not predictions:
        print("No pending prediction judgments.")
        return

    supported: set[str] = set()
    if not args.skip_catalog_check:
        catalog = fetch_openrouter_catalog()
        if args.judge_model not in catalog:
            raise ValueError(f"Judge model absent from OpenRouter catalog: {args.judge_model}")
        supported = set(catalog[args.judge_model].get("supported_parameters") or [])

    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                judge_one,
                pool[str(row["case_id"])],
                gold[str(row["case_id"])],
                row,
                args,
                supported,
                api_key,
                base_url,
            ): (row["model"], row["case_id"])
            for row in predictions
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc="Judge"
        ):
            row = future.result()
            if row.get("status") == "ok":
                append_jsonl(args.output, row)
            else:
                failures += 1
                model, case_id = futures[future]
                tqdm.write(
                    f"ERROR judge {model}/{case_id}: "
                    f"{row.get('error', 'unknown error')}"
                )
    print(
        f"Completed {len(predictions)} judgments with {failures} failures -> "
        f"{args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
