#!/usr/bin/env python3
"""Run the Task 3 information-bottleneck End-to-End condition.

This uses the same output schema as Task 3, but the evaluated model sees only
the underspecified specification. The private target defect is still used later
by the unchanged Task 3 judge and scorer.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import threading
import time
import urllib.error
from pathlib import Path
from typing import Any

from tqdm import tqdm

from task3_config import (
    DEFAULT_GOLD,
    DEFAULT_POOL,
    PREDICTION_JSON_SCHEMA,
    SYSTEM_PROMPT,
    fetch_openrouter_catalog,
    load_jsonl,
    model_filename,
    model_group,
    parse_json_object,
    selected_models,
    validate_prediction,
)
from run_baselines_openrouter import post_json


WRITE_LOCK = threading.Lock()
PROMPT_VERSION = "information_bottleneck_e2e_v1"


def build_e2e_prompt(input_text: str) -> str:
    return f"""Task: Clarification Action Generation

You are given an underspecified research idea. First identify the most likely
implementation-critical specification gap, then generate exactly one atomic
action that would obtain the missing information needed to resolve it before
implementation.

Allowed action types:
- clarification_question: ask one precise, answerable question of an author or
  implementer who knows the intended method.
- evidence_seeking: specify one concrete, executable inspection of an available
  artifact such as paper text, source code, configuration, data documentation,
  or an experiment, and state what implementation fact should be checked.

Requirements:
1. Address one implementation-critical ambiguity in the supplied specification.
2. Request enough information to determine the implementation-critical choice.
3. Be specific. Generic requests such as "provide more details" are
   insufficient.
4. Do not state, guess, presuppose, or recommend a value for the missing detail.
   Asking about explicit alternatives is allowed when it does not assert one.
5. Do not bundle unrelated questions or actions.
6. In expected_information, describe the information to be obtained, not its
   unknown value or the answer you expect to be true.

Underspecified specification:
<specification>
{input_text}
</specification>

Return only valid JSON:
{{
  "action_type": "clarification_question|evidence_seeking",
  "action": "one atomic clarification question or evidence-seeking action",
  "expected_information": "the unknown implementation information this action should obtain"
}}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Evaluation/task3/outputs/information_bottleneck/predictions/end_to_end"),
    )
    parser.add_argument("--models", default="openai/gpt-5.6-sol")
    parser.add_argument("--subset", choices=("all", "real", "synthetic"), default="real")
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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    with WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()


def processed_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {
        (str(row.get("model")), str(row.get("case_id")))
        for row in load_jsonl(path)
        if row.get("status") == "ok"
        and isinstance(row.get("prediction"), dict)
        and str(row.get("prompt_version")) == PROMPT_VERSION
    }


def model_capabilities(models: list[str], skip_catalog_check: bool) -> dict[str, set[str]]:
    if skip_catalog_check:
        return {model: set() for model in models}
    catalog = fetch_openrouter_catalog()
    missing = [model for model in models if model not in catalog]
    if missing:
        raise ValueError(
            "Model IDs absent from the current OpenRouter catalog: "
            + ", ".join(missing)
            + ". Use --skip-catalog-check only when access is known."
        )
    return {
        model: set(catalog[model].get("supported_parameters") or []) for model in models
    }


def build_payload(
    model: str,
    case: dict[str, Any],
    supported: set[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_e2e_prompt(str(case["input_text"]))},
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
            "json_schema": PREDICTION_JSON_SCHEMA,
        }
    return payload


def evaluate_one(
    case: dict[str, Any],
    model: str,
    supported: set[str],
    args: argparse.Namespace,
    api_key: str,
    base_url: str,
) -> dict[str, Any]:
    payload = build_payload(model, case, supported, args)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "IdeaAmbig-Task3-InformationBottleneck/1.0",
        "X-Title": "IdeaAmbig Task 3 Information Bottleneck",
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
            prediction = validate_prediction(parse_json_object(last_output))
            usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
            return {
                "status": "ok",
                "case_id": case["case_id"],
                "model": model,
                "model_group": model_group(model),
                "prompt_version": PROMPT_VERSION,
                "condition": "end_to_end",
                "prediction": prediction,
                "usage": usage,
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
        "model": model,
        "model_group": model_group(model),
        "prompt_version": PROMPT_VERSION,
        "condition": "end_to_end",
        "error": last_error,
    }
    if last_output:
        row["raw_output"] = last_output[:4000]
    return row


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    models = selected_models(args.models)
    output_paths = {
        model: args.output_dir / args.subset / model_filename(model)
        for model in models
    }
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
    if args.overwrite:
        for path in output_paths.values():
            if path.exists():
                path.unlink()

    capabilities = model_capabilities(models, args.skip_catalog_check)
    done_by_model = {model: processed_keys(path) for model, path in output_paths.items()}
    jobs = [
        (case, model)
        for model in models
        for case in cases
        if (model, str(case["case_id"])) not in done_by_model[model]
    ]
    if not jobs:
        print("No pending model/case pairs.")
        for model, path in output_paths.items():
            print(f"{model} -> {path.resolve()}")
        return

    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                evaluate_one,
                case,
                model,
                capabilities[model],
                args,
                api_key,
                base_url,
            ): (case["case_id"], model)
            for case, model in jobs
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc="E2E Task 3"
        ):
            row = future.result()
            if row.get("status") == "ok":
                append_jsonl(output_paths[str(row["model"])], row)
            else:
                failures += 1
                case_id, model = futures[future]
                tqdm.write(f"ERROR {model}/{case_id}: {row.get('error', 'unknown error')}")
    print(f"Completed {len(jobs)} jobs with {failures} failures.")
    for model, path in output_paths.items():
        print(f"{model} -> {path.resolve()}")


if __name__ == "__main__":
    main()
