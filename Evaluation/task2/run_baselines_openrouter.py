#!/usr/bin/env python3
"""Run the frozen Track 1 taxonomy-guided zero-shot prompt via OpenRouter."""

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

from track1_config import (
    DEFAULT_GOLD,
    DEFAULT_POOL,
    DEFAULT_PREDICTIONS,
    PROMPT_VERSIONS,
    PREDICTION_JSON_SCHEMA,
    SYSTEM_PROMPT,
    build_user_prompt,
    fetch_openrouter_catalog,
    load_jsonl,
    model_group,
    parse_json_object,
    selected_models,
    validate_prediction,
)


WRITE_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument(
        "--models", default="all", help="all, frontier, open_weight, or comma-separated IDs"
    )
    parser.add_argument("--subset", choices=("all", "real", "synthetic"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--prompt-version",
        choices=PROMPT_VERSIONS,
        default="v3",
        help="Prompt version to run. Track 1 now keeps only the frozen v3 prompt.",
    )
    parser.add_argument(
        "--reasoning-effort", choices=("none", "low", "medium", "high"), default="low"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-catalog-check",
        action="store_true",
        help="Attempt configured IDs without checking the live OpenRouter catalog.",
    )
    return parser.parse_args()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    with WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()


def processed_keys(path: Path, prompt_version: str) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {
        (str(row.get("model")), str(row.get("case_id")))
        for row in load_jsonl(path)
        if row.get("status") == "ok"
        and isinstance(row.get("prediction"), dict)
        and str(row.get("prompt_version") or "v3") == prompt_version
    }


def post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def model_capabilities(
    models: list[str], skip_catalog_check: bool
) -> dict[str, set[str]]:
    if skip_catalog_check:
        return {model: set() for model in models}
    catalog = fetch_openrouter_catalog()
    missing = [model for model in models if model not in catalog]
    if missing:
        raise ValueError(
            "Configured model IDs absent from the current OpenRouter catalog: "
            + ", ".join(missing)
            + ". Use --skip-catalog-check only if a provider has given you access to these IDs."
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
            {
                "role": "user",
                "content": build_user_prompt(
                    str(case["input_text"]), prompt_version=args.prompt_version
                ),
            },
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
        "User-Agent": "IdeaAmbig-Task2-Track1/1.0",
        "X-Title": "IdeaAmbig Task 2 Track 1",
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
            return {
                "status": "ok",
                "case_id": case["case_id"],
                "model": model,
                "model_group": model_group(model),
                "prompt_version": args.prompt_version,
                "prediction": prediction,
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
        "prompt_version": args.prompt_version,
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
    cases = load_jsonl(args.pool.resolve())
    if args.subset != "all":
        gold = {str(row["case_id"]): row for row in load_jsonl(args.gold.resolve())}
        missing = [str(case["case_id"]) for case in cases if str(case["case_id"]) not in gold]
        if missing:
            raise ValueError(f"Pool case missing from gold: {missing[0]}")
        cases = [case for case in cases if gold[str(case["case_id"])]["source_category"] == args.subset]
    if args.limit > 0:
        cases = cases[: args.limit]
    if args.overwrite and args.output.exists():
        args.output.unlink()

    capabilities = model_capabilities(models, args.skip_catalog_check)
    done = processed_keys(args.output, args.prompt_version)
    jobs = [
        (case, model)
        for model in models
        for case in cases
        if (model, str(case["case_id"])) not in done
    ]
    if not jobs:
        print("No pending model/case pairs.")
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
        with tqdm(total=len(futures), desc="Task 2 Track 1", unit="request") as progress:
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                append_jsonl(args.output, row)
                failures += row.get("status") != "ok"
                progress.update(1)

    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "models": models,
                "prompt_version": args.prompt_version,
                "subset": args.subset,
                "cases_per_model": len(cases),
                "requests_attempted": len(jobs),
                "failures": failures,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
