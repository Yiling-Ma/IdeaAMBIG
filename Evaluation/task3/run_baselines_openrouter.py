#!/usr/bin/env python3
"""Run Task 3 zero-shot baselines through OpenRouter."""

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
    DEFAULT_POOL,
    DEFAULT_PREDICTIONS_DIR,
    PREDICTION_JSON_SCHEMA,
    PROMPT_VERSIONS,
    SYSTEM_PROMPT,
    build_user_prompt,
    fetch_openrouter_catalog,
    load_jsonl,
    model_filename,
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
    parser.add_argument(
        "--output",
        type=Path,
        help="Explicit output file; allowed only when exactly one model is selected.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PREDICTIONS_DIR,
        help=(
            "Root directory for automatic per-model files. Files are written under "
            "<output-dir>/<prompt-version>/<subset>/<model>.jsonl."
        ),
    )
    parser.add_argument(
        "--models", default="all", help="all, frontier, open_weight, or comma-separated IDs"
    )
    parser.add_argument("--subset", choices=("all", "real", "synthetic"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--prompt-version", choices=PROMPT_VERSIONS, default="v1")
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


def processed_keys(path: Path, prompt_version: str) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {
        (str(row.get("model")), str(row.get("case_id")))
        for row in load_jsonl(path)
        if row.get("status") == "ok"
        and isinstance(row.get("prediction"), dict)
        and str(row.get("prompt_version") or "v1") == prompt_version
    }


def output_paths(
    models: list[str],
    explicit_output: Path | None,
    output_dir: Path,
    prompt_version: str,
    subset: str,
) -> dict[str, Path]:
    if explicit_output is not None:
        if len(models) != 1:
            raise ValueError("--output may be used only when exactly one model is selected")
        return {models[0]: explicit_output}
    return {
        model: output_dir / prompt_version / subset / model_filename(model)
        for model in models
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
            {
                "role": "user",
                "content": build_user_prompt(
                    str(case["input_text"]),
                    str(case["target_defect"]),
                    prompt_version=args.prompt_version,
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
        "User-Agent": "IdeaAmbig-Task3-Evaluation/1.0",
        "X-Title": "IdeaAmbig Task 3",
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
                "prompt_version": args.prompt_version,
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
    model_outputs = output_paths(
        models,
        args.output,
        args.output_dir,
        args.prompt_version,
        args.subset,
    )
    protected_inputs = {args.pool.resolve(), args.gold.resolve()}
    invalid_outputs = [
        path for path in model_outputs.values() if path.resolve() in protected_inputs
    ]
    if invalid_outputs:
        raise ValueError(
            "Prediction output must not overwrite the evaluation pool or gold file: "
            + ", ".join(str(path.resolve()) for path in invalid_outputs)
        )
    cases = load_jsonl(args.pool.resolve())
    gold = {str(row["case_id"]): row for row in load_jsonl(args.gold.resolve())}
    missing = [str(case["case_id"]) for case in cases if str(case["case_id"]) not in gold]
    if missing:
        raise ValueError(f"Pool case missing from gold: {missing[0]}")
    if args.subset != "all":
        cases = [
            case
            for case in cases
            if gold[str(case["case_id"])]["source_category"] == args.subset
        ]
    if args.limit > 0:
        cases = cases[: args.limit]
    if args.overwrite:
        for path in model_outputs.values():
            if path.exists():
                path.unlink()

    capabilities = model_capabilities(models, args.skip_catalog_check)
    done_by_model = {
        model: processed_keys(path, args.prompt_version)
        for model, path in model_outputs.items()
    }
    jobs = [
        (case, model)
        for model in models
        for case in cases
        if (model, str(case["case_id"])) not in done_by_model[model]
    ]
    if not jobs:
        print("No pending model/case pairs.")
        for model, path in model_outputs.items():
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
            concurrent.futures.as_completed(futures), total=len(futures), desc="Task 3"
        ):
            row = future.result()
            if row.get("status") == "ok":
                append_jsonl(model_outputs[str(row["model"])], row)
            else:
                failures += 1
                case_id, model = futures[future]
                tqdm.write(
                    f"ERROR {model}/{case_id}: {row.get('error', 'unknown error')}"
                )
    print(f"Completed {len(jobs)} jobs with {failures} failures.")
    for model, path in model_outputs.items():
        print(f"{model} -> {path.resolve()}")


if __name__ == "__main__":
    main()
