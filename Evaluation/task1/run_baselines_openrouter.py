#!/usr/bin/env python3
"""Run Task 1 zero-shot baselines through OpenRouter with resume support."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from tqdm import tqdm

from task1_config import (
    DEFAULT_GOLD,
    DEFAULT_POOL,
    DEFAULT_PREDICTIONS_DIR,
    MODEL_ALIASES,
    MODELS,
    PROMPT_VERSIONS,
    build_user_prompt,
    fetch_openrouter_catalog,
    load_jsonl,
    model_filename,
    option_mapping,
    resolved_model_registry,
    source_category,
    system_prompt,
)


WRITE_LOCK = threading.Lock()
MODELS_WITHOUT_REASONING = {
    "qwen/qwen3.5-397b-a17b",
}


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
            "<output-dir>/<prompt-version>/<pool-tag>/<subset>/<model>.jsonl."
        ),
    )
    parser.add_argument(
        "--prompt-version",
        choices=PROMPT_VERSIONS,
        default="v1",
        help=(
            "Prompt rubric version. Defaults to v1."
        ),
    )
    parser.add_argument(
        "--models",
        default="all",
        help="all, frontier, open_weight, or comma-separated OpenRouter model IDs",
    )
    parser.add_argument(
        "--subset",
        choices=("all", "real", "synthetic"),
        default="all",
        help="Evaluate all cases or select a source subset using private gold metadata.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Debug-only prefix limit; it does not preserve class balance. "
            "Use build_balanced_subset.py for reported evaluations."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--reasoning-effort", choices=("none", "low", "medium", "high"), default="low"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Includes hidden reasoning tokens for providers that bill them as completion tokens.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def selected_models(spec: str) -> list[str]:
    llm_models = {
        model: config
        for model, config in MODELS.items()
        if config["group"] in {"frontier_proprietary", "open_weight"}
    }
    if spec == "all":
        return list(llm_models)
    if spec == "frontier":
        return [m for m, c in llm_models.items() if c["group"] == "frontier_proprietary"]
    if spec == "open_weight":
        return [m for m, c in llm_models.items() if c["group"] == "open_weight"]
    raw_values = [value.strip() for value in spec.split(",") if value.strip()]
    values = [MODEL_ALIASES.get(value.lower(), value) for value in raw_values]
    unknown = sorted(set(values) - set(llm_models))
    if unknown:
        raise ValueError(f"Unknown configured OpenRouter models: {', '.join(unknown)}")
    return values


def processed_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {
        (str(row.get("model")), str(row.get("case_id")))
        for row in load_jsonl(path)
        if row.get("status") == "ok" and isinstance(row.get("predicted_ready"), bool)
    }


def pool_tag(pool_path: Path) -> str:
    """Distinguish the full pool from named frozen subsets such as real100."""
    resolved = pool_path.resolve()
    if resolved.parent == Path(__file__).resolve().parent:
        return "full"
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", resolved.parent.name).strip("._-")
    return value or "custom"


def output_paths(
    models: list[str],
    explicit_output: Path | None,
    output_dir: Path,
    prompt_version: str,
    selected_pool_tag: str,
    subset: str,
) -> dict[str, Path]:
    if explicit_output is not None:
        if len(models) != 1:
            raise ValueError("--output may be used only when exactly one model is selected")
        return {models[0]: explicit_output}
    return {
        model: (
            output_dir
            / prompt_version
            / selected_pool_tag
            / subset
            / model_filename(model)
        )
        for model in models
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    with WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()


def normalize_option_token(value: Any) -> str | None:
    text = str(value or "").strip().strip('"\'`').upper()
    # Some providers follow the semantics but spell the response as "Option B"
    # rather than the requested bare "B". Treat both as the same valid choice.
    match = re.match(r"^(?:OPTION\s+)?([AB])(?:\b|$)", text)
    return match.group(1) if match else None


def extract_reason_text(content: str) -> str | None:
    text = str(content or "").strip()
    if not text:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    first_line = lines[0]
    remainder = "\n".join(lines[1:]).strip()
    if remainder:
        reason = remainder
    else:
        match = re.match(r"^(?:OPTION\s+)?[AB](?::|\s+)(.+)$", first_line, flags=re.IGNORECASE)
        reason = match.group(1).strip() if match else ""
    if not reason:
        return None
    reason = re.sub(r"^\s*Reason:\s*", "", reason, flags=re.IGNORECASE).strip()
    return reason or None


def parse_prediction(content: str, mapping: dict[str, str]) -> tuple[str, bool, str | None]:
    option = normalize_option_token(content)
    if option is None:
        raise ValueError(f"Expected exactly A or B, received {content!r}")
    return option, mapping[option] == "READY", extract_reason_text(content)


def post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def build_payload(
    model: str,
    case: dict[str, Any],
    model_config: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, str]]:
    mapping = option_mapping(str(case["case_id"]), args.seed)
    supported = set(model_config.get("supported_parameters") or [])
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt(args.prompt_version)},
            {
                "role": "user",
                "content": build_user_prompt(
                    str(case["input_text"]),
                    mapping,
                    prompt_version=args.prompt_version,
                ),
            },
        ],
    }
    if "max_tokens" in supported:
        payload["max_tokens"] = args.max_tokens
    elif "max_completion_tokens" in supported:
        payload["max_completion_tokens"] = args.max_tokens
    if "seed" in supported:
        payload["seed"] = args.seed
    if (
        "reasoning" in supported
        and args.reasoning_effort != "none"
        and model not in MODELS_WITHOUT_REASONING
    ):
        payload["reasoning"] = {"effort": args.reasoning_effort, "exclude": True}
    return payload, mapping


def evaluate_one(
    case: dict[str, Any],
    model: str,
    model_config: dict[str, Any],
    args: argparse.Namespace,
    api_key: str,
    base_url: str,
) -> dict[str, Any]:
    payload, mapping = build_payload(model, case, model_config, args)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "IdeaAmbig-Task1-Evaluation/1.0",
        "X-Title": "IdeaAmbig Task 1 Evaluation",
    }
    last_error = "unknown error"
    for attempt in range(1, args.max_retries + 1):
        try:
            response = post_json(
                f"{base_url.rstrip('/')}/chat/completions", headers, payload, args.timeout
            )
            choice = (response.get("choices") or [])[0]
            message = choice.get("message") or {}
            raw_output = str(message.get("content") or "")
            if not raw_output.strip():
                raise RuntimeError("Empty model output")
            option, predicted_ready, reason_text = parse_prediction(raw_output, mapping)
            row = {
                "status": "ok",
                "case_id": case["case_id"],
                "model": model,
                "model_group": model_config["group"],
                "predicted_ready": predicted_ready,
                "selected_option": option,
                "option_mapping": mapping,
            }
            if reason_text:
                row["reason_text"] = reason_text
            return row
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
        "status": "error",
        "case_id": case["case_id"],
        "model": model,
        "model_group": model_config["group"],
        "error": last_error,
    }


def main() -> None:
    args = parse_args()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    models = selected_models(args.models)
    model_outputs = output_paths(
        models,
        args.output,
        args.output_dir,
        args.prompt_version,
        pool_tag(args.pool),
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
    if args.subset != "all":
        gold = {str(row["case_id"]): row for row in load_jsonl(args.gold.resolve())}
        unknown_case_ids = sorted(str(case["case_id"]) for case in cases if str(case["case_id"]) not in gold)
        if unknown_case_ids:
            raise ValueError(
                f"Pool contains {len(unknown_case_ids)} case IDs absent from gold; "
                f"first: {unknown_case_ids[0]}"
            )
        cases = [
            case
            for case in cases
            if source_category(gold[str(case["case_id"])]) == args.subset
        ]
    if args.limit > 0:
        cases = cases[: args.limit]
    if args.overwrite:
        for path in model_outputs.values():
            if path.exists():
                path.unlink()

    try:
        catalog = fetch_openrouter_catalog()
        registry = resolved_model_registry(catalog)
    except Exception as exc:
        registry = resolved_model_registry()
        print(
            f"Warning: using bundled model registry because catalog refresh failed: "
            f"{type(exc).__name__}: {exc}"
        )

    done_by_model = {
        model: processed_keys(path) for model, path in model_outputs.items()
    }
    jobs = [
        (case, model)
        for model in models
        for case in cases
        if (model, str(case["case_id"])) not in done_by_model[model]
    ]
    if not jobs:
        print("No pending cases; all selected model/case pairs are already complete.")
        for model, path in model_outputs.items():
            print(f"{model} -> {path.resolve()}")
        return

    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {
            executor.submit(
                evaluate_one,
                case,
                model,
                registry[model],
                args,
                api_key,
                base_url,
            ): (case["case_id"], model)
            for case, model in jobs
        }
        with tqdm(total=len(future_to_job), desc="Task 1", unit="request") as progress:
            for future in concurrent.futures.as_completed(future_to_job):
                row = future.result()
                if row.get("status") == "ok":
                    append_jsonl(model_outputs[str(row["model"])], row)
                else:
                    failures += 1
                    case_id, model = future_to_job[future]
                    tqdm.write(
                        f"ERROR {model}/{case_id}: "
                        f"{row.get('error', 'unknown error')}"
                    )
                progress.update(1)

    print(
        json.dumps(
            {
                "outputs": {
                    model: str(path.resolve()) for model, path in model_outputs.items()
                },
                "models": models,
                "subset": args.subset,
                "prompt_version": args.prompt_version,
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
