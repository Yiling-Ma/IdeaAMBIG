#!/usr/bin/env python3
"""Judge downstream specification quality for the clarification utility study."""

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
    TASK3_DIR,
    fetch_openrouter_catalog,
    load_jsonl,
    parse_json_object,
)


DEFAULT_DIR = TASK3_DIR / "outputs" / "synth" / "downstream_clarification"
WRITE_LOCK = threading.Lock()
ALLOWED_CONFIDENCE = {"high", "medium", "low"}

JUDGE_SYSTEM_PROMPT = """
You are a strict evaluator of implementation specifications for scientific
methods. Evaluate whether the generated specification faithfully resolves the
annotated implementation-critical defect without inventing unsupported details.
Return strict JSON only.
""".strip()

JUDGE_SCHEMA: dict[str, Any] = {
    "name": "ideaambig_downstream_spec_judgment",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "ready",
            "completeness",
            "missing_detail_recovered",
            "unsupported_assumption",
            "confidence",
            "reason",
        ],
        "properties": {
            "ready": {"type": "boolean"},
            "completeness": {"type": "integer", "minimum": 1, "maximum": 5},
            "missing_detail_recovered": {"type": "boolean"},
            "unsupported_assumption": {"type": "boolean"},
            "confidence": {"type": "string", "enum": sorted(ALLOWED_CONFIDENCE)},
            "reason": {"type": "string"},
        },
    },
}


def text(value: Any, default: str = "[not provided]") -> str:
    value = str(value or "").strip()
    return value or default


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    with WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()


def post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def build_judge_prompt(case: dict[str, Any], generation: dict[str, Any]) -> str:
    spec = generation.get("generation") if isinstance(generation.get("generation"), dict) else {}
    return f"""Evaluate the generated implementation specification.

Definitions:
- ready: The specification is implementation-ready with respect to the
  annotated target defect. It need not solve unrelated missing details.
- completeness: Rate from 1 to 5 how completely the specification describes
  implementation-critical details for the target defect and its local context.
  1 = unusable or mostly missing; 3 = partially usable but important details
  remain missing; 5 = complete enough to implement the target aspect.
- missing_detail_recovered: The specification explicitly recovers the private
  hidden resolution or an equivalent implementation detail.
- unsupported_assumption: The specification introduces an implementation choice
  not supported by the underspecified idea or, for clarification-assisted
  outputs, the oracle answer. Mark true for invented concrete values,
  unjustified defaults, or unsupported procedure choices.

Important rules:
- For the direct pipeline, the model did not receive the hidden resolution.
  Still judge whether its output recovered that detail and whether any guessed
  detail is unsupported by the original idea.
- For the clarification-assisted pipeline, the model received the oracle answer.
  It should incorporate that answer faithfully without adding unrelated guesses.
- Do not penalize the output merely for leaving unrelated non-target details as
  unresolved requirements.
- Evaluate the generated specification, not the quality of the original
  benchmark annotation.

Pipeline:
<pipeline>
{text(generation.get("pipeline"))}
</pipeline>

Underspecified idea:
<idea>
{text(case.get("input_text"))}
</idea>

Annotated target defect:
<target_defect>
{text(case.get("target_defect"))}
</target_defect>

Private hidden resolution:
<hidden_resolution>
{text(case.get("hidden_resolution"))}
</hidden_resolution>

Generated implementation specification:
<generated_specification>
{text(spec.get("implementation_specification"))}
</generated_specification>

Return only valid JSON:
{{
  "ready": true,
  "completeness": 5,
  "missing_detail_recovered": true,
  "unsupported_assumption": false,
  "confidence": "high|medium|low",
  "reason": "brief evidence-based explanation"
}}"""


def validate_judgment(value: dict[str, Any]) -> dict[str, Any]:
    for key in ("ready", "missing_detail_recovered", "unsupported_assumption"):
        if not isinstance(value.get(key), bool):
            raise ValueError(f"{key} must be boolean")
    completeness = value.get("completeness")
    if not isinstance(completeness, int) or not 1 <= completeness <= 5:
        raise ValueError("completeness must be an integer from 1 to 5")
    confidence = text(value.get("confidence"), "")
    if confidence not in ALLOWED_CONFIDENCE:
        raise ValueError(f"Invalid confidence: {confidence!r}")
    reason = text(value.get("reason"), "")
    if not reason:
        raise ValueError("reason must be non-empty")
    return {
        "ready": value["ready"],
        "completeness": completeness,
        "missing_detail_recovered": value["missing_detail_recovered"],
        "unsupported_assumption": value["unsupported_assumption"],
        "confidence": confidence,
        "reason": reason,
    }


def processed_keys(path: Path, judge_model: str) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    return {
        (str(row.get("case_id")), str(row.get("pipeline")), str(row.get("generator_model")))
        for row in load_jsonl(path)
        if row.get("status") == "ok" and row.get("judge_model") == judge_model
    }


def supported_parameters(model: str, skip_catalog_check: bool) -> set[str]:
    if skip_catalog_check:
        return set()
    catalog = fetch_openrouter_catalog()
    if model not in catalog:
        raise ValueError(
            f"Judge model absent from OpenRouter catalog: {model}. "
            "Use --skip-catalog-check only when access is known."
        )
    return set(catalog[model].get("supported_parameters") or [])


def judge_one(
    case: dict[str, Any],
    generation: dict[str, Any],
    args: argparse.Namespace,
    supported: set[str],
    api_key: str,
    base_url: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.judge_model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": build_judge_prompt(case, generation)},
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
        "User-Agent": "IdeaAmbig-Downstream-Judge/1.0",
        "X-Title": "IdeaAmbig Downstream Judge",
    }
    last_error = "unknown error"
    last_output = ""
    for attempt in range(1, args.max_retries + 1):
        try:
            response = post_json(
                f"{base_url.rstrip('/')}/chat/completions",
                headers,
                payload,
                args.timeout,
            )
            choice = (response.get("choices") or [])[0]
            last_output = text((choice.get("message") or {}).get("content"))
            judgment = validate_judgment(parse_json_object(last_output))
            return {
                "status": "ok",
                "case_id": generation["case_id"],
                "pipeline": generation["pipeline"],
                "generator_model": generation["model"],
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
        "case_id": generation.get("case_id"),
        "pipeline": generation.get("pipeline"),
        "generator_model": generation.get("model"),
        "judge_model": args.judge_model,
        "error": last_error,
    }
    if last_output:
        row["raw_output"] = last_output[:4000]
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_DIR / "selected_cases.jsonl")
    parser.add_argument("--pool", type=Path, default=TASK3_DIR / "outputs" / "synth" / "evaluation_pool.jsonl")
    parser.add_argument("--generations", type=Path, default=DEFAULT_DIR / "generations.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_DIR / "downstream_judgments.jsonl")
    parser.add_argument("--judge-model", default="anthropic/claude-opus-4.8")
    parser.add_argument("--generator-model", default="")
    parser.add_argument("--workers", type=int, default=4)
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


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    selected_cases = {str(row["case_id"]): row for row in load_jsonl(args.cases)}
    pool = {str(row["case_id"]): row for row in load_jsonl(args.pool)}
    cases = {
        case_id: {**pool.get(case_id, {}), **case}
        for case_id, case in selected_cases.items()
    }
    generations = [
        row
        for row in load_jsonl(args.generations)
        if row.get("status") == "ok"
        and str(row.get("case_id")) in cases
        and isinstance(row.get("generation"), dict)
        and (not args.generator_model or row.get("model") == args.generator_model)
    ]
    if not generations:
        raise ValueError("No downstream generations found for selected cases")
    if args.overwrite and args.output.exists():
        args.output.unlink()
    done = processed_keys(args.output, args.judge_model)
    jobs = [
        row
        for row in generations
        if (str(row["case_id"]), str(row["pipeline"]), str(row["model"])) not in done
    ]
    if not jobs:
        print("No pending downstream judgments.")
        return

    supported = supported_parameters(args.judge_model, args.skip_catalog_check)
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                judge_one,
                cases[str(row["case_id"])],
                row,
                args,
                supported,
                api_key,
                base_url,
            ): (row["case_id"], row["pipeline"])
            for row in jobs
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="DownstreamJudge",
        ):
            row = future.result()
            if row.get("status") == "ok":
                append_jsonl(args.output, row)
            else:
                failures += 1
                case_id, pipeline = futures[future]
                tqdm.write(
                    f"ERROR downstream judge {case_id}/{pipeline}: "
                    f"{row.get('error', 'unknown error')}"
                )
    print(
        f"Completed {len(jobs)} downstream judgments with {failures} failures -> "
        f"{args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
