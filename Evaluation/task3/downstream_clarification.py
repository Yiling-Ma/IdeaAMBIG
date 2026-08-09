#!/usr/bin/env python3
"""Run a downstream clarification utility study on Task 3 synthetic cases."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from task3_config import (
    TASK3_DIR,
    fetch_openrouter_catalog,
    load_jsonl,
    parse_json_object,
)


DEFAULT_SYNTH_POOL = TASK3_DIR / "outputs" / "synth" / "evaluation_pool.jsonl"
DEFAULT_SYNTH_GOLD = TASK3_DIR / "outputs" / "synth" / "evaluation_gold.jsonl"
DEFAULT_OUTPUT_DIR = TASK3_DIR / "outputs" / "synth" / "downstream_clarification"
WRITE_LOCK = threading.Lock()

SPEC_SCHEMA: dict[str, Any] = {
    "name": "ideaambig_downstream_specification",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["implementation_specification"],
        "properties": {
            "implementation_specification": {"type": "string", "minLength": 1},
        },
    },
}

SYSTEM_PROMPT = (
    "You are an expert scientific-method implementer. Your task is to rewrite "
    "underspecified research ideas into faithful, implementation-ready "
    "specifications. Return strict JSON only."
)


def text(value: Any) -> str:
    return str(value or "").strip()


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def atomic_write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    with WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()


def load_cases(pool_path: Path, gold_path: Path) -> list[dict[str, Any]]:
    pool = {str(row["case_id"]): row for row in load_jsonl(pool_path)}
    gold = {str(row["case_id"]): row for row in load_jsonl(gold_path)}
    if set(pool) != set(gold):
        raise ValueError("Pool and gold case_id sets do not match")
    cases: list[dict[str, Any]] = []
    for case_id in sorted(pool):
        gold_row = gold[case_id]
        cases.append(
            {
                "case_id": case_id,
                "input_text": text(pool[case_id].get("input_text")),
                "target_defect": text(pool[case_id].get("target_defect")),
                "level1": text(gold_row.get("level1")),
                "level2": text(gold_row.get("level2")),
                "hidden_resolution": text(gold_row.get("hidden_resolution")),
                "gold_action_type": text(gold_row.get("gold_action_type")),
                "gold_action": text(gold_row.get("gold_action")),
                "gold_evidence_to_seek": text(gold_row.get("gold_evidence_to_seek")),
                "paper_id": text(gold_row.get("paper_id")),
                "instance_id": text(gold_row.get("instance_id")),
            }
        )
    return cases


def stratified_sample(cases: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    if sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    if sample_size > len(cases):
        raise ValueError(f"--sample-size {sample_size} exceeds available cases {len(cases)}")
    by_level2: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_level2[case["level2"]].append(case)
    labels = sorted(by_level2)
    for label in labels:
        by_level2[label].sort(key=lambda row: stable_key(seed, row["case_id"]))

    selected: list[dict[str, Any]] = []
    base = sample_size // len(labels)
    remainder = sample_size % len(labels)
    label_order = sorted(labels, key=lambda label: stable_key(seed, label))
    quotas = {label: base + (1 if index < remainder else 0) for index, label in enumerate(label_order)}
    for label in labels:
        selected.extend(by_level2[label][: min(quotas[label], len(by_level2[label]))])

    if len(selected) < sample_size:
        selected_ids = {case["case_id"] for case in selected}
        leftovers = [
            case for case in cases if case["case_id"] not in selected_ids
        ]
        leftovers.sort(key=lambda row: stable_key(seed + 17, row["case_id"]))
        selected.extend(leftovers[: sample_size - len(selected)])

    selected.sort(key=lambda row: stable_key(seed + 101, row["case_id"]))
    return selected[:sample_size]


def build_direct_prompt(case: dict[str, Any]) -> str:
    return f"""Task: Direct Implementation Specification Generation

Given the following underspecified research idea, expand it into a faithful,
implementation-ready specification.

Requirements:
1. Do not ask clarification questions.
2. Preserve all information in the input.
3. Make implementation choices explicit only when they are supported by the
   input. If a detail is not determined, state it as an unresolved requirement
   rather than inventing a value.
4. Focus on details needed by an implementer: model, data, algorithm,
   training, inference, evaluation, and reproducibility settings.
5. Keep the specification concise, preferably under 900 words.

Underspecified research idea:
<idea>
{case["input_text"]}
</idea>

Return only valid JSON:
{{
  "implementation_specification": "complete implementation specification"
}}"""


def build_clarification_prompt(case: dict[str, Any]) -> str:
    return f"""Task: Clarification-Assisted Implementation Specification Generation

Given an underspecified research idea, an annotated target defect, a
clarification action, and the oracle clarification answer, revise the idea into
a faithful, implementation-ready specification.

Requirements:
1. Incorporate the oracle answer to resolve the target defect.
2. Preserve all non-target information in the input.
3. Do not add unsupported implementation choices beyond the input and oracle
   answer.
4. Make the resolved implementation detail explicit enough for an implementer.
5. Keep the specification concise, preferably under 900 words.

Underspecified research idea:
<idea>
{case["input_text"]}
</idea>

Annotated target defect:
<target_defect>
{case["target_defect"]}
</target_defect>

Clarification action:
<clarification_action>
Action type: {case["gold_action_type"]}
Action: {case["gold_action"]}
Evidence to seek: {case["gold_evidence_to_seek"]}
</clarification_action>

Oracle clarification answer:
<oracle_answer>
{case["hidden_resolution"]}
</oracle_answer>

Return only valid JSON:
{{
  "implementation_specification": "complete implementation specification"
}}"""


def validate_generation(value: dict[str, Any]) -> dict[str, str]:
    specification = text(value.get("implementation_specification"))
    if not specification:
        raise ValueError("implementation_specification must be non-empty")
    return {"implementation_specification": specification}


def post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def supported_parameters(model: str, skip_catalog_check: bool) -> set[str]:
    if skip_catalog_check:
        return set()
    catalog = fetch_openrouter_catalog()
    if model not in catalog:
        raise ValueError(
            f"Model ID absent from OpenRouter catalog: {model}. "
            "Use --skip-catalog-check only when access is known."
        )
    return set(catalog[model].get("supported_parameters") or [])


def build_payload(
    model: str,
    prompt: str,
    supported: set[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
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
        payload["response_format"] = {"type": "json_schema", "json_schema": SPEC_SCHEMA}
    return payload


def generate_one(
    case: dict[str, Any],
    pipeline: str,
    model: str,
    supported: set[str],
    args: argparse.Namespace,
    api_key: str,
    base_url: str,
) -> dict[str, Any]:
    prompt = build_direct_prompt(case) if pipeline == "direct" else build_clarification_prompt(case)
    payload = build_payload(model, prompt, supported, args)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "IdeaAmbig-Downstream-Clarification/1.0",
        "X-Title": "IdeaAmbig Downstream Clarification",
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
            generation = validate_generation(parse_json_object(last_output))
            usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
            return {
                "status": "ok",
                "case_id": case["case_id"],
                "pipeline": pipeline,
                "model": model,
                "generation": generation,
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
        "pipeline": pipeline,
        "model": model,
        "error": last_error,
    }
    if last_output:
        row["raw_output"] = last_output[:4000]
    return row


def processed_keys(path: Path, model: str) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {
        (str(row.get("case_id")), str(row.get("pipeline")))
        for row in load_jsonl(path)
        if row.get("status") == "ok" and row.get("model") == model
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_SYNTH_POOL)
    parser.add_argument("--gold", type=Path, default=DEFAULT_SYNTH_GOLD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="openai/gpt-5.6-sol")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--reasoning-effort", choices=("none", "low", "medium", "high"), default="low"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-catalog-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    cases = stratified_sample(
        load_cases(args.pool.resolve(), args.gold.resolve()),
        args.sample_size,
        args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = args.output_dir / "selected_cases.jsonl"
    prompts_path = args.output_dir / "generation_prompts.jsonl"
    generations_path = args.output_dir / "generations.jsonl"

    selected_rows = [
        {
            "case_id": case["case_id"],
            "instance_id": case["instance_id"],
            "paper_id": case["paper_id"],
            "level1": case["level1"],
            "level2": case["level2"],
            "target_defect": case["target_defect"],
            "hidden_resolution": case["hidden_resolution"],
            "gold_action_type": case["gold_action_type"],
            "gold_action": case["gold_action"],
        }
        for case in cases
    ]
    prompt_rows = []
    for case in cases:
        prompt_rows.append(
            {
                "case_id": case["case_id"],
                "pipeline": "direct",
                "model": args.model,
                "prompt": build_direct_prompt(case),
            }
        )
        prompt_rows.append(
            {
                "case_id": case["case_id"],
                "pipeline": "clarification_assisted",
                "model": args.model,
                "prompt": build_clarification_prompt(case),
            }
        )
    atomic_write_jsonl(selected_rows, cases_path)
    atomic_write_jsonl(prompt_rows, prompts_path)

    print(f"Selected {len(cases)} synthetic cases -> {cases_path.resolve()}")
    print(f"Wrote {len(prompt_rows)} generation prompts -> {prompts_path.resolve()}")
    if args.dry_run:
        return

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    supported = supported_parameters(args.model, args.skip_catalog_check)
    if args.overwrite and generations_path.exists():
        generations_path.unlink()
    done = processed_keys(generations_path, args.model)
    jobs = [
        (case, pipeline)
        for case in cases
        for pipeline in ("direct", "clarification_assisted")
        if (case["case_id"], pipeline) not in done
    ]
    if not jobs:
        print("No pending downstream generations.")
        return

    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                generate_one,
                case,
                pipeline,
                args.model,
                supported,
                args,
                api_key,
                base_url,
            ): (case["case_id"], pipeline)
            for case, pipeline in jobs
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Downstream",
        ):
            row = future.result()
            if row.get("status") == "ok":
                append_jsonl(generations_path, row)
            else:
                failures += 1
                case_id, pipeline = futures[future]
                tqdm.write(
                    f"ERROR downstream {case_id}/{pipeline}: "
                    f"{row.get('error', 'unknown error')}"
                )
    print(
        f"Completed {len(jobs)} downstream generations with {failures} failures -> "
        f"{generations_path.resolve()}"
    )


if __name__ == "__main__":
    main()
