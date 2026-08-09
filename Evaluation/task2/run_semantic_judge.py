#!/usr/bin/env python3
"""Judge whether each predicted description localizes the annotated target defect."""

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
    DEFAULT_JUDGMENTS,
    DEFAULT_POOL,
    DEFAULT_PREDICTIONS,
    fetch_openrouter_catalog,
    load_jsonl,
    parse_json_object,
)


WRITE_LOCK = threading.Lock()

MATCH_RELATIONSHIPS = {
    "exact_same_target",
    "same_target_wrong_characterization",
    "broader_but_uniquely_identifies_target",
}
ALLOWED_RELATIONSHIPS = MATCH_RELATIONSHIPS | {
    "neighboring_defect",
    "different_target",
    "too_vague_to_localize",
    "no_defect_identified",
}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}

JUDGE_SYSTEM_PROMPT = """
You are a strict target-defect identity judge for a scientific-method benchmark.

Do NOT judge whether the predicted defect is valid, important, or present
somewhere in the specification. A specification may contain multiple genuine
defects. Your only task is to decide whether the prediction identifies the same
annotated implementation decision or specification slot as the gold target.

Compare the affected component, the concrete implementation slot, and the
missing, ambiguous, or conflicting property at that slot. Shared terminology,
a shared broad module, or a shared downstream consequence is not sufficient.

A prediction may still match when its taxonomy or defect-type characterization
is wrong, provided that it uniquely localizes the same target slot.

Return strict JSON only.
""".strip()
JUDGE_SCHEMA: dict[str, Any] = {
    "name": "ideaambig_target_identity_match",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "gold_target_slot",
            "predicted_target_slot",
            "relationship",
            "match",
            "confidence",
            "reason",
        ],
        "properties": {
            "gold_target_slot": {"type": "string"},
            "predicted_target_slot": {"type": "string"},
            "relationship": {
                "type": "string",
                "enum": sorted(ALLOWED_RELATIONSHIPS),
            },
            "match": {"type": "boolean"},
            "confidence": {
                "type": "string",
                "enum": sorted(ALLOWED_CONFIDENCE),
            },
            "reason": {"type": "string"},
        },
    },
}


def text(value: Any, default: str = "[not provided]") -> str:
    if value is None:
        return default
    value = str(value).strip()
    return value or default


def extract_local_context(
    specification: str,
    surface_form: str | None,
    radius: int = 1800,
    fallback_chars: int = 3600,
) -> str:
    """Use surface-form-adjacent context to reduce distraction from other defects."""
    specification = specification or ""
    if not specification.strip():
        return "[not provided]"

    surface_form = (surface_form or "").strip()
    if not surface_form:
        return specification[:fallback_chars]

    index = specification.casefold().find(surface_form.casefold())
    if index < 0:
        return specification[:fallback_chars]

    start = max(0, index - radius)
    end = min(len(specification), index + len(surface_form) + radius)
    context = specification[start:end]
    if start > 0:
        context = "[...]\n" + context
    if end < len(specification):
        context += "\n[...]"
    return context


def build_judge_prompt(
    specification: str, target_defect: dict[str, Any], predicted_description: str
) -> str:
    """Build the target-identity prompt without taxonomy labels."""
    blocking = target_defect.get("blocking_missing_specs")
    blocking_text = (
        "; ".join(text(item) for item in blocking)
        if isinstance(blocking, list) and blocking
        else "[not provided]"
    )
    gold_description = text(target_defect.get("description"))
    surface_form = text(target_defect.get("surface_form"))
    resolution_reference = text(target_defect.get("missing_or_corrupted_detail"))
    target_signature = target_defect.get("target_signature")
    target_signature_text = (
        json.dumps(target_signature, ensure_ascii=False, indent=2, sort_keys=True)
        if isinstance(target_signature, dict) and target_signature
        else "[not provided]"
    )
    local_context = extract_local_context(
        specification,
        None if surface_form == "[not provided]" else surface_form,
    )
    prediction = text(predicted_description, "[empty prediction]")

    return f"""
Determine whether the prediction identifies the same concrete target defect as
the annotation.

IMPORTANT
The specification may contain multiple genuine defects. A prediction that finds
a valid but different defect must receive match=false.

First normalize both defects into:
1. affected component or entity;
2. concrete implementation decision, rule, variable, or specification slot;
3. property that is missing, ambiguous, or inconsistent.

Apply these checks:

DIRECT-FIX TEST
Would directly fixing the predicted issue also fix the annotated issue?

INDEPENDENT-COEXISTENCE TEST
Could both issues exist independently in the same specification? If yes, they
are normally different defects.

TARGET-SLOT TEST
Do both descriptions concern the same concrete rule, threshold, variable,
operation, module boundary, loss term, transformation, or stopping condition?

Decision rules:
- Paraphrases and different specificity may match.
- The prediction does not need to recover the correct hidden resolution.
- Wrong taxonomy may still match if the same target slot is localized.
- Same topic, component, algorithm, or downstream effect is not enough.
- A neighboring issue in the same component is match=false.
- A different valid defect is match=false.
- A generic statement that could refer to multiple defects is match=false.
- A repair without identifying the target slot is match=false.

Relationship labels:
- exact_same_target: same slot and substantially correct characterization.
- same_target_wrong_characterization: same slot, wrong defect characterization.
- broader_but_uniquely_identifies_target: broader wording, but uniquely same slot.
- neighboring_defect: related component, different implementation decision.
- different_target: different component, rule, variable, or decision.
- too_vague_to_localize: no unique concrete target slot.
- no_defect_identified: no actual defect is stated.

Set match=true only for:
- exact_same_target
- same_target_wrong_characterization
- broader_but_uniquely_identifies_target

Boundary examples:

Example 1
Gold: The tree-search stopping condition is inconsistent.
Prediction: The maximum tree depth is not specified.
Result: neighboring_defect, match=false. Same search module, different decision.

Example 2
Gold: Paper and code disagree on the event count defining a leaf.
Prediction: The leaf-node event threshold is underspecified.
Result: same_target_wrong_characterization, match=true. Same target slot, wrong
defect type.

Example 3
Gold: The weight combining two loss terms is unspecified.
Prediction: The optimizer learning rate is unspecified.
Result: different_target, match=false. Both affect training but are independent.

<gold_target>
Annotated defect:
{gold_description}

Relevant source wording:
{surface_form}

Canonical blocking specification / target slot:
{blocking_text}

Optional structured target signature:
{target_signature_text}

Resolution reference:
{resolution_reference}
</gold_target>

The resolution reference is only evidence for identifying the intended target
slot. The prediction does not need to recover that resolution. Do not establish
identity from shared consequences.

<specification_context>
{local_context}
</specification_context>

Use the context only to resolve terminology. Do not validate an alternative
defect from the context and count it as a match.

<prediction>
{prediction}
</prediction>

Return one JSON object with:
- gold_target_slot: concise normalized gold slot;
- predicted_target_slot: concise normalized predicted slot;
- relationship: one allowed relationship label;
- match: boolean consistent with the relationship;
- confidence: high, medium, or low;
- reason: briefly compare the two concrete implementation decisions.

The reason must state what implementation decision each defect refers to. Do
not merely say that they are similar or different.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--judge-model", default="anthropic/claude-opus-4.8")
    parser.add_argument("--models", default="all", help="all or evaluated model IDs to judge")
    parser.add_argument("--subset", choices=("all", "real", "synthetic"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=1024)
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


def post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
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
        and isinstance(row.get("semantic_match"), bool)
    }


def build_payload(
    judge_model: str,
    prompt: str,
    supported: set[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": judge_model,
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
    if "reasoning" in supported:
        payload["reasoning"] = {"effort": "low", "exclude": True}
    if "response_format" in supported or "structured_outputs" in supported:
        payload["response_format"] = {"type": "json_schema", "json_schema": JUDGE_SCHEMA}
    return payload


def validate_judge_output(value: dict[str, Any]) -> dict[str, Any]:
    relationship = text(value.get("relationship"), "")
    if relationship not in ALLOWED_RELATIONSHIPS:
        raise ValueError(f"Invalid relationship: {relationship!r}")

    match = value.get("match")
    if not isinstance(match, bool):
        raise ValueError("Judge output must contain boolean match")

    expected_match = relationship in MATCH_RELATIONSHIPS
    if match != expected_match:
        raise ValueError(
            f"Relationship {relationship!r} implies match={expected_match}, "
            f"but judge returned {match}"
        )

    confidence = text(value.get("confidence"), "")
    if confidence not in ALLOWED_CONFIDENCE:
        raise ValueError(f"Invalid confidence: {confidence!r}")

    result = {
        "gold_target_slot": text(value.get("gold_target_slot"), ""),
        "predicted_target_slot": text(value.get("predicted_target_slot"), ""),
        "relationship": relationship,
        "match": match,
        "confidence": confidence,
        "reason": text(value.get("reason"), ""),
    }
    for field in ("gold_target_slot", "predicted_target_slot", "reason"):
        if not result[field]:
            raise ValueError(f"{field} must be a non-empty string")
    return result


def judge_one(
    job: dict[str, Any],
    judge_model: str,
    supported: set[str],
    args: argparse.Namespace,
    api_key: str,
    base_url: str,
) -> dict[str, Any]:
    prompt = build_judge_prompt(
        job["input_text"], job["target_defect"], job["predicted_description"]
    )
    payload = build_payload(judge_model, prompt, supported, args)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "IdeaAmbig-Task2-Track1-Judge/1.0",
        "X-Title": "IdeaAmbig Task 2 Track 1 Semantic Judge",
    }
    last_error = "unknown error"
    for attempt in range(1, args.max_retries + 1):
        try:
            response = post_json(
                f"{base_url.rstrip('/')}/chat/completions", headers, payload, args.timeout
            )
            choice = (response.get("choices") or [])[0]
            content = str((choice.get("message") or {}).get("content") or "")
            value = parse_json_object(content)
            if not isinstance(value, dict):
                raise ValueError("Parsed judge output must be a JSON object")
            value = validate_judge_output(value)
            return {
                "status": "ok",
                "case_id": job["case_id"],
                "evaluated_model": job["evaluated_model"],
                "judge_model": judge_model,
                "semantic_match": value["match"],
                "relationship": value["relationship"],
                "gold_target_slot": value["gold_target_slot"],
                "predicted_target_slot": value["predicted_target_slot"],
                "confidence": value["confidence"],
                "reason": value["reason"],
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
    return {
        "status": "error",
        "case_id": job["case_id"],
        "evaluated_model": job["evaluated_model"],
        "judge_model": judge_model,
        "error": last_error,
    }


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    if not args.predictions.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {args.predictions.resolve()}\n"
            "Semantic matching is the second pipeline stage. First run "
            "run_baselines_openrouter.py, or pass --predictions PATH if the model "
            "predictions were written elsewhere."
        )

    pool = {str(row["case_id"]): row for row in load_jsonl(args.pool.resolve())}
    gold = {str(row["case_id"]): row for row in load_jsonl(args.gold.resolve())}
    if set(pool) != set(gold):
        raise ValueError("Pool and gold case_id sets do not match")

    prediction_rows = load_jsonl(args.predictions.resolve())
    successful_rows = [
        row
        for row in prediction_rows
        if row.get("status") == "ok" and isinstance(row.get("prediction"), dict)
    ]

    predictions: dict[tuple[str, str], dict[str, Any]] = {}
    available_models: set[str] = set()
    duplicate_keys: set[tuple[str, str]] = set()
    for row in successful_rows:
        model = str(row.get("model"))
        case_id = str(row.get("case_id"))
        available_models.add(model)
        key = (model, case_id)
        if key in predictions:
            duplicate_keys.add(key)
        predictions[key] = row
    if duplicate_keys:
        preview = sorted(duplicate_keys)[:5]
        raise ValueError(
            "Duplicate successful predictions found for the same model/case_id. "
            f"Examples: {preview}. Remove duplicates before judging."
        )
    if not predictions:
        observed_fields = sorted(
            {str(field) for row in prediction_rows[:20] for field in row}
        )
        raise ValueError(
            "No compatible Task 2 Track 1 predictions were found in "
            f"{args.predictions.resolve()}. Expected successful rows containing a "
            "'prediction' object with description, level1, and level2. Observed fields: "
            f"{observed_fields}. Task 1 readiness predictions cannot be used for Task 2; "
            "run run_baselines_openrouter.py from this Track 1 directory first."
        )
    prediction_case_ids = {case_id for _, case_id in predictions}
    if not prediction_case_ids.intersection(gold):
        example = sorted(prediction_case_ids)[:3]
        raise ValueError(
            "Prediction case IDs do not belong to the Task 2 Track 1 pool. "
            f"Examples: {example}. Use predictions generated from evaluation_pool.jsonl "
            "in this directory."
        )
    selected = available_models if args.models == "all" else {
        value.strip() for value in args.models.split(",") if value.strip()
    }
    unknown = sorted(selected - available_models)
    if unknown:
        raise ValueError("No predictions found for: " + ", ".join(unknown))

    if args.overwrite and args.output.exists():
        args.output.unlink()

    jobs: list[dict[str, Any]] = []
    done = processed_keys(args.output, args.judge_model)
    for (model, case_id), row in predictions.items():
        if model not in selected or (model, case_id) in done:
            continue
        gold_row = gold.get(case_id)
        if not gold_row:
            continue
        if args.subset != "all" and gold_row["source_category"] != args.subset:
            continue
        jobs.append(
            {
                "case_id": case_id,
                "evaluated_model": model,
                "input_text": pool[case_id]["input_text"],
                "target_defect": gold_row["target_defect"],
                "predicted_description": row["prediction"].get("description", ""),
            }
        )
    jobs.sort(key=lambda row: (row["evaluated_model"], row["case_id"]))
    if args.limit > 0:
        limited_jobs: list[dict[str, Any]] = []
        per_model_counts: dict[str, int] = {}
        for job in jobs:
            model = str(job["evaluated_model"])
            count = per_model_counts.get(model, 0)
            if count < args.limit:
                limited_jobs.append(job)
                per_model_counts[model] = count + 1
        jobs = limited_jobs
    if not jobs:
        print("No pending semantic judgments.")
        return

    supported: set[str] = set()
    if not args.skip_catalog_check:
        catalog = fetch_openrouter_catalog()
        if args.judge_model not in catalog:
            raise ValueError(
                f"Judge model {args.judge_model!r} is absent from the OpenRouter catalog"
            )
        supported = set(catalog[args.judge_model].get("supported_parameters") or [])

    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                judge_one,
                job,
                args.judge_model,
                supported,
                args,
                api_key,
                base_url,
            ): job
            for job in jobs
        }
        with tqdm(total=len(futures), desc="Semantic matching", unit="pair") as progress:
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                append_jsonl(args.output, row)
                failures += row.get("status") != "ok"
                progress.update(1)

    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "judge_model": args.judge_model,
                "judgments_attempted": len(jobs),
                "failures": failures,
            },
            indent=2,
        )
    )
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
