"""Configuration, prompts, schemas, and shared I/O for Task 3."""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Any


TASK3_DIR = Path(__file__).resolve().parent
DEFAULT_INSTANCES = TASK3_DIR.parent / "instances.jsonl"
DEFAULT_POOL = TASK3_DIR / "evaluation_pool.jsonl"
DEFAULT_GOLD = TASK3_DIR / "evaluation_gold.jsonl"
DEFAULT_PREDICTIONS_DIR = TASK3_DIR / "outputs" / "predictions"
DEFAULT_JUDGMENTS = TASK3_DIR / "outputs" / "action_judgments.jsonl"
DEFAULT_SCORES_DIR = TASK3_DIR / "outputs" / "scores"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

ACTION_TYPES = ("clarification_question", "evidence_seeking")

MODELS: dict[str, dict[str, str]] = {
    "google/gemini-3.1-pro-preview": {"group": "frontier_proprietary"},
    "anthropic/claude-sonnet-5": {"group": "frontier_proprietary"},
    "openai/gpt-5.6-sol": {"group": "frontier_proprietary"},
    "qwen/qwen3-8b": {"group": "open_weight"},
    "qwen/qwen3-32b": {"group": "open_weight"},
    "deepseek/deepseek-r1": {"group": "open_weight"},
    "openai/gpt-oss-120b": {"group": "open_weight"},
}

OPTIONAL_MODEL_GROUPS = {
    "deepseek/deepseek-v4-pro": "frontier_proprietary",
}

SYSTEM_PROMPT = (
    "You are an expert scientific-method reviewer proposing one clarification action "
    "for an already identified implementation-critical defect. Do not diagnose other "
    "defects, repair the method, or invent the missing detail. Return strict JSON only."
)

PROMPT_VERSIONS = ("v1",)


def build_user_prompt_v1(input_text: str, target_defect: str) -> str:
    return f"""Task: Clarification Action Generation

You are given an underspecified research idea and an independently annotated
target defect. The target defect has already been identified; do not perform
readiness assessment or defect localization again.

Generate exactly one atomic action that would obtain the missing information
needed to resolve the target defect before implementation.

Allowed action types:
- clarification_question: ask one precise, answerable question of an author or
  implementer who knows the intended method.
- evidence_seeking: specify one concrete, executable inspection of an available
  artifact such as paper text, source code, configuration, data documentation,
  or an experiment, and state what implementation fact should be checked.

Requirements:
1. Address only the supplied target defect.
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

Annotated target defect:
<target_defect>
{target_defect}
</target_defect>

Return only valid JSON:
{{
  "action_type": "clarification_question|evidence_seeking",
  "action": "one atomic clarification question or evidence-seeking action",
  "expected_information": "the unknown implementation information this action should obtain"
}}"""


def build_user_prompt(
    input_text: str, target_defect: str, prompt_version: str = "v1"
) -> str:
    if prompt_version == "v1":
        return build_user_prompt_v1(input_text, target_defect)
    raise ValueError(
        f"Unknown prompt version {prompt_version!r}; expected one of {PROMPT_VERSIONS}"
    )


PREDICTION_JSON_SCHEMA: dict[str, Any] = {
    "name": "ideaambig_task3_clarification_action",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["action_type", "action", "expected_information"],
        "properties": {
            "action_type": {"type": "string", "enum": list(ACTION_TYPES)},
            "action": {"type": "string", "minLength": 1},
            "expected_information": {"type": "string", "minLength": 1},
        },
    },
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def load_jsonl_collection(path: Path) -> list[dict[str, Any]]:
    """Load one JSONL file or recursively combine all JSONL files in a directory."""
    if path.is_file():
        return load_jsonl(path)
    if not path.is_dir():
        raise FileNotFoundError(f"JSONL input not found: {path}")
    files = sorted(candidate for candidate in path.rglob("*.jsonl") if candidate.is_file())
    if not files:
        raise FileNotFoundError(f"No JSONL files found under directory: {path}")
    rows: list[dict[str, Any]] = []
    for file_path in files:
        rows.extend(load_jsonl(file_path))
    return rows


def model_filename(model: str) -> str:
    """Create a stable, filesystem-safe filename from an OpenRouter model ID."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "__", model.strip()).strip("._-")
    if not value:
        raise ValueError(f"Cannot construct a filename from model ID: {model!r}")
    return f"{value}.jsonl"


def parse_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def validate_prediction(value: dict[str, Any]) -> dict[str, str]:
    action_type = str(value.get("action_type") or "").strip()
    action = str(value.get("action") or "").strip()
    expected_information = str(value.get("expected_information") or "").strip()
    if action_type not in ACTION_TYPES:
        raise ValueError(f"Unknown action_type: {action_type!r}")
    if not action:
        raise ValueError("action must be a non-empty string")
    if not expected_information:
        raise ValueError("expected_information must be a non-empty string")
    return {
        "action_type": action_type,
        "action": action,
        "expected_information": expected_information,
    }


def source_category(row: dict[str, Any]) -> str:
    source = row.get("source") if isinstance(row.get("source"), dict) else row
    values = " ".join(
        str(source.get(key) or "") for key in ("source_type", "construction_method")
    ).lower()
    if "real_gap" in values and "synthetic" not in values:
        return "real"
    if "synthetic" in values:
        return "synthetic"
    return "mixed_or_unknown"


def selected_models(spec: str) -> list[str]:
    if spec == "all":
        return list(MODELS)
    if spec == "frontier":
        return [m for m, c in MODELS.items() if c["group"] == "frontier_proprietary"]
    if spec == "open_weight":
        return [m for m, c in MODELS.items() if c["group"] == "open_weight"]
    values = [value.strip() for value in spec.split(",") if value.strip()]
    if not values:
        raise ValueError("No models selected")
    return values


def model_group(model: str) -> str:
    if model in MODELS:
        return MODELS[model]["group"]
    return OPTIONAL_MODEL_GROUPS.get(model, "additional_model")


def fetch_openrouter_catalog(timeout: int = 30) -> dict[str, dict[str, Any]]:
    request = urllib.request.Request(
        OPENROUTER_MODELS_URL,
        headers={"User-Agent": "IdeaAmbig-Task3-Evaluation/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return {row["id"]: row for row in payload.get("data", []) if "id" in row}
