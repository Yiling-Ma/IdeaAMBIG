"""Configuration, taxonomy, prompts, and shared I/O for Task 2 Track 1."""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Any


TRACK_DIR = Path(__file__).resolve().parent
DEFAULT_INSTANCES = TRACK_DIR.parent / "instances.jsonl"
DEFAULT_POOL = TRACK_DIR / "evaluation_pool.jsonl"
DEFAULT_GOLD = TRACK_DIR / "evaluation_gold.jsonl"
DEFAULT_PREDICTIONS = TRACK_DIR / "outputs" / "predictions.jsonl"
DEFAULT_JUDGMENTS = TRACK_DIR / "outputs" / "semantic_judgments.jsonl"
DEFAULT_SCORES_DIR = TRACK_DIR / "outputs" / "scores"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


LEVEL2_TO_LEVEL1 = {
    "ambiguous formal definition": "Ambiguity",
    "ambiguous method behavior": "Ambiguity",
    "missing algorithmic specification": "Incompleteness",
    "missing hyperparameter protocol": "Incompleteness",
    "missing model architecture": "Incompleteness",
    "missing evaluation protocol": "Incompleteness",
    "missing data/preprocessing protocol": "Incompleteness",
    "inconsistent objective or loss": "Inconsistency",
    "inconsistent architecture or pipeline": "Inconsistency",
    "inconsistent model specification": "Inconsistency",
}

LEVEL2_DEFINITIONS = {
    "ambiguous formal definition": (
        "A symbol, notation, mathematical object, or formal rule permits multiple "
        "implementation-relevant interpretations."
    ),
    "ambiguous method behavior": (
        "A method component, training procedure, inference rule, or stage behavior is "
        "described in a way that permits multiple materially different implementations."
    ),
    "missing algorithmic specification": (
        "An implementation-critical algorithm step, objective routing rule, update rule, "
        "training procedure, inference logic, decoding rule, or prompt construction detail "
        "is absent."
    ),
    "missing hyperparameter protocol": (
        "A non-standard or result-sensitive hyperparameter selection, search, adaptation, "
        "or scheduling protocol is absent."
    ),
    "missing model architecture": (
        "A structural choice affecting module composition, information flow, representation "
        "shape, pooling, normalization, dimensional mapping, or layer behavior is absent."
    ),
    "missing evaluation protocol": (
        "The evaluation setup is incomplete, including metric computation, data split, "
        "threshold, aggregation, sample selection, or evaluator configuration."
    ),
    "missing data/preprocessing protocol": (
        "The construction, filtering, labeling, tokenization, normalization, augmentation, "
        "windowing, or assembly of data or inputs is absent."
    ),
    "inconsistent objective or loss": (
        "Two explicit statements or sources prescribe conflicting objectives, loss terms, "
        "loss identities, signs, weights, or optimization targets."
    ),
    "inconsistent architecture or pipeline": (
        "Two explicit statements or sources prescribe conflicting architectures, processing "
        "stages, data flows, layer choices, or pipeline orders."
    ),
    "inconsistent model specification": (
        "Two explicit statements or sources conflict about the formal model, distribution, "
        "assumption, conditioning rule, or marginalization rule."
    ),
}

BOUNDARY_RULES = [
    "An absent loss identity, loss routing rule, training objective detail, inference rule, "
    "or update logic is missing algorithmic specification unless an explicit contradiction exists.",
    "A missing structural choice that changes model capacity, information flow, or representation "
    "shape is missing model architecture.",
    "Missing filtering, label construction, tokenization, normalization, augmentation, windowing, "
    "or data assembly is missing data/preprocessing protocol.",
    "Missing metric computation, split, threshold, evaluation sample selection, aggregation, or "
    "evaluation-model configuration is missing evaluation protocol.",
    "Use an inconsistency label only when the supplied specification contains explicit conflicting "
    "claims; a merely absent detail is an incompleteness.",
]

EXCLUSION_RULES = [
    "Do not flag ordinary choices such as batch size, learning rate, epoch count, random seed, "
    "hardware, runtime, local paths, package setup, or credentials unless they define a non-standard "
    "method-critical mechanism.",
    "Do not flag unavailable compute or data, pure performance mismatch, stylistic weakness, or a "
    "legitimate open design choice as the target defect.",
    "Do not invent the missing value or resolution. Diagnose what is underdetermined."
]

MODELS: dict[str, dict[str, str]] = {
    "qwen/qwen3-8b": {"group": "open_weight"},
    "qwen/qwen3.5-9b": {"group": "open_weight"},
    "qwen/qwen3.5-397b-a17b": {"group": "open_weight"},
    "deepseek/deepseek-r1-0528": {"group": "open_weight"},
    "openai/gpt-oss-120b": {"group": "open_weight"},
    "z-ai/glm-5.2": {"group": "open_weight"},
    "google/gemma-4-31b-it": {"group": "open_weight"},
    "moonshotai/kimi-k3": {"group": "open_weight"},
}

# Extra catalog models may be requested explicitly without changing the frozen
# model suite selected by --models all.
OPTIONAL_MODEL_GROUPS = {
    "google/gemini-3.1-pro-preview": "frontier_proprietary",
    "anthropic/claude-sonnet-5": "frontier_proprietary",
    "openai/gpt-5.6-sol": "frontier_proprietary",
    "deepseek/deepseek-v4-pro": "frontier_proprietary",
    "qwen/qwen3-32b": "open_weight",
    "deepseek/deepseek-r1": "open_weight",
}

SYSTEM_PROMPT = (
    "You are an expert scientific-method reviewer performing controlled defect "
    "localization. Follow the supplied IDEAAMBIG taxonomy. Do not repair the idea, "
    "invent missing details, or list secondary concerns. Return strict JSON only."
)

PROMPT_VERSIONS = ("v3",)


def taxonomy_text() -> str:
    blocks: list[str] = []
    for level1 in ("Ambiguity", "Incompleteness", "Inconsistency"):
        blocks.append(f"Level-1: {level1}")
        for level2, parent in LEVEL2_TO_LEVEL1.items():
            if parent == level1:
                blocks.append(f"- {level2}: {LEVEL2_DEFINITIONS[level2]}")
    blocks.append("Boundary rules:")
    blocks.extend(f"- {rule}" for rule in BOUNDARY_RULES)
    blocks.append("Exclusions:")
    blocks.extend(f"- {rule}" for rule in EXCLUSION_RULES)
    return "\n".join(blocks)


def build_user_prompt_v3(input_text: str) -> str:
    return f"""You are an expert scientific-method reviewer performing defect taxonomy
classification for research idea specifications.

Your task is to identify and classify the single implementation-critical
specification defect in the given research idea according to the IDEAAMBIG
taxonomy.

Each specification in this benchmark is constructed with exactly one annotated
target defect. Your goal is not to find all possible weaknesses. Your goal is
to recover the single target defect category that best explains why the
specification is not implementation-ready.

If multiple possible issues are visible, internally rank them and select only
the strongest defect that is:
- directly grounded in a specific statement, term, formula, component, or
  transition in the specification;
- necessary for faithful implementation or reproduction;
- not an ordinary default, minor missing detail, optional engineering choice,
  documentation improvement, or speculative concern;
- the most specific supported defect rather than a broad lack-of-detail claim.

Ambiguity/Incompleteness boundary:
Do not decide Ambiguity vs Incompleteness by whether implementation is unclear;
both categories make implementation unclear. Use Ambiguity only when the
specification states a concrete term, rule, behavior, configuration, or formula
whose stated content supports two or more plausible interpretations. Use
Incompleteness when the required algorithm, protocol, architecture,
hyperparameter selection rule, data process, or evaluation setup is not provided
at all.

Prefer the defect tied to the most specific unusual term, configuration, or
implementation decision in the specification, rather than broad missing details
about the overall architecture, training objective, or hyperparameters.

An implementation-critical defect is an omission, ambiguity, or internal
inconsistency that prevents a faithful initial implementation or experimental
prototype without an unsupported assumption about the core method.

The taxonomy is:

{taxonomy_text()}

Additional ambiguity boundary:
- Use ambiguous formal definition when the uncertainty concerns the meaning,
  value, counting convention, mathematical interpretation, or scope of a term,
  symbol, quantity, equation, set, or formally defined object.
- Use ambiguous method behavior when the uncertainty concerns what an
  algorithm, component, training stage, inference stage, or processing step
  operationally does.

Rules:
1. Select exactly one Level-1 category and one Level-2 category.
2. Do not output multiple defects or alternative labels.
3. Ignore ordinary implementation choices, standard defaults, minor missing
   details, and speculative concerns.
4. Prefer the most specific defect directly supported by the specification.
5. Do not repair the idea or propose solutions.
6. The description must identify the concrete missing, ambiguous, or
   conflicting implementation detail; repeating only a taxonomy label is not
   enough.

Specification:
<specification>
{input_text}
</specification>

Return only valid JSON:
{{
  "description": "one concrete atomic defect diagnosis",
  "level1": "Ambiguity|Incompleteness|Inconsistency",
  "level2": "one allowed Level-2 label"
}}"""


def build_user_prompt(input_text: str, prompt_version: str = "v3") -> str:
    if prompt_version == "v3":
        return build_user_prompt_v3(input_text)
    raise ValueError(
        f"Unknown prompt version {prompt_version!r}; expected one of {PROMPT_VERSIONS}"
    )


PREDICTION_JSON_SCHEMA: dict[str, Any] = {
    "name": "ideaambig_track1_diagnosis",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["description", "level1", "level2"],
        "properties": {
            "description": {"type": "string", "minLength": 1},
            "level1": {"type": "string", "enum": sorted(set(LEVEL2_TO_LEVEL1.values()))},
            "level2": {"type": "string", "enum": list(LEVEL2_TO_LEVEL1)},
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
    description = str(value.get("description") or "").strip()
    level1 = str(value.get("level1") or "").strip()
    level2 = str(value.get("level2") or "").strip()
    if not description:
        raise ValueError("description must be a non-empty string")
    if level1 not in set(LEVEL2_TO_LEVEL1.values()):
        raise ValueError(f"Unknown Level-1 label: {level1!r}")
    if level2 not in LEVEL2_TO_LEVEL1:
        raise ValueError(f"Unknown Level-2 label: {level2!r}")
    return {"description": description, "level1": level1, "level2": level2}


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
        headers={"User-Agent": "IdeaAmbig-Task2-Track1/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return {row["id"]: row for row in payload.get("data", []) if "id" in row}
