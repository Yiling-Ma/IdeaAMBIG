from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any


TASK1_DIR = Path(__file__).resolve().parent
DEFAULT_POOL = TASK1_DIR / "evaluation_pool.jsonl"
DEFAULT_GOLD = TASK1_DIR / "evaluation_gold.jsonl"
DEFAULT_PREDICTIONS_DIR = TASK1_DIR / "outputs" / "predictions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

MODELS: dict[str, dict[str, Any]] = {
    "baseline/always_ready": {
        "group": "baseline",
        "name": "Always Ready",
    },
    "baseline/always_not_ready": {
        "group": "baseline",
        "name": "Always Not Ready",
    },
    "baseline/random_empirical": {
        "group": "baseline",
        "name": "Random Guess",
    },
    "openai/gpt-5.6-sol": {
        "group": "frontier_proprietary",
    },
    "anthropic/claude-sonnet-5": {
        "group": "frontier_proprietary",
    },
    "google/gemini-3.1-pro-preview": {
        "group": "frontier_proprietary",
    },
    "deepseek/deepseek-v3.2": {
        "group": "frontier_proprietary",
    },
    "qwen/qwen3-8b": {
        "group": "open_weight",
    },
    "qwen/qwen3.5-9b": {
        "group": "open_weight",
    },
    "qwen/qwen3.5-397b-a17b": {
        "group": "open_weight",
    },
    "qwen/qwen3-32b": {
        "group": "open_weight",
    },
    "deepseek/deepseek-r1-0528": {
        "group": "open_weight",
    },
    "openai/gpt-oss-120b": {
        "group": "open_weight",
    },
    "z-ai/glm-5.2": {
        "group": "open_weight",
    },
    "google/gemma-4-31b-it": {
        "group": "open_weight",
    },
    "moonshotai/kimi-k3": {
        "group": "open_weight",
    },
}

MODEL_ALIASES: dict[str, str] = {
    "qwen/qwen3.5-9b": "qwen/qwen3.5-9b",
    "Qwen/Qwen3.5-9B".lower(): "qwen/qwen3.5-9b",
    "qwen3.5-9b": "qwen/qwen3.5-9b",
    "qwen/qwen3.5-397b-a17b": "qwen/qwen3.5-397b-a17b",
    "Qwen/Qwen3.5-397B-A17B".lower(): "qwen/qwen3.5-397b-a17b",
    "qwen3.5-397b-a17b": "qwen/qwen3.5-397b-a17b",
    "deepseek/deepseek-r1-0528": "deepseek/deepseek-r1-0528",
    "DeepSeek-R1-0528".lower(): "deepseek/deepseek-r1-0528",
    "deepseek-r1-0528": "deepseek/deepseek-r1-0528",
    "openai/gpt-oss-120b": "openai/gpt-oss-120b",
    "GPT-OSS-120B".lower(): "openai/gpt-oss-120b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "z-ai/glm-5.2": "z-ai/glm-5.2",
    "GLM-5.2".lower(): "z-ai/glm-5.2",
    "glm-5.2": "z-ai/glm-5.2",
    "google/gemma-4-31b-it": "google/gemma-4-31b-it",
    "Gemma-4-31B".lower(): "google/gemma-4-31b-it",
    "gemma-4-31b": "google/gemma-4-31b-it",
    "moonshotai/kimi-k3": "moonshotai/kimi-k3",
    "Kimi-K3".lower(): "moonshotai/kimi-k3",
    "kimi-k3": "moonshotai/kimi-k3",
}

SYSTEM_PROMPT_V1 = (
    "You are an expert scientific-method reviewer. Classify the supplied research "
    "idea specification using the operational definition provided. Do not invent "
    "missing details. Return exactly one option letter and nothing else."
)

SYSTEM_PROMPT_V2 = (
    "You are an expert scientific-method reviewer. Judge only the supplied "
    "specification. Do not consult or rely on the original paper, source code, "
    "or external implementation details, and do not invent missing information. "
    "Return exactly one option letter and nothing else."
)

SYSTEM_PROMPT_V3 = (
    "You are an expert scientific-method reviewer. Judge only the supplied "
    "specification using the operational definition provided. Do not rely on "
    "external knowledge about the source paper or implementation, and do not "
    "invent missing details. Follow the required output format exactly."
)

PROMPT_VERSIONS = ("v1", "v2", "v3")


def option_mapping(case_id: str, seed: int) -> dict[str, str]:
    """Counterbalance label-to-token mapping to reduce fixed option-token bias."""
    parity = hashlib.sha256(f"{seed}\0{case_id}".encode("utf-8")).digest()[0] % 2
    if parity == 0:
        return {"A": "READY", "B": "NOT_READY"}
    return {"A": "NOT_READY", "B": "READY"}


def system_prompt(prompt_version: str) -> str:
    if prompt_version == "v1":
        return SYSTEM_PROMPT_V1
    if prompt_version == "v2":
        return SYSTEM_PROMPT_V2
    if prompt_version == "v3":
        return SYSTEM_PROMPT_V3
    raise ValueError(f"Unknown prompt version: {prompt_version}")


def build_user_prompt(
    input_text: str,
    mapping: dict[str, str],
    prompt_version: str = "v1",
) -> str:
    if prompt_version == "v1":
        return f"""Task: Codification Readiness Assessment

A research idea specification is READY if an implementer can build a faithful
initial implementation or experimental prototype without inventing unsupported
assumptions about the core method. It need not fix every ordinary hyperparameter
or open design choice. It is NOT_READY when an omission, ambiguity, or internal
inconsistency leaves an implementation-critical choice underdetermined.

Option A: {mapping['A']}
Option B: {mapping['B']}

Specification:
<specification>
{input_text}
</specification>

Return only A or B."""
    if prompt_version == "v2":
        return f"""Task: Readiness Assessment

Determine whether the following research idea specification is ready for a
faithful initial implementation or experimental prototype.

An implementer may use standard machine-learning libraries, canonical
implementations of named standard components, and ordinary engineering
defaults.

A specification is READY if the core method can be implemented without
inventing an unsupported assumption. It does not need to specify every ordinary
hyperparameter, engineering detail, random seed, software version, or explicitly
open modular choice.

A specification is NOT_READY if an omission, ambiguity, or internal
inconsistency leaves an implementation-critical choice underdetermined. Such a
choice may concern the task or inputs and outputs, core algorithm, architecture,
objective or supervision, training procedure, data construction, inference or
decision rule, evaluation protocol, or consistency between these components.

Use the following decision boundary:
- Exact numerical reproduction is not required.
- Missing ordinary engineering details do not make a specification NOT_READY.
- An explicitly open modular choice does not make it NOT_READY when the
  specification clearly allows multiple alternatives.
- A missing evaluation detail is blocking only if it prevents a faithful
  assessment of the central research claim.
- If two competent implementers could follow the specification and produce
  materially different versions of the core method, with no evidence indicating
  which version is intended, classify it as NOT_READY.

Option A: {mapping['A']}
Option B: {mapping['B']}

Specification:
<specification>
{input_text}
</specification>

Return only A or B."""
    if prompt_version != "v3":
        raise ValueError(f"Unknown prompt version: {prompt_version}")

    return f"""Task: Readiness Assessment

Determine whether the following research idea specification is ready for a
faithful initial implementation or experimental prototype.

A specification is READY if the core method can be implemented faithfully
without inventing an unsupported assumption. It does not need to specify every
ordinary hyperparameter, routine engineering detail, software version, random
seed, or clearly open modular choice.

A specification is NOT_READY if an omission, ambiguity, or inconsistency leaves
an implementation-critical choice underdetermined. This can involve the task
setup, core algorithm, architecture, objective, training procedure, data
construction, inference rule, or evaluation protocol.

Use this boundary:
- Missing routine engineering details are acceptable.
- Exact numerical reproducibility is not required.
- An explicitly open modular choice is acceptable if the intended method
  remains clear.
- A missing detail is blocking only when two competent implementers could make
  materially different core-method decisions with no evidence about which one is
  intended.

Option A: {mapping['A']}
Option B: {mapping['B']}

Specification:
<specification>
{input_text}
</specification>

Output format:
First line: A or B
Second line: Reason: <one brief sentence explaining the main blocking issue, or
why the specification is implementable>

Do not output anything else."""


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
    """Load one JSONL file or recursively combine JSONL files in a directory."""
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


def source_category(gold: dict[str, Any]) -> str:
    """Map private provenance metadata to real/synthetic without exposing it."""
    source = gold.get("source") if isinstance(gold.get("source"), dict) else {}
    values: list[str] = []
    for key in ("source_type", "construction_method"):
        if source.get(key):
            values.append(str(source[key]))
    for key in ("source_types", "construction_methods"):
        if isinstance(source.get(key), list):
            values.extend(str(value) for value in source[key])
    lowered = " ".join(values).lower()
    has_real = "real_gap" in lowered
    has_synthetic = "synthetic" in lowered
    if has_real and not has_synthetic:
        return "real"
    if has_synthetic and not has_real:
        return "synthetic"
    return "mixed_or_unknown"


def fetch_openrouter_catalog(timeout: int = 30) -> dict[str, dict[str, Any]]:
    request = urllib.request.Request(
        OPENROUTER_MODELS_URL,
        headers={"User-Agent": "IdeaAmbig-Task1-Evaluation/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return {row["id"]: row for row in payload.get("data", []) if "id" in row}


def resolved_model_registry(
    catalog: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    registry = {model_id: dict(config) for model_id, config in MODELS.items()}
    if not catalog:
        return registry
    openrouter_model_ids = {
        model_id
        for model_id, config in registry.items()
        if config.get("group") in {"frontier_proprietary", "open_weight"}
    }
    missing = sorted(openrouter_model_ids - set(catalog))
    if missing:
        raise ValueError(f"Models absent from OpenRouter catalog: {', '.join(missing)}")
    for model_id, config in registry.items():
        if model_id not in openrouter_model_ids:
            continue
        remote = catalog[model_id]
        supported = set(remote.get("supported_parameters") or [])
        config.update(
            {
                "name": remote.get("name", model_id),
                "supported_parameters": sorted(supported),
                "context_length": remote.get("context_length"),
            }
        )
    return registry
