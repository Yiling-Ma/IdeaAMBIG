from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from common import (
    LLMJsonClient,
    add_common_args,
    ensure_dir,
    load_jsonl,
    save_json,
    save_jsonl,
)


SPEC_FIELDS = [
    "paper_title",
    "research_goal",
    "task",
    "inputs",
    "outputs",
    "core_method",
    "algorithm_steps",
    "training_or_optimization",
    "datasets",
    "evaluation_metrics",
    "baselines",
    "implementation_details",
    "reproducibility_relevant_details",
    "unknown_fields",
]

LIST_FIELDS = {
    "algorithm_steps",
    "datasets",
    "evaluation_metrics",
    "baselines",
    "implementation_details",
    "reproducibility_relevant_details",
    "unknown_fields",
}

RESULT_CLAIM_PATTERNS = [
    r"\byields?\b.*\b(gain|improvement|higher|better|substantial|significant)",
    r"\bachieves?\b.*\b(state-of-the-art|best|higher|better)",
    r"\boutperforms?\b",
    r"\bresults?\s+(show|indicate|demonstrate)",
]

METADATA_PHRASES_IN_REFERENCE = [
    "the repository provides",
    "the code is available",
    "available in the code",
    "not fully reproduced here",
    "requires an api key",
    "api key",
    "secrets file",
    "local path",
    "slurm",
    "gpu",
    "cuda",
]

UNSUPPORTED_TOOL_CLAIM_PATTERNS = [
    r"\bexternal retrieval\b",
    r"\bweb search\b",
    r"\bsearch engine\b",
    r"\bhuman evaluation\b",
    r"\bstatistical significance\b",
    r"\bablation study\b",
]

GOLD_CODIFICATION_READY_REFERENCE_DEFINITION = """
Gold codification-ready reference:
The codification_ready_reference is the evidence-grounded, defect-free implementation
specification for the target research idea or method. It is not a paper summary, not
the original idea, and not a list of all reproducibility trivia. It is the clarified
method specification that an implementer would need in order to faithfully codify the
core method.

It should include implementation-facing details when supported by the executed paper
or codebase, such as:
- task definition;
- concrete inputs and outputs;
- core algorithmic or prompting procedure;
- model, module, architecture, or representation structure;
- training, optimization, loss, inference, routing, decoding, or parsing logic;
- data construction, preprocessing, filtering, sampling, or input formatting;
- evaluation protocol, metric computation, baselines, and comparison setup;
- any target clarification detail needed to resolve a known ambiguity, omission, or
  inconsistency.

It should exclude:
- unsupported assumptions or guessed details;
- broad motivation, background, or contribution summaries;
- result claims such as "outperforms", "achieves better", or "results show";
- unknown fields, caveats, limitations, or unresolved contradictions;
- benchmark-construction metadata or synthetic-defect language;
- API keys, credentials, local paths, hardware-only details, runtime-only details,
  and environment setup unless they are themselves part of the scientific method.

Ordinary hyperparameters such as batch size, learning rate, epoch count, random seed,
or hardware should appear only when they encode a non-standard method mechanism or are
required to understand the implementation. Otherwise, keep them in
reproducibility_relevant_details, not in codification_ready_reference.
""".strip()


# ============================================================
# Basic text utilities
# ============================================================

def normalize_ws(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def word_count(text: str) -> int:
    return len(normalize_ws(text).split())


def truncate(text: str, max_chars: int) -> str:
    text = str(text or "")
    return text if len(text) <= max_chars else text[:max_chars]


def safe_regex_search(pattern: str, text: str) -> bool:
    try:
        return re.search(pattern, text, flags=re.I) is not None
    except re.error:
        return False


def contains_any_pattern(text: str, patterns: List[str]) -> bool:
    return any(safe_regex_search(p, text) for p in patterns)


def remove_references(text: str) -> str:
    if not text:
        return ""
    pattern = re.compile(r"(?im)^\s*(references|bibliography|works cited)\s*$")
    match = pattern.search(text)
    return text[: match.start()].strip() if match else text.strip()


def clean_code_summary(text: str) -> str:
    """Remove obvious credential-related snippets from code summary."""
    if not text:
        return ""

    blocked = [
        "api_key=xxx",
        "anthropic_api_key=xxx",
        "openai_api_key",
        "anthropic_api_key",
        "mistral_api_key",
        "replicate_api_key",
        "secrets_file",
        "put your api key",
        "bearer ",
        "x-api-key",
        "password",
        "credential",
    ]

    lines = []
    for line in text.splitlines():
        low = line.lower()
        if any(key in low for key in blocked):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def select_project_context(
    row: Dict[str, Any],
    max_idea_chars: int,
    max_paper_chars: int,
    max_code_chars: int,
    max_review_chars: int,
) -> Dict[str, str]:
    edited_idea = truncate(row.get("edited_idea") or "", max_idea_chars)
    original_idea = truncate(row.get("original_idea") or "", max_idea_chars)

    paper_text = remove_references(row.get("executed_paper_text") or "")
    paper_text = truncate(paper_text, max_paper_chars)

    code_summary = clean_code_summary(row.get("codebase_summary_text") or "")
    code_summary = truncate(code_summary, max_code_chars)

    reviews_text = truncate(row.get("reviews_text") or "", max_review_chars)

    return {
        "edited_idea": edited_idea,
        "original_idea": original_idea,
        "executed_paper_text": paper_text,
        "codebase_summary_text": code_summary,
        "reviews_text": reviews_text,
    }


# ============================================================
# Normalization
# ============================================================

def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = normalize_ws(value)
        return [value] if value else []
    return [value]


def normalize_string(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return normalize_ws(json.dumps(value, ensure_ascii=False))
    return normalize_ws(value)


def normalize_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(spec, dict):
        spec = {}

    for field in SPEC_FIELDS:
        if field in LIST_FIELDS:
            items = as_list(spec.get(field))
            cleaned = []
            for item in items:
                if isinstance(item, dict):
                    item_text = normalize_ws(json.dumps(item, ensure_ascii=False))
                else:
                    item_text = normalize_ws(item)
                if item_text:
                    cleaned.append(item_text)
            out[field] = cleaned
        else:
            out[field] = normalize_string(spec.get(field))
    return out


def normalize_gold_output(obj: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        obj = {}

    if "paper_derived_specification" in obj:
        paper_spec = obj.get("paper_derived_specification") or {}
    else:
        paper_spec = obj

    paper_spec = normalize_spec(paper_spec)
    reference = normalize_string(obj.get("codification_ready_reference"))

    if not paper_spec.get("paper_title"):
        paper_spec["paper_title"] = normalize_string(
            row.get("metadata", {}).get("matched_ideation_title") or row.get("project_id")
        )

    return {
        "project_id": row["project_id"],
        "split_group_id": row["split_group_id"],
        "source": row.get("source_urls", {}),
        "paper_derived_specification": paper_spec,
        "codification_ready_reference": reference,
    }


# ============================================================
# Quality checks
# ============================================================

def compute_completeness(spec: Dict[str, Any]) -> Dict[str, bool]:
    return {
        "has_task": bool(spec.get("task")),
        "has_inputs": bool(spec.get("inputs")),
        "has_outputs": bool(spec.get("outputs")),
        "has_core_method": bool(spec.get("core_method")),
        "has_algorithm_steps": bool(spec.get("algorithm_steps")),
        "has_training_or_optimization": bool(spec.get("training_or_optimization")),
        "has_datasets": bool(spec.get("datasets")),
        "has_evaluation_metrics": bool(spec.get("evaluation_metrics")),
        "has_baselines": bool(spec.get("baselines")),
        "has_implementation_details": bool(spec.get("implementation_details")),
    }


def audit_gold_spec(gold: Dict[str, Any]) -> List[str]:
    spec = gold.get("paper_derived_specification") or {}
    reference = gold.get("codification_ready_reference") or ""
    ref_low = reference.lower()
    warnings: List[str] = []

    unknown_fields = spec.get("unknown_fields") or []
    if isinstance(unknown_fields, list) and len(unknown_fields) >= 5:
        warnings.append("many_unknown_fields")

    if contains_any_pattern(ref_low, RESULT_CLAIM_PATTERNS):
        warnings.append("reference_contains_result_claim")

    if any(phrase in ref_low for phrase in METADATA_PHRASES_IN_REFERENCE):
        warnings.append("reference_contains_metadata_or_environment_phrase")

    if contains_any_pattern(ref_low, UNSUPPORTED_TOOL_CLAIM_PATTERNS):
        warnings.append("possible_unsupported_tool_or_evaluation_claim")

    if "unknown_fields" in ref_low or "not specified" in ref_low or "not fully" in ref_low:
        warnings.append("reference_contains_limitation_or_unknown_field_text")

    # Detect list-like references that look like concatenated schema fields
    # instead of a coherent research-idea paragraph.
    field_label_hits = 0
    for pat in [
        r"\btitle:\b",
        r"\bresearch overview:\b",
        r"\bmethod:\b",
        r"\bevaluation plan:\b",
        r"\bimplementation notes:\b",
        r"\binputs:\b",
        r"\boutputs:\b",
        r"\balgorithm steps:\b",
        r"\bdatasets:\b",
        r"\bevaluation metrics:\b",
        r"\bbaselines:\b",
    ]:
        if safe_regex_search(pat, reference):
            field_label_hits += 1
    if field_label_hits >= 3:
        warnings.append("reference_field_list_style")

    return sorted(set(warnings))


def assess_gold_quality(gold: Dict[str, Any], row: Dict[str, Any]) -> Tuple[str, List[str], Dict[str, bool]]:
    spec = gold.get("paper_derived_specification") or {}
    reference = gold.get("codification_ready_reference") or ""
    completeness = compute_completeness(spec)
    warnings: List[str] = []

    if not reference:
        warnings.append("empty_codification_ready_reference")
    elif word_count(reference) < 100:
        warnings.append("codification_ready_reference_too_short")
    elif word_count(reference) > 430:
        warnings.append("codification_ready_reference_too_long")

    for key, ok in completeness.items():
        if not ok:
            warnings.append(f"missing_{key.replace('has_', '')}")

    modalities = row.get("available_modalities") or {}
    has_paper = bool(modalities.get("has_executed_paper"))
    has_code = bool(modalities.get("has_codebase"))

    if not has_paper:
        warnings.append("missing_executed_paper_evidence")
    if not has_code:
        warnings.append("missing_codebase_evidence")

    warnings.extend(audit_gold_spec(gold))

    required_ok = (
        completeness["has_task"]
        and completeness["has_core_method"]
        and (completeness["has_algorithm_steps"] or completeness["has_implementation_details"])
        and bool(reference)
        and word_count(reference) >= 80
    )

    high_ok = (
        required_ok
        and completeness["has_inputs"]
        and completeness["has_outputs"]
        and completeness["has_training_or_optimization"]
        and completeness["has_datasets"]
        and completeness["has_evaluation_metrics"]
        and has_paper
        and has_code
    )

    if high_ok:
        source_quality = "high"
    elif required_ok:
        source_quality = "medium"
    else:
        source_quality = "low"

    return source_quality, sorted(set(warnings)), completeness


def build_gold_metadata(
    gold: Dict[str, Any],
    row: Dict[str, Any],
    model: str,
    use_llm: bool,
    source_quality: str,
    warnings: List[str],
    completeness: Dict[str, bool],
    selected_context_path: str,
) -> Dict[str, Any]:
    metadata = row.get("metadata") or {}
    modalities = row.get("available_modalities") or {}
    spec = gold.get("paper_derived_specification") or {}

    limitations = []
    for field in spec.get("unknown_fields", []):
        if field:
            limitations.append(str(field))

    for warning in warnings:
        if warning.startswith("missing_"):
            limitations.append(warning)

    return {
        "project_id": row["project_id"],
        "split_group_id": row["split_group_id"],
        "extraction_method": "llm_from_executed_paper_code" if use_llm else "heuristic_fallback",
        "model": model if use_llm else None,
        "source_quality": source_quality,
        "has_code_evidence": bool(modalities.get("has_codebase")),
        "has_paper_evidence": bool(modalities.get("has_executed_paper")),
        "gold_spec_completeness": completeness,
        "limitations": sorted(set(limitations)),
        "quality_warnings": warnings,
        "source_paths": {
            "executed_paper_path": row.get("executed_paper_path", ""),
            "codebase_path": row.get("codebase_path", ""),
            "idea_file": metadata.get("idea_file", ""),
            "paper_file": metadata.get("paper_file", ""),
            "code_zip": metadata.get("code_zip", ""),
        },
        "available_modalities": modalities,
        "selected_context_path": selected_context_path,
        "idea_source": row.get("idea_source", "unknown"),
        "complete_triple": bool(metadata.get("complete_triple")),
    }


def is_usable_gold(gold: Dict[str, Any]) -> Tuple[bool, str]:
    meta = gold.get("gold_extraction_metadata") or {}
    quality = meta.get("source_quality", "low")
    completeness = meta.get("gold_spec_completeness") or {}

    if quality == "low":
        return False, "low_source_quality"
    if not completeness.get("has_core_method"):
        return False, "missing_core_method"
    if not (completeness.get("has_algorithm_steps") or completeness.get("has_implementation_details")):
        return False, "missing_algorithm_steps_or_implementation_details"
    if word_count(gold.get("codification_ready_reference", "")) < 80:
        return False, "reference_too_short"
    return True, "usable"


# ============================================================
# Heuristic fallback
# ============================================================

def split_sentences(text: str, max_items: int = 12) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text))
    return [p.strip() for p in parts if 30 <= len(p.strip()) <= 500][:max_items]


def labeled_section(text: str, label: str, max_chars: int = 5000) -> str:
    pattern = (
        rf"(?:\b\d+\.\s*)?{re.escape(label)}\s*:\s*(.+?)"
        rf"(?=\s+\d+\.\s+[A-Z][A-Za-z -]{{3,40}}\s*:|"
        rf"\s+(?:Problem Statement|Motivation|Proposed Method|Step-by-Step Experiment Plan|Evaluation|Expected Outcomes|Implementation Details)\s*:|$)"
    )
    match = re.search(pattern, text, re.I | re.S)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()[:max_chars]


def numbered_steps(text: str, max_items: int = 10) -> List[str]:
    matches = re.findall(
        r"(?:Step\s*\d+|(?:\d+|[a-z])\))\s*[:.-]?\s*(.+?)(?=(?:\s+Step\s*\d+|\s+(?:\d+|[a-z])\)\s)|$)",
        text,
        re.I | re.S,
    )
    steps = [re.sub(r"\s+", " ", m).strip() for m in matches if len(re.sub(r"\s+", " ", m).strip()) > 25]
    return steps[:max_items]


def heuristic_gold(row: Dict[str, Any]) -> Dict[str, Any]:
    paper = row.get("executed_paper_text") or ""
    code = row.get("codebase_summary_text") or ""
    edited = row.get("edited_idea") or ""

    title = ""
    for line in paper.splitlines()[:40]:
        clean = line.strip(" #\t")
        if 8 <= len(clean) <= 180 and not clean.lower().startswith(("abstract", "introduction")):
            title = clean
            break

    problem = labeled_section(edited, "Problem Statement", 2500)
    motivation = labeled_section(edited, "Motivation", 2500)
    proposed = labeled_section(edited, "Proposed Method", 7000)
    plan = labeled_section(edited, "Step-by-Step Experiment Plan", 8000)

    abstract_or_intro = paper[:4000]
    method_text = proposed or abstract_or_intro
    steps = numbered_steps(plan, max_items=10) or split_sentences(method_text, max_items=8)

    datasets = sorted(
        set(
            re.findall(
                r"\b(?:TruthfulQA|ScienceQA|CIFAR-10|CIFAR100|ImageNet|MNIST|MMLU|GSM8K|HumanEval|MBPP|SQuAD|Natural Questions|HotpotQA|SimpleQA|Global MMLU|ConflictingQA|Personae Dataset)\b",
                paper + "\n" + edited + "\n" + code,
                re.I,
            )
        )
    )

    metrics = sorted(
        set(
            re.findall(
                r"\b(?:accuracy|exact match|ECE|expected calibration error|AUC|F1|BLEU|ROUGE|precision|recall|perplexity|pass@k|win rate|calibration|entropy|KL divergence)\b",
                paper + "\n" + edited + "\n" + code,
                re.I,
            )
        )
    )

    implementation_details = split_sentences(code, max_items=12)

    spec = {
        "paper_title": title or row.get("project_id"),
        "research_goal": (
            split_sentences(motivation or abstract_or_intro or edited, 1)[0]
            if split_sentences(motivation or abstract_or_intro or edited, 1)
            else ""
        ),
        "task": problem or "",
        "inputs": "",
        "outputs": "",
        "core_method": proposed or " ".join(split_sentences(method_text, 3)),
        "algorithm_steps": steps,
        "training_or_optimization": " ".join(split_sentences(paper, 4)),
        "datasets": datasets,
        "evaluation_metrics": metrics,
        "baselines": [],
        "implementation_details": implementation_details,
        "reproducibility_relevant_details": split_sentences(paper + "\n" + code, 8),
        "unknown_fields": [],
    }

    for field in ["task", "inputs", "outputs", "core_method"]:
        if not spec.get(field):
            spec["unknown_fields"].append(field)

    reference_parts = [
        spec["research_goal"],
        spec["task"],
        spec["core_method"],
        " ".join(spec["algorithm_steps"]),
        spec["training_or_optimization"],
        "Datasets: " + ", ".join(spec["datasets"]) if spec["datasets"] else "",
        "Evaluation metrics: " + ", ".join(spec["evaluation_metrics"]) if spec["evaluation_metrics"] else "",
        "Implementation details: " + " ".join(spec["implementation_details"][:5]) if spec["implementation_details"] else "",
    ]

    return {
        "project_id": row["project_id"],
        "split_group_id": row["split_group_id"],
        "source": row.get("source_urls", {}),
        "paper_derived_specification": normalize_spec(spec),
        "codification_ready_reference": normalize_ws(" ".join(p for p in reference_parts if p)),
    }


# ============================================================
# LLM extraction
# ============================================================

def build_llm_prompt(row: Dict[str, Any], context: Dict[str, str]) -> Tuple[str, str]:
    system = """
You extract faithful codification-ready specifications from executed research projects.

Return strict JSON only. Do not include markdown fences. Do not hallucinate.
Use only information supported by the executed paper and codebase summary.
If a field is missing or unsupported, leave it empty and list it in unknown_fields.
Do not turn speculative ideas, future work, limitations, or analysis suggestions into implemented method details.
""".strip()

    user_obj = {
        "benchmark_context": {
            "project": "IDEAAMBIG",
            "goal": (
                "Build synthetic-controlled benchmark instances for evaluating whether models can detect "
                "implementation-blocking underspecification in research ideas."
            ),
            "this_step": (
                "Extract a clean codification-ready gold specification from an executed project. Later steps "
                "will use this gold spec to generate single-defect candidates. Do not create defects or taxonomy labels in this step."
            ),
            "codification_ready_definition": GOLD_CODIFICATION_READY_REFERENCE_DEFINITION,
        },
        "evidence_priority": [
            "Primary evidence: executed_paper_text and codebase_summary_text.",
            "Secondary context: edited_idea and original_idea. Use them only to understand motivation, not to fill unsupported implementation details.",
            "Do not use reviews as gold evidence. Reviews are optional context only.",
            "If paper and code disagree, prefer implemented behavior in code and record the discrepancy in unknown_fields or reproducibility_relevant_details.",
        ],
        "strict_rules": [
            "Do not invent architecture, algorithm, training, dataset, inference, prompting, or evaluation details.",
            "Do not include a component unless the executed paper or codebase supports that it was actually used.",
            "Do not convert future work, fallback plans, limitations, error analysis suggestions, or speculative improvements into method details.",
            "Do not include external tools, retrieval systems, web search, human evaluation, ablations, or statistical tests unless the executed paper/code clearly implemented them.",
            "Do not include API keys, secrets, credential setup, organization IDs, local environment paths, hardware-only details, or rate-limit utilities as method details.",
            "Do not include benchmark-construction language, synthetic-defect language, route metadata, or annotation language in codification_ready_reference.",
            "Do not write phrases such as 'the repository provides', 'the code is available', 'not fully reproduced here', or 'available in the code' inside codification_ready_reference.",
            "Do not include unknown_fields, limitations, missing details, or caveats inside codification_ready_reference.",
            "Keep codification_ready_reference as one natural-language paragraph.",
            "Write codification_ready_reference as a coherent research idea / method specification paragraph, not as a field list or schema dump.",
            "Do not use section-label style such as 'Title:', 'Research Overview:', 'Method:', 'Evaluation Plan:', or 'Implementation Notes:' inside codification_ready_reference.",
            "The reference should be self-contained and implementation-oriented.",
            "The reference must describe how to implement the method, not whether the method worked or how much it improved results.",
            "Include method-critical low-level mechanisms such as language routing, embedding replacement, prompt parsing, logit selection, metric computation, or data filtering when they are required to faithfully implement the method.",
            "Do not move a detail into codification_ready_reference merely because it appears in code; include it only if it is method-relevant and supported by paper/code evidence.",
            "The reference should be 160 to 320 words if enough evidence is available.",
            "The reference may include exact prompts, formulas, thresholds, model names, dataset splits, or metric definitions only when explicitly supported by paper/code.",
            "unknown_fields must list important unsupported or missing details that would matter for faithful implementation or evaluation.",
        ],
        "field_guidance": {
            "paper_title": "Title of the executed paper/project.",
            "research_goal": "One concise sentence describing the research objective.",
            "task": "Concrete task solved by the method.",
            "inputs": "Concrete input objects and input format required by the method.",
            "outputs": "Concrete output objects produced by the method.",
            "core_method": "Main computational or prompting method, excluding background motivation.",
            "algorithm_steps": (
                "Ordered method steps. Preserve stage order, routing logic, formulas, prompt chaining, "
                "training/inference flow, aggregation rules, parsing rules, stopping conditions, and decision rules when supported."
            ),
            "training_or_optimization": (
                "Training, finetuning, optimization, or explicitly state no training if the method is prompting-only or inference-only. "
                "Include objective/loss only when actually used."
            ),
            "datasets": (
                "Datasets, dataset configurations, splits, sampling protocols, constructed data, data fields, "
                "and preprocessing/input-construction rules used in experiments."
            ),
            "evaluation_metrics": (
                "Metrics and metric-computation protocol. Include formulas, binning rules, judge setup, "
                "thresholding, ranking scores, and aggregation rules when supported."
            ),
            "baselines": "Baselines actually evaluated and the comparison setup.",
            "implementation_details": (
                "Method-relevant implementation details such as model component, prompt format, embedding/matrix shape, "
                "logit extraction, parsing behavior, stage count, formula details, or decoding settings. Exclude credentials and local environment noise."
            ),
            "reproducibility_relevant_details": "Details useful for reproducing method/evaluation, excluding credentials, API keys, and local paths.",
            "unknown_fields": "Unsupported or missing implementation-relevant details. Do not include these in the reference.",
            "codification_ready_reference": (
                "One self-contained paragraph that merges the evidence-supported method details into a clean, "
                "defect-free implementation-ready reference. It should resolve implementation-facing details "
                "that would otherwise block codification, while excluding result claims, unsupported assumptions, "
                "metadata, caveats, and ordinary reproducibility trivia."
            ),
        },
        "required_json_schema": {
            "paper_derived_specification": {
                "paper_title": "string",
                "research_goal": "string",
                "task": "string",
                "inputs": "string",
                "outputs": "string",
                "core_method": "string",
                "algorithm_steps": ["string"],
                "training_or_optimization": "string",
                "datasets": ["string"],
                "evaluation_metrics": ["string"],
                "baselines": ["string"],
                "implementation_details": ["string"],
                "reproducibility_relevant_details": ["string"],
                "unknown_fields": ["string"],
            },
            "codification_ready_reference": "string",
            "gold_extraction_metadata": {
                "source_quality": "high|medium|low",
                "limitations": ["string"],
            },
        },
        "project_metadata": {
            "project_id": row.get("project_id"),
            "split_group_id": row.get("split_group_id"),
            "idea_source": row.get("idea_source"),
            "available_modalities": row.get("available_modalities"),
            "source_urls": row.get("source_urls"),
        },
        "edited_idea_secondary_context": context.get("edited_idea", ""),
        "original_idea_secondary_context": context.get("original_idea", ""),
        "executed_paper_text_primary_evidence": context.get("executed_paper_text", ""),
        "codebase_summary_text_primary_evidence": context.get("codebase_summary_text", ""),
        "reviews_text_optional_do_not_use_as_gold": context.get("reviews_text", ""),
    }

    return system, json.dumps(user_obj, ensure_ascii=False, indent=2)


def llm_gold(
    row: Dict[str, Any],
    client: LLMJsonClient,
    out_dir: Path,
    model: str,
    max_idea_chars: int,
    max_paper_chars: int,
    max_code_chars: int,
    max_review_chars: int,
    save_prompt: bool,
    save_selected_context: bool,
) -> Dict[str, Any]:
    project_id = row["project_id"]
    project_dir = out_dir / "per_project" / project_id
    ensure_dir(project_dir)

    context = select_project_context(
        row=row,
        max_idea_chars=max_idea_chars,
        max_paper_chars=max_paper_chars,
        max_code_chars=max_code_chars,
        max_review_chars=max_review_chars,
    )

    selected_context_path = project_dir / "selected_context.json"
    if save_selected_context:
        save_json(context, selected_context_path)

    system, user = build_llm_prompt(row, context)

    if save_prompt:
        save_json({"system": system, "user": user}, project_dir / "prompt.json")

    raw_path = out_dir / "raw_llm" / f"{project_id}.txt"
    current_user = user
    last_gold: Dict[str, Any] | None = None

    for attempt in range(1, 4):
        attempt_raw_path = raw_path.with_name(f"{raw_path.stem}_attempt{attempt}.txt")
        obj = client.call_json(system, current_user, attempt_raw_path)
        gold = normalize_gold_output(obj, row)
        reference = gold.get("codification_ready_reference", "")

        if not safe_regex_search(
            r"\b(title:|research overview:|method:|evaluation plan:|implementation notes:)\b",
            reference,
        ):
            return gold

        last_gold = gold
        current_user = (
            user
            + "\n\nPrevious output issue: codification_ready_reference was formatted like a field list with section labels. "
            + "Rewrite it as ONE coherent natural-language paragraph that reads like a concrete research idea/method specification."
        )

    if last_gold is not None:
        return last_gold
    raise RuntimeError("LLM extraction failed to produce a valid codification_ready_reference.")


def finalize_gold_object(
    gold: Dict[str, Any],
    row: Dict[str, Any],
    model: str,
    use_llm: bool,
    selected_context_path: str,
) -> Dict[str, Any]:
    source_quality, warnings, completeness = assess_gold_quality(gold, row)
    metadata = build_gold_metadata(
        gold=gold,
        row=row,
        model=model,
        use_llm=use_llm,
        source_quality=source_quality,
        warnings=warnings,
        completeness=completeness,
        selected_context_path=selected_context_path,
    )
    gold["gold_extraction_metadata"] = metadata
    gold["quality_label"] = "usable" if source_quality in {"high", "medium"} else "unusable"
    gold["quality_warnings"] = warnings
    return gold


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)

    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--model", default="deepseek/deepseek-v4-pro")
    parser.add_argument("--max_idea_chars", type=int, default=20_000)
    parser.add_argument("--max_paper_chars", type=int, default=75_000)
    parser.add_argument("--max_code_chars", type=int, default=55_000)
    parser.add_argument("--max_review_chars", type=int, default=8_000)
    parser.add_argument("--save_prompt", action="store_true")
    parser.add_argument("--save_selected_context", action="store_true")
    parser.add_argument("--allow_heuristic_fallback", action="store_true")

    args = parser.parse_args()
    args.input_path = Path(args.input_path)
    args.out_dir = Path(args.out_dir)

    ensure_dir(args.out_dir)
    ensure_dir(args.out_dir / "per_project")
    ensure_dir(args.out_dir / "raw_llm")

    rows = load_jsonl(args.input_path)
    if args.limit:
        rows = rows[: args.limit]

    client = LLMJsonClient(args.model, args.use_llm)

    gold_specs: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    index_rows: List[Dict[str, Any]] = []

    for row in tqdm(rows, desc="Step2 extract gold specs"):
        project_id = row.get("project_id", "unknown_project")
        project_dir = args.out_dir / "per_project" / project_id
        ensure_dir(project_dir)

        out_file = project_dir / "gold_spec.json"
        selected_context_path = project_dir / "selected_context.json"

        if args.resume and out_file.exists() and out_file.stat().st_size > 50:
            try:
                gold = json.loads(out_file.read_text(encoding="utf-8"))
                usable, usable_reason = is_usable_gold(gold)
                index_rows.append(
                    {
                        "project_id": project_id,
                        "status": "skipped_exists",
                        "gold_spec_path": str(out_file),
                        "usable": usable,
                        "usable_reason": usable_reason,
                        "source_quality": gold.get("gold_extraction_metadata", {}).get("source_quality"),
                        "quality_warnings": gold.get("quality_warnings", []),
                    }
                )
                if usable:
                    gold_specs.append(gold)
                else:
                    rejected.append(
                        {
                            "project_id": project_id,
                            "reason": usable_reason,
                            "gold_spec_path": str(out_file),
                            "quality_warnings": gold.get("quality_warnings", []),
                        }
                    )
                continue
            except Exception:
                pass

        try:
            if args.use_llm:
                gold = llm_gold(
                    row=row,
                    client=client,
                    out_dir=args.out_dir,
                    model=args.model,
                    max_idea_chars=args.max_idea_chars,
                    max_paper_chars=args.max_paper_chars,
                    max_code_chars=args.max_code_chars,
                    max_review_chars=args.max_review_chars,
                    save_prompt=args.save_prompt,
                    save_selected_context=args.save_selected_context,
                )
            else:
                gold = heuristic_gold(row)

            gold = finalize_gold_object(
                gold=gold,
                row=row,
                model=args.model,
                use_llm=args.use_llm,
                selected_context_path=str(selected_context_path) if args.save_selected_context else "",
            )
            save_json(gold, out_file)

        except Exception as exc:
            if args.allow_heuristic_fallback:
                try:
                    gold = heuristic_gold(row)
                    gold = finalize_gold_object(
                        gold=gold,
                        row=row,
                        model=args.model,
                        use_llm=False,
                        selected_context_path="",
                    )
                    gold.setdefault("gold_extraction_metadata", {})["llm_failure_fallback_error"] = repr(exc)
                    save_json(gold, out_file)
                except Exception as fallback_exc:
                    rejected.append(
                        {
                            "project_id": project_id,
                            "reason": "gold_extraction_failed",
                            "error": repr(fallback_exc),
                            "llm_error": repr(exc),
                        }
                    )
                    index_rows.append(
                        {
                            "project_id": project_id,
                            "status": "failed",
                            "error": repr(fallback_exc),
                            "llm_error": repr(exc),
                        }
                    )
                    continue
            else:
                rejected.append(
                    {
                        "project_id": project_id,
                        "reason": "gold_extraction_failed",
                        "error": repr(exc),
                    }
                )
                index_rows.append(
                    {
                        "project_id": project_id,
                        "status": "failed",
                        "error": repr(exc),
                    }
                )
                continue

        usable, usable_reason = is_usable_gold(gold)
        index_row = {
            "project_id": project_id,
            "status": "saved",
            "gold_spec_path": str(out_file),
            "usable": usable,
            "usable_reason": usable_reason,
            "source_quality": gold.get("gold_extraction_metadata", {}).get("source_quality"),
            "quality_label": gold.get("quality_label"),
            "quality_warnings": gold.get("quality_warnings", []),
            "reference_word_count": word_count(gold.get("codification_ready_reference", "")),
            "has_code_evidence": gold.get("gold_extraction_metadata", {}).get("has_code_evidence"),
            "has_paper_evidence": gold.get("gold_extraction_metadata", {}).get("has_paper_evidence"),
        }
        index_rows.append(index_row)

        if usable:
            gold_specs.append(gold)
        else:
            rejected.append(
                {
                    "project_id": project_id,
                    "reason": usable_reason,
                    "gold_spec_path": str(out_file),
                    "source_quality": gold.get("gold_extraction_metadata", {}).get("source_quality"),
                    "quality_warnings": gold.get("quality_warnings", []),
                }
            )

        save_jsonl(gold_specs, args.out_dir / "gold_specs.jsonl")
        save_jsonl(rejected, args.out_dir / "rejected_gold_specs.jsonl")
        save_jsonl(index_rows, args.out_dir / "index.jsonl")

    save_jsonl(gold_specs, args.out_dir / "gold_specs.jsonl")
    save_jsonl(rejected, args.out_dir / "rejected_gold_specs.jsonl")
    save_jsonl(index_rows, args.out_dir / "index.jsonl")

    source_quality_counter = Counter(
        s.get("gold_extraction_metadata", {}).get("source_quality", "unknown")
        for s in gold_specs
    )
    warning_counter = Counter(
        warning for s in gold_specs for warning in s.get("quality_warnings", [])
    )
    rejected_counter = Counter(r.get("reason", "unknown") for r in rejected)

    summary = {
        "input_projects": len(rows),
        "gold_specs": len(gold_specs),
        "rejected_gold_specs": len(rejected),
        "use_llm": bool(args.use_llm),
        "model": args.model if args.use_llm else None,
        "source_quality": dict(source_quality_counter),
        "quality_warnings": dict(warning_counter),
        "rejection_reasons": dict(rejected_counter),
    }

    save_json(summary, args.out_dir / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()
