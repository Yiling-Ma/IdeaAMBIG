from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from taxonomy_utils import (
    infer_codification_slot_from_level2,
    strip_reference_language_from_text,
    find_reference_language_hits,
)


DEFAULT_INPUT = "Real_bench/github_issue_mining/outputs_step6/main_resolved.jsonl"
DEFAULT_OUT_DIR = (
    "Real_bench/github_issue_mining/outputs_step7/realgap_benchmark_instances"
)

# Records that passed automatic Step6 but are risky for final benchmark quality
# based on manual audit:
# - Flow-GRPO #42: qualitative diversity discussion over-claimed as quantitative N-JSD protocol
# - Flow-GRPO #22: KL logging interpretation, weak method-core codification value
KNOWN_RISKY_RECORD_IDS = {
    "github_yifan123__flow_grpo_issue_42__sde_diversity_evaluation_protocol_missing",
    "github_yifan123__flow_grpo_issue_22__kl_logging_metrics_meaning",
}

VALID_LEVEL2_BY_LEVEL1 = {
    "Ambiguity": {
        "ambiguous formal definition",
        "ambiguous method behavior",
    },
    "Incompleteness": {
        "missing algorithmic specification",
        "missing hyperparameter protocol",
        "missing model architecture",
        "missing evaluation protocol",
        "missing data/preprocessing protocol",
    },
    "Inconsistency": {
        "inconsistent objective or loss",
        "inconsistent architecture or pipeline",
        "inconsistent model specification",
    },
}

LEVEL2_TO_LEVEL1 = {
    level2: level1
    for level1, labels in VALID_LEVEL2_BY_LEVEL1.items()
    for level2 in labels
}

VALID_LEVEL1 = set(VALID_LEVEL2_BY_LEVEL1.keys())
VALID_LEVEL2 = set(LEVEL2_TO_LEVEL1.keys())

VALID_CODIFICATION_SLOTS = {
    "task",
    "input",
    "output",
    "core_method",
    "algorithm",
    "training",
    "evaluation",
    "implementation_detail",
    "code_behavior",
    "preprocessing",
    "data",
    "inference",
}

VALID_GRANULARITY = {"coarse", "medium", "fine"}

VALID_RESOLUTION_ROLES = {
    "implementation_blocker",
    "open_design_choice",
    "reproducibility_detail",
    "inconsistency_to_resolve",
}

VALID_ACTION_TYPES = {
    "clarification_question",
    "evidence_seeking",
    "experiment_selection",
}

REPO_LANGUAGE_TERMS = [
    "train_model",
    "timemoerunner",
    "v10postprocess",
    "latest commit",
    "codebase",
    "official validation script",
    "official code",
    "github.com",
    "github issue",
    "issue thread",
    "main branch",
    "runner's",
    "runner does not",
    "repo",
    "repository",
    "pull request",
    "commit",
]

INPUT_LEAKAGE_TERMS = [
    "ambiguous",
    "ambiguity",
    "underspecified",
    "under-specified",
    "missing",
    "incomplete",
    "not specified",
    "does not specify",
    "unclear",
    "undefined",
    "inconsistent",
    "inconsistency",
    "contradiction",
    "conflict",
    "github issue",
    "issue thread",
    "maintainer reply",
    "author clarification",
    "specification gap",
    "method gap",
    "implementation gap",
    "defect",
    "blocker",
]

PAPER_SPEC_DIAGNOSTIC_PATTERNS = [
    "does not specify",
    "not fully specified",
    "not specified",
    "is incomplete",
    "is ambiguous",
    "is undefined",
    "does not indicate",
    "does not describe",
    "does not state",
    "lack of specification",
    "missing specification",
    "unclear",
    "ambiguous",
    "underspecified",
]


def load_pipeline07():
    pipeline_path = (
        Path(__file__).resolve().parents[1]
        / "pipeline"
        / "step_07_build_realgap_bench_instances.py"
    )
    spec = importlib.util.spec_from_file_location("pipeline07", pipeline_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load pipeline module: {pipeline_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P7 = load_pipeline07()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return P7.load_jsonl(path)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    P7.save_json(obj, path)


def save_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    P7.save_jsonl(rows, path)


def read_text(path: str, max_chars: int) -> str:
    return P7.read_text(path, max_chars=max_chars)


def safe_filename(x: Any, max_len: int = 160) -> str:
    return P7.safe_filename(x, max_len=max_len)


def build_issue_thread_text(record: Dict[str, Any], max_chars: int = 30000) -> str:
    parts: List[str] = []

    title = str(record.get("title") or "").strip()
    body = str(record.get("body") or "").strip()

    if title:
        parts.append(f"Issue title: {title}")
    if body:
        parts.append(body)

    for comment in record.get("comments") or []:
        if not isinstance(comment, dict):
            continue

        author = ""
        user = comment.get("user")
        if isinstance(user, dict):
            author = str(user.get("login") or "")

        assoc = str(comment.get("author_association") or "")
        cbody = str(comment.get("body") or "").strip()
        if not cbody:
            continue

        header = f"Comment by {author}" if author else "Comment"
        if assoc:
            header += f" ({assoc})"

        parts.append(f"{header}:\n{cbody}")

    return "\n\n".join(parts)[:max_chars]


def prepare_item(item: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(item)
    record_id = str(out.get("record_id") or "").strip()

    if not out.get("realgap_id"):
        out["realgap_id"] = f"{record_id}_resolved_gap_001"

    if not out.get("source"):
        out["source"] = "GitHub_issue_real_gap"

    if not out.get("original_paper_title"):
        out["original_paper_title"] = out.get("paper_title_guess")

    # Normalize gap taxonomy if Step6 corrected it.
    gap = dict(out.get("gap") or {})
    level2 = str(gap.get("level2") or "").strip()
    if level2 in LEVEL2_TO_LEVEL1:
        gap["level1"] = LEVEL2_TO_LEVEL1[level2]
    out["gap"] = gap

    return out


def should_exclude_record(
    item: Dict[str, Any],
    *,
    drop_known_risky: bool,
    excluded_record_ids: set[str],
) -> Tuple[bool, str]:
    rid = str(item.get("record_id") or "").strip()
    if rid in excluded_record_ids:
        return True, "excluded_by_user_record_id"

    if drop_known_risky and rid in KNOWN_RISKY_RECORD_IDS:
        return True, "excluded_known_risky_manual_audit"

    return False, ""


def infer_github_resolution_role(gap: Dict[str, Any], granularity: str) -> str:
    hint = str(gap.get("resolution_role_hint") or "").strip()
    if hint in VALID_RESOLUTION_ROLES:
        return hint

    affected = str(gap.get("affected_component") or "").lower()
    level2 = str(gap.get("level2") or "").lower()
    codification_slot = infer_github_codification_slot(gap)

    if affected == "evaluation" or codification_slot == "evaluation":
        return "reproducibility_detail"

    if "evaluation protocol" in level2 or "data/preprocessing protocol" in level2:
        return "reproducibility_detail"

    role = P7.infer_resolution_role(gap, granularity)
    if role == "implementation_blocker" and (
        "hyperparameter" in level2 or "evaluation" in affected
    ):
        return "reproducibility_detail"

    return role


def infer_github_codification_slot(gap: Dict[str, Any]) -> str:
    level2 = str(gap.get("level2") or "").strip()
    if level2:
        text = " ".join(
            str(gap.get(k) or "")
            for k in (
                "gap_summary",
                "solution_summary",
                "gold_clarified_detail",
                "why_this_blocks_or_affects_codification",
            )
        )
        slot = infer_codification_slot_from_level2(level2, text)
        if slot in VALID_CODIFICATION_SLOTS:
            return slot

    affected = str(gap.get("affected_component") or "").strip().lower()
    if affected in VALID_CODIFICATION_SLOTS:
        return affected

    try:
        slot = P7.infer_codification_slot(gap)
        if slot in VALID_CODIFICATION_SLOTS:
            return slot
    except Exception:
        pass

    return "implementation_detail"


def infer_github_granularity(gap: Dict[str, Any]) -> str:
    try:
        g = P7.infer_granularity(gap)
        if g in VALID_GRANULARITY:
            return g
    except Exception:
        pass

    level2 = str(gap.get("level2") or "").lower()
    if "formal definition" in level2:
        return "fine"
    if "evaluation protocol" in level2 or "data/preprocessing protocol" in level2:
        return "medium"
    return "medium"


def build_github_source_object(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_type": "GitHub_issue_real_gap",
        "record_id": item.get("record_id"),
        "realgap_id": item.get("realgap_id"),
        "repo": item.get("repo"),
        "repo_url": item.get("repo_url"),
        "issue_number": item.get("issue_number"),
        "issue_url": item.get("issue_url"),
        "issue_title": item.get("title"),
        "original_paper_title": item.get("original_paper_title")
        or item.get("paper_title_guess"),
        "original_paper_url": item.get("original_paper_url"),
        "original_paper_pdf_url": item.get("original_paper_pdf_url"),
        "original_text_path": item.get("original_text_path"),
        "issue_evidence_source": "github_issue_thread",
    }


def build_github_prompt(
    item: Dict[str, Any],
    original_text: str,
    issue_thread_text: str,
    instance_id: str,
) -> str:
    gap = item.get("gap") or {}

    codification_slot = infer_github_codification_slot(gap)
    granularity = infer_github_granularity(gap)
    resolution_role = infer_github_resolution_role(gap, granularity)

    level2 = str(gap.get("level2", "")).strip()
    level1 = LEVEL2_TO_LEVEL1.get(level2, str(gap.get("level1", "")).strip())

    gold_detail = str(gap.get("gold_clarified_detail", "")).strip()
    solution_source_type = str(gap.get("solution_source_type", "")).strip()

    source = build_github_source_object(item)

    defect_seed = {
        "slot": codification_slot,
        "level1": level1,
        "level2": level2,
        "granularity": granularity,
        "resolution_role": resolution_role,
        "codification_slot": codification_slot,
        "gold_detail_removed_or_corrupted": gold_detail,
        "why_this_blocks_or_affects_codification": gap.get(
            "why_this_blocks_or_affects_codification", ""
        ),
    }

    evidence = {
        "gap_summary": gap.get("gap_summary", ""),
        "gap_quote": gap.get("gap_quote", ""),
        "solution_summary": gap.get("solution_summary", ""),
        "solution_quote": gap.get("solution_quote", ""),
        "solution_source_type": solution_source_type,
        "affected_component": gap.get("affected_component", ""),
        "gold_clarified_detail": gold_detail,
    }

    source_json = json.dumps(source, indent=2, ensure_ascii=False)
    defect_seed_json = json.dumps(defect_seed, indent=2, ensure_ascii=False)
    evidence_json = json.dumps(evidence, indent=2, ensure_ascii=False)

    return f"""
Construct one benchmark instance from a real GitHub issue specification gap.

Context:
We are building a benchmark for idea/specification ambiguity resolution. The benchmark input is an underspecified research-method specification derived from the original paper. A model should diagnose what is missing, ambiguous, or inconsistent before codification.

This instance comes from a GitHub issue thread about an official or community implementation repo. The original paper had a method-core specification gap. The issue thread provides concrete clarification evidence from authors or maintainers.

Your task:
Create a benchmark instance with the SAME schema as the MLRC/TMLR real-gap benchmark, but using GitHub issue evidence instead of a reproducibility report.

CRITICAL NO-LEAKAGE REQUIREMENT:
The input.underspecified_spec is what will be shown to evaluated models. It must NOT reveal that there is a gap, defect, ambiguity, inconsistency, or codification problem.

Do NOT use diagnostic/meta-evaluation language in input.underspecified_spec, including:
- ambiguous, ambiguity, underspecified, missing, incomplete
- not specified, does not specify, unclear, undefined
- inconsistent, inconsistency, contradiction, conflict
- GitHub issue, issue thread, maintainer reply, author clarification
- specification gap, method gap, implementation gap, defect, blocker

Important special case:
- The acronym "GAP" may mean global average pooling. Do NOT treat it as the word "gap".

Instead:
- Write input.underspecified_spec as a natural paper-style method description based ONLY on the original paper text excerpt.
- It should sound like a normal method paragraph from the original paper.
- Include the problematic surface form from the paper, but without explicitly saying it is problematic.
- Do NOT copy issue-thread diagnostic questions into underspecified_spec.
- Do NOT invent repository workflows, function names, script names, or API usage that are absent from the paper.
- If the issue is about repo code/API usage, still write underspecified_spec from the paper's high-level method/training/evaluation description only.
- For incompleteness, include only the high-level paper-side operation, not "the paper does not specify...".
- For inconsistency, include only the paper-side statement.

CRITICAL SURFACE-FORM RULE:
- defects[0].surface_form_in_underspecified_spec must be a phrase that appears in, or is a faithful paraphrase of, the original paper text excerpt.
- Never use the user's mistaken repo workflow as the surface form.
- Never use repo-only terms such as train_model, TimeMoeRunner, v10postprocess, latest commit, official validation script, GitHub, repository, main branch.

CRITICAL GOLD REFERENCE REQUIREMENT:
gold.codification_ready_reference should be a clean, standalone, codification-ready research idea specification.
It should include the gold_clarified_detail from the GitHub resolution evidence, rewritten as method-level specification language.
Avoid meta-language such as "as clarified in the issue", "the authors replied", "GitHub thread", "official code", "latest commit", "codebase", or "repository".
Prefer method-level wording over repo function names.
Do NOT use literature deferral, citation, or prior-work language in codification_ready_reference.
Banned phrases include: "as proven in the literature", "according to", "prior work", "previous work", "in the literature", "as described in", "as shown in", and bracket citations like [27].

Examples:
Bad gold: "Use v10postprocess in the official validation script."
Good gold: "During validation, all max_det predictions are passed to metric computation without confidence-threshold filtering."

CRITICAL PAPER-DERIVED SPECIFICATION RULE:
gold.paper_derived_specification must contain neutral extracted facts only.
Do NOT write gap-diagnostic language in any field, especially unknown_fields or reproducibility_relevant_details.
Do NOT say "the paper does not specify", "not fully specified", "ambiguous", "missing", "unclear", or "undefined" inside paper_derived_specification.
If a detail is unknown from the paper alone, leave unknown_fields empty or use neutral placeholders like "Exact validation filtering protocol for low-confidence predictions".
codification_readiness.reason may describe the blocker, but paper_derived_specification must stay neutral.

Both input.underspecified_spec and gold.codification_ready_reference must describe the full research idea:
1. research goal or motivation,
2. task being solved,
3. inputs and outputs,
4. core method or model structure,
5. the relevant method component containing the hidden issue.

Important constraints:
- Use exactly one defect corresponding to the provided real gap (ONE atomic gap only).
- If the provided gold_clarified_detail covers multiple independent slots, return:
  {{"reject_reason": "composite_gap_needs_split"}}
  and omit the normal benchmark schema.
- Evaluation protocol gaps (dataset split, metric computation, rasterization settings, prompt set, evaluation sampling) are reproducibility_detail, NOT implementation_blocker.
- Data/preprocessing protocol gaps (windowing, stride, filtering, label construction, tokenization, normalization) are reproducibility_detail or implementation_blocker depending on whether training cannot run without them.
- Do not merge multiple metrics or independent hyperparameters into one defect or one gold reference.
- Do not invent unsupported datasets, baselines, methods, architectures, or hyperparameters.
- Base paper-side content primarily on the original paper text excerpt.
- Use the GitHub issue evidence only to determine what clarification belongs in the gold reference.
- Keep the benchmark instance self-contained and understandable.
- If the provided gap is purely about repository API usage and cannot be translated into a paper/method specification, return:
  {{"reject_reason": "code_only_not_paper_spec_gap"}}
  and omit the normal benchmark schema.

Allowed labels:
Level-1 = Ambiguity | Incompleteness | Inconsistency
Level-2 = ambiguous formal definition | ambiguous method behavior | missing algorithmic specification | missing hyperparameter protocol | missing model architecture | missing evaluation protocol | missing data/preprocessing protocol | inconsistent objective or loss | inconsistent architecture or pipeline | inconsistent model specification
Granularity = coarse | medium | fine
Resolution role = implementation_blocker | open_design_choice | reproducibility_detail | inconsistency_to_resolve
Codification slot = task | input | output | core_method | algorithm | training | evaluation | implementation_detail | code_behavior | preprocessing | data | inference
Action type = clarification_question | evidence_seeking | experiment_selection

Fixed defect to use:
{defect_seed_json}

Real gap evidence from GitHub issue:
{evidence_json}

Source metadata:
{source_json}

Original paper text excerpt:
\"\"\"
{original_text}
\"\"\"

GitHub issue thread excerpt (resolution evidence only; do NOT leak issue language into underspecified_spec):
\"\"\"
{issue_thread_text}
\"\"\"

Return valid JSON only with exactly this schema:

{{
  "id": {json.dumps(instance_id, ensure_ascii=False)},
  "source": {source_json},
  "input": {{
    "underspecified_spec": ""
  }},
  "gold": {{
    "paper_derived_specification": {{
      "paper_title": "",
      "research_goal": "",
      "task": "",
      "inputs": "",
      "outputs": "",
      "core_method": "",
      "algorithm_steps": [],
      "training_or_optimization": "",
      "datasets": [],
      "evaluation_metrics": [],
      "baselines": [],
      "implementation_details": [],
      "reproducibility_relevant_details": [],
      "unknown_fields": []
    }},
    "codification_ready_reference": ""
  }},
  "defects": [
    {{
      "slot": {json.dumps(codification_slot, ensure_ascii=False)},
      "level1": {json.dumps(level1, ensure_ascii=False)},
      "level2": {json.dumps(level2, ensure_ascii=False)},
      "granularity": {json.dumps(granularity, ensure_ascii=False)},
      "resolution_role": {json.dumps(resolution_role, ensure_ascii=False)},
      "codification_slot": {json.dumps(codification_slot, ensure_ascii=False)},
      "gold_detail_removed_or_corrupted": {json.dumps(gold_detail, ensure_ascii=False)},
      "surface_form_in_underspecified_spec": "",
      "why_this_blocks_or_affects_codification": ""
    }}
  ],
  "open_design_choices": [],
  "codification_readiness": {{
    "is_ready": false,
    "readiness_score": 0,
    "blocking_missing_specs": [],
    "open_design_choices": [],
    "reason": ""
  }},
  "expected_clarification_actions": [
    {{
      "slot": {json.dumps(codification_slot, ensure_ascii=False)},
      "action_type": "",
      "question_or_action": "",
      "evidence_to_seek": ""
    }}
  ],
  "construction_metadata": {{
    "construction_method": "real_gap_from_github_issue",
    "paper_id": {json.dumps(item.get("record_id"), ensure_ascii=False)},
    "realgap_id": {json.dumps(item.get("realgap_id"), ensure_ascii=False)},
    "num_defects": 1,
    "selected_perturbations": [
      {defect_seed_json}
    ],
    "gap_quote": {json.dumps(gap.get("gap_quote", ""), ensure_ascii=False)},
    "solution_quote": {json.dumps(gap.get("solution_quote", ""), ensure_ascii=False)},
    "solution_source_type": {json.dumps(solution_source_type, ensure_ascii=False)}
  }}
}}

Output rules:
- defects must contain exactly one defect.
- The defect labels must exactly match the fixed defect above.
- underspecified_spec should be 120-220 words.
- codification_ready_reference should be 160-280 words.
- Both must start with research goal, task, or method context.
- codification_ready_reference must include the gold_clarified_detail.
- codification_ready_reference must not defer to literature, citations, or prior work.
- underspecified_spec must contain the paper-side gap surface form but must NOT include the gold_clarified_detail.
- input.underspecified_spec must NOT contain diagnostic/meta-evaluation leakage language.
- codification_readiness.is_ready must be false.
- readiness_score should be 2 or 3.
- expected_clarification_actions must contain exactly one action and must use only valid action_type labels.
- Do not create extra defects.
- paper_derived_specification fields must not contain gap-diagnostic wording.
"""


def normalize_text_for_check(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def is_diagnostic_paper_spec_sentence(text: str) -> bool:
    low = normalize_text_for_check(text)
    return any(pattern in low for pattern in PAPER_SPEC_DIAGNOSTIC_PATTERNS)


def sanitize_paper_derived_specification(spec: Dict[str, Any]) -> Dict[str, Any]:
    spec = dict(spec or {})

    string_fields = [
        "paper_title",
        "research_goal",
        "task",
        "inputs",
        "outputs",
        "core_method",
        "training_or_optimization",
    ]

    for field in string_fields:
        val = str(spec.get(field) or "")
        if is_diagnostic_paper_spec_sentence(val):
            spec[field] = ""

    list_fields = [
        "algorithm_steps",
        "datasets",
        "evaluation_metrics",
        "baselines",
        "implementation_details",
        "reproducibility_relevant_details",
        "unknown_fields",
    ]

    for field in list_fields:
        values = spec.get(field) or []
        if not isinstance(values, list):
            values = [str(values)] if values else []

        spec[field] = [
            item for item in values
            if not is_diagnostic_paper_spec_sentence(str(item))
        ]

    return spec


def find_repo_language_hits(text: str) -> List[str]:
    low = normalize_text_for_check(text)
    return [term for term in REPO_LANGUAGE_TERMS if term in low]


def find_input_leakage_hits(text: str) -> List[str]:
    low = normalize_text_for_check(text)
    hits = []

    for term in INPUT_LEAKAGE_TERMS:
        if term == "gap":
            continue
        if term in low:
            hits.append(term)

    return hits


def enforce_github_defect_roles(instance: Dict[str, Any]) -> Dict[str, Any]:
    defects = instance.get("defects") or []
    if not defects:
        return instance

    defect = dict(defects[0])
    slot = str(defect.get("codification_slot") or defect.get("slot") or "")
    level2 = str(defect.get("level2") or "").lower()
    role = str(defect.get("resolution_role") or "")

    if slot == "evaluation" or "evaluation protocol" in level2:
        if role == "implementation_blocker":
            defect["resolution_role"] = "reproducibility_detail"

    if "hyperparameter" in level2:
        if role == "implementation_blocker":
            defect["resolution_role"] = "reproducibility_detail"

    if "data/preprocessing protocol" in level2:
        if slot in {"data", "preprocessing"} and role not in VALID_RESOLUTION_ROLES:
            defect["resolution_role"] = "reproducibility_detail"

    instance["defects"] = [defect] + list(defects[1:])
    return instance


def force_fixed_defect_from_item(instance: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    gap = item.get("gap") or {}

    level2 = str(gap.get("level2") or "").strip()
    level1 = LEVEL2_TO_LEVEL1.get(level2, str(gap.get("level1") or "").strip())
    codification_slot = infer_github_codification_slot(gap)
    granularity = infer_github_granularity(gap)
    resolution_role = infer_github_resolution_role(gap, granularity)

    defects = instance.get("defects") or []
    defect = dict(defects[0]) if defects else {}

    defect["slot"] = codification_slot
    defect["level1"] = level1
    defect["level2"] = level2
    defect["granularity"] = granularity
    defect["resolution_role"] = resolution_role
    defect["codification_slot"] = codification_slot
    defect["gold_detail_removed_or_corrupted"] = str(
        gap.get("gold_clarified_detail") or ""
    )

    if not defect.get("why_this_blocks_or_affects_codification"):
        defect["why_this_blocks_or_affects_codification"] = str(
            gap.get("why_this_blocks_or_affects_codification") or ""
        )

    instance["defects"] = [defect]
    return instance


def postprocess_github_instance(
    instance: Dict[str, Any],
    item: Dict[str, Any],
    *,
    fail_on_repo_language: bool = False,
) -> Dict[str, Any]:
    instance = force_fixed_defect_from_item(instance, item)
    instance = enforce_github_defect_roles(instance)

    gold = instance.get("gold") or {}
    paper_spec = gold.get("paper_derived_specification") or {}
    gold["paper_derived_specification"] = sanitize_paper_derived_specification(paper_spec)

    reference = str(gold.get("codification_ready_reference") or "")
    cleaned_reference, ref_lang_hits = strip_reference_language_from_text(reference)
    if cleaned_reference:
        gold["codification_ready_reference"] = cleaned_reference
    instance["gold"] = gold

    warnings = list(instance.get("quality_warnings") or [])

    underspecified = str((instance.get("input") or {}).get("underspecified_spec") or "")
    reference = str(gold.get("codification_ready_reference") or "")

    for hit in ref_lang_hits:
        warnings.append(
            {"type": "reference_language_stripped_from_codification_ready_reference", "pattern": hit}
        )
    for hit in find_reference_language_hits(reference):
        warnings.append(
            {"type": "reference_language_in_codification_ready_reference", "pattern": hit}
        )

    for term in find_repo_language_hits(underspecified):
        warnings.append(
            {"type": "repo_language_in_underspecified_spec", "term": term}
        )

    for term in find_repo_language_hits(reference):
        warnings.append(
            {"type": "repo_language_in_codification_ready_reference", "term": term}
        )

    instance["quality_warnings"] = warnings

    if fail_on_repo_language and any(
        isinstance(w, dict) and str(w.get("type", "")).startswith("repo_language_in_")
        for w in warnings
    ):
        hits = [
            w.get("term")
            for w in warnings
            if isinstance(w, dict) and str(w.get("type", "")).startswith("repo_language_in_")
        ]
        raise ValueError(
            "Repo-specific language detected in benchmark text: "
            + ", ".join(str(x) for x in hits if x)
        )

    return instance


def force_github_instance_metadata(
    instance: Dict[str, Any],
    item: Dict[str, Any],
    instance_id: str,
) -> Dict[str, Any]:
    try:
        instance = P7.force_instance_metadata(
            instance=instance,
            item=item,
            instance_id=instance_id,
        )
    except Exception:
        instance["id"] = instance_id

    gap = item.get("gap") or {}
    instance["source"] = build_github_source_object(item)

    cm = dict(instance.get("construction_metadata") or {})
    cm["construction_method"] = "real_gap_from_github_issue"
    cm["repo"] = item.get("repo")
    cm["issue_url"] = item.get("issue_url")
    cm["issue_number"] = item.get("issue_number")
    cm["gap_summary"] = gap.get("gap_summary", "")
    cm["solution_summary"] = gap.get("solution_summary", "")
    cm["gold_clarified_detail"] = gap.get("gold_clarified_detail", "")
    instance["construction_metadata"] = cm

    return instance


def validate_github_instance_schema(
    instance: Dict[str, Any],
    *,
    fail_on_input_leakage: bool = False,
) -> None:
    if not isinstance(instance, dict):
        raise ValueError("instance_is_not_dict")

    for key in ("id", "source", "input", "gold", "defects", "codification_readiness"):
        if key not in instance:
            raise ValueError(f"missing_required_key:{key}")

    input_obj = instance.get("input") or {}
    underspecified = str(input_obj.get("underspecified_spec") or "").strip()
    if not underspecified:
        raise ValueError("missing_input_underspecified_spec")

    gold = instance.get("gold") or {}
    reference = str(gold.get("codification_ready_reference") or "").strip()
    if not reference:
        raise ValueError("missing_gold_codification_ready_reference")

    defects = instance.get("defects")
    if not isinstance(defects, list) or len(defects) != 1:
        raise ValueError("defects_must_contain_exactly_one_defect")

    defect = defects[0]
    if not isinstance(defect, dict):
        raise ValueError("defect_is_not_dict")

    level1 = str(defect.get("level1") or "")
    level2 = str(defect.get("level2") or "")
    slot = str(defect.get("codification_slot") or defect.get("slot") or "")
    granularity = str(defect.get("granularity") or "")
    role = str(defect.get("resolution_role") or "")

    if level1 not in VALID_LEVEL1:
        raise ValueError(f"invalid_level1:{level1}")

    if level2 not in VALID_LEVEL2:
        raise ValueError(f"invalid_level2:{level2}")

    if LEVEL2_TO_LEVEL1[level2] != level1:
        raise ValueError(f"level2_not_consistent_with_level1:{level1}/{level2}")

    if slot not in VALID_CODIFICATION_SLOTS:
        raise ValueError(f"invalid_codification_slot:{slot}")

    if granularity not in VALID_GRANULARITY:
        raise ValueError(f"invalid_granularity:{granularity}")

    if role not in VALID_RESOLUTION_ROLES:
        raise ValueError(f"invalid_resolution_role:{role}")

    actions = instance.get("expected_clarification_actions") or []
    if not isinstance(actions, list) or len(actions) != 1:
        raise ValueError("expected_clarification_actions_must_have_one_action")

    action = actions[0] if actions else {}
    action_type = str(action.get("action_type") or "")
    if action_type not in VALID_ACTION_TYPES:
        raise ValueError(f"invalid_action_type:{action_type}")

    readiness = instance.get("codification_readiness") or {}
    if readiness.get("is_ready") is not False:
        raise ValueError("codification_readiness_is_ready_must_be_false")

    try:
        score = int(readiness.get("readiness_score"))
    except Exception:
        raise ValueError("invalid_readiness_score") from None

    if score not in {1, 2, 3, 4}:
        raise ValueError(f"readiness_score_out_of_range:{score}")

    if fail_on_input_leakage:
        hits = find_input_leakage_hits(underspecified)
        if hits:
            raise ValueError(
                "input_leakage_detected:" + ",".join(sorted(set(hits)))
            )


def validate_instance_compat(
    instance: Dict[str, Any],
    *,
    fail_on_input_leakage: bool = False,
) -> None:
    """
    Try the shared pipeline validator first. If it fails due to old taxonomy
    support, fall back to GitHub-specific validation with the updated taxonomy.
    Leakage and schema errors are still enforced.
    """
    try:
        P7.validate_instance(
            instance,
            fail_on_input_leakage=fail_on_input_leakage,
        )
        return
    except Exception as e:
        err = repr(e)
        has_new_label = False
        defects = instance.get("defects") or []
        if defects and isinstance(defects[0], dict):
            level2 = str(defects[0].get("level2") or "")
            slot = str(defects[0].get("codification_slot") or defects[0].get("slot") or "")
            has_new_label = (
                level2 in {
                    "missing evaluation protocol",
                    "missing data/preprocessing protocol",
                }
                or slot in {"preprocessing", "data", "inference"}
            )

        if not has_new_label:
            raise

        # New taxonomy label may be unsupported by old pipeline07 validator.
        # Apply local strict validation instead.
        validate_github_instance_schema(
            instance,
            fail_on_input_leakage=fail_on_input_leakage,
        )
        instance.setdefault("quality_warnings", []).append(
            {
                "type": "validated_with_github_updated_taxonomy_fallback",
                "pipeline07_error": err,
            }
        )


def has_saved_instance(out_dir: Path, instance_id: str) -> bool:
    instance_path = out_dir / instance_id / "benchmark_instance.json"
    return instance_path.exists() and instance_path.stat().st_size > 50


def load_existing_index(out_dir: Path) -> List[Dict[str, Any]]:
    index_path = out_dir / "index.jsonl"
    return load_jsonl(index_path) if index_path.exists() else []


INDEX_STATUS_PRIORITY = {
    "saved": 4,
    "dry_run": 3,
    "failed": 2,
    "excluded": 1,
}


def dedupe_index_rows(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        instance_id = str(row.get("id") or "").strip()
        if not instance_id:
            continue
        prev = by_id.get(instance_id)
        if prev is None:
            by_id[instance_id] = row
            continue
        prev_rank = INDEX_STATUS_PRIORITY.get(str(prev.get("status") or ""), 0)
        row_rank = INDEX_STATUS_PRIORITY.get(str(row.get("status") or ""), 0)
        if row_rank >= prev_rank:
            by_id[instance_id] = row
    return by_id


def discover_saved_instances(out_dir: Path) -> Dict[str, str]:
    found: Dict[str, str] = {}
    if not out_dir.exists():
        return found
    for path in out_dir.glob("*/benchmark_instance.json"):
        if path.stat().st_size > 50:
            found[path.parent.name] = str(path)
    return found


def load_resume_index(out_dir: Path) -> Dict[str, Dict[str, Any]]:
    by_id = dedupe_index_rows(load_existing_index(out_dir))
    for instance_id, instance_path in discover_saved_instances(out_dir).items():
        row = dict(by_id.get(instance_id) or {})
        if str(row.get("status") or "") == "saved" and has_saved_instance(out_dir, instance_id):
            continue
        row.update({
            "id": instance_id,
            "benchmark_instance_path": instance_path,
            "status": "saved",
            "recovered_from_disk": True,
        })
        by_id[instance_id] = row
    return by_id


def upsert_index_row(
    index_by_id: Dict[str, Dict[str, Any]],
    row: Dict[str, Any],
) -> None:
    instance_id = str(row.get("id") or "").strip()
    if instance_id:
        index_by_id[instance_id] = row


def should_resume_skip(
    index_by_id: Dict[str, Dict[str, Any]],
    out_dir: Path,
    instance_id: str,
) -> bool:
    row = index_by_id.get(instance_id)
    if not row:
        return False
    if str(row.get("status") or "") != "saved":
        return False
    return has_saved_instance(out_dir, instance_id)


def finalize_index_rows(
    index_by_id: Dict[str, Dict[str, Any]],
    build_rows: List[Dict[str, Any]],
    excluded_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    ordered_ids: List[str] = []
    seen: set[str] = set()

    for item in build_rows + excluded_rows:
        instance_id = safe_filename(str(item.get("realgap_id") or item.get("record_id") or ""))
        if instance_id and instance_id not in seen:
            ordered_ids.append(instance_id)
            seen.add(instance_id)

    for instance_id in sorted(index_by_id):
        if instance_id not in seen:
            ordered_ids.append(instance_id)
            seen.add(instance_id)

    return [index_by_id[iid] for iid in ordered_ids if iid in index_by_id]


def save_index(index_by_id: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    rows = list(index_by_id.values())
    save_jsonl(rows, out_dir / "index.jsonl")


def build_index_row(
    *,
    instance: Optional[Dict[str, Any]],
    item: Dict[str, Any],
    instance_id: str,
    instance_path: str,
    status: str,
    error: str = "",
    error_path: str = "",
    excluded_reason: str = "",
) -> Dict[str, Any]:
    gap = item.get("gap") or {}
    row = {
        "id": instance_id,
        "realgap_id": item.get("realgap_id"),
        "record_id": item.get("record_id"),
        "source": item.get("source"),
        "repo": item.get("repo"),
        "issue_url": item.get("issue_url"),
        "original_paper_title": item.get("original_paper_title")
        or item.get("paper_title_guess"),
        "benchmark_instance_path": instance_path,
        "level1": gap.get("level1"),
        "level2": gap.get("level2"),
        "codification_slot": infer_github_codification_slot(gap),
        "construction_method": "real_gap_from_github_issue",
        "status": status,
    }

    if instance is not None:
        row["source_type"] = instance.get("source", {}).get("source_type")
        row["quality_warnings"] = instance.get("quality_warnings", [])
        if instance.get("defects"):
            row["level1"] = instance["defects"][0].get("level1")
            row["level2"] = instance["defects"][0].get("level2")
            row["codification_slot"] = instance["defects"][0].get("codification_slot")
    else:
        row["quality_warnings"] = []

    if error:
        row["error"] = error
    if error_path:
        row["error_path"] = error_path
    if excluded_reason:
        row["excluded_reason"] = excluded_reason

    return row


def parse_excluded_record_ids(raw_values: List[str]) -> set[str]:
    out: set[str] = set()
    for raw in raw_values or []:
        for part in str(raw or "").split(","):
            part = part.strip()
            if part:
                out.add(part)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default=DEFAULT_INPUT)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--model",
        default=os.getenv("OPENROUTER_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "deepseek/deepseek-v4-pro",
    )
    parser.add_argument("--max_original_chars", type=int, default=50000)
    parser.add_argument("--max_issue_chars", type=int, default=30000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save_prompt", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip instances already saved as benchmark_instance.json; retry failed/missing ones.",
    )
    parser.add_argument("--fail_on_input_leakage", action="store_true")
    parser.add_argument(
        "--fail_on_repo_language",
        action="store_true",
        help="Fail if underspecified_spec or codification_ready_reference contains repo/API terms.",
    )
    parser.add_argument(
        "--drop_known_risky",
        action="store_true",
        help="Drop manually identified risky Step6 main_resolved records from final Step7 construction.",
    )
    parser.add_argument(
        "--exclude_record_id",
        action="append",
        default=[],
        help="Record_id to exclude. Can be passed multiple times or as comma-separated values.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Build and save prompts only; do not call the LLM.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    excluded_record_ids = parse_excluded_record_ids(args.exclude_record_id)

    rows_all = [prepare_item(r) for r in load_jsonl(input_path)]

    filtered_rows: List[Dict[str, Any]] = []
    excluded_rows: List[Dict[str, Any]] = []

    for item in rows_all:
        exclude, reason = should_exclude_record(
            item,
            drop_known_risky=args.drop_known_risky,
            excluded_record_ids=excluded_record_ids,
        )
        if exclude:
            x = dict(item)
            x["_excluded_reason"] = reason
            excluded_rows.append(x)
        else:
            filtered_rows.append(item)

    rows = filtered_rows
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    client = None if args.dry_run else P7.build_client()

    index_by_id: Dict[str, Dict[str, Any]] = {}
    if args.resume:
        index_by_id = load_resume_index(out_dir)

    resume_skip_count = 0
    resume_retry_count = 0
    if args.resume:
        for item in rows:
            instance_id = safe_filename(str(item.get("realgap_id") or item.get("record_id") or ""))
            if should_resume_skip(index_by_id, out_dir, instance_id):
                resume_skip_count += 1
            else:
                resume_retry_count += 1

    print("\n===== Step 7: GitHub Real-Gap Benchmark Builder =====")
    print(f"Input:                  {input_path}")
    print(f"Output dir:             {out_dir}")
    print(f"Input rows total:        {len(rows_all)}")
    print(f"Excluded before build:   {len(excluded_rows)}")
    print(f"Rows to build:           {len(rows)}")
    print(f"Model:                  {args.model}")
    print(f"Resume:                 {args.resume}")
    if args.resume:
        print(f"Resume skip saved:      {resume_skip_count}")
        print(f"Resume retry/missing:   {resume_retry_count}")
    print(f"Dry run:                {args.dry_run}")
    print(f"Fail on input leakage:  {args.fail_on_input_leakage}")
    print(f"Fail on repo language:  {args.fail_on_repo_language}")
    print(f"Drop known risky:       {args.drop_known_risky}")
    print("=============================================================\n")

    for item in excluded_rows:
        realgap_id = str(item.get("realgap_id") or "")
        instance_id = safe_filename(realgap_id or item.get("record_id"))
        prev = index_by_id.get(instance_id) or {}
        if str(prev.get("status") or "") == "excluded":
            continue
        upsert_index_row(
            index_by_id,
            build_index_row(
                instance=None,
                item=item,
                instance_id=instance_id,
                instance_path="",
                status="excluded",
                excluded_reason=str(item.get("_excluded_reason") or ""),
            ),
        )

    if excluded_rows:
        save_index(index_by_id, out_dir)

    for idx, item in enumerate(tqdm(rows, desc="Build GitHub benchmark instances"), start=1):
        realgap_id = str(item.get("realgap_id") or "")
        instance_id = safe_filename(realgap_id or item.get("record_id"))

        if args.resume and should_resume_skip(index_by_id, out_dir, instance_id):
            tqdm.write(f"[{idx}/{len(rows)}] skip existing {instance_id}")
            continue

        original_text = read_text(
            str(item.get("original_text_path") or ""),
            max_chars=args.max_original_chars,
        )
        issue_thread_text = build_issue_thread_text(item, max_chars=args.max_issue_chars)

        prompt = build_github_prompt(
            item=item,
            original_text=original_text,
            issue_thread_text=issue_thread_text,
            instance_id=instance_id,
        )

        instance_dir = out_dir / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)

        if args.save_prompt or args.dry_run:
            (instance_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        if args.dry_run:
            upsert_index_row(
                index_by_id,
                build_index_row(
                    instance=None,
                    item=item,
                    instance_id=instance_id,
                    instance_path="",
                    status="dry_run",
                ),
            )
            save_index(index_by_id, out_dir)
            tqdm.write(f"[{idx}/{len(rows)}] dry-run prompt saved {instance_dir / 'prompt.txt'}")
            continue

        try:
            raw = P7.call_llm(client=client, model=args.model, prompt=prompt)
            (instance_dir / "raw_response.txt").write_text(raw, encoding="utf-8")

            instance = P7.safe_json_loads(raw)

            if instance.get("reject_reason") == "code_only_not_paper_spec_gap":
                raise ValueError("LLM rejected code-only gap: code_only_not_paper_spec_gap")

            if instance.get("reject_reason") == "composite_gap_needs_split":
                raise ValueError("LLM rejected composite gap: composite_gap_needs_split")

            instance = force_github_instance_metadata(
                instance=instance,
                item=item,
                instance_id=instance_id,
            )

            instance = postprocess_github_instance(
                instance,
                item=item,
                fail_on_repo_language=args.fail_on_repo_language,
            )

            validate_instance_compat(
                instance,
                fail_on_input_leakage=args.fail_on_input_leakage,
            )

            instance_path = instance_dir / "benchmark_instance.json"
            save_json(instance, instance_path)

            upsert_index_row(
                index_by_id,
                build_index_row(
                    instance=instance,
                    item=item,
                    instance_id=instance_id,
                    instance_path=str(instance_path),
                    status="saved",
                ),
            )
            save_index(index_by_id, out_dir)

            tqdm.write(f"[{idx}/{len(rows)}] saved {instance_path}")
            if instance.get("quality_warnings"):
                print(f"  warnings: {instance.get('quality_warnings')}")

        except Exception as e:
            error_path = instance_dir / "error.json"
            save_json(
                {
                    "id": instance_id,
                    "realgap_id": realgap_id,
                    "record_id": item.get("record_id"),
                    "error": repr(e),
                },
                error_path,
            )

            upsert_index_row(
                index_by_id,
                build_index_row(
                    instance=None,
                    item=item,
                    instance_id=instance_id,
                    instance_path="",
                    status="failed",
                    error=repr(e),
                    error_path=str(error_path),
                ),
            )
            save_index(index_by_id, out_dir)
            tqdm.write(f"[{idx}/{len(rows)}] failed {instance_id}: {repr(e)}")

        time.sleep(0.5)

    index_rows = finalize_index_rows(index_by_id, rows, excluded_rows)
    save_jsonl(index_rows, out_dir / "index.jsonl")

    saved = sum(r.get("status") == "saved" for r in index_rows)
    failed = sum(r.get("status") == "failed" for r in index_rows)
    dry = sum(r.get("status") == "dry_run" for r in index_rows)
    excluded = sum(r.get("status") == "excluded" for r in index_rows)

    warning_counter = Counter()
    level2_counter = Counter()
    status_counter = Counter()

    for row in index_rows:
        status_counter[str(row.get("status") or "")] += 1
        if row.get("level2"):
            level2_counter[str(row.get("level2"))] += 1

        for warning in row.get("quality_warnings", []) or []:
            if isinstance(warning, dict):
                warning_counter[warning.get("type", "dict_warning")] += 1
            else:
                warning_counter[str(warning)] += 1

    summary = {
        "script": "step_07_build_benchmark_instances.py",
        "input_path": str(input_path),
        "out_dir": str(out_dir),
        "model": args.model,
        "dry_run": args.dry_run,
        "resume": args.resume,
        "resume_skip_saved": resume_skip_count if args.resume else 0,
        "resume_retry_or_missing": resume_retry_count if args.resume else 0,
        "drop_known_risky": args.drop_known_risky,
        "excluded_record_ids": sorted(excluded_record_ids),
        "input_rows_total": len(rows_all),
        "input_rows_after_filter": len(rows),
        "instances_saved": saved,
        "instances_failed": failed,
        "instances_dry_run": dry,
        "instances_excluded": excluded,
        "status_distribution": dict(status_counter),
        "level2_distribution": dict(level2_counter),
        "quality_warning_distribution": dict(warning_counter),
        "index_path": str(out_dir / "index.jsonl"),
    }

    save_json(summary, out_dir / "summary.json")

    print("\n[done]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
