from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import OpenAI
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from taxonomy_utils import (
    TAXONOMY_BRIEF,
    GITHUB_TAXONOMY_BOUNDARY,
    LEVEL2_TO_LEVEL1,
    VALID_LEVEL2,
    normalize_level2,
    apply_taxonomy_correction,
    apply_taxonomy_correction_to_gap,
)

DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "deepseek/deepseek-v4-pro")


SYSTEM_PROMPT = """
You are an expert ML research benchmark annotator.

We are building IDEAAMBIG, a benchmark for idea/specification ambiguity resolution from real implementation evidence.

Your job is Step4 gate only:
1) decide whether a GitHub issue contains REAL, RESOLVED, method-core specification gap(s),
2) decide keep/reject,
3) if keep, split into up to 3 atomic candidates.

IMPORTANT:
- Step4 is gate only. Do NOT assign final taxonomy labels.
- Step4 is gate only. Do NOT write gold_clarified_detail rewrite.
- Keep only evidence-grounded atomic candidates with concrete gap_quote and solution_quote.

ATOMIC GAP RULE:
- One atomic gap = one independent missing/ambiguous/inconsistent implementation slot.
- If the issue clarifies multiple independent slots, return MULTIPLE entries in atomic_gaps, up to 3.
- Do NOT merge architecture, hyperparameter protocol, training, preprocessing, evaluation, and implementation behavior into one candidate.
- Each atomic gap needs its own gap_quote and solution_quote.
- Prefer gaps with clear implementation_blocker or reproducibility_detail impact.
- Reject gaps that are only visualization tips, resource requests, usage questions, or debugging support without codification impact.

A kept atomic gap must satisfy ALL gate criteria:
- is_spec_gap: the issue concerns an underspecified, ambiguous, or inconsistent method specification.
- is_actionable: resolving it changes or clarifies a concrete implementation decision.
- is_method_core_spec_gap: it affects method, model, algorithm, training, evaluation, preprocessing, data construction, inference, or implementation behavior.
- evidence_extractable: the issue thread provides enough concrete evidence for verification.

A resolved real gap requires concrete solution evidence in the issue/comments, such as:
- maintainer or author clarification,
- explicit code-derived behavior,
- a concrete implementation decision stated in the thread,
- or a clearly stated workaround/assumption that fills a concrete missing slot.

Reject:
- unresolved gaps,
- environment/setup issues,
- performance-only complaints,
- reproducer improvements,
- resource-only requests,
- implementation bugs / code defects (route separately, not main spec-gap),
- hyperparameter tuning advice / best-practice tricks without a paper-described missing slot,
- ordinary reproducibility seed questions unless they clarify a paper training protocol,
- code-only repo usage gaps that do not map to a method-core paper specification slot.

Also reject code-only or repo-usage gaps that do NOT map to a method-core paper specification slot, such as:
- repository API signatures,
- how to invoke a repo script,
- how to save outputs,
- debugging user mistakes,
- dependency/environment problems,
unless the thread clarifies a paper-described training, evaluation, preprocessing, inference, or model-behavior protocol.

Return STRICT JSON only.
"""


TAXONOMY_SYSTEM_PROMPT = """
You are an expert ML taxonomy annotator for IDEAAMBIG.

Your job is only taxonomy assignment for already-kept atomic gaps.

Critical constraints:
- Do NOT rewrite or add gold_clarified_detail.
- Do NOT expand scope beyond the provided atomic gap and issue evidence.
- Use only the allowed Level-2 labels from TAXONOMY_BRIEF.
- If uncertain, prefer Ambiguity or Incompleteness over Inconsistency unless explicit contradiction evidence exists.
- Use Inconsistency only when two concrete sources conflict, such as paper vs code, paper vs README, paper vs appendix, or two concrete implementation descriptions.
- For evaluation metric computation, evaluation prompt sets, evaluation thresholds, evaluation split, number of evaluation samples/seeds, evaluator configuration, or protocol for reproducing reported numbers, use missing evaluation protocol when the detail is absent.
- For input normalization, augmentation, tokenization, data filtering, data construction, label construction, segmentation, stride/windowing, or preprocessing before model input, use missing data/preprocessing protocol when the detail is absent.

Return STRICT JSON only.
"""


TOPIC_CLUSTERS = [
    ("eval_split", ["split", "train/eval", "9:1", "90%", "10%", "train/test", "validation"]),
    ("eval_fid", ["fid", "inception", "raster"]),
    ("eval_clip", ["clipscore", "clip model", "cosine similarity"]),
    ("eval_aesthetic", ["aesthetic", "laion"]),
    ("eval_prompt", ["prompt set", "drawbench", "test prompts", "table 2"]),
    ("eval_threshold", ["threshold", "validator", "max_det", "map"]),
    ("data_preprocess", ["preprocess", "normalization", "normalize", "rgb", "augmentation", "jpeg", "blur"]),
    ("data_windowing", ["stride", "windowing", "max_length", "segmentation"]),
    ("embedding_viz", ["embedding", "umap", "tsne", "visualization"]),
    ("interpretability", ["gamma", "attention mask", "feature importance"]),
    ("pretraining", ["pretraining_ratio", "pretrain", "reconstruction loss"]),
]


REASONING_PATTERNS = [
    r"\bbecause\b",
    r"\btherefore\b",
    r"\bso that\b",
    r"\bthis means\b",
    r"\bin order to\b",
    r"\bwe think\b",
    r"\bwe believe\b",
    r"\bjustification\b",
    r"\bexplanation\b",
]


EXECUTABLE_SIGNALS = [
    "set ",
    "use ",
    "apply ",
    "compute ",
    "return ",
    "pass ",
    "sample ",
    "mask ",
    "split ",
    "train ",
    "evaluate ",
    "with ",
    "=",
    "->",
    ":",
]


SEMANTIC_OVERLAP_THRESHOLD = 0.45
DEFAULT_MAX_GAPS_PER_ISSUE = 2


IMPLEMENTATION_BUG_PATTERNS = [
    r"\bcode-level bug\b",
    r"\brepository had a bug\b",
    r"\bfixed this bug\b",
    r"\bwas a bug\b",
    r"\bhad a bug\b",
    r"\bthere was a bug\b",
    r"\btestbug\b",
    r"\bwrong parameter\b",
    r"\bcomputes msle\b",
    r"\bnot rmsle\b",
    r"\bbatched\b.{0,40}\breward",
    r"\breward\b.{0,40}\bbatched\b",
    r"\breturn\b.{0,40}\bstd_dev",
    r"\bsqrt\(-1\*dt\)",
    r"\bmodified the original code\b",
    r"\bmodified the original repo\b",
]

TUNING_ADVICE_PATTERNS = [
    r"\btrick sharing\b",
    r"\bbest practice\b",
    r"\btuning advice\b",
    r"\bhyperparameter search\b",
    r"\bprocess of hyperparameter search\b",
    r"\bselected via hyperparameter\b",
    r"\bordinary hyperparameter\b",
    r"\bgradually increase\b",
    r"\bstarting with a very small\b",
    r"\btry starting\b",
    r"\bworthwhile\b",
    r"\bresolution-decoupled\b",
    r"\bgradient_accumulation\b",
    r"\bnum_batches_per_epoch\b",
    r"\bdepends on different datasets\b",
    r"\bno substantial difference\b",
    r"\bstrike an equilibrium\b",
    r"\bcomputational efficiency\b",
    r"\bnoise[- ]level\b",
    r"\bnoise level a\b",
    r"\bhyperparameter.{0,20}\bablation\b",
    r"\bablation.{0,20}\bhyperparameter\b",
]

REPRODUCIBILITY_SEED_PATTERNS = [
    r"\bconfigure_seed\b",
    r"\bseed parameter\b",
    r"\bidentical seed\b",
    r"\brandom seed\b",
    r"\breproducible aucs\b",
    r"reproducible_seed",
    r"\bfit method.{0,40}\bseed\b",
]


def gap_core_text_blob(gap: Dict[str, Any]) -> str:
    return " ".join(
        str(gap.get(k) or "")
        for k in (
            "gap_atom_id",
            "gap_summary",
            "gap_quote",
            "solution_summary",
            "solution_quote",
        )
    ).lower()


def gap_text_blob(gap: Dict[str, Any]) -> str:
    return gap_core_text_blob(gap) + " " + str(
        gap.get("why_this_blocks_or_affects_codification") or ""
    ).lower()


def matches_any_pattern(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text) for p in patterns)


def is_implementation_bug_gap(gap: Dict[str, Any]) -> bool:
    text = gap_core_text_blob(gap)
    gap_id = str(gap.get("gap_atom_id") or "").lower()
    if re.search(r"\bbug\b", gap_id):
        return True
    if re.search(r"\bbug\b", text) and any(
        cue in text
        for cue in (
            "fixed",
            "there was a bug",
            "had a bug",
            "repository",
            "original code",
            "original repo",
            "testbug",
        )
    ):
        return True
    return matches_any_pattern(text, IMPLEMENTATION_BUG_PATTERNS)


def is_tuning_advice_gap(gap: Dict[str, Any], issue: Optional[Dict[str, Any]] = None) -> bool:
    text = gap_core_text_blob(gap)
    if matches_any_pattern(text, TUNING_ADVICE_PATTERNS):
        return True
    if issue:
        title = str(issue.get("title") or "").lower()
        if any(x in title for x in ("trick sharing", "efficiency", "low-vram", "tuning trick")):
            if "protocol" not in text and "metric" not in text and "evaluation" not in text:
                return True
    return False


def is_reproducibility_seed_gap(gap: Dict[str, Any]) -> bool:
    return matches_any_pattern(gap_core_text_blob(gap), REPRODUCIBILITY_SEED_PATTERNS)


def postprocess_drop_reason(
    gap: Dict[str, Any],
    issue: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Return rejection reason if gap should not enter main spec-gap benchmark."""
    if is_implementation_bug_gap(gap):
        return "implementation_bug_not_spec_gap"
    if is_tuning_advice_gap(gap, issue):
        return "tuning_or_best_practice_advice"
    if is_reproducibility_seed_gap(gap):
        return "reproducibility_seed_not_method_spec"
    return None


def rank_atomic_gap(gap: Dict[str, Any]) -> Tuple[int, float, int]:
    return (
        strength_score(gap),
        float(gap.get("confidence") or 0.0),
        quote_quality(gap),
    )


def cap_gaps_per_issue(
    gaps: List[Dict[str, Any]],
    max_gaps: int,
) -> List[Dict[str, Any]]:
    if max_gaps <= 0 or len(gaps) <= max_gaps:
        return gaps
    ranked = sorted(gaps, key=rank_atomic_gap, reverse=True)
    kept = ranked[:max_gaps]
    for i, g in enumerate(kept, start=1):
        g["per_issue_cap_rank"] = i
        g["per_issue_cap_decision"] = f"keep_top_{max_gaps}"
    return kept


def safe_json_dumps(obj: Any) -> str:
    """JSON dumps with safe fallback for accidental sets or Paths."""
    def default(o: Any):
        if isinstance(o, set):
            return sorted(o)
        if isinstance(o, Path):
            return str(o)
        return str(o)

    return json.dumps(obj, ensure_ascii=False, default=default)


def sanitize_for_json(obj: Any) -> Any:
    """Recursively remove/convert non-JSON-serializable objects."""
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, set):
        return sorted(sanitize_for_json(x) for x in obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def topic_clusters_for_gap(gap: Dict[str, Any]) -> Set[str]:
    text = " ".join(
        str(gap.get(k) or "")
        for k in ("gap_summary", "gap_quote", "solution_summary", "solution_quote")
    ).lower()

    hits: Set[str] = set()
    for cid, terms in TOPIC_CLUSTERS:
        if any(t in text for t in terms):
            hits.add(cid)
    return hits


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def validate_solution_quote_executable(
    solution_quote: str,
    gap: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    """
  Light validation for operational solution quotes.
  Strong resolved gaps with substantive quotes are allowed even without explicit verbs.
    """
    q = str(solution_quote or "").strip()
    if not q:
        return False, ["solution_quote_empty"]

    low = q.lower()
    issues: List[str] = []

    has_executable_signal = any(sig in low for sig in EXECUTABLE_SIGNALS)
    strong_gap = (
        str((gap or {}).get("candidate_strength_hint") or "").lower() == "strong"
        and float((gap or {}).get("confidence") or 0.0) >= 0.85
        and len(q) >= 40
    )
    concrete_protocol_terms = [
        "incorporate",
        "removed",
        "split",
        "ratio",
        "metric",
        "loss",
        "hyperparameter",
        "protocol",
        "trained on",
        "evaluat",
        "normaliz",
        "window",
        "stride",
        "catboost",
        "guidance",
    ]
    has_concrete_protocol = any(term in low for term in concrete_protocol_terms)

    if not has_executable_signal and not (strong_gap and has_concrete_protocol):
        issues.append("missing_executable_signal")

    reasoning_hits = []
    for pattern in REASONING_PATTERNS:
        if re.search(pattern, low):
            reasoning_hits.append(f"reasoning_pattern:{pattern}")

    if reasoning_hits and issues:
        issues.extend(reasoning_hits)

    return not issues, issues


def strength_score(gap: Dict[str, Any]) -> int:
    val = str(gap.get("candidate_strength_hint") or "").lower()
    return {"strong": 3, "borderline": 2, "weak": 1}.get(val, 0)


def quote_quality(gap: Dict[str, Any]) -> int:
    return len(str(gap.get("solution_quote") or "").strip())


def semantic_dedup_atomic_gaps(atomic_gaps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate near-duplicate atomic gaps by rough topic clusters.
    Important fix: remove internal set fields before returning, otherwise JSONL writing fails.
    """
    if len(atomic_gaps) <= 1:
        return [sanitize_for_json(g) for g in atomic_gaps]

    enriched: List[Dict[str, Any]] = []
    for g in atomic_gaps:
        x = dict(g)
        cluster_set = topic_clusters_for_gap(g)
        x["_cluster_set"] = cluster_set
        x["topic_clusters"] = sorted(cluster_set)
        enriched.append(x)

    groups: List[List[Dict[str, Any]]] = []
    for g in enriched:
        placed = False
        for group in groups:
            if any(
                jaccard(g["_cluster_set"], member["_cluster_set"]) > SEMANTIC_OVERLAP_THRESHOLD
                for member in group
            ):
                group.append(g)
                placed = True
                break
        if not placed:
            groups.append([g])

    kept: List[Dict[str, Any]] = []
    for gi, group in enumerate(groups, start=1):
        ranked = sorted(
            group,
            key=lambda x: (
                strength_score(x),
                float(x.get("confidence") or 0.0),
                quote_quality(x),
            ),
            reverse=True,
        )
        best = dict(ranked[0])
        best["semantic_cluster_id"] = gi
        best["semantic_dedup_decision"] = (
            "keep_strongest_only" if len(group) > 1 else "keep_unique"
        )
        best.pop("_cluster_set", None)
        kept.append(sanitize_for_json(best))

    return kept


def should_keep_atomic_gap(result: Dict[str, Any]) -> bool:
    if result.get("keep") is False:
        return False
    if result.get("is_real_spec_gap") is False:
        return False

    keep_criteria = result.get("keep_criteria") or {}
    required = ["is_spec_gap", "is_actionable", "is_method_core_spec_gap", "evidence_extractable"]
    if not all(keep_criteria.get(k) is True for k in required):
        return False

    if not str(result.get("gap_quote") or "").strip():
        return False
    if not str(result.get("solution_quote") or "").strip():
        return False

    if result.get("is_resolved") is False:
        return False

    executable_ok, _ = validate_solution_quote_executable(
        str(result.get("solution_quote") or ""),
        gap=result,
    )
    if not executable_ok:
        return False

    return True


def should_keep_result(result: Dict[str, Any]) -> bool:
    if isinstance(result.get("atomic_gaps"), list):
        return any(should_keep_atomic_gap(g) for g in result["atomic_gaps"])
    return should_keep_atomic_gap(result)


def build_atomic_stub_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "keep_criteria": result.get("keep_criteria", {}),
        "affected_component_hint": result.get("affected_component_hint"),
        "gap_summary": result.get("gap_summary"),
        "gap_quote": result.get("gap_quote"),
        "solution_summary": result.get("solution_summary"),
        "solution_quote": result.get("solution_quote"),
        "why_this_blocks_or_affects_codification": result.get(
            "why_this_blocks_or_affects_codification", ""
        ),
        "candidate_strength_hint": result.get("candidate_strength_hint"),
        "confidence": result.get("confidence"),
    }


def iter_atomic_gaps(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    atomic = result.get("atomic_gaps")
    if isinstance(atomic, list) and atomic:
        return [g for g in atomic if isinstance(g, dict)]

    if should_keep_atomic_gap(result):
        legacy = dict(result)
        legacy.setdefault("gap_atom_id", "gap_001")
        return [legacy]

    return []


def issue_key(issue: Dict[str, Any]) -> str:
    record_id = str(issue.get("record_id") or "").strip()
    if record_id:
        return record_id

    issue_url = str(issue.get("issue_url") or "").strip()
    if issue_url:
        return issue_url

    repo = str(issue.get("repo") or "").strip()
    issue_number = issue.get("issue_number")
    if repo and issue_number is not None:
        return f"{repo}#{issue_number}"

    return safe_json_dumps(
        {"repo": repo, "issue_number": issue_number, "title": issue.get("title")}
    )


def build_prompt(issue: Dict[str, Any]) -> str:
    comments = issue.get("comments", [])[:10]
    comment_bodies = []
    for c in comments:
        if isinstance(c, dict):
            comment_bodies.append({
                "author": c.get("user", {}).get("login") if isinstance(c.get("user"), dict) else c.get("author"),
                "author_association": c.get("author_association"),
                "body": c.get("body", ""),
            })
        else:
            comment_bodies.append({"body": str(c)})

    return f"""
We are constructing benchmark instances for IDEAAMBIG: idea/specification ambiguity resolution.

You are given a closed/answered GitHub issue from a paper-linked repository.
Your task is to decide whether this issue contains one or more REAL, RESOLVED, ATOMIC method-core specification gaps,
and if yes extract up to 3 atomic gap candidates.

ATOMIC GAP RULE:
- One atomic gap = one independent missing/ambiguous/inconsistent implementation slot.
- NEVER merge architecture + hyperparameter protocol + training + preprocessing + evaluation into one candidate.
- If the thread resolves multiple independent slots, return multiple atomic_gaps entries, up to 3.
- Each atomic gap must have its own gap_quote and solution_quote.
- Evaluation-only gaps about reproducing reported numbers can be kept if they define metric computation, evaluation split, prompt set, sampling, threshold, or evaluator configuration.
- Reject atomic gaps that are only repo/API usage unless they clarify a paper-described protocol.

Benchmark keep criteria, all must be true for each kept atomic gap:
- is_spec_gap
- is_actionable
- is_method_core_spec_gap
- evidence_extractable

Resolved real gap requires concrete solution evidence in the issue/comments.
Reject unresolved gaps, environment/setup issues, performance-only complaints, resource requests, reproducer improvements, implementation bugs, and tuning/best-practice advice.

Also reject:
- implementation bugs or repo code defects (mark keep=false with rejection_reason=implementation_bug_not_spec_gap),
- hyperparameter tuning advice / efficiency tricks / batch-size tuning without a paper-described missing slot,
- ordinary reproducibility seed usage unless tied to a paper training protocol,
- repo/API usage gaps unless they clarify a paper-described protocol.

Step4 output scope:
- Gate only, no final taxonomy, no final gold rewrite.
- If keep_issue=true, return atomic_gaps with concrete evidence quotes.
- Atomic split is mandatory for mixed issues.
- solution_quote should contain an executable or operational specification, not only reasoning/explanation.

INPUT:
title: {issue.get("title")}
body: {issue.get("body")}
repo: {issue.get("repo")}
paper_title_guess: {issue.get("paper_title_guess")}
paper_url_guess: {issue.get("paper_url_guess")}
state: {issue.get("state")}
labels: {issue.get("label_names")}
comments:
{safe_json_dumps(comment_bodies)}

Return STRICT JSON only:
{{
  "keep_issue": true/false,
  "is_real_spec_gap": true/false,
  "rejection_reason": "resource_only|performance_only|ordinary_hyperparameter_value_only|uses_author_code_without_details|reproducer_improvement_not_spec_gap|insufficient_gap_evidence|insufficient_solution_evidence|not_method_core|code_only_not_paper_spec_gap|implementation_bug_not_spec_gap|tuning_or_best_practice_advice|reproducibility_seed_not_method_spec|composite_unsplit_gap|unresolved_gap|environment_or_setup|other|null",
  "atomic_gaps": [
    {{
      "gap_atom_id": "short_snake_case_id",
      "keep": true/false,
      "is_real_spec_gap": true/false,
      "keep_criteria": {{
        "is_spec_gap": true/false,
        "is_actionable": true/false,
        "is_method_core_spec_gap": true/false,
        "evidence_extractable": true/false
      }},
      "is_resolved": true/false,
      "affected_component_hint": "input|output|core_method|algorithm|training|evaluation|implementation_detail|model_architecture|hyperparameter|code_behavior|preprocessing|data|postprocessing|inference|null",
      "gap_summary": "",
      "gap_quote": "",
      "solution_summary": "",
      "solution_quote": "",
      "why_this_blocks_or_affects_codification": "",
      "candidate_strength_hint": "strong|borderline|weak",
      "confidence": 0.0
    }}
  ]
}}

If only one gap exists, still return atomic_gaps with exactly one entry.
Do not output taxonomy labels in Step4.
"""


def call_llm(client: OpenAI, model: str, prompt: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def build_taxonomy_prompt(issue: Dict[str, Any], atomic_gap: Dict[str, Any]) -> str:
    comments = issue.get("comments", [])[:10]
    comment_bodies = []
    for c in comments:
        if isinstance(c, dict):
            comment_bodies.append({
                "author": c.get("user", {}).get("login") if isinstance(c.get("user"), dict) else c.get("author"),
                "author_association": c.get("author_association"),
                "body": c.get("body", ""),
            })
        else:
            comment_bodies.append({"body": str(c)})

    allowed = "|".join(sorted(VALID_LEVEL2))

    return f"""
Assign taxonomy labels for one already-kept atomic gap.

Taxonomy brief:
{TAXONOMY_BRIEF}

GitHub issue boundary:
{GITHUB_TAXONOMY_BOUNDARY}

Important label-selection reminders:
- Use "missing evaluation protocol" for missing metric computation, evaluation data split, evaluation prompt set, evaluation threshold, evaluation sampling, number of evaluation seeds/samples, or evaluator configuration.
- Use "missing data/preprocessing protocol" for missing input normalization, augmentation, tokenization, data filtering, label construction, train/validation data construction, segmentation, stride/windowing, or preprocessing before model input.
- Use "missing algorithmic specification" for core method procedures, training-loop rules, update order, loss routing, interface operation, sampling/update rules, or termination conditions.
- Use "missing hyperparameter protocol" only when the missing issue is how to select/tune/validate a hyperparameter, not merely a single ordinary unreported value.
- Use "missing model architecture" for structural model choices such as activation, normalization, pooling, layer/module type, initialization, readout, dimensional mapping, or module wiring.
- Use inconsistency labels only when two concrete sources conflict.

Issue context:
title: {issue.get("title")}
body: {issue.get("body")}
repo: {issue.get("repo")}
paper_title_guess: {issue.get("paper_title_guess")}
paper_url_guess: {issue.get("paper_url_guess")}
comments:
{safe_json_dumps(comment_bodies)}

Atomic gap candidate:
{safe_json_dumps(atomic_gap)}

Return STRICT JSON only:
{{
  "level2": "{allowed}",
  "taxonomy_reason": "",
  "confidence": 0.0
}}
"""


def call_taxonomy_llm(client: OpenAI, model: str, prompt: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": TAXONOMY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def normalize_taxonomy_result(tax: Dict[str, Any], atomic_gap: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    level2_raw = normalize_level2(str(tax.get("level2") or "").strip())

    if level2_raw not in VALID_LEVEL2:
        level2_raw = "ambiguous method behavior"

    level1 = LEVEL2_TO_LEVEL1.get(level2_raw, "Ambiguity")

    out = {
        "level1": level1,
        "level2": level2_raw,
        "taxonomy_reason": str(tax.get("taxonomy_reason") or "").strip(),
        "taxonomy_confidence": float(tax.get("confidence") or 0.0),
    }

    context_text = ""
    if atomic_gap:
        context_text = " ".join(
            str(atomic_gap.get(k) or "")
            for k in ("gap_summary", "gap_quote", "solution_summary", "solution_quote", "why_this_blocks_or_affects_codification")
        )
    try:
        return apply_taxonomy_correction(out, context_text=context_text)
    except TypeError:
        return apply_taxonomy_correction(out)

def expand_kept_rows(
    issue: Dict[str, Any],
    result: Dict[str, Any],
    client: OpenAI,
    taxonomy_model: str,
    max_gaps_per_issue: int = DEFAULT_MAX_GAPS_PER_ISSUE,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
      - kept_rows for main spec-gap benchmark
      - implementation_bug_rows (tracked separately, not main benchmark)
      - atomic_gap_decisions for audit alignment
    """
    base_key = issue_key(issue)
    kept_rows: List[Dict[str, Any]] = []
    implementation_bug_rows: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    gate_passed: List[Dict[str, Any]] = []

    for atomic in iter_atomic_gaps(result):
        gap_atom_id = str(atomic.get("gap_atom_id") or "").strip() or "gap_unknown"
        decision: Dict[str, Any] = {
            "gap_atom_id": gap_atom_id,
            "llm_keep": atomic.get("keep"),
            "llm_is_real_spec_gap": atomic.get("is_real_spec_gap"),
            "candidate_strength_hint": atomic.get("candidate_strength_hint"),
            "confidence": atomic.get("confidence"),
            "final_keep_main_benchmark": False,
            "final_track": None,
            "drop_reason": None,
        }

        if not should_keep_atomic_gap(atomic):
            decision["drop_reason"] = "failed_gate_criteria"
            decisions.append(decision)
            continue

        executable_ok, executable_issues = validate_solution_quote_executable(
            str(atomic.get("solution_quote") or ""),
            gap=atomic,
        )
        if not executable_ok:
            decision["drop_reason"] = "solution_quote_not_executable"
            decision["executable_issues"] = executable_issues
            decisions.append(decision)
            continue

        track_reason = postprocess_drop_reason(atomic, issue)
        if track_reason == "implementation_bug_not_spec_gap":
            x = dict(atomic)
            x["gap_track"] = "implementation_bug"
            x["drop_reason"] = track_reason
            implementation_bug_rows.append(sanitize_for_json(x))
            decision["final_track"] = "implementation_bug"
            decision["drop_reason"] = track_reason
            decisions.append(decision)
            continue

        if track_reason:
            decision["drop_reason"] = track_reason
            decisions.append(decision)
            continue

        x = dict(atomic)
        x["solution_quote_executable_valid"] = executable_ok
        x["solution_quote_executable_issues"] = executable_issues
        x["topic_clusters"] = sorted(topic_clusters_for_gap(atomic))
        gate_passed.append(x)
        decision["gate_passed"] = True
        decisions.append(decision)

    deduped = semantic_dedup_atomic_gaps(gate_passed)
    deduped_ids = {str(g.get("gap_atom_id") or "") for g in deduped}
    for d in decisions:
        gid = str(d.get("gap_atom_id") or "")
        if d.get("gate_passed") and gid not in deduped_ids:
            d["drop_reason"] = "semantic_dedup"
            d["final_keep_main_benchmark"] = False

    capped = cap_gaps_per_issue(deduped, max_gaps=max_gaps_per_issue)
    kept_ids = {str(g.get("gap_atom_id") or "") for g in capped}

    for d in decisions:
        gid = str(d.get("gap_atom_id") or "")
        if d.get("gate_passed") and gid not in kept_ids:
            d["drop_reason"] = "per_issue_cap"
            d["final_keep_main_benchmark"] = False

    for idx, atomic in enumerate(capped, start=1):
        gap_atom_id = str(atomic.get("gap_atom_id") or f"gap_{idx:03d}").strip()
        gap_atom_id = re.sub(r"[^\w\-]+", "_", gap_atom_id).strip("_") or f"gap_{idx:03d}"

        for d in decisions:
            if str(d.get("gap_atom_id") or "") == gap_atom_id:
                d["final_keep_main_benchmark"] = True
                d["final_track"] = "spec_gap"
                d["drop_reason"] = None

        row = dict(issue)
        row["issue_record_id"] = base_key
        row["record_id"] = f"{base_key}__{gap_atom_id}"
        row["gap_atom_id"] = gap_atom_id
        row["gap_atom_index"] = idx
        row["realgap_id"] = f"{row['record_id']}_resolved_gap_001"
        row["gap_track"] = "spec_gap"

        taxonomy_prompt = build_taxonomy_prompt(issue, atomic)
        taxonomy_raw = call_taxonomy_llm(client, taxonomy_model, taxonomy_prompt)
        taxonomy = normalize_taxonomy_result(taxonomy_raw, atomic_gap=atomic)

        atomic_with_tax = dict(atomic)
        atomic_with_tax["level1"] = taxonomy["level1"]
        atomic_with_tax["level2"] = taxonomy["level2"]
        atomic_with_tax["taxonomy_reason"] = taxonomy.get("taxonomy_reason", "")
        atomic_with_tax["taxonomy_confidence"] = taxonomy.get("taxonomy_confidence", 0.0)
        atomic_with_tax["gap_track"] = "spec_gap"

        for k in (
            "taxonomy_auto_corrected",
            "taxonomy_correction_reason",
            "taxonomy_original_level1",
            "taxonomy_original_level2",
        ):
            if k in taxonomy:
                atomic_with_tax[k] = taxonomy[k]

        atomic_with_tax = apply_taxonomy_correction_to_gap(atomic_with_tax)

        row["step4_gate"] = sanitize_for_json(atomic_with_tax)
        row["step4_taxonomy"] = sanitize_for_json(taxonomy)

        atomic_stub = build_atomic_stub_from_result(atomic_with_tax)
        atomic_stub["level1"] = taxonomy["level1"]
        atomic_stub["level2"] = taxonomy["level2"]
        atomic_stub["taxonomy_reason"] = taxonomy.get("taxonomy_reason", "")
        atomic_stub["taxonomy_confidence"] = taxonomy.get("taxonomy_confidence", 0.0)
        atomic_stub["gap_track"] = "spec_gap"

        for k in (
            "taxonomy_auto_corrected",
            "taxonomy_correction_reason",
            "taxonomy_original_level1",
            "taxonomy_original_level2",
        ):
            if k in taxonomy:
                atomic_stub[k] = taxonomy[k]

        atomic_stub = apply_taxonomy_correction_to_gap(atomic_stub)

        row["atomic_gap"] = sanitize_for_json(atomic_stub)
        kept_rows.append(sanitize_for_json(row))

    return kept_rows, implementation_bug_rows, decisions


def is_noise(issue: Dict[str, Any]) -> bool:
    parts = [issue.get("title", ""), issue.get("body", "")]
    for c in issue.get("comments", [])[:5]:
        if isinstance(c, dict):
            parts.append(c.get("body", ""))
    t = "\n".join(str(x or "") for x in parts).lower()

    noise_terms = [
        "cuda", "oom", "out of memory", "gpu memory",
        "install", "installation", "pip install", "conda",
        "dependency", "requirements.txt", "modulenotfounderror",
        "runtime error", "docker", "permission denied",
        "dataset access", "download failed", "version conflict",
    ]
    return any(x in t for x in noise_terms)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = sanitize_for_json(row)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_processed_ids(audit_path: Path) -> Set[str]:
    processed: Set[str] = set()
    for row in load_jsonl(audit_path):
        key = str(row.get("record_id") or row.get("issue_key") or "").strip()
        if key:
            processed.add(key)
    return processed


def default_audit_path(out_path: str) -> Path:
    p = Path(out_path)
    return p.with_name(p.stem + ".audit.jsonl")


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0

    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def select_candidates(
    input_path: Path,
    processed_ids: Set[str],
    run_limit: Optional[int],
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for row in load_jsonl(input_path):
        key = issue_key(row)
        if key in processed_ids:
            continue
        selected.append(row)
        if run_limit is not None and len(selected) >= run_limit:
            break
    return selected


def default_sidecar_path(out_path: Path, suffix: str) -> Path:
    return out_path.with_name(f"{out_path.stem}.{suffix}.jsonl")


def summarize_final_rejection(decisions: List[Dict[str, Any]]) -> str:
    if not decisions:
        return "no_atomic_gaps_from_llm"
    reasons = [str(d.get("drop_reason") or "") for d in decisions if d.get("drop_reason")]
    if not reasons:
        return "postprocess_filtered_all"
    return max(set(reasons), key=reasons.count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="Real_bench/github_issue_mining/outputs_step3/step3_spec_gap_candidates.jsonl",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="Real_bench/github_issue_mining/outputs_step4/gated_atomic_candidates.jsonl",
    )
    parser.add_argument(
        "--audit_path",
        type=str,
        default="",
        help="Audit log for all processed issues. Default: <out_stem>.audit.jsonl",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_OPENAI_MODEL)
    parser.add_argument(
        "--limit",
        "--max_items",
        type=int,
        default=500,
        dest="limit",
        help="Max new issues to process per run. 0 means no limit.",
    )
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip issues already present in audit log and append new outputs.",
    )
    parser.add_argument(
        "--max_gaps_per_issue",
        type=int,
        default=DEFAULT_MAX_GAPS_PER_ISSUE,
        help="Max spec-gap atomic candidates kept per issue after dedup/ranking.",
    )
    parser.add_argument(
        "--implementation_bug_out_path",
        type=str,
        default="",
        help="Optional sidecar output for implementation_bug gaps. Default: <out_stem>.implementation_bug.jsonl",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    out_path = Path(args.out_path)
    audit_path = Path(args.audit_path) if args.audit_path else default_audit_path(args.out_path)
    implementation_bug_out_path = (
        Path(args.implementation_bug_out_path)
        if args.implementation_bug_out_path
        else default_sidecar_path(out_path, "implementation_bug")
    )
    run_limit = args.limit if args.limit > 0 else None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    implementation_bug_out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.resume:
        if out_path.exists():
            out_path.unlink()
        if audit_path.exists():
            audit_path.unlink()
        if implementation_bug_out_path.exists():
            implementation_bug_out_path.unlink()

    processed_ids = load_processed_ids(audit_path) if args.resume else set()
    candidates = select_candidates(input_path, processed_ids, run_limit)

    client = OpenAI()

    processed_before = len(processed_ids)
    total = 0
    kept = 0
    rejected = 0
    implementation_bug = 0
    errors = 0

    print("\n===== Step 4: Gate + Atomic Split + Taxonomy =====")
    print(f"Input path:                 {input_path}")
    print(f"Output path:                {out_path}")
    print(f"Implementation bug path:    {implementation_bug_out_path}")
    print(f"Audit path:                 {audit_path}")
    print(f"Model:                      {args.model}")
    print(f"Max gaps per issue:         {args.max_gaps_per_issue}")
    print(f"Resume:            {args.resume}")
    print(f"Limit new:         {run_limit if run_limit is not None else 'none'}")
    print(f"Already processed: {processed_before}")
    print(f"To process now:    {len(candidates)}")
    print("=========================================================\n")

    for issue in tqdm(candidates, desc="Step4 gate", unit="issue"):
        key = issue_key(issue)
        total += 1

        audit_row: Dict[str, Any] = {
            "record_id": key,
            "issue_key": key,
            "repo": issue.get("repo"),
            "issue_number": issue.get("issue_number"),
            "issue_url": issue.get("issue_url"),
            "title": issue.get("title"),
            "status": "rejected",
        }

        if is_noise(issue):
            rejected += 1
            audit_row["status"] = "noise"
            audit_row["step4_llm"] = {"keep": False, "reason": "noise_filter"}
            append_jsonl(audit_path, audit_row)
            continue

        try:
            prompt = build_prompt(issue)
            result = call_llm(client, args.model, prompt)
            result = sanitize_for_json(result)

            audit_row["step4_gate_llm"] = result

            kept_rows, bug_rows, gap_decisions = expand_kept_rows(
                issue,
                result,
                client,
                args.model,
                max_gaps_per_issue=args.max_gaps_per_issue,
            )
            audit_row["atomic_gap_decisions"] = gap_decisions

            if bug_rows:
                audit_row["implementation_bug_gap_count"] = len(bug_rows)
                audit_row["implementation_bug_gap_ids"] = [
                    r.get("gap_atom_id") for r in bug_rows
                ]
                for bug_row in bug_rows:
                    append_jsonl(implementation_bug_out_path, {
                        "issue_record_id": key,
                        "record_id": f"{key}__{bug_row.get('gap_atom_id')}",
                        "gap_atom_id": bug_row.get("gap_atom_id"),
                        "gap_track": "implementation_bug",
                        "repo": issue.get("repo"),
                        "issue_number": issue.get("issue_number"),
                        "issue_url": issue.get("issue_url"),
                        "title": issue.get("title"),
                        "atomic_gap": bug_row,
                    })
                implementation_bug += len(bug_rows)

            if kept_rows:
                audit_row["status"] = "kept"
                audit_row["atomic_gap_count"] = len(kept_rows)
                audit_row["gap_atom_ids"] = [r.get("gap_atom_id") for r in kept_rows]
                audit_row["step4_taxonomy"] = [
                    {
                        "gap_atom_id": r.get("gap_atom_id"),
                        "level1": (r.get("step4_taxonomy") or {}).get("level1"),
                        "level2": (r.get("step4_taxonomy") or {}).get("level2"),
                        "taxonomy_confidence": (r.get("step4_taxonomy") or {}).get("taxonomy_confidence"),
                        "taxonomy_auto_corrected": (r.get("step4_taxonomy") or {}).get("taxonomy_auto_corrected"),
                    }
                    for r in kept_rows
                ]

                for row in kept_rows:
                    append_jsonl(out_path, row)

                kept += len(kept_rows)
            else:
                rejected += 1
                audit_row["status"] = "rejected"
                audit_row["final_rejection_reason"] = summarize_final_rejection(gap_decisions)

        except Exception as e:
            errors += 1
            rejected += 1
            audit_row["status"] = "error"
            audit_row["error"] = repr(e)
            tqdm.write(f"error [{key}]: {e}")

        append_jsonl(audit_path, audit_row)
        time.sleep(args.sleep)

    print("\n[done]")
    print("resume:", args.resume)
    print("limit new:", run_limit if run_limit is not None else "none")
    print("processed this run:", total)
    print("kept this run:", kept)
    print("rejected this run:", rejected)
    print("implementation_bug this run:", implementation_bug)
    print("errors this run:", errors)
    print("kept total:", count_jsonl_lines(out_path))
    print("implementation_bug total:", count_jsonl_lines(implementation_bug_out_path))
    print("audit total:", count_jsonl_lines(audit_path))
    print("output:", out_path)
    print("implementation_bug:", implementation_bug_out_path)
    print("audit:", audit_path)


if __name__ == "__main__":
    main()
