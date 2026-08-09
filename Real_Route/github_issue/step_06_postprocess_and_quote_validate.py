from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from taxonomy_utils import apply_taxonomy_correction_to_gap, assess_taxonomy_mislabel, align_gap_labels_with_level2


DEFAULT_INPUT = (
    "Real_bench/github_issue_mining/outputs_step5/"
    "verified_atomic_instances_with_original_text.jsonl"
)
DEFAULT_OUT_DIR = "Real_bench/github_issue_mining/outputs_step6"

DEFAULT_OPENAI_MODEL = (
    os.getenv("OPENROUTER_MODEL")
    or os.getenv("OPENAI_MODEL")
    or "deepseek/deepseek-v4-pro"
)


VALID_LEVEL2_BY_LEVEL1 = {
    "Ambiguity": {
        "Ambiguous Definition",
        "Ambiguous Procedure",
    },
    "Incompleteness": {
        "Missing Method Procedure",
        "Missing Model Structure",
        "Missing Data Specification",
        "Missing Configuration Protocol",
        "Missing Evaluation Specification",
    },
    "Inconsistency": {
        "Conflicting Objective",
        "Conflicting Model Design",
        "Conflicting Formal Definition",
    },
}

LEVEL2_TO_LEVEL1 = {
    level2: level1
    for level1, labels in VALID_LEVEL2_BY_LEVEL1.items()
    for level2 in labels
}

VALID_LEVEL1 = set(VALID_LEVEL2_BY_LEVEL1.keys())
VALID_LEVEL2 = set(LEVEL2_TO_LEVEL1.keys())

VALID_AFFECTED_COMPONENTS = {
    "TASK_AND_IO",
    "CORE_ALGORITHM",
    "MODEL_ARCHITECTURE",
    "OBJECTIVE_AND_SUPERVISION",
    "TRAINING_PROCEDURE",
    "DATA_AND_PREPROCESSING",
    "INFERENCE_AND_DECISION",
    "EVALUATION_PROTOCOL",
    "INTERNAL_CONSISTENCY",
    "NONE",
}

VALID_SOLUTION_SOURCE_TYPES = {
    "author_clarification",
    "maintainer_clarification",
    "code_derived",
    "reproducer_assumption",
    "reproducer_workaround",
}

VALID_CANDIDATE_STRENGTHS = {"strong", "borderline", "weak"}

MAIN_READY_SOLUTION_SOURCES = {
    "author_clarification",
    "maintainer_clarification",
    "code_derived",
}

ORDINARY_HPARAM_TERMS = {
    "learning rate",
    "optimizer",
    "batch size",
    "weight decay",
    "dropout",
    "momentum",
    "epoch",
    "epochs",
    "random seed",
    "seed",
}

GENERIC_GAP_PHRASES = {
    "implementation details were not given",
    "not enough details were provided",
    "details were not provided",
    "not clearly specified",
    "not specified by the authors",
}

CODE_ONLY_SIGNALS = [
    "train_model",
    "timemoerunner",
    "does not accept a",
    "does not accept a `model`",
    "method does not accept",
    "will not be affected",
    "latest commit",
    "main branch",
    "v10postprocess",
    "how to save",
    "passed to the runner",
    "function signature",
    "api does not",
    "troubleshooting model reinitialization",
]

PAPER_PROTOCOL_SIGNALS = [
    "confidence threshold",
    "validation protocol",
    "metric computation",
    "evaluation protocol",
    "evaluation metric",
    "hyperparameter",
    "training protocol",
    "windowing stride",
    "batch composition",
    "dimensionality reduction",
    "loss",
    "architecture",
    "preprocessing",
    "post-processing",
    "postprocessing",
    "sampling",
    "inference uses",
    "ode",
    "sde",
    "prompt set",
    "drawbench",
    "pickscore",
    "fid",
    "inception",
    "rasterize",
]

COMPOSITE_TOPIC_CLUSTERS = [
    ("embedding_viz", ["internal embedding", "visualization", "umap", "tsne", "before the final mapping"]),
    ("interpretability", ["gamma controls", "attention mask", "feature importance", "interpretability", "mask can be interpreted"]),
    ("pretraining", ["pretraining_ratio", "pretrain", "reconstruction loss", "fraction of features masked", "p_s"]),
    ("eval_split", ["split", "9:1", "train:eval", "train/eval", "train/evaluation"]),
    ("eval_fid", ["fid", "inceptionv3", "inception", "rasteriz"]),
    ("eval_clip", ["clipscore", "clip model", "cosine similarity"]),
    ("eval_aesthetic", ["aesthetic score", "laion"]),
]

DEDUP_KEY_PHRASES = [
    "std_dev",
    "sqrt",
    "logprob",
    "sde",
    "pretraining_ratio",
    "gamma",
    "internal embedding",
    "fid",
    "clipscore",
    "pickscore",
    "windowing",
    "stride",
    "drawbench",
    "openpose feet",
    "interfacengan",
    "q(x)",
]

FORBIDDEN_GOLD_PATTERNS = [
    r"\bbecause\b",
    r"\btherefore\b",
    r"\bhowever\b",
    r"\bcompared to\b",
    r"\bwe assume\b",
    r"\bmight\b",
    r"\bmay\b",
    r"\bcould\b",
    r"\bin general\b",
    r"\bfor example\b",
    r"\bgithub\b",
    r"\bissue\b",
    r"\bauthor says\b",
    r"\bmaintainer says\b",
    r"\bpaper is ambiguous\b",
    r"\bnot specified\b",
    r"\bunclear\b",
]

STEP6_CLEANUP_SYSTEM_PROMPT = """
You are a strict post-filter and gold-spec cleanup assistant for IDEAAMBIG.

You are given one verified real-gap candidate, deterministic validation flags,
the GitHub issue thread, and the original paper excerpt.

Your job is limited:
1) repair gold_clarified_detail into a minimal implementation-ready specification,
2) optionally correct taxonomy if it is clearly wrong,
3) decide whether the candidate should be main_resolved, review_needed, or rejected.

Hard constraints:
- Do NOT introduce new technical details not supported by the issue thread or paper excerpt.
- Do NOT rewrite gap_quote or solution_quote.
- Do NOT change evidence.
- Do NOT keep implementation bugs, repo usage issues, unresolved gaps, or pure tuning advice as main benchmark instances.
- gold_clarified_detail must be 1-3 sentences.
- gold_clarified_detail must be concise and executable.
- gold_clarified_detail must not contain reasoning words such as because, therefore, however, may, might, could, for example.
- If the original evidence is insufficient, reject or mark review_needed.

Allowed Level-2 labels:
- Ambiguous Definition
- Ambiguous Procedure
- Missing Method Procedure
- Missing Configuration Protocol
- Missing Model Structure
- Missing Evaluation Specification
- Missing Data Specification
- Conflicting Objective
- Conflicting Model Design
- Conflicting Formal Definition

Labeling rules:
- Use Missing Evaluation Specification for metric computation, evaluation split, prompt set, threshold, sampling, seed/sample count, or evaluator configuration.
- Use Missing Data Specification for data construction, preprocessing, segmentation, stride/windowing, filtering, label construction, tokenization, or normalization.
- Use Missing Method Procedure for method procedure, training-loop rule, update order, loss routing, sampling rule, or termination condition.
- Use Ambiguous Procedure when the method permits multiple plausible operational behaviors.
- Use Ambiguous Definition when a mathematical/formal variable, sign, convention, or definition is unclear.
- Use Inconsistency only when two concrete sources conflict.

Return STRICT JSON only.
"""


def sentence_count(text: str) -> int:
    parts = re.split(r"(?<=[.!?])\s+", str(text or "").strip())
    return len([p for p in parts if p.strip()])


def validate_minimal_gold(g: Dict[str, Any]) -> Dict[str, Any]:
    gold = str(g.get("gold_clarified_detail") or "")
    sc = sentence_count(gold)
    issues: List[str] = []

    if sc < 1 or sc > 3:
        issues.append(f"gold_sentence_count_{sc}")

    if len(gold.split()) > 120:
        issues.append("gold_too_long")

    low = gold.lower()
    for pattern in FORBIDDEN_GOLD_PATTERNS:
        if re.search(pattern, low):
            issues.append(f"gold_forbidden:{pattern}")

    return {
        "gold_minimal_valid": not issues,
        "gold_sentence_count": sc,
        "issues": issues,
    }


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


def save_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def resume_key(row: Dict[str, Any]) -> str:
    """Stable key for skipping records already written to a Step6 split."""
    for key in ("realgap_id", "record_id", "issue_record_id", "gap_atom_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return f"{key}:{value}"

    gap = row.get("gap") or row.get("atomic_gap") or row.get("step4_gate") or {}
    if isinstance(gap, dict):
        for key in ("realgap_id", "record_id", "gap_atom_id"):
            value = str(gap.get(key) or "").strip()
            if value:
                return f"gap.{key}:{value}"

    source = row.get("source") or {}
    if isinstance(source, dict):
        for key in ("realgap_id", "record_id"):
            value = str(source.get(key) or "").strip()
            if value:
                return f"source.{key}:{value}"

    return "content:" + json.dumps(row, sort_keys=True, ensure_ascii=False)


def load_resume_splits(out_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], set[str]]:
    main_resolved = load_jsonl(out_dir / "main_resolved.jsonl")
    review_needed = load_jsonl(out_dir / "review_needed.jsonl")
    rejected = load_jsonl(out_dir / "rejected.jsonl")

    seen: set[str] = set()
    for rows in (main_resolved, review_needed, rejected):
        deduped: List[Dict[str, Any]] = []
        for row in rows:
            key = resume_key(row)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        rows[:] = deduped

    return main_resolved, review_needed, rejected, seen


def save_step6_outputs(
    out_dir: Path,
    main_resolved: List[Dict[str, Any]],
    review_needed: List[Dict[str, Any]],
    rejected: List[Dict[str, Any]],
) -> None:
    save_jsonl(main_resolved, out_dir / "main_resolved.jsonl")
    save_jsonl(review_needed, out_dir / "review_needed.jsonl")
    save_jsonl(rejected, out_dir / "rejected.jsonl")


def normalize_for_match(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def read_text(path: str, max_chars: int = 120000) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", errors="ignore")
    return text[:max_chars]


def build_issue_thread_text(record: Dict[str, Any]) -> str:
    parts: List[str] = []

    title = str(record.get("title") or "").strip()
    body = str(record.get("body") or "").strip()

    if title:
        parts.append(title)
    if body:
        parts.append(body)

    for comment in record.get("comments") or []:
        if not isinstance(comment, dict):
            continue
        cbody = str(comment.get("body") or "").strip()
        if cbody:
            parts.append(cbody)

    return "\n\n".join(parts)


def quote_in_text(quote: str, text: str) -> bool:
    q = normalize_for_match(quote)
    t = normalize_for_match(text)

    if not q:
        return False

    if q in t:
        return True

    words = [w for w in re.findall(r"\w+", q) if len(w) > 3]
    if len(words) < 4:
        return False

    hit = sum(1 for w in words if w in t)
    return hit / max(len(words), 1) >= 0.75


def split_quote_segments(quote: str) -> List[str]:
    parts = re.split(r"\.\.\.+", str(quote or ""))
    return [p.strip() for p in parts if p.strip()]


def quote_in_text_with_ellipsis(quote: str, text: str) -> Dict[str, Any]:
    quote = str(quote or "").strip()

    if not quote:
        return {
            "matched": False,
            "has_ellipsis": False,
            "segments": [],
            "matched_segments": 0,
            "total_segments": 0,
        }

    has_ellipsis = "..." in quote

    if not has_ellipsis:
        matched = quote_in_text(quote, text)
        return {
            "matched": matched,
            "has_ellipsis": False,
            "segments": [quote],
            "matched_segments": 1 if matched else 0,
            "total_segments": 1,
        }

    segments = split_quote_segments(quote)
    if not segments:
        return {
            "matched": False,
            "has_ellipsis": True,
            "segments": [],
            "matched_segments": 0,
            "total_segments": 0,
        }

    matched = [quote_in_text(seg, text) for seg in segments]
    matched_count = sum(1 for x in matched if x)

    longest = max(segments, key=len)
    longest_ok = quote_in_text(longest, text)
    all_ok = matched_count == len(segments)
    half_ok = matched_count >= max(1, len(segments) // 2)

    return {
        "matched": bool(all_ok or (longest_ok and half_ok)),
        "has_ellipsis": True,
        "segments": segments,
        "matched_segments": matched_count,
        "total_segments": len(segments),
        "longest_segment_matched": longest_ok,
    }


def contains_any(text: str, terms: set[str]) -> List[str]:
    low = text.lower()
    return sorted(t for t in terms if t in low)


def is_generic_gap_quote(quote: str) -> bool:
    q = normalize_for_match(quote)

    if not q:
        return True

    for phrase in GENERIC_GAP_PHRASES:
        if phrase in q:
            return True

    vague_terms = ["not specified", "not clear", "not provided", "no details"]
    has_vague = any(t in q for t in vague_terms)

    has_anchor = any(
        t in q
        for t in [
            "encoder",
            "decoder",
            "loss",
            "architecture",
            "layer",
            "hyperparameter",
            "training",
            "evaluation",
            "preprocessing",
            "batch",
            "prompt",
            "latent",
            "dimension",
            "parameter",
            "config",
            "metric",
            "dataset",
            "stride",
            "fid",
            "clipscore",
        ]
    )

    return has_vague and not has_anchor


def validate_taxonomy_label(level1: str, level2: str) -> bool:
    return level2 in VALID_LEVEL2_BY_LEVEL1.get(level1, set())


def contains_signal(text: str, signal: str) -> bool:
    text = str(text or "").lower()
    signal = signal.lower()

    if len(signal) <= 4 or signal in {"ode", "sde", "loss", "api", "fid"}:
        return bool(re.search(rf"\b{re.escape(signal)}\b", text))

    return signal in text


def assess_paper_core_gap(g: Dict[str, Any]) -> Tuple[bool, List[str]]:
    text = " ".join(
        str(g.get(k) or "")
        for k in (
            "gap_summary",
            "gap_quote",
            "solution_summary",
            "solution_quote",
            "gold_clarified_detail",
            "why_this_blocks_or_affects_codification",
        )
    ).lower()

    affected = str(g.get("affected_component") or "").lower()
    reasons: List[str] = []

    code_hits = [signal for signal in CODE_ONLY_SIGNALS if contains_signal(text, signal)]
    paper_hits = [
        signal for signal in PAPER_PROTOCOL_SIGNALS if contains_signal(text, signal)
    ]

    if code_hits and not paper_hits:
        reasons.append(f"code_only_not_paper_spec_gap:{code_hits[0]}")
        return False, reasons

    return True, reasons


def assess_composite_gap(g: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    text = " ".join(
        str(g.get(k) or "")
        for k in (
            "gap_summary",
            "gap_quote",
            "gold_clarified_detail",
            "solution_summary",
        )
    ).lower()

    hits: List[str] = []
    for cluster_id, terms in COMPOSITE_TOPIC_CLUSTERS:
        if any(term in text for term in terms):
            hits.append(cluster_id)

    reasons: List[str] = []
    is_composite = False

    eval_hits = [h for h in hits if h.startswith("eval_")]
    if len(eval_hits) >= 2:
        is_composite = True
        reasons.append(f"multiple_eval_metrics:{','.join(eval_hits)}")

    interpretability_mix = {
        "embedding_viz",
        "interpretability",
        "pretraining",
    }
    if len(set(hits) & interpretability_mix) >= 2:
        is_composite = True
        reasons.append(
            "interpretability_hyperparameter_training_mix:"
            + ",".join(sorted(set(hits) & interpretability_mix))
        )

    if len(hits) >= 3:
        is_composite = True
        reasons.append(f"multiple_topic_clusters:{','.join(hits)}")

    gap_summary = str(g.get("gap_summary") or "").lower()
    if re.search(r"\bnor (does|did|do)\b", gap_summary):
        is_composite = True
        reasons.append("gap_summary_multi_topic_nor_clause")

    return is_composite, hits, reasons


def gap_dedup_fingerprint(record: Dict[str, Any], g: Dict[str, Any]) -> Optional[str]:
    repo = str(record.get("repo") or "").lower()

    text = " ".join(
        str(g.get(k) or "")
        for k in ("gap_summary", "gold_clarified_detail", "gap_quote")
    ).lower()

    key_phrases = sorted({phrase for phrase in DEDUP_KEY_PHRASES if phrase in text})

    if len(key_phrases) < 2:
        return None

    return f"{repo}::{'|'.join(key_phrases)}"


def gap_dedup_rank(record: Dict[str, Any], g: Dict[str, Any]) -> Tuple[int, float, int]:
    strength_order = {"strong": 3, "borderline": 2, "weak": 1}
    strength = strength_order.get(str(g.get("candidate_strength") or ""), 0)

    try:
        confidence = float(g.get("confidence") or 0.0)
    except Exception:
        confidence = 0.0

    try:
        issue_number = int(record.get("issue_number") or 0)
    except Exception:
        issue_number = 0

    return strength, confidence, -issue_number


def llm_validation_flags(llm: Dict[str, Any]) -> Dict[str, bool]:
    if not llm:
        return {"is_resolved": False, "keep": False, "is_real_spec_gap": False}

    if llm.get("keep_criteria") or llm.get("gap_atom_id"):
        return {
            "is_resolved": llm.get("is_resolved") is True,
            "keep": llm.get("keep") is not False,
            "is_real_spec_gap": llm.get("is_real_spec_gap") is not False,
        }

    atomic = llm.get("atomic_gaps")
    if isinstance(atomic, list) and atomic:
        kept = [a for a in atomic if isinstance(a, dict) and a.get("keep") is not False]
        if kept:
            return {
                "is_resolved": all(a.get("is_resolved") is True for a in kept),
                "keep": llm.get("keep_issue") is True or any(
                    a.get("keep") is True for a in kept
                ),
                "is_real_spec_gap": llm.get("is_real_spec_gap") is True,
            }

    return {
        "is_resolved": llm.get("is_resolved") is True,
        "keep": llm.get("keep") is True or llm.get("keep_issue") is True,
        "is_real_spec_gap": llm.get("is_real_spec_gap") is True,
    }


def get_gap(record: Dict[str, Any]) -> Dict[str, Any]:
    return dict(record.get("gap") or {})


def original_paper_text_available(record: Dict[str, Any]) -> bool:
    stepb = record.get("stepB") or record.get("step5") or {}
    md_status = str(stepb.get("markdown_status") or "")

    if md_status in {"exists", "converted"}:
        return True

    lp = record.get("local_paths") or {}
    md_path = str(
        lp.get("original_paper_fast_text_markdown_path")
        or record.get("original_text_path")
        or ""
    )

    text = read_text(md_path, max_chars=200)
    return len(text.strip()) > 100


def safe_apply_taxonomy_correction(g: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return apply_taxonomy_correction_to_gap(g)
    except Exception:
        return g


def add_borderline_flags(g: Dict[str, Any]) -> Dict[str, Any]:
    g = dict(g)

    text = " ".join(
        str(g.get(k) or "")
        for k in (
            "gap_summary",
            "gap_quote",
            "solution_summary",
            "solution_quote",
            "gold_clarified_detail",
        )
    )

    ordinary = contains_any(text, ORDINARY_HPARAM_TERMS)
    flags = dict(g.get("borderline_flags") or {})

    flags["mentions_ordinary_hparams"] = bool(ordinary)
    flags["ordinary_hparam_terms"] = ordinary
    flags["generic_gap_quote"] = is_generic_gap_quote(g.get("gap_quote", ""))

    flags["needs_hparam_manual_review"] = (
        g.get("level2") == "Missing Configuration Protocol" and bool(ordinary)
    )
    flags["needs_generic_quote_manual_review"] = bool(flags["generic_gap_quote"])
    flags["solution_from_reproducer"] = g.get("solution_source_type") in {
        "reproducer_assumption",
        "reproducer_workaround",
    }

    g["borderline_flags"] = flags
    return g


def validate_gap(
    record: Dict[str, Any],
    issue_text: str,
    original_text: str,
) -> Dict[str, Any]:
    g = safe_apply_taxonomy_correction(add_borderline_flags(get_gap(record)))

    llm = (
        record.get("step5_verifier")
        or record.get("step5_llm")
        or record.get("step4_gate")
        or record.get("step4_llm")
        or {}
    )

    gap_q = str(g.get("gap_quote") or "")
    sol_q = str(g.get("solution_quote") or "")

    gap_match = quote_in_text_with_ellipsis(gap_q, issue_text)
    sol_match = quote_in_text_with_ellipsis(sol_q, issue_text)

    level1 = str(g.get("level1") or "")
    level2 = str(g.get("level2") or "")
    affected = str(g.get("affected_component") or "")
    solution_source = str(g.get("solution_source_type") or "")
    strength = str(g.get("candidate_strength") or "")
    kc = g.get("keep_criteria") or {}

    g["quote_validation"] = {
        "gap_quote_in_issue_thread": gap_match["matched"],
        "solution_quote_in_issue_thread": sol_match["matched"],
        "gap_quote_has_ellipsis": gap_match["has_ellipsis"],
        "solution_quote_has_ellipsis": sol_match["has_ellipsis"],
        "gap_quote_segment_match": {
            "matched_segments": gap_match["matched_segments"],
            "total_segments": gap_match["total_segments"],
        },
        "solution_quote_segment_match": {
            "matched_segments": sol_match["matched_segments"],
            "total_segments": sol_match["total_segments"],
        },
        "gap_quote_nonempty": bool(gap_q.strip()),
        "solution_quote_nonempty": bool(sol_q.strip()),
        "gap_quote_is_generic": g["borderline_flags"]["generic_gap_quote"],
        "validation_target": "github_issue_thread",
    }

    tax_extra = {}
    try:
        tax_extra = assess_taxonomy_mislabel(g)
    except Exception:
        tax_extra = {}

    g["taxonomy_validation"] = {
        "level1": level1,
        "level2": level2,
        "level2_consistent_with_level1": validate_taxonomy_label(level1, level2),
        "affected_component_valid": affected in VALID_AFFECTED_COMPONENTS,
        **tax_extra,
    }

    g["solution_source_validation"] = {
        "solution_source_type": solution_source,
        "solution_source_type_valid": solution_source in VALID_SOLUTION_SOURCE_TYPES,
    }

    g["candidate_strength_validation"] = {
        "candidate_strength": strength,
        "candidate_strength_valid": strength in VALID_CANDIDATE_STRENGTHS,
    }

    g["keep_criteria_validation"] = {
        "is_spec_gap_true": kc.get("is_spec_gap") is True,
        "is_actionable_true": kc.get("is_actionable") is True,
        "is_method_core_spec_gap_true": kc.get("is_method_core_spec_gap") is True,
        "gold_clarified_spec_extractable_true": kc.get(
            "gold_clarified_spec_extractable"
        ) is True,
    }

    g["step5_llm_validation"] = {
        **llm_validation_flags(llm),
    }

    g["original_paper_validation"] = {
        "original_paper_text_available": original_paper_text_available(record),
        "original_text_chars": len(original_text.strip()),
    }

    paper_core, paper_core_reasons = assess_paper_core_gap(g)
    g["paper_core_validation"] = {
        "paper_core_gap": paper_core,
        "reasons": paper_core_reasons,
    }

    g["minimal_gold_validation"] = validate_minimal_gold(g)

    composite, composite_hits, composite_reasons = assess_composite_gap(g)
    g["composite_validation"] = {
        "is_composite_gap": composite,
        "topic_clusters_hit": composite_hits,
        "reasons": composite_reasons,
    }

    g["passes_min_quote_validation"] = (
        g["quote_validation"]["gap_quote_nonempty"]
        and g["quote_validation"]["solution_quote_nonempty"]
        and g["quote_validation"]["gap_quote_in_issue_thread"]
        and g["quote_validation"]["solution_quote_in_issue_thread"]
    )

    g["passes_min_validation"] = (
        g["passes_min_quote_validation"]
        and g["taxonomy_validation"]["level2_consistent_with_level1"]
        and g["taxonomy_validation"]["affected_component_valid"]
        and g["solution_source_validation"]["solution_source_type_valid"]
        and g["candidate_strength_validation"]["candidate_strength_valid"]
        and g["keep_criteria_validation"]["is_spec_gap_true"]
        and g["keep_criteria_validation"]["is_actionable_true"]
        and g["keep_criteria_validation"]["is_method_core_spec_gap_true"]
        and g["keep_criteria_validation"]["gold_clarified_spec_extractable_true"]
        and g["step5_llm_validation"]["is_resolved"]
        and g["original_paper_validation"]["original_paper_text_available"]
        and g["paper_core_validation"]["paper_core_gap"]
        and g["minimal_gold_validation"]["gold_minimal_valid"]
    )

    gap_partial_ellipsis = (
        gap_match["has_ellipsis"]
        and gap_match["matched_segments"] < gap_match["total_segments"]
    )
    sol_partial_ellipsis = (
        sol_match["has_ellipsis"]
        and sol_match["matched_segments"] < sol_match["total_segments"]
    )

    g["recommended_manual_review"] = (
        strength != "strong"
        or g["borderline_flags"]["solution_from_reproducer"]
        or g["borderline_flags"]["needs_hparam_manual_review"]
        or g["borderline_flags"]["needs_generic_quote_manual_review"]
        or g["composite_validation"]["is_composite_gap"]
        or g["taxonomy_validation"].get("taxonomy_mislabel")
        or not g["minimal_gold_validation"]["gold_minimal_valid"]
        or gap_partial_ellipsis
        or sol_partial_ellipsis
    )

    g["main_ready_candidate"] = (
        g["passes_min_validation"]
        and strength == "strong"
        and solution_source in MAIN_READY_SOLUTION_SOURCES
        and not g["recommended_manual_review"]
        and not g["composite_validation"]["is_composite_gap"]
    )

    return g


def classify_record(
    record: Dict[str, Any],
    *,
    duplicate_gap: bool = False,
) -> Tuple[str, List[str]]:
    g = record.get("gap") or {}
    reasons: List[str] = []

    if duplicate_gap:
        reasons.append("duplicate_gap_in_repo")
        return "rejected", reasons

    paper_core = (g.get("paper_core_validation") or {}).get("paper_core_gap")
    if paper_core is False:
        reasons.extend((g.get("paper_core_validation") or {}).get("reasons") or [])
        if not reasons:
            reasons.append("code_only_not_paper_spec_gap")
        return "rejected", reasons

    composite = (g.get("composite_validation") or {}).get("is_composite_gap")
    if composite:
        reasons.extend((g.get("composite_validation") or {}).get("reasons") or [])
        if not reasons:
            reasons.append("composite_gap_needs_split")
        return "review_needed", reasons

    if (g.get("taxonomy_validation") or {}).get("taxonomy_mislabel"):
        tax = g.get("taxonomy_validation") or {}
        reasons.append("taxonomy_mislabel")
        if tax.get("mislabel_reason"):
            reasons.append(str(tax.get("mislabel_reason")))
        if tax.get("suggested_level2"):
            reasons.append(f"suggested_level2:{tax.get('suggested_level2')}")
        return "review_needed", reasons

    if g.get("main_ready_candidate"):
        reasons.append("main_ready_candidate_true")
        return "main_resolved", reasons

    if not g.get("passes_min_validation"):
        qv = g.get("quote_validation") or {}
        if not qv.get("gap_quote_in_issue_thread"):
            reasons.append("gap_quote_not_in_issue_thread")
        if not qv.get("solution_quote_in_issue_thread"):
            reasons.append("solution_quote_not_in_issue_thread")
        if not (g.get("original_paper_validation") or {}).get(
            "original_paper_text_available"
        ):
            reasons.append("original_paper_text_missing")
        if not (g.get("step5_llm_validation") or {}).get("is_resolved"):
            reasons.append("step5_not_resolved")

        tax = g.get("taxonomy_validation") or {}
        if not tax.get("level2_consistent_with_level1"):
            reasons.append("invalid_taxonomy")
        if not tax.get("affected_component_valid"):
            reasons.append("invalid_affected_component")

        mg = g.get("minimal_gold_validation") or {}
        if not mg.get("gold_minimal_valid"):
            reasons.extend(mg.get("issues") or ["gold_not_minimal"])

        if g.get("candidate_strength_validation", {}).get("candidate_strength") == "weak":
            reasons.append("candidate_strength_weak")

        if paper_core is False:
            reasons.extend((g.get("paper_core_validation") or {}).get("reasons") or [])

        if not reasons:
            reasons.append("does_not_pass_min_validation")

        return "rejected", reasons

    reasons.append("not_main_ready_candidate")

    if g.get("candidate_strength") != "strong":
        reasons.append(f"candidate_strength_{g.get('candidate_strength')}")

    if g.get("recommended_manual_review"):
        reasons.append("recommended_manual_review_true")

    src = g.get("solution_source_type")
    if src:
        reasons.append(f"solution_source_{src}")

    return "review_needed", reasons


def add_postprocess_info(
    record: Dict[str, Any],
    split: str,
    reasons: List[str],
) -> Dict[str, Any]:
    out = dict(record)
    out["postprocess_split"] = split
    out["postprocess_reasons"] = reasons
    out["source"] = "GitHub_issue_real_gap"
    return out


def process_record(record: Dict[str, Any]) -> Dict[str, Any]:
    issue_text = build_issue_thread_text(record)

    lp = record.get("local_paths") or {}
    original_path = str(
        lp.get("original_paper_fast_text_markdown_path")
        or record.get("original_text_path")
        or ""
    )
    original_text = read_text(original_path)

    out = dict(record)
    out["gap"] = validate_gap(out, issue_text, original_text)
    return out


def build_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=(
            os.getenv("OPENROUTER_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://openrouter.ai/api/v1"
        ),
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "IdeaAmbig-Bench"),
        },
    )


def call_cleanup_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    max_retries: int = 4,
    sleep: float = 2.0,
) -> Dict[str, Any]:
    last_err = ""

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": STEP6_CLEANUP_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content or "{}")

        except Exception as e:
            last_err = repr(e)
            if attempt < max_retries:
                time.sleep(sleep * attempt)

    raise RuntimeError(f"Step6 cleanup LLM failed after retries: {last_err}")


def build_cleanup_prompt(
    record: Dict[str, Any],
    issue_text: str,
    original_text: str,
) -> str:
    g = record.get("gap") or {}

    compact_validation = {
        "quote_validation": g.get("quote_validation"),
        "taxonomy_validation": g.get("taxonomy_validation"),
        "paper_core_validation": g.get("paper_core_validation"),
        "minimal_gold_validation": g.get("minimal_gold_validation"),
        "composite_validation": g.get("composite_validation"),
        "passes_min_validation": g.get("passes_min_validation"),
        "recommended_manual_review": g.get("recommended_manual_review"),
        "main_ready_candidate": g.get("main_ready_candidate"),
    }

    return f"""
Step6 LLM cleanup for one candidate.

Record ID:
{record.get("record_id")}

Current gap object:
{json.dumps(g, ensure_ascii=False, indent=2)}

Deterministic validation summary:
{json.dumps(compact_validation, ensure_ascii=False, indent=2)}

Original paper excerpt:
\"\"\"
{original_text[:12000]}
\"\"\"

GitHub issue thread excerpt:
\"\"\"
{issue_text[:12000]}
\"\"\"

Your task:
- Keep the same evidence quotes.
- Repair only gold_clarified_detail and taxonomy if needed.
- Decide final split.
- Do not add unsupported implementation details.

Return STRICT JSON only:
{{
  "final_decision": "main_resolved|review_needed|rejected",
  "decision_reason": "",
  "level1": "Ambiguity|Incompleteness|Inconsistency|null",
  "level2": "Ambiguous Definition|Ambiguous Procedure|Missing Method Procedure|Missing Configuration Protocol|Missing Model Structure|Missing Evaluation Specification|Missing Data Specification|Conflicting Objective|Conflicting Model Design|Conflicting Formal Definition|null",
  "affected_component": "TASK_AND_IO|CORE_ALGORITHM|MODEL_ARCHITECTURE|OBJECTIVE_AND_SUPERVISION|TRAINING_PROCEDURE|DATA_AND_PREPROCESSING|INFERENCE_AND_DECISION|EVALUATION_PROTOCOL|INTERNAL_CONSISTENCY|NONE|null",
  "gold_clarified_detail": "",
  "candidate_strength": "strong|borderline|weak",
  "confidence": 0.0,
  "cleanup_notes": ""
}}
"""


def needs_llm_cleanup(record: Dict[str, Any], split: str, reasons: List[str]) -> bool:
    g = record.get("gap") or {}
    mg = g.get("minimal_gold_validation") or {}
    tax = g.get("taxonomy_validation") or {}

    reason_text = " ".join(str(r) for r in reasons).lower()

    if split == "rejected" and ("gold" in reason_text):
        return True

    if not mg.get("gold_minimal_valid", True):
        return True

    if tax.get("taxonomy_mislabel"):
        return True

    if not tax.get("level2_consistent_with_level1", True):
        return True

    if split == "review_needed" and any(
        key in reason_text
        for key in [
            "taxonomy",
            "gold",
            "not_main_ready_candidate",
            "recommended_manual_review_true",
        ]
    ):
        return True

    return False


def normalize_cleanup_level2(level2: Any) -> str:
    label = str(level2 or "").strip()
    for valid in LEVEL2_TO_LEVEL1:
        if label.lower() == valid.lower():
            return valid
    return ""


def apply_llm_cleanup_result(
    record: Dict[str, Any],
    cleanup: Dict[str, Any],
) -> Dict[str, Any]:
    out = dict(record)
    g = dict(out.get("gap") or {})

    old_gold = g.get("gold_clarified_detail")
    old_level1 = g.get("level1")
    old_level2 = g.get("level2")
    old_affected = g.get("affected_component")
    old_strength = g.get("candidate_strength")

    new_gold = str(cleanup.get("gold_clarified_detail") or "").strip()
    if new_gold:
        g["gold_clarified_detail"] = new_gold

    new_level2 = normalize_cleanup_level2(cleanup.get("level2"))
    if new_level2:
        g["level2"] = new_level2
        g["level1"] = LEVEL2_TO_LEVEL1[new_level2]

    new_affected = str(cleanup.get("affected_component") or "").strip().upper().replace(" ", "_")
    new_affected = re.sub(r"_+", "_", new_affected)
    if new_affected in VALID_AFFECTED_COMPONENTS:
        g["affected_component"] = new_affected

    new_strength = str(cleanup.get("candidate_strength") or "").strip()
    if new_strength in VALID_CANDIDATE_STRENGTHS:
        g["candidate_strength"] = new_strength

    try:
        g["confidence"] = float(cleanup.get("confidence") or g.get("confidence") or 0.0)
    except Exception:
        pass

    g = align_gap_labels_with_level2(g)

    g["step6_llm_cleanup"] = {
        "applied": True,
        "final_decision": cleanup.get("final_decision"),
        "decision_reason": cleanup.get("decision_reason"),
        "cleanup_notes": cleanup.get("cleanup_notes"),
        "old_gold_clarified_detail": old_gold,
        "new_gold_clarified_detail": g.get("gold_clarified_detail"),
        "old_level1": old_level1,
        "old_level2": old_level2,
        "new_level1": g.get("level1"),
        "new_level2": g.get("level2"),
        "old_affected_component": old_affected,
        "new_affected_component": g.get("affected_component"),
        "old_candidate_strength": old_strength,
        "new_candidate_strength": g.get("candidate_strength"),
    }

    out["gap"] = g
    out["step6_llm_cleanup"] = g["step6_llm_cleanup"]
    return out


def summarize_splits(
    main_resolved: List[Dict[str, Any]],
    review_needed: List[Dict[str, Any]],
    rejected: List[Dict[str, Any]],
    input_count: int,
) -> Dict[str, Any]:
    def reason_counter(rows: List[Dict[str, Any]]) -> Dict[str, int]:
        c: Counter = Counter()
        for row in rows:
            for reason in row.get("postprocess_reasons", []):
                c[str(reason)] += 1
        return dict(c)

    def strength_counter(rows: List[Dict[str, Any]]) -> Dict[str, int]:
        c: Counter = Counter()
        for row in rows:
            g = row.get("gap") or {}
            c[str(g.get("candidate_strength"))] += 1
        return dict(c)

    def level2_counter(rows: List[Dict[str, Any]]) -> Dict[str, int]:
        c: Counter = Counter()
        for row in rows:
            g = row.get("gap") or {}
            c[str(g.get("level2"))] += 1
        return dict(c)

    def cleanup_counter(rows: List[Dict[str, Any]]) -> Dict[str, int]:
        c: Counter = Counter()
        for row in rows:
            cleanup = row.get("step6_llm_cleanup") or {}
            if cleanup.get("applied"):
                c["applied"] += 1
                decision = cleanup.get("final_decision")
                if decision:
                    c[f"decision_{decision}"] += 1
        return dict(c)

    return {
        "script": "step_06_postprocess_and_quote_validate.py",
        "input_count": input_count,
        "counts": {
            "main_resolved": len(main_resolved),
            "review_needed": len(review_needed),
            "rejected": len(rejected),
            "total": len(main_resolved) + len(review_needed) + len(rejected),
        },
        "main_resolved_strength_distribution": strength_counter(main_resolved),
        "review_needed_strength_distribution": strength_counter(review_needed),
        "rejected_strength_distribution": strength_counter(rejected),
        "main_resolved_level2_distribution": level2_counter(main_resolved),
        "review_needed_level2_distribution": level2_counter(review_needed),
        "rejected_level2_distribution": level2_counter(rejected),
        "review_needed_reasons": reason_counter(review_needed),
        "rejected_reasons": reason_counter(rejected),
        "step6_llm_cleanup": {
            "main_resolved": cleanup_counter(main_resolved),
            "review_needed": cleanup_counter(review_needed),
            "rejected": cleanup_counter(rejected),
        },
        "next_input_for_benchmark_builder": "main_resolved.jsonl",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default=DEFAULT_INPUT)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--use_llm_cleanup",
        action="store_true",
        help="Use LLM to clean gold_clarified_detail and repair taxonomy before final split.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_OPENAI_MODEL,
        help="LLM model for Step6 cleanup.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="Sleep/retry interval for LLM cleanup calls.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing Step6 split outputs and skip records already written.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    out_dir = Path(args.out_dir)

    rows_in = load_jsonl(input_path)
    if args.resume:
        main_resolved, review_needed, rejected, resume_seen = load_resume_splits(out_dir)
    else:
        main_resolved = []
        review_needed = []
        rejected = []
        resume_seen = set()

    processed_rows = [process_record(row) for row in rows_in]

    dedup_best: Dict[str, Tuple[Dict[str, Any], Tuple[int, float, int]]] = {}
    for processed in processed_rows:
        fp = gap_dedup_fingerprint(processed, processed.get("gap") or {})
        if fp is None:
            continue

        rank = gap_dedup_rank(processed, processed.get("gap") or {})
        prev = dedup_best.get(fp)

        if prev is None or rank > prev[1]:
            dedup_best[fp] = (processed, rank)

    dedup_losers: set[int] = set()
    for _fp, (winner, _rank) in dedup_best.items():
        winner_id = id(winner)
        for processed in processed_rows:
            if gap_dedup_fingerprint(processed, processed.get("gap") or {}) != _fp:
                continue
            if id(processed) != winner_id:
                dedup_losers.add(id(processed))

    client = build_client() if args.use_llm_cleanup else None

    print("\n===== Step 6: Postprocess + Quote Validate + Optional LLM Cleanup =====")
    print(f"Input:           {input_path}")
    print(f"Out dir:         {out_dir}")
    print(f"Rows:            {len(rows_in)}")
    print(f"Use LLM cleanup: {args.use_llm_cleanup}")
    print(f"Model:           {args.model if args.use_llm_cleanup else 'N/A'}")
    print(f"Resume:          {args.resume}")
    if args.resume:
        print(f"Already saved:   {len(resume_seen)}")
    print("======================================================================\n")

    skipped_resume = 0
    for processed in tqdm(processed_rows, desc="Step6 validate", unit="record"):
        key = resume_key(processed)
        if key in resume_seen:
            skipped_resume += 1
            continue

        duplicate_gap = id(processed) in dedup_losers
        split, reasons = classify_record(processed, duplicate_gap=duplicate_gap)

        if (
            args.use_llm_cleanup
            and client is not None
            and not duplicate_gap
            and needs_llm_cleanup(processed, split, reasons)
        ):
            try:
                issue_text = build_issue_thread_text(processed)

                lp = processed.get("local_paths") or {}
                original_path = str(
                    lp.get("original_paper_fast_text_markdown_path")
                    or processed.get("original_text_path")
                    or ""
                )
                original_text = read_text(original_path)

                prompt = build_cleanup_prompt(processed, issue_text, original_text)
                cleanup = call_cleanup_llm(
                    client=client,
                    model=args.model,
                    prompt=prompt,
                    sleep=args.sleep,
                )

                processed = apply_llm_cleanup_result(processed, cleanup)

                # Re-run deterministic validation after cleanup.
                processed = process_record(processed)
                split, reasons = classify_record(processed, duplicate_gap=False)

                # LLM can request stricter decisions, but cannot override hard validation into main.
                llm_decision = cleanup.get("final_decision")

                if llm_decision == "rejected":
                    split = "rejected"
                    reasons = [
                        "llm_cleanup_rejected",
                        str(cleanup.get("decision_reason") or ""),
                    ]
                elif llm_decision == "review_needed" and split == "main_resolved":
                    split = "review_needed"
                    reasons = [
                        "llm_cleanup_requested_review",
                        str(cleanup.get("decision_reason") or ""),
                    ]

            except Exception as e:
                processed["step6_llm_cleanup_error"] = repr(e)
                if split == "main_resolved":
                    split = "review_needed"
                    reasons = ["llm_cleanup_failed_review_needed", repr(e)]
                else:
                    reasons.append(f"llm_cleanup_failed:{repr(e)}")

        out = add_postprocess_info(processed, split, reasons)

        if split == "main_resolved":
            main_resolved.append(out)
        elif split == "review_needed":
            review_needed.append(out)
        else:
            rejected.append(out)

        resume_seen.add(key)
        save_step6_outputs(out_dir, main_resolved, review_needed, rejected)

    save_step6_outputs(out_dir, main_resolved, review_needed, rejected)

    summary = summarize_splits(main_resolved, review_needed, rejected, len(rows_in))
    summary["paths"] = {
        "input_path": str(input_path),
        "out_dir": str(out_dir),
        "main_resolved": str(out_dir / "main_resolved.jsonl"),
        "review_needed": str(out_dir / "review_needed.jsonl"),
        "rejected": str(out_dir / "rejected.jsonl"),
    }
    summary["use_llm_cleanup"] = args.use_llm_cleanup
    summary["model"] = args.model if args.use_llm_cleanup else None
    summary["resume"] = {
        "enabled": args.resume,
        "skipped_existing": skipped_resume,
        "seen_after_run": len(resume_seen),
    }

    save_json(summary, out_dir / "summary.json")

    print("\n[done]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
