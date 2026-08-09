from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common import (
    DIAGNOSTIC_LEAKAGE_PHRASES,
    LEVEL2_TO_LEVEL1,
    add_common_args,
    contains_leak,
    ensure_dir,
    load_jsonl,
    normalize_ws,
    save_json,
    save_jsonl,
)


# ============================================================
# Taxonomy
# ============================================================

FINAL_LEVEL2_TO_LEVEL1 = dict(LEVEL2_TO_LEVEL1)

DISALLOWED_LEVEL2 = {
    "missing objective/loss specification",
    "missing objective specification",
    "missing loss specification",
}

HARD_LEAKAGE_REASONS = {
    "gold_detail_leakage",
    "must_not_reveal_leakage",
    "diagnostic_language_leakage",
    "semantic_leakage_exact_prompt_phrase",
    "semantic_leakage_dataset_file_or_format",
    "semantic_leakage_huggingface_dataset",
    "semantic_leakage_json_keys_or_range",
    "semantic_leakage_stage_json_requirement",
    "semantic_leakage_count_of_alternatives",
    "semantic_leakage_threshold_values",
    "semantic_leakage_max_iteration_value",
    "semantic_leakage_model_or_library_identifier",
    "semantic_leakage_formula_or_code_identifier",
}


# ============================================================
# Text normalization
# ============================================================

def canonical_text(text: Any) -> str:
    """Normalize text for robust leakage checks."""
    s = str(text or "")
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
    }
    for a, b in replacements.items():
        s = s.replace(a, b)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def compact_for_match(text: Any) -> str:
    """More aggressive normalization that removes most punctuation."""
    s = canonical_text(text)
    s = re.sub(r"[^a-z0-9_./:+-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def phrase_in_text(text: str, phrase: str, min_len: int = 4) -> bool:
    p = canonical_text(phrase)
    if len(p) < min_len:
        return False
    return p in canonical_text(text)


def compact_phrase_in_text(text: str, phrase: str, min_len: int = 4) -> bool:
    p = compact_for_match(phrase)
    if len(p) < min_len:
        return False
    return p in compact_for_match(text)


def regex_in_text(text: str, pattern: str) -> bool:
    try:
        return re.search(pattern, canonical_text(text), flags=re.I) is not None
    except re.error:
        return False


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text or ""))


def window_contains(text: str, left_pattern: str, right_pattern: str, window: int = 220) -> bool:
    """
    Return True if left and right occur close to each other.
    This avoids rejecting generic words that appear elsewhere in a long spec.
    """
    s = canonical_text(text)
    left_matches = list(re.finditer(left_pattern, s, flags=re.I))
    right_matches = list(re.finditer(right_pattern, s, flags=re.I))

    for lm in left_matches:
        for rm in right_matches:
            if abs(lm.start() - rm.start()) <= window:
                return True
    return False


def get_nested(inst: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = inst
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


# ============================================================
# Semantic leakage helpers
# ============================================================

def extract_quoted_phrases(text: str, min_words: int = 5) -> List[str]:
    phrases = []
    for match in re.findall(r"['\"]([^'\"]+)['\"]", str(text or "")):
        cleaned = normalize_ws(match)
        if len(cleaned.split()) >= min_words:
            phrases.append(cleaned)
    return phrases


def extract_code_identifiers(text: str) -> List[str]:
    s = str(text or "")
    identifiers: List[str] = []

    # Dotted identifiers such as mc1_targets.choices or sklearn.roc_auc_score.
    identifiers.extend(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+\b", s))

    # Function-like names such as torch.randn.
    identifiers.extend(re.findall(r"\b(?:torch|np|numpy|sklearn|langid|datasets|transformers)\.[a-zA-Z_][a-zA-Z0-9_]*\b", s))

    # JSON/key-like quoted identifiers are handled separately and should not all be hard leaks.
    out = []
    seen = set()
    for item in identifiers:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def extract_file_names(text: str) -> List[str]:
    return re.findall(
        r"\b[\w.-]+\.(?:csv|json|jsonl|txt|tsv|py|pkl|pt|pth|npy|yaml|yml)\b",
        str(text or ""),
        flags=re.I,
    )


def extract_model_or_library_identifiers(text: str) -> List[str]:
    s = str(text or "")
    patterns = [
        r"\bgpt-4o-mini-[0-9-]+\b",
        r"\bgpt-3\.5-turbo\b",
        r"\bgpt-4\b",
        r"\bgpt2\b",
        r"\bfacebook/bart-large-mnli\b",
        r"\bcardiffnlp/twitter-roberta-base-sentiment\b",
        r"\blangid\b",
        r"\blangid\.classify\b",
        r"\bGPT2LMHeadModel\b",
        r"\bHugging Face Dataset\b",
        r"\bOpenAI ChatCompletion\b",
        r"\bresponse_format\b",
        r"\bjson_object\b",
    ]
    out = []
    for p in patterns:
        for m in re.findall(p, s, flags=re.I):
            out.append(m)
    seen = set()
    deduped = []
    for item in out:
        key = canonical_text(item)
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def semantic_leakage_warnings(inst: Dict[str, Any]) -> List[str]:
    """
    Detect semantic or cross-field leakage that exact string matching misses.

    Examples:
      - target removes "exactly five", but another field still says "five alternative..."
      - target removes JSON keys/range, but input still says revised confidence JSON or 1-100
      - target removes CSV/HuggingFace conversion, but input still mentions CSV or Hugging Face Dataset
      - target removes exact prompt text, but the quoted prompt remains elsewhere
    """
    text = get_nested(inst, ["input", "underspecified_spec"], "") or ""
    defect = (inst.get("defects") or [{}])[0]
    target = normalize_ws(defect.get("gold_detail_removed_or_corrupted", ""))
    safe = normalize_ws(defect.get("surface_form_in_underspecified_spec", ""))
    clarification = get_nested(inst, ["gold", "gold_clarification"], {}) or {}
    required_detail = normalize_ws(clarification.get("required_detail", ""))
    must_not = get_nested(inst, ["construction_metadata", "perturbation_instruction", "must_not_reveal"], []) or []

    combined_target = normalize_ws(" ".join([target, required_detail, " ".join(str(x) for x in must_not)]))
    target_low = canonical_text(combined_target)
    text_low = canonical_text(text)

    warnings: List[str] = []

    # 1. Exact prompt phrase leakage.
    for phrase in extract_quoted_phrases(combined_target, min_words=5):
        if phrase_in_text(text, phrase, min_len=20):
            # Ignore if the phrase is intentionally present in the safe surface form.
            if not phrase_in_text(safe, phrase, min_len=20):
                warnings.append("semantic_leakage_exact_prompt_phrase")
                break

    # 2. Dataset file and preprocessing format leakage.
    for filename in extract_file_names(combined_target):
        if compact_phrase_in_text(text, filename, min_len=6):
            warnings.append("semantic_leakage_dataset_file_or_format")
            break

    if any(k in target_low for k in ["csv", "dataset loaded", "file", "hugging face dataset"]):
        if "hugging face dataset" in target_low and "hugging face dataset" in text_low:
            warnings.append("semantic_leakage_huggingface_dataset")
        if "csv" in target_low and re.search(r"\bcsv\b", text_low):
            warnings.append("semantic_leakage_dataset_file_or_format")

    # 3. Count-of-alternatives leakage.
    # This catches cases where implementation_details changed "exactly five" to "several"
    # while core_method or algorithm_steps still says "five alternative...".
    if ("five" in target_low or "exactly five" in target_low or re.search(r"\b5\b", target_low)) and "alternative" in target_low:
        if re.search(r"\b(five|5)\s+(?:diverse\s+)?alternative", text_low):
            if not re.search(r"\b(five|5)\s+(?:diverse\s+)?alternative", canonical_text(safe)):
                warnings.append("semantic_leakage_count_of_alternatives")

    # 4. JSON key/range leakage.
    # We avoid rejecting all JSON mentions. We only reject when JSON appears near the
    # same target stage/prompt or when exact keys/range remain visible.
    json_target = "json" in target_low or "json_object" in target_low or "response_format" in target_low
    answer_conf_target = "answer" in target_low and "confidence" in target_low

    if json_target and answer_conf_target:
        if re.search(r'["\']answer["\']', text_low) or re.search(r'["\']confidence["\']', text_low):
            warnings.append("semantic_leakage_json_keys_or_range")
        if re.search(r"\b1\s*[-–]\s*100\b", text_low) and re.search(r"\b1\s*[-–]\s*100\b", target_low):
            warnings.append("semantic_leakage_json_keys_or_range")
        if window_contains(text, r"revised confidence", r"json", window=260) and "revised confidence" in target_low:
            warnings.append("semantic_leakage_stage_json_requirement")
        if window_contains(text, r"stage\s*1", r"json", window=260) and "stage 1" in target_low:
            warnings.append("semantic_leakage_stage_json_requirement")
        if window_contains(text, r"stage\s*2", r"json", window=260) and "stage 2" in target_low:
            warnings.append("semantic_leakage_stage_json_requirement")
        if window_contains(text, r"stage\s*3", r"json", window=260) and "stage 3" in target_low:
            warnings.append("semantic_leakage_stage_json_requirement")

    # 5. Threshold values leakage.
    if "threshold" in target_low and any(v in target_low for v in ["0.9", "0.6", "0.8"]):
        if re.search(r"\b0\.9\b", text_low) or re.search(r"\b0\.6\b", text_low) or re.search(r"\b0\.8\b", text_low):
            warnings.append("semantic_leakage_threshold_values")

    # 6. Maximum iteration value leakage.
    if ("maximum iterations" in target_low or "max_iter" in target_low or "max iterations" in target_low) and re.search(r"\b4\b", target_low):
        if re.search(r"(maximum|max)\s+(?:number\s+of\s+)?iterations[^.]{0,80}\b4\b", text_low) or re.search(r"\bmax_iter\s*=?\s*4\b", text_low):
            warnings.append("semantic_leakage_max_iteration_value")

    # 7. Model/library identifiers and exact code identifiers.
    for ident in extract_model_or_library_identifiers(combined_target):
        # Some model/library names may be global non-target details. Treat langid / response_format /
        # json_object / exact model snapshots as hard semantic leakage when they are target-specific.
        ident_low = canonical_text(ident)
        if ident_low in {"json", "api"}:
            continue
        if phrase_in_text(text, ident, min_len=5):
            if not phrase_in_text(safe, ident, min_len=5):
                warnings.append("semantic_leakage_model_or_library_identifier")
                break

    for ident in extract_code_identifiers(combined_target):
        if phrase_in_text(text, ident, min_len=5):
            if not phrase_in_text(safe, ident, min_len=5):
                warnings.append("semantic_leakage_formula_or_code_identifier")
                break

    # 8. Formula leakage.
    formula_like = re.findall(
        r"[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*=\s*[^.;]+",
        combined_target,
    )
    for formula in formula_like:
        if compact_phrase_in_text(text, formula, min_len=12):
            warnings.append("semantic_leakage_formula_or_code_identifier")
            break

    return sorted(set(warnings))


def has_strong_training_evidence(text: str) -> bool:
    low = canonical_text(text)
    patterns = [
        r"\btrain(?:ing|ed)?\b",
        r"\bfine[- ]?tun(?:e|ing)\b",
        r"\boptimizer\b",
        r"\blearning rate\b",
        r"\bloss\b",
        r"\bgradient\b",
        r"\bbackprop(?:agation)?\b",
        r"\bepochs?\b",
        r"\bbatch size\b",
    ]
    return any(regex_in_text(low, p) for p in patterns)


def is_no_training_defect(defect: Dict[str, Any]) -> bool:
    target = canonical_text(defect.get("gold_detail_removed_or_corrupted", ""))
    slot = canonical_text(defect.get("codification_slot", ""))
    source_field = canonical_text(defect.get("source_field", ""))
    no_train_markers = [
        r"\bno training\b",
        r"\bno optimization\b",
        r"\bzero[- ]?shot\b",
        r"\binference[- ]only\b",
        r"\bwithout fine[- ]?tuning\b",
        r"\bno fine[- ]?tuning\b",
    ]
    return (
        (slot in {"training", "training_objective"} or source_field == "training_or_optimization")
        and any(regex_in_text(target, p) for p in no_train_markers)
    )


def is_overloaded_core_method_target(defect: Dict[str, Any]) -> bool:
    source_field = canonical_text(defect.get("source_field", ""))
    target = canonical_text(defect.get("gold_detail_removed_or_corrupted", ""))
    if source_field != "core_method":
        return False
    has_stage = any(regex_in_text(target, p) for p in [r"\bstage\b", r"\bphase\b", r"\bmulti[- ]stage\b"])
    has_threshold = any(regex_in_text(target, p) for p in [r"\bthreshold\b", r"\b0\.\d+\b", r"\btop[- ]?k\b"])
    has_prompt_strategy = any(regex_in_text(target, p) for p in [r"\bprompt\b", r"\bchain[- ]of[- ]thought\b", r"\bself[- ]consistency\b"])
    has_termination = any(regex_in_text(target, p) for p in [r"\bstop\b", r"\buntil\b", r"\bterminate\b", r"\bmax(?:imum)? iterations?\b"])
    return has_stage and has_threshold and has_prompt_strategy and has_termination


def is_confidence_defect_leaking(defect: Dict[str, Any], underspecified_text: str) -> bool:
    target = canonical_text(defect.get("gold_detail_removed_or_corrupted", ""))
    text = canonical_text(underspecified_text)
    is_confidence = any(regex_in_text(target, p) for p in [r"\bconfidence\b", r"\bece\b", r"\bcalibration\b", r"\bauroc\b"])
    if not is_confidence:
        return False
    leak_patterns = [
        r"\bconfidence\b.{0,50}\b(?:threshold|bins?|range|score)\b",
        r"\bece\b",
        r"\bece\s*(?:bins?|bucket|calculation|compute|analysis)\b",
        r"\bexpected calibration error\b",
        r"\bcalibration\s*(?:curve|error|analysis|binning|bins?)\b",
        r"\bconfidence\s*(?:binning|bins?|bucket|histogram)\b",
        r"\b1\s*[-–]\s*100\b",
    ]
    return any(regex_in_text(text, p) for p in leak_patterns)


def prompt_overlap_too_high(defect: Dict[str, Any]) -> bool:
    target = normalize_ws(defect.get("gold_detail_removed_or_corrupted", ""))
    safe = normalize_ws(defect.get("surface_form_in_underspecified_spec", ""))
    if not target or not safe:
        return False
    is_prompt_like = any(
        regex_in_text(target, p)
        for p in [r"\bprompt\b", r"\bjson\b", r"\boutput\b", r"\binstruction\b", r"['\"].{20,}['\"]"]
    )
    if not is_prompt_like:
        return False
    target_tokens = [t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", canonical_text(target))]
    safe_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", canonical_text(safe)))
    if not target_tokens:
        return False
    overlap = sum(1 for t in target_tokens if t in safe_tokens) / max(1, len(target_tokens))
    return overlap >= 0.6


def has_diagnostic_meta_language(text: str) -> bool:
    """
    Detect explicit benchmark/meta diagnostic leakage.

    This should only catch cases where the underspecified spec itself tells the model
    that something is missing, unclear, unspecified, defective, or not codification-ready.

    Do NOT flag normal method phrases such as:
      - no training
      - without external tools
      - not attempted
      - no extra tokens
      - no fine-tuning
    """
    low = canonical_text(text)

    hard_patterns = [
        r"\bmissing\s+(detail|specification|information|algorithm|protocol|component|step|field|parameter|hyperparameter|setting|rule)\b",
        r"\bthe\s+(detail|method|algorithm|protocol|component|step|field|parameter|setting|rule)\s+is\s+missing\b",
        r"\bnot\s+specified\b",
        r"\bunspecified\b",
        r"\bundefined\b",
        r"\bnot\s+defined\b",
        r"\bunclear\s+(how|what|which|whether|when|where)\b",
        r"\bambiguous\s+(method|behavior|definition|specification|procedure|rule|protocol|setting)\b",
        r"\bspecification\s+gap\b",
        r"\bimplementation\s+gap\b",
        r"\bcodification\s+gap\b",
        r"\bdefect\b",
        r"\bnot\s+codification-ready\b",
        r"\bnot\s+codification\s+ready\b",
        r"\bcannot\s+be\s+implemented\b",
        r"\bblocks\s+codification\b",
        r"\bblocking\s+missing\s+spec\b",
        r"\bfails\s+to\s+specify\b",
        r"\bfails\s+to\s+define\b",
        r"\bwithout\s+specifying\b",
        r"\bwithout\s+defining\b",
        r"\bwithout\s+describing\b",
        r"\bdoes\s+not\s+specify\b",
        r"\bdoes\s+not\s+define\b",
        r"\bdoes\s+not\s+describe\b",
        r"\blacks\s+(the\s+)?(required\s+)?(detail|specification|protocol|definition|rule|parameter)\b",
    ]

    return any(regex_in_text(low, pattern) for pattern in hard_patterns)


# ============================================================
# Instance validation
# ============================================================

def validate_instance(inst: Dict[str, Any], allow_borderline: bool = False) -> List[str]:
    reasons: List[str] = []

    if "input" not in inst or "gold" not in inst or "source" not in inst:
        reasons.append("schema_missing_top_level_keys")

    if len(inst.get("defects", [])) != 1:
        reasons.append("defects_not_exactly_one")
        return reasons

    defect = inst["defects"][0]
    text = inst.get("input", {}).get("underspecified_spec", "") or ""

    if not text:
        reasons.append("missing_underspecified_spec")
    elif word_count(text) < 100:
        reasons.append("underspecified_spec_too_short")

    level2 = defect.get("level2")
    level1 = defect.get("level1")

    if level2 in DISALLOWED_LEVEL2:
        reasons.append("disallowed_level2_missing_objective_loss")

    if level2 not in FINAL_LEVEL2_TO_LEVEL1:
        reasons.append("invalid_level2")

    if level1 != FINAL_LEVEL2_TO_LEVEL1.get(level2):
        reasons.append("level1_level2_mismatch")

    eval_targets = inst.get("eval_targets", {})
    if "underspecified" not in eval_targets or "reference" not in eval_targets:
        reasons.append("missing_eval_target_modes")

    # Exact leakage checks.
    target_detail = defect.get("gold_detail_removed_or_corrupted", "") or ""
    if contains_leak(text, [target_detail]):
        reasons.append("gold_detail_leakage")

    must_not = inst.get("construction_metadata", {}).get("perturbation_instruction", {}).get("must_not_reveal", [])
    if contains_leak(text, must_not):
        reasons.append("must_not_reveal_leakage")

    # Diagnostic language leakage means the input itself explicitly says that something
    # is missing/unspecified/unclear. This must be narrow to avoid rejecting normal
    # method facts such as "No training", "without external tools", or "not attempted".
    if has_diagnostic_meta_language(text):
        reasons.append("diagnostic_language_leakage")

    # Semantic / cross-field leakage checks.
    reasons.extend(semantic_leakage_warnings(inst))

    # Rule 1: reject no-training / zero-shot defects for main benchmark unless
    # there is strong training evidence in the remaining context.
    if is_no_training_defect(defect):
        context = " ".join(
            [
                get_nested(inst, ["gold", "codification_ready_reference"], "") or "",
                get_nested(inst, ["gold", "paper_derived_specification", "training_or_optimization"], "") or "",
                text,
            ]
        )
        if not has_strong_training_evidence(context):
            reasons.append("no_training_or_zero_shot_defect_not_main_benchmark")

    # Rule 2: reject overloaded core_method target with multiple independent gaps.
    if is_overloaded_core_method_target(defect):
        reasons.append("core_method_target_overloaded_multi_gap")

    # Rule 3: confidence defects should not keep confidence-analysis details.
    if is_confidence_defect_leaking(defect, text):
        reasons.append("confidence_defect_remaining_threshold_or_analysis")

    # Rule 4: exact-prompt defects should not keep most prompt content words.
    if prompt_overlap_too_high(defect):
        reasons.append("exact_prompt_defect_safe_surface_too_similar")

    check = inst.get("construction_metadata", {}).get("preservation_check", {})
    if not check.get("target_detail_modified"):
        reasons.append("target_detail_not_modified")
    if not check.get("non_target_details_preserved"):
        reasons.append("non_target_details_not_preserved")
    if check.get("extra_defects_introduced"):
        reasons.append("extra_defects_introduced")

    rewrite_meta = inst.get("construction_metadata", {}).get("rewrite_metadata", {})
    if rewrite_meta:
        # Backward-compatible with old Step4: only enforce this check when
        # safe_surface_inserted is explicitly provided in rewrite metadata.
        if "safe_surface_inserted" in rewrite_meta and not rewrite_meta.get("safe_surface_inserted"):
            reasons.append("safe_surface_not_inserted")
        if rewrite_meta.get("rewrite_status") in {
            "target_detail_not_found",
            "rewrite_failed_target_detail_not_found",
            "failed",
        }:
            reasons.append("rewrite_failed_target_detail_not_found")

    # Rule 5: not an auto-pass. Keep for manual audit, do not include in clean final.
    instance_quality_warnings = inst.get("quality_warnings", []) or []
    if "safe_surface_form_not_exactly_found" in instance_quality_warnings:
        reasons.append("manual_audit_safe_surface_form_not_exactly_found")

    strength = inst.get("construction_metadata", {}).get("candidate_strength", "unknown")
    if strength == "weak":
        reasons.append("weak_candidate_strength")
    if strength == "borderline" and not allow_borderline:
        reasons.append("borderline_candidate_strength")

    candidate_warnings = inst.get("construction_metadata", {}).get("candidate_quality_warnings", [])
    if candidate_warnings and not allow_borderline:
        reasons.append("candidate_has_quality_warnings")

    source = inst.get("source", {})
    project_id = inst.get("construction_metadata", {}).get("project_id")
    if source.get("split_group_id") != project_id:
        reasons.append("split_group_id_not_project_id")

    if inst.get("construction_metadata", {}).get("num_defects") != 1:
        reasons.append("metadata_num_defects_not_one")

    return sorted(set(reasons))


def audit_row(inst: Dict[str, Any], reasons: List[str]) -> Dict[str, Any]:
    leakage_reasons = [r for r in reasons if "leakage" in r]
    semantic_reasons = [r for r in reasons if r.startswith("semantic_leakage")]
    label_reasons = {
        "invalid_level2",
        "level1_level2_mismatch",
        "disallowed_level2_missing_objective_loss",
    }
    schema_reasons = {
        "schema_missing_top_level_keys",
        "missing_eval_target_modes",
        "defects_not_exactly_one",
        "metadata_num_defects_not_one",
    }

    text = inst.get("input", {}).get("underspecified_spec", "") or ""
    strength = inst.get("construction_metadata", {}).get("candidate_strength", "unknown")

    return {
        "id": inst.get("id"),
        "project_id": inst.get("construction_metadata", {}).get("project_id"),
        "self_contained": bool(text) and word_count(text) >= 100,
        "single_defect": len(inst.get("defects", [])) == 1,
        "non_leaky_exact": not leakage_reasons,
        "non_leaky_semantic": not semantic_reasons,
        "semantic_leakage_reasons": semantic_reasons,
        "label_correct": not any(r in label_reasons for r in reasons),
        "codification_relevant": strength in {"strong", "medium"} or (strength == "borderline"),
        "candidate_strength": strength,
        "compatible_with_reproducibility_schema": not any(r in schema_reasons for r in reasons),
        "accepted": not reasons,
        "rejection_reasons": reasons,
    }


# ============================================================
# Finalization
# ============================================================

def select_with_project_cap(
    final: List[Dict[str, Any]],
    rejected: List[Dict[str, Any]],
    max_instances_per_project: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not max_instances_per_project or max_instances_per_project <= 0:
        return final, rejected

    kept: List[Dict[str, Any]] = []
    extra_rejected: List[Dict[str, Any]] = []
    counts: Counter = Counter()

    for inst in final:
        project_id = inst.get("construction_metadata", {}).get("project_id")
        if counts[project_id] < max_instances_per_project:
            kept.append(inst)
            counts[project_id] += 1
        else:
            row = dict(inst)
            row["rejection_reasons"] = ["exceeds_max_instances_per_project"]
            extra_rejected.append(row)

    return kept, rejected + extra_rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument(
        "--allow_borderline",
        action="store_true",
        help="Allow borderline candidates into final instances. Default is false.",
    )
    parser.add_argument(
        "--max_instances_per_project",
        type=int,
        default=5,
        help="Cap final instances per project. Use 0 or negative to disable.",
    )
    args = parser.parse_args()

    args.input_path = Path(args.input_path)
    args.out_dir = Path(args.out_dir)

    ensure_dir(args.out_dir)

    rows = load_jsonl(args.input_path)
    if args.limit:
        rows = rows[: args.limit]

    final: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []

    project_to_split: Dict[str, str] = {}
    seen_ids = set()

    for inst in rows:
        reasons = validate_instance(inst, allow_borderline=args.allow_borderline)

        inst_id = inst.get("id")
        if not inst_id:
            reasons.append("missing_instance_id")
        elif inst_id in seen_ids:
            reasons.append("duplicate_instance_id")
        seen_ids.add(inst_id)

        project_id = inst.get("construction_metadata", {}).get("project_id")
        split_id = inst.get("source", {}).get("split_group_id")

        if project_id in project_to_split and project_to_split[project_id] != split_id:
            reasons.append("project_split_group_inconsistent")
        project_to_split[project_id] = split_id

        reasons = sorted(set(reasons))
        audit.append(audit_row(inst, reasons))

        if reasons:
            row = dict(inst)
            row["rejection_reasons"] = reasons
            rejected.append(row)
        else:
            final.append(inst)

    final, rejected = select_with_project_cap(
        final=final,
        rejected=rejected,
        max_instances_per_project=args.max_instances_per_project,
    )

    # Rebuild audit after project-cap rejection so accepted status is consistent.
    rejected_ids = {r.get("id") for r in rejected}
    audit = []
    for inst in rows:
        inst_id = inst.get("id")
        if inst_id in rejected_ids:
            reasons = []
            for r in rejected:
                if r.get("id") == inst_id:
                    reasons = r.get("rejection_reasons", [])
                    break
            audit.append(audit_row(inst, reasons))
        else:
            audit.append(audit_row(inst, []))

    save_jsonl(final, args.out_dir / "final_instances.jsonl")
    save_jsonl(rejected, args.out_dir / "rejected_instances.jsonl")
    save_jsonl(audit, args.out_dir / "audit_instances.jsonl")

    summary = {
        "input_instances": len(rows),
        "final_instances": len(final),
        "rejected_instances": len(rejected),
        "unique_projects": len(set(i.get("construction_metadata", {}).get("project_id") for i in final)),
        "instances_per_project": dict(Counter(i.get("construction_metadata", {}).get("project_id") for i in final)),
        "level1_distribution": dict(Counter(i["defects"][0]["level1"] for i in final)),
        "level2_distribution": dict(Counter(i["defects"][0]["level2"] for i in final)),
        "codification_slot_distribution": dict(Counter(i["defects"][0]["codification_slot"] for i in final)),
        "candidate_strength_distribution": dict(Counter(i.get("construction_metadata", {}).get("candidate_strength", "unknown") for i in final)),
        "rejection_reasons": dict(Counter(r for i in rejected for r in i.get("rejection_reasons", []))),
        "semantic_leakage_rejections": dict(
            Counter(
                r
                for i in rejected
                for r in i.get("rejection_reasons", [])
                if str(r).startswith("semantic_leakage")
            )
        ),
    }

    save_json(summary, args.out_dir / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()
