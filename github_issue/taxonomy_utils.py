from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


LEVEL2_TO_LEVEL1: Dict[str, str] = {
    # Ambiguity
    "ambiguous formal definition": "Ambiguity",
    "ambiguous method behavior": "Ambiguity",

    # Incompleteness
    "missing algorithmic specification": "Incompleteness",
    "missing hyperparameter protocol": "Incompleteness",
    "missing model architecture": "Incompleteness",
    "missing evaluation protocol": "Incompleteness",
    "missing data/preprocessing protocol": "Incompleteness",

    # Inconsistency
    "inconsistent objective or loss": "Inconsistency",
    "inconsistent architecture or pipeline": "Inconsistency",
    "inconsistent model specification": "Inconsistency",
}

VALID_LEVEL2 = set(LEVEL2_TO_LEVEL1.keys())


TAXONOMY_BRIEF = """
Allowed Level-2 taxonomy labels:

Ambiguity
1. ambiguous formal definition
   A symbol, notation, mathematical object, or formal rule is underspecified,
   leaving multiple plausible interpretations that change what is computed.

2. ambiguous method behavior
   A method component, training procedure, inference rule, or evaluation behavior
   is described but its operational behavior is unclear, leaving multiple plausible
   implementations.

Incompleteness
3. missing algorithmic specification
   A core algorithmic step, interface, update rule, routing decision, or procedural
   detail is omitted, so the method cannot be faithfully implemented.

4. missing hyperparameter protocol
   A result-sensitive hyperparameter is introduced, but the paper does not specify
   how its value is chosen, tuned, or validated. The issue is the missing selection
   protocol, not just a single ordinary unreported value.

5. missing model architecture
   The paper states that a model or module is used but omits structural details
   such as layer type, normalization, pooling, activation, initialization, or
   dimensional mapping.

6. missing evaluation protocol
   The evaluation setup is incomplete, including missing metric computation rules,
   evaluation data splits, prompt sets, thresholds, sampling procedures, number of
   evaluation seeds/samples, or evaluation model configuration.

7. missing data/preprocessing protocol
   The data construction, filtering, labeling, augmentation, normalization,
   tokenization, segmentation, windowing, or input transformation procedure is not
   fully specified.

Inconsistency
8. inconsistent objective or loss
   The paper and another source specify different training objectives, loss terms,
   reward definitions, or optimization targets.

9. inconsistent architecture or pipeline
   The paper and another source specify different model architectures, module
   configurations, data pipelines, preprocessing steps, or training/evaluation
   pipelines.

10. inconsistent model specification
   Two sources specify conflicting formal model assumptions, such as distributions,
   conditioning sets, aggregation rules, sampling support, or probabilistic/inference
   definitions.
"""


GITHUB_TAXONOMY_BOUNDARY = """
Boundary rules for GitHub issue real-gap annotation:

General:
- Keep only method-core, implementation-facing, evidence-grounded specification gaps.
- Reject resource-only, installation-only, environment-only, script-usage-only, or
  performance-only issues unless they clarify a paper-described method, training,
  preprocessing, evaluation, inference, or model behavior protocol.
- If a gap can fit multiple labels, choose the label that best describes the primary
  implementation blocker.

Ambiguity vs. Incompleteness:
- Use Ambiguity when the source text describes something but leaves multiple plausible
  operational interpretations.
- Use Incompleteness when a necessary detail is simply absent.

missing algorithmic specification:
- Use for training-loop rules, update order, loss routing, interface operation,
  sampling/update rules, termination criteria, or algorithmic bookkeeping.
- Do not use it as a catch-all if the missing detail is clearly evaluation, data,
  preprocessing, hyperparameter selection, or model architecture.

missing hyperparameter protocol:
- Use when the missing detail is how to choose/tune/validate a hyperparameter.
- Do not use for a single ordinary unreported value unless the selection rule is
  itself the blocker.
- If a parameter controls a procedure or schedule, and the procedure is the blocker,
  prefer missing algorithmic specification.

missing model architecture:
- Use for missing structural model choices: activation, normalization, pooling,
  layer/module type, initialization, readout, dimensional mapping, or module wiring.
- If the missing detail is data transformation or evaluation behavior, do not use
  this label.

missing evaluation protocol:
- Use for metric computation, evaluation split, prompt set, evaluation threshold,
  inference-time measurement, evaluation sampling, number of evaluation seeds/samples,
  evaluator model configuration, or protocol needed to reproduce reported numbers.
- If the split or data processing is used to construct training data, prefer
  missing data/preprocessing protocol.

missing data/preprocessing protocol:
- Use for dataset construction, filtering, label construction, input normalization,
  augmentation, tokenization, serialization, segmentation/windowing, train/validation
  data construction, or preprocessing before model input.
- If the data operation is only for computing evaluation metrics, prefer
  missing evaluation protocol.

Inconsistency:
- Use Inconsistency only when there is explicit contradiction between paper/code,
  paper/appendix, paper/README, or two concrete sources.
- If the paper is merely vague and code makes one choice, prefer Incompleteness or
  Ambiguity.
"""


_ALIAS_MAP = {
    # common casing / punctuation variants
    "ambiguous formal definitions": "ambiguous formal definition",
    "ambiguous method behaviours": "ambiguous method behavior",
    "ambiguous method behaviour": "ambiguous method behavior",

    "missing algorithm specification": "missing algorithmic specification",
    "missing algorithmic details": "missing algorithmic specification",
    "missing algorithmic detail": "missing algorithmic specification",
    "missing training protocol": "missing algorithmic specification",
    "missing inference protocol": "missing algorithmic specification",

    "missing hyperparameter": "missing hyperparameter protocol",
    "missing hyperparameter value": "missing hyperparameter protocol",
    "missing hyperparameter selection": "missing hyperparameter protocol",
    "missing hyperparameter tuning protocol": "missing hyperparameter protocol",

    "missing architecture": "missing model architecture",
    "missing architectural specification": "missing model architecture",
    "missing architecture specification": "missing model architecture",

    "missing eval protocol": "missing evaluation protocol",
    "missing evaluation specification": "missing evaluation protocol",
    "missing metric protocol": "missing evaluation protocol",
    "missing metric computation protocol": "missing evaluation protocol",
    "missing evaluation metric protocol": "missing evaluation protocol",

    "missing preprocessing protocol": "missing data/preprocessing protocol",
    "missing data protocol": "missing data/preprocessing protocol",
    "missing data construction protocol": "missing data/preprocessing protocol",
    "missing data processing protocol": "missing data/preprocessing protocol",
    "missing input preprocessing protocol": "missing data/preprocessing protocol",
    "missing augmentation protocol": "missing data/preprocessing protocol",

    "inconsistent loss": "inconsistent objective or loss",
    "inconsistent objective": "inconsistent objective or loss",
    "inconsistent reward": "inconsistent objective or loss",
    "inconsistent objective/loss": "inconsistent objective or loss",

    "inconsistent architecture": "inconsistent architecture or pipeline",
    "inconsistent pipeline": "inconsistent architecture or pipeline",
    "inconsistent preprocessing pipeline": "inconsistent architecture or pipeline",

    "inconsistent model": "inconsistent model specification",
    "inconsistent formal model": "inconsistent model specification",
    "inconsistent model specificatio": "inconsistent model specification",
}


def _clean_label(s: str) -> str:
    s = str(s or "").strip().lower()
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_level2(level2: str) -> str:
    """
    Normalize a potentially noisy Level-2 string to one of VALID_LEVEL2.
    Returns the cleaned string if unknown; caller can decide fallback.
    """
    x = _clean_label(level2)

    if x in VALID_LEVEL2:
        return x

    if x in _ALIAS_MAP:
        return _ALIAS_MAP[x]

    # lightweight substring recovery
    for valid in VALID_LEVEL2:
        if valid in x:
            return valid

    for alias, valid in _ALIAS_MAP.items():
        if alias in x:
            return valid

    return x


INCONSISTENCY_LEVEL2 = {
    "inconsistent objective or loss",
    "inconsistent architecture or pipeline",
    "inconsistent model specification",
}

CONTRADICTION_SIGNALS = [
    "paper states",
    "paper describes",
    "paper specifies",
    "paper defines",
    "contradicts",
    "inconsistent with",
    "differs from the paper",
    "while the code",
    "paper says",
    "paper and code",
    "two parts of the paper",
    "conflicting objective",
    "conflicting loss",
]

GAP_TEXT_FIELDS = (
    "gap_summary",
    "gap_quote",
    "solution_summary",
    "solution_quote",
    "gold_clarified_detail",
    "why_this_blocks_or_affects_codification",
    "gold_detail_removed_or_corrupted",
)


def gap_text_blob(gap: Dict[str, Any]) -> str:
    return " ".join(str(gap.get(k) or "") for k in GAP_TEXT_FIELDS).lower()


def has_paper_code_contradiction(text: str) -> bool:
    if any(signal in text for signal in CONTRADICTION_SIGNALS):
        return True
    return bool(
        re.search(
            r"paper.{0,80}(code|implementation).{0,40}(differ|contradict|inconsistent|conflict)",
            text,
        )
    )


def suggest_inconsistency_relabel(gap: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    text = gap_text_blob(gap)
    evaluation_terms = [
        "eval_metric",
        "metric",
        "rmsle",
        "msle",
        "softmax",
        "logits",
        "activation",
        "cross entropy",
        "evaluation",
        "nllloss",
        "fid",
        "clipscore",
        "prompt set",
        "drawbench",
    ]
    implementation_bug_terms = [
        "should have",
        "missing squared",
        "bug",
        "wrong parameter",
        "computes msle",
        "not rmsle",
    ]

    if any(term in text for term in evaluation_terms):
        if any(term in text for term in implementation_bug_terms):
            return "Incompleteness", "missing evaluation protocol"
        return "Ambiguity", "ambiguous method behavior"
    return None


def validate_inconsistency_taxonomy(
    gap: Dict[str, Any],
) -> Tuple[bool, str, Optional[Tuple[str, str]]]:
    level2 = normalize_level2(str(gap.get("level2") or ""))
    if level2 not in INCONSISTENCY_LEVEL2:
        return True, "", None

    text = gap_text_blob(gap)
    if has_paper_code_contradiction(text):
        return True, "", None

    suggestion = suggest_inconsistency_relabel(gap)
    if suggestion:
        return False, "inconsistency_without_paper_code_contradiction", suggestion
    return False, "inconsistency_without_paper_code_contradiction", None


def assess_taxonomy_mislabel(gap: Dict[str, Any]) -> Dict[str, Any]:
    ok, reason, suggestion = validate_inconsistency_taxonomy(gap)
    return {
        "taxonomy_valid": ok,
        "mislabel_reason": "" if ok else reason,
        "suggested_level1": suggestion[0] if suggestion else None,
        "suggested_level2": suggestion[1] if suggestion else None,
        "taxonomy_mislabel": not ok,
    }


def apply_taxonomy_correction_to_gap(gap: Dict[str, Any]) -> Dict[str, Any]:
    """Apply taxonomy normalization/correction and merge results back into a gap dict."""
    taxonomy = {
        "level1": gap.get("level1"),
        "level2": gap.get("level2"),
        "taxonomy_reason": gap.get("taxonomy_reason", ""),
        "taxonomy_confidence": gap.get("taxonomy_confidence", 0.0),
    }
    context_text = gap_text_blob(gap)
    corrected = apply_taxonomy_correction(taxonomy, context_text=context_text)

    out = dict(gap)
    for key in (
        "level1",
        "level2",
        "taxonomy_auto_corrected",
        "taxonomy_correction_reason",
        "taxonomy_original_level1",
        "taxonomy_original_level2",
    ):
        if key in corrected:
            out[key] = corrected[key]
    return align_gap_labels_with_level2(out)


def apply_taxonomy_correction(
    taxonomy: Dict[str, Any],
    context_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Deterministic light correction after LLM taxonomy assignment.

    This function does NOT invent a new label aggressively. It mainly:
    - normalizes Level-2 aliases,
    - restores Level-1 from Level-2,
    - optionally applies conservative keyword corrections when context_text is given.
    """
    out = dict(taxonomy)

    original_level2 = str(out.get("level2") or "")
    normalized_level2 = normalize_level2(original_level2)

    if normalized_level2 not in VALID_LEVEL2:
        normalized_level2 = "ambiguous method behavior"

    corrected = False
    correction_reason = ""

    # Optional conservative correction using context text.
    # This is intentionally conservative and only triggers for very explicit cues.
    if context_text:
        text = context_text.lower()

        evaluation_cues = [
            "evaluation", "evaluate", "metric", "fid", "clipscore", "aesthetic score",
            "auc", "mAP".lower(), "score", "table", "prompt set", "test prompts",
            "drawbench", "eval_seeds", "threshold", "validator", "sampling seeds",
            "evaluation seeds", "inference uses", "ode during inference", "diversity",
            "n-jsd", "pickscore", "imagereward",
        ]
        data_preprocess_cues = [
            "preprocess", "preprocessing", "normalization", "normalize", "scale",
            "rgb", "augmentation", "augment", "blur", "jpeg", "crop", "resize",
            "tokenization", "tokenize", "serialization", "windowing", "stride",
            "segmentation", "dataset construction", "label construction",
            "training data", "train split", "validation split", "train/val",
            "input transformation", "rasterize", "rasterization",
        ]

        if normalized_level2 == "missing algorithmic specification":
            if any(cue in text for cue in evaluation_cues):
                normalized_level2 = "missing evaluation protocol"
                corrected = True
                correction_reason = "corrected missing algorithmic specification to missing evaluation protocol based on explicit evaluation/metric cues"
            elif any(cue in text for cue in data_preprocess_cues):
                normalized_level2 = "missing data/preprocessing protocol"
                corrected = True
                correction_reason = "corrected missing algorithmic specification to missing data/preprocessing protocol based on explicit data/preprocessing cues"

    out["level2"] = normalized_level2
    out["level1"] = LEVEL2_TO_LEVEL1[normalized_level2]

    if normalized_level2 != _clean_label(original_level2) or corrected:
        out["taxonomy_auto_corrected"] = True
        out["taxonomy_original_level2"] = original_level2
        out["taxonomy_correction_reason"] = correction_reason or "normalized taxonomy label alias"
    else:
        out["taxonomy_auto_corrected"] = False

    return out


LEVEL2_DEFAULT_CODIFICATION_SLOT: Dict[str, str] = {
    "ambiguous formal definition": "core_method",
    "ambiguous method behavior": "core_method",
    "missing algorithmic specification": "algorithm",
    "missing hyperparameter protocol": "training",
    "missing model architecture": "core_method",
    "missing evaluation protocol": "evaluation",
    "missing data/preprocessing protocol": "preprocessing",
    "inconsistent objective or loss": "training",
    "inconsistent architecture or pipeline": "core_method",
    "inconsistent model specification": "core_method",
}

LEVEL2_DEFAULT_AFFECTED_COMPONENT: Dict[str, str] = {
    "ambiguous formal definition": "core_method",
    "ambiguous method behavior": "core_method",
    "missing algorithmic specification": "algorithm",
    "missing hyperparameter protocol": "hyperparameter",
    "missing model architecture": "model_architecture",
    "missing evaluation protocol": "evaluation",
    "missing data/preprocessing protocol": "preprocessing",
    "inconsistent objective or loss": "training",
    "inconsistent architecture or pipeline": "core_method",
    "inconsistent model specification": "model_architecture",
}

DATA_PREPROCESS_SLOT_CUES = [
    "stride",
    "window",
    "token",
    "normalize",
    "preprocess",
    "augment",
    "filter",
    "crop",
    "resize",
    "segmentation",
    "label construction",
]

REFERENCE_LANGUAGE_PATTERNS = [
    re.compile(r"\bas proven in the literature\b", re.IGNORECASE),
    re.compile(r"\bin the literature\b", re.IGNORECASE),
    re.compile(r"\baccording to\b", re.IGNORECASE),
    re.compile(r"\bas described in\b", re.IGNORECASE),
    re.compile(r"\bas shown in\b", re.IGNORECASE),
    re.compile(r"\bas reported in\b", re.IGNORECASE),
    re.compile(r"\bas established in\b", re.IGNORECASE),
    re.compile(r"\bfrom the literature\b", re.IGNORECASE),
    re.compile(r"\bliterature shows\b", re.IGNORECASE),
    re.compile(r"\bprior work\b", re.IGNORECASE),
    re.compile(r"\bprevious work\b", re.IGNORECASE),
    re.compile(r"\bexisting literature\b", re.IGNORECASE),
    re.compile(r"\bcited in\b", re.IGNORECASE),
    re.compile(r"\bsee also\b", re.IGNORECASE),
    re.compile(r"\bas in prior\b", re.IGNORECASE),
    re.compile(r"\bas in the paper\b", re.IGNORECASE),
    re.compile(r"\[\d+(?:\s*,\s*\d+)*\]"),
]


def _split_sentences(text: str) -> List[str]:
    text = str(text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def infer_codification_slot_from_level2(
    level2: Any,
    context_text: str = "",
) -> str:
    normalized = normalize_level2(str(level2 or ""))
    if normalized not in VALID_LEVEL2:
        return "implementation_detail"

    if normalized == "missing data/preprocessing protocol":
        text = str(context_text or "").lower()
        if any(cue in text for cue in DATA_PREPROCESS_SLOT_CUES):
            return "preprocessing"
        return "data"

    return LEVEL2_DEFAULT_CODIFICATION_SLOT.get(normalized, "implementation_detail")


def infer_affected_component_from_level2(
    level2: Any,
    context_text: str = "",
) -> str:
    normalized = normalize_level2(str(level2 or ""))
    if normalized not in VALID_LEVEL2:
        return "implementation_detail"

    if normalized == "missing data/preprocessing protocol":
        slot = infer_codification_slot_from_level2(normalized, context_text)
        return "data" if slot == "data" else "preprocessing"

    return LEVEL2_DEFAULT_AFFECTED_COMPONENT.get(normalized, "implementation_detail")


def assess_slot_level2_consistency(gap: Dict[str, Any]) -> Dict[str, Any]:
    level2 = normalize_level2(str(gap.get("level2") or ""))
    if level2 not in VALID_LEVEL2:
        return {
            "slot_level2_consistent": True,
            "expected_codification_slot": None,
            "actual_codification_slot": None,
        }

    context = gap_text_blob(gap)
    expected = infer_codification_slot_from_level2(level2, context)
    actual = str(
        gap.get("codification_slot")
        or gap.get("slot")
        or gap.get("affected_component")
        or ""
    ).strip().lower()

    if actual in {"model_architecture", "hyperparameter"}:
        actual_mapped = infer_affected_component_from_level2(level2, context)
        consistent = actual == actual_mapped or actual in {expected}
    else:
        consistent = actual == expected

    return {
        "slot_level2_consistent": consistent,
        "expected_codification_slot": expected,
        "actual_codification_slot": actual or None,
        "level2": level2,
    }


def align_gap_labels_with_level2(gap: Dict[str, Any]) -> Dict[str, Any]:
    """Force codification slot and affected_component to match level2 taxonomy."""
    out = dict(gap)
    level2 = normalize_level2(str(out.get("level2") or ""))
    if level2 not in VALID_LEVEL2:
        return out

    context = gap_text_blob(out)
    expected_slot = infer_codification_slot_from_level2(level2, context)
    expected_affected = infer_affected_component_from_level2(level2, context)

    old_slot = str(out.get("codification_slot") or out.get("slot") or "").strip()
    old_affected = str(out.get("affected_component") or "").strip()

    out["codification_slot"] = expected_slot
    out["slot"] = expected_slot
    out["affected_component"] = expected_affected

    if old_slot != expected_slot or old_affected != expected_affected:
        out["slot_level2_aligned"] = True
        out["slot_alignment_reason"] = (
            f"aligned codification_slot={expected_slot} "
            f"affected_component={expected_affected} from level2={level2}"
        )
        if old_slot and old_slot != expected_slot:
            out["slot_original"] = old_slot
        if old_affected and old_affected != expected_affected:
            out["affected_component_original"] = old_affected
    else:
        out["slot_level2_aligned"] = False

    return out


def find_reference_language_hits(text: str) -> List[str]:
    hits: List[str] = []
    for pattern in REFERENCE_LANGUAGE_PATTERNS:
        if pattern.search(str(text or "")):
            hits.append(pattern.pattern)
    return hits


def strip_reference_language_from_text(text: str) -> Tuple[str, List[str]]:
    """Drop sentences that defer to literature/citations instead of stating the spec."""
    raw = str(text or "").strip()
    if not raw:
        return "", []

    kept: List[str] = []
    removed_hits: List[str] = []
    for sentence in _split_sentences(raw):
        sentence_hits = find_reference_language_hits(sentence)
        if sentence_hits:
            removed_hits.extend(sentence_hits)
            continue
        kept.append(sentence)

    cleaned = " ".join(kept).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned, sorted(set(removed_hits))