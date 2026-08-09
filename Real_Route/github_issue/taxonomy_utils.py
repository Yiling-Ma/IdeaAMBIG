from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


LEVEL2_TO_LEVEL1: Dict[str, str] = {
    # Ambiguity
    "Ambiguous Definition": "Ambiguity",
    "Ambiguous Procedure": "Ambiguity",

    # Incompleteness
    "Missing Method Procedure": "Incompleteness",
    "Missing Model Structure": "Incompleteness",
    "Missing Data Specification": "Incompleteness",
    "Missing Configuration Protocol": "Incompleteness",
    "Missing Evaluation Specification": "Incompleteness",

    # Inconsistency
    "Conflicting Objective": "Inconsistency",
    "Conflicting Model Design": "Inconsistency",
    "Conflicting Formal Definition": "Inconsistency",
}

VALID_LEVEL2 = set(LEVEL2_TO_LEVEL1.keys())

VALID_SPECIFICATION_SLOTS = {
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


TAXONOMY_BRIEF = """
Allowed Level-2 taxonomy labels:

Ambiguity
1. Ambiguous Definition
   A formal element, such as a symbol, notation, mathematical object, or rule, is
   described without a sufficiently precise meaning. Multiple plausible
   interpretations remain, leading implementers to compute or instantiate different
   objects.

2. Ambiguous Procedure
   A method operation, execution rule, inference behavior, or interaction between
   components is described but its operational procedure is unclear. Different
   implementations may follow different behaviors and produce different outcomes.

Incompleteness
3. Missing Method Procedure
   A required operational step, algorithmic rule, update mechanism, decision
   criterion, or execution procedure is omitted. Without this information, an
   implementer cannot faithfully reproduce how the method operates.

4. Missing Model Structure
   A model or computational component is mentioned, but its structural configuration
   is insufficiently specified. Missing details may include layer composition,
   module organization, dimensional mapping, normalization, activation, or
   parameterization choices that affect the instantiated model.

5. Missing Data Specification
   The construction or transformation of input data is incompletely described.
   Missing details may include data filtering, labeling, augmentation,
   normalization, tokenization, segmentation, or other preprocessing steps that
   affect the resulting inputs or supervision signals.

6. Missing Configuration Protocol
   A result-sensitive configuration choice is introduced, but the specification does
   not describe how the choice should be determined. Missing information concerns
   the selection, tuning, or validation procedure for important settings (e.g.,
   hyperparameters, thresholds, initialization choices, or sampling parameters),
   rather than merely an omitted value.

7. Missing Evaluation Specification
   The evaluation procedure is incompletely described, including missing metric
   definitions, evaluation protocols, data splits, sampling procedures, prompts,
   thresholds, or evaluation configurations. Such omissions prevent faithful
   reproduction or comparison of reported results.

Inconsistency
8. Conflicting Objective
   The specification and another source, such as code, appendix, or supplementary
   material, define different objectives, loss functions, reward signals, or
   optimization targets. Following different sources would optimize materially
   different goals.

9. Conflicting Model Design
   Different sources specify incompatible model components, architectures,
   preprocessing pipelines, or execution pipelines. The inconsistency makes it
   unclear which design should be implemented.

10. Conflicting Formal Definition
   Different sources provide incompatible formal assumptions or mathematical
   definitions, such as distributions, conditioning rules, aggregation operations,
   sampling assumptions, or inference formulations. The discrepancy changes the
   underlying formal model being implemented.
"""


SPECIFICATION_SLOT_BRIEF = """
Allowed specification slots for codification-readiness annotation:

1. TASK_AND_IO
   Use when the blocker concerns the research task, expected inputs, or expected
   outputs.

2. CORE_ALGORITHM
   Use when the blocker concerns the central computational procedure or sequence of
   operations that defines the method.

3. MODEL_ARCHITECTURE
   Use when the blocker concerns the structure or connectivity of the model.

4. OBJECTIVE_AND_SUPERVISION
   Use when the blocker concerns the objective being optimized or the supervision
   used to train the method.

5. TRAINING_PROCEDURE
   Use when the blocker concerns how the model or method is trained.

6. DATA_AND_PREPROCESSING
   Use when the blocker concerns how data are constructed, selected, transformed,
   or partitioned.

7. INFERENCE_AND_DECISION
   Use when the blocker concerns how model outputs are converted into final
   predictions, rankings, actions, or decisions.

8. EVALUATION_PROTOCOL
   Use when the blocker concerns how the proposed method or research claim is
   evaluated.

9. INTERNAL_CONSISTENCY
   Use only when two or more parts of the specification conflict.

10. NONE
    Select NONE only when the specification is labeled READY and no
    implementation-critical blocker is present.
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

Missing Method Procedure:
- Use for training-loop rules, update order, loss routing, interface operation,
  sampling/update rules, termination criteria, or algorithmic bookkeeping.
- Do not use it as a catch-all if the missing detail is clearly evaluation, data,
  preprocessing, configuration selection, or model structure.

Missing Configuration Protocol:
- Use when the missing detail is how to choose/tune/validate a configuration choice.
- Do not use for a single ordinary unreported value unless the selection rule is
  itself the blocker.
- If a parameter controls a procedure or schedule, and the procedure is the blocker,
  prefer Missing Method Procedure.

Missing Model Structure:
- Use for missing structural model choices: activation, normalization, pooling,
  layer/module type, initialization, readout, dimensional mapping, or module wiring.

Missing Evaluation Specification:
- Use for metric computation, evaluation split, prompt set, evaluation threshold,
  evaluation sampling, number of evaluation seeds/samples, evaluator configuration,
  or protocol needed to reproduce reported numbers.
- If the split or data processing is used to construct training data, prefer
  Missing Data Specification.

Missing Data Specification:
- Use for dataset construction, filtering, label construction, input normalization,
  augmentation, tokenization, serialization, segmentation/windowing, train/validation
  data construction, or preprocessing before model input.
- If the data operation is only for computing evaluation metrics, prefer
  Missing Evaluation Specification.

Inconsistency:
- Use Inconsistency only when there is explicit contradiction between paper/code,
  paper/appendix, paper/README, or two concrete sources.
- If the paper is merely vague and code makes one choice, prefer Incompleteness or
  Ambiguity.
"""


def _clean_label(s: str) -> str:
    s = str(s or "").strip()
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_level2(level2: str) -> str:
    """Normalize a Level-2 string to one of VALID_LEVEL2 via case-insensitive match."""
    x = _clean_label(level2)
    if not x:
        return x
    for valid in VALID_LEVEL2:
        if x.lower() == valid.lower():
            return valid
    return x


def normalize_specification_slot(slot: str) -> str:
    """Normalize a specification slot to one of VALID_SPECIFICATION_SLOTS."""
    x = str(slot or "").strip().upper().replace(" ", "_").replace("-", "_")
    x = re.sub(r"_+", "_", x)
    if x in VALID_SPECIFICATION_SLOTS:
        return x
    return str(slot or "").strip()


INCONSISTENCY_LEVEL2 = {
    "Conflicting Objective",
    "Conflicting Model Design",
    "Conflicting Formal Definition",
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
            return "Incompleteness", "Missing Evaluation Specification"
        return "Ambiguity", "Ambiguous Procedure"
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

    This function mainly:
    - normalizes Level-2 labels,
    - restores Level-1 from Level-2,
    - optionally applies conservative keyword corrections when context_text is given.
    """
    out = dict(taxonomy)

    original_level2 = str(out.get("level2") or "")
    normalized_level2 = normalize_level2(original_level2)

    if normalized_level2 not in VALID_LEVEL2:
        normalized_level2 = "Ambiguous Procedure"

    corrected = False
    correction_reason = ""

    if context_text:
        text = context_text.lower()

        evaluation_cues = [
            "evaluation", "evaluate", "metric", "fid", "clipscore", "aesthetic score",
            "auc", "map", "score", "table", "prompt set", "test prompts",
            "drawbench", "eval_seeds", "threshold", "validator", "sampling seeds",
            "evaluation seeds", "inference uses", "ode during inference", "diversity",
            "n-jsd", "pickscore", "imagereward",
        ]
        data_preprocess_cues = [
            "preprocess", "preprocessing", "normalization", "normalize", "scale",
            "rgb", "augmentation", "augment", "blur", "jpeg", "crop", "resize",
            "tokenization", "polygon", "serialization", "windowing", "stride",
            "segmentation", "dataset construction", "label construction",
            "training data", "train split", "validation split", "train/val",
            "input transformation", "rasterize", "rasterization",
        ]

        if normalized_level2 == "Missing Method Procedure":
            if any(cue in text for cue in evaluation_cues):
                normalized_level2 = "Missing Evaluation Specification"
                corrected = True
                correction_reason = (
                    "corrected Missing Method Procedure to Missing Evaluation "
                    "Specification based on explicit evaluation/metric cues"
                )
            elif any(cue in text for cue in data_preprocess_cues):
                normalized_level2 = "Missing Data Specification"
                corrected = True
                correction_reason = (
                    "corrected Missing Method Procedure to Missing Data "
                    "Specification based on explicit data/preprocessing cues"
                )

    out["level2"] = normalized_level2
    out["level1"] = LEVEL2_TO_LEVEL1[normalized_level2]

    if normalized_level2 != original_level2.strip() or corrected:
        out["taxonomy_auto_corrected"] = True
        out["taxonomy_original_level2"] = original_level2
        out["taxonomy_correction_reason"] = correction_reason or "normalized taxonomy label"
    else:
        out["taxonomy_auto_corrected"] = False

    return out


LEVEL2_DEFAULT_SPECIFICATION_SLOT: Dict[str, str] = {
    "Ambiguous Definition": "CORE_ALGORITHM",
    "Ambiguous Procedure": "CORE_ALGORITHM",
    "Missing Method Procedure": "CORE_ALGORITHM",
    "Missing Model Structure": "MODEL_ARCHITECTURE",
    "Missing Data Specification": "DATA_AND_PREPROCESSING",
    "Missing Configuration Protocol": "TRAINING_PROCEDURE",
    "Missing Evaluation Specification": "EVALUATION_PROTOCOL",
    "Conflicting Objective": "INTERNAL_CONSISTENCY",
    "Conflicting Model Design": "INTERNAL_CONSISTENCY",
    "Conflicting Formal Definition": "INTERNAL_CONSISTENCY",
}

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
    del context_text  # reserved for future context-aware overrides
    normalized = normalize_level2(str(level2 or ""))
    if normalized not in VALID_LEVEL2:
        return "NONE"
    return LEVEL2_DEFAULT_SPECIFICATION_SLOT.get(normalized, "NONE")


def infer_affected_component_from_level2(
    level2: Any,
    context_text: str = "",
) -> str:
    """Affected component uses the same vocabulary as specification slots."""
    return infer_codification_slot_from_level2(level2, context_text)


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
    actual_raw = str(
        gap.get("codification_slot")
        or gap.get("slot")
        or gap.get("affected_component")
        or ""
    ).strip()
    actual = normalize_specification_slot(actual_raw)
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
