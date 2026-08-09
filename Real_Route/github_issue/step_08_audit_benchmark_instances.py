from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from taxonomy_utils import (
    apply_taxonomy_correction_to_gap,
    assess_taxonomy_mislabel,
    assess_slot_level2_consistency,
    infer_codification_slot_from_level2,
    gap_text_blob,
    strip_reference_language_from_text,
    find_reference_language_hits,
)


DEFAULT_INPUT_PATH = "Real_bench/github_issue_mining/outputs_step7/all_benchmark_instances.jsonl"
DEFAULT_OUT_DIR = "Real_bench/github_issue_mining/outputs_step8"

REPO_LANGUAGE_TERMS = [
    "train_model",
    "timemoerunner",
    "v10postprocess",
    "latest commit",
    "codebase",
    "official validation script",
    "official code",
    "github.com",
    "main branch",
    "tab_network.py",
    "line 480",
]

TOPIC_CLUSTERS: List[Tuple[str, List[str]]] = [
    ("embedding_viz", ["internal embedding", "visualization", "umap", "tsne", "before the final mapping"]),
    ("interpretability", ["gamma controls", "attention mask", "feature importance", "interpretability"]),
    ("pretraining", ["pretraining_ratio", "pretrain", "reconstruction loss", "fraction of features masked"]),
    ("eval_split", ["9:1", "train:eval", "train/eval", "90% training", "10% evaluation"]),
    ("eval_fid", ["fid", "inceptionv3", "inception v3", "rasteriz"]),
    ("eval_clip", ["clipscore", "clip model", "cosine similarity"]),
    ("eval_aesthetic", ["aesthetic score", "laion"]),
    ("sde_logprob", ["std_dev", "logprob", "log prob", "sqrt(-dt)", "sd3_sde", "sde step"]),
    ("kl_penalty", ["kl divergence", "train.beta", "kl penalty"]),
    ("reward_norm", ["normalize", "normalized to the [0,1]", "reward normalization"]),
    ("dpo_data", ["preferred", "non-preferred", "dpo training"]),
    ("dual_head", ["one-to-many", "one-to-one", "dual head"]),
    ("conf_threshold", ["confidence threshold", "conf threshold", "max_det"]),
]

CITATION_REF_PATTERN = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")

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
    "confidence threshold",
    "one-to-many",
    "kl divergence",
]

HALLUCINATED_PATTERNS = [
    re.compile(r"\btypically\b", re.IGNORECASE),
    re.compile(r"\busually\b", re.IGNORECASE),
    re.compile(r"\brecommended\b", re.IGNORECASE),
    re.compile(r"\bstandard practice\b", re.IGNORECASE),
    re.compile(r"\bcommonly\b", re.IGNORECASE),
    re.compile(r"\bwe found no difference\b", re.IGNORECASE),
    re.compile(r"\bwe assume\b", re.IGNORECASE),
]

EXTERNAL_KNOWLEDGE_PATTERNS = [
    re.compile(r"\bin general\b", re.IGNORECASE),
    re.compile(r"\bas in .{0,40}paper\b", re.IGNORECASE),
    re.compile(r"\baccording to .{0,40}literature\b", re.IGNORECASE),
    re.compile(r"\bas proven in the literature\b", re.IGNORECASE),
    re.compile(r"\bin the literature\b", re.IGNORECASE),
    re.compile(r"\bprior work\b", re.IGNORECASE),
    re.compile(r"\bprevious work\b", re.IGNORECASE),
    re.compile(r"\bas described in\b", re.IGNORECASE),
    re.compile(r"\bas shown in\b", re.IGNORECASE),
    re.compile(r"\bstandard value\b", re.IGNORECASE),
    re.compile(r"\bdefault setting\b", re.IGNORECASE),
    re.compile(r"\bcommon value\b", re.IGNORECASE),
]

LEVEL2_NORMALIZED = {
    "Missing Model Structure": "MODEL_ARCHITECTURE",
    "Missing Configuration Protocol": "TRAINING_PROCEDURE",
    "Missing Method Procedure": "CORE_ALGORITHM",
    "Missing Evaluation Specification": "EVALUATION_PROTOCOL",
    "Missing Data Specification": "DATA_AND_PREPROCESSING",
    "Ambiguous Procedure": "CORE_ALGORITHM",
    "Ambiguous Definition": "CORE_ALGORITHM",
    "Conflicting Objective": "INTERNAL_CONSISTENCY",
    "Conflicting Model Design": "INTERNAL_CONSISTENCY",
    "Conflicting Formal Definition": "INTERNAL_CONSISTENCY",
}

CONCRETE_GOLD_SIGNALS = [
    "=",
    " set to ",
    " uses ",
    " computed ",
    " applied ",
    " split ",
    " layer ",
    " ratio ",
    " threshold ",
    " stride ",
    " returns ",
    " not ",
]


def load_pipeline07():
    pipeline_path = (
        Path(__file__).resolve().parents[1] / "step_07_build_realgap_bench_instances.py"
    )
    spec = importlib.util.spec_from_file_location("pipeline07", pipeline_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load pipeline module: {pipeline_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P7 = load_pipeline07()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_instances_from_dir(input_dir: Path) -> List[Dict[str, Any]]:
    paths = sorted(input_dir.glob("*/benchmark_instance.json"))
    if not paths:
        raise FileNotFoundError(f"No benchmark_instance.json under {input_dir}")
    rows: List[Dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows


def save_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_text(text: Any) -> str:
    text = "" if text is None else str(text)
    return re.sub(r"\s+", " ", text).strip().lower()


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", str(text or "")))


def split_sentences(text: str) -> List[str]:
    text = str(text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def find_citation_references(text: str) -> List[str]:
    return CITATION_REF_PATTERN.findall(str(text or ""))


def strip_citation_references(text: str) -> Tuple[str, List[str]]:
    raw = str(text or "")
    hits = find_citation_references(raw)
    if not hits:
        return raw, []

    cleaned = CITATION_REF_PATTERN.sub("", raw)
    cleaned = re.sub(r"\bfrom\s+(?=[,.;]|$)", "from prior work ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r",\s*,", ",", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\s+\.", ".", cleaned)
    cleaned = re.sub(r"\.\s*\.", ".", cleaned)
    return cleaned.strip(), hits


def sanitize_text_fields(obj: Any) -> Tuple[Any, List[str]]:
    """Recursively strip citation markers from strings inside dict/list structures."""
    all_hits: List[str] = []

    if isinstance(obj, str):
        cleaned, hits = strip_citation_references(obj)
        all_hits.extend(hits)
        return cleaned, all_hits

    if isinstance(obj, list):
        out_list: List[Any] = []
        for item in obj:
            cleaned_item, hits = sanitize_text_fields(item)
            all_hits.extend(hits)
            out_list.append(cleaned_item)
        return out_list, all_hits

    if isinstance(obj, dict):
        out_dict: Dict[str, Any] = {}
        for key, value in obj.items():
            cleaned_value, hits = sanitize_text_fields(value)
            all_hits.extend(hits)
            out_dict[key] = cleaned_value
        return out_dict, all_hits

    return obj, all_hits


def build_minimal_reference(underspecified: str, gold_detail: str, *, intro_words: int = 85) -> str:
    intro = " ".join(str(underspecified or "").split()[:intro_words]).strip()
    if intro and not intro.endswith((".", "!", "?")):
        intro = intro.rstrip(",;:") + "."
    detail = str(gold_detail or "").strip()
    if intro and detail:
        if not detail.endswith((".", "!", "?")):
            detail += "."
        return f"{intro} {detail}"
    return detail or intro


def contract_codification_ready_reference(instance: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    defect = (instance.get("defects") or [{}])[0]
    gold_detail = str(defect.get("gold_detail_removed_or_corrupted") or "").strip()
    underspecified = str((instance.get("input") or {}).get("underspecified_spec") or "")
    reference = str((instance.get("gold") or {}).get("codification_ready_reference") or "").strip()

    if not gold_detail:
        return reference, {"applied": False, "reason": "missing_gold_detail"}

    defect_clusters = topic_clusters(gold_detail)
    detail_tokens = set(significant_tokens(gold_detail, limit=30))

    kept: List[str] = []
    removed: List[str] = []
    for sentence in split_sentences(reference):
        sent_clusters = topic_clusters(sentence)
        extra = sent_clusters - defect_clusters
        sent_tokens = set(significant_tokens(sentence, limit=20))
        overlap = len(sent_tokens & detail_tokens) / max(len(detail_tokens), 1)

        if not extra or overlap >= 0.12 or not sent_clusters:
            kept.append(sentence)
        else:
            removed.append(sentence)

    contracted = " ".join(kept).strip()
    norm_detail = normalize_text(gold_detail)
    if norm_detail and norm_detail not in normalize_text(contracted):
        if contracted:
            contracted = contracted.rstrip(".") + ". " + gold_detail
        else:
            contracted = build_minimal_reference(underspecified, gold_detail)

    remaining_extra = sorted(topic_clusters(contracted) - defect_clusters)
    if remaining_extra:
        contracted = build_minimal_reference(underspecified, gold_detail)
        remaining_extra = sorted(topic_clusters(contracted) - defect_clusters)

    meta = {
        "applied": contracted != reference,
        "original_word_count": word_count(reference),
        "contracted_word_count": word_count(contracted),
        "removed_sentence_count": len(removed),
        "remaining_extra_clusters": remaining_extra,
        "defect_topic_clusters": sorted(defect_clusters),
    }
    return contracted, meta


def repair_benchmark_instance(instance: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    out = json.loads(json.dumps(instance, ensure_ascii=False))
    repair_log: Dict[str, Any] = {}

    inp = dict(out.get("input") or {})
    underspecified = str(inp.get("underspecified_spec") or "")
    cleaned_input, input_hits = strip_citation_references(underspecified)
    inp["underspecified_spec"] = cleaned_input
    out["input"] = inp
    repair_log["citation_hits_input"] = input_hits

    gold = dict(out.get("gold") or {})
    paper_spec = gold.get("paper_derived_specification")
    if paper_spec is not None:
        cleaned_spec, spec_hits = sanitize_text_fields(paper_spec)
        gold["paper_derived_specification"] = cleaned_spec
        repair_log["citation_hits_paper_derived_specification"] = spec_hits

    contracted_ref, contraction_meta = contract_codification_ready_reference(out)
    cleaned_ref, ref_hits = strip_citation_references(contracted_ref)
    cleaned_ref, ref_lang_hits = strip_reference_language_from_text(cleaned_ref)
    gold["codification_ready_reference"] = cleaned_ref
    out["gold"] = gold
    repair_log["citation_hits_codification_ready_reference"] = ref_hits
    repair_log["reference_language_hits_codification_ready_reference"] = ref_lang_hits
    repair_log["gold_spec_contraction"] = contraction_meta

    out, taxonomy_repair = repair_taxonomy_labels(out)
    repair_log["taxonomy_repair"] = taxonomy_repair

    cm = dict(out.get("construction_metadata") or {})
    cm["step8_repairs"] = {
        "citation_stripped": bool(
            input_hits or ref_hits or repair_log.get("citation_hits_paper_derived_specification")
        ),
        "gold_spec_contraction": contraction_meta,
        "taxonomy_repair": taxonomy_repair,
    }
    out["construction_metadata"] = cm

    return out, repair_log


def defect_to_gap_dict(instance: Dict[str, Any]) -> Dict[str, Any]:
    defect = (instance.get("defects") or [{}])[0]
    cm = instance.get("construction_metadata") or {}
    return {
        "level1": defect.get("level1"),
        "level2": defect.get("level2"),
        "gap_summary": cm.get("gap_summary", ""),
        "gap_quote": cm.get("gap_quote", ""),
        "solution_summary": cm.get("solution_summary", ""),
        "solution_quote": cm.get("solution_quote", ""),
        "gold_clarified_detail": defect.get("gold_detail_removed_or_corrupted", ""),
        "why_this_blocks_or_affects_codification": defect.get(
            "why_this_blocks_or_affects_codification", ""
        ),
    }


def repair_taxonomy_labels(instance: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    out = dict(instance)
    gap = defect_to_gap_dict(out)
    original_level1 = gap.get("level1")
    original_level2 = gap.get("level2")
    defects = list(out.get("defects") or [])
    original_slot = None
    if defects:
        original_slot = defects[0].get("codification_slot") or defects[0].get("slot")

    corrected = apply_taxonomy_correction_to_gap(gap)
    expected_slot = infer_codification_slot_from_level2(
        corrected.get("level2"),
        gap_text_blob(corrected),
    )

    meta = {
        "applied": corrected.get("level2") != original_level2,
        "original_level1": original_level1,
        "original_level2": original_level2,
        "corrected_level1": corrected.get("level1"),
        "corrected_level2": corrected.get("level2"),
        "taxonomy_correction_reason": corrected.get("taxonomy_correction_reason"),
        "slot_repair_applied": expected_slot != original_slot,
        "original_codification_slot": original_slot,
        "corrected_codification_slot": expected_slot,
        "slot_alignment_reason": corrected.get("slot_alignment_reason"),
    }

    if defects:
        defect = dict(defects[0])
        defect["level1"] = corrected.get("level1")
        defect["level2"] = corrected.get("level2")
        defect["slot"] = expected_slot
        defect["codification_slot"] = expected_slot
        defects[0] = defect
        out["defects"] = defects

    if not meta["applied"] and not meta["slot_repair_applied"]:
        return out, meta

    et = dict(out.get("eval_targets") or {})
    underspecified = dict(et.get("underspecified") or {})
    expected_defects = list(underspecified.get("expected_defects") or [])
    if expected_defects:
        ed = dict(expected_defects[0])
        ed["level1"] = corrected.get("level1")
        ed["level2"] = corrected.get("level2")
        ed["slot"] = expected_slot
        ed["codification_slot"] = expected_slot
        expected_defects[0] = ed
        underspecified["expected_defects"] = expected_defects
        et["underspecified"] = underspecified
        out["eval_targets"] = et

    selected = list((out.get("construction_metadata") or {}).get("selected_perturbations") or [])
    if selected:
        sp = dict(selected[0])
        sp["level1"] = corrected.get("level1")
        sp["level2"] = corrected.get("level2")
        sp["slot"] = expected_slot
        sp["codification_slot"] = expected_slot
        selected[0] = sp
        cm = dict(out.get("construction_metadata") or {})
        cm["selected_perturbations"] = selected
        out["construction_metadata"] = cm

    return out, meta


def assess_taxonomy_labels(instance: Dict[str, Any]) -> Dict[str, Any]:
    gap = defect_to_gap_dict(instance)
    defect = (instance.get("defects") or [{}])[0]
    gap["codification_slot"] = defect.get("codification_slot") or defect.get("slot")
    mislabel = assess_taxonomy_mislabel(gap)
    slot_consistency = assess_slot_level2_consistency(gap)
    bins = assess_taxonomy(instance)
    return {**bins, **mislabel, **slot_consistency}


def topic_clusters(text: str) -> Set[str]:
    low = normalize_text(text)
    hits: Set[str] = set()
    for cluster_id, terms in TOPIC_CLUSTERS:
        if any(term in low for term in terms):
            hits.add(cluster_id)
    return hits


def significant_tokens(text: str, limit: int = 14) -> List[str]:
    stop = {
        "that", "this", "with", "from", "into", "during", "should", "using",
        "used", "when", "where", "which", "their", "there", "these", "those",
        "model", "training", "evaluation", "method", "paper", "results",
    }
    words = re.findall(r"\b[a-z][a-z0-9_]{3,}\b", normalize_text(text))
    out: List[str] = []
    for w in words:
        if w in stop:
            continue
        if w not in out:
            out.append(w)
        if len(out) >= limit:
            break
    return out


def dedup_fingerprint(instance: Dict[str, Any]) -> str | None:
    source = instance.get("source") or {}
    repo = str(source.get("repo") or "").lower()
    defect = (instance.get("defects") or [{}])[0]
    gold = str(defect.get("gold_detail_removed_or_corrupted") or "")
    low = normalize_text(gold)

    key_phrases = sorted({p for p in DEDUP_KEY_PHRASES if p in low})
    if len(key_phrases) >= 2:
        slot = str(defect.get("codification_slot") or "")
        return f"{repo}::{slot}::{'|'.join(key_phrases)}"

    tokens = significant_tokens(gold, limit=10)
    if len(tokens) < 5:
        return None

    slot = str(defect.get("codification_slot") or "")
    level2 = str(defect.get("level2") or "")
    return f"{repo}::{slot}::{level2}::{'|'.join(tokens)}"


def jaccard_similarity(a: str, b: str) -> float:
    ta = set(significant_tokens(a, limit=30))
    tb = set(significant_tokens(b, limit=30))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def dedup_cluster_id(instance: Dict[str, Any], fp: str | None) -> str:
    if fp:
        return fp
    source = instance.get("source") or {}
    repo = str(source.get("repo") or "unknown")
    instance_id = str(instance.get("id") or "unknown")
    return f"singleton::{repo}::{instance_id}"


def instance_rank(instance: Dict[str, Any]) -> Tuple[int, int, int, str]:
    defect = (instance.get("defects") or [{}])[0]
    gold = str(defect.get("gold_detail_removed_or_corrupted") or "")
    ref = str((instance.get("gold") or {}).get("codification_ready_reference") or "")
    warnings = instance.get("quality_warnings") or []
    warning_penalty = sum(
        1 for w in warnings
        if isinstance(w, dict) or (isinstance(w, str) and w)
    )
    return (
        len(gold.split()),
        len(ref.split()),
        -warning_penalty,
        str(instance.get("id") or ""),
    )


def detect_repo_language(text: str) -> List[str]:
    low = normalize_text(text)
    return [term for term in REPO_LANGUAGE_TERMS if term in low]


def pattern_hits(text: str, patterns: List[re.Pattern[str]]) -> List[str]:
    hits: List[str] = []
    for pattern in patterns:
        if pattern.search(str(text or "")):
            hits.append(pattern.pattern)
    return hits


def benchmark_eval_text(instance: Dict[str, Any]) -> str:
    defect = (instance.get("defects") or [{}])[0]
    parts = [
        str((instance.get("input") or {}).get("underspecified_spec") or ""),
        str((instance.get("gold") or {}).get("codification_ready_reference") or ""),
        str(defect.get("gold_detail_removed_or_corrupted") or ""),
    ]
    return " ".join(parts)


def assess_hallucinated_language(instance: Dict[str, Any]) -> Dict[str, Any]:
    underspecified = str((instance.get("input") or {}).get("underspecified_spec") or "")
    reference = str((instance.get("gold") or {}).get("codification_ready_reference") or "")

    input_hits = pattern_hits(underspecified, HALLUCINATED_PATTERNS)
    reference_hits = pattern_hits(reference, HALLUCINATED_PATTERNS)

    return {
        "hallucinated_language_in_underspecified_spec": input_hits,
        "hallucinated_language_in_codification_ready_reference": reference_hits,
        "hallucinated_language_present": bool(input_hits or reference_hits),
    }


def assess_external_knowledge(instance: Dict[str, Any]) -> Dict[str, Any]:
    underspecified = str((instance.get("input") or {}).get("underspecified_spec") or "")
    reference = str((instance.get("gold") or {}).get("codification_ready_reference") or "")

    input_hits = pattern_hits(underspecified, EXTERNAL_KNOWLEDGE_PATTERNS)
    reference_hits = pattern_hits(reference, EXTERNAL_KNOWLEDGE_PATTERNS)

    return {
        "external_knowledge_in_underspecified_spec": input_hits,
        "external_knowledge_in_codification_ready_reference": reference_hits,
        "external_knowledge_injection": bool(input_hits or reference_hits),
    }


def assess_minimal_sufficiency(instance: Dict[str, Any]) -> Dict[str, Any]:
    defect = (instance.get("defects") or [{}])[0]
    gold_detail = str(defect.get("gold_detail_removed_or_corrupted") or "").strip()
    underspecified = str((instance.get("input") or {}).get("underspecified_spec") or "")
    reference = str((instance.get("gold") or {}).get("codification_ready_reference") or "")

    reasons: List[str] = []
    gold_words = word_count(gold_detail)

    if not gold_detail:
        reasons.append("missing_gold_detail")
    elif gold_words < 8:
        reasons.append("gold_detail_too_short")

    gold_low = gold_detail.lower()
    vague_markers = ["we assume", "we found no difference", "should probably", "may vary"]
    has_vague = any(marker in gold_low for marker in vague_markers)
    has_concrete = any(signal in gold_low for signal in CONCRETE_GOLD_SIGNALS)
    if has_vague and not has_concrete:
        reasons.append("uncertain_gold_detail_without_concrete_spec")

    if word_count(underspecified) < 60:
        reasons.append("underspecified_spec_too_short")
    if word_count(reference) < 80:
        reasons.append("codification_ready_reference_too_short")

    return {
        "minimal_sufficiency": not reasons,
        "reasons": reasons,
        "gold_detail_word_count": gold_words,
        "underspecified_word_count": word_count(underspecified),
        "reference_word_count": word_count(reference),
    }


def taxonomy_bin(instance: Dict[str, Any]) -> str:
    defect = (instance.get("defects") or [{}])[0]
    slot = str(defect.get("codification_slot") or defect.get("slot") or "").strip().upper()
    level1 = str(defect.get("level1") or "")
    level2 = str(defect.get("level2") or "").strip()

    if level2 == "Missing Evaluation Specification" or slot == "EVALUATION_PROTOCOL":
        return "EVALUATION_PROTOCOL"
    if level2 == "Missing Data Specification" or slot == "DATA_AND_PREPROCESSING":
        return "DATA_AND_PREPROCESSING"
    if slot == "MODEL_ARCHITECTURE" or level2 == "Missing Model Structure":
        return "MODEL_ARCHITECTURE"
    if (
        slot == "TRAINING_PROCEDURE"
        or level2 == "Missing Configuration Protocol"
        or level2 == "Conflicting Objective"
    ):
        return "TRAINING_PROCEDURE"
    if slot == "CORE_ALGORITHM" or level2 in {
        "Missing Method Procedure",
        "Ambiguous Definition",
        "Ambiguous Procedure",
    }:
        return "CORE_ALGORITHM"
    if slot == "TASK_AND_IO":
        return "TASK_AND_IO"
    if slot == "INFERENCE_AND_DECISION":
        return "INFERENCE_AND_DECISION"
    if slot == "OBJECTIVE_AND_SUPERVISION":
        return "OBJECTIVE_AND_SUPERVISION"
    if slot == "INTERNAL_CONSISTENCY" or level1 == "Inconsistency":
        return "INTERNAL_CONSISTENCY"
    if level1 == "Ambiguity":
        return "Ambiguity"
    return "other"


def assess_taxonomy(instance: Dict[str, Any]) -> Dict[str, Any]:
    defect = (instance.get("defects") or [{}])[0]
    level1 = str(defect.get("level1") or "")
    level2 = str(defect.get("level2") or "")
    slot = str(defect.get("codification_slot") or defect.get("slot") or "")
    granularity = str(defect.get("granularity") or "")

    return {
        "level1": level1,
        "level2": level2,
        "level2_normalized": LEVEL2_NORMALIZED.get(level2, level2),
        "codification_slot": slot,
        "granularity": granularity or P7.infer_granularity(defect),
        "taxonomy_bin": taxonomy_bin(instance),
    }


def attach_step8_metadata(instance: Dict[str, Any], audit: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(instance)
    cm = dict(out.get("construction_metadata") or {})
    checks = audit.get("checks") or {}
    cm["step8_audit"] = {
        "verdict": audit.get("verdict"),
        "reasons": audit.get("reasons"),
        "taxonomy": checks.get("taxonomy"),
        "minimal_sufficiency": checks.get("minimal_sufficiency"),
        "hallucinated_language": checks.get("hallucinated_language"),
        "external_knowledge": checks.get("external_knowledge"),
    }
    out["construction_metadata"] = cm
    return out


def build_taxonomy_bins(
    clean_rows: List[Dict[str, Any]],
    audits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    audit_by_id = {str(a.get("id") or ""): a for a in audits}

    by_slot: Counter = Counter()
    by_level1: Counter = Counter()
    by_bin: Counter = Counter()
    clean_ids_by_bin: Dict[str, List[str]] = defaultdict(list)
    clean_ids_by_slot: Dict[str, List[str]] = defaultdict(list)

    for row in clean_rows:
        iid = str(row.get("id") or "")
        audit = audit_by_id.get(iid) or {}
        taxonomy = (audit.get("checks") or {}).get("taxonomy") or assess_taxonomy(row)

        slot = str(taxonomy.get("codification_slot") or "unknown")
        level1 = str(taxonomy.get("level1") or "unknown")
        bin_name = str(taxonomy.get("taxonomy_bin") or "other")

        by_slot[slot] += 1
        by_level1[level1] += 1
        by_bin[bin_name] += 1
        clean_ids_by_bin[bin_name].append(iid)
        clean_ids_by_slot[slot].append(iid)

    return {
        "by_codification_slot": dict(by_slot),
        "by_level1": dict(by_level1),
        "by_taxonomy_bin": dict(by_bin),
        "clean_instance_ids_by_taxonomy_bin": dict(clean_ids_by_bin),
        "clean_instance_ids_by_codification_slot": dict(clean_ids_by_slot),
    }


def assess_over_specification(instance: Dict[str, Any]) -> Dict[str, Any]:
    defect = (instance.get("defects") or [{}])[0]
    gold_detail = str(defect.get("gold_detail_removed_or_corrupted") or "")
    reference = str((instance.get("gold") or {}).get("codification_ready_reference") or "")

    defect_clusters = topic_clusters(gold_detail)
    ref_clusters = topic_clusters(reference)
    extra_clusters = sorted(ref_clusters - defect_clusters)

    over_spec = bool(extra_clusters)
    if len(ref_clusters) >= 3 and len(defect_clusters) <= 1:
        over_spec = True

    return {
        "defect_topic_clusters": sorted(defect_clusters),
        "reference_topic_clusters": sorted(ref_clusters),
        "extra_reference_clusters": extra_clusters,
        "over_specification": over_spec,
    }


def assess_atomic_enforcement(instance: Dict[str, Any]) -> Dict[str, Any]:
    defects = instance.get("defects") or []
    readiness = instance.get("codification_readiness") or {}
    blocking = readiness.get("blocking_missing_specs") or []
    if not isinstance(blocking, list):
        blocking = [str(blocking)] if blocking else []

    defect = defects[0] if defects else {}
    gold_detail = str(defect.get("gold_detail_removed_or_corrupted") or "")
    detail_clusters = topic_clusters(gold_detail)

    reasons: List[str] = []
    if len(defects) != 1:
        reasons.append(f"defect_count_{len(defects)}")

    if len(blocking) > 1:
        reasons.append("multiple_blocking_missing_specs")

    interpretability_mix = {"embedding_viz", "interpretability", "pretraining"}
    if len(detail_clusters & interpretability_mix) >= 2:
        reasons.append("composite_defect_gold_detail")

    eval_hits = [c for c in detail_clusters if c.startswith("eval_")]
    if len(eval_hits) >= 2:
        reasons.append("composite_eval_metrics_in_defect")

    actions = instance.get("expected_clarification_actions") or []
    if len(actions) != 1:
        reasons.append(f"clarification_action_count_{len(actions)}")

    return {
        "atomic_violation": bool(reasons),
        "reasons": reasons,
        "blocking_missing_spec_count": len(blocking),
        "defect_topic_clusters": sorted(detail_clusters),
    }


def assess_citation_references(instance: Dict[str, Any]) -> Dict[str, Any]:
    underspecified = str((instance.get("input") or {}).get("underspecified_spec") or "")
    reference = str((instance.get("gold") or {}).get("codification_ready_reference") or "")
    paper_spec = (instance.get("gold") or {}).get("paper_derived_specification") or {}

    hits: List[str] = []
    hits.extend(find_citation_references(underspecified))
    hits.extend(find_citation_references(reference))

    def collect_from_obj(obj: Any) -> None:
        if isinstance(obj, str):
            hits.extend(find_citation_references(obj))
        elif isinstance(obj, list):
            for item in obj:
                collect_from_obj(item)
        elif isinstance(obj, dict):
            for value in obj.values():
                collect_from_obj(value)

    collect_from_obj(paper_spec)

    unique_hits = sorted(set(hits))
    return {
        "citation_references_present": bool(unique_hits),
        "citation_hits": unique_hits,
    }


def assess_leakage(instance: Dict[str, Any]) -> Dict[str, Any]:
    underspecified = str((instance.get("input") or {}).get("underspecified_spec") or "")
    reference = str((instance.get("gold") or {}).get("codification_ready_reference") or "")
    defect = (instance.get("defects") or [{}])[0]
    gold_detail = str(defect.get("gold_detail_removed_or_corrupted") or "")

    diagnostic_hits = P7.detect_diagnostic_leakage(underspecified)
    gold_leak = P7.contains_gold_detail(underspecified, gold_detail)
    repo_hits_input = detect_repo_language(underspecified)
    repo_hits_ref = detect_repo_language(reference)

    severe = bool(diagnostic_hits) or gold_leak

    return {
        "diagnostic_leakage_in_underspecified_spec": diagnostic_hits,
        "gold_detail_leakage_in_underspecified_spec": gold_leak,
        "repo_language_in_underspecified_spec": repo_hits_input,
        "repo_language_in_codification_ready_reference": repo_hits_ref,
        "severe_leakage": severe,
    }


def assess_schema(instance: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[Any] = []

    try:
        P7.validate_instance(instance, fail_on_input_leakage=False)
        warnings = list(instance.get("quality_warnings") or [])
    except ValueError as exc:
        errors.append(str(exc))

    return {
        "schema_valid": not errors,
        "schema_errors": errors,
        "pipeline_warnings": warnings,
    }


def assess_resolution_role(instance: Dict[str, Any]) -> Dict[str, Any]:
    defect = (instance.get("defects") or [{}])[0]
    slot = str(defect.get("codification_slot") or "").strip().upper()
    role = str(defect.get("resolution_role") or "")
    level2 = str(defect.get("level2") or "").strip()

    reproducibility_slots = {
        "EVALUATION_PROTOCOL",
        "DATA_AND_PREPROCESSING",
        "TRAINING_PROCEDURE",
    }
    reproducibility_level2 = level2 in {
        "Missing Evaluation Specification",
        "Missing Data Specification",
        "Missing Configuration Protocol",
    }
    wrong_role = (
        role == "implementation_blocker"
        and (slot in reproducibility_slots or reproducibility_level2)
    )
    return {
        "eval_or_hparam_as_implementation_blocker": wrong_role,
        "codification_slot": slot,
        "resolution_role": role,
    }


def choose_verdict(checks: Dict[str, Any]) -> str:
    if not checks["schema"]["schema_valid"]:
        return "reject"
    if checks["leakage"]["severe_leakage"]:
        return "reject"
    if checks.get("dedup_loser"):
        return "review"
    if checks["over_spec"]["over_specification"]:
        return "review"
    if checks["atomic"]["atomic_violation"]:
        return "review"
    if checks["resolution_role"]["eval_or_hparam_as_implementation_blocker"]:
        return "review"
    if checks["leakage"]["repo_language_in_underspecified_spec"]:
        return "review"
    if checks["leakage"]["repo_language_in_codification_ready_reference"]:
        return "review"
    if checks["citation"]["citation_references_present"]:
        return "review"
    if not checks["minimal_sufficiency"]["minimal_sufficiency"]:
        return "review"
    if checks["hallucinated_language"]["hallucinated_language_present"]:
        return "review"
    if checks["external_knowledge"]["external_knowledge_injection"]:
        return "review"
    if checks["taxonomy"]["taxonomy_mislabel"]:
        return "review"
    if not checks["taxonomy"].get("slot_level2_consistent", True):
        return "review"

    pipeline_warnings = checks["schema"]["pipeline_warnings"]
    for w in pipeline_warnings:
        if isinstance(w, dict):
            return "review"
        if isinstance(w, str) and w.endswith("_in_underspecified_spec"):
            return "review"

    return "clean"


def collect_review_reasons(checks: Dict[str, Any], verdict: str) -> List[str]:
    reasons: List[str] = []
    if verdict == "reject":
        reasons.extend(checks["schema"]["schema_errors"])
        if checks["leakage"]["severe_leakage"]:
            if checks["leakage"]["diagnostic_leakage_in_underspecified_spec"]:
                reasons.append("diagnostic_leakage_in_underspecified_spec")
            if checks["leakage"]["gold_detail_leakage_in_underspecified_spec"]:
                reasons.append("gold_detail_leakage_in_underspecified_spec")
        return reasons

    if checks.get("dedup_loser"):
        reasons.append("duplicate_gap_cluster_loser")
    if checks["over_spec"]["over_specification"]:
        reasons.append("over_specification_in_gold_reference")
        reasons.extend(
            f"extra_cluster:{c}"
            for c in checks["over_spec"]["extra_reference_clusters"]
        )
    reasons.extend(checks["atomic"]["reasons"])
    if checks["resolution_role"]["eval_or_hparam_as_implementation_blocker"]:
        reasons.append("eval_or_hparam_as_implementation_blocker")
    if checks["leakage"]["repo_language_in_underspecified_spec"]:
        reasons.append("repo_language_in_underspecified_spec")
    if checks["leakage"]["repo_language_in_codification_ready_reference"]:
        reasons.append("repo_language_in_codification_ready_reference")
    if checks["citation"]["citation_references_present"]:
        reasons.append("citation_reference_present")
        reasons.extend(f"citation:{hit}" for hit in checks["citation"]["citation_hits"])
    reasons.extend(checks["minimal_sufficiency"]["reasons"])
    if checks["hallucinated_language"]["hallucinated_language_present"]:
        reasons.append("hallucinated_language_present")
        reasons.extend(
            f"hallucinated_input:{hit}"
            for hit in checks["hallucinated_language"]["hallucinated_language_in_underspecified_spec"]
        )
        reasons.extend(
            f"hallucinated_reference:{hit}"
            for hit in checks["hallucinated_language"]["hallucinated_language_in_codification_ready_reference"]
        )
    if checks["external_knowledge"]["external_knowledge_injection"]:
        reasons.append("external_knowledge_injection")
        reasons.extend(
            f"external_input:{hit}"
            for hit in checks["external_knowledge"]["external_knowledge_in_underspecified_spec"]
        )
        reasons.extend(
            f"external_reference:{hit}"
            for hit in checks["external_knowledge"]["external_knowledge_in_codification_ready_reference"]
        )
    if checks["taxonomy"].get("taxonomy_mislabel"):
        reasons.append("taxonomy_mislabel")
        if checks["taxonomy"].get("mislabel_reason"):
            reasons.append(str(checks["taxonomy"].get("mislabel_reason")))
        if checks["taxonomy"].get("suggested_level2"):
            reasons.append(f"suggested_level2:{checks['taxonomy'].get('suggested_level2')}")
    if not checks["taxonomy"].get("slot_level2_consistent", True):
        reasons.append("slot_level2_inconsistent")
        expected = checks["taxonomy"].get("expected_codification_slot")
        actual = checks["taxonomy"].get("actual_codification_slot")
        if expected and actual:
            reasons.append(f"expected_slot:{expected}")
            reasons.append(f"actual_slot:{actual}")

    for w in checks["schema"]["pipeline_warnings"]:
        if isinstance(w, dict):
            reasons.append(str(w.get("type") or "pipeline_warning"))
        elif isinstance(w, str):
            reasons.append(w)

    return reasons


def assign_dedup_clusters(
    instances: List[Dict[str, Any]],
) -> Tuple[Dict[str, str], Set[str], Dict[str, List[str]]]:
    fp_by_id: Dict[str, str | None] = {}
    for inst in instances:
        fp_by_id[str(inst.get("id") or "")] = dedup_fingerprint(inst)

    cluster_members: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for inst in instances:
        iid = str(inst.get("id") or "")
        fp = fp_by_id.get(iid)
        cluster_id = dedup_cluster_id(inst, fp)
        cluster_members[cluster_id].append(inst)

    # Secondary merge: same repo + high gold_detail Jaccard within cluster groups
    merged: Dict[str, List[Dict[str, Any]]] = {}
    seen_ids: Set[str] = set()
    for cluster_id, members in cluster_members.items():
        if len(members) == 1:
            merged[cluster_id] = members
            continue

        groups: List[List[Dict[str, Any]]] = []
        for member in sorted(members, key=lambda x: str(x.get("id") or "")):
            mid = str(member.get("id") or "")
            if mid in seen_ids:
                continue
            defect = (member.get("defects") or [{}])[0]
            gold_a = str(defect.get("gold_detail_removed_or_corrupted") or "")
            placed = False
            for group in groups:
                gold_b = str(
                    (group[0].get("defects") or [{}])[0].get(
                        "gold_detail_removed_or_corrupted"
                    )
                    or ""
                )
                if jaccard_similarity(gold_a, gold_b) >= 0.72:
                    group.append(member)
                    placed = True
                    break
            if not placed:
                groups.append([member])

        for idx, group in enumerate(groups):
            sub_id = cluster_id if len(groups) == 1 else f"{cluster_id}::sub{idx}"
            merged[sub_id] = group
            for member in group:
                seen_ids.add(str(member.get("id") or ""))

    cluster_map: Dict[str, str] = {}
    cluster_index: Dict[str, List[str]] = {}
    losers: Set[str] = set()

    for cluster_id, members in merged.items():
        if len(members) == 1:
            iid = str(members[0].get("id") or "")
            cluster_map[iid] = cluster_id
            cluster_index[cluster_id] = [iid]
            continue

        ranked = sorted(members, key=instance_rank, reverse=True)
        winner_id = str(ranked[0].get("id") or "")
        member_ids = [str(m.get("id") or "") for m in members]
        cluster_index[cluster_id] = member_ids
        for member in members:
            iid = str(member.get("id") or "")
            cluster_map[iid] = cluster_id
            if iid != winner_id:
                losers.add(iid)

    return cluster_map, losers, cluster_index


def audit_instance(
    instance: Dict[str, Any],
    *,
    dedup_loser: bool,
    cluster_id: str,
    repair_log: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    checks = {
        "schema": assess_schema(instance),
        "leakage": assess_leakage(instance),
        "citation": assess_citation_references(instance),
        "over_spec": assess_over_specification(instance),
        "atomic": assess_atomic_enforcement(instance),
        "resolution_role": assess_resolution_role(instance),
        "minimal_sufficiency": assess_minimal_sufficiency(instance),
        "hallucinated_language": assess_hallucinated_language(instance),
        "external_knowledge": assess_external_knowledge(instance),
        "taxonomy": assess_taxonomy_labels(instance),
        "dedup_loser": dedup_loser,
        "dedup_cluster_id": cluster_id,
        "repairs": repair_log or {},
    }
    verdict = choose_verdict(checks)
    reasons = collect_review_reasons(checks, verdict)

    return {
        "id": instance.get("id"),
        "record_id": (instance.get("source") or {}).get("record_id"),
        "repo": (instance.get("source") or {}).get("repo"),
        "issue_number": (instance.get("source") or {}).get("issue_number"),
        "verdict": verdict,
        "reasons": reasons,
        "dedup_cluster_id": cluster_id,
        "dedup_loser": dedup_loser,
        "taxonomy_bin": checks["taxonomy"]["taxonomy_bin"],
        "checks": checks,
    }


def run_audit(
    instances: List[Dict[str, Any]],
    *,
    apply_repairs: bool = True,
) -> Dict[str, Any]:
    cluster_map, dedup_losers, cluster_index = assign_dedup_clusters(instances)

    audits: List[Dict[str, Any]] = []
    clean_rows: List[Dict[str, Any]] = []
    review_rows: List[Dict[str, Any]] = []
    reject_rows: List[Dict[str, Any]] = []

    for instance in tqdm(instances, desc="Step8 audit", unit="instance"):
        iid = str(instance.get("id") or "")
        repair_log: Dict[str, Any] = {}
        working = instance
        if apply_repairs:
            working, repair_log = repair_benchmark_instance(instance)

        audit = audit_instance(
            working,
            dedup_loser=iid in dedup_losers,
            cluster_id=cluster_map.get(iid, f"singleton::{iid}"),
            repair_log=repair_log,
        )
        audits.append(audit)

        if audit["verdict"] == "clean":
            clean_rows.append(attach_step8_metadata(working, audit))
        elif audit["verdict"] == "review":
            review_rows.append(attach_step8_metadata(working, audit))
        else:
            reject_rows.append(attach_step8_metadata(working, audit))

    verdict_counts = Counter(a["verdict"] for a in audits)
    reason_counts: Counter = Counter()
    for audit in audits:
        for reason in audit["reasons"]:
            reason_counts[reason.split(":")[0]] += 1

    taxonomy_bins = build_taxonomy_bins(clean_rows, audits)

    return {
        "summary": {
            "input_count": len(instances),
            "clean_count": len(clean_rows),
            "review_count": len(review_rows),
            "reject_count": len(reject_rows),
            "apply_repairs": apply_repairs,
            "verdict_counts": dict(verdict_counts),
            "top_reasons": dict(reason_counts.most_common(20)),
            "taxonomy_bins": taxonomy_bins["by_taxonomy_bin"],
            "dedup_clusters_with_multiple_members": {
                cid: ids
                for cid, ids in cluster_index.items()
                if len(ids) > 1
            },
        },
        "instances": audits,
        "taxonomy_bins": taxonomy_bins,
        "outputs": {
            "clean": clean_rows,
            "review": review_rows,
            "reject": reject_rows,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit benchmark instances: dedup, leakage, over-spec, atomic checks."
    )
    parser.add_argument("--input_path", type=Path, default=Path(DEFAULT_INPUT_PATH))
    parser.add_argument("--input_dir", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--no_apply_repairs",
        action="store_true",
        help="Skip citation stripping and gold spec contraction before audit.",
    )
    args = parser.parse_args()

    if args.input_dir is not None:
        instances = load_instances_from_dir(args.input_dir)
        input_label = str(args.input_dir)
    else:
        instances = load_jsonl(args.input_path)
        input_label = str(args.input_path)

    if not instances:
        raise ValueError(f"No benchmark instances loaded from {input_label}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_audit(instances, apply_repairs=not args.no_apply_repairs)

    clean_path = out_dir / "all_benchmark_instances.clean.jsonl"
    review_path = out_dir / "review_needed_instances.jsonl"
    audit_path = out_dir / "audit_report.json"
    taxonomy_path = out_dir / "taxonomy_bins.json"

    save_jsonl(result["outputs"]["clean"], clean_path)
    save_jsonl(result["outputs"]["review"], review_path)
    save_json(result["taxonomy_bins"], taxonomy_path)

    report = {
        "script": "step_08_audit_benchmark_instances.py",
        "input": input_label,
        "summary": result["summary"],
        "instances": result["instances"],
        "taxonomy_bins": result["taxonomy_bins"],
        "paths": {
            "clean": str(clean_path),
            "review_needed": str(review_path),
            "audit_report": str(audit_path),
            "taxonomy_bins": str(taxonomy_path),
        },
    }
    if result["outputs"]["reject"]:
        reject_path = out_dir / "rejected_instances.jsonl"
        save_jsonl(result["outputs"]["reject"], reject_path)
        report["paths"]["rejected"] = str(reject_path)
        report["summary"]["rejected_path"] = str(reject_path)

    save_json(report, audit_path)

    print("\n===== Step 8: Benchmark Audit =====")
    print(f"Input:  {input_label}")
    print(f"Out:    {out_dir}")
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print(f"\nClean:  {clean_path} ({result['summary']['clean_count']})")
    print(f"Review: {review_path} ({result['summary']['review_count']})")
    print(f"Audit:  {audit_path}")
    print(f"Bins:   {taxonomy_path}")
    print(json.dumps(result["taxonomy_bins"]["by_taxonomy_bin"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
