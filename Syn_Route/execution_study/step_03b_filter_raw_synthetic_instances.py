from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from common import (
    LEVEL2_TO_LEVEL1,
    LLMJsonClient,
    VALID_SPECIFICATION_SLOTS,
    add_common_args,
    contains_leak,
    ensure_dir,
    infer_codification_slot_from_level2,
    load_jsonl,
    normalize_level2,
    normalize_specification_slot,
    normalize_ws,
    safe_name,
    save_json,
    save_jsonl,
)


VALID_GRANULARITY = {"coarse", "medium", "fine"}
VALID_OPERATIONS = {
    "omit_detail",
    "abstract_detail",
    "make_ambiguous",
    "introduce_conflict",
}
DISALLOWED_LEVEL2 = {
    "missing objective/loss specification",
    "missing objective specification",
    "missing loss specification",
}

OPERATION_TO_STEP4 = {
    "omit_detail": "delete",
    "abstract_detail": "abstract",
    "make_ambiguous": "abstract",
    "introduce_conflict": "corrupt",
}


def load_gold_by_project(path: Path) -> Dict[str, Dict[str, Any]]:
    return {row["project_id"]: row for row in load_jsonl(path) if row.get("project_id")}


def normalize_granularity(raw: Any) -> str:
    value = normalize_ws(raw).lower()
    if value in VALID_GRANULARITY:
        return value
    if value == "low":
        return "fine"
    if value == "high":
        return "coarse"
    return "medium"


def normalize_operation(raw: Any) -> str:
    value = normalize_ws(raw).lower()
    aliases = {
        "delete": "omit_detail",
        "remove": "omit_detail",
        "omit": "omit_detail",
        "abstract": "abstract_detail",
        "make vague": "make_ambiguous",
        "ambiguous": "make_ambiguous",
        "corrupt": "introduce_conflict",
        "conflict": "introduce_conflict",
        "difference": "introduce_conflict",
    }
    value = aliases.get(value, value)
    return value if value in VALID_OPERATIONS else "abstract_detail"


def gold_text(gold: Dict[str, Any]) -> str:
    pds = gold.get("paper_derived_specification") or {}
    parts = [gold.get("codification_ready_reference", "")]
    for value in pds.values():
        if isinstance(value, list):
            parts.extend(str(x) for x in value)
        elif value:
            parts.append(str(value))
    return normalize_ws(" ".join(parts))


def detail_supported(detail: str, gold: Dict[str, Any]) -> bool:
    detail = normalize_ws(detail)
    if not detail:
        return False
    full = gold_text(gold)
    if detail.lower() in full.lower():
        return True
    tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", detail.lower()))
    if len(tokens) < 5:
        return False
    full_tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", full.lower()))
    return len(tokens & full_tokens) / max(1, len(tokens)) >= 0.55


def as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def detail_related_must_not_items(must_not: List[str], detail: str) -> List[str]:
    """Keep only must_not entries tied to the removed detail for leakage checking."""
    detail = normalize_ws(detail)
    if not detail:
        return []
    detail_low = detail.lower()
    detail_tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", detail_low))
    related: List[str] = []
    for item in must_not:
        phrase = normalize_ws(item)
        if not phrase or phrase == detail:
            continue
        phrase_low = phrase.lower()
        if phrase_low in detail_low or detail_low in phrase_low:
            related.append(phrase)
            continue
        phrase_tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", phrase_low))
        if not phrase_tokens:
            continue
        overlap = len(phrase_tokens & detail_tokens) / max(1, len(phrase_tokens))
        if overlap >= 0.5:
            related.append(phrase)
    return related


def llm_diagnostic_leakage(
    normalized: Dict[str, Any],
    client: LLMJsonClient,
    cache_dir: Path,
    resume: bool,
) -> Tuple[Optional[bool], List[str]]:
    instance_id = normalized.get("raw_instance_id") or f"{normalized.get('project_id', 'unknown')}_diag"
    cache_path = cache_dir / f"{safe_name(instance_id)}.json"
    if resume and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return bool(cached.get("has_diagnostic_leakage")), [str(x) for x in cached.get("reasons", [])][:8]

    system = (
        "You are an IDEAAMBIG benchmark leakage auditor. "
        "Decide whether underspecified_spec contains diagnostic/meta leakage. Return strict JSON only."
    )
    user_obj = {
        "task": (
            "Decide whether underspecified_spec contains diagnostic or benchmark-construction leakage."
        ),
        "diagnostic_leakage_means": [
            "The text explicitly tells the reader that a specification is missing, unspecified, unclear, defective, omitted, not defined, or not codification-ready.",
            "The text uses benchmark-construction/meta language such as defect, gold detail, removed detail, synthetic defect, or benchmark gap.",
        ],
        "not_diagnostic_examples": [
            "Normal method language such as 'bias benchmark dataset', 'inserting missing information', 'missing property rate', or 'missing tokens' in an algorithm step.",
            "Ordinary research-spec wording that describes method behavior without saying the spec itself is incomplete.",
        ],
        "decision_policy": [
            "Return has_diagnostic_leakage=true only for explicit meta/diagnostic leakage.",
            "Return false for normal English uses of words like missing, benchmark, omitted, or gap inside method descriptions.",
        ],
        "required_output_schema": {
            "has_diagnostic_leakage": "boolean",
            "confidence": "low|medium|high",
            "reasons": ["short strings"],
        },
        "instance": {
            "gold_detail_removed_or_corrupted": normalized.get("gold_detail_removed_or_corrupted", ""),
            "underspecified_spec": normalized.get("underspecified_spec", ""),
        },
    }
    raw_path = cache_dir / "raw_llm" / f"{safe_name(instance_id)}.txt"
    obj = client.call_json(system, json.dumps(user_obj, ensure_ascii=False, indent=2), raw_path)
    if not isinstance(obj, dict):
        raise RuntimeError("LLM diagnostic output is not a JSON object")
    has_leak = bool(obj.get("has_diagnostic_leakage"))
    reasons = [str(x) for x in (obj.get("reasons", []) or [])][:8]
    save_json(
        {
            "instance_id": instance_id,
            "has_diagnostic_leakage": has_leak,
            "confidence": str(obj.get("confidence", "medium")).strip().lower(),
            "reasons": reasons,
        },
        cache_path,
    )
    return has_leak, reasons


def validate_raw(raw: Dict[str, Any], gold: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    reasons: List[str] = []
    project_id = raw.get("project_id") or raw.get("record_id") or gold.get("project_id")
    level2 = normalize_level2(raw.get("level2"))
    level1 = normalize_ws(raw.get("level1")) or LEVEL2_TO_LEVEL1.get(level2, "")
    granularity = normalize_granularity(raw.get("granularity"))
    operation = normalize_operation(raw.get("defect_operation") or raw.get("operation_type"))
    detail = normalize_ws(raw.get("gold_detail_removed_or_corrupted") or raw.get("gold_detail_to_remove_or_corrupt"))
    underspecified = normalize_ws(raw.get("underspecified_spec"))
    surface = normalize_ws(raw.get("surface_form_in_underspecified_spec") or raw.get("safe_surface_form"))
    must_not = [normalize_ws(x) for x in as_list(raw.get("must_not_reveal")) if normalize_ws(x)]
    if detail and detail not in must_not:
        must_not.insert(0, detail)

    if not project_id:
        reasons.append("missing_project_id")
    if not underspecified:
        reasons.append("empty_underspecified_spec")
    if not detail:
        reasons.append("empty_gold_detail_removed_or_corrupted")
    if level2 in DISALLOWED_LEVEL2:
        reasons.append("disallowed_level2")
    if level2 not in LEVEL2_TO_LEVEL1:
        reasons.append("invalid_level2")
    if level1 != LEVEL2_TO_LEVEL1.get(level2):
        reasons.append("level1_level2_mismatch")
        level1 = LEVEL2_TO_LEVEL1.get(level2, level1)
    if not detail_supported(detail, gold):
        reasons.append("gold_detail_not_supported_by_gold_spec")
    if contains_leak(underspecified, [detail]):
        reasons.append("gold_detail_leakage")
    leaked = [x for x in detail_related_must_not_items(must_not, detail) if contains_leak(underspecified, [x])]
    if leaked:
        reasons.append("must_not_reveal_leakage")
    if len(raw.get("preservation_check", {}) or {}) == 0:
        reasons.append("missing_preservation_check")
    check = raw.get("preservation_check", {}) if isinstance(raw.get("preservation_check"), dict) else {}
    if check.get("extra_defects_introduced") is True:
        reasons.append("extra_defects_introduced")
    if check.get("target_detail_modified") is False:
        reasons.append("target_detail_not_modified")
    if check.get("non_target_details_preserved") is False:
        reasons.append("non_target_details_not_preserved")

    slot = normalize_specification_slot(raw.get("codification_slot"))
    if slot not in VALID_SPECIFICATION_SLOTS:
        slot = infer_codification_slot_from_level2(level2)

    normalized = dict(raw)
    normalized.update(
        {
            "project_id": project_id,
            "record_id": project_id,
            "target_slot": normalize_ws(raw.get("target_slot") or raw.get("source_field") or "implementation_details"),
            "source_field": normalize_ws(raw.get("source_field") or raw.get("target_slot") or "gold_reference"),
            "level1": level1,
            "level2": level2,
            "granularity": granularity,
            "resolution_role": "implementation_blocker",
            "codification_slot": slot,
            "defect_operation": operation,
            "gold_detail_removed_or_corrupted": detail,
            "gold_detail_to_remove_or_corrupt": detail,
            "underspecified_spec": underspecified,
            "surface_form_in_underspecified_spec": surface,
            "must_not_reveal": must_not[:20],
            "candidate_strength": normalize_ws(raw.get("candidate_strength") or "strong"),
            "quality_rationale": normalize_ws(raw.get("quality_rationale") or ""),
        }
    )
    return normalized, sorted(set(reasons))


def wrap_instance(raw: Dict[str, Any], gold: Dict[str, Any], idx: int, gold_specs_path: Path) -> Dict[str, Any]:
    project_id = raw["project_id"]
    defect_id = f"{project_id}_direct_synthetic_defect_{idx:03d}"
    defect = {
        "slot": raw["target_slot"],
        "level1": raw["level1"],
        "level2": raw["level2"],
        "granularity": raw["granularity"],
        "resolution_role": "implementation_blocker",
        "codification_slot": raw["codification_slot"],
        "gold_detail_removed_or_corrupted": raw["gold_detail_removed_or_corrupted"],
        "surface_form_in_underspecified_spec": raw["surface_form_in_underspecified_spec"],
        "why_this_blocks_or_affects_codification": normalize_ws(raw.get("why_this_blocks_or_affects_codification")),
    }
    return {
        "id": defect_id,
        "source": {
            "source_type": "IdeationExecutionGap_synthetic_controlled",
            "record_id": project_id,
            "source": "IdeationExecutionGap",
            "year": None,
            "venue_or_track": None,
            "report_title": None,
            "report_url": "https://openreview.net/forum?id=Fllp8l6Puy",
            "original_paper_title": gold.get("paper_derived_specification", {}).get("paper_title"),
            "original_paper_url": "https://openreview.net/forum?id=Fllp8l6Puy",
            "route": "synthetic_controlled_direct_llm",
            "routing_reason": "Executed paper/code provide codification-ready reference; LLM generated one controlled underspecified specification and deterministic checks validated it.",
            "route_confidence": 1.0,
            "paper_id": project_id,
            "split_group_id": project_id,
            "split_group_type": "project",
            "synthetic_defect_id": defect_id,
            "gold_spec_path": str(gold_specs_path),
        },
        "input": {"underspecified_spec": raw["underspecified_spec"]},
        "gold": {
            "paper_derived_specification": gold.get("paper_derived_specification", {}),
            "codification_ready_reference": gold.get("codification_ready_reference", ""),
        },
        "defects": [defect],
        "open_design_choices": [],
        "codification_readiness": {
            "is_ready": False,
            "readiness_score": 2,
            "blocking_missing_specs": [{"slot": defect["slot"], "reason": defect["why_this_blocks_or_affects_codification"]}],
            "open_design_choices": [],
            "reason": "The specification is intentionally not codification-ready because one implementation-relevant detail has been removed, abstracted, or corrupted.",
        },
        "expected_clarification_actions": [
            {
                "slot": defect["slot"],
                "action_type": "clarification_question",
                "question_or_action": normalize_ws(raw.get("expected_clarification_question")),
                "evidence_to_seek": f"Find the exact {defect['codification_slot']} detail in the executed paper or code.",
            }
        ],
        "construction_metadata": {
            "construction_method": "synthetic_controlled_direct_llm_single_defect_from_executed_project_gold_spec",
            "project_id": project_id,
            "split_group_id": project_id,
            "split_group_type": "project",
            "synthetic_defect_id": defect_id,
            "gold_spec_path": str(gold_specs_path),
            "candidate_strength": raw.get("candidate_strength", "strong"),
            "quality_rationale": raw.get("quality_rationale", ""),
            "num_defects": 1,
            "selected_perturbations": [raw],
            "perturbation_instruction": {
                "operation_type": OPERATION_TO_STEP4.get(raw["defect_operation"], "abstract"),
                "defect_operation": raw["defect_operation"],
                "how_to_modify_reference": "Modify exactly one implementation-critical detail in the codification-ready reference.",
                "must_not_reveal": raw.get("must_not_reveal", []),
                "safe_surface_form": raw.get("surface_form_in_underspecified_spec", ""),
            },
            "preservation_check": raw.get("preservation_check") or {
                "target_detail_modified": True,
                "non_target_details_preserved": True,
                "extra_defects_introduced": False,
                "brief_explanation": "Validated by Step 3b deterministic checks.",
            },
        },
        "eval_targets": {
            "underspecified": {
                "expected_ready": False,
                "expected_readiness_score": 2,
                "expected_defects": [
                    {
                        "defect_id": "d1",
                        "slot": defect["slot"],
                        "level1": defect["level1"],
                        "level2": defect["level2"],
                        "granularity": defect["granularity"],
                        "resolution_role": "implementation_blocker",
                        "codification_slot": defect["codification_slot"],
                    }
                ],
                "expected_clarification_actions": [{"slot": defect["slot"], "action_type": "clarification_question"}],
            },
            "reference": {
                "expected_ready": True,
                "expected_readiness_score": 5,
                "expected_defects": [],
                "expected_clarification_actions": [],
            },
        },
        "quality_warnings": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--gold_specs_path", type=Path, required=True)
    parser.add_argument("--max_instances_per_project", type=int, default=5)
    parser.add_argument("--use_llm", action="store_true", help="Use LLM to filter diagnostic/meta leakage.")
    parser.add_argument("--model", default="deepseek/deepseek-v4-pro")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    diagnostic_cache_dir = args.out_dir / "llm_diagnostic"
    ensure_dir(diagnostic_cache_dir)
    raw_rows = load_jsonl(args.input_path)
    if args.limit:
        raw_rows = raw_rows[: args.limit]
    gold_by_project = load_gold_by_project(args.gold_specs_path)
    client = LLMJsonClient(args.model, args.use_llm)

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    diagnostic_llm_failures = 0

    for raw in tqdm(raw_rows, desc="Step3b filter raw instances"):
        project_id = raw.get("project_id") or raw.get("record_id")
        gold = gold_by_project.get(project_id)
        if not gold:
            rejected.append({"raw": raw, "rejection_reasons": ["missing_gold_spec"], "project_id": project_id})
            continue
        normalized, reasons = validate_raw(raw, gold)
        diagnostic_llm_reasons: List[str] = []
        if not reasons and args.use_llm:
            try:
                has_diagnostic_leak, diagnostic_llm_reasons = llm_diagnostic_leakage(
                    normalized,
                    client,
                    diagnostic_cache_dir,
                    args.resume,
                )
                if has_diagnostic_leak:
                    reasons.append("diagnostic_language_leakage")
            except Exception as exc:
                diagnostic_llm_failures += 1
                diagnostic_llm_reasons = [f"diagnostic_llm_check_failed: {exc!r}"]
        audit.append(
            {
                "raw_instance_id": raw.get("raw_instance_id"),
                "project_id": normalized.get("project_id"),
                "accepted": not reasons,
                "rejection_reasons": reasons,
                "diagnostic_llm_reasons": diagnostic_llm_reasons,
                "level2": normalized.get("level2"),
                "granularity": normalized.get("granularity"),
                "defect_operation": normalized.get("defect_operation"),
            }
        )
        if reasons:
            rejected.append({"raw": normalized, "rejection_reasons": reasons, "project_id": project_id})
            continue
        if counts[project_id] >= args.max_instances_per_project:
            rejected.append({"raw": normalized, "rejection_reasons": ["max_instances_per_project_exceeded"], "project_id": project_id})
            continue
        counts[project_id] += 1
        accepted.append(wrap_instance(normalized, gold, counts[project_id], args.gold_specs_path))

    save_jsonl(accepted, args.out_dir / "all_instances.jsonl")
    save_jsonl(rejected, args.out_dir / "rejected_instances.jsonl")
    save_jsonl(audit, args.out_dir / "audit_instances.jsonl")
    summary = {
        "input_raw_instances": len(raw_rows),
        "accepted_instances": len(accepted),
        "rejected_instances": len(rejected),
        "unique_projects": len(set(i["construction_metadata"]["project_id"] for i in accepted)),
        "instances_per_project": dict(Counter(i["construction_metadata"]["project_id"] for i in accepted)),
        "level1_distribution": dict(Counter(i["defects"][0]["level1"] for i in accepted)),
        "level2_distribution": dict(Counter(i["defects"][0]["level2"] for i in accepted)),
        "codification_slot_distribution": dict(Counter(i["defects"][0]["codification_slot"] for i in accepted)),
        "granularity_distribution": dict(Counter(i["defects"][0]["granularity"] for i in accepted)),
        "defect_operation_distribution": dict(Counter(i["construction_metadata"]["perturbation_instruction"].get("defect_operation") for i in accepted)),
        "rejection_reasons": dict(Counter(r for item in rejected for r in item.get("rejection_reasons", []))),
        "diagnostic_llm_failures": diagnostic_llm_failures,
        "recommended_next_step_input": str(args.out_dir / "all_instances.jsonl"),
    }
    save_json(summary, args.out_dir / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()
