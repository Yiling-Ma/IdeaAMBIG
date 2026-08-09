from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from common import LLMJsonClient, add_common_args, ensure_dir, load_jsonl, normalize_level2, normalize_ws, safe_name, save_json, save_jsonl


ROUTINE_DETAIL_PATTERNS = [
    r"\bcross[- ]entropy loss\b",
    r"\bclassification loss\b",
    r"\badam\b",
    r"\bsgd\b",
    r"\blearning rate\b",
    r"\bbatch size\b",
    r"\bepochs?\b",
    r"\brandom seed\b",
    r"\bweight decay\b",
    r"\bmomentum\b",
    r"\bdropout\b",
]

METHOD_CRITICAL_HYPERPARAMETER_HINTS = [
    "temperature",
    "candidate actions",
    "noise standard deviation",
    "policy smoothing",
    "softmax estimation",
    "subsets",
    "threshold",
    "radius",
    "trade-off",
    "samples",
    "centers",
    "class centers",
]

ALLOWED_LEVEL2 = {
    "Ambiguous Definition",
    "Ambiguous Procedure",
    "Missing Method Procedure",
    "Missing Configuration Protocol",
    "Missing Model Structure",
    "Missing Evaluation Specification",
    "Missing Data Specification",
    "Conflicting Objective",
    "Conflicting Model Design",
    "Conflicting Formal Definition",
}


def regex_any(patterns: List[str], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)


def get_defect(inst: Dict[str, Any]) -> Dict[str, Any]:
    defects = inst.get("defects") or []
    return defects[0] if defects and isinstance(defects[0], dict) else {}


def deterministic_reasons(inst: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    defects = inst.get("defects") or []
    if len(defects) != 1:
        reasons.append("not_exactly_one_defect")
        return reasons

    defect = get_defect(inst)
    detail = normalize_ws(defect.get("gold_detail_removed_or_corrupted"))
    detail_low = detail.lower()
    level2 = normalize_level2(defect.get("level2"))
    text = normalize_ws(inst.get("input", {}).get("underspecified_spec"))

    if not text:
        reasons.append("missing_underspecified_spec")
    if level2 not in ALLOWED_LEVEL2:
        reasons.append("invalid_level2")
    if "missing objective/loss specification" in str(defect.get("level2") or "").lower():
        reasons.append("disallowed_missing_objective_loss_label")

    if detail and detail.lower() in text.lower():
        reasons.append("gold_detail_exact_leakage")

    if regex_any(ROUTINE_DETAIL_PATTERNS, detail_low):
        method_critical = any(hint in detail_low for hint in METHOD_CRITICAL_HYPERPARAMETER_HINTS)
        if not method_critical:
            reasons.append("routine_training_or_hyperparameter_detail")

    if level2 == "Missing Configuration Protocol":
        method_critical = any(hint in detail_low for hint in METHOD_CRITICAL_HYPERPARAMETER_HINTS)
        if not method_critical:
            reasons.append("weak_hyperparameter_protocol")

    return sorted(set(reasons))


def build_review_prompt(inst: Dict[str, Any]) -> Tuple[str, str]:
    defect = get_defect(inst)
    system = """
You are an expert quality auditor for IDEAAMBIG synthetic benchmark instances.
Return strict JSON only. Do not include markdown fences.
""".strip()
    user_obj = {
        "task": (
            "Decide whether this instance should remain in the final benchmark. "
            "Reject weak or leaky instances even if the schema is valid."
        ),
        "acceptance_criteria": [
            "The input is self-contained and reads like a natural research/method specification.",
            "There is exactly one target defect.",
            "The removed/corrupted gold detail is not revealed in the input.",
            "The defect blocks or materially affects faithful implementation, not just exact reproduction of a routine setting.",
            "The Level-1/Level-2 label matches the defect.",
            "The expected clarification question targets the missing or ambiguous implementation detail.",
        ],
        "reject_if": [
            "The target detail is an ordinary training recipe detail such as plain cross-entropy loss, Adam, SGD, learning rate, batch size, epoch count, random seed, dropout, or weight decay.",
            "The target detail is a hyperparameter value that is not part of a non-standard method mechanism.",
            "The input leaks the removed detail or a close paraphrase.",
            "The instance introduces multiple defects.",
            "The defect is only about runtime, hardware, environment, or incidental reproducibility trivia.",
            "The label is incorrect or too broad for the target detail.",
        ],
        "decision_policy": [
            "Use accept=true only for strong or medium-quality IDEAAMBIG instances.",
            "Use accept=false for weak instances.",
            "A method-specific hyperparameter can be accepted if it is central to an algorithmic mechanism, such as SD3 softmax target estimation or IFSL feature stratification.",
            "Plain cross-entropy loss for a normal classifier is weak and should be rejected unless the paper's method specifically redefines the loss.",
        ],
        "required_output_schema": {
            "accept": "boolean",
            "quality_strength": "strong|medium|weak",
            "rejection_reasons": ["short strings; empty if accepted"],
            "quality_notes": "short explanation",
            "label_correct": "boolean",
            "single_defect": "boolean",
            "non_leaky": "boolean",
            "codification_relevant": "boolean",
        },
        "instance": {
            "id": inst.get("id"),
            "underspecified_spec": inst.get("input", {}).get("underspecified_spec", ""),
            "gold_reference": inst.get("gold", {}).get("codification_ready_reference", ""),
            "defect": defect,
            "expected_clarification_actions": inst.get("expected_clarification_actions", []),
            "construction_metadata": {
                "project_id": inst.get("construction_metadata", {}).get("project_id"),
                "candidate_strength": inst.get("construction_metadata", {}).get("candidate_strength"),
                "quality_rationale": inst.get("construction_metadata", {}).get("quality_rationale"),
            },
        },
    }
    return system, json.dumps(user_obj, ensure_ascii=False, indent=2)


def parse_review(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {
            "accept": False,
            "quality_strength": "weak",
            "rejection_reasons": ["llm_review_not_object"],
            "quality_notes": "",
            "label_correct": False,
            "single_defect": False,
            "non_leaky": False,
            "codification_relevant": False,
        }
    reasons = obj.get("rejection_reasons") or []
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    return {
        "accept": bool(obj.get("accept")),
        "quality_strength": normalize_ws(obj.get("quality_strength") or "weak").lower(),
        "rejection_reasons": [normalize_ws(x) for x in reasons if normalize_ws(x)],
        "quality_notes": normalize_ws(obj.get("quality_notes")),
        "label_correct": bool(obj.get("label_correct")),
        "single_defect": bool(obj.get("single_defect")),
        "non_leaky": bool(obj.get("non_leaky")),
        "codification_relevant": bool(obj.get("codification_relevant")),
    }


def llm_review(inst: Dict[str, Any], client: LLMJsonClient, out_dir: Path, resume: bool) -> Dict[str, Any]:
    inst_id = inst.get("id") or "unknown_instance"
    review_path = out_dir / "per_instance" / safe_name(inst_id) / "review.json"
    ensure_dir(review_path.parent)
    if resume and review_path.exists() and review_path.stat().st_size > 50:
        return json.loads(review_path.read_text(encoding="utf-8"))

    system, user = build_review_prompt(inst)
    obj = client.call_json(system, user, out_dir / "raw_llm" / f"{safe_name(inst_id)}.txt")
    review = parse_review(obj)
    review["id"] = inst_id
    save_json(review, review_path)
    return review


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--model", default="deepseek/deepseek-v4-pro")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(args.out_dir / "per_instance")
    ensure_dir(args.out_dir / "raw_llm")

    instances = load_jsonl(args.input_path)
    if args.limit:
        instances = instances[: args.limit]

    client = LLMJsonClient(args.model, args.use_llm)
    final: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []

    for inst in tqdm(instances, desc="Step5 quality review"):
        det_reasons = deterministic_reasons(inst)
        review: Dict[str, Any] = {
            "accept": not det_reasons,
            "quality_strength": "medium" if not det_reasons else "weak",
            "rejection_reasons": det_reasons,
            "quality_notes": "deterministic review only",
            "label_correct": True,
            "single_defect": len(inst.get("defects") or []) == 1,
            "non_leaky": "gold_detail_exact_leakage" not in det_reasons,
            "codification_relevant": not det_reasons,
        }
        if args.use_llm and not det_reasons:
            try:
                review = llm_review(inst, client, args.out_dir, args.resume)
            except Exception as exc:
                review = {
                    "accept": False,
                    "quality_strength": "weak",
                    "rejection_reasons": ["llm_review_failed"],
                    "quality_notes": repr(exc),
                    "label_correct": False,
                    "single_defect": False,
                    "non_leaky": False,
                    "codification_relevant": False,
                }

        combined_reasons = sorted(set(det_reasons + (review.get("rejection_reasons") or [])))
        accept = not det_reasons and bool(review.get("accept"))
        row = {
            "id": inst.get("id"),
            "project_id": inst.get("construction_metadata", {}).get("project_id"),
            "accepted": accept,
            "deterministic_reasons": det_reasons,
            "llm_review": review,
            "combined_rejection_reasons": combined_reasons,
            "level2": get_defect(inst).get("level2"),
            "codification_slot": get_defect(inst).get("codification_slot"),
            "gold_detail_removed_or_corrupted": get_defect(inst).get("gold_detail_removed_or_corrupted"),
        }
        audit.append(row)

        if accept:
            inst = dict(inst)
            inst.setdefault("construction_metadata", {})["step5_quality_review"] = review
            final.append(inst)
        else:
            rejected.append({"instance": inst, "audit": row, "rejection_reasons": combined_reasons})

        save_jsonl(final, args.out_dir / "final_instances.jsonl")
        save_jsonl(rejected, args.out_dir / "rejected_instances.jsonl")
        save_jsonl(audit, args.out_dir / "audit_instances.jsonl")

    summary = {
        "input_instances": len(instances),
        "final_instances": len(final),
        "rejected_instances": len(rejected),
        "unique_projects": len(set(i.get("construction_metadata", {}).get("project_id") for i in final)),
        "instances_per_project": dict(Counter(i.get("construction_metadata", {}).get("project_id") for i in final)),
        "level1_distribution": dict(Counter(get_defect(i).get("level1") for i in final)),
        "level2_distribution": dict(Counter(get_defect(i).get("level2") for i in final)),
        "codification_slot_distribution": dict(Counter(get_defect(i).get("codification_slot") for i in final)),
        "quality_strength_distribution": dict(Counter(i.get("construction_metadata", {}).get("step5_quality_review", {}).get("quality_strength", "unknown") for i in final)),
        "rejection_reasons": dict(Counter(r for item in rejected for r in item.get("rejection_reasons", []))),
    }
    save_json(summary, args.out_dir / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()
