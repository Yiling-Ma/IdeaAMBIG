from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from tqdm import tqdm

from common import LLMJsonClient, add_common_args, ensure_dir, load_jsonl, normalize_ws, safe_name, save_json, save_jsonl


HARD_WARNING_PATTERNS = {
    "equation_like_expression": [
        r"\b[A-Za-z][A-Za-z0-9_^{}\\]*\s*=",
        r"=\s*[^,.;]{8,}",
    ],
    "dense_formula_detail": [
        r"\\sum",
        r"\\prod",
        r"\\partial",
        r"∑",
        r"∏",
        r"∂",
        r"\|det",
        r"\blog\s*Z\b",
        r"\bL\s*=",
        r"\bp\([^)]*\)\s*=",
        r"\bA_\{",
        r"\bsoftmax_β\s*=",
    ],
    "low_level_symbolic_parameterization": [
        r"\bdelta\s*=",
        r"\bδ\s*=",
        r"\br_δ\b",
        r"\bv\^\(",
        r"\bW_q",
        r"\bW_key\b",
        r"\blog[- ]?jacobian",
        r"\bjacobian\b",
        r"\bξ_i\b",
        r"\bη_i\b",
        r"\bh_φ\b",
        r"\bλ_θ\b",
    ],
    "optimizer_recipe": [
        r"\busing adam\b",
        r"\bwith adam\b",
        r"\busing sgd\b",
        r"\bwith sgd\b",
        r"\bstandard optimizers?\b",
    ],
    "routine_training_loss": [
        r"\btraining minimizes cross[- ]entropy\b",
        r"\btraining uses cross[- ]entropy\b",
        r"\bcross[- ]entropy loss using\b",
    ],
    "claimed_method_advantage": [
        r"\bshows?\b.*\b(faster|better|improved|stronger|superior)",
        r"\bimproves?\b.*\b(robustness|accuracy|performance|convergence)",
        r"\bdesigned to improve\b",
        r"\binducing high[- ]density\b",
        r"\bhigh[- ]density feature\b",
        r"\bintra[- ]class compactness\b",
        r"\bbetter separation\b",
        r"\bavoids?\b.*\b(collapse|degenerate|failure)",
    ],
    "inferred_decision_rule": [
        r"\bstandard softmax logits\b",
        r"\bsoftmax logits are only applied at inference\b",
        r"\bsoftmax is only applied at inference\b",
        r"\bsoftmax\b.*\be\.g\.",
        r"\be\.g\.,?\s*negative squared distances\b",
        r"\bpaper-specified decision rule\b",
    ],
    "uncertainty_text": [
        r"\bnot explicitly specified\b",
        r"\bnot specified in the paper\b",
        r"\bexact .* is not specified\b",
    ],
    "experiment_only_data_generation_phase": [
        r"\(\s*1\s*\)\s*data generation",
        r"\bphase[s]?:\s*\(1\)\s*data generation",
        r"\bdata generation:\s*synthetic sources\b",
    ],
}


SOFT_WARNING_PATTERNS = {
    "many_numeric_implementation_constants": [
        r"\b\d+\s+(heads|layers|flow steps|bins|epochs|tasks|classes)\b",
        r"\bhidden dimension\s+\d+\b",
        r"\bbatch size\s+\d+\b",
        r"\blearning rate\s+[0-9]\b",
        r"\be\.g\.,?\s*\d+\b",
    ],
}


def regex_any(patterns: Iterable[str], text: str) -> bool:
    for pattern in patterns:
        try:
            if re.search(pattern, text, flags=re.I):
                return True
        except re.error:
            continue
    return False


def word_count(text: Any) -> int:
    return len(normalize_ws(text).split())


def audit_reference(reference: str) -> List[str]:
    warnings: List[str] = []
    for name, patterns in HARD_WARNING_PATTERNS.items():
        if regex_any(patterns, reference):
            warnings.append(name)
    for name, patterns in SOFT_WARNING_PATTERNS.items():
        hits = sum(1 for pattern in patterns if regex_any([pattern], reference))
        if hits >= 2:
            warnings.append(name)
    wc = word_count(reference)
    if wc < 100:
        warnings.append("reference_too_short")
    if wc > 260:
        warnings.append("reference_too_long")
    return sorted(set(warnings))


def hard_warnings(warnings: List[str]) -> List[str]:
    hard = set(HARD_WARNING_PATTERNS) | {"reference_too_short", "reference_too_long"}
    return [w for w in warnings if w in hard]


def compact_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in [
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
    ]:
        value = spec.get(key)
        if isinstance(value, list):
            out[key] = [normalize_ws(x)[:700] for x in value[:12] if normalize_ws(x)]
        elif normalize_ws(value):
            out[key] = normalize_ws(value)[:1600]
    return out


def build_repair_prompt(gold: Dict[str, Any], warnings: List[str]) -> Tuple[str, str]:
    spec = gold.get("paper_derived_specification") or {}
    system = """
You repair codification_ready_reference fields for IDEAAMBIG.
Return strict JSON only. Do not include markdown fences.
""".strip()
    user_obj = {
        "task": (
            "Rewrite only codification_ready_reference into a compact method handoff specification. "
            "Keep the scientific method faithful to the structured specification."
        ),
        "style_target": [
            "One natural-language paragraph, 120-230 words.",
            "Method handoff style: method name/type, inputs, 2-4 implementation phases, evaluation setup.",
            "Use operational descriptions, not equations.",
            "The result should resemble a clear method spec such as: 'The method, X, is a ... Given ..., it proceeds in three phases: (1) ..., (2) ..., (3) ... . The approach is evaluated on ... against ... using ... .'",
        ],
        "must_remove": [
            "exact symbolic equations or expressions containing '='",
            "low-level symbolic parameterizations such as W_q, r_delta, h_phi, lambda_theta, Jacobian terms, target-value equations, or density normalizers",
            "ordinary optimizer names such as Adam or SGD",
            "routine training recipe details such as learning rate, batch size, epochs, random seeds, dropout, momentum, or weight decay",
            "claimed advantages or empirical conclusions such as improves robustness, high-density features, faster convergence, avoids collapse, or better separation",
            "inferred decision-rule examples such as 'e.g., negative squared distances'",
            "uncertainty text such as 'not explicitly specified'",
            "experiment-only data generation as a method phase, unless data generation is the proposed method",
            "many numeric architecture constants such as exact layer counts, heads, hidden dimensions, flow steps, or bins",
        ],
        "must_preserve": [
            "task and inputs/outputs",
            "core method mechanism",
            "method-defining objective or training logic described in words",
            "inference behavior only when supported by the structured specification",
            "datasets, baselines, and metrics at a concise evaluation-protocol level",
        ],
        "known_warnings_to_fix": warnings,
        "current_codification_ready_reference": gold.get("codification_ready_reference", ""),
        "structured_gold_specification": compact_spec(spec),
        "required_output_schema": {
            "codification_ready_reference": "string",
            "repair_notes": ["short strings explaining what was removed or compressed"],
        },
    }
    return system, json.dumps(user_obj, ensure_ascii=False, indent=2)


def parse_repair(obj: Any) -> Tuple[str, List[str]]:
    if isinstance(obj, dict):
        ref = normalize_ws(obj.get("codification_ready_reference"))
        notes = obj.get("repair_notes") or []
        if not isinstance(notes, list):
            notes = [str(notes)]
        return ref, [normalize_ws(x) for x in notes if normalize_ws(x)]
    return "", ["repair_output_not_object"]


def repair_one(gold: Dict[str, Any], client: LLMJsonClient, out_dir: Path, resume: bool) -> Dict[str, Any]:
    record_id = gold.get("record_id") or gold.get("project_id") or "unknown_record"
    record_dir = out_dir / "per_record" / safe_name(record_id)
    ensure_dir(record_dir)
    repair_path = record_dir / "repair.json"
    initial_reference = gold.get("codification_ready_reference", "")
    initial_warnings = audit_reference(initial_reference)

    if resume and repair_path.exists() and repair_path.stat().st_size > 50:
        repaired_obj = json.loads(repair_path.read_text(encoding="utf-8"))
        repaired = dict(gold)
        repaired["codification_ready_reference"] = repaired_obj.get("codification_ready_reference", "")
        repaired.setdefault("gold_extraction_metadata", {})["reference_repair_metadata"] = repaired_obj.get("repair_metadata", {})
        repaired["quality_warnings"] = repaired_obj.get("final_warnings", [])
        return repaired

    if not initial_warnings:
        repaired = dict(gold)
        repaired.setdefault("gold_extraction_metadata", {})["reference_repair_metadata"] = {
            "status": "not_needed",
            "initial_warnings": [],
            "final_warnings": [],
        }
        return repaired

    system, user = build_repair_prompt(gold, initial_warnings)
    obj = client.call_json(system, user, out_dir / "raw_llm" / f"{safe_name(record_id)}.txt")
    repaired_reference, repair_notes = parse_repair(obj)
    final_warnings = audit_reference(repaired_reference)

    repaired = dict(gold)
    repaired["codification_ready_reference"] = repaired_reference
    existing_warnings = [
        w
        for w in (gold.get("quality_warnings") or [])
        if not str(w).startswith("reference_")
    ]
    repaired["quality_warnings"] = sorted(set(existing_warnings + final_warnings))
    repaired.setdefault("gold_extraction_metadata", {})["reference_repair_metadata"] = {
        "status": "repaired",
        "initial_warnings": initial_warnings,
        "final_warnings": final_warnings,
        "repair_notes": repair_notes,
    }

    save_json(
        {
            "record_id": record_id,
            "initial_codification_ready_reference": initial_reference,
            "codification_ready_reference": repaired_reference,
            "repair_metadata": repaired["gold_extraction_metadata"]["reference_repair_metadata"],
            "final_warnings": final_warnings,
        },
        repair_path,
    )
    return repaired


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--model", default="deepseek/deepseek-v4-pro")
    args = parser.parse_args()

    if not args.use_llm:
        raise SystemExit("Step 2b requires --use_llm because it repairs references with an LLM.")

    ensure_dir(args.out_dir)
    ensure_dir(args.out_dir / "raw_llm")
    ensure_dir(args.out_dir / "per_record")

    rows = load_jsonl(args.input_path)
    if args.limit:
        rows = rows[: args.limit]

    client = LLMJsonClient(args.model, True)
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []

    for gold in tqdm(rows, desc="Step2b repair gold references"):
        record_id = gold.get("record_id") or gold.get("project_id") or "unknown_record"
        try:
            repaired = repair_one(gold, client, args.out_dir, args.resume)
            final_warnings = audit_reference(repaired.get("codification_ready_reference", ""))
            final_hard = hard_warnings(final_warnings)
            row = {
                "record_id": record_id,
                "accepted": not final_hard,
                "initial_warnings": audit_reference(gold.get("codification_ready_reference", "")),
                "final_warnings": final_warnings,
                "final_hard_warnings": final_hard,
                "word_count": word_count(repaired.get("codification_ready_reference", "")),
            }
            audit.append(row)
            if final_hard:
                rejected.append({"record_id": record_id, "reason": "reference_repair_failed_quality_gate", "audit": row, "gold": repaired})
            else:
                accepted.append(repaired)
        except Exception as exc:
            row = {
                "record_id": record_id,
                "accepted": False,
                "error": repr(exc),
            }
            audit.append(row)
            rejected.append({"record_id": record_id, "reason": "reference_repair_exception", "error": repr(exc), "gold": gold})

        save_jsonl(accepted, args.out_dir / "gold_specs.jsonl")
        save_jsonl(rejected, args.out_dir / "rejected_gold_specs.jsonl")
        save_jsonl(audit, args.out_dir / "audit_references.jsonl")

    warning_counts = Counter(w for row in audit for w in row.get("final_warnings", []))
    initial_warning_counts = Counter(w for row in audit for w in row.get("initial_warnings", []))
    summary = {
        "input_gold_specs": len(rows),
        "accepted_gold_specs": len(accepted),
        "rejected_gold_specs": len(rejected),
        "initial_warning_distribution": dict(initial_warning_counts),
        "final_warning_distribution": dict(warning_counts),
        "rejection_reasons": dict(Counter(r.get("reason") for r in rejected)),
        "recommended_next_step_input": str(args.out_dir / "gold_specs.jsonl"),
    }
    save_json(summary, args.out_dir / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()
