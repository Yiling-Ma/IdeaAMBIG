from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from common import LLMJsonClient, add_common_args, ensure_dir, load_jsonl, save_json, save_jsonl


def compact_instance(inst: Dict[str, Any]) -> Dict[str, Any]:
    defect = (inst.get("defects") or [{}])[0]
    return {
        "id": inst.get("id"),
        "project_id": inst.get("construction_metadata", {}).get("project_id"),
        "underspecified_spec": str(inst.get("input", {}).get("underspecified_spec", ""))[:6000],
        "gold_clarification": inst.get("gold", {}).get("gold_clarification", {}),
        "defect": defect,
        "quality_warnings": inst.get("quality_warnings", []),
        "candidate_strength": inst.get("construction_metadata", {}).get("candidate_strength"),
    }


def llm_audit_decision(
    inst: Dict[str, Any],
    client: LLMJsonClient,
    out_dir: Path,
) -> Dict[str, Any]:
    system = (
        "You are an IDEAAMBIG benchmark auditor. "
        "Decide keep/revise/drop conservatively. Return strict JSON only."
    )
    user_obj = {
        "task": (
            "Audit this single synthetic instance and decide one label: keep, revise, or drop. "
            "Use strict benchmark criteria."
        ),
        "decision_criteria": {
            "single_defect": "The instance must contain exactly one implementation-relevant hidden defect.",
            "no_leakage": "Input should not reveal target detail or benchmark-construction cues.",
            "code_ready_relevance": "Defect should materially block or alter implementation/evaluation decisions.",
            "gold_recoverability": "A clear clarification question could recover the hidden detail from gold evidence.",
        },
        "decision_policy": [
            "keep: criteria are satisfied; minor style issues only.",
            "revise: potentially usable but needs concrete rewrite/fix.",
            "drop: fundamental leakage, multi-defect, or weak/non-recoverable target.",
        ],
        "required_output_schema": {
            "decision": "keep|revise|drop",
            "confidence": "low|medium|high",
            "reasons": ["short strings"],
            "fix_suggestions": ["short strings"],
        },
        "instance": compact_instance(inst),
    }
    raw_path = out_dir / "raw_llm" / f"{inst.get('id', 'unknown')}.txt"
    obj = client.call_json(system, json.dumps(user_obj, ensure_ascii=False, indent=2), raw_path)
    if not isinstance(obj, dict):
        raise RuntimeError("LLM audit output is not a JSON object")
    decision = str(obj.get("decision", "")).strip().lower()
    if decision not in {"keep", "revise", "drop"}:
        raise RuntimeError(f"Invalid audit decision: {decision}")
    return {
        "id": inst.get("id"),
        "project_id": inst.get("construction_metadata", {}).get("project_id"),
        "decision": decision,
        "confidence": str(obj.get("confidence", "medium")).strip().lower(),
        "reasons": [str(x) for x in (obj.get("reasons", []) or [])][:12],
        "fix_suggestions": [str(x) for x in (obj.get("fix_suggestions", []) or [])][:12],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--model", default="deepseek/deepseek-v4-pro")
    parser.add_argument("--use_llm_judge", action="store_true")
    args = parser.parse_args()

    args.input_path = Path(args.input_path)
    args.out_dir = Path(args.out_dir)
    ensure_dir(args.out_dir)
    ensure_dir(args.out_dir / "raw_llm")

    rows = load_jsonl(args.input_path)
    if args.limit:
        rows = rows[: args.limit]

    client = LLMJsonClient(args.model, args.use_llm_judge)
    audits: List[Dict[str, Any]] = []
    keep_rows: List[Dict[str, Any]] = []
    revise_rows: List[Dict[str, Any]] = []
    drop_rows: List[Dict[str, Any]] = []

    for inst in tqdm(rows, desc="Step6 LLM audit"):
        try:
            audit = llm_audit_decision(inst, client, args.out_dir)
        except Exception as exc:
            audit = {
                "id": inst.get("id"),
                "project_id": inst.get("construction_metadata", {}).get("project_id"),
                "decision": "drop",
                "confidence": "low",
                "reasons": [f"audit_failed: {repr(exc)}"],
                "fix_suggestions": [],
            }

        audits.append(audit)
        merged = dict(inst)
        merged["llm_audit"] = audit
        if audit["decision"] == "keep":
            keep_rows.append(merged)
        elif audit["decision"] == "revise":
            revise_rows.append(merged)
        else:
            drop_rows.append(merged)

    save_jsonl(audits, args.out_dir / "llm_audit.jsonl")
    save_jsonl(keep_rows, args.out_dir / "keep_instances.jsonl")
    save_jsonl(revise_rows, args.out_dir / "revise_instances.jsonl")
    save_jsonl(drop_rows, args.out_dir / "drop_instances.jsonl")

    summary = {
        "input_instances": len(rows),
        "keep": len(keep_rows),
        "revise": len(revise_rows),
        "drop": len(drop_rows),
        "decision_distribution": dict(Counter(a["decision"] for a in audits)),
        "confidence_distribution": dict(Counter(a["confidence"] for a in audits)),
    }
    save_json(summary, args.out_dir / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()

