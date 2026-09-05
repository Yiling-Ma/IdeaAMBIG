from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from common import (
    LEVEL2_TO_LEVEL1,
    LLMJsonClient,
    add_common_args,
    ensure_dir,
    load_jsonl,
    normalize_ws,
    save_json,
    save_jsonl,
)


VALID_OPERATIONS = {
    "omit_detail",
    "abstract_detail",
    "make_ambiguous",
    "introduce_conflict",
}

VALID_GRANULARITY = {"coarse", "medium", "fine"}


def compact(value: Any, max_chars: int) -> str:
    text = normalize_ws(value)
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "..."


def load_manifest_by_project(path: Path | None) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    return {row.get("project_id"): row for row in load_jsonl(path) if row.get("project_id")}


def compact_gold_spec(gold: Dict[str, Any]) -> Dict[str, Any]:
    pds = gold.get("paper_derived_specification") or {}
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
        value = pds.get(key)
        if isinstance(value, list):
            out[key] = [compact(x, 700) for x in value[:12] if normalize_ws(x)]
        else:
            text = compact(value, 1600)
            if text:
                out[key] = text
    return out


def build_prompt(
    gold: Dict[str, Any],
    manifest: Dict[str, Any],
    max_instances: int,
) -> Tuple[str, str]:
    system = """
You are generating synthetic-controlled IDEAAMBIG benchmark instances.

Return strict JSON only. Do not include markdown fences.
Use only details supported by the provided codification-ready reference, structured gold specification, executed paper evidence, or codebase evidence.
Do not invent unsupported details.
""".strip()

    user_obj = {
        "task": (
            f"Create up to {max_instances} realistic underspecified research specifications by removing, "
            "abstracting, or lightly corrupting exactly one implementation-critical detail from the codification-ready reference."
        ),
        "benchmark_goal": (
            "IDEAAMBIG evaluates whether models can detect research-specification-to-implementation handoff gaps "
            "before codification."
        ),
        "source_interpretation": [
            "The executed paper and codebase are gold evidence.",
            "The original or edited idea may be underspecified and should not be treated as gold by itself.",
            "Do not create paper-code discrepancies. Create research specification codification gaps.",
        ],
        "critical_rules": [
            "Generate at most the requested number of independent instances.",
            "Each instance must contain exactly one defect.",
            "The defect must be created by deleting, abstracting, making ambiguous, or lightly conflicting exactly one implementation-critical detail.",
            "Preserve all non-target details.",
            "Do not introduce a second defect.",
            "Do not reveal the removed or corrupted gold detail in underspecified_spec.",
            "Do not use benchmark/meta diagnostic language that explicitly says the specification is missing, unspecified, unclear, defective, or not codification-ready.",
            "Normal method phrases such as 'benchmark dataset', 'missing tokens', or 'missing information' inside algorithm steps are allowed.",
            "underspecified_spec must read like a natural research or method specification, not an annotation.",
            "underspecified_spec must be between 100 and 200 words.",
            "Do not choose weak details such as ordinary batch size, learning rate, epoch count, random seed, hardware, runtime, API key, or local path unless the value is part of a non-standard method mechanism.",
            "Prefer details whose absence blocks faithful implementation: algorithm step, model architecture/module, loss or training procedure, data/preprocessing/input construction, evaluation protocol/metric computation, inference/decoding/prompt construction.",
        ],
        "taxonomy": {
            "allowed_level2_to_level1": LEVEL2_TO_LEVEL1,
            "disallowed_labels": [
                "missing objective/loss specification",
                "missing objective specification",
                "missing loss specification",
            ],
            "label_guidance": [
                "If a loss identity, loss routing, objective detail, training logic, or inference-time decision logic is removed, usually use Missing Method Procedure.",
                "If paper/code conflict on the loss or objective, use Conflicting Objective.",
                "If prompt construction, parsing, routing, stage order, formula, aggregation, or decoding logic is removed, use Missing Method Procedure.",
                "If layer type, embedding mapping, pooling, normalization, module structure, or dimensional connection is removed, use Missing Model Structure.",
                "If dataset construction, split, sampling, filtering, labeling, tokenization, preprocessing, or input construction is removed, use Missing Data Specification.",
                "If metric computation, baseline setup, thresholding, evaluation split, or judge/model configuration is removed, use Missing Evaluation Specification.",
            ],
        },
        "schema_alignment": {
            "granularity_allowed_values": ["coarse", "medium", "fine"],
            "granularity_guidance": [
                "coarse: an entire method component is missing or undefined (the codification slot itself is essentially unaddressed), so multiple downstream implementation steps are underdetermined at once.",
                "medium: one specific operation, step, or parameter is missing or wrong within an otherwise well-specified component; the surrounding module is clear. Example: a graph-attention score is fully specified except that a LeakyReLU nonlinearity before softmax is omitted.",
                "fine: a local wording ambiguity with a small, enumerable set of plausible readings, with no operation missing. Example: '8x8 cell grid' could mean 8x8-pixel cells or an 8x8 grid spanning the full image.",
                "Vague quantifiers (e.g., 'multiple', 'a threshold') whose gold value is a concrete number or setting are medium, unless the concrete value changes the qualitative behavior of the method, in which case they are coarse.",
            ],
            "defect_operation_allowed_values": [
                "omit_detail",
                "abstract_detail",
                "make_ambiguous",
                "introduce_conflict",
            ],
            "operation_guidance": [
                "Use omit_detail when the concrete detail is removed.",
                "Use abstract_detail when the concrete detail is replaced by a vague high-level phrase.",
                "Use make_ambiguous when a component remains mentioned but its operational behavior becomes unclear.",
                "Use introduce_conflict only when the synthetic instance intentionally creates a contradiction. Avoid this for most instances.",
            ],
        },
        "output_requirements": [
            "gold_detail_removed_or_corrupted must be a concrete detail traceable to the reference or structured gold specification.",
            "surface_form_in_underspecified_spec must be the exact vague/replacement phrase or sentence in underspecified_spec.",
            "underspecified_spec length must be 100-200 words.",
            "must_not_reveal must contain ONLY (1) the full removed/corrupted gold detail and (2) distinctive answer-leaking phrases directly derived from that detail.",
            "Do NOT list generic context words such as 'JSON object', 'benchmark', dataset category names, or broad format labels unless they uniquely reveal the removed detail.",
            "evidence.quote must quote or paraphrase the supporting gold detail tightly enough for deterministic grounding checks.",
            "preservation_check is required, but Step 3b will verify it independently.",
        ],
        "required_output_schema": {
            "instances": [
                {
                    "project_id": "string",
                    "target_slot": "string",
                    "source_field": "paper_derived_specification field or gold_reference",
                    "level1": "Ambiguity|Incompleteness|Inconsistency",
                    "level2": "one allowed Level-2 label",
                    "granularity": "coarse|medium|fine",
                    "resolution_role": "implementation_blocker",
                    "codification_slot": "TASK_AND_IO|CORE_ALGORITHM|MODEL_ARCHITECTURE|OBJECTIVE_AND_SUPERVISION|TRAINING_PROCEDURE|DATA_AND_PREPROCESSING|INFERENCE_AND_DECISION|EVALUATION_PROTOCOL|INTERNAL_CONSISTENCY|NONE",
                    "defect_operation": "omit_detail|abstract_detail|make_ambiguous|introduce_conflict",
                    "gold_detail_removed_or_corrupted": "string",
                    "underspecified_spec": "string",
                    "surface_form_in_underspecified_spec": "string",
                    "why_this_blocks_or_affects_codification": "string",
                    "expected_clarification_question": "string",
                    "must_not_reveal": ["string"],
                    "evidence": [{"source": "gold_reference|paper_derived_specification|executed_paper|codebase", "quote": "string"}],
                    "preservation_check": {
                        "target_detail_modified": True,
                        "non_target_details_preserved": True,
                        "extra_defects_introduced": False,
                        "brief_explanation": "string",
                    },
                    "quality_rationale": "string",
                }
            ],
            "rejected_candidates": [{"gold_detail": "string", "reason": "string"}],
        },
        "inputs": {
            "project_id": gold.get("project_id"),
            "edited_idea_secondary_context": compact(manifest.get("edited_idea", ""), 6000),
            "executed_paper_evidence_excerpt": compact(manifest.get("executed_paper_text", ""), 16000),
            "codebase_evidence_excerpt": compact(manifest.get("codebase_summary_text", ""), 12000),
            "codification_ready_reference": gold.get("codification_ready_reference", ""),
            "gold_structured_specification": compact_gold_spec(gold),
        },
    }
    return system, json.dumps(user_obj, ensure_ascii=False, indent=2)


def parse_llm_instances(obj: Any) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)], []
    if not isinstance(obj, dict):
        return [], [{"reason": "llm_output_not_object"}]
    instances = obj.get("instances", [])
    rejected = obj.get("rejected_candidates", [])
    if not isinstance(instances, list):
        instances = []
    if not isinstance(rejected, list):
        rejected = []
    return [x for x in instances if isinstance(x, dict)], [x for x in rejected if isinstance(x, dict)]


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--manifest_path", type=Path, default=None)
    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--model", default="deepseek/deepseek-v4-pro")
    parser.add_argument("--max_instances_per_project", type=int, default=5)
    parser.add_argument("--save_prompt", action="store_true")
    args = parser.parse_args()

    if not args.use_llm:
        raise SystemExit("Step 3a requires --use_llm because it directly writes underspecified specs.")

    ensure_dir(args.out_dir)
    ensure_dir(args.out_dir / "raw_llm")
    ensure_dir(args.out_dir / "per_project")

    gold_specs = load_jsonl(args.input_path)
    if args.limit:
        gold_specs = gold_specs[: args.limit]
    manifest_by_project = load_manifest_by_project(args.manifest_path)
    client = LLMJsonClient(args.model, True)

    raw_rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []
    index_rows: List[Dict[str, Any]] = []

    for gold in tqdm(gold_specs, desc="Step3a generate raw instances"):
        project_id = gold.get("project_id", "unknown_project")
        project_dir = args.out_dir / "per_project" / project_id
        ensure_dir(project_dir)
        out_path = project_dir / "raw_instances.json"

        if args.resume and out_path.exists() and out_path.stat().st_size > 50:
            obj = json.loads(out_path.read_text(encoding="utf-8"))
            instances = obj.get("instances", [])
            rejected = obj.get("rejected_candidates", [])
            raw_rows.extend(instances)
            rejected_rows.extend(rejected)
            index_rows.append(
                {
                    "project_id": project_id,
                    "status": "skipped_exists",
                    "raw_instances": len(instances),
                    "rejected_candidates": len(rejected),
                }
            )
            continue

        manifest = manifest_by_project.get(project_id, {})
        system, user = build_prompt(gold, manifest, args.max_instances_per_project)
        if args.save_prompt:
            save_json({"system": system, "user": json.loads(user)}, project_dir / "prompt.json")

        try:
            obj = client.call_json(system, user, args.out_dir / "raw_llm" / f"{project_id}.txt")
            instances, rejected = parse_llm_instances(obj)
            for i, instance in enumerate(instances, start=1):
                instance["project_id"] = instance.get("project_id") or project_id
                instance["record_id"] = project_id
                instance["raw_instance_id"] = f"{project_id}_raw_{i:03d}"
            for item in rejected:
                item["project_id"] = project_id
            save_json(
                {
                    "project_id": project_id,
                    "instances": instances,
                    "rejected_candidates": rejected,
                },
                out_path,
            )
            raw_rows.extend(instances)
            rejected_rows.extend(rejected)
            index_rows.append(
                {
                    "project_id": project_id,
                    "status": "saved",
                    "raw_instances": len(instances),
                    "rejected_candidates": len(rejected),
                }
            )
        except Exception as exc:
            row = {
                "project_id": project_id,
                "reason": "llm_generation_failed",
                "error": repr(exc),
            }
            rejected_rows.append(row)
            index_rows.append(
                {
                    "project_id": project_id,
                    "status": "failed",
                    "raw_instances": 0,
                    "rejected_candidates": 1,
                    "error": repr(exc),
                }
            )

        save_jsonl(raw_rows, args.out_dir / "raw_instances.jsonl")
        save_jsonl(rejected_rows, args.out_dir / "rejected_raw_outputs.jsonl")
        save_jsonl(index_rows, args.out_dir / "index.jsonl")

    save_jsonl(raw_rows, args.out_dir / "raw_instances.jsonl")
    save_jsonl(rejected_rows, args.out_dir / "rejected_raw_outputs.jsonl")
    save_jsonl(index_rows, args.out_dir / "index.jsonl")

    summary = {
        "input_gold_specs": len(gold_specs),
        "raw_instances": len(raw_rows),
        "rejected_raw_outputs": len(rejected_rows),
        "instances_per_project": dict(Counter(r.get("project_id") for r in raw_rows)),
        "level2_distribution": dict(Counter(r.get("level2", "unknown") for r in raw_rows)),
        "operation_distribution": dict(Counter(r.get("defect_operation", "unknown") for r in raw_rows)),
        "granularity_distribution": dict(Counter(r.get("granularity", "unknown") for r in raw_rows)),
    }
    save_json(summary, args.out_dir / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()
