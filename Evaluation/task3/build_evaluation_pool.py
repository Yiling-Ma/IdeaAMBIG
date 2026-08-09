#!/usr/bin/env python3
"""Build public inputs and private gold for isolated Task 3 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from task3_config import DEFAULT_INSTANCES, TASK3_DIR, load_jsonl, source_category


def nested_dict(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key)
    return value if isinstance(value, dict) else {}


def nonempty(value: Any) -> str:
    return str(value or "").strip()


def permutation_key(seed: int, instance_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{instance_id}".encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def target_defect_description(instance: dict[str, Any], defect: dict[str, Any]) -> str:
    """Create a gold diagnosis without exposing the hidden resolution.

    Real instances generally have an adjudicated gap summary. Synthetic instances
    often store a concise target phrase in blocking_missing_specs[0].slot; this is
    an instance-specific description, not a codification-slot ontology label.
    """
    hidden_resolution = nonempty(defect.get("gold_detail_removed_or_corrupted")) or nonempty(
        nested_dict(instance, "construction_metadata").get("gold_clarified_detail")
    )
    construction = nested_dict(instance, "construction_metadata")
    forbidden_phrases: list[str] = []
    perturbation_instruction = construction.get("perturbation_instruction")
    if isinstance(perturbation_instruction, dict):
        values = perturbation_instruction.get("must_not_reveal")
        if isinstance(values, list):
            forbidden_phrases.extend(nonempty(value) for value in values if nonempty(value))
    perturbations = construction.get("selected_perturbations")
    if isinstance(perturbations, list) and perturbations and isinstance(perturbations[0], dict):
        values = perturbations[0].get("must_not_reveal")
        if isinstance(values, list):
            forbidden_phrases.extend(nonempty(value) for value in values if nonempty(value))

    def normalized(value: str) -> str:
        return re.sub(r"\W+", " ", value.casefold()).strip()

    def is_non_leaking(candidate: str) -> bool:
        candidate_norm = normalized(candidate)
        hidden_norm = normalized(hidden_resolution)
        if not candidate_norm or not hidden_norm:
            return bool(candidate_norm)
        if hidden_norm in candidate_norm:
            return False
        if candidate_norm in hidden_norm and len(candidate_norm.split()) >= 2:
            return False
        return not any(
            normalized(phrase) and normalized(phrase) in candidate_norm
            for phrase in forbidden_phrases
        )

    gap_summary = nonempty(construction.get("gap_summary"))
    if gap_summary and is_non_leaking(gap_summary):
        return gap_summary

    readiness = nested_dict(instance, "codification_readiness")
    blocking = readiness.get("blocking_missing_specs")
    if isinstance(blocking, list):
        for item in blocking:
            if isinstance(item, dict):
                target = nonempty(item.get("slot"))
                if target and is_non_leaking(target):
                    level1 = nonempty(defect.get("level1"))
                    if level1 == "Ambiguity":
                        return f"The specification leaves {target} ambiguous."
                    if level1 == "Inconsistency":
                        return f"The specification is internally inconsistent about {target}."
                    return f"The specification does not specify {target}."
            else:
                text = nonempty(item)
                if text and is_non_leaking(text):
                    return text

    surface = nonempty(defect.get("surface_form_in_underspecified_spec"))
    if surface:
        level1 = nonempty(defect.get("level1")).lower()
        return (
            f"The implementation-critical detail associated with the following passage is "
            f"{level1 or 'underspecified'}: {surface}"
        )
    raise ValueError(f"{instance.get('id')}: cannot derive a non-leaking target defect")


def build_case(instance: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    instance_id = nonempty(instance.get("id"))
    if not instance_id:
        raise ValueError("Instance has no non-empty id")
    defects = instance.get("defects")
    if not isinstance(defects, list) or len(defects) != 1 or not isinstance(defects[0], dict):
        raise ValueError(f"{instance_id}: Task 3 requires exactly one defect")
    actions = instance.get("expected_clarification_actions")
    if not isinstance(actions, list) or len(actions) != 1 or not isinstance(actions[0], dict):
        raise ValueError(f"{instance_id}: Task 3 requires exactly one gold action")

    defect = defects[0]
    action = actions[0]
    input_text = nonempty(nested_dict(instance, "input").get("underspecified_spec"))
    if not input_text:
        raise ValueError(f"{instance_id}: missing input.underspecified_spec")
    target_description = target_defect_description(instance, defect)
    hidden_resolution = nonempty(defect.get("gold_detail_removed_or_corrupted")) or nonempty(
        nested_dict(instance, "construction_metadata").get("gold_clarified_detail")
    )
    if not hidden_resolution:
        raise ValueError(f"{instance_id}: missing hidden gold resolution")

    action_type = nonempty(action.get("action_type"))
    if action_type not in {"clarification_question", "evidence_seeking"}:
        raise ValueError(f"{instance_id}: invalid gold action type {action_type!r}")
    expected_action = nonempty(action.get("question_or_action"))
    if not expected_action:
        raise ValueError(f"{instance_id}: missing gold question_or_action")

    construction = nested_dict(instance, "construction_metadata")
    source = nested_dict(instance, "source")
    source_info = {
        "source_type": source.get("source_type"),
        "construction_method": construction.get("construction_method"),
    }
    public = {
        "input_text": input_text,
        "target_defect": target_description,
    }
    private = {
        "instance_id": instance_id,
        "paper_id": construction.get("paper_id") or source.get("paper_id") or instance_id,
        "source": source_info,
        "source_category": source_category(source_info),
        "level1": nonempty(defect.get("level1")),
        "level2": nonempty(defect.get("level2")),
        "hidden_resolution": hidden_resolution,
        "surface_form": nonempty(defect.get("surface_form_in_underspecified_spec")),
        "blocking_effect": nonempty(defect.get("why_this_blocks_or_affects_codification")),
        "gold_action_type": action_type,
        "gold_action": expected_action,
        "gold_evidence_to_seek": nonempty(action.get("evidence_to_seek")),
        "solution_evidence": {
            "solution_summary": nonempty(construction.get("solution_summary")),
            "solution_quote": nonempty(construction.get("solution_quote")),
            "solution_source_type": nonempty(construction.get("solution_source_type")),
        },
    }
    return public, private


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument("--output-dir", type=Path, default=TASK3_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", choices=("all", "real", "synthetic"), default="all")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = output_dir / "evaluation_pool.jsonl"
    gold_path = output_dir / "evaluation_gold.jsonl"
    manifest_path = output_dir / "evaluation_manifest.json"
    existing = [p for p in (pool_path, gold_path, manifest_path) if p.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Output exists; pass --force to replace: " + ", ".join(str(p) for p in existing)
        )

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for instance in load_jsonl(args.input.resolve()):
        public, private = build_case(instance)
        instance_id = private["instance_id"]
        if instance_id in seen:
            raise ValueError(f"Duplicate instance id: {instance_id}")
        seen.add(instance_id)
        if args.subset != "all" and private["source_category"] != args.subset:
            continue
        candidates.append({"public": public, "private": private})

    candidates.sort(
        key=lambda row: permutation_key(args.seed, row["private"]["instance_id"])
    )
    public_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    for position, candidate in enumerate(candidates, start=1):
        case_id = f"task3_{position:06d}"
        public_rows.append({"case_id": case_id, **candidate["public"]})
        gold_rows.append({"case_id": case_id, **candidate["private"]})

    atomic_write_jsonl(public_rows, pool_path)
    atomic_write_jsonl(gold_rows, gold_path)
    manifest = {
        "benchmark": "IdeaAmbig",
        "task": "Task 3: Clarification Action Generation",
        "setting": "gold-defect-conditioned zero-shot; one atomic action per instance",
        "input_path": str(args.input.resolve()),
        "subset": args.subset,
        "seed": args.seed,
        "evaluation_cases": len(public_rows),
        "source_counts": dict(sorted(Counter(r["source_category"] for r in gold_rows).items())),
        "level1_counts": dict(sorted(Counter(r["level1"] for r in gold_rows).items())),
        "level2_counts": dict(sorted(Counter(r["level2"] for r in gold_rows).items())),
        "gold_action_type_counts": dict(
            sorted(Counter(r["gold_action_type"] for r in gold_rows).items())
        ),
        "public_pool_fields": ["case_id", "input_text", "target_defect"],
        "private_gold_file": gold_path.name,
        "notes": [
            "The public target_defect is the gold Task 2 diagnosis, not a model prediction.",
            "No codification_slot or taxonomy label is exposed to the evaluated model.",
            "The hidden resolution, reference action, solution evidence, taxonomy, and provenance are private.",
            "Gold action type is diagnostic only; a different action type may still receive full credit.",
        ],
        "files": {
            pool_path.name: {"sha256": file_sha256(pool_path)},
            gold_path.name: {"sha256": file_sha256(gold_path)},
        },
    }
    atomic_write_json(manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
