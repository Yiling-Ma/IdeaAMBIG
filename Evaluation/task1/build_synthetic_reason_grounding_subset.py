#!/usr/bin/env python3
"""Build a balanced synthetic Task 1 subset with reason-grounding gold."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT_DIR / "Syn_bench" / "syn_benchmark_instance.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ready-count", type=int, default=60)
    parser.add_argument("--not-ready-count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default="synthetic120")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def short_hash(text: str, length: int = 20) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def normalized_text(text: str) -> str:
    return " ".join(str(text).split())


def simplified_defects(row: dict[str, Any]) -> list[dict[str, Any]]:
    defects = row.get("defects") if isinstance(row.get("defects"), list) else []
    return [
        {
            "level1": defect.get("level1"),
            "level2": defect.get("level2"),
            "codification_slot": defect.get("codification_slot") or defect.get("slot"),
            "resolution_role": defect.get("resolution_role"),
        }
        for defect in defects
        if isinstance(defect, dict)
    ]


def synthesize_ready_reason(source_row: dict[str, Any]) -> str:
    reference = (
        ((source_row.get("eval_targets") or {}).get("reference") or {})
        if isinstance(source_row.get("eval_targets"), dict)
        else {}
    )
    if reference.get("expected_ready") is True:
        return (
            "The codification-ready reference resolves the original blocker and leaves "
            "no implementation-critical missing specification."
        )
    return (
        "The ready version contains no gold implementation-critical blocker under the "
        "benchmark annotation."
    )


def reason_grounding_gold(row: dict[str, Any], variant: str) -> dict[str, Any]:
    codification = row.get("codification_readiness") or {}
    if variant == "incomplete":
        return {
            "gold_label": "NOT_READY",
            "defects": row.get("defects") or [],
            "blocking_missing_specs": codification.get("blocking_missing_specs") or [],
            "readiness_reason": codification.get("reason") or "",
            "expected_clarification_actions": row.get("expected_clarification_actions") or [],
        }
    return {
        "gold_label": "READY",
        "defects": [],
        "blocking_missing_specs": [],
        "readiness_reason": synthesize_ready_reason(row),
        "expected_clarification_actions": [],
        "resolved_from_source_blockers": codification.get("blocking_missing_specs") or [],
        "source_not_ready_reason": codification.get("reason") or "",
    }


def source_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_types = sorted(
        {str(row.get("source", {}).get("source_type")) for row in rows if row.get("source", {}).get("source_type")}
    )
    paper_ids = sorted(
        {str(row.get("source", {}).get("paper_id")) for row in rows if row.get("source", {}).get("paper_id")}
    )
    construction_methods = sorted(
        {
            str(row.get("construction_metadata", {}).get("construction_method"))
            for row in rows
            if row.get("construction_metadata", {}).get("construction_method")
        }
    )
    return {
        "source_types": source_types,
        "construction_methods": construction_methods,
        "paper_ids": paper_ids,
    }


def build_rows(
    rows: list[dict[str, Any]],
    ready_count: int,
    not_ready_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_id = str((row.get("source") or {}).get("split_group_id") or row["id"])
        groups[group_id].append(row)

    ready_groups = sorted(groups)
    if len(ready_groups) < ready_count:
        raise ValueError(f"Need {ready_count} ready groups but found only {len(ready_groups)}")
    rng.shuffle(ready_groups)
    selected_ready_groups = ready_groups[:ready_count]

    not_ready_candidates = [row for row in rows if not bool((row.get("codification_readiness") or {}).get("is_ready"))]
    if len(not_ready_candidates) < not_ready_count:
        raise ValueError(
            f"Need {not_ready_count} not-ready instances but found only {len(not_ready_candidates)}"
        )
    rng.shuffle(not_ready_candidates)
    selected_not_ready = not_ready_candidates[:not_ready_count]

    cases: list[dict[str, Any]] = []
    case_index = 1

    for group_id in selected_ready_groups:
        group_rows = groups[group_id]
        anchor = group_rows[0]
        reference_text = str(((anchor.get("gold") or {}).get("codification_ready_reference")) or "").strip()
        if not reference_text:
            raise ValueError(f"Missing gold.codification_ready_reference for group {group_id}")
        pair_group_id = f"spec_{short_hash(normalized_text(reference_text))}"
        case_id = f"task1_syn_{case_index:06d}"
        case_index += 1
        cases.append(
            {
                "pool": {"case_id": case_id, "input_text": reference_text},
                "gold": {
                    "case_id": case_id,
                    "expected_ready": True,
                    "variant": "complete",
                    "instance_id": str(anchor["id"]),
                    "source_instance_ids": sorted(str(row["id"]) for row in group_rows),
                    "pair_group_id": pair_group_id,
                    "source": source_summary(group_rows),
                    "defects": [],
                    "reason_grounding_gold": reason_grounding_gold(anchor, "complete"),
                },
            }
        )

    for row in selected_not_ready:
        input_text = str(((row.get("input") or {}).get("underspecified_spec")) or "").strip()
        if not input_text:
            raise ValueError(f"Missing input.underspecified_spec for {row['id']}")
        reference_text = str(((row.get("gold") or {}).get("codification_ready_reference")) or "").strip()
        pair_group_id = f"spec_{short_hash(normalized_text(reference_text or str(row['id'])))}"
        case_id = f"task1_syn_{case_index:06d}"
        case_index += 1
        case_source = {
            "source_type": (row.get("source") or {}).get("source_type"),
            "construction_method": (row.get("construction_metadata") or {}).get("construction_method"),
            "paper_id": (row.get("source") or {}).get("paper_id"),
        }
        cases.append(
            {
                "pool": {"case_id": case_id, "input_text": input_text},
                "gold": {
                    "case_id": case_id,
                    "expected_ready": False,
                    "variant": "incomplete",
                    "instance_id": str(row["id"]),
                    "source_instance_ids": [str(row["id"])],
                    "pair_group_id": pair_group_id,
                    "source": case_source,
                    "defects": simplified_defects(row),
                    "reason_grounding_gold": reason_grounding_gold(row, "incomplete"),
                },
            }
        )

    rng.shuffle(cases)
    manifest = {
        "seed": seed,
        "ready_count": ready_count,
        "not_ready_count": not_ready_count,
        "total_cases": len(cases),
        "source_file": str(DEFAULT_INPUT),
        "selected_ready_groups": selected_ready_groups,
        "selected_not_ready_instance_ids": [str(row["id"]) for row in selected_not_ready],
    }
    return [case["pool"] for case in cases], [case["gold"] for case in cases], manifest


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    pool_path = output_dir / f"{args.prefix}_evaluation_pool.jsonl"
    gold_path = output_dir / f"{args.prefix}_evaluation_gold_reason_grounding.jsonl"
    manifest_path = output_dir / f"{args.prefix}_evaluation_manifest.json"

    if not args.overwrite:
        for path in (pool_path, gold_path, manifest_path):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing file: {path}")

    rows = load_jsonl(args.input.resolve())
    pool_rows, gold_rows, manifest = build_rows(
        rows,
        ready_count=args.ready_count,
        not_ready_count=args.not_ready_count,
        seed=args.seed,
    )
    write_jsonl(pool_path, pool_rows)
    write_jsonl(gold_path, gold_rows)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "pool": str(pool_path),
                "gold": str(gold_path),
                "manifest": str(manifest_path),
                "ready_count": args.ready_count,
                "not_ready_count": args.not_ready_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
