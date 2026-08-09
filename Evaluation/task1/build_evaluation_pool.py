#!/usr/bin/env python3
"""Build the flattened, shuffled zero-shot evaluation pool for Task 1.

The public pool contains only opaque case IDs and specification text. Labels,
source information, taxonomy annotations, and complete/incomplete provenance are
written to a separate private gold file so they cannot leak into model prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "instances.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No instances found in {path}")
    return rows


def nested_get(row: dict[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def normalized_text(text: str) -> str:
    """Normalize text only for hashing/deduplication, not for model input."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def short_hash(text: str, length: int = 20) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def permutation_key(seed: int, internal_id: str) -> str:
    """Cross-run deterministic pseudo-random ordering independent of input order."""
    return hashlib.sha256(f"{seed}\0{internal_id}".encode("utf-8")).hexdigest()


def instance_id(row: dict[str, Any], line_index: int) -> str:
    value = row.get("id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Instance at logical row {line_index} has no non-empty string id")
    return value


def source_summary(row: dict[str, Any]) -> dict[str, Any]:
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    construction = (
        row.get("construction_metadata")
        if isinstance(row.get("construction_metadata"), dict)
        else {}
    )
    return {
        "source_type": source.get("source_type"),
        "construction_method": construction.get("construction_method"),
        "paper_id": construction.get("paper_id") or source.get("paper_id"),
    }


def defect_summary(row: dict[str, Any]) -> list[dict[str, Any]]:
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


def validate_expected_label(
    row: dict[str, Any], variant: str, expected_value: bool, row_id: str
) -> None:
    target_name = "underspecified" if variant == "incomplete" else "reference"
    value = nested_get(row, "eval_targets", target_name, "expected_ready")
    if value is not None and value is not expected_value:
        raise ValueError(
            f"{row_id}: eval_targets.{target_name}.expected_ready={value!r}; "
            f"expected {expected_value!r}"
        )


def build_candidates(
    rows: list[dict[str, Any]], complete_mode: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates: list[dict[str, Any]] = []
    complete_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()

    for index, row in enumerate(rows, start=1):
        row_id = instance_id(row, index)
        if row_id in seen_ids:
            raise ValueError(f"Duplicate instance id: {row_id}")
        seen_ids.add(row_id)

        incomplete_text = nested_get(row, "input", "underspecified_spec")
        complete_text = nested_get(row, "gold", "codification_ready_reference")
        if not isinstance(incomplete_text, str) or not incomplete_text.strip():
            raise ValueError(f"{row_id}: missing input.underspecified_spec")
        if not isinstance(complete_text, str) or not complete_text.strip():
            raise ValueError(f"{row_id}: missing gold.codification_ready_reference")

        validate_expected_label(row, "incomplete", False, row_id)
        validate_expected_label(row, "complete", True, row_id)
        normalized_reference = normalized_text(complete_text)
        pair_group_id = f"spec_{short_hash(normalized_reference)}"
        source = source_summary(row)

        candidates.append(
            {
                "internal_id": f"incomplete\0{row_id}",
                "input_text": incomplete_text,
                "gold": {
                    "expected_ready": False,
                    "variant": "incomplete",
                    "instance_id": row_id,
                    "source_instance_ids": [row_id],
                    "pair_group_id": pair_group_id,
                    "source": source,
                    "defects": defect_summary(row),
                },
            }
        )

        complete_groups[normalized_reference].append(
            {
                "instance_id": row_id,
                "input_text": complete_text,
                "pair_group_id": pair_group_id,
                "source": source,
            }
        )

    if complete_mode == "per-instance":
        for group in complete_groups.values():
            for item in group:
                row_id = item["instance_id"]
                candidates.append(
                    {
                        "internal_id": f"complete\0{row_id}",
                        "input_text": item["input_text"],
                        "gold": {
                            "expected_ready": True,
                            "variant": "complete",
                            "instance_id": row_id,
                            "source_instance_ids": [row_id],
                            "pair_group_id": item["pair_group_id"],
                            "source": item["source"],
                            "defects": [],
                        },
                    }
                )
    else:
        for normalized_reference, group in complete_groups.items():
            source_instance_ids = sorted(item["instance_id"] for item in group)
            source_types = sorted(
                {
                    item["source"].get("source_type")
                    for item in group
                    if item["source"].get("source_type")
                }
            )
            construction_methods = sorted(
                {
                    item["source"].get("construction_method")
                    for item in group
                    if item["source"].get("construction_method")
                }
            )
            paper_ids = sorted(
                {
                    item["source"].get("paper_id")
                    for item in group
                    if item["source"].get("paper_id")
                }
            )
            candidates.append(
                {
                    "internal_id": f"complete\0{short_hash(normalized_reference, 64)}",
                    "input_text": group[0]["input_text"],
                    "gold": {
                        "expected_ready": True,
                        "variant": "complete",
                        "instance_id": source_instance_ids[0],
                        "source_instance_ids": source_instance_ids,
                        "pair_group_id": group[0]["pair_group_id"],
                        "source": {
                            "source_types": source_types,
                            "construction_methods": construction_methods,
                            "paper_ids": paper_ids,
                        },
                        "defects": [],
                    },
                }
            )

    counts = {
        "input_instances": len(rows),
        "incomplete_candidates": len(rows),
        "complete_candidates_before_deduplication": len(rows),
        "unique_complete_references": len(complete_groups),
        "duplicate_complete_references_removed": (
            len(rows) - len(complete_groups) if complete_mode == "unique" else 0
        ),
    }
    return candidates, counts


def atomic_write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a flattened, deterministically shuffled Task 1 evaluation pool."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--complete-mode",
        choices=("unique", "per-instance"),
        default="unique",
        help=(
            "unique emits each distinct complete reference once (recommended); "
            "per-instance emits one complete case for every input instance"
        ),
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace existing generated files."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = output_dir / "evaluation_pool.jsonl"
    gold_path = output_dir / "evaluation_gold.jsonl"
    manifest_path = output_dir / "evaluation_manifest.json"

    existing = [path for path in (pool_path, gold_path, manifest_path) if path.exists()]
    if existing and not args.force:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output exists; pass --force to replace: {joined}")

    input_path = args.input.resolve()
    rows = load_jsonl(input_path)
    candidates, source_counts = build_candidates(rows, args.complete_mode)
    candidates.sort(key=lambda item: permutation_key(args.seed, item["internal_id"]))

    public_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    for position, candidate in enumerate(candidates, start=1):
        case_id = f"task1_{position:06d}"
        public_rows.append({"case_id": case_id, "input_text": candidate["input_text"]})
        gold_rows.append({"case_id": case_id, **candidate["gold"]})

    atomic_write_jsonl(public_rows, pool_path)
    atomic_write_jsonl(gold_rows, gold_path)

    label_counts = Counter(row["expected_ready"] for row in gold_rows)
    variant_counts = Counter(row["variant"] for row in gold_rows)
    manifest = {
        "benchmark": "IdeaAmbig",
        "task": "Task 1: Readiness Assessment",
        "setting": "zero-shot; one specification per sample; no train/dev/test split",
        "input_path": str(input_path),
        "seed": args.seed,
        "shuffle_method": "SHA-256(seed, internal_case_id) deterministic permutation",
        "complete_mode": args.complete_mode,
        **source_counts,
        "evaluation_cases": len(public_rows),
        "variant_counts": dict(sorted(variant_counts.items())),
        "label_counts": {
            "ready": label_counts[True],
            "not_ready": label_counts[False],
        },
        "public_pool_fields": ["case_id", "input_text"],
        "private_gold_file": gold_path.name,
        "notes": [
            "Complete and incomplete specifications are never jointly presented.",
            "Variant labels and provenance are absent from the public evaluation pool.",
            "pair_group_id is private metadata for post-hoc matched analysis only.",
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
