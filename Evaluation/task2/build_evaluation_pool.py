#!/usr/bin/env python3
"""Build the private-gold/public-input pool for Task 2 Track 1.

Track 1 is conditional atomic localization: every public input is an
underspecified specification with exactly one validated target defect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from track1_config import (
    DEFAULT_INSTANCES,
    LEVEL2_TO_LEVEL1,
    TRACK_DIR,
    load_jsonl,
    source_category,
)


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


def balanced_sample(
    candidates: list[dict[str, Any]],
    sample_size: int,
    stratify_by: str,
    seed: int,
) -> list[dict[str, Any]]:
    if sample_size <= 0:
        raise ValueError("--sample-size must be > 0")
    if sample_size > len(candidates):
        raise ValueError(
            f"--sample-size {sample_size} exceeds available candidates {len(candidates)}"
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        gold = candidate["gold"]
        if stratify_by == "level1":
            key = str(gold["level1"])
        elif stratify_by == "level2":
            key = str(gold["level2"])
        elif stratify_by == "source_category":
            key = str(gold["source_category"])
        else:
            raise ValueError(f"Unsupported stratification key: {stratify_by}")
        groups[key].append(candidate)

    if not groups:
        return []

    rng = random.Random(seed)
    for rows in groups.values():
        rng.shuffle(rows)

    labels = sorted(groups)
    target_counts = {label: 0 for label in labels}
    remaining = sample_size

    # Allocate samples as evenly as possible while respecting per-class capacity.
    active = labels[:]
    while remaining > 0 and active:
        base = remaining // len(active)
        extra = remaining % len(active)
        progressed = False
        next_active: list[str] = []
        for index, label in enumerate(active):
            capacity = len(groups[label]) - target_counts[label]
            desired = base + (1 if index < extra else 0)
            take = min(capacity, desired)
            if take > 0:
                target_counts[label] += take
                remaining -= take
                progressed = True
            if len(groups[label]) - target_counts[label] > 0:
                next_active.append(label)
        if not progressed:
            raise ValueError(
                "Unable to allocate balanced sample; not enough class capacity remains."
            )
        active = next_active

    sampled: list[dict[str, Any]] = []
    for label in labels:
        sampled.extend(groups[label][: target_counts[label]])

    rng.shuffle(sampled)
    return sampled


def atomic_write_json(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_gold(instance: dict[str, Any]) -> dict[str, Any]:
    instance_id = nonempty(instance.get("id"))
    if not instance_id:
        raise ValueError("Instance has no non-empty id")
    defects = instance.get("defects")
    if not isinstance(defects, list) or len(defects) != 1 or not isinstance(defects[0], dict):
        raise ValueError(f"{instance_id}: Track 1 requires exactly one defect")
    defect = defects[0]
    level1 = nonempty(defect.get("level1"))
    level2 = nonempty(defect.get("level2"))
    if level2 not in LEVEL2_TO_LEVEL1:
        raise ValueError(f"{instance_id}: invalid Level-2 label {level2!r}")
    if LEVEL2_TO_LEVEL1[level2] != level1:
        raise ValueError(
            f"{instance_id}: inconsistent taxonomy {level1!r}/{level2!r}; "
            f"expected {LEVEL2_TO_LEVEL1[level2]!r}"
        )

    construction = nested_dict(instance, "construction_metadata")
    source = nested_dict(instance, "source")
    readiness = nested_dict(instance, "codification_readiness")
    eval_targets = nested_dict(instance, "eval_targets")
    under_target = eval_targets.get("underspecified")
    if isinstance(under_target, dict) and under_target.get("expected_ready") is not False:
        raise ValueError(f"{instance_id}: underspecified target is not explicitly NOT_READY")

    blocking = readiness.get("blocking_missing_specs")
    blocking_items: list[str] = []
    if isinstance(blocking, list):
        for item in blocking:
            if isinstance(item, dict):
                text = nonempty(item.get("reason")) or nonempty(item.get("detail"))
            else:
                text = nonempty(item)
            if text:
                blocking_items.append(text)
    target = {
        "description": nonempty(construction.get("gap_summary"))
        or nonempty(defect.get("why_this_blocks_or_affects_codification"))
        or nonempty(readiness.get("reason")),
        "surface_form": nonempty(defect.get("surface_form_in_underspecified_spec")),
        "missing_or_corrupted_detail": nonempty(defect.get("gold_detail_removed_or_corrupted"))
        or nonempty(construction.get("gold_clarified_detail")),
        "blocking_effect": nonempty(defect.get("why_this_blocks_or_affects_codification")),
        "blocking_missing_specs": [item for item in blocking_items if item],
    }
    if not any(target.values()):
        raise ValueError(f"{instance_id}: no semantic gold information for defect matching")

    source_info = {
        "source_type": source.get("source_type"),
        "construction_method": construction.get("construction_method"),
    }
    return {
        "instance_id": instance_id,
        "paper_id": construction.get("paper_id") or source.get("paper_id") or instance_id,
        "source": source_info,
        "source_category": source_category(source_info),
        "level1": level1,
        "level2": level2,
        "target_defect": target,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument("--output-dir", type=Path, default=TRACK_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--subset",
        choices=("all", "real", "synthetic"),
        default="all",
        help="Build all atomic instances or only one provenance subset.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Optional sampled pool size after subset filtering; 0 keeps all eligible cases.",
    )
    parser.add_argument(
        "--stratify-by",
        choices=("none", "level1", "level2", "source_category"),
        default="none",
        help="Optional balanced sampling key used only when --sample-size > 0.",
    )
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
        gold = build_gold(instance)
        instance_id = gold["instance_id"]
        if instance_id in seen:
            raise ValueError(f"Duplicate instance id: {instance_id}")
        seen.add(instance_id)
        if args.subset != "all" and gold["source_category"] != args.subset:
            continue
        input_obj = nested_dict(instance, "input")
        input_text = nonempty(input_obj.get("underspecified_spec"))
        if not input_text:
            raise ValueError(f"{instance_id}: missing input.underspecified_spec")
        candidates.append({"input_text": input_text, "gold": gold})

    candidates.sort(key=lambda row: permutation_key(args.seed, row["gold"]["instance_id"]))
    if args.sample_size > 0:
        if args.stratify_by == "none":
            candidates = candidates[: args.sample_size]
        else:
            candidates = balanced_sample(
                candidates,
                sample_size=args.sample_size,
                stratify_by=args.stratify_by,
                seed=args.seed,
            )

    public_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    for position, candidate in enumerate(candidates, start=1):
        case_id = f"task2_track1_{position:06d}"
        public_rows.append({"case_id": case_id, "input_text": candidate["input_text"]})
        gold_rows.append({"case_id": case_id, **candidate["gold"]})

    atomic_write_jsonl(public_rows, pool_path)
    atomic_write_jsonl(gold_rows, gold_path)
    manifest = {
        "benchmark": "IdeaAmbig",
        "task": "Task 2: Defect Localization",
        "track": "Track 1: Controlled Atomic Defect Localization",
        "setting": "taxonomy-guided zero-shot; exactly one target defect per input",
        "input_path": str(args.input.resolve()),
        "subset": args.subset,
        "seed": args.seed,
        "sample_size": args.sample_size or len(public_rows),
        "stratify_by": args.stratify_by,
        "evaluation_cases": len(public_rows),
        "source_counts": dict(sorted(Counter(r["source_category"] for r in gold_rows).items())),
        "level1_counts": dict(sorted(Counter(r["level1"] for r in gold_rows).items())),
        "level2_counts": dict(sorted(Counter(r["level2"] for r in gold_rows).items())),
        "public_pool_fields": ["case_id", "input_text"],
        "private_gold_file": gold_path.name,
        "notes": [
            "Only underspecified specifications are included.",
            "Taxonomy labels, gold defect details, and provenance are private.",
            "Every accepted source instance must contain exactly one taxonomy-consistent defect.",
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
