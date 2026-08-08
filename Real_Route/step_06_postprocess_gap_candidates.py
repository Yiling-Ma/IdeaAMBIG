import json
import argparse
from pathlib import Path
from typing import Any, Dict, List
from collections import Counter

# ============================================================
# IO
# ============================================================

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path or not path.exists():
        return []

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def save_jsonl(rows: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_json(obj: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ============================================================
# Universal source helpers
# ============================================================

VALID_SOURCE_FILTERS = {
    "all",
    "OpenReview_MLRC",
    "OpenReview_TMLR",
}


def source_matches(record: Dict[str, Any], source_filter: str) -> bool:
    if source_filter == "all":
        return True
    return str(record.get("source") or "") == source_filter


def infer_dataset_tag_from_source(source_filter: str) -> str:
    if source_filter == "OpenReview_MLRC":
        return "mlrc"
    if source_filter == "OpenReview_TMLR":
        return "tmlr"
    return "mixed"


def default_paths(dataset_tag: str) -> Dict[str, Path]:
    base = Path(f"Real_bench/{dataset_tag}")
    outputs = base / "outputs"
    post = base / "postprocessed_real_gap"

    return {
        "resolved_path": outputs / f"{dataset_tag}_resolved_real_gap_candidates_main_ready.jsonl",
        "unresolved_path": outputs / f"{dataset_tag}_unresolved_audit_only.jsonl",
        "invalid_path": outputs / f"{dataset_tag}_invalid_or_rejected_candidates.jsonl",
        "out_dir": post,
    }


# ============================================================
# Helpers
# ============================================================

def get_gap(item: Dict[str, Any]) -> Dict[str, Any]:
    return item.get("gap") or {}


def get_candidate(item: Dict[str, Any]) -> Dict[str, Any]:
    return item.get("candidate") or {}


def slim_source_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Strip MinerU tqdm logs from embedded source_record."""
    r = dict(record)
    mineru_status = r.get("mineru_selected_status")
    if not isinstance(mineru_status, dict):
        return r

    slim = dict(mineru_status)
    conv = slim.get("status")
    if isinstance(conv, dict):
        slim_conv = {}
        for key, val in conv.items():
            if not isinstance(val, dict):
                slim_conv[key] = val
                continue

            slim_val = {
                "ok": val.get("ok"),
                "message": val.get("message"),
            }

            extra = val.get("extra")
            if isinstance(extra, dict):
                slim_extra = {
                    k: v
                    for k, v in extra.items()
                    if k not in {"stdout_tail", "stderr_tail"}
                }
                slim_val["extra"] = slim_extra

            slim_conv[key] = slim_val

        slim["status"] = slim_conv

    r["mineru_selected_status"] = slim
    return r


def clean_output_item(item: Dict[str, Any], *, drop_source_record: bool = False) -> Dict[str, Any]:
    out = dict(item)

    if drop_source_record:
        out.pop("source_record", None)
    elif "source_record" in out and isinstance(out["source_record"], dict):
        out["source_record"] = slim_source_record(out["source_record"])

    return out


def add_postprocess_info(
    item: Dict[str, Any],
    split: str,
    reasons: List[str],
) -> Dict[str, Any]:
    out = dict(item)
    out["postprocess_split"] = split
    out["postprocess_reasons"] = reasons
    return out


def filter_by_source(rows: List[Dict[str, Any]], source_filter: str) -> List[Dict[str, Any]]:
    return [r for r in rows if source_matches(r, source_filter)]


# ============================================================
# Classification
# ============================================================

def classify_resolved(item: Dict[str, Any]) -> tuple[str, List[str]]:
    """
    Classify resolved real-gap candidates.

    Main benchmark rule:
      Use only main_ready_candidate=True.

    If Step 05 was run with --keep_only_main_ready, almost all candidates here
    should go to main_resolved.
    """
    g = get_gap(item)
    reasons = []

    passes_min = bool(g.get("passes_min_validation"))
    main_ready = bool(g.get("main_ready_candidate"))
    strength = g.get("candidate_strength", "")
    solution_source = g.get("solution_source_type", "")
    manual = bool(g.get("recommended_manual_review"))

    if main_ready:
        reasons.append("main_ready_candidate_true")
        return "main_resolved", reasons

    if not passes_min:
        reasons.append("does_not_pass_min_validation")
        return "rejected", reasons

    if strength == "weak":
        reasons.append("candidate_strength_weak")
        return "rejected", reasons

    # Anything not main-ready but not clearly invalid goes to manual review.
    reasons.append("not_main_ready_candidate")

    if strength:
        reasons.append(f"candidate_strength_{strength}")

    if solution_source:
        reasons.append(f"solution_source_{solution_source}")

    if manual:
        reasons.append("recommended_manual_review_true")

    return "review_needed", reasons


def classify_unresolved(item: Dict[str, Any]) -> tuple[str, List[str]]:
    """
    Unresolved candidates are audit-only.

    They cannot be used for benchmark construction because no gold clarified
    specification is extractable.
    """
    g = get_gap(item)
    reasons = [
        "unresolved_gap_audit_only",
        "no_gold_clarified_specification",
    ]

    if g.get("passes_min_validation"):
        reasons.append("passes_min_validation")
    else:
        reasons.append("does_not_pass_min_validation")

    strength = g.get("candidate_strength", "")
    if strength:
        reasons.append(f"candidate_strength_{strength}")

    return "unresolved_audit_only", reasons


def classify_invalid(item: Dict[str, Any]) -> tuple[str, List[str]]:
    c = get_candidate(item)
    reason = c.get("rejection_reason", "unknown")
    return "rejected", [f"invalid_or_rejected_candidate:{reason}"]


# ============================================================
# Summary
# ============================================================

def summarize_splits(
    main_resolved: List[Dict[str, Any]],
    review_needed: List[Dict[str, Any]],
    unresolved_audit_only: List[Dict[str, Any]],
    rejected: List[Dict[str, Any]],
    source_filter: str,
    resolved_input_count: int,
    unresolved_input_count: int,
    invalid_input_count: int,
) -> Dict[str, Any]:

    def label_counter(rows: List[Dict[str, Any]]) -> Dict[str, int]:
        c: Counter = Counter()
        for r in rows:
            g = r.get("gap") or {}
            if g:
                c[f"{g.get('level1')} / {g.get('level2')}"] += 1
        return dict(c)

    def strength_counter(rows: List[Dict[str, Any]]) -> Dict[str, int]:
        c: Counter = Counter()
        for r in rows:
            g = r.get("gap") or {}
            if g:
                c[str(g.get("candidate_strength"))] += 1
        return dict(c)

    def reason_counter(rows: List[Dict[str, Any]]) -> Dict[str, int]:
        c: Counter = Counter()
        for r in rows:
            for reason in r.get("postprocess_reasons", []):
                c[reason] += 1
        return dict(c)

    def solution_source_counter(rows: List[Dict[str, Any]]) -> Dict[str, int]:
        c: Counter = Counter()
        for r in rows:
            g = r.get("gap") or {}
            src = g.get("solution_source_type", "n/a")
            c[src] += 1
        return dict(c)

    def source_counter(rows: List[Dict[str, Any]]) -> Dict[str, int]:
        c: Counter = Counter()
        for r in rows:
            c[str(r.get("source") or "missing")] += 1
        return dict(c)

    summary = {
        "script": "step_06_postprocess_gap_candidates.py",
        "source_filter": source_filter,
        "input_counts_after_source_filter": {
            "resolved": resolved_input_count,
            "unresolved": unresolved_input_count,
            "invalid": invalid_input_count,
        },
        "counts": {
            "main_resolved": len(main_resolved),
            "review_needed": len(review_needed),
            "unresolved_audit_only": len(unresolved_audit_only),
            "rejected": len(rejected),
            "total": (
                len(main_resolved)
                + len(review_needed)
                + len(unresolved_audit_only)
                + len(rejected)
            ),
        },
        "source_distribution": {
            "main_resolved": source_counter(main_resolved),
            "review_needed": source_counter(review_needed),
            "unresolved_audit_only": source_counter(unresolved_audit_only),
            "rejected": source_counter(rejected),
        },
        "main_resolved_label_distribution": label_counter(main_resolved),
        "review_needed_label_distribution": label_counter(review_needed),
        "unresolved_audit_label_distribution": label_counter(unresolved_audit_only),
        "main_resolved_strength_distribution": strength_counter(main_resolved),
        "review_needed_strength_distribution": strength_counter(review_needed),
        "unresolved_audit_strength_distribution": strength_counter(unresolved_audit_only),
        "main_resolved_solution_source_distribution": solution_source_counter(main_resolved),
        "review_needed_solution_source_distribution": solution_source_counter(review_needed),
        "review_needed_reasons": reason_counter(review_needed),
        "unresolved_audit_reasons": reason_counter(unresolved_audit_only),
        "rejected_reasons": reason_counter(rejected),
        "next_input_for_real_gap_instance_builder": "main_resolved.jsonl",
    }

    return summary


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source_filter",
        default="all",
        choices=["all", "OpenReview_MLRC", "OpenReview_TMLR"],
        help="Which source to post-process.",
    )

    parser.add_argument(
        "--dataset_tag",
        default="",
        help="Dataset tag used for default paths. If omitted, inferred from source_filter.",
    )

    parser.add_argument(
        "--resolved_path",
        default="",
        help="Step 05 resolved candidates path. If omitted, inferred from dataset_tag.",
    )

    parser.add_argument(
        "--unresolved_path",
        default="",
        help="Step 05 unresolved audit-only path. If omitted, inferred from dataset_tag.",
    )

    parser.add_argument(
        "--invalid_path",
        default="",
        help="Step 05 invalid/rejected candidates path. If omitted, inferred from dataset_tag.",
    )

    parser.add_argument(
        "--out_dir",
        default="",
        help="Output directory. If omitted, inferred from dataset_tag.",
    )

    parser.add_argument(
        "--keep_source_record_in_main",
        action="store_true",
        help="Keep source_record in main_resolved.jsonl. Default drops it.",
    )

    args = parser.parse_args()

    dataset_tag = args.dataset_tag.strip() or infer_dataset_tag_from_source(args.source_filter)
    paths = default_paths(dataset_tag)

    resolved_path = Path(args.resolved_path) if args.resolved_path else paths["resolved_path"]
    unresolved_path = Path(args.unresolved_path) if args.unresolved_path else paths["unresolved_path"]
    invalid_path = Path(args.invalid_path) if args.invalid_path else paths["invalid_path"]
    out_dir = Path(args.out_dir) if args.out_dir else paths["out_dir"]

    resolved_rows_all = load_jsonl(resolved_path)
    unresolved_rows_all = load_jsonl(unresolved_path)
    invalid_rows_all = load_jsonl(invalid_path)

    resolved_rows = filter_by_source(resolved_rows_all, args.source_filter)
    unresolved_rows = filter_by_source(unresolved_rows_all, args.source_filter)
    invalid_rows = filter_by_source(invalid_rows_all, args.source_filter)

    main_resolved: List[Dict[str, Any]] = []
    review_needed: List[Dict[str, Any]] = []
    unresolved_audit_only: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for item in resolved_rows:
        split, reasons = classify_resolved(item)
        out = add_postprocess_info(item, split, reasons)
        out = clean_output_item(
            out,
            drop_source_record=(split == "main_resolved" and not args.keep_source_record_in_main),
        )

        if split == "main_resolved":
            main_resolved.append(out)
        elif split == "review_needed":
            review_needed.append(out)
        else:
            rejected.append(out)

    for item in unresolved_rows:
        split, reasons = classify_unresolved(item)
        out = add_postprocess_info(item, split, reasons)
        out = clean_output_item(out)

        # Always audit-only. Never benchmark input.
        unresolved_audit_only.append(out)

    for item in invalid_rows:
        split, reasons = classify_invalid(item)
        out = add_postprocess_info(item, split, reasons)
        out = clean_output_item(out)
        rejected.append(out)

    save_jsonl(main_resolved, out_dir / "main_resolved.jsonl")
    save_jsonl(review_needed, out_dir / "review_needed.jsonl")
    save_jsonl(unresolved_audit_only, out_dir / "unresolved_audit_only.jsonl")
    save_jsonl(rejected, out_dir / "rejected.jsonl")

    summary = summarize_splits(
        main_resolved=main_resolved,
        review_needed=review_needed,
        unresolved_audit_only=unresolved_audit_only,
        rejected=rejected,
        source_filter=args.source_filter,
        resolved_input_count=len(resolved_rows),
        unresolved_input_count=len(unresolved_rows),
        invalid_input_count=len(invalid_rows),
    )

    summary["paths"] = {
        "resolved_path": str(resolved_path),
        "unresolved_path": str(unresolved_path),
        "invalid_path": str(invalid_path),
        "out_dir": str(out_dir),
        "main_resolved": str(out_dir / "main_resolved.jsonl"),
        "review_needed": str(out_dir / "review_needed.jsonl"),
        "unresolved_audit_only": str(out_dir / "unresolved_audit_only.jsonl"),
        "rejected": str(out_dir / "rejected.jsonl"),
        "summary": str(out_dir / "summary.json"),
    }

    save_json(summary, out_dir / "summary.json")

    print("\n===== Step 06 Post-processing Summary =====")
    print(f"Source filter:              {args.source_filter}")
    print(f"Dataset tag:                {dataset_tag}")
    print(f"Resolved path:              {resolved_path}")
    print(f"Unresolved path:            {unresolved_path}")
    print(f"Invalid path:               {invalid_path}")
    print(f"Resolved input all/source:  {len(resolved_rows_all)} / {len(resolved_rows)}")
    print(f"Unresolved input all/source:{len(unresolved_rows_all)} / {len(unresolved_rows)}")
    print(f"Invalid input all/source:   {len(invalid_rows_all)} / {len(invalid_rows)}")
    print(f"Main resolved:              {len(main_resolved)}")
    print(f"Review needed:              {len(review_needed)}")
    print(f"Unresolved audit only:      {len(unresolved_audit_only)}")
    print(f"Rejected:                   {len(rejected)}")
    print(f"Output dir:                 {out_dir}")
    print("\nNext input for real-gap instance builder:")
    print(f"  {out_dir / 'main_resolved.jsonl'}")


if __name__ == "__main__":
    main()
