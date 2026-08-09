import json
import argparse
from pathlib import Path
from typing import Any, Dict, List
from collections import Counter



# ============================================================
# IO
# ============================================================

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_load_jsonl_if_exists(path: Path) -> List[Dict[str, Any]]:
    if not path or not path.exists():
        return []
    return load_jsonl(path)


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
# Filtering helpers
# ============================================================

def get_primary_route(record: Dict[str, Any]) -> str:
    route = record.get("llm_route_classification") or {}
    return str(route.get("primary_route") or "").strip()


def has_original_text(record: Dict[str, Any]) -> bool:
    """
    Check whether the record has usable original paper text path.
    This can be fast PyMuPDF markdown or MinerU markdown.
    """
    local_paths = record.get("local_paths") or {}

    candidates = [
        local_paths.get("original_paper_fast_text_markdown_path"),
        local_paths.get("original_paper_mineru_markdown_path"),
    ]

    for p in candidates:
        if p and Path(p).exists() and Path(p).stat().st_size > 50:
            return True

    return False


def has_original_pdf(record: Dict[str, Any]) -> bool:
    local_paths = record.get("local_paths") or {}
    p = local_paths.get("original_paper_pdf_path")
    if p and Path(p).exists() and Path(p).stat().st_size > 50:
        return True

    return bool(record.get("has_original_paper_pdf_local"))


def passes_clean_synthetic_filter(
    record: Dict[str, Any],
    require_original_text: bool = False,
    require_original_pdf: bool = False,
    require_single_target: bool = False,
) -> bool:
    """
    Base condition:
      primary_route == synthetic_controlled

    Optional strict conditions:
      require_original_text
      require_original_pdf
      require_single_target
    """
    if get_primary_route(record) != "synthetic_controlled":
        return False

    if require_single_target and record.get("is_single_target") is not True:
        return False

    if require_original_text and not has_original_text(record):
        return False

    if require_original_pdf and not has_original_pdf(record):
        return False

    return True


def add_synthetic_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep original record structure, but add a small metadata field for downstream synthetic route.
    """
    out = dict(record)

    route = out.get("llm_route_classification") or {}
    local_paths = out.get("local_paths") or {}

    out["synthetic_selection"] = {
        "selected": True,
        "selection_route": "synthetic_controlled",
        "primary_route": route.get("primary_route"),
        "routing_reason": route.get("routing_reason", ""),
        "confidence": route.get("confidence", None),
        "has_original_fast_text_markdown": bool(local_paths.get("original_paper_fast_text_markdown_path")),
        "has_original_mineru_markdown": bool(local_paths.get("original_paper_mineru_markdown_path")),
        "has_original_pdf": bool(local_paths.get("original_paper_pdf_path")) or bool(out.get("has_original_paper_pdf_local")),
        "next_step": (
            "Use original paper markdown to construct a codification-ready gold specification, "
            "then inject exactly one synthetic defect using the shared benchmark schema."
        ),
    }

    return out


def has_valid_synthetic_selection(record: Dict[str, Any]) -> bool:
    sel = record.get("synthetic_selection")
    if not isinstance(sel, dict):
        return False
    return sel.get("selected") is True and sel.get("selection_route") == "synthetic_controlled"


# ============================================================
# Summary
# ============================================================

def build_summary(
    all_rows: List[Dict[str, Any]],
    selected_rows: List[Dict[str, Any]],
    require_original_text: bool,
    require_original_pdf: bool,
    require_single_target: bool,
    input_limit: int = 0,
) -> Dict[str, Any]:
    route_counter = Counter(get_primary_route(r) for r in all_rows)

    selected_year_counter = Counter(str(r.get("year", "unknown")) for r in selected_rows)

    selected_has_original_text = sum(has_original_text(r) for r in selected_rows)
    selected_has_original_pdf = sum(has_original_pdf(r) for r in selected_rows)

    selected_confidences = []
    for r in selected_rows:
        route = r.get("llm_route_classification") or {}
        conf = route.get("confidence")
        if isinstance(conf, (int, float)):
            selected_confidences.append(float(conf))

    if selected_confidences:
        avg_conf = sum(selected_confidences) / len(selected_confidences)
    else:
        avg_conf = None

    return {
        "input_records": len(all_rows),
        "selected_synthetic_controlled": len(selected_rows),
        "route_distribution_all": dict(route_counter),
        "selected_year_distribution": dict(selected_year_counter),
        "selected_has_original_text": selected_has_original_text,
        "selected_has_original_pdf": selected_has_original_pdf,
        "selected_average_confidence": avg_conf,
        "filters": {
            "primary_route": "synthetic_controlled",
            "require_original_text": require_original_text,
            "require_original_pdf": require_original_pdf,
            "require_single_target": require_single_target,
            "input_limit": input_limit if input_limit > 0 else None,
        },
        "recommended_next_step": (
            "Run synthetic gold-spec construction on selected records. "
            "For each selected paper, build a complete codification-ready paper-derived specification, "
            "then generate one single-defect synthetic instance using the shared real/synthetic schema."
        ),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--summary_path", required=True)

    parser.add_argument(
        "--require_original_text",
        action="store_true",
        help="Keep only synthetic_controlled records with original paper markdown available.",
    )

    parser.add_argument(
        "--require_original_pdf",
        action="store_true",
        help="Keep only synthetic_controlled records with original paper PDF available.",
    )

    parser.add_argument(
        "--require_single_target",
        action="store_true",
        help="Keep only records where is_single_target == True.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N input records (0 = no limit).",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing --out_path file. Skips record_ids already present with valid synthetic_selection.",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    out_path = Path(args.out_path)
    summary_path = Path(args.summary_path)

    rows = load_jsonl(input_path)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    selected: List[Dict[str, Any]] = []
    done_idx: Dict[str, int] = {}

    if args.resume and out_path.exists():
        selected = safe_load_jsonl_if_exists(out_path)
        for i, r in enumerate(selected):
            rid = str(r.get("record_id") or "").strip()
            if rid:
                done_idx[rid] = i

    num_skipped = 0
    num_added = 0
    num_updated = 0

    for r in rows:
        if not passes_clean_synthetic_filter(
            r,
            require_original_text=args.require_original_text,
            require_original_pdf=args.require_original_pdf,
            require_single_target=args.require_single_target,
        ):
            continue

        rid = str(r.get("record_id") or "").strip()
        out_row = add_synthetic_metadata(r)

        if args.resume and rid:
            prior_i = done_idx.get(rid)
            if prior_i is not None and has_valid_synthetic_selection(selected[prior_i]):
                num_skipped += 1
                continue
            if prior_i is not None:
                selected[prior_i] = out_row
                num_updated += 1
                continue

        if rid:
            done_idx[rid] = len(selected)
        selected.append(out_row)
        num_added += 1

    save_jsonl(selected, out_path)

    summary = build_summary(
        all_rows=rows,
        selected_rows=selected,
        require_original_text=args.require_original_text,
        require_original_pdf=args.require_original_pdf,
        require_single_target=args.require_single_target,
        input_limit=args.limit,
    )
    save_json(summary, summary_path)

    print("\n===== Synthetic Controlled Filtering Summary =====")
    print(f"Input records: {len(rows)}")
    print(f"Selected synthetic_controlled: {len(selected)}")
    if args.resume:
        print(f"Resume skipped: {num_skipped}")
        print(f"Resume added: {num_added}")
        print(f"Resume updated: {num_updated}")
    print(f"Output: {out_path}")
    print(f"Summary: {summary_path}")
    print("\nRoute distribution:")
    for k, v in summary["route_distribution_all"].items():
        print(f"  {k}: {v}")
    print("\nSelected availability:")
    print(f"  has original text: {summary['selected_has_original_text']} / {len(selected)}")
    print(f"  has original PDF:  {summary['selected_has_original_pdf']} / {len(selected)}")


if __name__ == "__main__":
    main()
