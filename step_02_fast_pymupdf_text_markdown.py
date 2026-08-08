import re
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import Counter

import fitz
from tqdm import tqdm


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


def save_jsonl(rows: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_json(obj: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_load_jsonl_if_exists(path: Path) -> List[Dict[str, Any]]:
    if not path or not path.exists():
        return []
    try:
        return load_jsonl(path)
    except Exception:
        return []


# ============================================================
# Text / path helpers
# ============================================================

def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def normalize_markdown(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def is_existing_text(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 50


def is_probably_pdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 5:
        return False
    try:
        with path.open("rb") as f:
            return f.read(5) == b"%PDF-"
    except Exception:
        return False


def safe_filename(x: Any, max_len: int = 160) -> str:
    s = normalize_text(x)
    if not s:
        s = "unknown"
    s = re.sub(r"[^\w\-.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] or "unknown"


def get_record_id(record: Dict[str, Any], idx: int) -> str:
    rid = record.get("record_id")
    if rid:
        return safe_filename(rid)
    return f"record_{idx:05d}"


# ============================================================
# PDF -> fast text markdown
# ============================================================

def pdf_to_text_markdown(pdf_path: Path, md_path: Path) -> Tuple[bool, str, Dict[str, Any]]:
    if is_existing_text(md_path):
        return True, "exists", {
            "num_pages": None,
            "num_chars": md_path.stat().st_size,
        }

    if not pdf_path.exists():
        return False, "pdf_missing", {
            "num_pages": 0,
            "num_chars": 0,
        }

    if not is_probably_pdf(pdf_path):
        return False, "not_valid_pdf", {
            "num_pages": 0,
            "num_chars": 0,
        }

    try:
        md_path.parent.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(str(pdf_path))
        pages = []

        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            text = normalize_markdown(text)
            pages.append(f"\n\n# Page {i}\n\n{text}")

        num_pages = len(doc)
        doc.close()

        md = normalize_markdown("\n".join(pages))

        if len(md) < 50:
            return False, "extracted_text_too_short", {
                "num_pages": num_pages,
                "num_chars": len(md),
            }

        with md_path.open("w", encoding="utf-8") as f:
            f.write(md)

        return True, "converted", {
            "num_pages": num_pages,
            "num_chars": len(md),
        }

    except Exception as e:
        return False, f"conversion_failed: {repr(e)}", {
            "num_pages": 0,
            "num_chars": 0,
        }


def make_fast_md_path(
    pdf_path: str,
    fast_md_dir: Path,
    role: str,
    record: Dict[str, Any],
    idx: int,
) -> Path:
    p = Path(pdf_path)
    rid = get_record_id(record, idx)
    name = f"{rid}_{p.stem}.md"
    return fast_md_dir / role / name


# ============================================================
# Record processing
# ============================================================

def requested_markdown_status(record: Dict[str, Any]) -> Dict[str, bool]:
    lp = record.get("local_paths") or {}
    return {
        "needs_report": bool(lp.get("report_pdf_path")),
        "needs_original": bool(lp.get("original_paper_pdf_path")),
    }


def has_valid_requested_markdowns(record: Dict[str, Any]) -> bool:
    lp = record.get("local_paths") or {}

    report_pdf = str(lp.get("report_pdf_path") or "")
    original_pdf = str(lp.get("original_paper_pdf_path") or "")

    report_md = str(lp.get("report_fast_text_markdown_path") or "")
    original_md = str(lp.get("original_paper_fast_text_markdown_path") or "")

    if report_pdf:
        if not report_md or not is_existing_text(Path(report_md)):
            return False

    if original_pdf:
        if not original_md or not is_existing_text(Path(original_md)):
            return False

    # If neither PDF exists, there is nothing to convert. We do not skip it
    # before it has conversion status, because we still want the record in output.
    return bool(report_pdf or original_pdf)


def merge_prior_local_paths(record: Dict[str, Any], prior: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(record)
    lp = dict(record.get("local_paths") or {})
    prior_lp = prior.get("local_paths") or {}

    for key in [
        "report_pdf_path",
        "original_paper_pdf_path",
        "report_fast_text_markdown_path",
        "original_paper_fast_text_markdown_path",
    ]:
        if prior_lp.get(key) and not lp.get(key):
            lp[key] = prior_lp[key]

    merged["local_paths"] = lp
    return merged


def process_record(
    record: Dict[str, Any],
    idx: int,
    fast_md_dir: Path,
    convert_report: bool,
    convert_original: bool,
) -> Dict[str, Any]:
    r = dict(record)
    local_paths = dict(r.get("local_paths") or {})

    report_pdf_path = str(local_paths.get("report_pdf_path") or "")
    original_pdf_path = str(local_paths.get("original_paper_pdf_path") or "")

    status = dict(r.get("fast_text_conversion_status") or {})

    # ----------------------------
    # Report PDF -> markdown
    # ----------------------------
    if not convert_report:
        report_md = Path(str(local_paths.get("report_fast_text_markdown_path") or ""))
        status["report_fast_text_conversion"] = {
            "ok": bool(report_md and is_existing_text(report_md)),
            "message": "skipped_by_flag",
            "pdf_path": report_pdf_path,
            "markdown_path": str(report_md) if report_md else "",
        }

    elif report_pdf_path:
        report_md = make_fast_md_path(
            report_pdf_path,
            fast_md_dir,
            "reports",
            r,
            idx,
        )
        ok, msg, meta = pdf_to_text_markdown(Path(report_pdf_path), report_md)
        status["report_fast_text_conversion"] = {
            "ok": ok,
            "message": msg,
            "pdf_path": report_pdf_path,
            "markdown_path": str(report_md),
            **meta,
        }
        if ok and is_existing_text(report_md):
            local_paths["report_fast_text_markdown_path"] = str(report_md)
        else:
            local_paths["report_fast_text_markdown_path"] = ""

    else:
        report_md = Path("")
        status["report_fast_text_conversion"] = {
            "ok": False,
            "message": "missing_report_pdf_path",
            "pdf_path": "",
            "markdown_path": "",
            "num_pages": 0,
            "num_chars": 0,
        }
        local_paths["report_fast_text_markdown_path"] = ""

    # ----------------------------
    # Original PDF -> markdown
    # ----------------------------
    if not convert_original:
        original_md = Path(str(local_paths.get("original_paper_fast_text_markdown_path") or ""))
        status["original_fast_text_conversion"] = {
            "ok": bool(original_md and is_existing_text(original_md)),
            "message": "skipped_by_flag",
            "pdf_path": original_pdf_path,
            "markdown_path": str(original_md) if original_md else "",
        }

    elif original_pdf_path:
        original_md = make_fast_md_path(
            original_pdf_path,
            fast_md_dir,
            "originals",
            r,
            idx,
        )
        ok, msg, meta = pdf_to_text_markdown(Path(original_pdf_path), original_md)
        status["original_fast_text_conversion"] = {
            "ok": ok,
            "message": msg,
            "pdf_path": original_pdf_path,
            "markdown_path": str(original_md),
            **meta,
        }
        if ok and is_existing_text(original_md):
            local_paths["original_paper_fast_text_markdown_path"] = str(original_md)
        else:
            local_paths["original_paper_fast_text_markdown_path"] = ""

    else:
        original_md = Path("")
        status["original_fast_text_conversion"] = {
            "ok": False,
            "message": "missing_original_pdf_path",
            "pdf_path": "",
            "markdown_path": "",
            "num_pages": 0,
            "num_chars": 0,
        }
        local_paths["original_paper_fast_text_markdown_path"] = ""

    r["local_paths"] = local_paths
    r["fast_text_conversion_status"] = status

    r["has_report_fast_text_markdown"] = bool(
        local_paths.get("report_fast_text_markdown_path")
        and is_existing_text(Path(local_paths["report_fast_text_markdown_path"]))
    )
    r["has_original_paper_fast_text_markdown"] = bool(
        local_paths.get("original_paper_fast_text_markdown_path")
        and is_existing_text(Path(local_paths["original_paper_fast_text_markdown_path"]))
    )
    r["has_both_fast_text_markdowns"] = (
        r["has_report_fast_text_markdown"]
        and r["has_original_paper_fast_text_markdown"]
    )

    # More useful for TMLR: record is conversion-complete if all locally available PDFs
    # have markdown. If no original PDF is available, report-only can still be complete.
    requested = requested_markdown_status(r)
    r["has_all_available_fast_text_markdowns"] = (
        (not requested["needs_report"] or r["has_report_fast_text_markdown"])
        and (not requested["needs_original"] or r["has_original_paper_fast_text_markdown"])
        and (requested["needs_report"] or requested["needs_original"])
    )

    return r


# ============================================================
# Filtering
# ============================================================

def filter_rows(
    rows: List[Dict[str, Any]],
    only_mlrc: bool,
    only_tmlr: bool,
    require_original_pdf_local: bool,
    require_both_pdf_local: bool,
) -> List[Dict[str, Any]]:
    out = []

    for r in rows:
        source = r.get("source")
        lp = r.get("local_paths") or {}

        if only_mlrc and source != "OpenReview_MLRC":
            continue

        if only_tmlr and source != "OpenReview_TMLR":
            continue

        if require_original_pdf_local and not lp.get("original_paper_pdf_path"):
            continue

        if require_both_pdf_local and not (
            lp.get("report_pdf_path") and lp.get("original_paper_pdf_path")
        ):
            continue

        out.append(r)

    return out


# ============================================================
# Summary
# ============================================================

def count_has(rows: List[Dict[str, Any]], key: str) -> int:
    return sum(1 for r in rows if r.get(key))


def get_message_distribution(rows: List[Dict[str, Any]], status_key: str) -> Dict[str, int]:
    c = Counter()
    for r in rows:
        st = r.get("fast_text_conversion_status") or {}
        msg = (st.get(status_key) or {}).get("message", "missing")
        c[msg] += 1
    return dict(c)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument(
        "--fast_md_dir",
        default="repro_paper_candidates/markdown_fast_pymupdf",
    )

    parser.add_argument("--only_mlrc", action="store_true")
    parser.add_argument("--only_tmlr", action="store_true")
    parser.add_argument(
        "--require_original_pdf_local",
        action="store_true",
        help="Only keep records that already have local_paths.original_paper_pdf_path.",
    )
    parser.add_argument(
        "--require_both_pdf_local",
        action="store_true",
        help="Only keep records that have both report and original PDFs locally.",
    )

    parser.add_argument(
        "--skip_report",
        action="store_true",
        help="Do not convert report PDFs.",
    )
    parser.add_argument(
        "--skip_original",
        action="store_true",
        help="Do not convert original paper PDFs.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing --out_path file.",
    )
    parser.add_argument(
        "--summary_path",
        default=None,
        help="If omitted, use out_path with .summary.json.",
    )

    args = parser.parse_args()

    if args.only_mlrc and args.only_tmlr:
        raise ValueError("Use only one of --only_mlrc or --only_tmlr, not both.")

    input_path = Path(args.input_path)
    out_path = Path(args.out_path)
    fast_md_dir = Path(args.fast_md_dir)
    summary_path = Path(args.summary_path) if args.summary_path else out_path.with_suffix(".summary.json")

    rows_all = load_jsonl(input_path)
    rows = filter_rows(
        rows_all,
        only_mlrc=args.only_mlrc,
        only_tmlr=args.only_tmlr,
        require_original_pdf_local=args.require_original_pdf_local,
        require_both_pdf_local=args.require_both_pdf_local,
    )

    convert_report = not args.skip_report
    convert_original = not args.skip_original

    processed: List[Dict[str, Any]] = []
    done_idx: Dict[str, int] = {}

    if args.resume and out_path.exists():
        processed = safe_load_jsonl_if_exists(out_path)
        for i, r in enumerate(processed):
            rid = str(r.get("record_id") or "").strip()
            if rid:
                done_idx[rid] = i

    print("\n===== Universal PyMuPDF Fast Markdown Converter =====")
    print(f"Input path: {input_path}")
    print(f"Output path: {out_path}")
    print(f"Fast markdown dir: {fast_md_dir}")
    print(f"Input rows before filter: {len(rows_all)}")
    print(f"Rows after filter: {len(rows)}")
    print(f"Convert report PDFs: {convert_report}")
    print(f"Convert original PDFs: {convert_original}")
    print(f"Only MLRC: {args.only_mlrc}")
    print(f"Only TMLR: {args.only_tmlr}")
    print(f"Require original local PDF: {args.require_original_pdf_local}")
    print(f"Require both local PDFs: {args.require_both_pdf_local}")
    print("=====================================================\n")

    for idx, r in enumerate(tqdm(rows, desc="Step 02: PyMuPDF fast text markdown"), start=1):
        rid = str(r.get("record_id") or "").strip()

        if args.resume and rid:
            prior_i = done_idx.get(rid)
            if prior_i is not None:
                prior = processed[prior_i]
                if has_valid_requested_markdowns(prior):
                    continue
                r = merge_prior_local_paths(r, prior)

        out = process_record(
            record=r,
            idx=idx,
            fast_md_dir=fast_md_dir,
            convert_report=convert_report,
            convert_original=convert_original,
        )

        if rid and rid in done_idx:
            processed[done_idx[rid]] = out
        else:
            if rid:
                done_idx[rid] = len(processed)
            processed.append(out)

        save_jsonl(processed, out_path)

    save_jsonl(processed, out_path)

    source_dist = Counter(r.get("source") for r in processed)

    summary = {
        "script": "step_02_fast_pymupdf_text_markdown.py",
        "input_path": str(input_path),
        "out_path": str(out_path),
        "fast_md_dir": str(fast_md_dir),
        "input_rows_before_filter": len(rows_all),
        "records": len(processed),
        "source_distribution": dict(source_dist),
        "convert_report": convert_report,
        "convert_original": convert_original,
        "only_mlrc": args.only_mlrc,
        "only_tmlr": args.only_tmlr,
        "require_original_pdf_local": args.require_original_pdf_local,
        "require_both_pdf_local": args.require_both_pdf_local,

        "has_report_pdf_local": sum(1 for r in processed if (r.get("local_paths") or {}).get("report_pdf_path")),
        "has_original_paper_pdf_local": sum(1 for r in processed if (r.get("local_paths") or {}).get("original_paper_pdf_path")),
        "has_report_fast_text_markdown": count_has(processed, "has_report_fast_text_markdown"),
        "has_original_paper_fast_text_markdown": count_has(processed, "has_original_paper_fast_text_markdown"),
        "has_both_fast_text_markdowns": count_has(processed, "has_both_fast_text_markdowns"),
        "has_all_available_fast_text_markdowns": count_has(processed, "has_all_available_fast_text_markdowns"),

        "report_conversion_message_distribution": get_message_distribution(
            processed,
            "report_fast_text_conversion",
        ),
        "original_conversion_message_distribution": get_message_distribution(
            processed,
            "original_fast_text_conversion",
        ),

        "next_step": (
            "Run PDF-level LLM filtering/extraction or construct benchmark instances. "
            "Records without original markdown can still be kept and handled later."
        ),
    }

    save_json(summary, summary_path)

    print("\n===== Step 02 Summary =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
