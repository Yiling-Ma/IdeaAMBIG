import re
import json
import time
import hashlib
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

from tqdm import tqdm


HEADERS = {
    "User-Agent": "Mozilla/5.0 (research; benchmark construction)"
}


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
# Helpers
# ============================================================

def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def safe_filename(x: Any, max_len: int = 120) -> str:
    s = normalize_text(x)
    if not s:
        s = "unknown"
    s = re.sub(r"[^\w\-.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] or "unknown"


def short_hash(x: str, n: int = 10) -> str:
    return hashlib.md5(str(x).encode("utf-8")).hexdigest()[:n]


def first_nonempty(*xs):
    for x in xs:
        if x is not None and x != "" and x != []:
            return x
    return None


def get_record_id(record: Dict[str, Any], idx: int) -> str:
    rid = record.get("record_id")
    if rid:
        return safe_filename(rid)
    return f"record_{idx:05d}"


# ============================================================
# PDF URL helpers
# ============================================================

def normalize_pdf_url(url: Optional[str]) -> str:
    if not url:
        return ""

    url = str(url).strip()

    # OpenReview forum URL -> PDF URL
    if "openreview.net/forum" in url:
        m = re.search(r"[?&]id=([^&]+)", url)
        if m:
            rid = m.group(1).strip()
            rid = re.sub(r"[^0-9A-Za-z_-].*$", "", rid)
            return f"https://openreview.net/pdf?id={rid}"

    # OpenReview relative PDF path
    if url.startswith("/pdf/"):
        return "https://openreview.net" + url

    # arXiv abs -> PDF
    m = re.match(r"https?://arxiv\.org/abs/([^?#]+)", url)
    if m:
        arxiv_id = m.group(1).replace(".pdf", "")
        return f"https://arxiv.org/pdf/{arxiv_id}"

    # arXiv pdf without .pdf is still valid, keep normalized without forcing .pdf.
    m = re.match(r"https?://arxiv\.org/pdf/([^?#]+)", url)
    if m:
        arxiv_id = m.group(1).replace(".pdf", "")
        return f"https://arxiv.org/pdf/{arxiv_id}"

    return url


def is_probably_pdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 5:
        return False
    try:
        with path.open("rb") as f:
            return f.read(5) == b"%PDF-"
    except Exception:
        return False


def download_pdf(
    url: str,
    out_path: Path,
    max_retries: int = 5,
    sleep: float = 5.0,
    timeout: int = 60,
) -> Tuple[bool, str]:
    url = normalize_pdf_url(url)

    if not url:
        return False, "empty_url"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and is_probably_pdf(out_path):
        return True, "exists"

    last_err = ""

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()

            if not data or len(data) < 100:
                last_err = "empty_or_too_small_response"
                time.sleep(sleep)
                continue

            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            with tmp.open("wb") as f:
                f.write(data)

            if not is_probably_pdf(tmp):
                first = data[:120]
                tmp.unlink(missing_ok=True)
                last_err = f"downloaded_file_not_pdf_first_bytes={first!r}"
                time.sleep(sleep)
                continue

            tmp.rename(out_path)
            return True, "downloaded"

        except urllib.error.HTTPError as e:
            last_err = f"http_error_{e.code}"
            wait = sleep * attempt
            time.sleep(wait)

        except Exception as e:
            last_err = repr(e)
            wait = sleep * attempt
            time.sleep(wait)

    return False, f"failed_after_retries: {last_err}"


# ============================================================
# Existing path helpers
# ============================================================

def get_existing_valid_pdf_path(record: Dict[str, Any], path_key: str) -> str:
    lp = record.get("local_paths") or {}
    p = str(lp.get(path_key) or "")
    if p and is_probably_pdf(Path(p)):
        return p
    return ""


def has_valid_report_pdf(r: Dict[str, Any]) -> bool:
    return bool(get_existing_valid_pdf_path(r, "report_pdf_path"))


def has_valid_original_pdf(r: Dict[str, Any]) -> bool:
    return bool(get_existing_valid_pdf_path(r, "original_paper_pdf_path"))


def has_valid_pdf_pair(r: Dict[str, Any]) -> bool:
    return has_valid_report_pdf(r) and has_valid_original_pdf(r)


def build_processed_by_id(processed: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for r in processed:
        rid = str(r.get("record_id") or "").strip()
        if rid:
            by_id[rid] = r
    return by_id


def should_skip_on_resume(
    record: Dict[str, Any],
    idx: int,
    resume: bool,
    done_idx: Dict[str, int],
    processed: List[Dict[str, Any]],
    download_report: bool,
    download_original: bool,
) -> bool:
    if not resume:
        return False

    rid = get_record_id(record, idx)
    prior_i = done_idx.get(rid)
    if prior_i is None:
        return False

    prior = processed[prior_i]

    report_needed = bool(normalize_pdf_url(record.get("report_pdf_url"))) and download_report
    original_needed = bool(normalize_pdf_url(record.get("original_paper_pdf_url"))) and download_original

    report_ok = (not report_needed) or has_valid_report_pdf(prior)
    original_ok = (not original_needed) or has_valid_original_pdf(prior)

    return report_ok and original_ok


# ============================================================
# Progress
# ============================================================

def compute_progress_for_input(
    rows: List[Dict[str, Any]],
    processed_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, int]:
    report_ok = original_ok = both_ok = 0
    original_url_count = 0
    report_url_count = 0

    for idx, record in enumerate(rows, start=1):
        rid = get_record_id(record, idx)
        r = processed_by_id.get(rid)

        if normalize_pdf_url(record.get("report_pdf_url")):
            report_url_count += 1
        if normalize_pdf_url(record.get("original_paper_pdf_url")):
            original_url_count += 1

        if not r:
            continue

        if r.get("has_report_pdf_local"):
            report_ok += 1
        if r.get("has_original_paper_pdf_local"):
            original_ok += 1
        if r.get("has_both_pdfs_local"):
            both_ok += 1

    return {
        "total_records": len(rows),
        "report_pdf_url_records": report_url_count,
        "original_pdf_url_records": original_url_count,
        "report_pdf_ok": report_ok,
        "original_pdf_ok": original_ok,
        "both_pdf_ok": both_ok,
    }


def format_progress_line(stats: Dict[str, int]) -> str:
    n = stats["total_records"]
    return (
        f"report={stats['report_pdf_ok']}/{n} "
        f"original={stats['original_pdf_ok']}/{stats['original_pdf_url_records']}url "
        f"both={stats['both_pdf_ok']}/{n}"
    )


# ============================================================
# Record processing
# ============================================================

def make_pdf_paths(
    record: Dict[str, Any],
    idx: int,
    pdf_dir: Path,
) -> Tuple[Path, Path]:
    rid = get_record_id(record, idx)

    report_url = normalize_pdf_url(record.get("report_pdf_url"))
    original_url = normalize_pdf_url(record.get("original_paper_pdf_url"))

    report_hash = short_hash(report_url) if report_url else "no_report"
    original_hash = short_hash(original_url) if original_url else "no_original"

    report_pdf = pdf_dir / "reports" / f"{rid}_{report_hash}.pdf"
    original_pdf = pdf_dir / "originals" / f"{rid}_{original_hash}.pdf"

    return report_pdf, original_pdf


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
    pdf_dir: Path,
    sleep: float,
    max_retries: int,
    download_report: bool,
    download_original: bool,
) -> Dict[str, Any]:
    r = dict(record)
    local_paths = dict(r.get("local_paths") or {})

    report_url = normalize_pdf_url(r.get("report_pdf_url"))
    original_url = normalize_pdf_url(r.get("original_paper_pdf_url"))

    report_pdf, original_pdf = make_pdf_paths(r, idx, pdf_dir)

    status = dict(r.get("download_status") or {})

    # ----------------------------
    # Report PDF
    # ----------------------------
    existing_report = get_existing_valid_pdf_path(r, "report_pdf_path")

    if existing_report:
        status["report_pdf_download"] = {
            "ok": True,
            "message": "exists_from_input_local_paths",
            "url": report_url,
            "path": existing_report,
        }
        local_paths["report_pdf_path"] = existing_report

    elif not download_report:
        status["report_pdf_download"] = {
            "ok": False,
            "message": "skipped_by_flag",
            "url": report_url,
            "path": "",
        }
        local_paths["report_pdf_path"] = local_paths.get("report_pdf_path", "")

    elif not report_url:
        status["report_pdf_download"] = {
            "ok": False,
            "message": "missing_report_pdf_url",
            "url": "",
            "path": "",
        }
        local_paths["report_pdf_path"] = ""

    else:
        ok, msg = download_pdf(
            report_url,
            report_pdf,
            sleep=sleep,
            max_retries=max_retries,
        )
        status["report_pdf_download"] = {
            "ok": ok,
            "message": msg,
            "url": report_url,
            "path": str(report_pdf),
        }
        local_paths["report_pdf_path"] = str(report_pdf) if is_probably_pdf(report_pdf) else ""

    # ----------------------------
    # Original paper PDF
    # ----------------------------
    existing_original = get_existing_valid_pdf_path(r, "original_paper_pdf_path")

    if existing_original:
        status["original_paper_pdf_download"] = {
            "ok": True,
            "message": "exists_from_input_local_paths",
            "url": original_url,
            "path": existing_original,
        }
        local_paths["original_paper_pdf_path"] = existing_original

    elif not download_original:
        status["original_paper_pdf_download"] = {
            "ok": False,
            "message": "skipped_by_flag",
            "url": original_url,
            "path": "",
        }
        local_paths["original_paper_pdf_path"] = local_paths.get("original_paper_pdf_path", "")

    elif not original_url:
        status["original_paper_pdf_download"] = {
            "ok": False,
            "message": "missing_original_paper_pdf_url",
            "url": "",
            "path": "",
        }
        local_paths["original_paper_pdf_path"] = ""

    else:
        ok, msg = download_pdf(
            original_url,
            original_pdf,
            sleep=sleep,
            max_retries=max_retries,
        )
        status["original_paper_pdf_download"] = {
            "ok": ok,
            "message": msg,
            "url": original_url,
            "path": str(original_pdf),
        }
        local_paths["original_paper_pdf_path"] = str(original_pdf) if is_probably_pdf(original_pdf) else ""

    r["local_paths"] = local_paths
    r["download_status"] = status

    r["has_report_pdf"] = bool(report_url)
    r["has_original_paper_pdf"] = bool(original_url)

    r["has_report_pdf_local"] = bool(get_existing_valid_pdf_path(r, "report_pdf_path"))
    r["has_original_paper_pdf_local"] = bool(get_existing_valid_pdf_path(r, "original_paper_pdf_path"))
    r["has_both_pdfs_local"] = (
        r["has_report_pdf_local"] and r["has_original_paper_pdf_local"]
    )

    return r


# ============================================================
# Filtering
# ============================================================

def filter_rows(
    rows: List[Dict[str, Any]],
    only_mlrc: bool,
    only_tmlr: bool,
    require_original_pdf_url: bool,
) -> List[Dict[str, Any]]:
    out = []

    for r in rows:
        source = r.get("source")

        if only_mlrc and source != "OpenReview_MLRC":
            continue

        if only_tmlr and source != "OpenReview_TMLR":
            continue

        if require_original_pdf_url and not normalize_pdf_url(r.get("original_paper_pdf_url")):
            continue

        out.append(r)

    return out


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--pdf_dir", default="repro_paper_candidates/pdfs")

    parser.add_argument("--only_mlrc", action="store_true")
    parser.add_argument("--only_tmlr", action="store_true")
    parser.add_argument(
        "--require_original_pdf_url",
        action="store_true",
        help="Only keep records that already have original_paper_pdf_url.",
    )

    parser.add_argument(
        "--skip_report",
        action="store_true",
        help="Do not download report PDFs. Existing local_paths.report_pdf_path will still be preserved.",
    )
    parser.add_argument(
        "--skip_original",
        action="store_true",
        help="Do not download original paper PDFs. Existing local_paths.original_paper_pdf_path will still be preserved.",
    )

    parser.add_argument("--sleep", type=float, default=5.0)
    parser.add_argument("--max_retries", type=int, default=5)

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing --out_path file.",
    )
    parser.add_argument(
        "--summary_path",
        default=None,
        help="If omitted, use out_path with .summary.json",
    )

    args = parser.parse_args()

    if args.only_mlrc and args.only_tmlr:
        raise ValueError("Use only one of --only_mlrc or --only_tmlr, not both.")

    input_path = Path(args.input_path)
    out_path = Path(args.out_path)
    pdf_dir = Path(args.pdf_dir)
    summary_path = Path(args.summary_path) if args.summary_path else out_path.with_suffix(".summary.json")

    rows_all = load_jsonl(input_path)
    rows = filter_rows(
        rows_all,
        only_mlrc=args.only_mlrc,
        only_tmlr=args.only_tmlr,
        require_original_pdf_url=args.require_original_pdf_url,
    )

    download_report = not args.skip_report
    download_original = not args.skip_original

    processed: List[Dict[str, Any]] = []
    done_idx: Dict[str, int] = {}

    if args.resume and out_path.exists():
        processed = safe_load_jsonl_if_exists(out_path)
        for i, r in enumerate(processed):
            rid = str(r.get("record_id") or "").strip()
            if rid:
                done_idx[rid] = i

        if processed:
            by_id = build_processed_by_id(processed)
            ckpt = compute_progress_for_input(rows, by_id)
            print("[resume] loaded checkpoint:")
            print(
                f"  rows_in_out={len(processed)} | "
                f"{format_progress_line(ckpt)}"
            )

    processed_by_id = build_processed_by_id(processed)

    pre_skipped = sum(
        1
        for idx, record in enumerate(rows, start=1)
        if should_skip_on_resume(
            record=record,
            idx=idx,
            resume=args.resume,
            done_idx=done_idx,
            processed=processed,
            download_report=download_report,
            download_original=download_original,
        )
    )

    if args.resume and pre_skipped > 0:
        print(
            f"[resume] will skip {pre_skipped}/{len(rows)} records "
            f"(requested PDFs already valid)"
        )

    print("\n===== Universal PDF Downloader =====")
    print(f"Input path: {input_path}")
    print(f"Output path: {out_path}")
    print(f"PDF dir: {pdf_dir}")
    print(f"Input rows before filter: {len(rows_all)}")
    print(f"Rows after filter: {len(rows)}")
    print(f"Download report PDFs: {download_report}")
    print(f"Download original PDFs: {download_original}")
    print(f"Only MLRC: {args.only_mlrc}")
    print(f"Only TMLR: {args.only_tmlr}")
    print(f"Require original PDF URL: {args.require_original_pdf_url}")
    print("====================================\n")

    pbar = tqdm(total=len(rows), desc="Downloading PDFs")

    if pre_skipped > 0:
        pbar.update(pre_skipped)
        init_stats = compute_progress_for_input(rows, processed_by_id)
        pbar.set_postfix_str(format_progress_line(init_stats), refresh=True)

    for idx, record in enumerate(rows, start=1):
        rid = get_record_id(record, idx)

        if should_skip_on_resume(
            record=record,
            idx=idx,
            resume=args.resume,
            done_idx=done_idx,
            processed=processed,
            download_report=download_report,
            download_original=download_original,
        ):
            continue

        if args.resume:
            prior_i = done_idx.get(rid)
            if prior_i is not None:
                prior = processed[prior_i]
                record = merge_prior_local_paths(record, prior)

        out = process_record(
            record=record,
            idx=idx,
            pdf_dir=pdf_dir,
            sleep=args.sleep,
            max_retries=args.max_retries,
            download_report=download_report,
            download_original=download_original,
        )

        prior_i = done_idx.get(rid)
        if prior_i is not None:
            processed[prior_i] = out
        else:
            done_idx[rid] = len(processed)
            processed.append(out)

        processed_by_id[rid] = out

        save_jsonl(processed, out_path)

        pbar.update(1)
        stats = compute_progress_for_input(rows, processed_by_id)
        pbar.set_postfix_str(format_progress_line(stats), refresh=True)

        title = str(out.get("report_title") or rid)[:70]
        print(
            f"[{idx}/{len(rows)}] {title} | "
            f"report={'ok' if out.get('has_report_pdf_local') else 'fail'} "
            f"original={'ok' if out.get('has_original_paper_pdf_local') else 'fail'} "
            f"| original_url={'yes' if normalize_pdf_url(out.get('original_paper_pdf_url')) else 'no'}"
        )

    pbar.close()
    save_jsonl(processed, out_path)

    final = compute_progress_for_input(rows, processed_by_id)

    source_dist = Counter(r.get("source") for r in processed)
    report_msg_dist = Counter()
    original_msg_dist = Counter()

    for r in processed:
        ds = r.get("download_status") or {}
        report_msg = (ds.get("report_pdf_download") or {}).get("message", "missing")
        original_msg = (ds.get("original_paper_pdf_download") or {}).get("message", "missing")
        report_msg_dist[report_msg] += 1
        original_msg_dist[original_msg] += 1

    summary = {
        "script": "step_01_download_pdfs.py",
        "input_path": str(input_path),
        "out_path": str(out_path),
        "pdf_dir": str(pdf_dir),
        "input_rows_before_filter": len(rows_all),
        "records": len(rows),
        "source_distribution": dict(source_dist),
        "download_report": download_report,
        "download_original": download_original,
        "only_mlrc": args.only_mlrc,
        "only_tmlr": args.only_tmlr,
        "require_original_pdf_url": args.require_original_pdf_url,
        "report_pdf_url_records": final["report_pdf_url_records"],
        "original_pdf_url_records": final["original_pdf_url_records"],
        "report_pdfs_local": final["report_pdf_ok"],
        "original_pdfs_local": final["original_pdf_ok"],
        "both_pdfs_local": final["both_pdf_ok"],
        "report_download_message_distribution": dict(report_msg_dist),
        "original_download_message_distribution": dict(original_msg_dist),
        "next_step": (
            "Convert report and original PDFs to markdown. Records without original PDFs "
            "can still be kept for later LLM metadata extraction or filtering."
        ),
    }

    save_json(summary, summary_path)

    print("\n===== PDF Download Summary =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
