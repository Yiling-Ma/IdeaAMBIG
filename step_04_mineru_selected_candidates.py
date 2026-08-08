import os
import re
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def safe_load_jsonl_if_exists(path: Path) -> List[Dict[str, Any]]:
    if not path or not path.exists():
        return []
    try:
        return load_jsonl(path)
    except Exception:
        return []


def safe_filename(x: Any, max_len: int = 160) -> str:
    s = re.sub(r"\s+", " ", str(x or "")).strip()
    if not s:
        s = "unknown"
    s = re.sub(r"[^\w\-.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len]


def find_markdown_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.md"))


def choose_best_markdown(md_files: List[Path]) -> Optional[Path]:
    md_files = [p for p in md_files if p.exists() and p.stat().st_size > 50]
    if not md_files:
        return None
    return max(md_files, key=lambda p: p.stat().st_size)


def is_existing_text(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 50


# ============================================================
# Universal source / route logic
# ============================================================

VALID_ROUTE_FILTERS = {
    "resolved_real_gap",
    "synthetic_controlled",
}

VALID_SOURCE_FILTERS = {
    "all",
    "OpenReview_MLRC",
    "OpenReview_TMLR",
}


def source_matches(record: Dict[str, Any], source_filter: str) -> bool:
    if source_filter == "all":
        return True
    return str(record.get("source") or "") == source_filter


def infer_dataset_tag(rows: List[Dict[str, Any]], source_filter: str) -> str:
    if source_filter == "OpenReview_MLRC":
        return "mlrc"
    if source_filter == "OpenReview_TMLR":
        return "tmlr"

    sources = sorted(set(str(r.get("source") or "unknown") for r in rows))
    if sources == ["OpenReview_MLRC"]:
        return "mlrc"
    if sources == ["OpenReview_TMLR"]:
        return "tmlr"
    return "mixed"


def get_primary_route(record: Dict[str, Any]) -> str:
    route = record.get("llm_route_classification") or {}
    return str(route.get("primary_route") or "").strip()


def should_select_record(record: Dict[str, Any], route_filter: str) -> bool:
    primary = get_primary_route(record)
    return primary == route_filter


def required_scope_for_route(route_filter: str) -> str:
    if route_filter == "resolved_real_gap":
        return "both"
    if route_filter == "synthetic_controlled":
        return "original_only"
    raise ValueError(f"Unsupported route_filter: {route_filter}")


def get_source_prefix(record: Dict[str, Any]) -> str:
    source = str(record.get("source") or "").strip()
    if source == "OpenReview_MLRC":
        return "mlrc"
    if source == "OpenReview_TMLR":
        return "tmlr"
    return safe_filename(source or "unknown_source", max_len=40)


def get_record_id(record: Dict[str, Any], idx: int) -> str:
    """
    Filename-safe record id.

    We include a source prefix to prevent possible filename collisions when
    MLRC and TMLR records are processed in one mixed file.
    """
    prefix = get_source_prefix(record)
    rid = record.get("record_id")
    if rid:
        return safe_filename(f"{prefix}_{rid}")
    return f"{prefix}_record_{idx:05d}"


# ============================================================
# MinerU
# ============================================================

def make_subprocess_env(num_threads: int = 1) -> Dict[str, str]:
    """
    Reduce thread-affinity warnings and oversubscription.

    These env vars usually help reduce:
      onnxruntime pthread_setaffinity_np failed ...
    """
    env = os.environ.copy()

    thread_vars = {
        "OMP_NUM_THREADS": str(num_threads),
        "MKL_NUM_THREADS": str(num_threads),
        "OPENBLAS_NUM_THREADS": str(num_threads),
        "NUMEXPR_NUM_THREADS": str(num_threads),
        "VECLIB_MAXIMUM_THREADS": str(num_threads),
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_WAIT_POLICY": "PASSIVE",
    }

    for k, v in thread_vars.items():
        env[k] = v

    return env


def convert_with_mineru(
    pdf_path: Path,
    final_md_path: Path,
    raw_output_dir: Path,
    mineru_bin: str = "mineru",
    method: str = "txt",
    backend: str = "pipeline",
    lang: str = "en",
    timeout: int = 3600,
    extra_args: Optional[List[str]] = None,
    stream_logs: bool = False,
    quiet: bool = True,
    subprocess_threads: int = 1,
) -> Tuple[bool, str, Dict[str, Any]]:

    extra_args = extra_args or []

    if is_existing_text(final_md_path):
        return True, "exists", {"final_md_path": str(final_md_path)}

    if not pdf_path.exists():
        return False, "pdf_missing", {"pdf_path": str(pdf_path)}

    raw_output_dir.mkdir(parents=True, exist_ok=True)
    final_md_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        mineru_bin,
        "-p", str(pdf_path),
        "-o", str(raw_output_dir),
        "-m", method,
        "-b", backend,
        "-l", lang,
    ] + extra_args

    env = make_subprocess_env(num_threads=subprocess_threads)

    try:
        if stream_logs and not quiet:
            proc = subprocess.run(
                cmd,
                text=True,
                timeout=timeout,
                env=env,
            )
            stdout_tail = ""
            stderr_tail = ""
        else:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                env=env,
            )
            stdout_tail = (proc.stdout or "")[-4000:]
            stderr_tail = (proc.stderr or "")[-4000:]

        extra = {
            "cmd": " ".join(cmd),
            "returncode": proc.returncode,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "raw_output_dir": str(raw_output_dir),
        }

        if proc.returncode != 0:
            return False, f"mineru_failed_returncode_{proc.returncode}", extra

        md_files = find_markdown_files(raw_output_dir)
        best_md = choose_best_markdown(md_files)

        extra["num_markdown_files_found"] = len(md_files)
        extra["best_markdown"] = str(best_md) if best_md else ""

        if best_md is None:
            return False, "mineru_no_markdown_found", extra

        shutil.copy2(best_md, final_md_path)

        if is_existing_text(final_md_path):
            return True, "converted", extra

        return False, "final_markdown_too_short", extra

    except FileNotFoundError:
        return False, "mineru_command_not_found", {"cmd": " ".join(cmd)}
    except subprocess.TimeoutExpired:
        return False, "mineru_timeout", {"cmd": " ".join(cmd)}
    except Exception as e:
        return False, f"mineru_conversion_failed: {repr(e)}", {"cmd": " ".join(cmd)}


# ============================================================
# Progress
# ============================================================

def compute_progress_stats(
    rows: List[Dict[str, Any]],
    route_filter: str,
) -> Dict[str, int]:
    scope = required_scope_for_route(route_filter)

    selected = [
        r for r in rows
        if should_select_record(r, route_filter=route_filter)
    ]

    n_selected = len(selected)

    report_ok = sum(bool(r.get("has_report_mineru_markdown")) for r in selected)
    original_ok = sum(bool(r.get("has_original_paper_mineru_markdown")) for r in selected)

    if scope == "both":
        required_ok = sum(bool(r.get("has_required_mineru_markdowns")) for r in selected)
    elif scope == "original_only":
        required_ok = original_ok
    else:
        required_ok = 0

    return {
        "total_records": len(rows),
        "selected_for_mineru": n_selected,
        "report_md_ok": report_ok,
        "original_md_ok": original_ok,
        "required_ok": required_ok,
    }


def format_progress_line(stats: Dict[str, int], route_filter: str) -> str:
    n = stats["selected_for_mineru"]
    required = stats["required_ok"]
    report = stats["report_md_ok"]
    original = stats["original_md_ok"]

    if n == 0:
        return (
            f"route={route_filter} selected=0/{stats['total_records']} "
            f"required=0 report=0 original=0"
        )

    return (
        f"route={route_filter} selected={n}/{stats['total_records']} "
        f"required={required}/{n} report={report}/{n} original={original}/{n}"
    )


# ============================================================
# Existing output / resume
# ============================================================

def has_required_mineru_outputs(r: Dict[str, Any], route_filter: str) -> bool:
    lp = r.get("local_paths") or {}

    report_md = str(lp.get("report_mineru_markdown_path") or "")
    original_md = str(lp.get("original_paper_mineru_markdown_path") or "")

    report_ok = bool(report_md) and is_existing_text(Path(report_md))
    original_ok = bool(original_md) and is_existing_text(Path(original_md))

    if route_filter == "resolved_real_gap":
        return report_ok and original_ok

    if route_filter == "synthetic_controlled":
        return original_ok

    return False


def merge_prior_mineru_fields(current: Dict[str, Any], prior: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge prior MinerU output fields into the current source record on resume.
    """
    merged = dict(current)

    prior_lp = prior.get("local_paths")
    if isinstance(prior_lp, dict):
        lp = dict(merged.get("local_paths") or {})
        for k, v in prior_lp.items():
            if v:
                lp[k] = v
        merged["local_paths"] = lp

    for key in [
        "mineru_selected_status",
        "has_report_mineru_markdown",
        "has_original_paper_mineru_markdown",
        "has_both_mineru_markdowns",
        "has_required_mineru_markdowns",
    ]:
        if key in prior:
            merged[key] = prior[key]

    return merged


# ============================================================
# Record processing
# ============================================================

def mark_unselected_record(
    record: Dict[str, Any],
    route_filter: str,
) -> Dict[str, Any]:
    r = dict(record)
    primary_route = get_primary_route(r)

    r["mineru_selected_status"] = {
        "selected": False,
        "route_filter": route_filter,
        "primary_route": primary_route,
        "reason": f"primary_route={primary_route} does not match route_filter={route_filter}",
    }

    r["has_report_mineru_markdown"] = bool(r.get("has_report_mineru_markdown", False))
    r["has_original_paper_mineru_markdown"] = bool(r.get("has_original_paper_mineru_markdown", False))
    r["has_both_mineru_markdowns"] = bool(r.get("has_both_mineru_markdowns", False))
    r["has_required_mineru_markdowns"] = False

    return r


def process_selected_record(
    record: Dict[str, Any],
    idx: int,
    mineru_md_dir: Path,
    mineru_raw_dir: Path,
    mineru_bin: str,
    method: str,
    backend: str,
    lang: str,
    timeout: int,
    extra_args: List[str],
    stream_logs: bool,
    quiet: bool,
    subprocess_threads: int,
    route_filter: str,
) -> Dict[str, Any]:

    r = dict(record)
    local_paths = dict(r.get("local_paths") or {})
    rid = get_record_id(r, idx)
    primary_route = get_primary_route(r)
    scope = required_scope_for_route(route_filter)

    status = {}

    report_pdf = local_paths.get("report_pdf_path") or ""
    original_pdf = local_paths.get("original_paper_pdf_path") or ""

    report_md = mineru_md_dir / "reports" / f"{rid}.md"
    original_md = mineru_md_dir / "originals" / f"{rid}.md"

    report_raw = mineru_raw_dir / "reports" / rid
    original_raw = mineru_raw_dir / "originals" / rid

    # ------------------------------
    # Report conversion
    # resolved_real_gap only
    # ------------------------------
    if scope == "both":
        if report_pdf:
            ok, msg, extra = convert_with_mineru(
                pdf_path=Path(report_pdf),
                final_md_path=report_md,
                raw_output_dir=report_raw,
                mineru_bin=mineru_bin,
                method=method,
                backend=backend,
                lang=lang,
                timeout=timeout,
                extra_args=extra_args,
                stream_logs=stream_logs,
                quiet=quiet,
                subprocess_threads=subprocess_threads,
            )
            status["report_mineru_conversion"] = {
                "ok": ok,
                "message": msg,
                "extra": extra,
            }
        else:
            status["report_mineru_conversion"] = {
                "ok": False,
                "message": "missing_report_pdf_path",
                "extra": {},
            }
    else:
        status["report_mineru_conversion"] = {
            "ok": False,
            "message": f"skipped_for_route_filter_{route_filter}",
            "extra": {},
        }

    # ------------------------------
    # Original paper conversion
    # both resolved_real_gap and synthetic_controlled
    # ------------------------------
    if original_pdf:
        ok, msg, extra = convert_with_mineru(
            pdf_path=Path(original_pdf),
            final_md_path=original_md,
            raw_output_dir=original_raw,
            mineru_bin=mineru_bin,
            method=method,
            backend=backend,
            lang=lang,
            timeout=timeout,
            extra_args=extra_args,
            stream_logs=stream_logs,
            quiet=quiet,
            subprocess_threads=subprocess_threads,
        )
        status["original_paper_mineru_conversion"] = {
            "ok": ok,
            "message": msg,
            "extra": extra,
        }
    else:
        status["original_paper_mineru_conversion"] = {
            "ok": False,
            "message": "missing_original_pdf_path",
            "extra": {},
        }

    # ------------------------------
    # Save paths
    # ------------------------------
    local_paths["report_mineru_markdown_path"] = str(report_md) if is_existing_text(report_md) else ""
    local_paths["original_paper_mineru_markdown_path"] = str(original_md) if is_existing_text(original_md) else ""

    local_paths["report_mineru_raw_output_dir"] = str(report_raw) if report_raw.exists() else ""
    local_paths["original_paper_mineru_raw_output_dir"] = str(original_raw) if original_raw.exists() else ""

    r["local_paths"] = local_paths

    r["mineru_selected_status"] = {
        "selected": True,
        "route_filter": route_filter,
        "primary_route": primary_route,
        "required_scope": scope,
        "status": status,
    }

    r["has_report_mineru_markdown"] = bool(
        local_paths.get("report_mineru_markdown_path")
        and is_existing_text(Path(local_paths["report_mineru_markdown_path"]))
    )

    r["has_original_paper_mineru_markdown"] = bool(
        local_paths.get("original_paper_mineru_markdown_path")
        and is_existing_text(Path(local_paths["original_paper_mineru_markdown_path"]))
    )

    r["has_both_mineru_markdowns"] = (
        r["has_report_mineru_markdown"]
        and r["has_original_paper_mineru_markdown"]
    )

    if route_filter == "resolved_real_gap":
        r["has_required_mineru_markdowns"] = r["has_both_mineru_markdowns"]
    elif route_filter == "synthetic_controlled":
        r["has_required_mineru_markdowns"] = r["has_original_paper_mineru_markdown"]
    else:
        r["has_required_mineru_markdowns"] = False

    return r


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", required=True)
    parser.add_argument("--out_path", required=True)

    parser.add_argument(
        "--source_filter",
        default="all",
        choices=["all", "OpenReview_MLRC", "OpenReview_TMLR"],
        help="Which source to process. Use OpenReview_MLRC, OpenReview_TMLR, or all.",
    )

    parser.add_argument(
        "--dataset_tag",
        default="",
        help="Optional dataset tag for output directories, e.g., mlrc, tmlr, mixed. If omitted, inferred.",
    )

    parser.add_argument(
        "--mineru_md_dir",
        default="",
        help="Output directory for selected MinerU markdown files. If omitted, inferred from dataset_tag and route_filter.",
    )

    parser.add_argument(
        "--mineru_raw_dir",
        default="",
        help="Output directory for raw MinerU outputs. If omitted, inferred from dataset_tag and route_filter.",
    )

    parser.add_argument("--mineru_bin", default="mineru")
    parser.add_argument("--mineru_method", default="txt", choices=["auto", "txt", "ocr"])
    parser.add_argument("--mineru_backend", default="pipeline")
    parser.add_argument("--mineru_lang", default="en")
    parser.add_argument("--timeout", type=int, default=3600)

    parser.add_argument(
        "--stream_logs",
        action="store_true",
        help="Print MinerU logs directly. Not recommended because onnxruntime may print noisy affinity warnings.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        default=True,
        help="Capture MinerU stdout/stderr and only print progress. Default: true.",
    )

    parser.add_argument("--mineru_extra_args", nargs="*", default=[])

    parser.add_argument(
        "--route_filter",
        required=True,
        choices=["resolved_real_gap", "synthetic_controlled"],
        help=(
            "Which primary_route to convert with MinerU. "
            "resolved_real_gap converts report+original. "
            "synthetic_controlled converts original only."
        ),
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel MinerU record workers. Default: 4.",
    )

    parser.add_argument(
        "--subprocess_threads",
        type=int,
        default=1,
        help="Thread env vars per MinerU subprocess. Default: 1.",
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Checkpoint every N completed selected records. Default: 1.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing --out_path file. Skips records whose required MinerU markdown already exists.",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    out_path = Path(args.out_path)

    if args.source_filter not in VALID_SOURCE_FILTERS:
        raise ValueError(f"Unsupported source_filter: {args.source_filter}")

    rows_all = load_jsonl(input_path)
    rows = [r for r in rows_all if source_matches(r, args.source_filter)]

    dataset_tag = args.dataset_tag.strip() or infer_dataset_tag(rows, args.source_filter)

    if args.mineru_md_dir:
        mineru_md_dir = Path(args.mineru_md_dir)
    else:
        mineru_md_dir = Path(f"Real_bench/{dataset_tag}/markdown_mineru_{args.route_filter}")

    if args.mineru_raw_dir:
        mineru_raw_dir = Path(args.mineru_raw_dir)
    else:
        mineru_raw_dir = Path(f"Real_bench/{dataset_tag}/mineru_raw_outputs_{args.route_filter}")

    processed_map: Dict[str, Dict[str, Any]] = {}

    if args.resume and out_path.exists():
        prior_rows = safe_load_jsonl_if_exists(out_path)
        for i, r in enumerate(prior_rows):
            rid = str(r.get("record_id") or f"prior_{i:05d}").strip()
            processed_map[rid] = r

        if prior_rows:
            init_stats = compute_progress_stats(prior_rows, route_filter=args.route_filter)
            print("[resume] loaded checkpoint:")
            print(f"  {format_progress_line(init_stats, route_filter=args.route_filter)}")

    # Build output baseline.
    all_records_by_id: Dict[str, Dict[str, Any]] = {}
    all_order: List[str] = []

    for idx, r in enumerate(rows, start=1):
        rid = str(r.get("record_id") or f"record_{idx:05d}").strip()
        all_order.append(rid)

        if rid in processed_map:
            all_records_by_id[rid] = merge_prior_mineru_fields(r, processed_map[rid])
        else:
            all_records_by_id[rid] = dict(r)

    # Mark unselected records once, so output file remains complete.
    for rid in all_order:
        rec = all_records_by_id[rid]
        if not should_select_record(rec, route_filter=args.route_filter):
            all_records_by_id[rid] = mark_unselected_record(
                rec,
                route_filter=args.route_filter,
            )

    # Select records that need conversion.
    tasks = []
    skipped_existing = 0

    for idx, rid in enumerate(all_order, start=1):
        rec = all_records_by_id[rid]

        if not should_select_record(rec, route_filter=args.route_filter):
            continue

        if args.resume and has_required_mineru_outputs(rec, route_filter=args.route_filter):
            skipped_existing += 1
            continue

        tasks.append((idx, rid, rec))

    # Save initial state.
    save_jsonl([all_records_by_id[rid] for rid in all_order], out_path)

    selected_total = sum(
        1 for rid in all_order
        if should_select_record(all_records_by_id[rid], route_filter=args.route_filter)
    )

    print("\n===== Step 4 MinerU Parallel Start =====")
    print(f"Source filter: {args.source_filter}")
    print(f"Dataset tag: {dataset_tag}")
    print(f"Route filter: {args.route_filter}")
    print(f"Input records before source filter: {len(rows_all)}")
    print(f"Input records after source filter: {len(rows)}")
    print(f"Selected records: {selected_total}")
    print(f"Already complete skipped by resume: {skipped_existing}")
    print(f"Remaining conversion tasks: {len(tasks)}")
    print(f"num_workers: {args.num_workers}")
    print(f"mineru_method: {args.mineru_method}")
    print(f"mineru_backend: {args.mineru_backend}")
    print(f"mineru_md_dir: {mineru_md_dir}")
    print(f"mineru_raw_dir: {mineru_raw_dir}")
    print(f"quiet: {args.quiet}")
    print("========================================\n")

    completed_since_save = 0

    with ThreadPoolExecutor(max_workers=max(args.num_workers, 1)) as executor:
        futures = {}

        for idx, rid, rec in tasks:
            fut = executor.submit(
                process_selected_record,
                record=rec,
                idx=idx,
                mineru_md_dir=mineru_md_dir,
                mineru_raw_dir=mineru_raw_dir,
                mineru_bin=args.mineru_bin,
                method=args.mineru_method,
                backend=args.mineru_backend,
                lang=args.mineru_lang,
                timeout=args.timeout,
                extra_args=args.mineru_extra_args,
                stream_logs=args.stream_logs,
                quiet=args.quiet,
                subprocess_threads=args.subprocess_threads,
                route_filter=args.route_filter,
            )
            futures[fut] = (idx, rid)

        pbar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Step 4 MinerU parallel source={args.source_filter} route={args.route_filter}",
        )

        for fut in pbar:
            idx, rid = futures[fut]

            try:
                out = fut.result()
            except Exception as e:
                out = dict(all_records_by_id[rid])
                out["mineru_selected_status"] = {
                    "selected": True,
                    "route_filter": args.route_filter,
                    "primary_route": get_primary_route(out),
                    "error": f"worker_failed: {repr(e)}",
                }
                out["has_report_mineru_markdown"] = False
                out["has_original_paper_mineru_markdown"] = False
                out["has_both_mineru_markdowns"] = False
                out["has_required_mineru_markdowns"] = False

            all_records_by_id[rid] = out
            completed_since_save += 1

            stats = compute_progress_stats(
                [all_records_by_id[x] for x in all_order],
                route_filter=args.route_filter,
            )
            pbar.set_postfix_str(
                format_progress_line(stats, route_filter=args.route_filter),
                refresh=True,
            )

            if completed_since_save >= max(args.save_every, 1):
                save_jsonl([all_records_by_id[x] for x in all_order], out_path)
                completed_since_save = 0

    save_jsonl([all_records_by_id[rid] for rid in all_order], out_path)

    final_rows = [all_records_by_id[rid] for rid in all_order]
    final = compute_progress_stats(final_rows, route_filter=args.route_filter)

    print("\n===== Step 4 Summary =====")
    print(f"Source filter: {args.source_filter}")
    print(f"Dataset tag: {dataset_tag}")
    print(f"Route filter: {args.route_filter}")
    print(f"Input records before source filter: {len(rows_all)}")
    print(f"Input records after source filter: {len(rows)}")
    print(f"Selected for MinerU: {final['selected_for_mineru']}")
    print(f"Required MinerU markdown success: {final['required_ok']} / {final['selected_for_mineru']}")
    print(f"Report MinerU markdown: {final['report_md_ok']} / {final['selected_for_mineru']}")
    print(f"Original MinerU markdown: {final['original_md_ok']} / {final['selected_for_mineru']}")
    print(f"Progress: {format_progress_line(final, route_filter=args.route_filter)}")
    print(f"Output written to: {args.out_path}")


if __name__ == "__main__":
    main()
