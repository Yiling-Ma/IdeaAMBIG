from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import re
import socket
import shlex
import shutil
import ssl
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_filename(value: Any, max_len: int = 180) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^\w\-.]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] if text else "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def route_is_synthetic_controlled(row: Dict[str, Any]) -> bool:
    route = row.get("llm_route_classification") or {}
    selection = row.get("synthetic_selection") or {}
    return (
        route.get("primary_route") == "synthetic_controlled"
        or selection.get("primary_route") == "synthetic_controlled"
        or selection.get("selection_route") == "synthetic_controlled"
    )


def existing_path(value: Any, base_dir: Path) -> Optional[Path]:
    if not value:
        return None
    path = Path(str(value))
    candidates = [path]
    if not path.is_absolute():
        candidates.append(base_dir / path)
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 50:
            return candidate
    return None


def openreview_pdf_url(url: str) -> Optional[str]:
    parsed = urllib.parse.urlparse(url)
    if "openreview.net" not in parsed.netloc:
        return None
    query = urllib.parse.parse_qs(parsed.query)
    ids = query.get("id")
    if not ids:
        return None
    return f"https://openreview.net/pdf?id={ids[0]}"


def arxiv_pdf_url(url: str) -> Optional[str]:
    parsed = urllib.parse.urlparse(url)
    if "arxiv.org" not in parsed.netloc:
        return None
    path = parsed.path
    if path.startswith("/pdf/"):
        paper_id = path[len("/pdf/") :].replace(".pdf", "")
        return f"https://arxiv.org/pdf/{paper_id}"
    if path.startswith("/abs/"):
        paper_id = path[len("/abs/") :]
        return f"https://arxiv.org/pdf/{paper_id}"
    return None


def candidate_pdf_urls(row: Dict[str, Any]) -> List[str]:
    urls: List[str] = []
    for key in ["original_paper_pdf_url", "original_paper_url"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            urls.append(value.strip())

    expanded: List[str] = []
    for url in urls:
        expanded.append(url)
        for fn in [openreview_pdf_url, arxiv_pdf_url]:
            derived = fn(url)
            if derived:
                expanded.insert(0, derived)

    deduped: List[str] = []
    seen = set()
    for url in expanded:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


def download_url(url: str, output_path: Path, timeout: int, retries: int, sleep: float) -> Tuple[bool, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error = ""

    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read()

            if len(data) < 200:
                raise ValueError(f"downloaded_file_too_small:{len(data)}")

            if not data.lstrip().startswith(b"%PDF"):
                preview = data[:80].decode("utf-8", errors="ignore").replace("\n", " ")
                raise ValueError(f"downloaded_content_is_not_pdf:{preview}")

            tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
            tmp_path.write_bytes(data)
            tmp_path.replace(output_path)
            return True, ""

        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            ValueError,
            http.client.IncompleteRead,
            http.client.HTTPException,
            socket.timeout,
            ssl.SSLError,
            OSError,
        ) as exc:
            last_error = repr(exc)
            if attempt < retries:
                time.sleep(sleep * attempt)

    return False, last_error


def find_mineru_markdown(mineru_out_dir: Path) -> Optional[Path]:
    markdown_files = [
        p
        for p in mineru_out_dir.rglob("*.md")
        if p.exists() and p.stat().st_size > 500 and not p.name.startswith(".")
    ]
    if not markdown_files:
        return None
    markdown_files.sort(key=lambda p: p.stat().st_size, reverse=True)
    return markdown_files[0]


def build_mineru_command(args: argparse.Namespace, pdf_path: Path, mineru_out_dir: Path) -> List[str]:
    if args.mineru_args_template:
        rendered = args.mineru_args_template.format(
            pdf=str(pdf_path),
            out_dir=str(mineru_out_dir),
            mode=args.mineru_mode,
        )
        return [args.mineru_cmd] + shlex.split(rendered)

    command_name = Path(args.mineru_cmd).name
    if command_name == "magic-pdf":
        return [
            args.mineru_cmd,
            "-p",
            str(pdf_path),
            "-o",
            str(mineru_out_dir),
            "-m",
            args.mineru_mode,
        ]

    return [
        args.mineru_cmd,
        "-p",
        str(pdf_path),
        "-o",
        str(mineru_out_dir),
        "-m",
        args.mineru_mode,
    ]


def convert_pdf_to_mineru_markdown(
    pdf_path: Path,
    markdown_path: Path,
    mineru_out_dir: Path,
    title: str,
    record_id: str,
    args: argparse.Namespace,
) -> Tuple[bool, str]:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    mineru_out_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which(args.mineru_cmd) is None and not Path(args.mineru_cmd).exists():
        return False, f"mineru_command_not_found:{args.mineru_cmd}"

    if args.clean_mineru_raw and mineru_out_dir.exists():
        shutil.rmtree(mineru_out_dir)
        mineru_out_dir.mkdir(parents=True, exist_ok=True)

    command = build_mineru_command(args, pdf_path=pdf_path, mineru_out_dir=mineru_out_dir)
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=args.mineru_timeout,
        )
        mineru_markdown_path = find_mineru_markdown(mineru_out_dir)
        if not mineru_markdown_path:
            return False, "mineru_markdown_not_found"

        text = mineru_markdown_path.read_text(encoding="utf-8", errors="ignore").strip()
        if len(text) < 500:
            return False, "mineru_markdown_too_short"

        heading = title.strip() or record_id
        if text.lstrip().startswith("#"):
            markdown = f"<!-- record_id: {record_id} -->\n<!-- mineru_source: {mineru_markdown_path} -->\n\n{text}\n"
        else:
            markdown = (
                f"# {heading}\n\n"
                f"<!-- record_id: {record_id} -->\n"
                f"<!-- mineru_source: {mineru_markdown_path} -->\n\n"
                f"{text}\n"
            )
        markdown_path.write_text(markdown, encoding="utf-8")
        return True, ""
    except subprocess.TimeoutExpired:
        return False, f"mineru_timeout_after_seconds:{args.mineru_timeout}"
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout
        return False, f"mineru_failed:{detail[:800]}"


def copy_existing_markdown(source_path: Path, markdown_path: Path, title: str, record_id: str) -> Tuple[bool, str]:
    try:
        text = source_path.read_text(encoding="utf-8", errors="ignore").strip()
        if len(text) < 500:
            return False, "existing_markdown_too_short"
        if not text.lstrip().startswith("#"):
            heading = title.strip() or record_id
            text = f"# {heading}\n\n<!-- record_id: {record_id} -->\n\n{text}"
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(text.rstrip() + "\n", encoding="utf-8")
        return True, ""
    except Exception as exc:
        return False, repr(exc)


def process_one(row: Dict[str, Any], args: argparse.Namespace, base_dir: Path) -> Dict[str, Any]:
    record_id = row.get("record_id") or safe_filename(row.get("report_url")) or "unknown"
    safe_id = safe_filename(record_id)
    title = row.get("original_paper_title") or row.get("report_title") or record_id

    pdf_path = args.out_dir / "pdfs" / f"{safe_id}.pdf"
    markdown_path = args.out_dir / "markdown" / f"{safe_id}.md"
    mineru_out_dir = args.out_dir / "mineru_raw" / safe_id

    if args.resume and markdown_path.exists() and markdown_path.stat().st_size > 500:
        existing_text = markdown_path.read_text(encoding="utf-8", errors="ignore")[:2000]
        if "mineru_source:" not in existing_text and "<!-- mineru" not in existing_text:
            # Existing text from an older fast-text run should not satisfy MinerU resume.
            pass
        else:
            return {
                "record_id": record_id,
                "status": "skipped_mineru_markdown_exists",
                "original_paper_markdown_path": str(markdown_path),
                "original_paper_pdf_path": str(pdf_path) if pdf_path.exists() else "",
                "source_url_used": "",
                "error": "",
            }

    local_paths = row.get("local_paths") or {}
    existing_md = existing_path(local_paths.get("original_paper_mineru_markdown_path"), base_dir)
    if existing_md:
        ok, err = copy_existing_markdown(existing_md, markdown_path, title=title, record_id=record_id)
        if ok:
            text = markdown_path.read_text(encoding="utf-8", errors="ignore")
            if "mineru_source:" not in text[:2000]:
                markdown_path.write_text(
                    f"<!-- record_id: {record_id} -->\n<!-- mineru_source: {existing_md} -->\n\n{text.rstrip()}\n",
                    encoding="utf-8",
                )
            return {
                "record_id": record_id,
                "status": "copied_existing_mineru_markdown",
                "original_paper_markdown_path": str(markdown_path),
                "original_paper_pdf_path": str(pdf_path) if pdf_path.exists() else "",
                "source_markdown_path": str(existing_md),
                "source_url_used": "",
                "error": "",
            }
        return {
            "record_id": record_id,
            "status": "failed",
            "original_paper_markdown_path": "",
            "original_paper_pdf_path": "",
            "source_markdown_path": str(existing_md),
            "source_url_used": "",
            "error": err,
        }

    existing_pdf = existing_path(local_paths.get("original_paper_pdf_path"), base_dir)
    if existing_pdf and (not pdf_path.exists() or pdf_path.stat().st_size <= 200):
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(existing_pdf, pdf_path)

    source_url_used = ""
    if not pdf_path.exists() or pdf_path.stat().st_size <= 200:
        errors = []
        for url in candidate_pdf_urls(row):
            ok, err = download_url(
                url=url,
                output_path=pdf_path,
                timeout=args.timeout,
                retries=args.retries,
                sleep=args.retry_sleep,
            )
            if ok:
                source_url_used = url
                break
            errors.append({"url": url, "error": err})

        if not pdf_path.exists() or pdf_path.stat().st_size <= 200:
            return {
                "record_id": record_id,
                "status": "failed",
                "original_paper_markdown_path": "",
                "original_paper_pdf_path": "",
                "source_url_used": "",
                "download_errors": errors,
                "error": "original_paper_pdf_download_failed",
            }

    ok, err = convert_pdf_to_mineru_markdown(
        pdf_path=pdf_path,
        markdown_path=markdown_path,
        mineru_out_dir=mineru_out_dir,
        title=title,
        record_id=record_id,
        args=args,
    )
    if not ok:
        return {
            "record_id": record_id,
            "status": "failed",
            "original_paper_markdown_path": "",
            "original_paper_pdf_path": str(pdf_path),
            "mineru_raw_dir": str(mineru_out_dir),
            "source_url_used": source_url_used,
            "error": err,
        }

    return {
        "record_id": record_id,
        "status": "saved",
        "original_paper_markdown_path": str(markdown_path),
        "original_paper_pdf_path": str(pdf_path),
        "mineru_raw_dir": str(mineru_out_dir),
        "source_url_used": source_url_used,
        "pdf_sha256": sha256_file(pdf_path),
        "markdown_num_chars": markdown_path.stat().st_size,
        "error": "",
    }


def build_updated_candidate(row: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    local_paths = dict(out.get("local_paths") or {})
    if result.get("original_paper_pdf_path"):
        local_paths["original_paper_pdf_path"] = result["original_paper_pdf_path"]
    if result.get("original_paper_markdown_path"):
        local_paths["original_paper_mineru_markdown_path"] = result["original_paper_markdown_path"]
    out["local_paths"] = local_paths
    out["paper_markdown_preparation"] = {
        "status": result.get("status"),
        "error": result.get("error", ""),
        "source_url_used": result.get("source_url_used", ""),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--mineru_cmd", default="magic-pdf")
    parser.add_argument("--mineru_mode", default="auto")
    parser.add_argument(
        "--mineru_args_template",
        default="",
        help=(
            "Optional argument template for custom MinerU commands. "
            "Available placeholders: {pdf}, {out_dir}, {mode}. "
            "Example: '-p {pdf} -o {out_dir} -m {mode}'"
        ),
    )
    parser.add_argument("--mineru_timeout", type=int, default=900)
    parser.add_argument("--clean_mineru_raw", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pdfs").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "markdown").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "mineru_raw").mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.input_path)
    rows = [row for row in rows if route_is_synthetic_controlled(row)]
    rows = sorted(rows, key=lambda x: str(x.get("record_id") or ""))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    # Candidate paths in existing metadata are usually relative to the repo root.
    base_dir = args.input_path.resolve().parents[2]

    print("\n===== Step R0: Prepare Original Paper Markdown =====")
    print(f"Input path: {args.input_path}")
    print(f"Output dir: {args.out_dir}")
    print(f"Candidates: {len(rows)}")
    print(f"Resume: {args.resume}")
    print("====================================================\n")

    index_rows: List[Dict[str, Any]] = []
    failed_rows: List[Dict[str, Any]] = []
    updated_candidates: List[Dict[str, Any]] = []

    for row in tqdm(rows, desc="Prepare original paper markdown"):
        result = process_one(row, args=args, base_dir=base_dir)
        index_rows.append(result)
        updated_candidates.append(build_updated_candidate(row, result))
        if result.get("status") == "failed":
            failed_rows.append(result)

        save_jsonl(index_rows, args.out_dir / "index.jsonl")
        save_jsonl(failed_rows, args.out_dir / "failed.jsonl")
        save_jsonl(updated_candidates, args.out_dir / "synthetic_controlled_candidates_with_markdown.jsonl")

    status_counts = Counter(row.get("status", "unknown") for row in index_rows)
    success_statuses = {"saved", "copied_existing_mineru_markdown", "skipped_mineru_markdown_exists"}
    successful = sum(count for status, count in status_counts.items() if status in success_statuses)
    failed = status_counts.get("failed", 0)

    inventory = {
        "input_path": str(args.input_path),
        "out_dir": str(args.out_dir),
        "num_candidates": len(rows),
        "status_counts": dict(status_counts),
        "pdf_dir": str(args.out_dir / "pdfs"),
        "markdown_dir": str(args.out_dir / "markdown"),
        "index_path": str(args.out_dir / "index.jsonl"),
        "failed_path": str(args.out_dir / "failed.jsonl"),
        "updated_candidates_path": str(args.out_dir / "synthetic_controlled_candidates_with_markdown.jsonl"),
        "conversion_tool": shutil.which(args.mineru_cmd) or args.mineru_cmd,
        "conversion_method": "mineru",
        "mineru_cmd": args.mineru_cmd,
        "mineru_mode": args.mineru_mode,
        "mineru_raw_dir": str(args.out_dir / "mineru_raw"),
    }
    summary = {
        "input_candidates": len(rows),
        "successful_markdown": successful,
        "failed": failed,
        "status_counts": dict(status_counts),
        "recommended_next_step_input": str(args.out_dir / "synthetic_controlled_candidates_with_markdown.jsonl"),
    }

    save_json(inventory, args.out_dir / "dataset_inventory.json")
    save_json(summary, args.out_dir / "summary.json")

    print("\n===== Step R0 Summary =====")
    print(f"Input candidates: {len(rows)}")
    print(f"Successful markdown: {successful}")
    print(f"Failed: {failed}")
    print(f"Index: {args.out_dir / 'index.jsonl'}")
    print(f"Updated candidates: {args.out_dir / 'synthetic_controlled_candidates_with_markdown.jsonl'}")
    print(f"Summary: {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
