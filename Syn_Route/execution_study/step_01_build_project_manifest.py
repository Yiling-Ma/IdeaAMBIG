from __future__ import annotations

import argparse
import csv
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common import (
    add_common_args,
    choose_codebase_dir,
    choose_idea_file,
    choose_paper_file,
    compact_reviews_text,
    collect_candidate_project_dirs,
    ensure_dir,
    extract_idea_fields,
    find_extracted_data_root,
    group_reviews_by_project,
    maybe_limit,
    read_text,
    save_json,
    save_jsonl,
    stable_id,
    summarize_codebase,
)


SOURCE_URLS = {
    "paper": "https://openreview.net/forum?id=Fllp8l6Puy",
    "repository": "https://github.com/NoviScl/AI-Researcher",
}


def parse_paper_number(path: Path) -> Optional[int]:
    """Extract N from Paper#N.pdf."""
    match = re.search(r"#(\d+)", path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def stable_execution_project_id(num: int | str) -> str:
    """Create a stable project ID based on the execution-study paper number."""
    try:
        n = int(num)
        return f"ai_researcher_execution_{n:03d}"
    except Exception:
        return f"ai_researcher_execution_{stable_id(str(num))}"


def infer_project_id(project_dir: Path, reviews: Dict[str, List[Dict[str, Any]]]) -> str:
    """
    Fallback ID inference for non-flat directory layouts.

    The primary PaperSubmissions2 layout should use stable_execution_project_id().
    This function is only used if the release data has a project-directory layout.
    """
    name = project_dir.name.strip()
    if name in reviews:
        return name

    for idea_id in reviews:
        if idea_id and (idea_id in name or name in idea_id):
            return idea_id

    return name or stable_id(str(project_dir))


def load_title_mapping(repo_dir: Path) -> List[Dict[str, str]]:
    """
    Load ideation ID-to-title mapping from the AI-Researcher repository.

    This mapping is useful for linking execution projects to ideation IDs and
    reviews, but it should NOT be used as the stable project_id.
    """
    path = repo_dir / "reviews_ideation" / "id_title_mapping.csv"
    if not path.exists():
        return []

    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "ID": row.get("ID", "") or "",
                    "Title": row.get("Title / Filename", "") or "",
                }
            )
    return rows


def title_from_idea(text: str) -> str:
    """Extract a plausible idea title from an idea document."""
    if not text:
        return ""

    match = re.search(
        r"\bTitle\s*:\s*(.+?)(?:\n| 1\.| Problem Statement:)",
        text,
        re.I | re.S,
    )
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()

    first = re.sub(r"\s+", " ", text).strip()
    return first[:160]


def norm_title(text: str) -> str:
    """Normalize a title for fuzzy matching."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def jaccard_score(a: str, b: str) -> float:
    a_tokens = set(norm_title(a).split())
    b_tokens = set(norm_title(b).split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))


def match_mapping(title: str, mapping: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Match an idea title to the ideation ID mapping.

    The result is advisory metadata only. It should not define project_id.
    """
    nt = norm_title(title)
    best: Dict[str, Any] = {"ID": "", "Title": "", "score": 0.0}

    if not nt:
        return best

    for row in mapping:
        mapped_title = row.get("Title", "")
        mt = norm_title(mapped_title)
        if not mt:
            continue

        score = jaccard_score(title, mapped_title)

        map_tokens = set(mt.split())
        title_tokens = set(nt.split())

        if (
            mapped_title.strip().endswith(".json")
            and len(map_tokens) >= 2
            and map_tokens.issubset(title_tokens)
        ):
            score = max(score, 0.90)

        if mt in nt or nt in mt:
            score = max(score, 0.95)

        if score > float(best["score"]):
            best = {
                "ID": row.get("ID", "") or "",
                "Title": mapped_title,
                "score": float(score),
            }

    if float(best["score"]) >= 0.55:
        return best

    return {"ID": "", "Title": "", "score": 0.0}


def infer_idea_source_from_mapping_id(mapped_id: str) -> str:
    """Infer AI/Human source from mapped ideation ID when available."""
    if mapped_id.endswith("_AI"):
        return "AI"
    if mapped_id.endswith("_Human"):
        return "Human"
    return "unknown"


def get_reviews_for_project(
    reviews: Dict[str, List[Dict[str, Any]]],
    project_id: str,
    mapped_id: str,
) -> List[Dict[str, Any]]:
    """
    Retrieve execution reviews.

    Reviews are usually keyed by ideation IDs, not by our stable project IDs.
    Therefore we try mapped_id first, then project_id.
    """
    if mapped_id and mapped_id in reviews:
        return reviews[mapped_id]
    if project_id in reviews:
        return reviews[project_id]
    return []


def extract_code_zip(zip_path: Path, out_root: Path, resume: bool) -> Path:
    """
    Extract Code#N.zip into out_root/Code#N.

    Extraction failure is recorded implicitly by the lack of code summary later.
    """
    target = out_root / zip_path.stem

    if target.exists() and resume:
        return target

    if target.exists():
        shutil.rmtree(target)

    target.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target)
    except Exception as exc:
        # Keep target path so downstream summary can show that extraction was attempted.
        (target / "_extract_error.txt").write_text(
            f"{type(exc).__name__}: {exc}",
            encoding="utf-8",
        )

    return target


def build_flat_rows(
    data_root: Path,
    repo_dir: Path,
    out_dir: Path,
    reviews: Dict[str, List[Dict[str, Any]]],
    limit: int | None,
    resume: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build manifest rows from the flat PaperSubmissions2 layout.

    Expected files:
      PaperSubmissions2/Idea#N.docx
      PaperSubmissions2/Paper#N.pdf
      PaperSubmissions2/Code#N.zip
    """
    mapping = load_title_mapping(repo_dir)
    flat_dir = data_root / "PaperSubmissions2"

    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    if not flat_dir.exists():
        return rows, skipped

    paper_files = []
    for path in flat_dir.glob("Paper#*.pdf"):
        num = parse_paper_number(path)
        if num is not None:
            paper_files.append((num, path))

    paper_files = sorted(paper_files, key=lambda x: x[0])
    if limit is not None:
        paper_files = paper_files[:limit]

    for num, paper_file in paper_files:
        project_id = stable_execution_project_id(num)

        idea_file = flat_dir / f"Idea#{num}.docx"
        code_zip = flat_dir / f"Code#{num}.zip"

        has_idea_docx = idea_file.exists()
        has_paper_pdf = paper_file.exists()
        has_code_zip = code_zip.exists()
        complete_triple = has_idea_docx and has_paper_pdf and has_code_zip

        idea_text = read_text(idea_file, max_chars=80_000) if has_idea_docx else ""
        original_idea, edited_idea, parsed_idea_source = extract_idea_fields(idea_text)

        if not original_idea:
            original_idea = idea_text
        if not edited_idea:
            edited_idea = idea_text

        title = title_from_idea(edited_idea or original_idea or idea_text)
        match = match_mapping(title, mapping)
        mapped_id = match.get("ID", "") or ""

        idea_source = infer_idea_source_from_mapping_id(mapped_id)
        if idea_source == "unknown" and parsed_idea_source:
            idea_source = parsed_idea_source

        code_dir = (
            extract_code_zip(code_zip, out_dir / "extracted_code", resume=resume)
            if has_code_zip
            else None
        )

        paper_text = read_text(paper_file, max_chars=220_000) if has_paper_pdf else ""
        code_summary = summarize_codebase(code_dir, max_chars=80_000) if code_dir else ""

        review_rows = get_reviews_for_project(
            reviews=reviews,
            project_id=project_id,
            mapped_id=mapped_id,
        )
        reviews_text = compact_reviews_text(review_rows)

        if not (idea_text or paper_text or code_summary):
            skipped.append(
                {
                    "project_id": project_id,
                    "project_number": num,
                    "reason": "no_idea_paper_or_code_text_extracted",
                    "idea_file": str(idea_file),
                    "paper_file": str(paper_file),
                    "code_zip": str(code_zip),
                }
            )
            continue

        rows.append(
            {
                "project_id": project_id,
                "split_group_id": project_id,
                "split_group_type": "project",
                "idea_source": idea_source,
                "original_idea": original_idea,
                "edited_idea": edited_idea,
                "executed_paper_path": str(paper_file) if has_paper_pdf else "",
                "executed_paper_text": paper_text,
                "codebase_path": str(code_dir) if code_dir else "",
                "codebase_summary_text": code_summary,
                "review_paths": [
                    str(repo_dir / "reviews_execution" / "data_points_all_execution.json")
                ]
                if review_rows
                else [],
                "reviews_text": reviews_text,
                "metadata": {
                    "project_number": num,
                    "project_id_source": "stable_paper_submission_number",
                    "idea_file": str(idea_file) if has_idea_docx else "",
                    "paper_file": str(paper_file) if has_paper_pdf else "",
                    "code_zip": str(code_zip) if has_code_zip else "",
                    "complete_triple": complete_triple,
                    "has_idea_docx": has_idea_docx,
                    "has_paper_pdf": has_paper_pdf,
                    "has_code_zip": has_code_zip,
                    "matched_ideation_id": mapped_id,
                    "matched_ideation_title": match.get("Title", ""),
                    "title_match_score": match.get("score", 0.0),
                    "num_execution_reviews": len(review_rows),
                    "flat_layout_dir": str(flat_dir),
                },
                "source_urls": SOURCE_URLS,
                "available_modalities": {
                    "has_original_idea": bool(original_idea),
                    "has_edited_idea": bool(edited_idea),
                    "has_executed_paper": bool(paper_text),
                    "has_codebase": bool(code_summary),
                    "has_reviews": bool(reviews_text),
                },
            }
        )

    return rows, skipped


def build_project_dir_rows(
    data_root: Path,
    repo_dir: Path,
    reviews: Dict[str, List[Dict[str, Any]]],
    limit: int | None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Fallback manifest builder for project-directory layouts.

    This is only used if the expected PaperSubmissions2 flat layout is absent
    or yields no rows.
    """
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    project_dirs = maybe_limit(collect_candidate_project_dirs(data_root), limit)

    for project_dir in project_dirs:
        project_id = infer_project_id(project_dir, reviews)

        paper_file = choose_paper_file(project_dir)
        idea_file = choose_idea_file(project_dir)
        code_dir = choose_codebase_dir(project_dir)

        idea_text = read_text(idea_file, max_chars=80_000) if idea_file else ""
        original_idea, edited_idea, idea_source = extract_idea_fields(idea_text)

        if not original_idea:
            original_idea = idea_text
        if not edited_idea:
            edited_idea = idea_text

        paper_text = read_text(paper_file, max_chars=220_000) if paper_file else ""
        code_summary = summarize_codebase(code_dir, max_chars=80_000) if code_dir else ""

        review_rows = get_reviews_for_project(
            reviews=reviews,
            project_id=project_id,
            mapped_id="",
        )
        reviews_text = compact_reviews_text(review_rows)

        if not (paper_text or code_summary or edited_idea):
            skipped.append(
                {
                    "project_id": project_id,
                    "project_dir": str(project_dir),
                    "reason": "no_paper_code_or_idea_text_extracted",
                }
            )
            continue

        rows.append(
            {
                "project_id": project_id,
                "split_group_id": project_id,
                "split_group_type": "project",
                "idea_source": idea_source or "unknown",
                "original_idea": original_idea,
                "edited_idea": edited_idea,
                "executed_paper_path": str(paper_file) if paper_file else "",
                "executed_paper_text": paper_text,
                "codebase_path": str(code_dir) if code_dir else "",
                "codebase_summary_text": code_summary,
                "review_paths": [
                    str(repo_dir / "reviews_execution" / "data_points_all_execution.json")
                ]
                if review_rows
                else [],
                "reviews_text": reviews_text,
                "metadata": {
                    "project_dir": str(project_dir),
                    "project_id_source": "fallback_project_directory",
                    "num_execution_reviews": len(review_rows),
                    "idea_file": str(idea_file) if idea_file else "",
                    "paper_file": str(paper_file) if paper_file else "",
                    "codebase_dir": str(code_dir) if code_dir else "",
                    "complete_triple": bool(idea_file and paper_file and code_dir),
                    "has_idea_docx": bool(idea_file),
                    "has_paper_pdf": bool(paper_file),
                    "has_code_zip": False,
                },
                "source_urls": SOURCE_URLS,
                "available_modalities": {
                    "has_original_idea": bool(original_idea),
                    "has_edited_idea": bool(edited_idea),
                    "has_executed_paper": bool(paper_text),
                    "has_codebase": bool(code_summary),
                    "has_reviews": bool(reviews_text),
                },
            }
        )

    return rows, skipped


def summarize_manifest(
    manifest_rows: List[Dict[str, Any]],
    skipped: List[Dict[str, Any]],
    raw_dir: Path,
    data_root: Optional[Path],
) -> Dict[str, Any]:
    """Build a manifest summary."""
    modality_counter = Counter()
    idea_source_counter = Counter()
    complete_triples = 0
    missing_idea = 0
    missing_paper = 0
    missing_code = 0
    project_ids = Counter()

    for row in manifest_rows:
        project_ids[row.get("project_id", "")] += 1

        for key, value in row.get("available_modalities", {}).items():
            if value:
                modality_counter[key] += 1

        idea_source_counter[row.get("idea_source", "unknown")] += 1

        metadata = row.get("metadata", {})
        if metadata.get("complete_triple"):
            complete_triples += 1
        if not metadata.get("has_idea_docx"):
            missing_idea += 1
        if not metadata.get("has_paper_pdf"):
            missing_paper += 1
        if not metadata.get("has_code_zip"):
            missing_code += 1

    duplicate_project_ids = {
        project_id: count
        for project_id, count in project_ids.items()
        if project_id and count > 1
    }

    return {
        "input_path": str(raw_dir),
        "data_root": str(data_root) if data_root else None,
        "projects": len(manifest_rows),
        "skipped_projects": len(skipped),
        "complete_triples": complete_triples,
        "missing_idea_docx": missing_idea,
        "missing_paper_pdf": missing_paper,
        "missing_code_zip": missing_code,
        "available_modalities": dict(modality_counter),
        "idea_source_distribution": dict(idea_source_counter),
        "duplicate_project_ids": duplicate_project_ids,
        "project_id_policy": "stable ai_researcher_execution_XXX for PaperSubmissions2 layout",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()

    raw_dir: Path = args.input_path or Path("execution_study/outputs_step0_raw")
    out_dir: Path = args.out_dir

    ensure_dir(out_dir)

    repo_dir = raw_dir / "AI-Researcher"
    data_root = find_extracted_data_root(raw_dir)
    reviews = group_reviews_by_project(repo_dir)

    manifest_rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    if not data_root or not data_root.exists():
        skipped.append(
            {
                "project_id": None,
                "reason": "execution_study_zip_not_available",
                "detail": (
                    "Run step 0 and place/download Execution_Study_Data.zip, "
                    "then rerun with --resume."
                ),
            }
        )
    else:
        manifest_rows, flat_skipped = build_flat_rows(
            data_root=data_root,
            repo_dir=repo_dir,
            out_dir=out_dir,
            reviews=reviews,
            limit=args.limit,
            resume=args.resume,
        )
        skipped.extend(flat_skipped)

        if not manifest_rows:
            fallback_rows, fallback_skipped = build_project_dir_rows(
                data_root=data_root,
                repo_dir=repo_dir,
                reviews=reviews,
                limit=args.limit,
            )
            manifest_rows.extend(fallback_rows)
            skipped.extend(fallback_skipped)

    save_jsonl(manifest_rows, out_dir / "project_manifest.jsonl")
    save_jsonl(skipped, out_dir / "skipped_projects.jsonl")

    summary = summarize_manifest(
        manifest_rows=manifest_rows,
        skipped=skipped,
        raw_dir=raw_dir,
        data_root=data_root,
    )
    save_json(summary, out_dir / "summary.json")

    print(summary)


if __name__ == "__main__":
    main()
