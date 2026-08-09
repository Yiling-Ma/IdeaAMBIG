from __future__ import annotations

import argparse
import re
from pathlib import Path

from common import (
    DATA_FILE_ID,
    DATA_URL,
    PAPER_URL,
    REPO_URL,
    add_common_args,
    clone_repo,
    ensure_dir,
    file_inventory,
    git_commit,
    save_json,
    summarize_inventory,
    try_download_google_drive,
    unzip_if_needed,
)


def read_excerpt(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    marker = "## Execution Study Data"
    idx = text.find(marker)
    if idx >= 0:
        return text[idx : idx + max_chars]
    return text[:max_chars]


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    ensure_dir(out_dir)
    repo_dir = out_dir / "AI-Researcher"
    zip_path = out_dir / "Execution_Study_Data.zip"
    extract_dir = out_dir / "Execution_Study_Data"

    repo_result = clone_repo(repo_dir, resume=args.resume)
    download_result = try_download_google_drive(DATA_FILE_ID, zip_path)
    unzip_result = unzip_if_needed(zip_path, extract_dir, resume=args.resume)

    readme = repo_dir / "README.md"
    license_path = repo_dir / "LICENSE"
    license_text = license_path.read_text(encoding="utf-8", errors="ignore")[:2000] if license_path.exists() else ""
    readme_excerpt = read_excerpt(readme)

    inventory = {
        "paper_url": PAPER_URL,
        "repository_url": REPO_URL,
        "repository_path": str(repo_dir),
        "repository_commit": git_commit(repo_dir),
        "release_data_url": DATA_URL,
        "release_file_id": DATA_FILE_ID,
        "release_zip_path": str(zip_path),
        "release_extract_path": str(extract_dir),
        "download_result": download_result,
        "unzip_result": unzip_result,
        "license_information": {
            "license_path": str(license_path) if license_path.exists() else None,
            "license_excerpt": license_text,
        },
        "README_excerpts_explaining_data": readme_excerpt,
        "repository_inventory": summarize_inventory(repo_dir) if repo_dir.exists() else {},
        "release_inventory": summarize_inventory(extract_dir) if extract_dir.exists() else {},
    }
    save_json(inventory, out_dir / "dataset_inventory.json")

    summary = {
        "repository": repo_result,
        "download": download_result,
        "unzip": unzip_result,
        "manual_download_required": download_result.get("status") == "manual_download_required",
        "manual_download_instructions": download_result if download_result.get("status") == "manual_download_required" else None,
        "dataset_inventory_path": str(out_dir / "dataset_inventory.json"),
    }
    save_json(summary, out_dir / "download_summary.json")
    print(summary)


if __name__ == "__main__":
    main()

