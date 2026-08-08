import argparse
import csv
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


GITHUB_API = "https://api.github.com"


PAPER_URL_PATTERNS = [
    r"https?://arxiv\.org/(?:abs|pdf)/[0-9]+\.[0-9]+(?:v[0-9]+)?",
    r"https?://openreview\.net/(?:forum|pdf)\?id=[A-Za-z0-9_\-]+",
    r"https?://proceedings\.neurips\.cc/[^)\s\"'>]+",
    r"https?://openaccess\.thecvf\.com/[^)\s\"'>]+",
    r"https?://aclanthology\.org/[^)\s\"'>]+",
    r"https?://proceedings\.mlr\.press/[^)\s\"'>]+",
    r"https?://dl\.acm\.org/doi/[^)\s\"'>]+",
    r"https?://ieeexplore\.ieee\.org/[^)\s\"'>]+",
    r"https?://link\.springer\.com/[^)\s\"'>]+",
]

PAPER_TITLE_PATTERNS = [
    r"(?:paper|title)\s*[:：]\s*(.+)",
    r"#\s*(.+)",
]

OFFICIAL_HINTS = [
    "official implementation",
    "official code",
    "code for the paper",
    "implementation of our paper",
    "this repository contains the code",
    "paper code",
]

REPRO_HINTS = [
    "reproduction",
    "reproduce",
    "reimplementation",
    "replication",
    "unofficial implementation",
]

RESEARCH_HINTS = [
    "arxiv",
    "openreview",
    "neurips",
    "iclr",
    "icml",
    "cvpr",
    "eccv",
    "iccv",
    "acl",
    "emnlp",
    "naacl",
    "sigir",
    "recsys",
    "kdd",
    "www",
    "icde",
    "aaai",
    "ijcai",
    "paper",
]


def build_queries(min_stars: int, min_issues: int) -> List[str]:
    """
    GitHub repository search supports qualifiers such as:
    stars:>=20 archived:false fork:false language:Python

    We deliberately use multiple high-precision queries rather than one broad query.
    """
    base = f"stars:>={min_stars} archived:false fork:false"

    queries = []

    venue_terms = [
        "arxiv",
        "openreview",
        "neurips",
        "iclr",
        "icml",
        "cvpr",
        "eccv",
        "iccv",
        "acl",
        "emnlp",
        "sigir",
        "recsys",
        "kdd",
        "aaai",
    ]

    phrase_terms = [
        '"official implementation"',
        '"paper code"',
        '"code for the paper"',
        '"reproduce paper"',
        '"reproduction"',
        '"reimplementation"',
    ]

    languages = ["Python", "Jupyter Notebook", "C++", "R"]

    for term in venue_terms:
        for lang in languages:
            queries.append(f"{term} {base} language:{lang}")

    for phrase in phrase_terms:
        for lang in languages:
            queries.append(f"{phrase} {base} language:{lang}")

    # Some repositories have no dominant language detected.
    for term in ["arxiv", "openreview", '"official implementation"', '"paper code"']:
        queries.append(f"{term} {base}")

    # GitHub search cannot directly filter by issue count.
    # We apply min_issues after fetching repo metadata.
    return queries


def github_headers(token: Optional[str]) -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "ideaambig-bench-miner",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def request_json(
    url: str,
    headers: Dict[str, str],
    params: Optional[Dict[str, Any]] = None,
    sleep: float = 1.0,
    max_retries: int = 5,
) -> Optional[Dict[str, Any]]:
    for attempt in range(max_retries):
        resp = requests.get(url, headers=headers, params=params, timeout=30)

        if resp.status_code == 200:
            time.sleep(sleep)
            return resp.json()

        # Rate limit or secondary rate limit.
        if resp.status_code in {403, 429}:
            reset_ts = resp.headers.get("X-RateLimit-Reset")
            remaining = resp.headers.get("X-RateLimit-Remaining")
            if remaining == "0" and reset_ts:
                wait_s = max(5, int(reset_ts) - int(time.time()) + 5)
                print(f"[rate-limit] sleeping {wait_s}s")
                time.sleep(wait_s)
            else:
                wait_s = min(60, 5 * (attempt + 1))
                print(f"[secondary-limit] {resp.status_code}; sleeping {wait_s}s")
                time.sleep(wait_s)
            continue

        if resp.status_code >= 500:
            wait_s = min(60, 5 * (attempt + 1))
            print(f"[server-error] {resp.status_code}; sleeping {wait_s}s")
            time.sleep(wait_s)
            continue

        print(f"[warn] request failed: {resp.status_code} {url}")
        try:
            print(resp.json())
        except Exception:
            print(resp.text[:500])
        return None

    print(f"[error] max retries exceeded: {url}")
    return None


def search_repositories(
    query: str,
    headers: Dict[str, str],
    max_repos: int,
    sleep: float,
) -> List[Dict[str, Any]]:
    """
    GitHub Search API returns at most 1000 results per query.
    For pilot mining, we limit max_repos_per_query.
    """
    results = []
    per_page = 100
    max_pages = max(1, (max_repos + per_page - 1) // per_page)

    for page in range(1, max_pages + 1):
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": per_page,
            "page": page,
        }
        data = request_json(
            f"{GITHUB_API}/search/repositories",
            headers=headers,
            params=params,
            sleep=sleep,
        )
        if not data:
            break

        items = data.get("items", [])
        if not items:
            break

        results.extend(items)
        if len(results) >= max_repos:
            break

    return results[:max_repos]


def fetch_readme_text(
    full_name: str,
    headers: Dict[str, str],
    sleep: float,
) -> str:
    """
    Fetch README through GitHub API.
    We request raw content using Accept: raw.
    """
    raw_headers = dict(headers)
    raw_headers["Accept"] = "application/vnd.github.raw"

    url = f"{GITHUB_API}/repos/{full_name}/readme"
    resp = requests.get(url, headers=raw_headers, timeout=30)

    if resp.status_code == 200:
        time.sleep(sleep)
        return resp.text[:200000]

    time.sleep(sleep)
    return ""


def extract_paper_urls(text: str) -> List[str]:
    urls = []
    for pat in PAPER_URL_PATTERNS:
        urls.extend(re.findall(pat, text, flags=re.IGNORECASE))
    # normalize arxiv pdf to abs if needed? keep original for evidence.
    unique = []
    seen = set()
    for u in urls:
        u = u.rstrip(").,;]")
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def guess_paper_title(readme: str, repo_name: str) -> str:
    """
    Lightweight title guess.
    We do not need perfect title here; later steps can repair using arXiv/OpenReview metadata.
    """
    lines = [ln.strip() for ln in readme.splitlines() if ln.strip()]
    candidates = []

    for ln in lines[:80]:
        low = ln.lower()

        # Skip badges, images, links-only lines.
        if "badge" in low or "<img" in low or ln.startswith("!["):
            continue

        for pat in PAPER_TITLE_PATTERNS:
            m = re.search(pat, ln, flags=re.IGNORECASE)
            if m:
                title = m.group(1).strip()
                title = re.sub(r"[*_`#\[\]\(\)]", "", title).strip()
                if 8 <= len(title) <= 180:
                    candidates.append(title)

        # Common markdown style:
        # [Paper Title](https://arxiv.org/abs/...)
        if "arxiv.org" in low or "openreview.net" in low:
            m = re.search(r"\[([^\]]{8,180})\]\(", ln)
            if m:
                title = re.sub(r"[*_`#]", "", m.group(1)).strip()
                candidates.append(title)

    if candidates:
        # Prefer candidates that do not look like generic repo names.
        candidates = sorted(candidates, key=lambda x: (len(x), x), reverse=True)
        return candidates[0]

    return repo_name


def classify_repo_type(text: str) -> str:
    low = text.lower()

    if any(h in low for h in OFFICIAL_HINTS):
        return "official_implementation"

    if any(h in low for h in REPRO_HINTS):
        return "reproduction_or_reimplementation"

    if any(h in low for h in RESEARCH_HINTS):
        return "paper_linked_repo"

    return "unknown"


def compute_score(repo: Dict[str, Any], readme: str, paper_urls: List[str], repo_type: str) -> int:
    """
    Simple ranking score.
    Higher score = more likely useful for issue mining.
    """
    score = 0

    stars = repo.get("stargazers_count") or 0
    issues = repo.get("open_issues_count") or 0
    forks = repo.get("forks_count") or 0

    if stars >= 500:
        score += 5
    elif stars >= 100:
        score += 4
    elif stars >= 20:
        score += 3
    elif stars >= 5:
        score += 1

    if forks >= 50:
        score += 3
    elif forks >= 10:
        score += 2
    elif forks >= 3:
        score += 1

    if issues >= 20:
        score += 4
    elif issues >= 5:
        score += 3
    elif issues >= 1:
        score += 2

    if paper_urls:
        score += 5

    if repo_type == "official_implementation":
        score += 5
    elif repo_type == "reproduction_or_reimplementation":
        score += 4
    elif repo_type == "paper_linked_repo":
        score += 2

    low = readme.lower()
    if "reproduce" in low or "reproduction" in low:
        score += 3
    if "paper" in low:
        score += 2
    if "config" in low or "hyperparameter" in low or "training" in low:
        score += 1

    return score


def detect_evidence_source(repo: Dict[str, Any], readme: str, paper_urls: List[str]) -> str:
    homepage = repo.get("homepage") or ""
    description = repo.get("description") or ""

    if extract_paper_urls(homepage):
        return "homepage"
    if extract_paper_urls(description):
        return "description"
    if paper_urls:
        return "readme"
    return "search_query"


def normalize_repo_record(
    repo: Dict[str, Any],
    query: str,
    readme: str,
) -> Dict[str, Any]:
    full_name = repo.get("full_name", "")
    combined_text = "\n".join([
        repo.get("description") or "",
        repo.get("homepage") or "",
        readme or "",
    ])

    paper_urls = extract_paper_urls(combined_text)
    repo_type = classify_repo_type(combined_text)
    paper_title_guess = guess_paper_title(readme, repo.get("name", full_name))
    evidence_source = detect_evidence_source(repo, readme, paper_urls)
    score = compute_score(repo, readme, paper_urls, repo_type)

    return {
        "repo": full_name,
        "repo_url": repo.get("html_url", ""),
        "owner": (repo.get("owner") or {}).get("login", ""),
        "name": repo.get("name", ""),
        "description": repo.get("description") or "",
        "homepage": repo.get("homepage") or "",
        "language": repo.get("language") or "",
        "stars": repo.get("stargazers_count") or 0,
        "forks": repo.get("forks_count") or 0,
        "open_issues_count": repo.get("open_issues_count") or 0,
        "created_at": repo.get("created_at") or "",
        "updated_at": repo.get("updated_at") or "",
        "pushed_at": repo.get("pushed_at") or "",
        "archived": repo.get("archived", False),
        "fork": repo.get("fork", False),
        "license": ((repo.get("license") or {}).get("spdx_id") or ""),
        "topics": "|".join(repo.get("topics") or []),
        "paper_title_guess": paper_title_guess,
        "paper_url_guess": paper_urls[0] if paper_urls else "",
        "all_paper_urls": "|".join(paper_urls),
        "repo_type_guess": repo_type,
        "evidence_source": evidence_source,
        "score": score,
        "matched_query": query,
    }


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default="Real_bench/github_issue_mining/outputs_step1",
    )
    parser.add_argument("--min_stars", type=int, default=20)
    parser.add_argument("--min_issues", type=int, default=1)
    parser.add_argument("--max_repos_per_query", type=int, default=100)
    parser.add_argument("--max_total_repos", type=int, default=1000)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument(
        "--github_token",
        type=str,
        default=os.environ.get("GITHUB_TOKEN", ""),
        help="Optional. Prefer setting env var GITHUB_TOKEN.",
    )
    parser.add_argument(
        "--fetch_readme",
        action="store_true",
        help="Fetch README for better paper URL/title detection. Recommended.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    token = args.github_token.strip() or None
    headers = github_headers(token)

    queries = build_queries(args.min_stars, args.min_issues)
    print(f"[info] num queries: {len(queries)}")

    seen = set()
    rows: List[Dict[str, Any]] = []

    for qi, query in enumerate(queries, start=1):
        print(f"\n[query {qi}/{len(queries)}] {query}")

        repos = search_repositories(
            query=query,
            headers=headers,
            max_repos=args.max_repos_per_query,
            sleep=args.sleep,
        )
        print(f"[info] fetched repos: {len(repos)}")

        for repo in repos:
            full_name = repo.get("full_name", "")
            if not full_name or full_name in seen:
                continue

            if repo.get("archived") or repo.get("fork"):
                continue

            if (repo.get("stargazers_count") or 0) < args.min_stars:
                continue

            if (repo.get("open_issues_count") or 0) < args.min_issues:
                continue

            seen.add(full_name)

            readme = ""
            if args.fetch_readme:
                readme = fetch_readme_text(
                    full_name=full_name,
                    headers=headers,
                    sleep=args.sleep,
                )

            row = normalize_repo_record(repo, query, readme)

            # Keep only repos with some research signal.
            has_research_signal = (
                bool(row["paper_url_guess"])
                or row["repo_type_guess"] in {
                    "official_implementation",
                    "reproduction_or_reimplementation",
                    "paper_linked_repo",
                }
            )

            if not has_research_signal:
                continue

            rows.append(row)

            if len(rows) >= args.max_total_repos:
                break

        print(f"[info] accumulated unique candidate repos: {len(rows)}")

        if len(rows) >= args.max_total_repos:
            break

    # Sort by score then stars.
    rows = sorted(
        rows,
        key=lambda r: (int(r["score"]), int(r["stars"]), int(r["open_issues_count"])),
        reverse=True,
    )

    csv_path = os.path.join(args.out_dir, "seed_repos_tier1_pilot.csv")
    jsonl_path = os.path.join(args.out_dir, "seed_repos_tier1_pilot.jsonl")

    write_csv(csv_path, rows)
    write_jsonl(jsonl_path, rows)

    print("\n[done]")
    print(f"num repos: {len(rows)}")
    print(f"csv:   {csv_path}")
    print(f"jsonl: {jsonl_path}")

    if rows:
        print("\n[top 10]")
        for i, r in enumerate(rows[:10], start=1):
            print(
                f"{i:02d}. {r['repo']} | stars={r['stars']} | "
                f"issues={r['open_issues_count']} | type={r['repo_type_guess']} | "
                f"score={r['score']} | paper={r['paper_url_guess'][:80]}"
            )


if __name__ == "__main__":
    main()
