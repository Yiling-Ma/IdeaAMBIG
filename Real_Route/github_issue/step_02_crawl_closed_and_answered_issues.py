import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


GITHUB_API = "https://api.github.com"


POSITIVE_SPEC_GAP_KEYWORDS = [
    # paper / reproduction signal
    "paper",
    "implementation",
    "implement",
    "reproduce",
    "reproduction",
    "replicate",
    "replication",
    "reproducing",
    "result",
    "results",
    "table",
    "figure",
    "equation",
    "algorithm",
    "appendix",

    # missing / ambiguity signal
    "not specified",
    "unspecified",
    "unclear",
    "ambiguous",
    "missing",
    "not mentioned",
    "does not mention",
    "could you clarify",
    "clarify",
    "how did you",
    "what value",
    "which setting",
    "which config",
    "what is the",
    "where is the",
    "inconsistent",
    "does not match",
    "different from the paper",
    "mismatch",

    # implementation-critical slots
    "config",
    "configuration",
    "hyperparameter",
    "hyperparameters",
    "learning rate",
    "lr",
    "batch size",
    "epoch",
    "epochs",
    "seed",
    "split",
    "dataset split",
    "preprocess",
    "preprocessing",
    "normalization",
    "normalize",
    "loss",
    "objective",
    "architecture",
    "layer",
    "dimension",
    "hidden size",
    "checkpoint",
    "training",
    "train",
    "inference",
    "evaluation",
    "evaluate",
    "metric",
    "metrics",
    "baseline",
    "ablation",
]

NEGATIVE_ENGINEERING_KEYWORDS = [
    "install",
    "installation",
    "pip",
    "conda",
    "importerror",
    "modulenotfounderror",
    "cuda",
    "gpu memory",
    "out of memory",
    "oom",
    "docker",
    "path error",
    "permission denied",
    "download link",
    "broken link",
    "dataset access",
    "syntax error",
    "version conflict",
    "dependency",
    "requirements.txt",
    "environment",
    "windows",
    "ubuntu",
    "macos",
]

RESOLUTION_KEYWORDS = [
    "fixed",
    "solved",
    "resolved",
    "thanks",
    "thank you",
    "works",
    "worked",
    "correct",
    "you should",
    "please use",
    "we use",
    "we used",
    "try",
    "set",
    "change",
    "update",
    "merged",
    "commit",
    "pull request",
    "pr",
    "see",
]

AUTHOR_ASSOCIATIONS_MAINTAINER = {"OWNER", "MEMBER", "COLLABORATOR"}


COMMIT_URL_RE = re.compile(
    r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/commit/[A-Fa-f0-9]{7,40}"
)

PR_URL_RE = re.compile(
    r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/pull/[0-9]+"
)

ISSUE_URL_RE = re.compile(
    r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/issues/[0-9]+"
)

SHA_RE = re.compile(r"\b[0-9a-f]{7,40}\b", re.IGNORECASE)


def github_headers(token: Optional[str]) -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "ideaambig-bench-issue-crawler",
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
    max_retries: int = 6,
) -> Optional[Any]:
    """
    Robust GitHub GET request with rate-limit handling.
    """
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=40)
        except requests.RequestException as e:
            wait_s = min(60, 5 * (attempt + 1))
            print(f"[request-error] {e}; sleeping {wait_s}s", file=sys.stderr)
            time.sleep(wait_s)
            continue

        if resp.status_code == 200:
            time.sleep(sleep)
            return resp.json()

        # Rate limit or secondary rate limit.
        if resp.status_code in {403, 429}:
            remaining = resp.headers.get("X-RateLimit-Remaining")
            reset_ts = resp.headers.get("X-RateLimit-Reset")

            # Hard rate limit.
            if remaining == "0" and reset_ts:
                wait_s = max(5, int(reset_ts) - int(time.time()) + 5)
                print(f"[rate-limit] sleeping {wait_s}s", file=sys.stderr)
                time.sleep(wait_s)
                continue

            # Secondary rate limit.
            wait_s = min(120, 10 * (attempt + 1))
            print(
                f"[secondary-rate-limit] status={resp.status_code}; sleeping {wait_s}s; url={url}",
                file=sys.stderr,
            )
            time.sleep(wait_s)
            continue

        if resp.status_code in {500, 502, 503, 504}:
            wait_s = min(120, 10 * (attempt + 1))
            print(f"[server-error] {resp.status_code}; sleeping {wait_s}s", file=sys.stderr)
            time.sleep(wait_s)
            continue

        # Not found / forbidden repo / validation error etc.
        print(f"[warn] request failed: status={resp.status_code} url={url}", file=sys.stderr)
        try:
            print(json.dumps(resp.json(), ensure_ascii=False)[:1000], file=sys.stderr)
        except Exception:
            print(resp.text[:1000], file=sys.stderr)
        return None

    print(f"[error] max retries exceeded: {url}", file=sys.stderr)
    return None


def paginate_get(
    url: str,
    headers: Dict[str, str],
    params: Dict[str, Any],
    sleep: float,
    max_items: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Page through GitHub list endpoint.
    """
    all_items: List[Dict[str, Any]] = []
    page = 1
    per_page = int(params.get("per_page", 100))

    while True:
        page_params = dict(params)
        page_params["page"] = page
        page_params["per_page"] = per_page

        data = request_json(url, headers=headers, params=page_params, sleep=sleep)
        if data is None:
            break
        if not isinstance(data, list):
            break
        if not data:
            break

        all_items.extend(data)

        if max_items is not None and len(all_items) >= max_items:
            return all_items[:max_items]

        if len(data) < per_page:
            break

        page += 1

    return all_items


def load_seed_repos(input_csv: str, max_repos: Optional[int]) -> List[Dict[str, str]]:
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    def safe_int(x: Any, default: int = 0) -> int:
        try:
            return int(float(x))
        except Exception:
            return default

    # Step 1 output is already sorted, but sorting again makes the script robust.
    rows = sorted(
        rows,
        key=lambda r: (
            safe_int(r.get("score", 0)),
            safe_int(r.get("stars", 0)),
            safe_int(r.get("open_issues_count", 0)),
        ),
        reverse=True,
    )

    if max_repos is not None and max_repos > 0:
        rows = rows[:max_repos]

    return rows


def compact_user(user_obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not user_obj:
        return {}
    return {
        "login": user_obj.get("login", ""),
        "id": user_obj.get("id", ""),
        "type": user_obj.get("type", ""),
        "site_admin": user_obj.get("site_admin", False),
        "html_url": user_obj.get("html_url", ""),
    }


def compact_label(label_obj: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": label_obj.get("name", ""),
        "description": label_obj.get("description", ""),
        "color": label_obj.get("color", ""),
    }


def extract_links_from_text(text: str) -> Dict[str, List[str]]:
    if not text:
        text = ""

    pr_urls = sorted(set(PR_URL_RE.findall(text)))
    commit_urls = sorted(set(COMMIT_URL_RE.findall(text)))
    issue_urls = sorted(set(ISSUE_URL_RE.findall(text)))

    # Raw SHA mentions can be noisy, so keep them separate.
    sha_mentions = sorted(set(SHA_RE.findall(text)))

    return {
        "pr_urls": pr_urls,
        "commit_urls": commit_urls,
        "issue_urls": issue_urls,
        "sha_mentions": sha_mentions,
    }


def count_keyword_hits(text: str, keywords: List[str]) -> List[str]:
    low = (text or "").lower()
    hits = []
    for kw in keywords:
        if kw.lower() in low:
            hits.append(kw)
    return hits


def issue_text_for_signals(issue: Dict[str, Any], comments: List[Dict[str, Any]]) -> str:
    parts = [
        issue.get("title") or "",
        issue.get("body") or "",
    ]
    for c in comments:
        parts.append(c.get("body") or "")
    return "\n".join(parts)


def summarize_comments(comments_raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    comments = []
    for c in comments_raw:
        comments.append(
            {
                "id": c.get("id"),
                "user": compact_user(c.get("user")),
                "author_association": c.get("author_association", ""),
                "created_at": c.get("created_at", ""),
                "updated_at": c.get("updated_at", ""),
                "body": c.get("body") or "",
                "html_url": c.get("html_url", ""),
            }
        )
    return comments


def has_maintainer_comment(comments: List[Dict[str, Any]]) -> bool:
    for c in comments:
        if c.get("author_association") in AUTHOR_ASSOCIATIONS_MAINTAINER:
            return True
    return False


def get_maintainer_comments(comments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        c
        for c in comments
        if c.get("author_association") in AUTHOR_ASSOCIATIONS_MAINTAINER
    ]


def infer_closed_resolution_signal(issue: Dict[str, Any], comments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    This is NOT final labeling. It only creates useful weak signals.

    High-value signals:
    - closed issue
    - maintainer/member comment
    - resolution keywords in maintainer comments or final comments
    - linked PR / commit
    - user says thanks/worked
    """
    text_all = issue_text_for_signals(issue, comments)
    links = extract_links_from_text(text_all)

    maint_comments = get_maintainer_comments(comments)
    maint_text = "\n".join(c.get("body", "") for c in maint_comments)

    final_comments = comments[-3:] if len(comments) >= 3 else comments
    final_text = "\n".join(c.get("body", "") for c in final_comments)

    maint_resolution_hits = count_keyword_hits(maint_text, RESOLUTION_KEYWORDS)
    final_resolution_hits = count_keyword_hits(final_text, RESOLUTION_KEYWORDS)

    return {
        "has_maintainer_comment": bool(maint_comments),
        "num_maintainer_comments": len(maint_comments),
        "has_linked_pr": bool(links["pr_urls"]),
        "has_linked_commit": bool(links["commit_urls"]),
        "num_pr_urls": len(links["pr_urls"]),
        "num_commit_urls": len(links["commit_urls"]),
        "maintainer_resolution_keyword_hits": maint_resolution_hits,
        "final_resolution_keyword_hits": final_resolution_hits,
        "weak_has_resolution_signal": (
            bool(maint_comments)
            or bool(links["pr_urls"])
            or bool(links["commit_urls"])
            or bool(maint_resolution_hits)
            or bool(final_resolution_hits)
        ),
    }


def normalize_issue_record(
    repo_row: Dict[str, str],
    issue: Dict[str, Any],
    comments_raw: List[Dict[str, Any]],
) -> Dict[str, Any]:
    comments = summarize_comments(comments_raw)
    text_all = issue_text_for_signals(issue, comments)

    positive_hits = count_keyword_hits(text_all, POSITIVE_SPEC_GAP_KEYWORDS)
    negative_hits = count_keyword_hits(text_all, NEGATIVE_ENGINEERING_KEYWORDS)
    resolution_hits = count_keyword_hits(text_all, RESOLUTION_KEYWORDS)
    links = extract_links_from_text(text_all)
    resolution_signal = infer_closed_resolution_signal(issue, comments)

    labels = [compact_label(x) for x in issue.get("labels", [])]

    return {
        "record_id": f"github_{repo_row.get('repo', '').replace('/', '__')}_issue_{issue.get('number')}",
        "source_type": "github_issue",
        "repo": repo_row.get("repo", ""),
        "repo_url": repo_row.get("repo_url", ""),
        "repo_type_guess": repo_row.get("repo_type_guess", ""),
        "paper_title_guess": repo_row.get("paper_title_guess", ""),
        "paper_url_guess": repo_row.get("paper_url_guess", ""),
        "all_paper_urls": repo_row.get("all_paper_urls", ""),
        "repo_score": repo_row.get("score", ""),
        "repo_stars": repo_row.get("stars", ""),
        "repo_open_issues_count": repo_row.get("open_issues_count", ""),

        "issue_number": issue.get("number"),
        "issue_id": issue.get("id"),
        "issue_url": issue.get("html_url", ""),
        "api_url": issue.get("url", ""),
        "title": issue.get("title") or "",
        "body": issue.get("body") or "",
        "state": issue.get("state", ""),
        "state_reason": issue.get("state_reason", ""),
        "locked": issue.get("locked", False),
        "created_at": issue.get("created_at", ""),
        "updated_at": issue.get("updated_at", ""),
        "closed_at": issue.get("closed_at", ""),
        "author_association": issue.get("author_association", ""),
        "user": compact_user(issue.get("user")),
        "labels": labels,
        "label_names": [x["name"] for x in labels],
        "comments_count": issue.get("comments", 0),
        "comments": comments,

        "linked_pr_urls": links["pr_urls"],
        "linked_commit_urls": links["commit_urls"],
        "linked_issue_urls": links["issue_urls"],
        "sha_mentions": links["sha_mentions"][:20],

        "positive_spec_gap_keyword_hits": positive_hits,
        "negative_engineering_keyword_hits": negative_hits,
        "resolution_keyword_hits": resolution_hits,
        "num_positive_spec_gap_keyword_hits": len(positive_hits),
        "num_negative_engineering_keyword_hits": len(negative_hits),
        "num_resolution_keyword_hits": len(resolution_hits),

        "weak_resolution_signal": resolution_signal,
    }


def fetch_issue_comments(
    comments_url: str,
    headers: Dict[str, str],
    sleep: float,
    max_comments_per_issue: int,
) -> List[Dict[str, Any]]:
    if not comments_url:
        return []

    comments = paginate_get(
        comments_url,
        headers=headers,
        params={
            "per_page": 100,
        },
        sleep=sleep,
        max_items=max_comments_per_issue,
    )

    return comments


def crawl_repo_issues(
    repo_row: Dict[str, str],
    headers: Dict[str, str],
    state: str,
    sleep: float,
    max_issues_per_repo: int,
    max_comments_per_issue: int,
    since: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    repo = repo_row.get("repo", "")
    url = f"{GITHUB_API}/repos/{repo}/issues"

    params: Dict[str, Any] = {
        "state": state,
        "sort": "updated",
        "direction": "desc",
        "per_page": 100,
    }
    if since:
        params["since"] = since

    issues = paginate_get(
        url,
        headers=headers,
        params=params,
        sleep=sleep,
        max_items=max_issues_per_repo,
    )

    records: List[Dict[str, Any]] = []
    skipped_prs = 0
    fetched_comments_total = 0
    failed_comments = 0

    for idx, issue in enumerate(issues, start=1):
        # GitHub issue endpoint includes PRs. Skip PRs.
        if "pull_request" in issue:
            skipped_prs += 1
            continue

        comments_url = issue.get("comments_url", "")
        comments_raw = []

        if int(issue.get("comments", 0) or 0) > 0:
            comments_raw = fetch_issue_comments(
                comments_url=comments_url,
                headers=headers,
                sleep=sleep,
                max_comments_per_issue=max_comments_per_issue,
            )
            fetched_comments_total += len(comments_raw)

            # If comments_count > 0 but we got none, mark possible failure.
            if len(comments_raw) == 0:
                failed_comments += 1

        rec = normalize_issue_record(
            repo_row=repo_row,
            issue=issue,
            comments_raw=comments_raw,
        )
        records.append(rec)

    summary = {
        "repo": repo,
        "repo_url": repo_row.get("repo_url", ""),
        "paper_title_guess": repo_row.get("paper_title_guess", ""),
        "paper_url_guess": repo_row.get("paper_url_guess", ""),
        "repo_type_guess": repo_row.get("repo_type_guess", ""),
        "repo_score": repo_row.get("score", ""),
        "repo_stars": repo_row.get("stars", ""),
        "requested_state": state,
        "raw_items_fetched": len(issues),
        "issues_saved": len(records),
        "prs_skipped": skipped_prs,
        "comments_fetched": fetched_comments_total,
        "failed_comment_fetch_count": failed_comments,
        "issues_with_comments": sum(1 for r in records if len(r.get("comments", [])) > 0),
        "issues_with_maintainer_comment": sum(
            1 for r in records
            if r.get("weak_resolution_signal", {}).get("has_maintainer_comment")
        ),
        "issues_with_linked_pr": sum(
            1 for r in records
            if r.get("weak_resolution_signal", {}).get("has_linked_pr")
        ),
        "issues_with_linked_commit": sum(
            1 for r in records
            if r.get("weak_resolution_signal", {}).get("has_linked_commit")
        ),
        "issues_with_positive_spec_gap_keywords": sum(
            1 for r in records
            if r.get("num_positive_spec_gap_keyword_hits", 0) > 0
        ),
        "issues_with_weak_resolution_signal": sum(
            1 for r in records
            if r.get("weak_resolution_signal", {}).get("weak_has_resolution_signal")
        ),
    }

    return records, summary


def append_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    # Include any extra keys if later rows have them.
    seen = set(fieldnames)
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_existing_record_ids(path: str) -> set:
    """
    For resumable crawling.
    If output jsonl already exists, avoid writing duplicate issue records.
    """
    ids = set()
    if not os.path.exists(path):
        return ids

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rid = obj.get("record_id")
                if rid:
                    ids.add(rid)
            except Exception:
                continue

    return ids


def load_existing_crawled_repos(summary_path: str) -> set:
    """
    Resume at repo level if crawl_summary.csv already exists.
    """
    repos = set()
    if not os.path.exists(summary_path):
        return repos

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("repo"):
                    repos.add(row["repo"])
    except Exception:
        return repos

    return repos


def print_global_summary(summaries: List[Dict[str, Any]]) -> None:
    if not summaries:
        print("[summary] no repos crawled")
        return

    int_keys = [
        "raw_items_fetched",
        "issues_saved",
        "prs_skipped",
        "comments_fetched",
        "failed_comment_fetch_count",
        "issues_with_comments",
        "issues_with_maintainer_comment",
        "issues_with_linked_pr",
        "issues_with_linked_commit",
        "issues_with_positive_spec_gap_keywords",
        "issues_with_weak_resolution_signal",
    ]

    totals = Counter()
    for s in summaries:
        for k in int_keys:
            try:
                totals[k] += int(s.get(k, 0))
            except Exception:
                pass

    print("\n[global summary]")
    print(f"repos crawled: {len(summaries)}")
    for k in int_keys:
        print(f"{k}: {totals[k]}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Step 1 seed repo CSV.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="Real_bench/github_issue_mining/outputs_step2",
    )
    parser.add_argument(
        "--state",
        type=str,
        default="closed",
        choices=["open", "closed", "all"],
        help="Issue state to crawl. For gold-solution mining, use closed.",
    )
    parser.add_argument(
        "--max_repos",
        type=int,
        default=50,
        help="Number of top repos to crawl. Use small number for pilot.",
    )
    parser.add_argument(
        "--max_issues_per_repo",
        type=int,
        default=200,
        help="Max issue items fetched per repo. PRs will be skipped.",
    )
    parser.add_argument(
        "--max_comments_per_issue",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--github_token",
        type=str,
        default=os.environ.get("GITHUB_TOKEN", ""),
        help="Optional. Prefer setting env var GITHUB_TOKEN.",
    )
    parser.add_argument(
        "--since",
        type=str,
        default="",
        help="Optional ISO timestamp, e.g. 2020-01-01T00:00:00Z.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output files and skip repos already in summary.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="raw_issues_closed_answered",
        help="Base filename prefix for output jsonl.",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    out_jsonl = os.path.join(args.out_dir, f"{args.output_prefix}.jsonl")
    summary_csv = os.path.join(args.out_dir, "crawl_summary.csv")
    failed_jsonl = os.path.join(args.out_dir, "failed_repos.jsonl")

    token = args.github_token.strip() or None
    headers = github_headers(token)

    repo_rows = load_seed_repos(
        input_csv=args.input_csv,
        max_repos=args.max_repos,
    )

    print(f"[info] input repos selected: {len(repo_rows)}")
    print(f"[info] state: {args.state}")
    print(f"[info] max_issues_per_repo: {args.max_issues_per_repo}")
    print(f"[info] output: {out_jsonl}")

    existing_ids = set()
    crawled_repos = set()

    if args.resume:
        existing_ids = load_existing_record_ids(out_jsonl)
        crawled_repos = load_existing_crawled_repos(summary_csv)
        print(f"[resume] existing issue records: {len(existing_ids)}")
        print(f"[resume] already summarized repos: {len(crawled_repos)}")

    summaries: List[Dict[str, Any]] = []
    failed_repos: List[Dict[str, Any]] = []

    # If resume is false, overwrite previous outputs.
    if not args.resume:
        with open(out_jsonl, "w", encoding="utf-8") as f:
            pass
        with open(failed_jsonl, "w", encoding="utf-8") as f:
            pass

    for ri, repo_row in enumerate(repo_rows, start=1):
        repo = repo_row.get("repo", "")

        if args.resume and repo in crawled_repos:
            print(f"[skip resume] {repo}")
            continue

        print(f"\n[{ri}/{len(repo_rows)}] crawling repo: {repo}")

        try:
            records, summary = crawl_repo_issues(
                repo_row=repo_row,
                headers=headers,
                state=args.state,
                sleep=args.sleep,
                max_issues_per_repo=args.max_issues_per_repo,
                max_comments_per_issue=args.max_comments_per_issue,
                since=args.since.strip() or None,
            )

            # Deduplicate if resuming.
            if existing_ids:
                new_records = []
                for r in records:
                    rid = r.get("record_id")
                    if rid not in existing_ids:
                        new_records.append(r)
                        existing_ids.add(rid)
                records = new_records

            append_jsonl(out_jsonl, records)
            summaries.append(summary)

            print(
                f"[repo done] saved={len(records)} "
                f"comments={summary['comments_fetched']} "
                f"maintainer_comment_issues={summary['issues_with_maintainer_comment']} "
                f"weak_resolution_signal={summary['issues_with_weak_resolution_signal']}"
            )

        except Exception as e:
            failed = {
                "repo": repo,
                "error": repr(e),
                "paper_url_guess": repo_row.get("paper_url_guess", ""),
                "repo_url": repo_row.get("repo_url", ""),
            }
            failed_repos.append(failed)
            append_jsonl(failed_jsonl, [failed])
            print(f"[error] failed repo {repo}: {repr(e)}", file=sys.stderr)

    # If resume and existing summary exists, merge old + new summary.
    if args.resume and os.path.exists(summary_csv):
        old_summaries = []
        try:
            with open(summary_csv, "r", encoding="utf-8") as f:
                old_summaries = list(csv.DictReader(f))
        except Exception:
            old_summaries = []

        # Keep old summaries, add new summaries. If duplicated repo, latest wins.
        merged = {}
        for s in old_summaries:
            if s.get("repo"):
                merged[s["repo"]] = s
        for s in summaries:
            if s.get("repo"):
                merged[s["repo"]] = s
        summaries_to_write = list(merged.values())
    else:
        summaries_to_write = summaries

    write_csv(summary_csv, summaries_to_write)

    print("\n[done]")
    print(f"issues jsonl: {out_jsonl}")
    print(f"summary csv:  {summary_csv}")
    print(f"failed repos:  {failed_jsonl}")
    print_global_summary(summaries_to_write)


if __name__ == "__main__":
    main()
