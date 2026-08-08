from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz
from openai import OpenAI
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from taxonomy_utils import align_gap_labels_with_level2


HEADERS = {
    "User-Agent": "Mozilla/5.0 (research; benchmark construction)",
}

DEFAULT_INPUT = "Real_bench/github_issue_mining/outputs_step4/gated_atomic_candidates.jsonl"
DEFAULT_OUTPUT = (
    "Real_bench/github_issue_mining/outputs_step5/"
    "verified_atomic_instances_with_original_text.jsonl"
)
DEFAULT_PDF_DIR = "Real_bench/github_issue_mining/pdfs/originals"
DEFAULT_MD_DIR = "Real_bench/github_issue_mining/markdown_fast_pymupdf/originals"

DEFAULT_OPENAI_MODEL = (
    os.getenv("OPENROUTER_MODEL")
    or os.getenv("OPENAI_MODEL")
    or "deepseek/deepseek-v4-pro"
)

VALID_LEVEL2_BY_LEVEL1 = {
    "Ambiguity": {
        "Ambiguous Definition",
        "Ambiguous Procedure",
    },
    "Incompleteness": {
        "Missing Algorithmic Procedure",
        "Missing Configuration Protocol",
        "Missing Model Specification",
        "Missing Evaluation Specification",
        "Missing Data Specification",
    },
    "Inconsistency": {
        "Conflicting Objective",
        "Conflicting Model Design",
        "Conflicting Formal Definition",
    },
}

LEVEL2_TO_LEVEL1 = {
    level2: level1
    for level1, values in VALID_LEVEL2_BY_LEVEL1.items()
    for level2 in values
}

VALID_LEVEL1 = set(VALID_LEVEL2_BY_LEVEL1.keys())
VALID_LEVEL2 = set(LEVEL2_TO_LEVEL1.keys())

VALID_AFFECTED_COMPONENTS = {
    "TASK_AND_IO",
    "CORE_ALGORITHM",
    "MODEL_ARCHITECTURE",
    "OBJECTIVE_AND_SUPERVISION",
    "TRAINING_PROCEDURE",
    "DATA_AND_PREPROCESSING",
    "INFERENCE_AND_DECISION",
    "EVALUATION_PROTOCOL",
    "INTERNAL_CONSISTENCY",
    "NONE",
}

VALID_SOLUTION_SOURCE_TYPES = {
    "author_clarification",
    "maintainer_clarification",
    "code_derived",
    "reproducer_assumption",
    "reproducer_workaround",
}

VALID_RESOLUTION_ROLES = {
    "implementation_blocker",
    "reproducibility_detail",
    "open_design_choice",
    "inconsistency_to_resolve",
}

VALID_CANDIDATE_STRENGTHS = {
    "strong",
    "borderline",
    "weak",
}

STEP5_SYSTEM_PROMPT = f"""
You are a strict verifier for GitHub real-gap benchmark instances.

This is Step5. Step4 already gated and split atomic candidates.

Your task is ONLY:
1) verify whether the atomic candidate is still valid,
2) assign taxonomy using the allowed IDEAAMBIG taxonomy,
3) output a minimal gold_clarified_detail rewrite.

Hard constraints:
- Do NOT invent details not supported by the paper excerpt or issue thread.
- Do NOT add external knowledge.
- Do NOT expand the atomic gap into multiple independent gaps.
- If the evidence is insufficient, reject the candidate.
- If the issue is an implementation bug, pure repo usage problem, or tuning advice rather than a method-core spec gap, reject it.
- Evaluation protocol gaps are valid if they define metric computation, evaluation split, prompt set, threshold, sampling, seed/sample count, or evaluator configuration.
- Data/preprocessing gaps are valid if they define normalization, augmentation, data filtering, label construction, train/validation construction, segmentation, stride/windowing, or preprocessing before model input.

Allowed Level-1 labels:
- Ambiguity
- Incompleteness
- Inconsistency

Allowed Level-2 labels:
Ambiguity:
- Ambiguous Definition
- Ambiguous Procedure

Incompleteness:
- Missing Algorithmic Procedure
- Missing Configuration Protocol
- Missing Model Specification
- Missing Evaluation Specification
- Missing Data Specification

Inconsistency:
- Conflicting Objective
- Conflicting Model Design
- Conflicting Formal Definition

Labeling rules:
- Use "Missing Evaluation Specification" for missing metric computation, evaluation data split, evaluation prompt set, evaluation threshold, evaluation sampling, number of evaluation seeds/samples, or evaluator configuration.
- Use "Missing Data Specification" for missing input normalization, augmentation, tokenization, data filtering, label construction, train/validation data construction, segmentation, stride/windowing, or preprocessing before model input.
- Use "Missing Algorithmic Procedure" for missing core method procedure, training-loop rule, update order, loss routing, sampling/update rule, or termination condition.
- Use "Missing Configuration Protocol" only when the missing issue is how to select/tune/validate a hyperparameter, not merely a single ordinary unreported value.
- Use "Missing Model Specification" for structural model choices such as activation, normalization, pooling, layer/module type, initialization, readout, dimensional mapping, or module wiring.
- Use Inconsistency only when two concrete sources conflict, such as paper vs code, paper vs README, paper vs appendix, or two concrete implementation descriptions.
- If uncertain, prefer Ambiguity or Incompleteness over Inconsistency unless explicit contradiction evidence exists.

Hard constraints for gold_clarified_detail:
- 1 to 3 sentences only.
- no explanation, no reasoning, no comparison.
- no dataset speculation, no guessing.
- concise implementation-ready specification only.
- no meta-language such as "the issue clarifies", "the author says", "GitHub thread", or "paper is ambiguous".
- write as a method-level specification that can be used for implementation.
- do not defer to literature, citations, or prior work (e.g. "as proven in the literature", "according to", "[27]").

Return STRICT JSON only.
"""

BANNED_GOLD_PATTERNS = [
    r"\bbecause\b",
    r"\btherefore\b",
    r"\bhowever\b",
    r"\bcompared to\b",
    r"\bwe assume\b",
    r"\bmight\b",
    r"\bmay\b",
    r"\bcould\b",
    r"\bin general\b",
    r"\bfor example\b",
    r"\bgithub\b",
    r"\bissue\b",
    r"\bauthor says\b",
    r"\bmaintainer says\b",
    r"\bpaper is ambiguous\b",
    r"\bnot specified\b",
    r"\bunclear\b",
]

REJECTION_REASONS = {
    "insufficient_evidence",
    "not_real_spec_gap",
    "not_resolved",
    "not_method_core",
    "gold_not_minimal",
    "missing_paper_url",
    "implementation_bug_not_spec_gap",
    "repo_usage_not_method_spec_gap",
    "tuning_or_best_practice_advice",
    "other",
    "null",
    None,
}


def sentence_count(text: str) -> int:
    parts = re.split(r"(?<=[.!?])\s+", str(text or "").strip())
    return len([p for p in parts if p.strip()])


def violates_gold_constraints(text: str) -> List[str]:
    issues: List[str] = []
    raw = str(text or "").strip()
    sc = sentence_count(raw)

    if sc < 1 or sc > 3:
        issues.append(f"gold_sentence_count_{sc}")

    low = raw.lower()
    if len(low.split()) > 120:
        issues.append("gold_too_long")

    for pattern in BANNED_GOLD_PATTERNS:
        if re.search(pattern, low):
            issues.append(f"gold_banned_pattern:{pattern}")

    return issues


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_level2(label: Any) -> str:
    raw = normalize_space(label)
    for valid in VALID_LEVEL2:
        if raw.lower() == valid.lower():
            return valid
    return ""


def normalize_nullable_enum(value: Any, valid_values: set[str]) -> Optional[str]:
    raw = normalize_space(value)
    if not raw or raw.lower() == "null":
        return None
    if raw in valid_values:
        return raw
    low = raw.lower()
    for v in valid_values:
        if v.lower() == low:
            return v
    return None


def normalize_step5_result(result: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(result)

    keep_criteria = out.get("keep_criteria")
    if not isinstance(keep_criteria, dict):
        keep_criteria = {}
    for key in (
        "is_spec_gap",
        "is_actionable",
        "is_method_core_spec_gap",
        "gold_clarified_spec_extractable",
    ):
        keep_criteria[key] = bool(keep_criteria.get(key))
    out["keep_criteria"] = keep_criteria

    level2 = normalize_level2(out.get("level2"))
    if level2:
        out["level2"] = level2
        out["level1"] = LEVEL2_TO_LEVEL1[level2]
    else:
        out["level1"] = None
        out["level2"] = None

    out["affected_component"] = normalize_nullable_enum(
        out.get("affected_component"), VALID_AFFECTED_COMPONENTS
    )
    out["solution_source_type"] = normalize_nullable_enum(
        out.get("solution_source_type"), VALID_SOLUTION_SOURCE_TYPES
    )
    out["resolution_role_hint"] = normalize_nullable_enum(
        out.get("resolution_role_hint"), VALID_RESOLUTION_ROLES
    )

    strength = normalize_space(out.get("candidate_strength")).lower()
    out["candidate_strength"] = strength if strength in VALID_CANDIDATE_STRENGTHS else "weak"

    try:
        out["confidence"] = float(out.get("confidence") or 0.0)
    except Exception:
        out["confidence"] = 0.0

    rejection = out.get("rejection_reason")
    if rejection == "null":
        rejection = None
    if rejection not in REJECTION_REASONS:
        rejection = "other"
    out["rejection_reason"] = rejection

    required_keep = [
        out.get("is_real_spec_gap") is True,
        out.get("is_resolved") is True,
        keep_criteria.get("is_spec_gap") is True,
        keep_criteria.get("is_actionable") is True,
        keep_criteria.get("is_method_core_spec_gap") is True,
        keep_criteria.get("gold_clarified_spec_extractable") is True,
        bool(out.get("gap_quote")),
        bool(out.get("solution_quote")),
        bool(out.get("gold_clarified_detail")),
        bool(out.get("level1")),
        bool(out.get("level2")),
    ]

    if not all(required_keep):
        out["keep"] = False
        if out.get("rejection_reason") is None:
            out["rejection_reason"] = "insufficient_evidence"

    out = align_gap_labels_with_level2(out)
    return out


def build_step5_prompt(record: Dict[str, Any], original_text: str) -> str:
    atomic = record.get("atomic_gap") or {}
    comments = []
    for c in record.get("comments") or []:
        if isinstance(c, dict):
            comments.append(str(c.get("body") or ""))
    thread_text = "\n\n".join(
        [
            str(record.get("title") or ""),
            str(record.get("body") or ""),
            *comments[:10],
        ]
    )

    return f"""
Step5 strict verification for one atomic candidate.

Issue metadata:
- repo: {record.get("repo")}
- issue_url: {record.get("issue_url")}
- title: {record.get("title")}
- paper_title_guess: {record.get("paper_title_guess")}
- paper_url_guess: {record.get("paper_url_guess")}

Atomic candidate from Step4:
{json.dumps(atomic, ensure_ascii=False, indent=2)}

Original paper excerpt:
\"\"\"
{original_text[:20000]}
\"\"\"

Issue thread excerpt:
\"\"\"
{thread_text[:20000]}
\"\"\"

Verification instructions:
- Verify that the atomic candidate is a real, resolved, method-core specification gap.
- Use the original paper excerpt to check whether the gap maps to paper/method specification, not only repo usage.
- Use the issue thread only as resolution evidence.
- Reject if the solution evidence is insufficient.
- Reject if this is merely an implementation bug, environment issue, repo usage issue, or tuning advice.
- Keep evaluation protocol gaps if they specify reproducible metric computation, data split, prompt set, threshold, sampling, seed/sample count, or evaluator configuration.
- Keep data/preprocessing gaps if they specify reproducible data construction, preprocessing, segmentation, stride/windowing, filtering, label construction, or tokenization.
- gold_clarified_detail must be the minimal implementation-ready clarification, not an explanation.

Return STRICT JSON:
{{
  "keep": true/false,
  "is_real_spec_gap": true/false,
  "is_resolved": true/false,
  "keep_criteria": {{
    "is_spec_gap": true/false,
    "is_actionable": true/false,
    "is_method_core_spec_gap": true/false,
    "gold_clarified_spec_extractable": true/false
  }},
  "rejection_reason": "insufficient_evidence|not_real_spec_gap|not_resolved|not_method_core|implementation_bug_not_spec_gap|repo_usage_not_method_spec_gap|tuning_or_best_practice_advice|other|null",
  "level1": "Ambiguity|Incompleteness|Inconsistency|null",
  "level2": "Ambiguous Definition|Ambiguous Procedure|Missing Algorithmic Procedure|Missing Configuration Protocol|Missing Model Specification|Missing Evaluation Specification|Missing Data Specification|Conflicting Objective|Conflicting Model Design|Conflicting Formal Definition|null",
  "affected_component": "TASK_AND_IO|CORE_ALGORITHM|MODEL_ARCHITECTURE|OBJECTIVE_AND_SUPERVISION|TRAINING_PROCEDURE|DATA_AND_PREPROCESSING|INFERENCE_AND_DECISION|EVALUATION_PROTOCOL|INTERNAL_CONSISTENCY|NONE|null",
  "solution_source_type": "author_clarification|maintainer_clarification|code_derived|reproducer_assumption|reproducer_workaround|null",
  "resolution_role_hint": "implementation_blocker|reproducibility_detail|open_design_choice|inconsistency_to_resolve|null",
  "gap_summary": "",
  "gap_quote": "",
  "solution_summary": "",
  "solution_quote": "",
  "gold_clarified_detail": "",
  "why_this_blocks_or_affects_codification": "",
  "candidate_strength": "strong|borderline|weak",
  "confidence": 0.0
}}
"""


def build_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=(
            os.getenv("OPENROUTER_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://openrouter.ai/api/v1"
        ),
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "IdeaAmbig-Bench"),
        },
    )


def call_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    max_retries: int = 4,
    sleep: float = 2.0,
) -> Dict[str, Any]:
    last_err = ""
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": STEP5_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or ""
            return json.loads(content)
        except Exception as e:
            last_err = repr(e)
            if attempt < max_retries:
                time.sleep(sleep * attempt)
    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_err}")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
    return rows


def save_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_pdf_url(url: Optional[str]) -> str:
    if not url:
        return ""

    url = str(url).strip()

    m = re.match(r"https?://arxiv\.org/abs/([^?#|]+)", url)
    if m:
        arxiv_id = m.group(1).replace(".pdf", "")
        return f"https://arxiv.org/pdf/{arxiv_id}"

    m = re.match(r"https?://arxiv\.org/pdf/([^?#|]+)", url)
    if m:
        arxiv_id = m.group(1).replace(".pdf", "")
        return f"https://arxiv.org/pdf/{arxiv_id}"

    if "openreview.net/forum" in url:
        m = re.search(r"[?&]id=([^&|]+)", url)
        if m:
            rid = re.sub(r"[^0-9A-Za-z_-].*$", "", m.group(1).strip())
            return f"https://openreview.net/pdf?id={rid}"

    if url.startswith("/pdf/"):
        return "https://openreview.net" + url

    return url.split("|")[0].strip()


def normalize_paper_page_url(url: Optional[str]) -> str:
    pdf_url = normalize_pdf_url(url)
    if not pdf_url:
        return ""

    m = re.match(r"https?://arxiv\.org/pdf/([^?#]+)", pdf_url)
    if m:
        return f"https://arxiv.org/abs/{m.group(1)}"

    if "openreview.net/pdf" in pdf_url:
        m = re.search(r"id=([^&]+)", pdf_url)
        if m:
            return f"https://openreview.net/forum?id={m.group(1)}"

    return pdf_url


def collect_paper_url_candidates(record: Dict[str, Any]) -> List[str]:
    candidates: List[str] = []

    for key in ("paper_url_guess", "all_paper_urls", "original_paper_url"):
        raw = str(record.get(key) or "").strip()
        if not raw:
            continue
        for part in re.split(r"[|,]", raw):
            part = part.strip()
            if part and part not in candidates:
                candidates.append(part)

    return candidates


def pick_paper_url(record: Dict[str, Any]) -> str:
    for candidate in collect_paper_url_candidates(record):
        pdf_url = normalize_pdf_url(candidate)
        if pdf_url:
            return pdf_url
    return ""


def is_probably_pdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 5:
        return False
    try:
        with path.open("rb") as f:
            return f.read(5) == b"%PDF-"
    except Exception:
        return False


def is_existing_text(path: Path, min_chars: int = 50) -> bool:
    if not path.exists():
        return False
    try:
        return len(path.read_text(encoding="utf-8", errors="ignore").strip()) >= min_chars
    except Exception:
        return False


def normalize_markdown(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def url_slug(url: str) -> str:
    pdf_url = normalize_pdf_url(url)

    m = re.match(r"https?://arxiv\.org/pdf/([^?#]+)", pdf_url)
    if m:
        return "arxiv_" + re.sub(r"[^A-Za-z0-9._-]+", "_", m.group(1))

    m = re.search(r"id=([^&]+)", pdf_url)
    if m:
        return "openreview_" + re.sub(r"[^A-Za-z0-9._-]+", "_", m.group(1))

    digest = hashlib.sha1(pdf_url.encode("utf-8")).hexdigest()[:16]
    return f"paper_{digest}"


def download_pdf(
    url: str,
    out_path: Path,
    max_retries: int = 5,
    sleep: float = 2.0,
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
            tmp.write_bytes(data)

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
            time.sleep(sleep * attempt)
        except Exception as e:
            last_err = repr(e)
            time.sleep(sleep * attempt)

    return False, f"failed_after_retries: {last_err}"


def pdf_to_text_markdown(pdf_path: Path, md_path: Path) -> Tuple[bool, str, Dict[str, Any]]:
    if is_existing_text(md_path):
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        return True, "exists", {"num_pages": None, "num_chars": len(text)}

    if not pdf_path.exists():
        return False, "pdf_missing", {"num_pages": 0, "num_chars": 0}

    if not is_probably_pdf(pdf_path):
        return False, "not_valid_pdf", {"num_pages": 0, "num_chars": 0}

    try:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        doc = fitz.open(str(pdf_path))
        pages: List[str] = []

        for i, page in enumerate(doc, start=1):
            text = normalize_markdown(page.get_text("text") or "")
            pages.append(f"\n\n# Page {i}\n\n{text}")

        num_pages = len(doc)
        doc.close()

        md = normalize_markdown("\n".join(pages))
        if len(md) < 50:
            return False, "extracted_text_too_short", {
                "num_pages": num_pages,
                "num_chars": len(md),
            }

        md_path.write_text(md, encoding="utf-8")
        return True, "converted", {"num_pages": num_pages, "num_chars": len(md)}

    except Exception as e:
        return False, f"conversion_failed: {repr(e)}", {"num_pages": 0, "num_chars": 0}


def ensure_local_paths(record: Dict[str, Any]) -> Dict[str, Any]:
    lp = record.get("local_paths")
    if not isinstance(lp, dict):
        lp = {}
    record["local_paths"] = lp
    return lp


def finalize_step5_result(
    raw_result: Dict[str, Any],
    paper_missing: bool = False,
    allow_missing_paper_url: bool = False,
) -> Dict[str, Any]:
    result = normalize_step5_result(raw_result)

    gold_text = str(result.get("gold_clarified_detail") or "")
    gold_issues = violates_gold_constraints(gold_text)

    if gold_issues:
        result["keep"] = False
        result["rejection_reason"] = "gold_not_minimal"
        result["gold_constraint_issues"] = gold_issues

    if paper_missing and result.get("keep") is True:
        warnings = list(result.get("verification_warnings") or [])
        warnings.append("missing_paper_url_or_original_text")
        result["verification_warnings"] = warnings
        result["verified_status"] = "review_needed" if allow_missing_paper_url else "rejected"
        if not allow_missing_paper_url:
            result["keep"] = False
            result["rejection_reason"] = "missing_paper_url"
    else:
        result["verified_status"] = "kept" if result.get("keep") is True else "rejected"

    return result


def run_step5_verifier(
    out: Dict[str, Any],
    original_text: str,
    client: OpenAI,
    model: str,
    sleep: float,
    paper_missing: bool = False,
    allow_missing_paper_url: bool = False,
) -> Dict[str, Any]:
    prompt = build_step5_prompt(out, original_text)
    raw_result = call_llm(client, model, prompt, sleep=sleep)
    result = finalize_step5_result(
        raw_result,
        paper_missing=paper_missing,
        allow_missing_paper_url=allow_missing_paper_url,
    )
    out["step5_verifier"] = result
    out["gap"] = result
    return out


def process_record(
    record: Dict[str, Any],
    pdf_dir: Path,
    md_dir: Path,
    sleep: float,
    client: OpenAI,
    model: str,
    url_cache: Dict[str, Dict[str, Any]],
    allow_missing_paper_url: bool = False,
) -> Dict[str, Any]:
    out = dict(record)
    lp = ensure_local_paths(out)

    paper_pdf_url = pick_paper_url(out)
    out["original_paper_pdf_url"] = paper_pdf_url
    out["original_paper_url"] = normalize_paper_page_url(paper_pdf_url)
    out["original_paper_title"] = out.get("original_paper_title") or out.get("paper_title_guess")

    stepb: Dict[str, Any] = {
        "paper_url_candidates": collect_paper_url_candidates(out),
        "selected_paper_pdf_url": paper_pdf_url,
    }

    if not paper_pdf_url:
        stepb["pdf_status"] = "missing_paper_url"
        stepb["markdown_status"] = "skipped_no_url"
        out["original_text_source"] = "none"
        out["stepB"] = stepb

        if allow_missing_paper_url:
            return run_step5_verifier(
                out=out,
                original_text="",
                client=client,
                model=model,
                sleep=sleep,
                paper_missing=True,
                allow_missing_paper_url=True,
            )

        out["step5_verifier"] = {
            "keep": False,
            "is_real_spec_gap": False,
            "is_resolved": False,
            "rejection_reason": "missing_paper_url",
            "verified_status": "rejected",
            "verification_warnings": ["missing_paper_url"],
        }
        out["gap"] = out["step5_verifier"]
        return out

    if paper_pdf_url in url_cache:
        cached = url_cache[paper_pdf_url]
        lp["original_paper_pdf_path"] = cached.get("original_paper_pdf_path", "")
        lp["original_paper_fast_text_markdown_path"] = cached.get(
            "original_paper_fast_text_markdown_path", ""
        )
        out["original_text_path"] = lp.get("original_paper_fast_text_markdown_path", "")
        out["original_text_source"] = cached.get("original_text_source", "none")
        out["stepB"] = {**stepb, **cached.get("stepB", {}), "reused_from_cache": True}

        original_text = ""
        md_path = Path(str(lp.get("original_paper_fast_text_markdown_path") or ""))
        if md_path.exists():
            original_text = md_path.read_text(encoding="utf-8", errors="ignore")

        paper_missing = not bool(original_text.strip())
        return run_step5_verifier(
            out=out,
            original_text=original_text,
            client=client,
            model=model,
            sleep=sleep,
            paper_missing=paper_missing,
            allow_missing_paper_url=allow_missing_paper_url,
        )

    slug = url_slug(paper_pdf_url)
    pdf_path = pdf_dir / f"{slug}.pdf"
    md_path = md_dir / f"{slug}.md"

    pdf_ok, pdf_msg = download_pdf(paper_pdf_url, pdf_path, sleep=sleep)
    stepb["pdf_status"] = pdf_msg
    stepb["original_paper_pdf_path"] = str(pdf_path) if pdf_ok else ""

    if pdf_ok:
        md_ok, md_msg, md_meta = pdf_to_text_markdown(pdf_path, md_path)
        stepb["markdown_status"] = md_msg
        stepb.update(md_meta)

        if md_ok:
            lp["original_paper_pdf_path"] = str(pdf_path)
            lp["original_paper_fast_text_markdown_path"] = str(md_path)
            out["original_text_path"] = str(md_path)
            out["original_text_source"] = "fast_pymupdf"
        else:
            out["original_text_source"] = "none"
    else:
        stepb["markdown_status"] = "skipped_pdf_failed"
        out["original_text_source"] = "none"

    out["stepB"] = stepb

    url_cache[paper_pdf_url] = {
        "original_paper_pdf_path": lp.get("original_paper_pdf_path", ""),
        "original_paper_fast_text_markdown_path": lp.get(
            "original_paper_fast_text_markdown_path", ""
        ),
        "original_text_source": out.get("original_text_source", "none"),
        "stepB": stepb,
    }

    original_text = ""
    md_path_local = Path(str(lp.get("original_paper_fast_text_markdown_path") or ""))
    if md_path_local.exists():
        original_text = md_path_local.read_text(encoding="utf-8", errors="ignore")

    paper_missing = not bool(original_text.strip())

    out = run_step5_verifier(
        out=out,
        original_text=original_text,
        client=client,
        model=model,
        sleep=sleep,
        paper_missing=paper_missing,
        allow_missing_paper_url=allow_missing_paper_url,
    )

    time.sleep(sleep)
    return out


def build_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    pdf_ok = 0
    md_ok = 0
    missing_url = 0

    verified_kept = 0
    verified_rejected = 0
    review_needed = 0

    rejection_counter: Counter[str] = Counter()
    level1_counter: Counter[str] = Counter()
    level2_counter: Counter[str] = Counter()
    gold_issue_counter: Counter[str] = Counter()

    for row in rows:
        stepb = row.get("stepB") or {}

        if not stepb.get("selected_paper_pdf_url"):
            missing_url += 1

        if stepb.get("pdf_status") in {"exists", "downloaded"}:
            pdf_ok += 1

        if stepb.get("markdown_status") in {"exists", "converted"}:
            md_ok += 1

        verifier = row.get("step5_verifier") or {}
        if verifier.get("verified_status") == "review_needed":
            review_needed += 1

        if verifier.get("keep") is True:
            verified_kept += 1
            if verifier.get("level1"):
                level1_counter[str(verifier.get("level1"))] += 1
            if verifier.get("level2"):
                level2_counter[str(verifier.get("level2"))] += 1
        else:
            verified_rejected += 1
            reason = str(verifier.get("rejection_reason") or "unknown")
            rejection_counter[reason] += 1

        for issue in verifier.get("gold_constraint_issues") or []:
            gold_issue_counter[str(issue)] += 1

    unique_urls = {
        str((row.get("stepB") or {}).get("selected_paper_pdf_url") or "")
        for row in rows
    }
    unique_urls.discard("")

    return {
        "total_records": len(rows),
        "unique_paper_urls": len(unique_urls),
        "missing_paper_url": missing_url,
        "pdf_ok": pdf_ok,
        "markdown_ok": md_ok,
        "verified_kept": verified_kept,
        "verified_rejected": verified_rejected,
        "review_needed": review_needed,
        "rejection_reasons": dict(rejection_counter),
        "level1_distribution": dict(level1_counter),
        "level2_distribution": dict(level2_counter),
        "gold_constraint_issues": dict(gold_issue_counter),
    }


def should_reuse_cached_row(cached_row: Dict[str, Any]) -> bool:
    verifier = cached_row.get("step5_verifier")
    if not isinstance(verifier, dict):
        return False

    if cached_row.get("step5_error"):
        return False

    lp = cached_row.get("local_paths") or {}
    pdf_path = Path(str(lp.get("original_paper_pdf_path") or ""))
    md_path = Path(str(lp.get("original_paper_fast_text_markdown_path") or ""))

    stepb = cached_row.get("stepB") or {}
    missing_url = stepb.get("pdf_status") == "missing_paper_url"

    if missing_url:
        return True

    return (
        pdf_path.exists()
        and is_probably_pdf(pdf_path)
        and is_existing_text(md_path)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default=DEFAULT_INPUT)
    parser.add_argument("--out_path", default=DEFAULT_OUTPUT)
    parser.add_argument("--pdf_dir", default=DEFAULT_PDF_DIR)
    parser.add_argument("--markdown_dir", default=DEFAULT_MD_DIR)
    parser.add_argument("--model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing output rows by record_id and skip re-download when files exist.",
    )
    parser.add_argument(
        "--allow_missing_paper_url",
        action="store_true",
        help="Run Step5 verifier even when paper URL/text is missing. Such cases are marked review_needed if kept.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    out_path = Path(args.out_path)
    pdf_dir = Path(args.pdf_dir)
    md_dir = Path(args.markdown_dir)
    summary_path = out_path.with_name(out_path.stem + ".summary.json")

    client = build_client()

    rows_in = load_jsonl(input_path)

    existing_by_id: Dict[str, Dict[str, Any]] = {}
    if args.resume and out_path.exists():
        for row in load_jsonl(out_path):
            rid = str(row.get("record_id") or row.get("issue_key") or "").strip()
            if rid:
                existing_by_id[rid] = row

    url_cache: Dict[str, Dict[str, Any]] = {}
    if args.resume:
        for row in existing_by_id.values():
            stepb = row.get("stepB") or {}
            url = str(stepb.get("selected_paper_pdf_url") or row.get("original_paper_pdf_url") or "")
            if url:
                lp = row.get("local_paths") or {}
                url_cache[url] = {
                    "original_paper_pdf_path": lp.get("original_paper_pdf_path", ""),
                    "original_paper_fast_text_markdown_path": lp.get(
                        "original_paper_fast_text_markdown_path", ""
                    ),
                    "original_text_source": row.get("original_text_source", "none"),
                    "stepB": stepb,
                }

    print("\n===== Step 5: Strict Verifier + Minimal Gold Rewrite =====")
    print(f"Input:        {input_path}")
    print(f"Output:       {out_path}")
    print(f"PDF dir:      {pdf_dir}")
    print(f"Markdown dir: {md_dir}")
    print(f"Resume:       {args.resume}")
    print(f"Model:        {args.model}")
    print(f"Input rows:   {len(rows_in)}")
    print(f"Allow missing paper URL: {args.allow_missing_paper_url}")
    print("================================================\n")

    rows_out: List[Dict[str, Any]] = []

    for row in tqdm(rows_in, desc="Step5 original papers + verifier", unit="record"):
        rid = str(row.get("record_id") or row.get("issue_key") or "").strip()

        if args.resume and rid in existing_by_id:
            cached_row = existing_by_id[rid]
            if should_reuse_cached_row(cached_row):
                rows_out.append(cached_row)
                continue

        try:
            rows_out.append(
                process_record(
                    record=row,
                    pdf_dir=pdf_dir,
                    md_dir=md_dir,
                    sleep=args.sleep,
                    client=client,
                    model=args.model,
                    url_cache=url_cache,
                    allow_missing_paper_url=args.allow_missing_paper_url,
                )
            )
        except Exception as e:
            error_row = dict(row)
            error_row["step5_error"] = repr(e)
            error_row["step5_verifier"] = {
                "keep": False,
                "is_real_spec_gap": False,
                "is_resolved": False,
                "rejection_reason": "other",
                "verified_status": "error",
                "error": repr(e),
            }
            error_row["gap"] = error_row["step5_verifier"]
            rows_out.append(error_row)

    save_jsonl(rows_out, out_path)

    summary = build_summary(rows_out)
    summary["input_path"] = str(input_path)
    summary["out_path"] = str(out_path)
    summary["pdf_dir"] = str(pdf_dir)
    summary["markdown_dir"] = str(md_dir)
    summary["model"] = args.model
    summary["allow_missing_paper_url"] = args.allow_missing_paper_url

    save_json(summary, summary_path)

    print("\n[done]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
