import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import Counter

from tqdm import tqdm
from openai import OpenAI


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


def read_text(path: str, max_chars: int = 50000) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", errors="ignore")
    return text[:max_chars]


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def first_nonempty(*xs):
    for x in xs:
        if x is not None and x != "" and x != []:
            return x
    return None


# ============================================================
# JSON parsing
# ============================================================

def safe_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    if not text.startswith("{"):
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            text = m.group(0)

    return json.loads(text)


# ============================================================
# LLM client
# ============================================================

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


def call_llm_json(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_retries: int = 3,
    sleep: float = 5.0,
) -> Dict[str, Any]:
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are routing machine learning reproducibility reports "
                            "for benchmark construction. Return valid JSON only. "
                            "Be conservative. Do not invent unsupported details."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            content = resp.choices[0].message.content
            if content is None:
                raise ValueError("LLM returned empty content.")

            return safe_json_loads(content)

        except Exception as e:
            last_err = e
            time.sleep(sleep * attempt)

    raise RuntimeError(f"LLM call failed after retries: {repr(last_err)}")


# ============================================================
# Taxonomy
# ============================================================

TAXONOMY_BRIEF = """
Use this compact taxonomy for implementation-relevant method-core specification gaps.

Level-1: Ambiguity

1. ambiguous formal definition
A symbol, notation, mathematical object, or formal rule is underspecified, leaving multiple plausible interpretations. The ambiguity changes what an implementer would compute or instantiate.

2. ambiguous method behavior
A method component, training procedure, inference rule, or evaluation behavior is described but its operational behavior is unclear. Multiple plausible implementations are possible and may lead to different results.

Level-1: Incompleteness

3. missing algorithmic specification
A core algorithmic step, interface, update rule, routing decision, or procedural detail is omitted. Without this detail, the method cannot be faithfully implemented.

4. missing hyperparameter protocol
A result-sensitive hyperparameter is introduced, but the paper does not specify how its value is chosen, tuned, or validated. The issue is the missing selection protocol rather than a single ordinary unreported value.

5. missing model architecture
The paper states that a model or module is used but omits structural details such as layer type, normalization, pooling, activation, initialization, or dimensional mapping. These omissions materially affect the implemented model.

6. missing evaluation protocol
The evaluation setup is incomplete, including missing metric computation rules, evaluation data splits, prompt sets, thresholds, sampling procedures, or evaluation model configuration. The omission prevents faithful reproduction of reported results.

7. missing data/preprocessing protocol
The data construction, filtering, labeling, augmentation, normalization, tokenization, segmentation, or input transformation procedure is not fully specified. As a result, implementers may construct different inputs or supervision signals.

Level-1: Inconsistency

8. inconsistent objective or loss
The paper and another source, such as code or an appendix, specify different training objectives, loss terms, reward definitions, or optimization targets. Following each source would optimize a materially different objective.

9. inconsistent architecture or pipeline
The paper and another source specify different model architectures, module configurations, data pipelines, preprocessing steps, or training/evaluation pipelines. The inconsistency makes it unclear which version should be followed.

10. inconsistent model specification
Two sources specify conflicting formal model assumptions, such as distributions, conditioning sets, aggregation rules, sampling support, or probabilistic/inference definitions. The discrepancy changes the underlying model being implemented.

Boundary rules:
- Do not count ordinary missing values such as batch size or learning rate unless the missing value or selection protocol materially blocks implementation or changes the method.
- Do not count GPU, compute, memory, package installation, training time, dataset access, licensing, or random seed variation as sufficient by themselves.
- Do not count pure performance mismatch unless the report links it to a concrete method-core specification gap.
- Do not count "hard to reproduce" unless the report identifies a specific method-core specification problem.
- If the term "reproducing" means "reproducing kernel Hilbert space" or another technical phrase unrelated to reproducibility, classify the record as not a reproduction report.
"""


VALID_PRIMARY_ROUTES = {
    "resolved_real_gap",
    "synthetic_controlled",
    "unusable",
}

VALID_LEVEL1 = {
    "Ambiguity",
    "Incompleteness",
    "Inconsistency",
    "none",
}

VALID_LEVEL2 = {
    "ambiguous formal definition",
    "ambiguous method behavior",
    "missing algorithmic specification",
    "missing hyperparameter protocol",
    "missing model architecture",
    "missing evaluation protocol",
    "missing data/preprocessing protocol",
    "inconsistent objective or loss",
    "inconsistent architecture or pipeline",
    "inconsistent model specification",
    "none",
}

LEVEL2_TO_LEVEL1 = {
    "ambiguous formal definition": "Ambiguity",
    "ambiguous method behavior": "Ambiguity",
    "missing algorithmic specification": "Incompleteness",
    "missing hyperparameter protocol": "Incompleteness",
    "missing model architecture": "Incompleteness",
    "missing evaluation protocol": "Incompleteness",
    "missing data/preprocessing protocol": "Incompleteness",
    "inconsistent objective or loss": "Inconsistency",
    "inconsistent architecture or pipeline": "Inconsistency",
    "inconsistent model specification": "Inconsistency",
}


# ============================================================
# Prompt
# ============================================================

def build_prompt(
    record: Dict[str, Any],
    max_report_chars: int,
    max_original_chars: int,
) -> str:
    local_paths = record.get("local_paths") or {}

    report_text = read_text(
        local_paths.get("report_fast_text_markdown_path", ""),
        max_chars=max_report_chars,
    )

    original_text = read_text(
        local_paths.get("original_paper_fast_text_markdown_path", ""),
        max_chars=max_original_chars,
    )

    metadata = {
        "record_id": record.get("record_id"),
        "source": record.get("source"),
        "year": record.get("year"),
        "venue_or_track": record.get("venue_or_track"),

        "report_title": record.get("report_title"),
        "report_url": record.get("report_url"),
        "report_pdf_url": record.get("report_pdf_url"),
        "report_abstract": record.get("report_abstract"),

        "original_paper_title": record.get("original_paper_title"),
        "original_paper_url": record.get("original_paper_url"),
        "original_paper_pdf_url": record.get("original_paper_pdf_url"),
        "original_code_url": record.get("original_code_url"),

        "is_single_target": record.get("is_single_target"),
        "is_reproduction_report": record.get("is_reproduction_report"),
        "has_report_pdf_local": record.get("has_report_pdf_local"),
        "has_original_paper_pdf_local": record.get("has_original_paper_pdf_local"),
        "has_report_fast_text_markdown": record.get("has_report_fast_text_markdown"),
        "has_original_paper_fast_text_markdown": record.get("has_original_paper_fast_text_markdown"),
    }

    report_abstract = normalize_text(record.get("report_abstract"))

    return f"""
We are building a benchmark for idea/specification ambiguity resolution from machine learning reproducibility artifacts.

Your task has two parts:

Part 1: Metadata extraction and verification.
Verify whether this record is actually a reproduction / reproducibility / replication report. Then extract or correct the original paper metadata if the report provides evidence.

Part 2: Route classification.
Classify the record into exactly one construction route:
A. resolved_real_gap
B. synthetic_controlled
C. unusable

Be conservative. Do not invent unsupported paper titles, URLs, code URLs, or gap details.

Compact taxonomy:
{TAXONOMY_BRIEF}

Definitions:

A. resolved_real_gap
Use this route only when BOTH are true:
1. The report identifies a concrete implementation-relevant method-core specification gap in the original paper.
2. The report provides concrete resolution evidence, such as a workaround, author clarification, code-derived decision, corrected implementation detail, or explicit implementation choice sufficient to write a gold clarified specification.

B. synthetic_controlled
Use this route only when ALL are true:
1. The record is actually a reproduction / reproducibility / replication report.
2. The report does not identify a concrete real method-core specification gap.
3. The record is single-target or clearly centered on one original paper/method.
4. The original paper text/PDF is available and sufficiently complete to derive a clean gold specification.
5. The report is mainly a faithful reproduction, implementation, ablation, extension, or evaluation of the original paper.

C. unusable
Use this route if ANY of the following hold:
- It is not actually a reproduction / reproducibility / replication report.
- It is a false positive caused by terms like "reproducing kernel Hilbert space".
- The report identifies a real method-core gap but does not provide concrete solution evidence.
- The problem is only compute, GPU, memory, package, training time, dataset access, licensing, or random seed variation.
- The report only reports performance mismatch without a concrete method-core specification gap and solution evidence.
- The record is not single-target.
- The original paper text/PDF is unavailable or insufficient for synthetic-controlled construction.
- The report is too vague to determine whether a clean gold specification can be built.

Important routing policy:
- If a real method-core gap is present but unresolved, primary_route must be "unusable".
- If no real gap is present but the original paper is available and single-target, primary_route can be "synthetic_controlled".
- If the report itself contains enough source evidence and concrete solution evidence for a real gap, primary_route can be "resolved_real_gap".
- Do not force a taxonomy label if the evidence is only resource limitation, dataset access, random seed, or performance mismatch.
- Evidence snippets should be short direct quotes from the provided text when possible.

Return JSON only with this exact schema:

{{
  "llm_metadata_extraction": {{
    "is_reproduction_report": true/false,
    "is_false_positive": true/false,
    "is_single_target": true/false,
    "metadata_confidence": 0.0,
    "extracted_original_paper_title": null,
    "extracted_original_paper_url": null,
    "extracted_original_paper_pdf_url": null,
    "extracted_original_code_url": null,
    "metadata_reason": "",
    "metadata_evidence_snippets": [
      {{
        "type": "reproduction_signal|false_positive_signal|original_paper|code|url|single_target",
        "quote": "",
        "interpretation": ""
      }}
    ]
  }},
  "llm_route_classification": {{
    "primary_route": "resolved_real_gap|synthetic_controlled|unusable",
    "all_applicable_routes": [
      "resolved_real_gap|synthetic_controlled|unusable"
    ],
    "should_run_mineru": true/false,
    "should_run_detailed_extraction": true/false,
    "routing_reason": "",
    "taxonomy_candidate": {{
      "is_spec_gap": true/false,
      "is_method_core_spec_gap": true/false,
      "gold_clarified_spec_extractable": true/false,
      "level1_candidate": "Ambiguity|Incompleteness|Inconsistency|none",
      "level2_candidate": "ambiguous formal definition|ambiguous method behavior|missing algorithmic specification|missing hyperparameter protocol|missing model architecture|missing evaluation protocol|missing data/preprocessing protocol|inconsistent objective or loss|inconsistent architecture or pipeline|inconsistent model specification|none",
      "taxonomy_reason": ""
    }},
    "evidence_snippets": [
      {{
        "type": "gap|solution|resource_only|synthetic_source|unresolved_gap|unusable_reason",
        "quote": "",
        "interpretation": ""
      }}
    ],
    "potential_gap_summaries": [""],
    "potential_solution_summaries": [""],
    "confidence": 0.0
  }}
}}

How to set should_run_mineru:
- true if primary_route is resolved_real_gap.
- true if primary_route is synthetic_controlled and better original-paper markdown is needed for gold construction.
- false if primary_route is unusable.

How to set should_run_detailed_extraction:
- true only for resolved_real_gap.
- false for synthetic_controlled.
- false for unusable.

Existing metadata:
{json.dumps(metadata, indent=2, ensure_ascii=False)}

Report abstract:
\"\"\"
{report_abstract}
\"\"\"

Report fast text markdown:
\"\"\"
{report_text}
\"\"\"

Original paper fast text markdown excerpt:
\"\"\"
{original_text}
\"\"\"
"""


# ============================================================
# Normalization
# ============================================================

def normalize_metadata_output(obj: Dict[str, Any]) -> Dict[str, Any]:
    md = obj.get("llm_metadata_extraction")
    if not isinstance(md, dict):
        md = {}

    out = {
        "is_reproduction_report": bool(md.get("is_reproduction_report")),
        "is_false_positive": bool(md.get("is_false_positive")),
        "is_single_target": bool(md.get("is_single_target")),
        "metadata_confidence": 0.0,
        "extracted_original_paper_title": md.get("extracted_original_paper_title"),
        "extracted_original_paper_url": md.get("extracted_original_paper_url"),
        "extracted_original_paper_pdf_url": md.get("extracted_original_paper_pdf_url"),
        "extracted_original_code_url": md.get("extracted_original_code_url"),
        "metadata_reason": str(md.get("metadata_reason", "")),
        "metadata_evidence_snippets": md.get("metadata_evidence_snippets") if isinstance(md.get("metadata_evidence_snippets"), list) else [],
    }

    try:
        out["metadata_confidence"] = float(md.get("metadata_confidence", 0.0))
    except Exception:
        out["metadata_confidence"] = 0.0

    for key in [
        "extracted_original_paper_title",
        "extracted_original_paper_url",
        "extracted_original_paper_pdf_url",
        "extracted_original_code_url",
    ]:
        if out[key] == "":
            out[key] = None

    if out["is_false_positive"]:
        out["is_reproduction_report"] = False

    return out


def normalize_route_output(obj: Dict[str, Any], metadata_obj: Dict[str, Any]) -> Dict[str, Any]:
    route = obj.get("llm_route_classification")
    if not isinstance(route, dict):
        route = {}

    primary = route.get("primary_route")

    if primary == "unresolved_real_gap":
        primary = "unusable"

    if primary not in VALID_PRIMARY_ROUTES:
        primary = "unusable"

    routes = route.get("all_applicable_routes")
    if not isinstance(routes, list):
        routes = []

    cleaned_routes = []
    for r in routes:
        if r == "unresolved_real_gap":
            r = "unusable"
        if r in VALID_PRIMARY_ROUTES and r not in cleaned_routes:
            cleaned_routes.append(r)

    if primary not in cleaned_routes:
        cleaned_routes.insert(0, primary)

    if primary != "unusable":
        cleaned_routes = [r for r in cleaned_routes if r != "unusable"]

    taxonomy = route.get("taxonomy_candidate")
    if not isinstance(taxonomy, dict):
        taxonomy = {}

    level1 = taxonomy.get("level1_candidate", "none")
    if level1 not in VALID_LEVEL1:
        level1 = "none"

    level2 = taxonomy.get("level2_candidate", "none")
    if level2 not in VALID_LEVEL2:
        level2 = "none"
    if level2 in LEVEL2_TO_LEVEL1:
        level1 = LEVEL2_TO_LEVEL1[level2]

    taxonomy_out = {
        "is_spec_gap": bool(taxonomy.get("is_spec_gap")),
        "is_method_core_spec_gap": bool(taxonomy.get("is_method_core_spec_gap")),
        "gold_clarified_spec_extractable": bool(taxonomy.get("gold_clarified_spec_extractable")),
        "level1_candidate": level1,
        "level2_candidate": level2,
        "taxonomy_reason": str(taxonomy.get("taxonomy_reason", "")),
    }

    # Safety: false positives cannot be routed to real/synthetic sources.
    if metadata_obj.get("is_false_positive") or not metadata_obj.get("is_reproduction_report"):
        primary = "unusable"
        cleaned_routes = ["unusable"]

    if primary == "resolved_real_gap":
        taxonomy_out["is_spec_gap"] = True
        taxonomy_out["is_method_core_spec_gap"] = True
        taxonomy_out["gold_clarified_spec_extractable"] = True

        if taxonomy_out["level1_candidate"] == "none":
            taxonomy_out["level1_candidate"] = "Incompleteness"
        if taxonomy_out["level2_candidate"] == "none":
            taxonomy_out["level2_candidate"] = "missing algorithmic specification"

    elif primary == "synthetic_controlled":
        taxonomy_out["is_spec_gap"] = False
        taxonomy_out["is_method_core_spec_gap"] = False
        taxonomy_out["gold_clarified_spec_extractable"] = False
        taxonomy_out["level1_candidate"] = "none"
        taxonomy_out["level2_candidate"] = "none"

    else:
        taxonomy_out["gold_clarified_spec_extractable"] = False

    evidence = route.get("evidence_snippets")
    if not isinstance(evidence, list):
        evidence = []

    gaps = route.get("potential_gap_summaries")
    if not isinstance(gaps, list):
        gaps = []

    sols = route.get("potential_solution_summaries")
    if not isinstance(sols, list):
        sols = []

    try:
        confidence = float(route.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    should_run_mineru = bool(route.get("should_run_mineru"))
    should_run_detailed = bool(route.get("should_run_detailed_extraction"))

    if primary == "resolved_real_gap":
        should_run_mineru = True
        should_run_detailed = True
    elif primary == "synthetic_controlled":
        should_run_detailed = False
    else:
        should_run_mineru = False
        should_run_detailed = False

    return {
        "primary_route": primary,
        "all_applicable_routes": cleaned_routes,
        "should_run_mineru": should_run_mineru,
        "should_run_detailed_extraction": should_run_detailed,
        "routing_reason": str(route.get("routing_reason", "")),
        "taxonomy_candidate": taxonomy_out,
        "evidence_snippets": evidence,
        "potential_gap_summaries": gaps,
        "potential_solution_summaries": sols,
        "confidence": confidence,
    }


def normalize_llm_output(obj: Dict[str, Any]) -> Dict[str, Any]:
    metadata_obj = normalize_metadata_output(obj)
    route_obj = normalize_route_output(obj, metadata_obj)
    return {
        "llm_metadata_extraction": metadata_obj,
        "llm_route_classification": route_obj,
    }


# ============================================================
# Record updates
# ============================================================

def apply_metadata_updates(record: Dict[str, Any], metadata_obj: Dict[str, Any]) -> Dict[str, Any]:
    r = dict(record)

    if metadata_obj.get("is_reproduction_report") is True:
        r["is_reproduction_report"] = True
    elif metadata_obj.get("is_reproduction_report") is False:
        r["is_reproduction_report"] = False

    if metadata_obj.get("is_single_target") is True:
        r["is_single_target"] = True
    elif metadata_obj.get("is_single_target") is False:
        r["is_single_target"] = False

    if metadata_obj.get("is_false_positive"):
        r["exclude_reason"] = first_nonempty(
            r.get("exclude_reason"),
            "LLM metadata extraction classified this record as a false positive, not a reproduction report.",
        )

    field_map = {
        "extracted_original_paper_title": "original_paper_title",
        "extracted_original_paper_url": "original_paper_url",
        "extracted_original_paper_pdf_url": "original_paper_pdf_url",
        "extracted_original_code_url": "original_code_url",
    }

    for src_key, dst_key in field_map.items():
        val = metadata_obj.get(src_key)
        if val and not r.get(dst_key):
            r[dst_key] = val

    r["has_open_source_code"] = bool(r.get("original_code_url"))
    r["has_original_paper_pdf"] = bool(r.get("original_paper_pdf_url"))

    return r


def failure_output(error: Exception) -> Dict[str, Any]:
    return {
        "llm_metadata_extraction": {
            "is_reproduction_report": False,
            "is_false_positive": False,
            "is_single_target": False,
            "metadata_confidence": 0.0,
            "extracted_original_paper_title": None,
            "extracted_original_paper_url": None,
            "extracted_original_paper_pdf_url": None,
            "extracted_original_code_url": None,
            "metadata_reason": f"LLM metadata extraction failed: {repr(error)}",
            "metadata_evidence_snippets": [],
        },
        "llm_route_classification": {
            "primary_route": "unusable",
            "all_applicable_routes": ["unusable"],
            "should_run_mineru": False,
            "should_run_detailed_extraction": False,
            "routing_reason": f"LLM routing failed: {repr(error)}",
            "taxonomy_candidate": {
                "is_spec_gap": False,
                "is_method_core_spec_gap": False,
                "gold_clarified_spec_extractable": False,
                "level1_candidate": "none",
                "level2_candidate": "none",
                "taxonomy_reason": "Routing failed.",
            },
            "evidence_snippets": [],
            "potential_gap_summaries": [],
            "potential_solution_summaries": [],
            "confidence": 0.0,
        },
    }


# ============================================================
# Filtering
# ============================================================

def filter_rows(
    rows: List[Dict[str, Any]],
    only_mlrc: bool,
    only_tmlr: bool,
    require_report_text: bool,
    require_original_text: bool,
    limit: int,
) -> List[Dict[str, Any]]:
    out = []

    for r in rows:
        source = r.get("source")
        lp = r.get("local_paths") or {}

        if only_mlrc and source != "OpenReview_MLRC":
            continue

        if only_tmlr and source != "OpenReview_TMLR":
            continue

        if require_report_text and not lp.get("report_fast_text_markdown_path"):
            continue

        if require_original_text and not lp.get("original_paper_fast_text_markdown_path"):
            continue

        out.append(r)

        if limit and limit > 0 and len(out) >= limit:
            break

    return out


# ============================================================
# Resume helpers
# ============================================================

def has_valid_llm_outputs(r: Dict[str, Any]) -> bool:
    md = r.get("llm_metadata_extraction")
    route = r.get("llm_route_classification")

    if not isinstance(md, dict):
        return False

    if not isinstance(route, dict):
        return False

    if route.get("primary_route") not in VALID_PRIMARY_ROUTES:
        return False

    if not isinstance(route.get("taxonomy_candidate"), dict):
        return False

    return True


def should_skip_on_resume(
    record: Dict[str, Any],
    resume: bool,
    done_idx: Dict[str, int],
    outputs: List[Dict[str, Any]],
) -> bool:
    if not resume:
        return False

    rid = str(record.get("record_id") or "").strip()
    if not rid:
        return False

    prior_i = done_idx.get(rid)
    if prior_i is None:
        return False

    return has_valid_llm_outputs(outputs[prior_i])


def merge_prior_fields(record: Dict[str, Any], prior: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(record)

    if isinstance(prior.get("local_paths"), dict):
        lp = dict(record.get("local_paths") or {})
        prior_lp = prior.get("local_paths") or {}
        for k, v in prior_lp.items():
            if v and not lp.get(k):
                lp[k] = v
        merged["local_paths"] = lp

    for key in [
        "original_paper_title",
        "original_paper_url",
        "original_paper_pdf_url",
        "original_code_url",
        "is_reproduction_report",
        "is_single_target",
        "exclude_reason",
    ]:
        if prior.get(key) and not merged.get(key):
            merged[key] = prior[key]

    return merged


# ============================================================
# Routing
# ============================================================

def route_records(
    input_path: Path,
    out_path: Path,
    summary_path: Path,
    model: str,
    max_report_chars: int,
    max_original_chars: int,
    only_mlrc: bool,
    only_tmlr: bool,
    require_report_text: bool,
    require_original_text: bool,
    limit: int,
    resume: bool,
    sleep: float,
):
    rows_all = load_jsonl(input_path)
    rows = filter_rows(
        rows_all,
        only_mlrc=only_mlrc,
        only_tmlr=only_tmlr,
        require_report_text=require_report_text,
        require_original_text=require_original_text,
        limit=limit,
    )

    client = build_client()

    outputs: List[Dict[str, Any]] = []
    done_idx: Dict[str, int] = {}

    if resume and out_path.exists():
        outputs = safe_load_jsonl_if_exists(out_path)
        for i, r in enumerate(outputs):
            rid = str(r.get("record_id") or "").strip()
            if rid:
                done_idx[rid] = i
        if outputs:
            print(
                f"[resume] loaded checkpoint: rows_in_out={len(outputs)} "
                f"| valid_llm_outputs={sum(1 for r in outputs if has_valid_llm_outputs(r))}"
            )

    pre_skipped = sum(
        1
        for record in rows
        if should_skip_on_resume(
            record=record,
            resume=resume,
            done_idx=done_idx,
            outputs=outputs,
        )
    )

    print("\n===== Universal Step 03: LLM Metadata + Route Classification =====")
    print(f"Input path: {input_path}")
    print(f"Output path: {out_path}")
    print(f"Input rows before filter: {len(rows_all)}")
    print(f"Rows after filter: {len(rows)}")
    print(f"Resume: {resume}")
    if resume and pre_skipped > 0:
        print(
            f"[resume] will skip {pre_skipped}/{len(rows)} records "
            f"(valid llm_metadata_extraction + llm_route_classification already present)"
        )
    print(f"Model: {model}")
    print(f"Only MLRC: {only_mlrc}")
    print(f"Only TMLR: {only_tmlr}")
    print(f"Require report text: {require_report_text}")
    print(f"Require original text: {require_original_text}")
    print(f"Max report chars: {max_report_chars}")
    print(f"Max original chars: {max_original_chars}")
    print("==================================================================\n")

    pbar = tqdm(total=len(rows), desc="Step 03: LLM metadata + routing")
    if pre_skipped > 0:
        pbar.update(pre_skipped)
        pbar.set_postfix_str(
            f"skipped={pre_skipped} remaining={len(rows) - pre_skipped}",
            refresh=True,
        )

    for idx, record in enumerate(rows, start=1):
        rid = str(record.get("record_id") or "").strip()

        if should_skip_on_resume(
            record=record,
            resume=resume,
            done_idx=done_idx,
            outputs=outputs,
        ):
            continue

        if resume and rid:
            prior_i = done_idx.get(rid)
            if prior_i is not None:
                record = merge_prior_fields(record, outputs[prior_i])

        try:
            raw_obj = call_llm_json(
                client=client,
                model=model,
                prompt=build_prompt(
                    record=record,
                    max_report_chars=max_report_chars,
                    max_original_chars=max_original_chars,
                ),
                temperature=0.0,
            )
            norm_obj = normalize_llm_output(raw_obj)

        except Exception as e:
            norm_obj = failure_output(e)

        md_obj = norm_obj["llm_metadata_extraction"]
        route_obj = norm_obj["llm_route_classification"]

        out = apply_metadata_updates(record, md_obj)
        out["llm_metadata_extraction"] = md_obj
        out["llm_route_classification"] = route_obj

        if rid and rid in done_idx:
            outputs[done_idx[rid]] = out
        else:
            if rid:
                done_idx[rid] = len(outputs)
            outputs.append(out)

        save_jsonl(outputs, out_path)

        taxonomy = route_obj.get("taxonomy_candidate", {})
        pbar.update(1)
        pbar.set_postfix_str(
            f"skipped={pre_skipped} done={pbar.n}/{len(rows)}",
            refresh=True,
        )
        print(
            f"[{idx}/{len(rows)}] "
            f"repro={md_obj.get('is_reproduction_report')} "
            f"false_pos={md_obj.get('is_false_positive')} "
            f"route={route_obj.get('primary_route')} "
            f"mineru={route_obj.get('should_run_mineru')} "
            f"detail={route_obj.get('should_run_detailed_extraction')} "
            f"level1={taxonomy.get('level1_candidate')} "
            f"level2={taxonomy.get('level2_candidate')} "
            f"title={str(record.get('report_title'))[:80]}"
        )

        if sleep and sleep > 0:
            time.sleep(sleep)

    pbar.close()
    save_jsonl(outputs, out_path)

    route_counter = Counter()
    level1_counter = Counter()
    level2_counter = Counter()
    source_counter = Counter()
    false_pos = 0
    repro_true = 0
    single_target_true = 0
    mineru_count = 0
    detail_count = 0

    for r in outputs:
        source_counter[r.get("source")] += 1

        md = r.get("llm_metadata_extraction") or {}
        route = r.get("llm_route_classification") or {}
        taxonomy = route.get("taxonomy_candidate") or {}

        if md.get("is_false_positive"):
            false_pos += 1
        if md.get("is_reproduction_report"):
            repro_true += 1
        if md.get("is_single_target"):
            single_target_true += 1

        route_counter[route.get("primary_route", "missing")] += 1
        level1_counter[taxonomy.get("level1_candidate", "missing")] += 1
        level2_counter[taxonomy.get("level2_candidate", "missing")] += 1

        if route.get("should_run_mineru"):
            mineru_count += 1
        if route.get("should_run_detailed_extraction"):
            detail_count += 1

    summary = {
        "script": "step_03_llm_route_classification.py",
        "input_path": str(input_path),
        "out_path": str(out_path),
        "input_rows_before_filter": len(rows_all),
        "records": len(outputs),
        "model": model,
        "source_distribution": dict(source_counter),
        "only_mlrc": only_mlrc,
        "only_tmlr": only_tmlr,
        "require_report_text": require_report_text,
        "require_original_text": require_original_text,
        "max_report_chars": max_report_chars,
        "max_original_chars": max_original_chars,
        "llm_is_reproduction_report_true": repro_true,
        "llm_false_positive": false_pos,
        "llm_single_target_true": single_target_true,
        "route_distribution": dict(route_counter),
        "level1_distribution": dict(level1_counter),
        "level2_distribution": dict(level2_counter),
        "should_run_mineru": mineru_count,
        "should_run_detailed_extraction": detail_count,
        "next_step": (
            "For resolved_real_gap records, run detailed real-gap extraction. "
            "For synthetic_controlled records, run synthetic defect construction. "
            "If LLM extracted new original_paper_pdf_url values, rerun step_01_download_pdfs.py "
            "to download additional original PDFs."
        ),
    }

    save_json(summary, summary_path)

    print("\n===== Step 03 Summary =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--summary_path", default=None)

    parser.add_argument(
        "--model",
        default=os.getenv("OPENROUTER_MODEL") or os.getenv("OPENAI_MODEL") or "deepseek/deepseek-v4-pro",
    )

    parser.add_argument("--max_report_chars", type=int, default=50000)
    parser.add_argument("--max_original_chars", type=int, default=15000)

    parser.add_argument("--only_mlrc", action="store_true")
    parser.add_argument("--only_tmlr", action="store_true")

    parser.add_argument(
        "--require_report_text",
        action="store_true",
        help="Only keep records with report_fast_text_markdown_path.",
    )
    parser.add_argument(
        "--require_original_text",
        action="store_true",
        help="Only keep records with original_paper_fast_text_markdown_path.",
    )

    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output and skip records with valid LLM outputs.",
    )

    args = parser.parse_args()

    if args.only_mlrc and args.only_tmlr:
        raise ValueError("Use only one of --only_mlrc or --only_tmlr, not both.")

    out_path = Path(args.out_path)
    summary_path = Path(args.summary_path) if args.summary_path else out_path.with_suffix(".summary.json")

    route_records(
        input_path=Path(args.input_path),
        out_path=out_path,
        summary_path=summary_path,
        model=args.model,
        max_report_chars=args.max_report_chars,
        max_original_chars=args.max_original_chars,
        only_mlrc=args.only_mlrc,
        only_tmlr=args.only_tmlr,
        require_report_text=args.require_report_text,
        require_original_text=args.require_original_text,
        limit=args.limit,
        resume=args.resume,
        sleep=args.sleep,
    )


if __name__ == "__main__":
    main()
