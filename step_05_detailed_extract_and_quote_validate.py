import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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


def read_text(path: str, max_chars: int = 80000) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", errors="ignore")
    text = sanitize_text_for_llm(text)
    return text[:max_chars]


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


def normalize_for_match(x: Any) -> str:
    x = "" if x is None else str(x)
    x = re.sub(r"\s+", " ", x).strip().lower()
    return x


# ============================================================
# Text cleaning
# ============================================================

TQDM_LINE_RE = re.compile(
    r"^(Processing pages:|Predicting:|Extracting:|Downloading:|\d+%\|)",
    re.IGNORECASE,
)
PROGRESS_BAR_RE = re.compile(r"\d+%\|[^|\n]*\|")
TQDM_TIMING_RE = re.compile(r"\[\d+:\d+<\d+:\d+,\s*[\d.]+it/s\]")


def sanitize_text_for_llm(text: str) -> str:
    """Remove tqdm/progress-bar log lines before sending text to the LLM."""
    if not text:
        return ""

    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if TQDM_LINE_RE.search(stripped):
            continue
        if PROGRESS_BAR_RE.search(stripped) and len(stripped) < 300:
            continue
        if TQDM_TIMING_RE.search(stripped) and len(stripped) < 300:
            continue
        cleaned_lines.append(line)

    out = "\n".join(cleaned_lines)
    out = re.sub(r"\n{4,}", "\n\n\n", out)
    return out.strip()


def is_log_noise_quote(quote: str) -> bool:
    q = str(quote or "").strip()
    if not q:
        return False
    if "Processing pages:" in q:
        return True
    if PROGRESS_BAR_RE.search(q):
        return True
    if TQDM_TIMING_RE.search(q):
        return True
    return False


def slim_source_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Drop bulky MinerU subprocess logs from embedded source_record.
    Step 04 may store stdout/stderr tails in mineru_selected_status.
    """
    r = dict(record)
    mineru_status = r.get("mineru_selected_status")
    if not isinstance(mineru_status, dict):
        return r

    slim = dict(mineru_status)
    conv = slim.get("status")
    if isinstance(conv, dict):
        slim_conv = {}
        for key, val in conv.items():
            if not isinstance(val, dict):
                slim_conv[key] = val
                continue

            slim_val = {
                "ok": val.get("ok"),
                "message": val.get("message"),
            }

            extra = val.get("extra")
            if isinstance(extra, dict):
                slim_extra = {
                    k: v
                    for k, v in extra.items()
                    if k not in {"stdout_tail", "stderr_tail"}
                }
                slim_val["extra"] = slim_extra

            slim_conv[key] = slim_val

        slim["status"] = slim_conv

    r["mineru_selected_status"] = slim
    return r


# ============================================================
# Universal source / route logic
# ============================================================

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


def should_run_detailed(record: Dict[str, Any]) -> bool:
    """
    Step 05 only runs for records routed as resolved_real_gap.

    We require:
      primary_route == resolved_real_gap
      should_run_detailed_extraction == True

    We do NOT require original MinerU markdown because some valid real-gap
    cases can be extracted from the report alone. Original paper text is used
    when available.
    """
    route = record.get("llm_route_classification") or {}
    return (
        route.get("primary_route") == "resolved_real_gap"
        and bool(route.get("should_run_detailed_extraction"))
    )


# ============================================================
# Text selection
# ============================================================

def get_best_report_text(record: Dict[str, Any], max_chars: int) -> Dict[str, str]:
    local_paths = record.get("local_paths") or {}

    candidates = [
        ("mineru", local_paths.get("report_mineru_markdown_path", "")),
        ("fast_pymupdf", local_paths.get("report_fast_text_markdown_path", "")),
        ("top_level", record.get("report_text_path", "")),
    ]

    for source, path in candidates:
        text = read_text(path, max_chars=max_chars)
        if len(text.strip()) > 100:
            return {"source": source, "path": path, "text": text}

    return {"source": "none", "path": "", "text": ""}


def get_best_original_text(record: Dict[str, Any], max_chars: int) -> Dict[str, str]:
    local_paths = record.get("local_paths") or {}

    candidates = [
        ("mineru", local_paths.get("original_paper_mineru_markdown_path", "")),
        ("fast_pymupdf", local_paths.get("original_paper_fast_text_markdown_path", "")),
        ("top_level", record.get("original_text_path", "")),
    ]

    for source, path in candidates:
        text = read_text(path, max_chars=max_chars)
        if len(text.strip()) > 100:
            return {"source": source, "path": path, "text": text}

    return {"source": "none", "path": "", "text": ""}


# ============================================================
# Taxonomy / validation constants
# ============================================================

VALID_LEVEL2_BY_LEVEL1 = {
    "Ambiguity": {
        "ambiguous formal definition",
        "ambiguous method behavior",
    },
    "Incompleteness": {
        "missing algorithmic specification",
        "missing hyperparameter protocol",
        "missing model architecture",
        "missing evaluation protocol",
        "missing data/preprocessing protocol",
    },
    "Inconsistency": {
        "inconsistent objective or loss",
        "inconsistent architecture or pipeline",
        "inconsistent model specification",
    },
}

VALID_LEVEL2_TO_LEVEL1 = {
    level2: level1
    for level1, labels in VALID_LEVEL2_BY_LEVEL1.items()
    for level2 in labels
}

VALID_AFFECTED_COMPONENTS = {
    "input",
    "output",
    "core_method",
    "algorithm",
    "training",
    "evaluation",
    "implementation_detail",
    "code_behavior",
    "preprocessing",
    "data",
    "inference",
}

AFFECTED_COMPONENT_ALIASES = {
    "model_architecture": "core_method",
    "hyperparameter": "training",
    "postprocessing": "output",
}

VALID_SOLUTION_SOURCE_TYPES = {
    "author_clarification",
    "code_derived",
    "reproducer_assumption",
    "reproducer_workaround",
}

VALID_CANDIDATE_STRENGTHS = {
    "strong",
    "borderline",
    "weak",
}

VALID_REJECTION_REASONS = {
    "resource_only",
    "performance_only",
    "ordinary_hyperparameter_value_only",
    "uses_author_code_without_details",
    "reproducer_improvement_not_spec_gap",
    "insufficient_gap_evidence",
    "insufficient_solution_evidence",
    "not_method_core",
    "duplicate_or_subsumed",
    "other",
}

ORDINARY_HPARAM_TERMS = {
    "learning rate",
    "optimizer",
    "batch size",
    "weight decay",
    "dropout",
    "momentum",
    "epoch",
    "epochs",
    "number of epochs",
    "training epochs",
    "adam",
    "sgd",
    "random seed",
    "seed",
    "train/test split",
    "train test split",
    "training/test split",
    "training testing split",
    "split ratio",
    "train ratio",
    "test ratio",
}

IMPROVEMENT_TERMS = {
    "we added",
    "we add",
    "we propose",
    "we proposed",
    "we modified",
    "we introduce",
    "we introduced",
    "stabilize",
    "stabilization",
    "nan",
    "not a number",
    "extra loss",
    "additional loss",
    "new supervision",
    "auxiliary supervision",
    "uniform memory loss",
}

GENERIC_GAP_PHRASES = {
    "the main difficulty arose from the fact that certain hyperparameters and model structures were not specified",
    "certain hyperparameters and model structures were not specified",
    "implementation details were not given",
    "not enough details were provided",
    "details were not provided",
    "not clearly specified",
    "not specified by the authors",
}


def validate_taxonomy_label(level1: str, level2: str) -> bool:
    return level2 in VALID_LEVEL2_BY_LEVEL1.get(level1, set())


def normalize_level2_label(x: Any) -> str:
    raw = str(x or "").strip().lower()
    aliases = {
        "missing evaluation": "missing evaluation protocol",
        "missing metric protocol": "missing evaluation protocol",
        "missing evaluation metric protocol": "missing evaluation protocol",
        "missing data protocol": "missing data/preprocessing protocol",
        "missing preprocessing protocol": "missing data/preprocessing protocol",
        "missing data construction protocol": "missing data/preprocessing protocol",
        "missing data preprocessing protocol": "missing data/preprocessing protocol",
        "missing architecture": "missing model architecture",
        "missing model specification": "missing model architecture",
        "missing algorithm": "missing algorithmic specification",
        "missing algorithmic detail": "missing algorithmic specification",
        "inconsistent loss": "inconsistent objective or loss",
        "inconsistent objective": "inconsistent objective or loss",
        "inconsistent pipeline": "inconsistent architecture or pipeline",
        "inconsistent architecture": "inconsistent architecture or pipeline",
        "inconsistent model": "inconsistent model specification",
    }
    if raw in aliases:
        return aliases[raw]
    for labels in VALID_LEVEL2_BY_LEVEL1.values():
        if raw in labels:
            return raw
    for labels in VALID_LEVEL2_BY_LEVEL1.values():
        for label in labels:
            if label in raw:
                return label
    return raw


def validate_affected_component(x: str) -> bool:
    return x in VALID_AFFECTED_COMPONENTS


def normalize_affected_component(x: Any) -> str:
    raw = str(x or "").strip().lower()
    return AFFECTED_COMPONENT_ALIASES.get(raw, raw)


def validate_solution_source_type(x: str) -> bool:
    return x in VALID_SOLUTION_SOURCE_TYPES


def validate_candidate_strength(x: str) -> bool:
    return x in VALID_CANDIDATE_STRENGTHS


def validate_rejection_reason(x: str) -> bool:
    return x in VALID_REJECTION_REASONS


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
                            "You extract benchmark construction evidence from ML reproducibility reports. "
                            "Return valid JSON only. Do not invent unsupported details. "
                            "Follow the fixed taxonomy exactly."
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
# Prompt
# ============================================================

def build_prompt(
    record: Dict[str, Any],
    report_pack: Dict[str, str],
    original_pack: Dict[str, str],
) -> str:
    metadata = {
        "record_id": record.get("record_id"),
        "source": record.get("source"),
        "report_title": record.get("report_title"),
        "report_url": record.get("report_url"),
        "report_pdf_url": record.get("report_pdf_url"),
        "original_paper_title": record.get("original_paper_title"),
        "original_paper_url": record.get("original_paper_url"),
        "original_paper_pdf_url": record.get("original_paper_pdf_url"),
        "year": record.get("year"),
        "route_classification": record.get("llm_route_classification"),
        "report_text_source": report_pack.get("source"),
        "report_text_path": report_pack.get("path"),
        "original_text_source": original_pack.get("source"),
        "original_text_path": original_pack.get("path"),
    }

    return f"""
We are constructing benchmark instances for idea/specification ambiguity resolution.

You are given:
1. A machine learning reproducibility report.
2. The corresponding original paper text, when available.

Your task is to extract implementation-relevant RESOLVED specification gaps from the reproducibility report and classify them using the fixed taxonomy below.

Important pipeline context:
- This record has been routed as resolved_real_gap.
- We only want gaps that have concrete solution evidence.
- Unresolved gaps may be listed for audit, but they will NOT be used to construct benchmark gold.
- Synthetic-controlled instances are handled in a separate pipeline and should not be extracted here.

The goal is NOT to collect every reproduction difficulty.
The goal is to collect method-core specification gaps that can support benchmark instances.

A kept resolved gap must satisfy ALL benchmark keep criteria:
- is_spec_gap = true: the issue concerns an underspecified, ambiguous, or inconsistent method specification in the original paper.
- is_actionable = true: resolving the issue changes or clarifies a concrete implementation decision.
- is_method_core_spec_gap = true: the issue affects the method, model, algorithm, training procedure, evaluation protocol, data/preprocessing pipeline, or implementation behavior.
- gold_clarified_spec_extractable = true: the report provides enough concrete evidence to write a gold clarified specification.
- The gap must be about the original method specification, not merely about the reproducer's engineering difficulty.

Resolved real gap:
A gap is resolved only if the report gives a concrete, extractable resolution:
- an author-confirmed clarification,
- an explicit code-derived behavior,
- a concrete implementation decision stated in the report,
- or a clearly described workaround/assumption that fills a concrete missing slot.

A resolved gap must have:
- a specific gap_quote from the report,
- a specific solution_quote from the report,
- and a gold_clarified_detail that can be implemented.

Unresolved real gap:
A gap is unresolved if:
- the report identifies an implementation-relevant specification gap,
- but the report does not give enough concrete detail to write a gold clarified specification.

Important resolved/unresolved distinction:
- If the report only says "we used the author's code" but does not state the relevant implementation detail, mark unresolved or invalid.
- If the report says "we contacted the authors" but does not include the concrete answer, mark unresolved.
- If the report says "the authors provided a screenshot/code/explanation" but the report does not spell out the implementation detail, mark unresolved.
- If the reproducer made an assumption, mark resolved only if the assumption is explicitly stated and can be written as a reproduction-specific clarified specification.
- If the reproducer merely says they could not implement something "in limited time", this is not enough evidence of a valid spec gap.

Invalid cases:
Do NOT include these as gap candidates:
- only performance mismatch,
- only compute/GPU/memory/training time,
- only dataset unavailable,
- only random seed variation,
- only software/framework engineering difficulty,
- only "we could not implement it in limited time",
- only "we used the author's code" without extracting concrete implementation details,
- a reproducer-proposed improvement or stabilization trick that is not described as filling a missing/ambiguous/inconsistent original specification,
- a bug fix or extra loss added by the reproducer unless the report explicitly frames it as resolving an original paper specification gap,
- ordinary missing values of standard hyperparameters such as batch size, learning rate, optimizer, number of epochs, dropout, momentum, weight decay, random seed, or train/test split ratio, unless the missing selection/tuning/stopping/data-construction protocol is itself the primary blocker,
- reader confusion where the original paper already specifies the detail,
- pure typo where the intended meaning is obvious.

Specific quote requirement:
- Do not use a broad generic difficulty sentence as the only gap_quote for a specific candidate.
- A gap_quote should identify the concrete missing/ambiguous/inconsistent slot.
- If only a generic quote is available, mark the candidate as borderline or invalid unless the solution_quote clearly specifies the same missing slot.
- Do not reuse the same broad gap_quote across multiple candidates unless each candidate has a distinct, specific solution_quote and the missing slots are clearly different.

Special hyperparameter / ordinary value rule:
- If a missing slot is only an ordinary training hyperparameter value, reject it.
- If a missing slot is only a concrete train/test split ratio, reject it unless the report explicitly identifies a nontrivial data-construction or selection protocol as the blocker.
- If a missing slot is a non-standard, method-defining hyperparameter, it may be kept.
- If ordinary hyperparameters are mentioned together with a valid method-specific gap, extract only the method-specific gap and exclude ordinary optimizer/lr/batch-size details from gap_summary and gold_clarified_detail unless they are inseparable from the reported solution.
- For number of epochs, keep only if the real missing slot is a stopping/training-duration protocol, not merely an epoch count.
- Weight initialization should usually be treated as missing hyperparameter protocol unless the evidence explicitly frames it as an architecture or paper-code inconsistency.

Special reproducer-improvement rule:
- A reproducer assumption or workaround is valid only if it fills a clearly identified missing or ambiguous slot in the original paper.
- Do not mark a case as resolved if the solution is merely a reproducer-added improvement, stabilization trick, auxiliary supervision, new loss, NaN fix, or performance-motivated modification.
- If such an issue appears, put it under invalid_or_rejected_candidates with rejection_reason = "reproducer_improvement_not_spec_gap".

Gold clarified detail writing rules:
- The field gold_clarified_detail is the final target specification that a benchmark model should recover.
- Write gold_clarified_detail as a direct implementation specification, not as an annotation note, citation instruction, evidence explanation, or review comment.
- The gold_clarified_detail must contain only implementation details directly supported by the solution_quote or by another explicit contiguous quote in the provided report text.
- If the solution_quote only supports a named design choice, write only that named design choice. Do not expand it into layer-by-layer details, hyperparameter values, reference-paper mechanisms, or code behavior unless those details are explicitly stated in the report text.
- Do not add details from external papers, references, GitHub repositories, common knowledge, or your own knowledge unless the report text explicitly states those details.
- Do not write "as described in reference [X]", "following reference [X]", "following the repository", "according to the code", "unless implementation code is available", "refer to", "check", or similar evidence-management language inside gold_clarified_detail.
- If the evidence says that the solution came from code, author clarification, repository, or updated script, use that only to set solution_source_type and solution_summary. The gold_clarified_detail itself should state the resolved behavior directly.
- If the solution is author-confirmed or directly code-derived, write it as a clarified specification.
- If the solution is a reproducer assumption, interpretation, or workaround, write it explicitly as reproduction-specific, using phrases like:
  "For the reproduction, implement this by..."
  "The reproduction assumes..."
  "The reproduction operationalizes this as..."
- Do NOT present a reproducer assumption as the only intended original-paper specification.
- Do NOT add explanatory claims such as "this approximates the intended behavior" unless the report explicitly states that.
- Do NOT include explanatory consequences in gold_clarified_detail. Put consequences in why_this_blocks_or_affects_codification instead.
- The gold_clarified_detail should be executable and concrete, not a high-level summary.
- Prefer concise imperative or declarative rules.

Gold-support check:
Before finalizing each resolved_gap_candidate, verify that every concrete noun phrase or implementation choice in gold_clarified_detail is supported by solution_quote. If gold_clarified_detail contains details not present in solution_quote, either:
1. shorten gold_clarified_detail to only the supported content;
2. choose a more specific contiguous solution_quote that supports those details; or
3. mark the case unresolved due to insufficient_solution_evidence.

Reference and named-architecture rules:
- If the solution_quote says a component uses a named architecture such as SN-PatchGAN or StyleGAN2 discriminator, gold_clarified_detail may name that architecture.
- Do not add details of that architecture, cite the reference paper, describe its internals, or mention the reference number unless the report text explicitly states those details.
- Correct example: "In the adversarial training component, implement the discriminator as an SN-PatchGAN discriminator."
- Correct example: "For the StylEx encoder, use the same architecture as the StyleGAN2 discriminator."
- Incorrect example: "Implement the discriminator as SN-PatchGAN as described in reference [17]."
- Incorrect example: "Use the StyleGAN2 discriminator architecture, including progressive growth, skip connections, and leaky ReLU activations" unless those exact details are explicitly stated in the report text.

Atomicity and granularity rules:
- Each resolved_gap_candidate should target one missing, ambiguous, or inconsistent implementation decision.
- If the report provides two independent clarified decisions, split them into two candidates unless they are inseparable parts of the same mechanism and share the same implementation slot.
- Do not merge architecture details with training hyperparameter protocol details unless they are inseparable in the same quoted solution.
- Do not include extra parameter ranges, optimizer settings, data annotations, preprocessing details, or evaluation details in gold_clarified_detail unless they are necessary for the target defect and explicitly supported by solution_quote.
- If a broad quote mentions multiple missing details, use the solution_quote and target implementation slot to decide whether to split into multiple atomic candidates.
- If one broad gap has several concrete sub-decisions solved together by the same implementation section and they form a single mechanism, keep them as one candidate.

Common human-revision failure cases to avoid:
- For named architectures, do not include external reference details in gold.
- For Blur+JPEG-like augmentation ambiguity, gold should resolve independent vs joint application unless the target defect specifically concerns parameter ranges.
- For dataset/classifier training-data gaps, do not add label counts or annotation descriptions unless supported by the solution_quote.
- For updated evaluation scripts, gold should state the missing processing step directly, not speculate about internal removal rules unless stated in the report.
- For figure-code architecture mismatches, gold should state the concrete connection or tensor-flow rule, not add unrelated module internals.
- For RGB scaling or normalization gaps, gold should state the target scaling range directly and avoid unnecessary native-range assumptions unless supported.

Evidence quote rules:
- gap_quote must be an exact contiguous snippet copied from the reproducibility report.
- solution_quote must be an exact contiguous snippet copied from the reproducibility report.
- Do NOT use ellipses such as "...".
- If evidence is spread across multiple places, choose the shortest single contiguous quote that supports the claim.
- If no single solution quote exists, the case is unresolved.
- source_specification_quote_from_original_paper is optional. Use it only if you find a direct original-paper snippet showing the underspecified/ambiguous/inconsistent source specification. Otherwise leave it empty.
- Do not use a broad generic quote if a more specific quote exists.
- If the solution_quote is short and only supports a high-level named behavior, keep gold_clarified_detail at the same level of specificity.

Candidate strength:
For resolved_gap_candidates:
- strong: clearly method-core, strong evidence, concrete quote support, gold is extractable, suitable for benchmark main set.
- borderline: potentially useful but includes ordinary hyperparameters, reproducer assumptions, weak original-paper evidence, generic quotes, mixed issues requiring manual review, or a gold that can only be written at a named-component level.
- weak: likely not suitable for the main benchmark.

For unresolved_gap_candidates:
- Since unresolved gaps cannot provide gold clarified specifications, they should usually be "borderline".
- Use "strong" for unresolved only if the gap evidence is highly specific, method-core, and directly supported by a precise report quote.
- If evidence is generic or partially specific, use "borderline".
- If the issue is mostly ordinary hyperparameter or unclear, use "weak" or invalid.

Fixed taxonomy:

Level-1 = Ambiguity

1. ambiguous formal definition
Definition: A symbol, notation, mathematical object, or formal rule is underspecified, leaving multiple plausible interpretations. The ambiguity changes what an implementer would compute or instantiate.

2. ambiguous method behavior
Definition: A method component, training procedure, inference rule, or evaluation behavior is described but its operational behavior is unclear. Multiple plausible implementations are possible and may lead to different results.

Level-1 = Incompleteness

3. missing algorithmic specification
Definition: A core algorithmic step, interface, update rule, routing decision, or procedural detail is omitted. Without this detail, the method cannot be faithfully implemented.

4. missing hyperparameter protocol
Definition: A result-sensitive hyperparameter is introduced, but the paper does not specify how its value is chosen, tuned, or validated. The issue is the missing selection protocol rather than a single ordinary unreported value.

5. missing model architecture
Definition: The paper states that a model or module is used but omits structural details such as layer type, normalization, pooling, activation, initialization, or dimensional mapping. These omissions materially affect the implemented model.

6. missing evaluation protocol
Definition: The evaluation setup is incomplete, including missing metric computation rules, evaluation data splits, prompt sets, thresholds, sampling procedures, or evaluation model configuration. The omission prevents faithful reproduction of reported results.

7. missing data/preprocessing protocol
Definition: The data construction, filtering, labeling, augmentation, normalization, tokenization, segmentation, or input transformation procedure is not fully specified. As a result, implementers may construct different inputs or supervision signals.

Level-1 = Inconsistency

8. inconsistent objective or loss
Definition: The paper and another source, such as code or an appendix, specify different training objectives, loss terms, reward definitions, or optimization targets. Following each source would optimize a materially different objective.

9. inconsistent architecture or pipeline
Definition: The paper and another source specify different model architectures, module configurations, data pipelines, preprocessing steps, or training/evaluation pipelines. The inconsistency makes it unclear which version should be followed.

10. inconsistent model specification
Definition: Two sources specify conflicting formal model assumptions, such as distributions, conditioning sets, aggregation rules, sampling support, or probabilistic/inference definitions. The discrepancy changes the underlying model being implemented.

Taxonomy decision rules:
- Assign exactly one Level-1 label for each kept gap.
- Assign exactly one Level-2 label from the list above.
- If two categories fit, choose the one that best describes the primary implementation blocker.
- If the issue is about architecture structure, layer split, activation placement, or named architecture choice, prefer "missing model architecture" unless it is explicitly a paper-code contradiction.
- If the issue is about metric computation, evaluation split, prompt set, threshold, sampling, or evaluator configuration, prefer "missing evaluation protocol".
- If the issue is about data construction, filtering, labeling, augmentation, normalization, tokenization, segmentation, or input transformation, prefer "missing data/preprocessing protocol".
- If the issue is about a core algorithmic procedure or update rule that is not evaluation/data/preprocessing, prefer "missing algorithmic specification".
- If the issue is about a missing tuning, initialization, stopping, selection, or hyperparameter protocol, prefer "missing hyperparameter protocol".
- If the issue is paper-code contradiction, prefer an inconsistency label.
- If the issue is a paper figure or diagram that implies a different architecture than the authors' code, prefer "Inconsistency / inconsistent architecture or pipeline".
- The Level-2 label must be consistent with the Level-1 label.

Return JSON only with exactly this schema:

{{
  "record_keep_assessment": {{
    "has_any_method_core_spec_gap": true/false,
    "has_any_resolved_real_gap": true/false,
    "has_any_unresolved_real_gap": true/false,
    "overall_reason": ""
  }},
  "resolved_gap_candidates": [
    {{
      "keep_criteria": {{
        "is_spec_gap": true/false,
        "is_actionable": true/false,
        "is_method_core_spec_gap": true/false,
        "gold_clarified_spec_extractable": true
      }},
      "gap_summary": "",
      "gap_quote": "",
      "solution_summary": "",
      "solution_quote": "",
      "solution_source_type": "author_clarification|code_derived|reproducer_assumption|reproducer_workaround",
      "source_specification_quote_from_original_paper": "",
      "affected_component": "input|output|core_method|algorithm|training|evaluation|implementation_detail|code_behavior|preprocessing|data|inference",
      "level1": "Ambiguity|Incompleteness|Inconsistency",
      "level2": "ambiguous formal definition|ambiguous method behavior|missing algorithmic specification|missing hyperparameter protocol|missing model architecture|missing evaluation protocol|missing data/preprocessing protocol|inconsistent objective or loss|inconsistent architecture or pipeline|inconsistent model specification",
      "taxonomy_rationale": "",
      "gold_clarified_detail": "",
      "why_this_blocks_or_affects_codification": "",
      "candidate_strength": "strong|borderline|weak",
      "strength_rationale": "",
      "confidence": 0.0
    }}
  ],
  "unresolved_gap_candidates": [
    {{
      "keep_criteria": {{
        "is_spec_gap": true/false,
        "is_actionable": true/false,
        "is_method_core_spec_gap": true/false,
        "gold_clarified_spec_extractable": false
      }},
      "gap_summary": "",
      "gap_quote": "",
      "why_unresolved": "",
      "affected_component": "input|output|core_method|algorithm|training|evaluation|implementation_detail|code_behavior|preprocessing|data|inference",
      "level1": "Ambiguity|Incompleteness|Inconsistency",
      "level2": "ambiguous formal definition|ambiguous method behavior|missing algorithmic specification|missing hyperparameter protocol|missing model architecture|missing evaluation protocol|missing data/preprocessing protocol|inconsistent objective or loss|inconsistent architecture or pipeline|inconsistent model specification",
      "taxonomy_rationale": "",
      "candidate_strength": "strong|borderline|weak",
      "strength_rationale": "",
      "confidence": 0.0
    }}
  ],
  "invalid_or_rejected_candidates": [
    {{
      "issue_summary": "",
      "evidence_quote": "",
      "rejection_reason": "resource_only|performance_only|ordinary_hyperparameter_value_only|uses_author_code_without_details|reproducer_improvement_not_spec_gap|insufficient_gap_evidence|insufficient_solution_evidence|not_method_core|duplicate_or_subsumed|other"
    }}
  ],
  "rejection_reason_if_no_gap": ""
}}

Duplicate and granularity rules:
- Do not split one gap into many candidates if they share the same missing slot and same solution.
- If one broad gap has several concrete sub-decisions, include it as one candidate if they are solved together by the same implementation section and form one mechanism.
- Split into multiple candidates when one quote contains clearly independent implementation decisions, such as a model architecture decision and a training hyperparameter protocol decision.
- If two candidates use nearly the same gap_quote and solution_quote, keep only the one with the clearer implementation blocker unless they target clearly distinct missing slots.
- Do not include both a broad generic gap and a more specific version of the same gap.
- Do not combine a gold detail about "what component/architecture to use" with a separate gold detail about "how to initialize/tune/train it" unless the report presents them as one inseparable mechanism.

Metadata:
{json.dumps(metadata, indent=2, ensure_ascii=False)}

Reproducibility report text:
\"\"\"
{report_pack.get("text", "")}
\"\"\"

Original paper text:
\"\"\"
{original_pack.get("text", "")}
\"\"\"
"""


# ============================================================
# Quote validation
# ============================================================

def quote_in_text(quote: str, text: str) -> bool:
    q = normalize_for_match(quote)
    t = normalize_for_match(text)

    if not q:
        return False

    if q in t:
        return True

    words = [w for w in re.findall(r"\w+", q) if len(w) > 3]
    if len(words) < 4:
        return False

    hit = sum(1 for w in words if w in t)
    return hit / max(len(words), 1) >= 0.75


def contains_any(text: str, terms: set) -> List[str]:
    low = text.lower()
    return sorted([t for t in terms if t in low])


def is_generic_gap_quote(quote: str) -> bool:
    q = normalize_for_match(quote)
    if not q:
        return True

    for p in GENERIC_GAP_PHRASES:
        if p in q:
            return True

    vague_terms = ["not specified", "not clear", "not provided", "no details"]
    has_vague = any(t in q for t in vague_terms)
    has_specific_anchor = any(
        t in q
        for t in [
            "encoder", "decoder", "loss", "objective", "architecture", "activation",
            "initialization", "sampling", "preprocessing", "postprocessing",
            "evaluation", "training", "split", "lambda", "covariance", "prior",
            "convolution", "batchnorm", "pool", "rnn", "gan", "discriminator",
            "metric", "auc", "normalization", "filter", "channel", "abf",
            "classifier", "augmentation", "jpeg", "blur",
        ]
    )
    return has_vague and not has_specific_anchor


def add_borderline_flags(g: Dict[str, Any]) -> Dict[str, Any]:
    g = dict(g)

    text = " ".join([
        str(g.get("gap_summary", "")),
        str(g.get("gap_quote", "")),
        str(g.get("solution_summary", "")),
        str(g.get("solution_quote", "")),
        str(g.get("gold_clarified_detail", "")),
        str(g.get("why_unresolved", "")),
        str(g.get("taxonomy_rationale", "")),
    ])

    ordinary_hparams = contains_any(text, ORDINARY_HPARAM_TERMS)
    improvement_terms = contains_any(text, IMPROVEMENT_TERMS)

    flags = g.get("borderline_flags") or {}
    flags["mentions_ordinary_hparams"] = bool(ordinary_hparams)
    flags["ordinary_hparam_terms"] = ordinary_hparams
    flags["mentions_reproducer_improvement_terms"] = bool(improvement_terms)
    flags["reproducer_improvement_terms"] = improvement_terms
    flags["generic_gap_quote"] = is_generic_gap_quote(g.get("gap_quote", ""))

    flags["needs_hparam_manual_review"] = (
        g.get("level2") == "missing hyperparameter protocol"
        and bool(ordinary_hparams)
    )
    flags["needs_improvement_manual_review"] = bool(improvement_terms)
    flags["needs_generic_quote_manual_review"] = bool(flags["generic_gap_quote"])

    g["borderline_flags"] = flags
    return g


# ============================================================
# Candidate validation
# ============================================================

def validate_resolved_gap(g: Dict[str, Any], report_text: str, original_text: str) -> Dict[str, Any]:
    g = dict(g)
    g = add_borderline_flags(g)

    gap_quote = g.get("gap_quote", "")
    solution_quote = g.get("solution_quote", "")
    source_quote = g.get("source_specification_quote_from_original_paper", "")

    level2 = normalize_level2_label(g.get("level2", ""))
    level1 = VALID_LEVEL2_TO_LEVEL1.get(level2, g.get("level1", ""))
    affected_component = normalize_affected_component(g.get("affected_component", ""))
    g["level1"] = level1
    g["level2"] = level2
    g["affected_component"] = affected_component
    solution_source_type = g.get("solution_source_type", "")
    candidate_strength = g.get("candidate_strength", "")

    kc = g.get("keep_criteria") or {}

    gap_noise = is_log_noise_quote(gap_quote)
    solution_noise = is_log_noise_quote(solution_quote)
    source_noise = is_log_noise_quote(source_quote) if source_quote else False

    g["quote_validation"] = {
        "gap_quote_in_report": False if gap_noise else quote_in_text(gap_quote, report_text),
        "solution_quote_in_report": False if solution_noise else quote_in_text(solution_quote, report_text),
        "source_specification_quote_in_original": (
            False
            if source_noise
            else (quote_in_text(source_quote, original_text) if source_quote else False)
        ),
        "gap_quote_is_log_noise": gap_noise,
        "solution_quote_is_log_noise": solution_noise,
        "source_quote_is_log_noise": source_noise,
        "gap_quote_has_no_ellipsis": "..." not in gap_quote,
        "solution_quote_has_no_ellipsis": "..." not in solution_quote,
        "gap_quote_nonempty": bool(str(gap_quote).strip()),
        "solution_quote_nonempty": bool(str(solution_quote).strip()),
        "gap_quote_is_generic": g["borderline_flags"]["generic_gap_quote"],
    }

    g["taxonomy_validation"] = {
        "level1": level1,
        "level2": level2,
        "level2_consistent_with_level1": validate_taxonomy_label(level1, level2),
        "affected_component_valid": validate_affected_component(affected_component),
    }

    g["solution_source_validation"] = {
        "solution_source_type": solution_source_type,
        "solution_source_type_valid": validate_solution_source_type(solution_source_type),
    }

    g["candidate_strength_validation"] = {
        "candidate_strength": candidate_strength,
        "candidate_strength_valid": validate_candidate_strength(candidate_strength),
    }

    g["keep_criteria_validation"] = {
        "is_spec_gap_true": kc.get("is_spec_gap") is True,
        "is_actionable_true": kc.get("is_actionable") is True,
        "is_method_core_spec_gap_true": kc.get("is_method_core_spec_gap") is True,
        "gold_clarified_spec_extractable_true": kc.get("gold_clarified_spec_extractable") is True,
    }

    g["passes_min_quote_validation"] = (
        g["quote_validation"]["gap_quote_nonempty"]
        and g["quote_validation"]["solution_quote_nonempty"]
        and not gap_noise
        and not solution_noise
        and g["quote_validation"]["gap_quote_in_report"]
        and g["quote_validation"]["solution_quote_in_report"]
        and g["quote_validation"]["gap_quote_has_no_ellipsis"]
        and g["quote_validation"]["solution_quote_has_no_ellipsis"]
    )

    g["passes_min_validation"] = (
        g["passes_min_quote_validation"]
        and g["taxonomy_validation"]["level2_consistent_with_level1"]
        and g["taxonomy_validation"]["affected_component_valid"]
        and g["solution_source_validation"]["solution_source_type_valid"]
        and g["candidate_strength_validation"]["candidate_strength_valid"]
        and g["keep_criteria_validation"]["is_spec_gap_true"]
        and g["keep_criteria_validation"]["is_actionable_true"]
        and g["keep_criteria_validation"]["is_method_core_spec_gap_true"]
        and g["keep_criteria_validation"]["gold_clarified_spec_extractable_true"]
    )

    g["recommended_manual_review"] = (
        candidate_strength != "strong"
        or solution_source_type in {"reproducer_assumption", "reproducer_workaround"}
        or g["borderline_flags"]["needs_hparam_manual_review"]
        or g["borderline_flags"]["needs_improvement_manual_review"]
        or g["borderline_flags"]["needs_generic_quote_manual_review"]
    )

    g["main_ready_candidate"] = (
        g["passes_min_validation"]
        and candidate_strength == "strong"
        and solution_source_type in {"author_clarification", "code_derived"}
        and not g["recommended_manual_review"]
    )

    return g


def validate_unresolved_gap(g: Dict[str, Any], report_text: str) -> Dict[str, Any]:
    g = dict(g)
    g = add_borderline_flags(g)

    if g.get("candidate_strength") == "strong":
        g["original_candidate_strength"] = "strong"
        g["candidate_strength"] = "borderline"
        g["unresolved_strength_downgraded"] = True
    else:
        g["original_candidate_strength"] = g.get("candidate_strength", "")
        g["unresolved_strength_downgraded"] = False

    gap_quote = g.get("gap_quote", "")
    level2 = normalize_level2_label(g.get("level2", ""))
    level1 = VALID_LEVEL2_TO_LEVEL1.get(level2, g.get("level1", ""))
    affected_component = normalize_affected_component(g.get("affected_component", ""))
    g["level1"] = level1
    g["level2"] = level2
    g["affected_component"] = affected_component
    candidate_strength = g.get("candidate_strength", "")

    kc = g.get("keep_criteria") or {}
    gap_noise = is_log_noise_quote(gap_quote)

    g["quote_validation"] = {
        "gap_quote_in_report": False if gap_noise else quote_in_text(gap_quote, report_text),
        "gap_quote_has_no_ellipsis": "..." not in gap_quote,
        "gap_quote_nonempty": bool(str(gap_quote).strip()),
        "gap_quote_is_generic": g["borderline_flags"]["generic_gap_quote"],
        "gap_quote_is_log_noise": gap_noise,
    }

    g["taxonomy_validation"] = {
        "level1": level1,
        "level2": level2,
        "level2_consistent_with_level1": validate_taxonomy_label(level1, level2),
        "affected_component_valid": validate_affected_component(affected_component),
    }

    g["candidate_strength_validation"] = {
        "candidate_strength": candidate_strength,
        "candidate_strength_valid": validate_candidate_strength(candidate_strength),
    }

    g["keep_criteria_validation"] = {
        "is_spec_gap_true": kc.get("is_spec_gap") is True,
        "is_actionable_true": kc.get("is_actionable") is True,
        "is_method_core_spec_gap_true": kc.get("is_method_core_spec_gap") is True,
        "gold_clarified_spec_extractable_false": kc.get("gold_clarified_spec_extractable") is False,
    }

    g["passes_min_quote_validation"] = (
        g["quote_validation"]["gap_quote_nonempty"]
        and not gap_noise
        and g["quote_validation"]["gap_quote_in_report"]
        and g["quote_validation"]["gap_quote_has_no_ellipsis"]
    )

    g["passes_min_validation"] = (
        g["passes_min_quote_validation"]
        and g["taxonomy_validation"]["level2_consistent_with_level1"]
        and g["taxonomy_validation"]["affected_component_valid"]
        and g["candidate_strength_validation"]["candidate_strength_valid"]
        and g["keep_criteria_validation"]["is_spec_gap_true"]
        and g["keep_criteria_validation"]["is_actionable_true"]
        and g["keep_criteria_validation"]["is_method_core_spec_gap_true"]
        and g["keep_criteria_validation"]["gold_clarified_spec_extractable_false"]
    )

    g["recommended_manual_review"] = True
    g["main_ready_candidate"] = False

    return g


def validate_invalid_candidate(g: Dict[str, Any], report_text: str) -> Dict[str, Any]:
    g = dict(g)

    evidence_quote = g.get("evidence_quote", "")
    rejection_reason = g.get("rejection_reason", "")
    evidence_noise = is_log_noise_quote(evidence_quote) if evidence_quote else False

    g["quote_validation"] = {
        "evidence_quote_in_report": (
            False
            if evidence_noise
            else (quote_in_text(evidence_quote, report_text) if evidence_quote else False)
        ),
        "evidence_quote_has_no_ellipsis": "..." not in evidence_quote,
        "evidence_quote_nonempty": bool(str(evidence_quote).strip()),
        "evidence_quote_is_log_noise": evidence_noise,
    }

    g["rejection_validation"] = {
        "rejection_reason": rejection_reason,
        "rejection_reason_valid": validate_rejection_reason(rejection_reason),
    }

    g["passes_invalid_candidate_validation"] = (
        g["rejection_validation"]["rejection_reason_valid"]
        and (
            not evidence_quote
            or (
                not evidence_noise
                and g["quote_validation"]["evidence_quote_in_report"]
                and g["quote_validation"]["evidence_quote_has_no_ellipsis"]
            )
        )
    )

    return g


def validate_extraction(obj: Dict[str, Any], report_text: str, original_text: str) -> Dict[str, Any]:
    obj = dict(obj)

    resolved_validated = [
        validate_resolved_gap(g, report_text=report_text, original_text=original_text)
        for g in obj.get("resolved_gap_candidates", []) or []
    ]

    unresolved_validated = [
        validate_unresolved_gap(g, report_text=report_text)
        for g in obj.get("unresolved_gap_candidates", []) or []
    ]

    invalid_validated = [
        validate_invalid_candidate(g, report_text=report_text)
        for g in obj.get("invalid_or_rejected_candidates", []) or []
    ]

    obj["resolved_gap_candidates"] = resolved_validated
    obj["unresolved_gap_candidates"] = unresolved_validated
    obj["invalid_or_rejected_candidates"] = invalid_validated

    obj["validation_summary"] = {
        "num_resolved_candidates": len(resolved_validated),
        "num_unresolved_candidates": len(unresolved_validated),
        "num_invalid_or_rejected_candidates": len(invalid_validated),
        "num_resolved_pass_min_validation": sum(bool(g.get("passes_min_validation")) for g in resolved_validated),
        "num_unresolved_pass_min_validation": sum(bool(g.get("passes_min_validation")) for g in unresolved_validated),
        "num_invalid_pass_validation": sum(bool(g.get("passes_invalid_candidate_validation")) for g in invalid_validated),
        "num_resolved_strong": sum(g.get("candidate_strength") == "strong" for g in resolved_validated),
        "num_unresolved_strong_after_downgrade": sum(g.get("candidate_strength") == "strong" for g in unresolved_validated),
        "num_unresolved_original_strong": sum(g.get("original_candidate_strength") == "strong" for g in unresolved_validated),
        "num_resolved_manual_review": sum(bool(g.get("recommended_manual_review")) for g in resolved_validated),
        "num_unresolved_manual_review": sum(bool(g.get("recommended_manual_review")) for g in unresolved_validated),
        "num_resolved_main_ready": sum(bool(g.get("main_ready_candidate")) for g in resolved_validated),
    }

    return obj


# ============================================================
# Output item builders
# ============================================================

def build_base_record(record: Dict[str, Any], report_pack: Dict[str, str], original_pack: Dict[str, str]) -> Dict[str, Any]:
    return {
        "record_id": record.get("record_id"),
        "source": record.get("source"),
        "report_title": record.get("report_title"),
        "report_url": record.get("report_url"),
        "report_pdf_url": record.get("report_pdf_url"),
        "original_paper_title": record.get("original_paper_title"),
        "original_paper_url": record.get("original_paper_url"),
        "original_paper_pdf_url": record.get("original_paper_pdf_url"),
        "year": record.get("year"),
        "report_text_source": report_pack.get("source"),
        "report_text_path": report_pack.get("path"),
        "original_text_source": original_pack.get("source"),
        "original_text_path": original_pack.get("path"),
    }


def candidate_passes_filters(
    g: Dict[str, Any],
    keep_only_validated: bool,
    keep_only_strong: bool,
    keep_only_main_ready: bool,
) -> bool:
    if keep_only_validated and not g.get("passes_min_validation"):
        return False
    if keep_only_strong and g.get("candidate_strength") != "strong":
        return False
    if keep_only_main_ready and not g.get("main_ready_candidate"):
        return False
    return True


def make_resolved_item(
    base: Dict[str, Any],
    record: Dict[str, Any],
    candidate: Dict[str, Any],
    j: int,
) -> Dict[str, Any]:
    return {
        **base,
        "realgap_id": f"{record.get('record_id')}_resolved_gap_{j+1:03d}",
        "gap": candidate,
        "route": "resolved_real_gap",
        "postprocess_split": "main_resolved",
        "postprocess_reasons": ["main_ready_candidate_true"] if candidate.get("main_ready_candidate") else [],
        "source_record": slim_source_record(record),
    }


def make_unresolved_item(
    base: Dict[str, Any],
    record: Dict[str, Any],
    candidate: Dict[str, Any],
    j: int,
) -> Dict[str, Any]:
    return {
        **base,
        "unresolved_gap_id": f"{record.get('record_id')}_unresolved_gap_{j+1:03d}",
        "gap": candidate,
        "route": "unresolved_real_gap_audit_only",
        "source_record": slim_source_record(record),
    }


def make_invalid_item(
    base: Dict[str, Any],
    record: Dict[str, Any],
    candidate: Dict[str, Any],
    j: int,
) -> Dict[str, Any]:
    return {
        **base,
        "invalid_candidate_id": f"{record.get('record_id')}_invalid_{j+1:03d}",
        "candidate": candidate,
        "route": "invalid_or_rejected",
        "source_record": slim_source_record(record),
    }


# ============================================================
# Resume helpers
# ============================================================

def has_valid_detailed_output(r: Dict[str, Any]) -> bool:
    obj = r.get("detailed_realgap_extraction")
    if not isinstance(obj, dict):
        return False
    return isinstance(obj.get("validation_summary"), dict)


def load_prior_outputs(out_path: Path) -> Dict[str, Dict[str, Any]]:
    prior = {}
    if not out_path.exists():
        return prior

    rows = safe_load_jsonl_if_exists(out_path)
    for r in rows:
        rid = str(r.get("record_id") or "").strip()
        if rid:
            prior[rid] = r
    return prior


# ============================================================
# Main extraction
# ============================================================

def detailed_extract(
    input_path: Path,
    out_path: Path,
    resolved_path: Path,
    unresolved_path: Path,
    invalid_path: Path,
    summary_path: Path,
    source_filter: str,
    model: str,
    max_report_chars: int,
    max_original_chars: int,
    keep_only_validated: bool,
    keep_only_strong: bool,
    keep_only_main_ready: bool,
    resume: bool,
    sleep: float,
):

    rows_all = load_jsonl(input_path)

    rows_after_source = [
        r for r in rows_all
        if source_matches(r, source_filter)
    ]

    rows = [
        r for r in rows_after_source
        if should_run_detailed(r)
    ]


    prior_by_id = load_prior_outputs(out_path) if resume else {}

    client = build_client()

    all_outputs_by_id: Dict[str, Dict[str, Any]] = {}
    all_order: List[str] = []

    resolved_items: List[Dict[str, Any]] = []
    unresolved_items: List[Dict[str, Any]] = []
    invalid_items: List[Dict[str, Any]] = []

    print("\n===== Universal Step 05: Detailed Real-Gap Extraction =====")
    print(f"Input path: {input_path}")
    print(f"Output path: {out_path}")
    print(f"Resolved path: {resolved_path}")
    print(f"Unresolved path: {unresolved_path}")
    print(f"Invalid path: {invalid_path}")
    print(f"Summary path: {summary_path}")
    print(f"Input records before source filter: {len(rows_all)}")
    print(f"Input records after source filter: {len(rows)}")
    print(f"Source filter: {source_filter}")
    print(f"Model: {model}")
    print(f"keep_only_validated: {keep_only_validated}")
    print(f"keep_only_strong: {keep_only_strong}")
    print(f"keep_only_main_ready: {keep_only_main_ready}")
    print(f"Resume: {resume}")
    print("==========================================================\n")

    for idx, record in enumerate(
        tqdm(rows, desc="Step 05: detailed extraction + validation"),
        start=1,
    ):
        rid = str(record.get("record_id") or f"record_{idx:05d}").strip()
        all_order.append(rid)

        if resume and rid in prior_by_id and has_valid_detailed_output(prior_by_id[rid]):
            out = prior_by_id[rid]
            all_outputs_by_id[rid] = out
            obj = out.get("detailed_realgap_extraction") or {}
            report_pack = {
                "source": out.get("report_text_source", "prior"),
                "path": out.get("report_text_path", ""),
                "text": "",
            }
            original_pack = {
                "source": out.get("original_text_source", "prior"),
                "path": out.get("original_text_path", ""),
                "text": "",
            }
            base = build_base_record(out, report_pack, original_pack)

        else:
            if not should_run_detailed(record):
                out = dict(record)
                route = record.get("llm_route_classification") or {}
                out["detailed_extraction_skipped"] = True
                out["detailed_extraction_skip_reason"] = (
                    "Step 05 only runs for primary_route=resolved_real_gap and "
                    "should_run_detailed_extraction=True; got "
                    f"primary_route={route.get('primary_route')}, "
                    f"should_run_detailed_extraction={route.get('should_run_detailed_extraction')}"
                )
                all_outputs_by_id[rid] = out
                continue

            report_pack = get_best_report_text(record, max_chars=max_report_chars)
            original_pack = get_best_original_text(record, max_chars=max_original_chars)

            if len(report_pack.get("text", "").strip()) <= 100:
                obj = {
                    "record_keep_assessment": {
                        "has_any_method_core_spec_gap": False,
                        "has_any_resolved_real_gap": False,
                        "has_any_unresolved_real_gap": False,
                        "overall_reason": "Skipped because report text is missing or too short.",
                    },
                    "resolved_gap_candidates": [],
                    "unresolved_gap_candidates": [],
                    "invalid_or_rejected_candidates": [],
                    "rejection_reason_if_no_gap": "missing_or_too_short_report_text",
                    "validation_summary": {
                        "num_resolved_candidates": 0,
                        "num_unresolved_candidates": 0,
                        "num_invalid_or_rejected_candidates": 0,
                        "num_resolved_pass_min_validation": 0,
                        "num_unresolved_pass_min_validation": 0,
                        "num_invalid_pass_validation": 0,
                        "num_resolved_strong": 0,
                        "num_unresolved_strong_after_downgrade": 0,
                        "num_unresolved_original_strong": 0,
                        "num_resolved_manual_review": 0,
                        "num_unresolved_manual_review": 0,
                        "num_resolved_main_ready": 0,
                    },
                }
            else:
                try:
                    raw_obj = call_llm_json(
                        client=client,
                        model=model,
                        prompt=build_prompt(record, report_pack, original_pack),
                        temperature=0.0,
                    )

                    obj = validate_extraction(
                        raw_obj,
                        report_text=report_pack.get("text", ""),
                        original_text=original_pack.get("text", ""),
                    )

                except Exception as e:
                    obj = {
                        "record_keep_assessment": {
                            "has_any_method_core_spec_gap": False,
                            "has_any_resolved_real_gap": False,
                            "has_any_unresolved_real_gap": False,
                            "overall_reason": f"LLM detailed extraction failed: {repr(e)}",
                        },
                        "resolved_gap_candidates": [],
                        "unresolved_gap_candidates": [],
                        "invalid_or_rejected_candidates": [],
                        "rejection_reason_if_no_gap": f"LLM detailed extraction failed: {repr(e)}",
                        "validation_summary": {
                            "num_resolved_candidates": 0,
                            "num_unresolved_candidates": 0,
                            "num_invalid_or_rejected_candidates": 0,
                            "num_resolved_pass_min_validation": 0,
                            "num_unresolved_pass_min_validation": 0,
                            "num_invalid_pass_validation": 0,
                            "num_resolved_strong": 0,
                            "num_unresolved_strong_after_downgrade": 0,
                            "num_unresolved_original_strong": 0,
                            "num_resolved_manual_review": 0,
                            "num_unresolved_manual_review": 0,
                            "num_resolved_main_ready": 0,
                        },
                    }

            out = dict(record)
            out["detailed_extraction_skipped"] = False
            out["detailed_realgap_extraction"] = obj
            all_outputs_by_id[rid] = out

            base = build_base_record(record, report_pack, original_pack)

        # Build candidate-level output files from current/prior detailed object.
        obj = all_outputs_by_id[rid].get("detailed_realgap_extraction") or {}

        for j, g in enumerate(obj.get("resolved_gap_candidates", []) or []):
            if not candidate_passes_filters(
                g,
                keep_only_validated=keep_only_validated,
                keep_only_strong=keep_only_strong,
                keep_only_main_ready=keep_only_main_ready,
            ):
                continue
            resolved_items.append(make_resolved_item(base, all_outputs_by_id[rid], g, j))

        for j, g in enumerate(obj.get("unresolved_gap_candidates", []) or []):
            # Unresolved gaps are audit-only. They cannot construct gold.
            if keep_only_validated and not g.get("passes_min_validation"):
                continue
            if keep_only_strong and g.get("candidate_strength") != "strong":
                continue
            if keep_only_main_ready:
                continue
            unresolved_items.append(make_unresolved_item(base, all_outputs_by_id[rid], g, j))

        for j, g in enumerate(obj.get("invalid_or_rejected_candidates", []) or []):
            invalid_items.append(make_invalid_item(base, all_outputs_by_id[rid], g, j))

        # Checkpoint after each record.
        save_jsonl([all_outputs_by_id[x] for x in all_order], out_path)
        save_jsonl(resolved_items, resolved_path)
        save_jsonl(unresolved_items, unresolved_path)
        save_jsonl(invalid_items, invalid_path)

        vs = obj.get("validation_summary", {})
        print(
            f"[{idx}/{len(rows)}] "
            f"source={record.get('source')} "
            f"route={get_primary_route(record)} "
            f"report={report_pack.get('source')} "
            f"original={original_pack.get('source')} "
            f"resolved={vs.get('num_resolved_candidates')} "
            f"main_ready={vs.get('num_resolved_main_ready')} "
            f"title={str(record.get('report_title'))[:80]}"
        )

        if sleep and sleep > 0:
            time.sleep(sleep)

    save_jsonl([all_outputs_by_id[x] for x in all_order], out_path)
    save_jsonl(resolved_items, resolved_path)
    save_jsonl(unresolved_items, unresolved_path)
    save_jsonl(invalid_items, invalid_path)

    route_counter = Counter()
    source_counter = Counter()
    report_text_source_counter = Counter()
    original_text_source_counter = Counter()
    num_should_run = 0
    num_skipped = 0
    num_records_with_resolved = 0
    num_records_with_main_ready = 0

    total_resolved_candidates = 0
    total_main_ready_candidates = 0

    for rid in all_order:
        r = all_outputs_by_id[rid]
        source_counter[r.get("source")] += 1
        route_counter[get_primary_route(r)] += 1

        if r.get("detailed_extraction_skipped"):
            num_skipped += 1

        if should_run_detailed(r):
            num_should_run += 1

        obj = r.get("detailed_realgap_extraction") or {}
        vs = obj.get("validation_summary") or {}

        n_res = int(vs.get("num_resolved_candidates") or 0)
        n_main = int(vs.get("num_resolved_main_ready") or 0)

        total_resolved_candidates += n_res
        total_main_ready_candidates += n_main

        if n_res > 0:
            num_records_with_resolved += 1
        if n_main > 0:
            num_records_with_main_ready += 1

        report_text_source_counter[r.get("report_text_source", "unknown")] += 1
        original_text_source_counter[r.get("original_text_source", "unknown")] += 1

    summary = {
        "script": "step_05_detailed_extract_and_quote_validate.py",
        "input_path": str(input_path),
        "out_path": str(out_path),
        "resolved_path": str(resolved_path),
        "unresolved_path": str(unresolved_path),
        "invalid_path": str(invalid_path),
        "records_before_source_filter": len(rows_all),
        "records_after_source_filter": len(rows),
        "source_filter": source_filter,
        "model": model,
        "source_distribution": dict(source_counter),
        "route_distribution": dict(route_counter),
        "report_text_source_distribution": dict(report_text_source_counter),
        "original_text_source_distribution": dict(original_text_source_counter),
        "records_should_run_detailed": num_should_run,
        "records_skipped": num_skipped,
        "records_with_resolved_candidates": num_records_with_resolved,
        "records_with_main_ready_candidates": num_records_with_main_ready,
        "total_resolved_candidates_extracted": total_resolved_candidates,
        "total_main_ready_candidates_extracted": total_main_ready_candidates,
        "resolved_candidates_saved": len(resolved_items),
        "unresolved_audit_only_candidates_saved": len(unresolved_items),
        "invalid_or_rejected_candidates_saved": len(invalid_items),
        "keep_only_validated": keep_only_validated,
        "keep_only_strong": keep_only_strong,
        "keep_only_main_ready": keep_only_main_ready,
        "next_step": (
            "Human-check resolved candidates, revise gold_clarified_detail / taxonomy / atomicity, "
            "then build final real-gap benchmark instances."
        ),
    }

    save_json(summary, summary_path)

    print("\n===== Step 05 Summary =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", required=True)

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
        "--out_path",
        default="",
        help="All detailed extraction outputs. If omitted, inferred from dataset_tag.",
    )
    parser.add_argument(
        "--resolved_path",
        default="",
        help="Resolved real-gap candidates output. If omitted, inferred from dataset_tag.",
    )
    parser.add_argument(
        "--unresolved_path",
        default="",
        help="Unresolved audit-only candidates output. If omitted, inferred from dataset_tag.",
    )
    parser.add_argument(
        "--invalid_path",
        default="",
        help="Invalid/rejected candidates output. If omitted, inferred from dataset_tag.",
    )
    parser.add_argument(
        "--summary_path",
        default="",
        help="Summary JSON output. If omitted, inferred from out_path.",
    )

    parser.add_argument(
        "--model",
        default=os.getenv("OPENROUTER_MODEL") or os.getenv("OPENAI_MODEL") or "deepseek/deepseek-v4-pro",
    )

    parser.add_argument("--max_report_chars", type=int, default=80000)
    parser.add_argument("--max_original_chars", type=int, default=50000)

    parser.add_argument(
        "--keep_only_validated",
        action="store_true",
        help="Only save candidates that pass quote + taxonomy + keep-criteria validation.",
    )

    parser.add_argument(
        "--keep_only_strong",
        action="store_true",
        help="Only save candidates with candidate_strength == strong.",
    )

    parser.add_argument(
        "--keep_only_main_ready",
        action="store_true",
        help=(
            "Only save resolved candidates that are main_ready_candidate=True. "
            "Unresolved candidates are not saved in this mode."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing --out_path and skip records with detailed output.",
    )

    parser.add_argument("--sleep", type=float, default=0.0)

    args = parser.parse_args()

    input_path = Path(args.input_path)
    rows_for_tag = [
        r for r in load_jsonl(input_path)
        if source_matches(r, args.source_filter)
    ]
    dataset_tag = args.dataset_tag.strip() or infer_dataset_tag(rows_for_tag, args.source_filter)

    default_dir = Path(f"Real_bench/{dataset_tag}/outputs")
    default_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out_path) if args.out_path else default_dir / f"{dataset_tag}_detailed_extraction_all.jsonl"
    resolved_path = Path(args.resolved_path) if args.resolved_path else default_dir / f"{dataset_tag}_resolved_real_gap_candidates_main_ready.jsonl"
    unresolved_path = Path(args.unresolved_path) if args.unresolved_path else default_dir / f"{dataset_tag}_unresolved_audit_only.jsonl"
    invalid_path = Path(args.invalid_path) if args.invalid_path else default_dir / f"{dataset_tag}_invalid_or_rejected_candidates.jsonl"
    summary_path = Path(args.summary_path) if args.summary_path else out_path.with_suffix(".summary.json")

    detailed_extract(
        input_path=input_path,
        out_path=out_path,
        resolved_path=resolved_path,
        unresolved_path=unresolved_path,
        invalid_path=invalid_path,
        summary_path=summary_path,
        source_filter=args.source_filter,
        model=args.model,
        max_report_chars=args.max_report_chars,
        max_original_chars=args.max_original_chars,
        keep_only_validated=args.keep_only_validated,
        keep_only_strong=args.keep_only_strong,
        keep_only_main_ready=args.keep_only_main_ready,
        resume=args.resume,
        sleep=args.sleep,
    )


if __name__ == "__main__":
    main()
