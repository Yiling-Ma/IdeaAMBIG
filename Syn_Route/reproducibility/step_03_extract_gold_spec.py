import os
import re
import json
import time
import argparse
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from openai import OpenAI


REPRODUCIBILITY_DIR = Path(__file__).resolve().parent
PAPER_META_MARKDOWN_DIR = REPRODUCIBILITY_DIR / "paper_meta" / "markdown"


SPEC_FIELDS = [
    "paper_title",
    "research_goal",
    "task",
    "inputs",
    "outputs",
    "core_method",
    "algorithm_steps",
    "training_or_optimization",
    "datasets",
    "evaluation_metrics",
    "baselines",
    "implementation_details",
    "reproducibility_relevant_details",
    "unknown_fields",
]

LIST_FIELDS = {
    "algorithm_steps",
    "datasets",
    "evaluation_metrics",
    "baselines",
    "implementation_details",
    "reproducibility_relevant_details",
    "unknown_fields",
}

RESULT_CLAIM_PATTERNS = [
    r"\byields?\b.*\b(gain|improvement|higher|better|substantial|significant)",
    r"\bachieves?\b.*\b(state-of-the-art|best|higher|better)",
    r"\boutperforms?\b",
    r"\bresults?\s+(show|indicate|demonstrate)",
    r"\bshows?\b.*\b(faster|better|improved|stronger|superior)",
    r"\bimproves?\b.*\b(robustness|accuracy|performance|convergence)",
    r"\bavoids?\b.*\b(collapse|degenerate|failure)",
    r"\breduces?\b.*\b(error|attack success|distortion)",
    r"\bleads?\s+to\b.*\b(robust|accurate|compact|separated|better)",
]

MECHANISM_CLAIM_PHRASES = [
    "shows faster convergence",
    "faster convergence",
    "avoids degenerate collapse",
    "avoids collapse",
    "shows improved",
    "improves robustness",
    "high sample density",
    "intra-class compactness",
    "maintaining separation",
]

UNSUPPORTED_DECISION_RULE_PATTERNS = [
    r"\bstandard softmax logits\b",
    r"\bsoftmax logits are only applied at inference\b",
    r"\bsoftmax is only applied at inference\b",
    r"\bsoftmax\b.*\be\.g\.",
    r"\be\.g\.,?\s*negative squared distances\b",
]

METADATA_PHRASES_IN_REFERENCE = [
    "the repository provides",
    "the code is available",
    "available in the code",
    "not fully reproduced here",
    "requires an api key",
    "api key",
    "secrets file",
    "local path",
    "slurm",
    "gpu",
    "cuda",
    "reproducibility report",
    "synthetic",
    "gold reference",
    "route classification",
]

UNSUPPORTED_TOOL_CLAIM_PATTERNS = [
    r"\bexternal retrieval\b",
    r"\bweb search\b",
    r"\bsearch engine\b",
    r"\bhuman evaluation\b",
    r"\bstatistical significance\b",
    r"\bablation study\b",
]

TUNABLE_RECIPE_PATTERNS = [
    r"\blearning rate\b",
    r"\blr\s*=?\s*[0-9]",
    r"\bbatch size\b",
    r"\bepochs?\b",
    r"\bweight decay\b",
    r"\bmomentum\b",
    r"\bdropout\b",
    r"\brandom seed\b",
    r"\bseed\b",
]

DENSE_FORMULA_PATTERNS = [
    r"\\prod",
    r"\\sum",
    r"\\partial",
    r"\\theta",
    r"\\phi",
    r"\\lambda",
    r"\\exp",
    r"\\log",
    r"\bp\([^)]*\)\s*=",
    r"\bL\s*=",
    r"\|\|",
    r"∏",
    r"∑",
    r"∂",
    r"λ",
    r"φ",
    r"θ",
]

NUMERIC_CONFIG_PATTERNS = [
    r"\b\d+\s+flow steps\b",
    r"\b\d+\s+bins\b",
    r"\b\d+\s+heads\b",
    r"\bhidden dimension\s+\d+\b",
    r"\bintermediate dimension\s+\d+\b",
    r"\b\d+\s+layers\b",
    r"\be\.g\.,?\s*six\b",
    r"\be\.g\.,?\s*\d+\b",
]

OPTIMIZER_NAME_PATTERNS = [
    r"\busing adam\b",
    r"\bwith adam\b",
    r"\busing sgd\b",
    r"\bwith sgd\b",
    r"\bstandard optimizers?\b",
]

EXPERIMENT_ONLY_PHASE_PATTERNS = [
    r"\(\s*1\s*\)\s*data generation",
    r"\bphase[s]?:\s*\(1\)\s*data generation",
    r"\bdata generation:\s*synthetic sources\b",
]

UNCERTAINTY_IN_REFERENCE_PATTERNS = [
    r"\bnot explicitly specified\b",
    r"\bnot specified in the paper\b",
    r"\bexact .* is not specified\b",
]

LOW_LEVEL_SYMBOLIC_PATTERNS = [
    r"\bdelta\s*=",
    r"\bδ\s*=",
    r"\br_δ\b",
    r"\bv\^\(",
    r"\bW_q",
    r"\bW_key\b",
    r"\blog[- ]?jacobian",
    r"\bjacobian\b",
    r"\bξ_i\b",
    r"\bη_i\b",
]

ROUTINE_LOSS_PATTERNS = [
    r"\btraining minimizes cross[- ]entropy\b",
    r"\btraining uses cross[- ]entropy\b",
    r"\bcross[- ]entropy loss using\b",
]

EQUATION_LIKE_REFERENCE_PATTERNS = [
    r"\b[A-Za-z][A-Za-z0-9_^\{\}\\]*\s*=",
    r"=\s*[^,.;]{8,}",
]

GOLD_CODIFICATION_READY_REFERENCE_DEFINITION = """
Gold codification-ready reference:
The codification_ready_reference is the evidence-grounded, defect-free implementation
specification for the target research idea or method. It is not a paper summary, not
the original abstract, and not a list of reproducibility trivia. It is the clarified
method specification that an implementer would need in order to faithfully codify the
core method.

The intended style is a compact method handoff specification, not a dense reproduction
record. It should read like: "The method, <name>, is a <method type> for <task>. Given
<inputs>, it proceeds in <N> phases: (1) ..., (2) ..., ... . The approach is evaluated
on <datasets/tasks> against <baselines> using <metrics>." Prefer this phase/step
structure when the paper describes a multi-stage method.

The reference should explain the implementation plan at the same level as a senior
researcher handing off a method to an engineer. It should not be a compressed equation
section. Use names and operational descriptions for losses, priors, target values,
attention rules, and adjustment procedures. By default, do not include symbolic
equations in codification_ready_reference; put exact equations in the structured
paper_derived_specification fields.

For this reproducibility synthetic-controlled route, the original paper is the primary
evidence source for the reference. The associated reproducibility report is used only
to justify that the paper-report pair was routed to synthetic controlled: the report
indicates successful reproduction and does not provide a resolved method-core gap. Do
not use the report as a source of artificial gaps.

It should include implementation-facing details when supported by the original paper,
such as:
- task definition;
- concrete inputs and outputs;
- core algorithmic procedure;
- model, module, architecture, or representation roles;
- training, optimization, loss, inference, routing, decoding, or parsing logic;
- data construction, preprocessing, filtering, sampling, or input formatting;
- evaluation protocol, metric computation, baselines, and comparison setup.

It should exclude:
- unsupported assumptions or guessed details;
- broad motivation, background, or contribution summaries;
- result claims or claimed advantages such as "outperforms", "achieves better",
  "results show", "improves robustness", "shows faster convergence", "avoids
  collapse", or "induces compactness";
- unknown fields, caveats, limitations, or unresolved contradictions;
- benchmark-construction metadata or synthetic-defect language;
- API keys, credentials, local paths, hardware-only details, runtime-only details,
  and environment setup unless they are themselves part of the scientific method.
- low-level mathematical expansions, derivations, density normalizers, exact likelihood
  formulas, or architecture counts when a concise operational description would still
  let an implementer build the method.
- experiment-only data generation procedures unless synthetic data generation is itself
  the proposed method;
- ordinary optimizer choices, training schedules, data augmentation settings, exact
  layer counts, hidden dimensions, number of heads, number of bins, or number of epochs
  unless they define the method interface or a non-standard mechanism.

Do not infer prediction or inference rules from common practice. If the paper does not
explicitly specify whether prediction uses center distance, logits, softmax, nearest
prototype, decoding, thresholding, or another decision rule, write the decision rule
conservatively as "the paper-specified decision rule" and place the uncertainty in
unknown_fields.

Ordinary tunable training recipe details such as learning rate, batch size, epoch count,
momentum, weight decay, dropout rate, random seed, optimizer schedule, and hardware
should NOT appear in codification_ready_reference unless the value defines a non-standard
scientific mechanism of the method. Put these details in reproducibility_relevant_details
instead. The reference should describe what must be implemented, not one plausible
hyperparameter setting for training it.

Mathematical formulas should appear in codification_ready_reference only when they define
a method-critical object, update rule, loss, scoring function, inference rule, or data
transformation. Do not include long derivations or secondary equations that are not needed
to build the initial implementation.

Summarize formulas by role rather than spelling them out. For example, say "train the
flow by exact conditional likelihood under a factorized exponential-family prior" rather
than writing every term of the likelihood. Say "use a squared Euclidean regression loss
to pull features toward fixed class centers" rather than writing the full loss formula.
For reinforcement learning, say how the target value is constructed rather than writing
the full Bellman equation.

Before returning JSON, revise codification_ready_reference by removing:
1. ordinary optimizer or training recipe details;
2. long equations and normalizers;
3. numeric architecture counts that are not method-defining;
4. claimed advantages or empirical conclusions;
5. benchmark data generation procedures that are not part of the method;
6. any text that says an exact detail is not specified;
7. invented examples of unspecified decision rules, such as "e.g., negative distances"
   when the paper does not state that rule.

Use these style conversions:
- Bad: "δ = k-q, r_δ = (...), v = ..., softmax(v^T r_δ) peaks at ..."
  Good: "each head computes attention from a quadratic relative-position encoding with
  content projections suppressed, causing heads to specialize to relative pixel shifts."
- Bad: "The method improves robustness by inducing high sample density."
  Good: "The method trains features with a fixed-center regression objective and is
  evaluated under adversarial attacks."
- Bad: "classification uses softmax scores, e.g., negative squared distances" when the
  paper does not state the scoring rule.
  Good: "predictions use the paper-specified center-based decision rule" or omit the
  decision rule if it is not needed for the method handoff.
- Bad: "training uses SGD/Adam/cross-entropy" when that is routine and not the proposed
  contribution.
  Good: omit it from codification_ready_reference and place it in reproducibility_relevant_details.
""".strip()


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


def save_json(obj: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_jsonl(rows: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def safe_filename(x: Any, max_len: int = 160) -> str:
    s = str(x or "").strip()
    s = re.sub(r"[^\w\-.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if s else "unknown"


def safe_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            text = match.group(0)

    return json.loads(text.strip())


def normalize_ws(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def word_count(text: Any) -> int:
    return len(normalize_ws(text).split())


def safe_regex_search(pattern: str, text: str) -> bool:
    try:
        return re.search(pattern, text, flags=re.I) is not None
    except re.error:
        return False


def contains_any_pattern(text: str, patterns: List[str]) -> bool:
    return any(safe_regex_search(pattern, text) for pattern in patterns)


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = normalize_ws(value)
        return [value] if value else []
    return [value]


def normalize_string(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return normalize_ws(json.dumps(value, ensure_ascii=False))
    return normalize_ws(value)


# ============================================================
# Client
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


def extract_sse_content(text: str) -> str:
    """
    Compatibility for proxies that return SSE-style raw strings.
    """
    contents = []

    for line in text.splitlines():
        line = line.strip()

        if not line.startswith("data:"):
            continue

        payload = line[len("data:"):].strip()

        if payload == "[DONE]":
            continue

        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue

        choices = obj.get("choices", [])

        for choice in choices:
            delta = choice.get("delta", {})
            if isinstance(delta, dict) and delta.get("content"):
                contents.append(delta["content"])

            message = choice.get("message", {})
            if isinstance(message, dict) and message.get("content"):
                contents.append(message["content"])

            if choice.get("text"):
                contents.append(choice["text"])

    return "".join(contents).strip()


def call_llm(
    client: OpenAI,
    prompt: str,
    model: str,
    max_retries: int = 3,
    retry_sleep: float = 5.0,
) -> str:
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You extract faithful codification-ready research method "
                            "specifications from papers. Return valid JSON only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            if hasattr(response, "choices"):
                if not response.choices:
                    raise ValueError("LLM API returned no choices.")
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("LLM API returned message.content=None.")
                return content.strip()

            if isinstance(response, str):
                raw = response.strip()

                if raw.startswith("{") or raw.startswith("["):
                    return raw

                if "data:" in raw:
                    content = extract_sse_content(raw)
                    if content:
                        return content

                raise ValueError(
                    "LLM API returned raw string but no completion content was found.\n"
                    f"Response preview:\n{raw[:1000]}"
                )

            if isinstance(response, dict):
                if response.get("choices"):
                    choice = response["choices"][0]

                    if "message" in choice and "content" in choice["message"]:
                        return choice["message"]["content"].strip()

                    if "text" in choice:
                        return choice["text"].strip()

                if "content" in response:
                    return str(response["content"]).strip()

            raise TypeError(
                f"Unexpected LLM response type: {type(response)}\n"
                f"Response preview:\n{str(response)[:1000]}"
            )

        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(retry_sleep * attempt)

    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_err}")


# ============================================================
# Text processing
# ============================================================

def remove_reference_sections(text: str) -> str:
    lines = text.splitlines()
    output = []
    skip = False

    reference_heading_pattern = re.compile(
        r"^#+\s*(references|bibliography|works cited|acknowledgements|acknowledgments|acknowledgement|acknowledgment)\s*$",
        re.IGNORECASE,
    )

    heading_pattern = re.compile(r"^#+")

    for line in lines:
        stripped = line.strip()

        if reference_heading_pattern.match(stripped):
            skip = True
            continue

        if skip and heading_pattern.match(stripped):
            lower = stripped.lower()
            if not any(
                term in lower
                for term in [
                    "references",
                    "bibliography",
                    "works cited",
                    "acknowledgements",
                    "acknowledgments",
                    "acknowledgement",
                    "acknowledgment",
                ]
            ):
                skip = False

        if not skip:
            output.append(line)

    return "\n".join(output)


def extract_relevant_sections(text: str, max_chars: int = 50000) -> str:
    """
    Extract method / experiment / appendix-related sections.
    References are removed to save tokens.
    Appendix is allowed because it may contain implementation details.

    Version 1:
    - We do NOT split one paper into multiple chunks.
    - We select relevant sections and truncate to max_chars.
    - One paper produces one final gold spec.
    """
    text = remove_reference_sections(text)
    lines = text.splitlines()

    selected = []
    keep = False

    start_patterns = [
        r"^#+\s*(abstract|summary)",
        r"^#+\s*(introduction|background)",
        r"^#+\s*(method|methods|methodology|approach|model|algorithm|architecture)",
        r"^#+\s*(experiment|experiments|experimental setup|implementation details)",
        r"^#+\s*(evaluation|results|training|dataset|datasets|data)",
        r"^#+\s*(appendix|appendices|supplementary|supplemental)",
        r"^#+\s*(hyperparameter|hyperparameters|reproducibility|additional details|training details)",
    ]

    stop_patterns = [
        r"^#+\s*(related work|conclusion|discussion|references|bibliography|works cited|acknowledgements|acknowledgments|acknowledgement|acknowledgment)",
    ]

    for line in lines:
        low = line.strip().lower()

        if any(re.search(p, low) for p in start_patterns):
            keep = True

        if keep:
            selected.append(line)

        if keep and any(re.search(p, low) for p in stop_patterns):
            keep = False

    section_text = "\n".join(selected).strip()

    # Fallback: if section extraction fails, use the full cleaned text.
    if len(section_text) < 2000:
        section_text = text

    section_text = remove_reference_sections(section_text)

    return section_text[:max_chars]


def choose_original_markdown_path(record: Dict[str, Any]) -> str:
    """
    Prefer MinerU original markdown if available.
    Otherwise use fast PyMuPDF original markdown.
    Finally fall back to paper_meta/markdown/<record_id>.md.
    """
    local_paths = record.get("local_paths") or {}
    record_id = record.get("record_id") or ""

    candidates = [
        local_paths.get("original_paper_mineru_markdown_path"),
        local_paths.get("original_paper_fast_text_markdown_path"),
    ]
    if record_id:
        candidates.append(PAPER_META_MARKDOWN_DIR / f"{safe_filename(record_id)}.md")

    for p in candidates:
        if p and Path(p).exists() and Path(p).stat().st_size > 50:
            return str(Path(p).resolve())

    return ""


def is_synthetic_controlled(record: Dict[str, Any]) -> bool:
    route = record.get("llm_route_classification") or {}
    selection = record.get("synthetic_selection") or {}
    return (
        route.get("primary_route") == "synthetic_controlled"
        or selection.get("primary_route") == "synthetic_controlled"
        or selection.get("selection_route") == "synthetic_controlled"
    )


# ============================================================
# Normalization
# ============================================================

def normalize_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        spec = {}

    out: Dict[str, Any] = {}
    for key in SPEC_FIELDS:
        if key in LIST_FIELDS:
            items = as_list(spec.get(key))
            cleaned = []
            for item in items:
                item_text = normalize_string(item)
                if item_text:
                    cleaned.append(item_text)
            out[key] = cleaned
        else:
            out[key] = normalize_string(spec.get(key))

    return out


def normalize_extracted_output(extracted: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expected LLM output:
    {
      "paper_derived_specification": {...},
      "codification_ready_reference": "..."
    }

    Also supports fallback if the model accidentally returns old flat schema.
    """
    if "paper_derived_specification" in extracted:
        paper_spec = extracted.get("paper_derived_specification", {})
        reference = extracted.get("codification_ready_reference", "")
    else:
        paper_spec = extracted
        reference = extracted.get("codification_ready_reference", "")

    return {
        "paper_derived_specification": normalize_spec(paper_spec),
        "codification_ready_reference": normalize_string(reference),
    }


def compute_completeness(spec: Dict[str, Any]) -> Dict[str, bool]:
    return {
        "has_task": bool(spec.get("task")),
        "has_inputs": bool(spec.get("inputs")),
        "has_outputs": bool(spec.get("outputs")),
        "has_core_method": bool(spec.get("core_method")),
        "has_algorithm_steps": bool(spec.get("algorithm_steps")),
        "has_training_or_optimization": bool(spec.get("training_or_optimization")),
        "has_datasets": bool(spec.get("datasets")),
        "has_evaluation_metrics": bool(spec.get("evaluation_metrics")),
        "has_baselines": bool(spec.get("baselines")),
        "has_implementation_details": bool(spec.get("implementation_details")),
    }


def audit_gold_spec(paper_spec: Dict[str, Any], reference: str) -> List[str]:
    warnings: List[str] = []
    ref_low = reference.lower()
    unknown_fields = paper_spec.get("unknown_fields") or []

    if isinstance(unknown_fields, list) and len(unknown_fields) >= 5:
        warnings.append("many_unknown_fields")

    if contains_any_pattern(ref_low, RESULT_CLAIM_PATTERNS):
        warnings.append("reference_contains_result_claim")

    if any(phrase in ref_low for phrase in MECHANISM_CLAIM_PHRASES):
        warnings.append("reference_contains_claimed_method_advantage")

    if contains_any_pattern(ref_low, UNSUPPORTED_DECISION_RULE_PATTERNS):
        warnings.append("reference_may_contain_inferred_decision_rule")

    if contains_any_pattern(ref_low, OPTIMIZER_NAME_PATTERNS):
        warnings.append("reference_contains_optimizer_recipe")

    if contains_any_pattern(ref_low, EXPERIMENT_ONLY_PHASE_PATTERNS):
        warnings.append("reference_contains_experiment_only_data_generation_phase")

    if contains_any_pattern(ref_low, UNCERTAINTY_IN_REFERENCE_PATTERNS):
        warnings.append("reference_contains_uncertainty_text")

    if any(phrase in ref_low for phrase in METADATA_PHRASES_IN_REFERENCE):
        warnings.append("reference_contains_metadata_or_environment_phrase")

    if contains_any_pattern(ref_low, UNSUPPORTED_TOOL_CLAIM_PATTERNS):
        warnings.append("possible_unsupported_tool_or_evaluation_claim")

    tunable_hits = sum(1 for pattern in TUNABLE_RECIPE_PATTERNS if safe_regex_search(pattern, ref_low))
    if tunable_hits >= 3:
        warnings.append("reference_contains_tunable_training_recipe")

    dense_formula_hits = sum(1 for pattern in DENSE_FORMULA_PATTERNS if safe_regex_search(pattern, reference))
    if dense_formula_hits >= 3:
        warnings.append("reference_contains_dense_formula_details")

    equation_like_hits = sum(1 for pattern in EQUATION_LIKE_REFERENCE_PATTERNS if safe_regex_search(pattern, reference))
    if equation_like_hits >= 2:
        warnings.append("reference_contains_equation_like_expressions")

    numeric_config_hits = sum(1 for pattern in NUMERIC_CONFIG_PATTERNS if safe_regex_search(pattern, ref_low))
    if numeric_config_hits >= 3:
        warnings.append("reference_contains_many_numeric_implementation_constants")

    if contains_any_pattern(reference, LOW_LEVEL_SYMBOLIC_PATTERNS):
        warnings.append("reference_contains_low_level_symbolic_parameterization")

    if contains_any_pattern(ref_low, ROUTINE_LOSS_PATTERNS):
        warnings.append("reference_contains_routine_training_loss")

    if "unknown_fields" in ref_low or "not specified" in ref_low or "not fully" in ref_low:
        warnings.append("reference_contains_limitation_or_unknown_field_text")

    field_label_hits = 0
    for pat in [
        r"\btitle:\b",
        r"\bresearch overview:\b",
        r"\bmethod:\b",
        r"\bevaluation plan:\b",
        r"\bimplementation notes:\b",
        r"\binputs:\b",
        r"\boutputs:\b",
        r"\balgorithm steps:\b",
        r"\bdatasets:\b",
        r"\bevaluation metrics:\b",
        r"\bbaselines:\b",
    ]:
        if safe_regex_search(pat, reference):
            field_label_hits += 1
    if field_label_hits >= 3:
        warnings.append("reference_field_list_style")

    return sorted(set(warnings))


def assess_gold_quality(paper_spec: Dict[str, Any], reference: str) -> Tuple[str, List[str], Dict[str, bool]]:
    completeness = compute_completeness(paper_spec)
    warnings: List[str] = []
    ref_words = word_count(reference)

    if not reference:
        warnings.append("empty_codification_ready_reference")
    elif ref_words < 100:
        warnings.append("codification_ready_reference_too_short")
    elif ref_words > 430:
        warnings.append("codification_ready_reference_too_long")

    for key, ok in completeness.items():
        if not ok:
            warnings.append(f"missing_{key.replace('has_', '')}")

    warnings.extend(audit_gold_spec(paper_spec, reference))

    required_ok = (
        completeness["has_task"]
        and completeness["has_core_method"]
        and (completeness["has_algorithm_steps"] or completeness["has_implementation_details"])
        and bool(reference)
        and ref_words >= 80
    )

    high_ok = (
        required_ok
        and completeness["has_inputs"]
        and completeness["has_outputs"]
        and completeness["has_training_or_optimization"]
        and completeness["has_datasets"]
        and completeness["has_evaluation_metrics"]
    )

    if high_ok:
        source_quality = "high"
    elif required_ok:
        source_quality = "medium"
    else:
        source_quality = "low"

    return source_quality, sorted(set(warnings)), completeness


def build_source_metadata(
    record: Dict[str, Any],
    original_text_path: str,
) -> Dict[str, Any]:
    route = record.get("llm_route_classification") or {}
    selection = record.get("synthetic_selection") or {}
    source = record.get("source") or "OpenReview_Reproducibility"

    return {
        "source_type": f"{source}_synthetic_controlled",
        "record_id": record.get("record_id"),
        "source": record.get("source"),
        "year": record.get("year"),
        "venue_or_track": record.get("venue_or_track"),
        "report_title": record.get("report_title"),
        "report_url": record.get("report_url"),
        "report_pdf_url": record.get("report_pdf_url"),
        "original_paper_title": record.get("original_paper_title"),
        "original_paper_url": record.get("original_paper_url"),
        "original_paper_pdf_url": record.get("original_paper_pdf_url"),
        "original_text_path": original_text_path,
        "route": route.get("primary_route"),
        "routing_reason": route.get("routing_reason", ""),
        "route_confidence": route.get("confidence"),
        "synthetic_selection": selection,
    }


def wrap_output_schema(
    extracted: Dict[str, Any],
    record: Dict[str, Any],
    original_text_path: str,
    selected_text_path: str,
    model: str,
    max_chars: int,
) -> Dict[str, Any]:
    normalized = normalize_extracted_output(extracted)

    reference = normalized["codification_ready_reference"]
    paper_spec = normalized["paper_derived_specification"]
    record_id = record.get("record_id") or "unknown_record"

    source_quality, quality_warnings, completeness = assess_gold_quality(paper_spec, reference)
    quality_label = "usable" if source_quality in {"high", "medium"} else "unusable"

    limitations = []
    for field in paper_spec.get("unknown_fields", []):
        if field:
            limitations.append(str(field))
    for warning in quality_warnings:
        if warning.startswith("missing_"):
            limitations.append(warning)

    return {
        "project_id": record_id,
        "record_id": record_id,
        "split_group_id": record_id,
        "split_group_type": "paper",
        "source": build_source_metadata(record, original_text_path=original_text_path),
        "paper_derived_specification": paper_spec,
        "codification_ready_reference": reference,
        "gold_extraction_metadata": {
            "record_id": record_id,
            "split_group_id": record_id,
            "split_group_type": "paper",
            "extraction_method": "llm_from_original_paper_mineru_markdown",
            "model": model,
            "source_quality": source_quality,
            "has_original_paper_evidence": bool(original_text_path),
            "has_reproducibility_report_evidence": bool(record.get("report_url") or record.get("report_pdf_url")),
            "gold_spec_completeness": completeness,
            "limitations": sorted(set(limitations)),
            "source_paths": {
                "original_text_path": original_text_path,
                "selected_text_path": selected_text_path,
                "original_paper_pdf_path": (record.get("local_paths") or {}).get("original_paper_pdf_path", ""),
                "original_paper_mineru_markdown_path": (record.get("local_paths") or {}).get("original_paper_mineru_markdown_path", ""),
            },
            "route": (record.get("llm_route_classification") or {}).get("primary_route"),
            "routing_reason": (record.get("llm_route_classification") or {}).get("routing_reason", ""),
        },
        "quality_label": quality_label,
        "quality_warnings": quality_warnings,
        "extraction_metadata": {
            "extraction_method": "llm_from_original_paper_mineru_markdown",
            "model": model,
            "max_chars": max_chars,
            "selected_text_path": selected_text_path,
        },
    }


# ============================================================
# Prompt
# ============================================================

def build_prompt(record: Dict[str, Any], relevant_text: str) -> str:
    metadata = {
        "record_id": record.get("record_id"),
        "report_title": record.get("report_title"),
        "original_paper_title": record.get("original_paper_title"),
        "original_paper_url": record.get("original_paper_url"),
        "original_paper_pdf_url": record.get("original_paper_pdf_url"),
        "year": record.get("year"),
        "route_classification": record.get("llm_route_classification"),
    }

    return f"""
You are given markdown text from an original machine learning paper.

This paper has been selected for the synthetic-controlled route of an idea/specification ambiguity benchmark:
- The associated reproducibility report indicates successful reproduction and did NOT identify a resolved method-core specification gap.
- The original paper is the primary evidence source from which we derive a codification-ready gold specification.
- The reproducibility report justifies source selection; it is not the source of a defect in this step.
- Later steps will inject exactly one synthetic defect into this gold specification.
- Your task here is ONLY to extract the clean gold specification.
- Do NOT create any defects in this step.

Benchmark context:
An idea specification is a research idea or method description that expresses a scientific direction and provides partial information about how the idea may be implemented.

{GOLD_CODIFICATION_READY_REFERENCE_DEFINITION}

If a detail is not supported by the paper, do NOT invent it. Put it in unknown_fields instead.

Rules for the structured paper-derived specification:
- Extract only details supported by the paper text.
- Do NOT invent details.
- If information is missing, use "unknown" or record it in "unknown_fields".
- Keep fields concise.
- Use arrays for list-like fields.
- Include implementation-relevant details useful for later controlled perturbation.
- Prefer concrete method details over background or related work.
- Put exact equations, derivations, normalizers, architecture counts, optimizer settings, random seeds, training schedules, and experiment-only data generation details in the structured fields when supported, not in codification_ready_reference.

Rules for codification_ready_reference:
- Write one natural-language paragraph.
- Length should be 120-260 words.
- It should sound like a compact method handoff specification, not a paper summary, not a JSON field list, and not a reproduction recipe.
- Prefer a phase/step structure when possible: "Given <inputs>, the method proceeds in <N> phases: (1) ..., (2) ..., ...".
- Use at most four phases unless the method truly requires more. Each phase should name an implementation action, not an experimental observation.
- It should be clean, complete, and implementation-oriented at the level of method logic.
- Include the task, inputs, outputs, core method, training or estimation logic, and evaluation protocol.
- Include necessary architectural or algorithmic details only when they define the method interface, module roles, or stage behavior.
- Prefer concrete method details that a coder needs over broad motivation or result claims.
- Include architectural or algorithmic constants only when changing them would change the method identity, interface, or core mechanism.
- State what the method does, not why it is better. Do not include claimed advantages such as faster convergence, avoiding collapse, high sample density, intra-class compactness, better separation, or improved robustness unless the phrase is rewritten as an operational mechanism required for implementation.
- Do not guess the inference or prediction decision rule. If the paper does not explicitly specify whether prediction uses softmax logits, nearest center, distance scoring, thresholds, or decoding, use a conservative phrase such as "the paper-specified decision rule" and record the uncertainty in unknown_fields.
- Do not add example decision rules using "e.g." when the paper does not explicitly specify them.
- Exclude ordinary tunable training recipe values from codification_ready_reference, including learning rate, batch size, epoch count, momentum, weight decay, dropout rate, optimizer schedule, random seed, and hardware, unless the value is itself a non-standard method mechanism.
- Put ordinary tunable training recipe values in reproducibility_relevant_details instead of codification_ready_reference.
- Do not include exact optimizer names such as Adam or SGD in codification_ready_reference unless the optimizer is a scientific contribution of the method.
- Do not include exact symbolic formulas in codification_ready_reference. Prefer concise operational descriptions of equations, losses, priors, likelihoods, attention scores, Bellman targets, and constraints. Keep exact equations in paper_derived_specification fields instead.
- Do not include long formulas, derivations, normalizing constants, dense mathematical expansions, or equation-like expressions with "=" in codification_ready_reference.
- Do not include low-level symbolic parameterizations with subscripts or superscripts, such as relative-position vectors, per-head vectors, density parameters, Jacobian terms, or full target-value formulas. Replace them with a named operational mechanism.
- Do not include many numeric architecture counts or implementation constants unless they are central to the method identity. Put secondary details in implementation_details or reproducibility_relevant_details.
- Do not include example numeric architecture settings such as "e.g., six layers", "9 heads", "hidden dimension 400", or "10 flow steps"; put them in implementation_details.
- Do not make experiment-only data generation a method phase unless the proposed method is itself a data generator. Put benchmark data generation in datasets or evaluation details.
- Do not include routine training losses such as cross-entropy when they are only used to train a downstream classifier and are not the proposed method contribution.
- Do not say a detail is "not explicitly specified" inside codification_ready_reference. Put that uncertainty in unknown_fields.
- Do not combine a theoretical construction and an experimental architecture into one dense reference unless both are necessary for the same implementation target. Prefer the implementation-facing method used for the benchmark instance.
- Do NOT include unsupported details.
- Do NOT include unknown_fields in this paragraph.
- Do NOT mention that this is synthetic, a benchmark, a gold reference, or a route classification.
- Do NOT mention the reproducibility report.
- Do NOT mention future defect injection.
- Do NOT include open or unresolved details in this paragraph.
- Do NOT include comparative result claims such as "outperforms", "achieves better", or "results show".
- Do NOT include claimed benefits such as "shows faster convergence", "avoids degenerate collapse", "improves robustness", "induces high sample density", or "maintains compactness" unless they are restated as an implementation operation.
- Do NOT include random seeds, exact compute resources, or incidental implementation details unless they are essential to the core method.
- Preserve the paper's research idea faithfully.

Return valid JSON only with this exact schema:

{{
  "paper_derived_specification": {{
    "paper_title": "",
    "research_goal": "",
    "task": "",
    "inputs": "",
    "outputs": "",
    "core_method": "",
    "algorithm_steps": [],
    "training_or_optimization": "",
    "datasets": [],
    "evaluation_metrics": [],
    "baselines": [],
    "implementation_details": [],
    "reproducibility_relevant_details": [],
    "unknown_fields": []
  }},
  "codification_ready_reference": ""
}}

Source metadata:
{json.dumps(metadata, indent=2, ensure_ascii=False)}

Original paper markdown:
\"\"\"
{relevant_text}
\"\"\"
"""


# ============================================================
# Main extraction
# ============================================================

def process_one_record(
    record: Dict[str, Any],
    out_dir: Path,
    client: OpenAI,
    model: str,
    max_chars: int,
    save_selected_sections: bool,
    save_prompt: bool,
    resume: bool,
) -> Dict[str, Any]:
    record_id = record.get("record_id") or "unknown_record"
    record_dir = out_dir / safe_filename(record_id)
    record_dir.mkdir(parents=True, exist_ok=True)

    output_path = record_dir / "gold_spec.json"
    raw_response_path = record_dir / "raw_response.txt"
    error_path = record_dir / "error.json"
    selected_text_path = record_dir / "selected_sections.md"
    prompt_path = record_dir / "prompt.txt"

    if resume and output_path.exists() and output_path.stat().st_size > 50:
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            return {
                "record_id": record_id,
                "status": "skipped_exists",
                "gold_spec_path": str(output_path),
                "quality_label": existing.get("quality_label", "unknown"),
                "quality_warnings": existing.get("quality_warnings", []),
                "original_text_path": existing.get("source", {}).get("original_text_path", ""),
            }
        except Exception:
            pass

    original_text_path = choose_original_markdown_path(record)

    if not original_text_path:
        err = {
            "record_id": record_id,
            "error": "missing_original_paper_markdown",
        }
        save_json(err, error_path)

        return {
            "record_id": record_id,
            "status": "failed",
            "error": "missing_original_paper_markdown",
            "error_path": str(error_path),
            "gold_spec_path": "",
            "quality_label": "unusable",
            "quality_warnings": ["missing_original_paper_markdown"],
            "original_text_path": "",
        }

    paper_text = Path(original_text_path).read_text(encoding="utf-8", errors="ignore")

    relevant_text = extract_relevant_sections(
        paper_text,
        max_chars=max_chars,
    )

    if save_selected_sections:
        selected_text_path.write_text(relevant_text, encoding="utf-8")

    prompt = build_prompt(record, relevant_text)

    if save_prompt:
        prompt_path.write_text(prompt, encoding="utf-8")

    try:
        raw = call_llm(
            client=client,
            prompt=prompt,
            model=model,
        )

        raw_response_path.write_text(raw, encoding="utf-8")

        extracted = safe_json_loads(raw)

        output_obj = wrap_output_schema(
            extracted=extracted,
            record=record,
            original_text_path=original_text_path,
            selected_text_path=str(selected_text_path) if save_selected_sections else "",
            model=model,
            max_chars=max_chars,
        )

        save_json(output_obj, output_path)

        return {
            "record_id": record_id,
            "status": "saved",
            "gold_spec_path": str(output_path),
            "quality_label": output_obj.get("quality_label"),
            "quality_warnings": output_obj.get("quality_warnings", []),
            "original_text_path": original_text_path,
            "paper_title": output_obj.get("paper_derived_specification", {}).get("paper_title", ""),
            "reference_word_count": len(output_obj.get("codification_ready_reference", "").split()),
        }

    except Exception as e:
        err = {
            "record_id": record_id,
            "error": repr(e),
            "original_text_path": original_text_path,
        }
        save_json(err, error_path)

        return {
            "record_id": record_id,
            "status": "failed",
            "error": repr(e),
            "error_path": str(error_path),
            "gold_spec_path": "",
            "quality_label": "unusable",
            "quality_warnings": ["llm_extraction_failed"],
            "original_text_path": original_text_path,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--model",
        default=os.getenv("OPENROUTER_MODEL") or os.getenv("OPENAI_MODEL") or "deepseek/deepseek-v4-pro",
    )
    parser.add_argument("--max_chars", type=int, default=50000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save_selected_sections", action="store_true")
    parser.add_argument("--save_prompt", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_path)

    # Safety: only process synthetic_controlled records.
    rows = [r for r in rows if is_synthetic_controlled(r)]

    if args.limit and args.limit > 0:
        rows = rows[:args.limit]

    client = build_client()
    index_rows = []
    accepted_gold_specs = []
    rejected_gold_specs = []

    print("\n===== Syn Step 1: Gold Spec Extraction =====")
    print(f"Input path: {input_path}")
    print(f"Output dir: {out_dir}")
    print(f"Records to process: {len(rows)}")
    print(f"Model: {args.model}")
    print(f"Max chars: {args.max_chars}")
    print("===========================================\n")

    for record in tqdm(rows, desc="Extract synthetic gold specs"):
        result = process_one_record(
            record=record,
            out_dir=out_dir,
            client=client,
            model=args.model,
            max_chars=args.max_chars,
            save_selected_sections=args.save_selected_sections,
            save_prompt=args.save_prompt,
            resume=args.resume,
        )

        index_rows.append(result)
        save_jsonl(index_rows, out_dir / "index.jsonl")

        if result.get("gold_spec_path"):
            try:
                gold_obj = json.loads(Path(result["gold_spec_path"]).read_text(encoding="utf-8"))
                if result.get("quality_label") == "usable":
                    accepted_gold_specs.append(gold_obj)
                else:
                    rejected_gold_specs.append(
                        {
                            "record_id": result.get("record_id"),
                            "reason": "unusable_gold_spec",
                            "quality_label": result.get("quality_label"),
                            "quality_warnings": result.get("quality_warnings", []),
                            "gold_spec_path": result.get("gold_spec_path"),
                            "gold": gold_obj,
                        }
                    )
            except Exception as exc:
                rejected_gold_specs.append(
                    {
                        "record_id": result.get("record_id"),
                        "reason": "gold_spec_read_failed",
                        "error": repr(exc),
                        "gold_spec_path": result.get("gold_spec_path", ""),
                    }
                )
        elif result.get("status") == "failed":
            rejected_gold_specs.append(
                {
                    "record_id": result.get("record_id"),
                    "reason": result.get("error", "failed"),
                    "quality_label": result.get("quality_label"),
                    "quality_warnings": result.get("quality_warnings", []),
                    "error_path": result.get("error_path", ""),
                }
            )

        save_jsonl(accepted_gold_specs, out_dir / "gold_specs.jsonl")
        save_jsonl(rejected_gold_specs, out_dir / "rejected_gold_specs.jsonl")

        if result.get("status") == "saved":
            print(
                f"[saved] {result.get('record_id')} | "
                f"quality={result.get('quality_label')} | "
                f"words={result.get('reference_word_count')} | "
                f"warnings={result.get('quality_warnings')}"
            )
        elif result.get("status") == "skipped_exists":
            print(
                f"[skip] {result.get('record_id')} | "
                f"quality={result.get('quality_label')} | "
                f"warnings={result.get('quality_warnings')}"
            )
        else:
            print(
                f"[failed] {result.get('record_id')} | "
                f"error={result.get('error')}"
            )

    save_jsonl(index_rows, out_dir / "index.jsonl")

    saved = sum(r.get("status") == "saved" for r in index_rows)
    skipped = sum(r.get("status") == "skipped_exists" for r in index_rows)
    failed = sum(r.get("status") == "failed" for r in index_rows)
    usable = sum(r.get("quality_label") == "usable" for r in index_rows)
    source_quality_counts = Counter()
    warning_counts = Counter()
    completeness_counts = Counter()
    for gold in accepted_gold_specs:
        meta = gold.get("gold_extraction_metadata") or {}
        source_quality_counts[meta.get("source_quality", "unknown")] += 1
        for warning in gold.get("quality_warnings", []):
            warning_counts[warning] += 1
        completeness = meta.get("gold_spec_completeness") or {}
        for key, value in completeness.items():
            if value:
                completeness_counts[key] += 1

    summary = {
        "input_path": str(input_path),
        "out_dir": str(out_dir),
        "records_processed": len(index_rows),
        "saved": saved,
        "skipped_exists": skipped,
        "failed": failed,
        "usable": usable,
        "accepted_gold_specs": len(accepted_gold_specs),
        "rejected_gold_specs": len(rejected_gold_specs),
        "source_quality_distribution": dict(source_quality_counts),
        "quality_warning_distribution": dict(warning_counts),
        "completeness_true_counts": dict(completeness_counts),
        "model": args.model,
        "max_chars": args.max_chars,
        "gold_specs_path": str(out_dir / "gold_specs.jsonl"),
        "rejected_gold_specs_path": str(out_dir / "rejected_gold_specs.jsonl"),
    }
    save_json(summary, out_dir / "summary.json")

    print("\n===== Syn Step 1 Summary =====")
    print(f"Processed: {len(index_rows)}")
    print(f"Saved: {saved}")
    print(f"Skipped existing: {skipped}")
    print(f"Failed: {failed}")
    print(f"Usable: {usable}")
    print(f"Accepted gold specs: {len(accepted_gold_specs)}")
    print(f"Rejected gold specs: {len(rejected_gold_specs)}")
    print(f"Index: {out_dir / 'index.jsonl'}")
    print(f"Gold specs: {out_dir / 'gold_specs.jsonl'}")
    print(f"Rejected gold specs: {out_dir / 'rejected_gold_specs.jsonl'}")
    print(f"Summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
