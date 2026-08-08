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
# Fixed labels
# ============================================================

LEVEL2_TO_LEVEL1 = {
    "Ambiguous Definition": "Ambiguity",
    "Ambiguous Procedure": "Ambiguity",
    "Missing Algorithmic Procedure": "Incompleteness",
    "Missing Configuration Protocol": "Incompleteness",
    "Missing Model Specification": "Incompleteness",
    "Missing Evaluation Specification": "Incompleteness",
    "Missing Data Specification": "Incompleteness",
    "Conflicting Objective": "Inconsistency",
    "Conflicting Model Design": "Inconsistency",
    "Conflicting Formal Definition": "Inconsistency",
}

LEVEL1_LABELS = {
    "Ambiguity",
    "Incompleteness",
    "Inconsistency",
}

LEVEL2_LABELS = set(LEVEL2_TO_LEVEL1.keys())

GRANULARITY_LABELS = {
    "coarse",
    "medium",
    "fine",
}

RESOLUTION_ROLE_LABELS = {
    "implementation_blocker",
    "open_design_choice",
    "reproducibility_detail",
    "inconsistency_to_resolve",
}

CODIFICATION_SLOT_LABELS = {
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

ACTION_TYPE_LABELS = {
    "clarification_question",
    "evidence_seeking",
    "experiment_selection",
}


LEVEL2_TO_CODIFICATION_SLOT = {
    "Ambiguous Definition": "CORE_ALGORITHM",
    "Ambiguous Procedure": "CORE_ALGORITHM",
    "Missing Algorithmic Procedure": "CORE_ALGORITHM",
    "Missing Model Specification": "MODEL_ARCHITECTURE",
    "Missing Data Specification": "DATA_AND_PREPROCESSING",
    "Missing Configuration Protocol": "TRAINING_PROCEDURE",
    "Missing Evaluation Specification": "EVALUATION_PROTOCOL",
    "Conflicting Objective": "INTERNAL_CONSISTENCY",
    "Conflicting Model Design": "INTERNAL_CONSISTENCY",
    "Conflicting Formal Definition": "INTERNAL_CONSISTENCY",
}


# ============================================================
# Universal source helpers
# ============================================================

VALID_SOURCE_FILTERS = {
    "all",
    "OpenReview_MLRC",
    "OpenReview_TMLR",
}

DATASET_ROOTS = {
    "mlrc": Path("Real_bench/repro_paper_candidates"),
    "tmlr": Path("Real_bench/TMLR"),
    "mixed": Path("Real_bench/shared"),
}


def dataset_root(dataset_tag: str) -> Path:
    return DATASET_ROOTS.get(dataset_tag, Path(f"Real_bench/{dataset_tag}"))


def infer_record_source(record: Dict[str, Any]) -> str:
    explicit = str(record.get("source") or "").strip()
    if explicit:
        return explicit

    record_id = str(record.get("record_id") or "")
    realgap_id = str(record.get("realgap_id") or "")

    if record_id.startswith("openreview_mlrc_") or realgap_id.startswith("openreview_mlrc_"):
        return "OpenReview_MLRC"
    if record_id.startswith("tmlr_") or realgap_id.startswith("tmlr_"):
        return "OpenReview_TMLR"

    return ""


def ensure_record_source(record: Dict[str, Any]) -> Dict[str, Any]:
    if not str(record.get("source") or "").strip():
        inferred = infer_record_source(record)
        if inferred:
            record = dict(record)
            record["source"] = inferred
    return record


def source_matches(record: Dict[str, Any], source_filter: str) -> bool:
    if source_filter == "all":
        return True
    return infer_record_source(record) == source_filter


def infer_dataset_tag_from_source(source_filter: str) -> str:
    if source_filter == "OpenReview_MLRC":
        return "mlrc"
    if source_filter == "OpenReview_TMLR":
        return "tmlr"
    return "mixed"


def infer_source_type(source: str) -> str:
    if source == "OpenReview_MLRC":
        return "OpenReview_MLRC_real_gap"
    if source == "OpenReview_TMLR":
        return "OpenReview_TMLR_real_gap"
    if source:
        return f"{source}_real_gap"
    return "unknown_real_gap"


def default_input_path(dataset_tag: str) -> Path:
    root = dataset_root(dataset_tag)
    if dataset_tag == "mlrc":
        return root / "postprocessed" / "main_resolved.jsonl"
    return root / "postprocessed_real_gap" / "main_resolved.jsonl"


def default_out_dir(dataset_tag: str) -> Path:
    return dataset_root(dataset_tag) / "realgap_benchmark_instances"


# ============================================================
# IO
# ============================================================

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows

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


def read_text(path: str, max_chars: int) -> str:
    if not path:
        return ""

    p = Path(path)
    if not p.exists():
        return ""

    return p.read_text(encoding="utf-8", errors="ignore")[:max_chars]


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
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            text = m.group(0)

    return json.loads(text)


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


def call_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    max_retries: int = 3,
    retry_sleep: float = 5.0,
) -> str:
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You construct benchmark instances for diagnosing underspecified "
                            "scientific method specifications. Return valid JSON only."
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

            return content.strip()

        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(retry_sleep * attempt)

    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_err}")


# ============================================================
# Normalization helpers
# ============================================================

def normalize_level2_label(raw: Any) -> str:
    text = str(raw or "").strip()
    for label in LEVEL2_LABELS:
        if text.lower() == label.lower():
            return label
    return text


def normalize_codification_slot(raw: str) -> str:
    text = str(raw or "").strip().upper().replace(" ", "_").replace("-", "_")
    text = re.sub(r"_+", "_", text)
    if text in CODIFICATION_SLOT_LABELS:
        return text
    return "NONE"


def infer_codification_slot(gap: Dict[str, Any]) -> str:
    level2 = normalize_level2_label(gap.get("level2", ""))
    if level2 in LEVEL2_TO_CODIFICATION_SLOT:
        return LEVEL2_TO_CODIFICATION_SLOT[level2]

    affected = normalize_codification_slot(str(gap.get("affected_component", "")).strip())
    if affected in CODIFICATION_SLOT_LABELS:
        return affected
    return "NONE"


def infer_granularity(gap: Dict[str, Any]) -> str:
    level2 = normalize_level2_label(gap.get("level2", ""))
    affected = normalize_codification_slot(str(gap.get("affected_component", "")))
    text = " ".join([
        str(gap.get("gap_summary", "")),
        str(gap.get("gold_clarified_detail", "")),
    ]).lower()

    if level2 == "Missing Configuration Protocol":
        return "fine"

    if affected in {"TRAINING_PROCEDURE", "NONE"}:
        return "fine"

    if any(k in text for k in ["learning rate", "batch size", "epoch", "threshold", "k=", "k =", "lambda"]):
        return "fine"

    if affected in {
        "TASK_AND_IO",
        "CORE_ALGORITHM",
        "DATA_AND_PREPROCESSING",
        "INFERENCE_AND_DECISION",
        "MODEL_ARCHITECTURE",
    }:
        return "medium"

    return "medium"


def infer_resolution_role(gap: Dict[str, Any], granularity: str) -> str:
    level2 = normalize_level2_label(gap.get("level2", ""))

    if level2.startswith("Conflicting"):
        return "inconsistency_to_resolve"

    if level2 in {"Missing Evaluation Specification", "Missing Data Specification"}:
        return "reproducibility_detail"

    if granularity == "fine" or level2 == "Missing Configuration Protocol":
        return "reproducibility_detail"

    return "implementation_blocker"


def normalize_action_type(raw: str) -> str:
    raw = str(raw or "").strip().lower()

    if raw in ACTION_TYPE_LABELS:
        return raw

    if any(k in raw for k in ["evidence", "code", "paper", "report", "inspect", "look up", "seek"]):
        return "evidence_seeking"

    if any(k in raw for k in ["experiment", "select", "choose", "compare", "ablation"]):
        return "experiment_selection"

    return "clarification_question"


def normalize_paper_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    schema = {
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
        "unknown_fields": [],
    }

    if not isinstance(spec, dict):
        return schema

    for k in schema:
        if k in spec:
            schema[k] = spec[k]

    list_fields = [
        "algorithm_steps",
        "datasets",
        "evaluation_metrics",
        "baselines",
        "implementation_details",
        "reproducibility_relevant_details",
        "unknown_fields",
    ]

    for k in list_fields:
        if not isinstance(schema[k], list):
            schema[k] = [str(schema[k])] if schema[k] else []

    return schema


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", str(text or "")))


def contains_gold_detail(text: str, gold_detail: str) -> bool:
    """
    Conservative leakage check.

    We only flag likely leakage when:
    1. highly distinctive numeric/architecture tokens from the gold detail appear in the input, or
    2. a long contiguous distinctive phrase from the gold detail appears in the input.
    """
    text_low = str(text or "").lower()
    gold_low = str(gold_detail or "").lower()

    if not text_low or not gold_low:
        return False

    numeric_patterns = re.findall(
        r"(?:\d+\s*,\s*){2,}\d+|conv_[a-z]\s*[-+]\s*conv_[a-z]|real output|imaginary output",
        gold_low,
    )

    for p in numeric_patterns:
        if p and p in text_low:
            return True

    gold_tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{3,}\b", gold_low)

    stop = {
        "reproduction", "implement", "implementation", "detail", "complex",
        "model", "method", "input", "output", "layer", "layers", "feature",
        "features", "network", "classification", "training", "dataset",
        "paper", "research", "convolution", "convolutional", "respectively",
        "using", "with", "from", "into", "part", "parts", "given",
        "separate", "apply", "combine", "form", "respect", "operation",
    }

    distinctive = [w for w in gold_tokens if w not in stop]

    if len(distinctive) < 5:
        return False

    for i in range(0, len(distinctive) - 4):
        phrase = " ".join(distinctive[i:i + 5])
        if phrase in text_low:
            return True

    return False


# Important: do NOT include bare "gap" here.
# It creates false positives for Conv-4 GAP = global average pooling.
DIAGNOSTIC_LEAKAGE_TERMS = {
    "ambiguous",
    "ambiguity",
    "underspecified",
    "under-specified",
    "not specified",
    "not fully specified",
    "does not specify",
    "doesn't specify",
    "not clearly specified",
    "missing",
    "missing detail",
    "undefined",
    "unclear",
    "inconsistent",
    "inconsistency",
    "contradiction",
    "contradicts",
    "conflict",
    "conflicting",
    "actual code",
    "code implementation",
    "reported in the code",
    "authors' code",
    "author's code",
    "clarified by",
    "as clarified",
    "reproducibility report",
    "specification gap",
    "method gap",
    "implementation gap",
    "defect",
    "codification impossible",
    "cannot be codified",
    "blocks codification",
    "implementation blocker",
}


def detect_diagnostic_leakage(text: str) -> List[str]:
    raw = str(text or "")
    low = raw.lower()
    hits = []

    for term in sorted(DIAGNOSTIC_LEAKAGE_TERMS):
        if " " in term:
            if term in low:
                hits.append(term)
        else:
            if re.search(rf"\b{re.escape(term)}\b", low):
                hits.append(term)

    return hits


def starts_like_local_fix(text: str) -> bool:
    return bool(re.match(
        r"^\s*(to implement|use|set|apply|compute|inspect|choose|replace|fix)\b",
        str(text or "").lower(),
    ))


# ============================================================
# Source object helper
# ============================================================

def build_source_object(item: Dict[str, Any]) -> Dict[str, Any]:
    source_name = infer_record_source(item)
    return {
        "source_type": infer_source_type(source_name),
        "record_id": item.get("record_id"),
        "realgap_id": item.get("realgap_id"),
        "report_title": item.get("report_title"),
        "report_url": item.get("report_url"),
        "report_pdf_url": item.get("report_pdf_url"),
        "original_paper_title": item.get("original_paper_title"),
        "original_paper_url": item.get("original_paper_url"),
        "original_paper_pdf_url": item.get("original_paper_pdf_url"),
        "year": item.get("year"),
        "report_text_path": item.get("report_text_path"),
        "original_text_path": item.get("original_text_path"),
    }


# ============================================================
# Prompt
# ============================================================

def build_prompt(
    item: Dict[str, Any],
    original_text: str,
    report_text: str,
    instance_id: str,
) -> str:
    gap = item.get("gap") or {}

    codification_slot = infer_codification_slot(gap)
    granularity = infer_granularity(gap)
    resolution_role = infer_resolution_role(gap, granularity)

    level2 = normalize_level2_label(gap.get("level2", ""))
    level1 = LEVEL2_TO_LEVEL1.get(level2, str(gap.get("level1", "")).strip())
    gold_detail = str(gap.get("gold_clarified_detail", "")).strip()
    solution_source_type = str(gap.get("solution_source_type", "")).strip()

    source = build_source_object(item)

    defect_seed = {
        "slot": codification_slot,
        "level1": level1,
        "level2": level2,
        "granularity": granularity,
        "resolution_role": resolution_role,
        "codification_slot": codification_slot,
        "gold_detail_removed_or_corrupted": gold_detail,
        "why_this_blocks_or_affects_codification": gap.get("why_this_blocks_or_affects_codification", ""),
    }

    evidence = {
        "gap_summary": gap.get("gap_summary", ""),
        "gap_quote": gap.get("gap_quote", ""),
        "solution_summary": gap.get("solution_summary", ""),
        "solution_quote": gap.get("solution_quote", ""),
        "solution_source_type": solution_source_type,
        "source_specification_quote_from_original_paper": gap.get("source_specification_quote_from_original_paper", ""),
        "affected_component": gap.get("affected_component", ""),
        "taxonomy_rationale": gap.get("taxonomy_rationale", ""),
        "gold_clarified_detail": gold_detail,
    }

    source_json = json.dumps(source, indent=2, ensure_ascii=False)
    defect_seed_json = json.dumps(defect_seed, indent=2, ensure_ascii=False)
    evidence_json = json.dumps(evidence, indent=2, ensure_ascii=False)

    return f"""
Construct one benchmark instance from a real reproducibility gap.

Context:
We are building a benchmark for idea/specification ambiguity resolution. The benchmark input is an underspecified research-method specification. A model should diagnose what is missing, ambiguous, or inconsistent before codification.

This instance comes from a real ML reproducibility report. The original paper had a method-core specification gap, and the report provides a concrete clarified implementation detail.

Your task:
Create a benchmark instance with the SAME schema as the synthetic perturbation route, but using the real gap evidence instead of artificial corruption.

CRITICAL NO-LEAKAGE REQUIREMENT:
The input.underspecified_spec is what will be shown to evaluated models. It must NOT reveal that there is a gap, defect, ambiguity, inconsistency, or codification problem.

Do NOT use any of the following diagnostic/meta-evaluation language in input.underspecified_spec:
- ambiguous, ambiguity
- underspecified, missing, incomplete
- not specified, does not specify, unclear, undefined
- inconsistent, inconsistency, contradiction, conflict
- code says, actual code, authors' code, code implementation
- specification gap, method gap, implementation gap, defect, blocker, codification impossible
- reproduction report, clarified by, as clarified

Important special case:
- The acronym "GAP" may mean global average pooling. Do NOT treat it as the word "gap".
- You may use "GAP" only when it refers to global average pooling.

Instead:
- Write input.underspecified_spec as a natural paper-style method description.
- It should sound like a normal method paragraph from the original paper.
- It should include the problematic surface form, but without explicitly saying it is problematic.
- For an inconsistency case, include only the paper-side statement, not the code-derived correction and not that it conflicts with code.
- For an incompleteness case, include only the high-level paper-side operation, not "the paper does not specify...".
- For an ambiguity case, describe the method at the same level as the paper, without saying it is ambiguous.

Examples:
Bad input:
"The paper does not specify how to implement complex convolution, making codification impossible."

Good input:
"The processing module uses a complex convolutional layer with the bias term removed so that convolution preserves the rotation structure of the complex feature representation."

Bad input:
"The paper says all layers use 64 channels, but this is inconsistent with the code."

Good input:
"The Conv-4 model is described as a four-layer convolutional network in which each convolutional layer uses 3x3 kernels and 64 output channels."

CRITICAL GOLD REFERENCE REQUIREMENT:
gold.codification_ready_reference should be a clean, standalone, codification-ready research idea specification.
It should include the clarified detail, but should NOT be written as a local fix only.
It should avoid meta-language such as "this resolves the inconsistency", "as clarified by the report", or "the paper had a gap".
If the detail is code-derived or reproduction-specific, you may write "For this reproduction..." only when necessary, but prefer a clean implementation specification.

Both input.underspecified_spec and gold.codification_ready_reference must describe the full research idea:
1. research goal or motivation,
2. task being solved,
3. inputs and outputs,
4. core method or model structure,
5. the relevant method component containing the hidden issue.

The difference:
- underspecified_spec contains the original paper-style surface form and does NOT reveal the gold detail.
- codification_ready_reference contains the same research idea context PLUS the gold_clarified_detail.

Important constraints:
- Use exactly one defect corresponding to the provided real gap.
- Do not invent unsupported datasets, baselines, methods, architectures, or hyperparameters.
- Keep the benchmark instance self-contained and understandable.
- The underspecified_spec and codification_ready_reference should be parallel: same research idea and method context, but only the gold reference resolves the hidden slot.

Allowed labels:
Level-1 = Ambiguity | Incompleteness | Inconsistency
Level-2 = Ambiguous Definition | Ambiguous Procedure | Missing Algorithmic Procedure | Missing Configuration Protocol | Missing Model Specification | Missing Evaluation Specification | Missing Data Specification | Conflicting Objective | Conflicting Model Design | Conflicting Formal Definition
Granularity = coarse | medium | fine
Resolution role = implementation_blocker | open_design_choice | reproducibility_detail | inconsistency_to_resolve
Codification slot = TASK_AND_IO | CORE_ALGORITHM | MODEL_ARCHITECTURE | OBJECTIVE_AND_SUPERVISION | TRAINING_PROCEDURE | DATA_AND_PREPROCESSING | INFERENCE_AND_DECISION | EVALUATION_PROTOCOL | INTERNAL_CONSISTENCY | NONE
Action type = clarification_question | evidence_seeking | experiment_selection

Fixed defect to use:
{defect_seed_json}

Real gap evidence:
{evidence_json}

Source metadata:
{source_json}

Original paper text excerpt:
\"\"\"
{original_text}
\"\"\"

Reproducibility report text excerpt:
\"\"\"
{report_text}
\"\"\"

Return valid JSON only with exactly this schema:

{{
  "id": {json.dumps(instance_id, ensure_ascii=False)},
  "source": {source_json},
  "input": {{
    "underspecified_spec": ""
  }},
  "gold": {{
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
  }},
  "defects": [
    {{
      "slot": {json.dumps(codification_slot, ensure_ascii=False)},
      "level1": {json.dumps(level1, ensure_ascii=False)},
      "level2": {json.dumps(level2, ensure_ascii=False)},
      "granularity": {json.dumps(granularity, ensure_ascii=False)},
      "resolution_role": {json.dumps(resolution_role, ensure_ascii=False)},
      "codification_slot": {json.dumps(codification_slot, ensure_ascii=False)},
      "gold_detail_removed_or_corrupted": {json.dumps(gold_detail, ensure_ascii=False)},
      "surface_form_in_underspecified_spec": "",
      "why_this_blocks_or_affects_codification": ""
    }}
  ],
  "open_design_choices": [],
  "codification_readiness": {{
    "is_ready": false,
    "readiness_score": 0,
    "blocking_missing_specs": [],
    "open_design_choices": [],
    "reason": ""
  }},
  "expected_clarification_actions": [
    {{
      "slot": {json.dumps(codification_slot, ensure_ascii=False)},
      "action_type": "",
      "question_or_action": "",
      "evidence_to_seek": ""
    }}
  ],
  "construction_metadata": {{
    "construction_method": "real_gap_from_reproducibility_report",
    "paper_id": {json.dumps(item.get("record_id"), ensure_ascii=False)},
    "realgap_id": {json.dumps(item.get("realgap_id"), ensure_ascii=False)},
    "num_defects": 1,
    "selected_perturbations": [
      {defect_seed_json}
    ],
    "gap_quote": {json.dumps(gap.get("gap_quote", ""), ensure_ascii=False)},
    "solution_quote": {json.dumps(gap.get("solution_quote", ""), ensure_ascii=False)},
    "solution_source_type": {json.dumps(solution_source_type, ensure_ascii=False)}
  }}
}}

Output rules:
- defects must contain exactly one defect.
- The defect labels must exactly match the fixed defect above.
- underspecified_spec should be 120-220 words.
- codification_ready_reference should be 160-280 words.
- Both underspecified_spec and codification_ready_reference must start with research goal, task, or method context.
- Both must mention the paper's full research idea, not only the local implementation detail.
- codification_ready_reference must include the gold_clarified_detail.
- underspecified_spec must contain the gap surface form but must NOT include the gold_clarified_detail.
- input.underspecified_spec must NOT contain diagnostic/meta-evaluation leakage language.
- codification_readiness.is_ready must be false.
- readiness_score should be 2 or 3.
- expected_clarification_actions must contain exactly one action and must use only valid action_type labels.
- Do not create extra defects.
"""


# ============================================================
# Validation and eval targets
# ============================================================

def make_single_defect_from_gap(
    gap: Dict[str, Any],
    instance_defect: Dict[str, Any],
    codification_slot: str,
    granularity: str,
    resolution_role: str,
) -> Dict[str, Any]:
    level2 = normalize_level2_label(gap.get("level2"))
    level1 = LEVEL2_TO_LEVEL1.get(level2, gap.get("level1"))
    return {
        "slot": codification_slot,
        "level1": level1,
        "level2": level2,
        "granularity": granularity,
        "resolution_role": resolution_role,
        "codification_slot": codification_slot,
        "gold_detail_removed_or_corrupted": gap.get("gold_clarified_detail", ""),
        "surface_form_in_underspecified_spec": str(
            instance_defect.get("surface_form_in_underspecified_spec", "")
        ).strip(),
        "why_this_blocks_or_affects_codification": str(
            instance_defect.get("why_this_blocks_or_affects_codification", "")
            or gap.get("why_this_blocks_or_affects_codification", "")
        ).strip(),
    }


def force_single_clarification_action(
    actions: List[Dict[str, Any]],
    codification_slot: str,
) -> List[Dict[str, Any]]:
    fixed = []

    for a in actions:
        if not isinstance(a, dict):
            continue

        fixed.append({
            "slot": codification_slot,
            "action_type": normalize_action_type(a.get("action_type", "clarification_question")),
            "question_or_action": str(a.get("question_or_action", "")).strip(),
            "evidence_to_seek": str(a.get("evidence_to_seek", "")).strip(),
        })

    if fixed:
        return [fixed[0]]

    return [{
        "slot": codification_slot,
        "action_type": "clarification_question",
        "question_or_action": "Clarify the missing or ambiguous implementation detail for this method component.",
        "evidence_to_seek": "Author clarification, official code, appendix, or reproducibility report evidence.",
    }]


def normalize_actions(instance: Dict[str, Any]) -> Dict[str, Any]:
    defects = instance.get("defects", [])
    defect = defects[0] if isinstance(defects, list) and defects else {}
    codification_slot = defect.get("codification_slot", "NONE")

    actions = instance.get("expected_clarification_actions", [])
    if not isinstance(actions, list):
        actions = []

    instance["expected_clarification_actions"] = force_single_clarification_action(
        actions=actions,
        codification_slot=codification_slot,
    )
    return instance


def build_eval_targets(instance: Dict[str, Any]) -> Dict[str, Any]:
    eval_defects = []

    for idx, d in enumerate(instance.get("defects", []), start=1):
        eval_defects.append({
            "defect_id": f"d{idx}",
            "slot": d.get("slot", ""),
            "level1": d.get("level1", ""),
            "level2": d.get("level2", ""),
            "granularity": d.get("granularity", ""),
            "resolution_role": d.get("resolution_role", ""),
            "codification_slot": d.get("codification_slot", ""),
        })

    readiness = instance.get("codification_readiness", {})
    score = readiness.get("readiness_score", 2)

    eval_actions = [
        {
            "slot": a.get("slot", ""),
            "action_type": a.get("action_type", ""),
        }
        for a in instance.get("expected_clarification_actions", [])
    ]

    return {
        "underspecified": {
            "expected_ready": False,
            "expected_readiness_score": score,
            "expected_defects": eval_defects,
            "expected_clarification_actions": eval_actions,
        },
        "reference": {
            "expected_ready": True,
            "expected_readiness_score": 5,
            "expected_defects": [],
            "expected_clarification_actions": [],
        },
    }


def force_instance_metadata(
    instance: Dict[str, Any],
    item: Dict[str, Any],
    instance_id: str,
) -> Dict[str, Any]:
    gap = item.get("gap") or {}

    codification_slot = infer_codification_slot(gap)
    granularity = infer_granularity(gap)
    resolution_role = infer_resolution_role(gap, granularity)

    source = build_source_object(item)

    instance["id"] = instance_id
    instance["source"] = source

    # ------------------------------
    # Force gold schema
    # ------------------------------
    if "gold" not in instance or not isinstance(instance["gold"], dict):
        instance["gold"] = {}

    paper_spec = instance["gold"].get("paper_derived_specification", {})
    instance["gold"]["paper_derived_specification"] = normalize_paper_spec(paper_spec)

    instance["gold"]["codification_ready_reference"] = str(
        instance["gold"].get("codification_ready_reference", "")
    ).strip()

    # ------------------------------
    # Force exactly one canonical defect
    # ------------------------------
    original_instance_defect = (
        instance.get("defects", [{}])[0]
        if isinstance(instance.get("defects"), list) and instance.get("defects")
        else {}
    )

    single_defect = make_single_defect_from_gap(
        gap=gap,
        instance_defect=original_instance_defect,
        codification_slot=codification_slot,
        granularity=granularity,
        resolution_role=resolution_role,
    )

    instance["defects"] = [single_defect]

    # ------------------------------
    # Force input schema
    # ------------------------------
    if "input" not in instance or not isinstance(instance["input"], dict):
        instance["input"] = {"underspecified_spec": ""}

    instance["input"]["underspecified_spec"] = str(
        instance["input"].get("underspecified_spec", "")
    ).strip()

    # ------------------------------
    # Force no open design choices in atomic one-defect setting
    # ------------------------------
    instance["open_design_choices"] = []

    # ------------------------------
    # Force codification readiness
    # ------------------------------
    instance.setdefault("codification_readiness", {})
    instance["codification_readiness"]["is_ready"] = False

    try:
        score = int(instance["codification_readiness"].get("readiness_score", 2))
    except Exception:
        score = 2

    score = max(1, min(5, score))
    if score >= 5:
        score = 3

    instance["codification_readiness"]["readiness_score"] = score
    instance["codification_readiness"].setdefault("blocking_missing_specs", [])
    instance["codification_readiness"]["open_design_choices"] = []
    instance["codification_readiness"].setdefault("reason", "")

    # ------------------------------
    # Force exactly one clarification action
    # ------------------------------
    instance = normalize_actions(instance)

    # ------------------------------
    # Force construction metadata to match shared schema
    # selected_perturbations[0] is the same schema as defects[0]
    # ------------------------------
    instance["construction_metadata"] = {
        "construction_method": "real_gap_from_reproducibility_report",
        "paper_id": item.get("record_id"),
        "realgap_id": item.get("realgap_id"),
        "num_defects": 1,
        "selected_perturbations": [single_defect],
        "gap_quote": gap.get("gap_quote", ""),
        "solution_quote": gap.get("solution_quote", ""),
        "solution_source_type": gap.get("solution_source_type", ""),
        "gap_summary": gap.get("gap_summary", ""),
        "solution_summary": gap.get("solution_summary", ""),
    }

    # ------------------------------
    # Eval targets use the forced one-defect / one-action structure
    # ------------------------------
    instance["eval_targets"] = build_eval_targets(instance)

    return instance


def validate_instance(instance: Dict[str, Any], fail_on_input_leakage: bool = False):
    underspecified = instance.get("input", {}).get("underspecified_spec", "")
    reference = instance.get("gold", {}).get("codification_ready_reference", "")

    if not underspecified:
        raise ValueError("Missing input.underspecified_spec")

    if not reference:
        raise ValueError("Missing gold.codification_ready_reference")

    defects = instance.get("defects", [])
    if not isinstance(defects, list) or len(defects) != 1:
        raise ValueError("Expected exactly one defect")

    d = defects[0]

    if d.get("level1") not in LEVEL1_LABELS:
        raise ValueError(f"Invalid level1: {d.get('level1')}")

    if d.get("level2") not in LEVEL2_LABELS:
        raise ValueError(f"Invalid level2: {d.get('level2')}")

    expected_l1 = LEVEL2_TO_LEVEL1.get(d.get("level2"))
    if expected_l1 and d.get("level1") != expected_l1:
        raise ValueError(f"level1/level2 mismatch: {d.get('level1')} / {d.get('level2')}")

    if d.get("granularity") not in GRANULARITY_LABELS:
        raise ValueError(f"Invalid granularity: {d.get('granularity')}")

    if d.get("resolution_role") not in RESOLUTION_ROLE_LABELS:
        raise ValueError(f"Invalid resolution_role: {d.get('resolution_role')}")

    if d.get("codification_slot") not in CODIFICATION_SLOT_LABELS:
        raise ValueError(f"Invalid codification_slot: {d.get('codification_slot')}")

    if not d.get("gold_detail_removed_or_corrupted"):
        raise ValueError("Missing defects[0].gold_detail_removed_or_corrupted")

    actions = instance.get("expected_clarification_actions", [])
    if not isinstance(actions, list) or len(actions) != 1:
        raise ValueError(
            f"Expected exactly one expected_clarification_action, got "
            f"{len(actions) if isinstance(actions, list) else 'non-list'}"
        )

    for a in actions:
        if a.get("slot") not in CODIFICATION_SLOT_LABELS:
            raise ValueError(f"Invalid action slot: {a.get('slot')}")
        if a.get("action_type") not in ACTION_TYPE_LABELS:
            raise ValueError(f"Invalid action_type: {a.get('action_type')}")

    if instance.get("open_design_choices") not in ([], None):
        raise ValueError("Expected open_design_choices to be empty for one-defect instance")

    cm = instance.get("construction_metadata", {})
    if cm.get("num_defects") != 1:
        raise ValueError(
            f"construction_metadata.num_defects must be 1, got {cm.get('num_defects')}"
        )

    selected = cm.get("selected_perturbations", [])
    if not isinstance(selected, list) or len(selected) != 1:
        raise ValueError("construction_metadata.selected_perturbations must contain exactly one item")

    et = instance.get("eval_targets", {}).get("underspecified", {})
    expected_defects = et.get("expected_defects", [])
    expected_actions = et.get("expected_clarification_actions", [])

    if not isinstance(expected_defects, list) or len(expected_defects) != 1:
        raise ValueError("eval_targets.underspecified.expected_defects must contain exactly one item")

    if not isinstance(expected_actions, list) or len(expected_actions) != 1:
        raise ValueError("eval_targets.underspecified.expected_clarification_actions must contain exactly one item")

    ref_target = instance.get("eval_targets", {}).get("reference", {})
    if ref_target.get("expected_ready") is not True:
        raise ValueError("eval_targets.reference.expected_ready must be true")

    if ref_target.get("expected_defects") != []:
        raise ValueError("eval_targets.reference.expected_defects must be []")

    if ref_target.get("expected_clarification_actions") != []:
        raise ValueError("eval_targets.reference.expected_clarification_actions must be []")

    warnings = []

    if word_count(underspecified) < 90:
        warnings.append("underspecified_spec_may_be_too_short")

    if word_count(reference) < 130:
        warnings.append("codification_ready_reference_may_be_too_short")

    if starts_like_local_fix(reference):
        warnings.append("codification_ready_reference_starts_like_local_fix")

    gold_detail = d.get("gold_detail_removed_or_corrupted", "")
    if contains_gold_detail(underspecified, gold_detail):
        warnings.append("possible_gold_detail_leakage_in_underspecified_spec")

    leakage_hits = detect_diagnostic_leakage(underspecified)
    if leakage_hits:
        warnings.append({
            "type": "diagnostic_leakage_in_underspecified_spec",
            "terms": leakage_hits,
        })

    reference_meta_hits = []
    for term in [
        "this resolves the inconsistency",
        "this resolves the ambiguity",
        "this resolves the gap",
        "as clarified by",
        "reproducibility report",
        "specification gap",
        "method gap",
        "implementation gap",
        "defect",
    ]:
        if term in reference.lower():
            reference_meta_hits.append(term)

    if reference_meta_hits:
        warnings.append({
            "type": "meta_language_in_codification_ready_reference",
            "terms": reference_meta_hits,
        })

    instance.setdefault("quality_warnings", [])
    instance["quality_warnings"] = warnings

    if fail_on_input_leakage and leakage_hits:
        raise ValueError(
            "Diagnostic leakage detected in input.underspecified_spec: "
            + ", ".join(leakage_hits)
        )


# ============================================================
# Resume helpers
# ============================================================

def has_saved_instance(out_dir: Path, instance_id: str) -> bool:
    instance_path = out_dir / instance_id / "benchmark_instance.json"
    return instance_path.exists() and instance_path.stat().st_size > 50


def load_existing_index(out_dir: Path) -> List[Dict[str, Any]]:
    index_path = out_dir / "index.jsonl"
    return load_jsonl(index_path) if index_path.exists() else []


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        default="",
        help="Input main_resolved.jsonl from Step 06. If omitted, inferred from dataset_tag.",
    )

    parser.add_argument(
        "--out_dir",
        default="",
        help="Output directory for benchmark instances. If omitted, inferred from dataset_tag.",
    )

    parser.add_argument(
        "--source_filter",
        default="all",
        choices=["all", "OpenReview_MLRC", "OpenReview_TMLR"],
        help="Which source to build instances for.",
    )

    parser.add_argument(
        "--dataset_tag",
        default="",
        help="Dataset tag, e.g., mlrc, tmlr, mixed. If omitted, inferred from source_filter.",
    )

    parser.add_argument(
        "--model",
        default=os.getenv("OPENROUTER_MODEL") or os.getenv("OPENAI_MODEL") or "deepseek/deepseek-v4-pro",
    )

    parser.add_argument("--max_original_chars", type=int, default=50000)
    parser.add_argument("--max_report_chars", type=int, default=40000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save_prompt", action="store_true")

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip records whose benchmark_instance.json already exists.",
    )

    parser.add_argument(
        "--fail_on_input_leakage",
        action="store_true",
        help="Fail the run if input.underspecified_spec contains diagnostic leakage terms.",
    )

    args = parser.parse_args()

    dataset_tag = args.dataset_tag.strip() or infer_dataset_tag_from_source(args.source_filter)

    input_path = Path(args.input_path) if args.input_path else default_input_path(dataset_tag)
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(dataset_tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_all = load_jsonl(input_path)
    rows = [r for r in rows_all if source_matches(r, args.source_filter)]

    if args.limit and args.limit > 0:
        rows = rows[:args.limit]

    client = build_client()

    existing_index = load_existing_index(out_dir) if args.resume else []
    index_rows = existing_index[:] if args.resume else []
    existing_ids = {str(r.get("id") or "") for r in existing_index}

    print("\n===== Universal Step 07: Build Real-Gap Benchmark Instances =====")
    print(f"Input path: {input_path}")
    print(f"Output dir: {out_dir}")
    print(f"Source filter: {args.source_filter}")
    print(f"Dataset tag: {dataset_tag}")
    print(f"Input rows before source filter: {len(rows_all)}")
    print(f"Input rows after source filter/limit: {len(rows)}")
    print(f"Model: {args.model}")
    print(f"Resume: {args.resume}")
    print(f"Save prompt: {args.save_prompt}")
    print(f"Fail on input leakage: {args.fail_on_input_leakage}")
    print("===============================================================\n")

    for idx, item in enumerate(tqdm(rows, desc="Build real-gap benchmark instances"), start=1):
        item = ensure_record_source(item)
        realgap_id = item.get("realgap_id") or f"realgap_{idx:05d}"
        instance_id = safe_filename(realgap_id)

        if args.resume and instance_id in existing_ids and has_saved_instance(out_dir, instance_id):
            print(f"[{idx}/{len(rows)}] skip existing {instance_id}")
            continue

        original_text = read_text(
            item.get("original_text_path", ""),
            max_chars=args.max_original_chars,
        )
        report_text = read_text(
            item.get("report_text_path", ""),
            max_chars=args.max_report_chars,
        )

        prompt = build_prompt(
            item=item,
            original_text=original_text,
            report_text=report_text,
            instance_id=instance_id,
        )

        instance_dir = out_dir / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)

        if args.save_prompt:
            (instance_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        try:
            raw = call_llm(client=client, model=args.model, prompt=prompt)
            (instance_dir / "raw_response.txt").write_text(raw, encoding="utf-8")

            instance = safe_json_loads(raw)
            instance = force_instance_metadata(
                instance=instance,
                item=item,
                instance_id=instance_id,
            )

            validate_instance(
                instance,
                fail_on_input_leakage=args.fail_on_input_leakage,
            )

            instance_path = instance_dir / "benchmark_instance.json"
            save_json(instance, instance_path)

            index_row = {
                "id": instance_id,
                "realgap_id": realgap_id,
                "record_id": item.get("record_id"),
                "source": infer_record_source(item),
                "source_type": instance.get("source", {}).get("source_type"),
                "report_title": item.get("report_title"),
                "original_paper_title": item.get("original_paper_title"),
                "benchmark_instance_path": str(instance_path),
                "level1": instance["defects"][0]["level1"],
                "level2": instance["defects"][0]["level2"],
                "codification_slot": instance["defects"][0]["codification_slot"],
                "construction_method": "real_gap_from_reproducibility_report",
                "quality_warnings": instance.get("quality_warnings", []),
                "status": "saved",
            }

            index_rows.append(index_row)
            save_jsonl(index_rows, out_dir / "index.jsonl")

            print(f"[{idx}/{len(rows)}] saved {instance_path}")
            if instance.get("quality_warnings"):
                print(f"  warnings: {instance.get('quality_warnings')}")

        except Exception as e:
            error_path = instance_dir / "error.json"
            save_json(
                {
                    "id": instance_id,
                    "realgap_id": realgap_id,
                    "record_id": item.get("record_id"),
                    "source": infer_record_source(item),
                    "error": repr(e),
                },
                error_path,
            )

            index_row = {
                "id": instance_id,
                "realgap_id": realgap_id,
                "record_id": item.get("record_id"),
                "source": infer_record_source(item),
                "report_title": item.get("report_title"),
                "original_paper_title": item.get("original_paper_title"),
                "benchmark_instance_path": "",
                "level1": item.get("gap", {}).get("level1"),
                "level2": item.get("gap", {}).get("level2"),
                "codification_slot": infer_codification_slot(item.get("gap", {}) or {}),
                "construction_method": "real_gap_from_reproducibility_report",
                "quality_warnings": [],
                "status": "failed",
                "error_path": str(error_path),
                "error": repr(e),
            }

            index_rows.append(index_row)
            save_jsonl(index_rows, out_dir / "index.jsonl")

            print(f"[{idx}/{len(rows)}] failed {instance_id}: {repr(e)}")

    save_jsonl(index_rows, out_dir / "index.jsonl")

    saved = sum(r.get("status") == "saved" for r in index_rows)
    failed = sum(r.get("status") == "failed" for r in index_rows)

    source_counter = Counter(str(r.get("source") or "missing") for r in index_rows)
    level1_counter = Counter(str(r.get("level1") or "missing") for r in index_rows if r.get("status") == "saved")
    level2_counter = Counter(str(r.get("level2") or "missing") for r in index_rows if r.get("status") == "saved")
    warning_counter = Counter()

    for r in index_rows:
        for w in r.get("quality_warnings", []) or []:
            if isinstance(w, dict):
                warning_counter[w.get("type", "dict_warning")] += 1
            else:
                warning_counter[str(w)] += 1

    summary = {
        "script": "step_07_build_realgap_bench_instances.py",
        "input_path": str(input_path),
        "out_dir": str(out_dir),
        "source_filter": args.source_filter,
        "dataset_tag": dataset_tag,
        "input_rows_before_source_filter": len(rows_all),
        "input_rows_after_source_filter_or_limit": len(rows),
        "model": args.model,
        "instances_saved": saved,
        "instances_failed": failed,
        "source_distribution": dict(source_counter),
        "saved_level1_distribution": dict(level1_counter),
        "saved_level2_distribution": dict(level2_counter),
        "quality_warning_distribution": dict(warning_counter),
        "index_path": str(out_dir / "index.jsonl"),
        "next_step": (
            "Human inspect generated benchmark_instance.json files for leakage, "
            "gold faithfulness, and parallelism between underspecified_spec and codification_ready_reference."
        ),
    }

    save_json(summary, out_dir / "summary.json")

    print("\n===== Real-gap benchmark construction summary =====")
    print(f"Input gaps: {len(rows)}")
    print(f"Instances saved: {saved}")
    print(f"Instances failed: {failed}")
    print(f"Output dir: {out_dir}")
    print(f"Index: {out_dir / 'index.jsonl'}")
    print(f"Summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
