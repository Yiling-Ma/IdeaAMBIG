<p align="center">
  <img src="docs/identity/ideaambig-logo.svg" alt="IdeaAMBIG logo: a specification gap and its missing piece" width="180">
</p>

<h1 align="center">
  <img src="docs/identity/ideaambig-wordmark.svg" alt="IdeaAMBIG:" width="205" height="36" align="top"> Benchmarking Implementation-Critical<br>
  Gaps in Research-Idea Specifications
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2609.10539">
    <img src="docs/identity/badges/arxiv.svg" alt="arXiv: 2609.10539">
  </a>
  <a href="https://yiling-ma.github.io/IdeaAMBIG-website/">
    <img src="docs/identity/badges/website.svg" alt="Website: Homepage" title="IdeaAMBIG project website">
  </a>
  <a href="https://github.com/Yiling-Ma/IdeaAMBIG">
    <img src="docs/identity/badges/code.svg" alt="Code: GitHub">
  </a>
</p>

<p align="center">
  <a href="#overview">Overview</a> &nbsp;&middot;&nbsp;
  <a href="#benchmark">Benchmark</a> &nbsp;&middot;&nbsp;
  <a href="#construction-pipeline">Construction</a> &nbsp;&middot;&nbsp;
  <a href="#results">Results</a> &nbsp;&middot;&nbsp;
  <a href="#quick-start">Quick Start</a> &nbsp;&middot;&nbsp;
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="docs/figures/pipeline.png" width="100%" alt="IdeaAMBIG construction pipeline: source artifacts, instance construction, and two-stage human verification">
</p>
<p align="center">
  <em><strong>Construction pipeline of IdeaAMBIG.</strong> ① Collect source artifacts from GitHub issues, reproducibility reports, and executed research. ② Reconstruct real-world gaps and inject single defects into codification-ready references. ③ Two-stage human verification yields 660 evidence-grounded instances.</em>
</p>

---

## Overview

**IdeaAMBIG** benchmarks **codification readiness**: whether a research-method specification provides enough information for a faithful initial implementation without unsupported assumptions about the core method.

- **660 evidence-grounded instances**: 163 real-world gaps and 497 controlled synthetic gaps.
- **Paired specifications**: each instance includes a `NOT_READY` specification and a `READY` counterpart that resolves only its target defect.
- **Three diagnostic tasks**: readiness assessment, defect localization, and clarification action generation.
- **A shared taxonomy**: 3 defect types and 10 categories across 10 research domains.
- **13 evaluated LLMs**: proprietary and open-weight models, with separate real-world and synthetic results.

**The central finding:** discovering the implementation blocker is harder than formulating a clarification once it is known. The strongest evaluated model reaches **9.6% Macro DRR** on real-world defect localization, versus **80.6% Macro-CAS** when clarification is guided by the annotated blocker.

---

## Benchmark

### Three Evaluation Tasks

| Task | Input | Output | Metric |
| --- | --- | --- | --- |
| **1 · Readiness Assessment** | One specification | `READY` / `NOT_READY` + rationale | Macro-F1 |
| **2 · Defect Localization** | Underspecified specification | Target blocker + taxonomy labels | Macro DRR |
| **3 · Clarification Generation** | Specification + **annotated defect** | Question or evidence-seeking action + required information | Macro-CAS |

Macro DRR requires recovery of the target blocker and both taxonomy labels. Macro-CAS requires a relevant action that requests sufficient information and avoids unsupported assumptions. Task 3 evaluates clarification formulation given a known blocker; the end-to-end variant evaluates clarification without this hint.

### Dataset Composition

| Subset | Source | Instances |
| --- | --- | ---: |
| Real-world | GitHub issues | 121 |
| Real-world | Reproducibility reports | 42 |
| **Real-world subtotal** | | **163** |
| Controlled synthetic | Reproducibility reports | 358 |
| Controlled synthetic | Ideation–execution trajectories | 139 |
| **Synthetic subtotal** | | **497** |
| **Total** | | **660** |

Every main-benchmark instance contains one atomic, implementation-critical defect, its supported resolution, and a gold clarification action. See [Data/README.md](Data/README.md) for the JSONL schema and the separate two-defect ablation.

### Defect Taxonomy

| Defect type | Category |
| --- | --- |
| **Ambiguity** | Ambiguous Definition |
|  | Ambiguous Procedure |
| **Incompleteness** | Missing Method Procedure |
|  | Missing Model Structure |
|  | Missing Data Specification |
|  | Missing Configuration Protocol |
|  | Missing Evaluation Specification |
| **Inconsistency** | Conflicting Objective |
|  | Conflicting Model Design |
|  | Conflicting Formal Definition |

---

## Construction Pipeline

Source artifacts provide evidence for resolved real-world gaps or codification-ready references. Real-world instances reconstruct documented gaps; controlled synthetic instances alter exactly one method-defining detail in a ready reference. Primary screening and independent second review retain atomic, self-contained, evidence-supported cases.

| Route | Sources | Documentation |
| --- | --- | --- |
| **Real-world** | GitHub issues and reproducibility reports | [Real_Route](Real_Route/README.md) |
| **Controlled synthetic** | Reproducibility reports and executed projects | [Syn_Route](Syn_Route/README.md) |

---

## Results

### Main Results Across 13 LLMs

#### Real-world

| Model | Readiness ↑ | Localization ↑ | Clarification ↑ |
| --- | ---: | ---: | ---: |
| *Frontier proprietary models* |  |  |  |
| GPT-5.6-Sol | **67.5** | **9.6** | **80.6** |
| Claude Sonnet 5 | *59.3* | *6.8* | *76.8* |
| Gemini 3.1 Pro Preview | 45.8 | 3.4 | 68.9 |
| DeepSeek-V3.2 | 44.5 | 5.9 | 62.9 |
| *Open-weight reasoning models* |  |  |  |
| Qwen3.5-397B-A17B | 57.5 | 3.3 | 72.0 |
| GLM-5.2 | 45.1 | 5.7 | 63.6 |
| DeepSeek-R1-0528 | 44.3 | 1.8 | 68.0 |
| Kimi-K3 | 32.7 | 6.6 | 60.6 |
| *Open-weight general models* |  |  |  |
| Gemma-4-31B-IT | 47.1 | 4.7 | 67.7 |
| Qwen3-8B | 40.1 | 3.2 | 53.8 |
| Qwen3-32B | 34.5 | 3.6 | 61.9 |
| Qwen3.5-9B | 33.3 | 5.3 | 64.6 |
| GPT-OSS-120B | 30.0 | 2.5 | 53.4 |

#### Controlled synthetic

| Model | Readiness ↑ | Localization ↑ | Clarification ↑ |
| --- | ---: | ---: | ---: |
| *Frontier proprietary models* |  |  |  |
| GPT-5.6-Sol | **86.4** | **12.2** | **96.2** |
| Claude Sonnet 5 | *76.2* | *11.4* | *94.5* |
| Gemini 3.1 Pro Preview | 67.3 | 7.6 | 91.6 |
| DeepSeek-V3.2 | 67.2 | 8.0 | 93.0 |
| *Open-weight reasoning models* |  |  |  |
| Qwen3.5-397B-A17B | 55.9 | 4.6 | 89.0 |
| GLM-5.2 | 65.9 | 7.0 | 77.0 |
| DeepSeek-R1-0528 | 59.9 | 2.2 | 68.5 |
| Kimi-K3 | 43.1 | 8.2 | 92.5 |
| *Open-weight general models* |  |  |  |
| Gemma-4-31B-IT | 56.9 | 7.0 | 73.0 |
| Qwen3-8B | 48.7 | 4.6 | 82.6 |
| Qwen3-32B | 37.0 | 5.8 | 76.3 |
| Qwen3.5-9B | 54.3 | 5.5 | 81.2 |
| GPT-OSS-120B | 44.4 | 3.3 | 65.3 |

<sub><em>All scores are percentages; higher is better. **Bold** and *italics* indicate the best and second-best scores in each column. Readiness = Macro-F1; Localization = Macro DRR; Clarification = Macro-CAS. Task 1 uses 100 READY and 100 NOT_READY specifications per subset. Tasks 2 and 3 use all 163 real-world instances and a fixed sample of 200 synthetic instances.</em></sub>

### What Changes When the Blocker Is Given?

With GPT-5.6-Sol on the same 163 real-world instances, providing the annotated blocker raises clarification success by **67.0 percentage points**.

| Metric | End-to-End | Defect-Guided |
| --- | ---: | ---: |
| Macro-CAS ↑ | 13.6 | **80.6** |
| Sufficiency ↑ | 8.6 | **80.4** |
| No-Assumption ↑ | **96.3** | 95.7 |

### Does Resolving the Gap Help Implementation?

In the **oracle clarification study**, supplying the gold resolution raises the READY rate from **14% to 98%** on 50 real-world instances, evaluated by two human annotators blinded to the generation condition. On 20 controlled instances with bounded implementation components, oracle clarification also improves executable code:

| Metric | Direct Generation | Clarification-Assisted |
| --- | ---: | ---: |
| Runnable ↑ | 70.0 | **95.0** |
| All Tests Pass ↑ | 45.0 | **85.0** |
| Target Fidelity ↑ | 30.0 | **90.0** |
| Further Clarification ↓ | 35.0 | **5.0** |

These utility studies supply the gold resolution and measure the benefit of successful clarification under oracle conditions.

---

## Quick Start

### 1. Get the Dataset

```bash
git clone https://github.com/Yiling-Ma/IdeaAMBIG.git
cd IdeaAMBIG
```

Read the released JSONL files with Python 3; no extra packages are needed:

```python
import json
from pathlib import Path

def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

real = load_jsonl("Data/real_benchmark_instance.jsonl")
synthetic = load_jsonl("Data/syn_benchmark_instance.jsonl")
print(f"Real: {len(real)} | Synthetic: {len(synthetic)}")

case = real[0]
print(case["input"]["underspecified_spec"])
print(case["defects"][0])
print(case["expected_clarification_actions"])
```

### 2. Build a Task 1 Evaluation Pool

This example builds a pool from the real-world release and separates model-visible inputs from gold annotations:

```bash
python3 Evaluation/task1/build_evaluation_pool.py \
  --input Data/real_benchmark_instance.jsonl \
  --output-dir /tmp/ideaambig-task1
```

This creates `evaluation_pool.jsonl`, `evaluation_gold.jsonl`, and `evaluation_manifest.json`. The pool includes all underspecified inputs and deduplicated ready references; the paper's balanced Task 1 evaluation uses 100 specifications per class and subset.

### 3. Run a Model

```bash
python3 -m pip install tqdm
export OPENROUTER_API_KEY="YOUR_API_KEY"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"

python3 Evaluation/task1/run_baselines_openrouter.py \
  --pool /tmp/ideaambig-task1/evaluation_pool.jsonl \
  --gold /tmp/ideaambig-task1/evaluation_gold.jsonl \
  --models openai/gpt-5.6-sol \
  --prompt-version v3 \
  --subset real \
  --workers 4
```

See [Evaluation/README.md](Evaluation/README.md) for per-task runners, judging, scoring, and clarification ablations. Construction routes have their own dependencies and usage instructions.

### Repository Structure

| Directory | Contents |
| --- | --- |
| [Data](Data/README.md) | Benchmark files, schema, and multi-defect data |
| [Real_Route](Real_Route/README.md) | Real-world gap construction |
| [Syn_Route](Syn_Route/README.md) | Controlled synthetic construction |
| [Evaluation](Evaluation/README.md) | Task runners, judging, scoring, and clarification studies |
| [docs](docs/README.md) | Logo, badges, and paper figures |

---

## Citation

```bibtex
@article{ma2026ideaambig,
  title  = {IdeaAMBIG: Benchmarking Implementation-Critical Gaps in Research-Idea Specifications},
  author = {Ma, Yiling and Zhao, Yilun and Wu, Sihong and Patwardhan, Manasi and Cohan, Arman},
  year   = {2026},
  journal = {arXiv preprint arXiv:2609.10539},
  url    = {https://arxiv.org/abs/2609.10539}
}
```

## Taxonomy consistency

The final three Level-1 types and ten Level-2 categories are specified in
[TAXONOMY.md](TAXONOMY.md) and [taxonomy.json](taxonomy.json), matching both
paper versions and the project website. Run `python3 scripts/check_taxonomy.py`
to verify released records and the construction/evaluation label declarations.
