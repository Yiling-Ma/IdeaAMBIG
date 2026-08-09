# IdeaAmbig

Code accompanying the anonymous ACL submission **"IdeaAMBIG: Benchmarking
Implementation-Critical Gaps in Research-Idea Specifications."**

IdeaAMBIG evaluates whether a research-idea specification is *codification-ready*
— whether a competent implementer or coding agent could build a faithful
initial implementation without inventing unsupported assumptions about the
core method — through three tasks: **Readiness Assessment**, **Defect
Localization**, and **Clarification Action Generation**. The benchmark
contains 660 evidence-grounded, single-defect instances (163 real-world +
497 controlled synthetic; Table 3), each pairing an underspecified `NOT_READY`
specification with a `READY` counterpart, a labeled target defect, and a gold
clarification action.

This repository contains the **public construction and evaluation code**. It
does not include the released dataset, cached LLM outputs, PDFs, or other
large intermediate artifacts — see each subdirectory's `.gitignore`.

## Repository layout

```
ARR_codes/
├── Real_Route/        Real-world gap construction (GitHub issues + reproducibility reports) → 163 instances
├── Syn_Route/          Controlled synthetic-defect construction → 497 instances
└── Evaluation/         Task 1/2/3 evaluation pipeline (prompting, judging, scoring)
```

Each subdirectory has its own README with step-by-step usage:

- [`Real_Route/README.md`](Real_Route/README.md) — real-world instance
  construction from GitHub issues and reproducibility reports.
- [`Syn_Route/README.md`](Syn_Route/README.md) — controlled synthetic-defect
  construction from reproducibility reports and AI-Researcher
  ideation–execution trajectories.
- [`Evaluation/README.md`](Evaluation/README.md) — running the 13 evaluated
  LLMs on Tasks 1–3, LLM-judge scoring, the Table 9 information-bottleneck
  ablation, and the Table 2 oracle clarification-utility study.

## Data construction pipeline (overview)

```
                 ┌───────────── Real Gaps ─────────────┐   ┌────── Synthetic Gaps ──────┐
                 │                                      │   │                            │
        GitHub Issues                        Reproducibility Reports         AI-Researcher Execution Study
   (Real_Route/github_issue/)                  (Real_Route/step_01–07)         (Syn_Route/execution_study/)
                 │                                      │   │                            │
                 ▼                                      ▼   ▼                            ▼
          resolved real gap              resolved real gap / synthetic-controlled   codification-ready
          (42 instances)                  (121 instances)   / unusable               reference + 5 candidate
                                                                  │                    defect injections
                                                                  ▼                    (139 instances)
                                                    Syn_Route/reproducibility/
                                                       (358 instances)
```

Both real and synthetic tracks share the same **3 Level-1 defect types**
(Ambiguity, Incompleteness, Inconsistency) and **10 Level-2 categories**,
defined in `Real_Route/github_issue/taxonomy_utils.py` and mirrored across
`Syn_Route/`:

| Level-1 | Level-2 categories |
|---|---|
| Ambiguity | Ambiguous Definition, Ambiguous Procedure |
| Incompleteness | Missing Method Procedure, Missing Configuration Protocol, Missing Model Structure, Missing Evaluation Specification, Missing Data Specification |
| Inconsistency | Conflicting Objective, Conflicting Model Design, Conflicting Formal Definition |

All construction paths apply the same two-stage human verification protocol
(primary screen + independent second review of retained/uncertain cases) and
enforce that each retained instance isolates exactly one atomic,
evidence-supported, implementation-critical defect.

## Evaluation

`Evaluation/` runs 13 LLMs (accessed via OpenRouter) on the three benchmark
tasks and reports the paper's primary metrics: Macro-F1 (Task 1 readiness
assessment), Macro Defect Recovery Rate (Task 2 defect localization), and
Macro Clarification Action Success Rate (Task 3 clarification generation).
Semantic and rubric-based judging uses `anthropic/claude-opus-4.8`. See
[`Evaluation/README.md`](Evaluation/README.md) for exact commands, including
the Table 9 end-to-end vs. defect-guided ablation and the Table 2 oracle
clarification-utility study.

## Setup

Each subdirectory is a standalone Python 3 project (no top-level
`requirements.txt` is provided; install packages as import errors surface —
common dependencies include `requests`, `tqdm`, `beautifulsoup4`, `pymupdf`,
and `gdown`). Scripts that call an LLM provider read credentials from
environment variables, e.g.:

```bash
export OPENROUTER_API_KEY="..."
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
```

No credentials, personal paths, or author-identifying information are
included in this release.

## License and anonymity

This is an anonymous submission repository. Please do not include information
that could reveal author identity in issues, commits, or forks of this
repository during the review period.
