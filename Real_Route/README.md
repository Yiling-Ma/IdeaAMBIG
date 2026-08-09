# Real Route

This directory contains the public-code version of the real-world (naturally
occurring) gap construction pipeline. It has two independent tracks that feed
the two real-world sources described in the paper:

- Top-level `step_01`–`step_07`: **reproducibility-report track**
  (ML Reproducibility Challenge, ECIR, TMLR reports paired with the paper they
  reproduce) → 121 real instances.
- `github_issue/`: **GitHub-issue track**
  (closed/answered issues on paper-linked repositories, 2017–2025) → 42 real
  instances.

121 + 42 = 163 real-world instances (Table 3).

## Reproducibility-report track (`step_01`–`step_07`)

| Step | Script | Purpose |
|---|---|---|
| 1 | `step_01_download_pdfs.py` | Download reproducibility-report and paired source-paper PDFs (MLRC / ECIR / TMLR). |
| 2 | `step_02_fast_pymupdf_text_markdown.py` | Fast PyMuPDF-based PDF→Markdown conversion, used as a first-pass text extraction before the heavier MinerU pass. |
| 3 | `step_03_llm_route_classification.py` | LLM routing/classification of each paper–report pair into `resolved_real_gap`, `synthetic_controlled`, or `unusable` (Figure 2 / Figure 7). |
| 4 | `step_04_mineru_selected_candidates.py` | Re-runs MinerU (layout-aware PDF parsing) only on the candidates selected in step 3, for higher-fidelity text needed for detailed extraction. |
| 5 | `step_05_detailed_extract_and_quote_validate.py` | LLM extraction of atomic gap/resolution candidates from the routed pairs, with quote-level validation against the source text. |
| 6 | `step_06_postprocess_gap_candidates.py` | Deduplication, normalization, and taxonomy-label cleanup of extracted candidates. |
| 7 | `step_07_build_realgap_bench_instances.py` | Assembles the final `x⁻`/`x⁺`/defect/clarification benchmark records for the reproducibility-report track. |

Each script follows the same `--input_path` / `--out_path` convention, so
steps can be chained by feeding one step's `--out_path` as the next step's
`--input_path`. Human verification (two-stage review, see paper §3.2 and
Appendix E.1) is applied to the step-7 output outside of this code.

## GitHub-issue track (`github_issue/`)

| Step | Script | Purpose |
|---|---|---|
| 1 | `step_01_seed_repos.py` | Seed the set of paper-linked GitHub repositories (2017–2025) to crawl. |
| 2 | `step_02_crawl_closed_and_answered_issues.py` | Crawl closed/answered issues from the seeded repositories. |
| 3 | `step_03_prefilter_spec_gap_issues.py` | Lexical/heuristic prefilter for issues containing gap-and-resolution signals. |
| 4 | `step_04_llm_validate.py` | LLM classification and decomposition of prefiltered issues into candidate specification gaps. |
| 5 | `step_05_download_original_papers.py` | Download the source paper associated with each validated candidate's repository. |
| 6 | `step_06_postprocess_and_quote_validate.py` | Postprocessing and quote-level validation of candidates against the paper/issue evidence. |
| 7 | `step_07_build_benchmark_instances.py` | Assembles the final benchmark records for the GitHub-issue track. |
| 8 | `step_08_audit_benchmark_instances.py` | Schema/taxonomy audit of the assembled instances before human review. |

`taxonomy_utils.py` holds the shared Level-1/Level-2 taxonomy definitions
(`LEVEL2_TO_LEVEL1`, boundary rules, normalization helpers) used across the
GitHub-issue track; it mirrors the taxonomy in the paper's Appendix B.3.

`step_07_build_benchmark_instances.py` loads `step_07_build_realgap_bench_instances.py`
(one directory up) at runtime for shared helpers — this is intentional
cross-track reuse, not a duplicate implementation.

## Taxonomy

Both tracks use the same 3 Level-1 types and 10 Level-2 categories defined in
the paper (Appendix B.3):

- **Ambiguity**: Ambiguous Definition, Ambiguous Procedure
- **Incompleteness**: Missing Method Procedure, Missing Configuration Protocol,
  Missing Model Structure, Missing Evaluation Specification, Missing Data
  Specification
- **Inconsistency**: Conflicting Objective, Conflicting Model Design,
  Conflicting Formal Definition

## Notes

- Human & LLM verification (two-stage review, target-uniqueness audit) is
  applied downstream of these scripts, as described in Appendix E.1–E.2 of
  the paper.
- Large intermediate artifacts (PDFs, Markdown dumps, MinerU output, cached
  LLM responses, per-step JSONL outputs) are ignored by `.gitignore` and are
  not part of this public code release.
