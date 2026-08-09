# Syn Route

This directory contains the cleaned public-code version of the controlled
synthetic-instance construction pipeline. It produces the 497 controlled
synthetic instances in Table 3 (358 reproducibility + 139 ideation-execution).

## Layout

- `reproducibility/`
  Synthetic-controlled pipeline for reproducibility-report sources, covering ECIR, MLRC, and TMLR.
  Starts from the same paper–report pairs routed to `synthetic_controlled` by
  `Real_Route/step_03_llm_route_classification.py`; venue-specific PDF/source
  collection is handled by `Real_Route/step_01_download_pdfs.py` and
  `step_02_fast_pymupdf_text_markdown.py` and is not duplicated here.

  | Step | Script | Purpose |
  |---|---|---|
  | 1 | `step_01_filter_synthetic_controlled.py` | Filter routed pairs down to the synthetic-controlled candidates. |
  | 2 | `step_02_prepare_original_paper_markdown.py` | Prepare Markdown of the source paper for gold-spec extraction. |
  | 3 | `step_03_extract_gold_spec.py` | Extract the codification-ready gold reference specification. |
  | 4 | `step_04_repair_gold_references.py` | Repair/complete gold references using report + paper evidence. |
  | 5a | `step_05a_generate_raw_synthetic_instances.py` | Generate up to 5 controlled candidates per project by removing/abstracting exactly one implementation-critical detail. |
  | 5b | `step_05b_filter_raw_synthetic_instances.py` | Filter raw candidates for atomicity and validity. |
  | 6 | `step_06_quality_filter_and_finalize.py` | Final quality filtering and assembly of benchmark-ready instances. |

  `stats_final_index_labels.py` reports label/index statistics over the
  finalized set.

- `execution_study/`
  Synthetic construction from the AI-Researcher ideation–execution release
  (43 executed projects with a final paper + reference codebase each).

  | Step | Script | Purpose |
  |---|---|---|
  | 0 | `step_00_download_dataset.py` | Download the ideation-execution dataset release. |
  | 1 | `step_01_build_project_manifest.py` | Build a manifest of executed projects with paper + codebase pairs. |
  | 2 | `step_02_extract_gold_specs.py` | Extract the codification-ready reference specification per project. |
  | 3a | `step_03a_generate_raw_synthetic_instances.py` | Generate controlled candidates by removing/abstracting one implementation-critical detail. |
  | 3b | `step_03b_filter_raw_synthetic_instances.py` | Filter raw candidates for atomicity and validity. |
  | 4 | `step_04_quality_filter_and_finalize.py` | Final quality filtering and assembly. |
  | 5 | `step_05_llm_audit_instances.py` | LLM audit pass before human verification. |

Both tracks share `common.py` (I/O and download helpers) and
`quality_patterns.py` (shared heuristics for enforcing "exactly one
implementation-critical detail changed, all non-target content preserved" —
see paper Appendix E.1, criteria 1–2 for controlled synthetic candidates).

## Naming convention

- Step scripts use `step_XX_description.py`.
- Helper modules use plain snake_case, e.g. `common.py`, `quality_patterns.py`.
- Directories use lowercase snake_case.

## Notes

- Output folders, PDFs, markdown dumps, prompts, cached LLM responses, and local virtual environments are ignored by `.gitignore`.
- This directory is intended to keep the public code only; historical local outputs remain reproducible artifacts and should stay outside version control.
