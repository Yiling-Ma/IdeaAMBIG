# Data

The released IdeaAMBIG benchmark: 660 evidence-grounded, single-defect
instances (Table 3).

| File | Instances | Source |
|---|---|---|
| `real_benchmark_instance.jsonl` | 163 | 121 from GitHub issues + 42 from reproducibility reports (`Real_Route/`) |
| `syn_benchmark_instance.jsonl` | 497 | 358 from reproducibility reports + 139 from AI-Researcher ideation–execution trajectories (`Syn_Route/`) |

Each line is one JSON instance: an underspecified `NOT_READY` specification
(`input.underspecified_spec`) paired with its `READY` counterpart
(`gold.codification_ready_reference`), a single annotated target defect, and
a gold clarification action.

## Schema

```
id                          unique instance identifier
source                      provenance: source type, paper/report title and URL(s),
                             GitHub repo/issue (real-world) or construction method
input.underspecified_spec   the NOT_READY specification (x⁻)
gold.codification_ready_reference   the READY specification (x⁺)
defects[0]                  the single target defect:
                               level1              Ambiguity | Incompleteness | Inconsistency
                               level2              one of the 10 categories below
                               codification_slot    one of the 10 specification slots below
                               granularity, resolution_role
                               gold_detail_removed_or_corrupted   the hidden/altered gold detail
                               surface_form_in_underspecified_spec
                               why_this_blocks_or_affects_codification
open_design_choices          explicitly open, non-blocking design choices (if any)
codification_readiness       is_ready, readiness_score, blocking_missing_specs, reason
expected_clarification_actions   gold clarification action(s): action_type, question_or_action,
                                  evidence_to_seek
evidence                     (real_benchmark_instance.jsonl only) supporting quotes:
                               gap_quote, solution_quote, solution_source_type,
                               gap_summary, solution_summary
eval_targets                  per-task evaluation targets used by Evaluation/:
                               underspecified   expected labels for the NOT_READY side
                               reference        expected labels for the READY side
                                                 (always is_ready=true, empty defects/actions)
```

## Taxonomy

3 Level-1 types and 10 Level-2 categories (definitions in the paper's
Appendix B.3; the same taxonomy is implemented in
`Real_Route/github_issue/taxonomy_utils.py` and mirrored across `Syn_Route/`):

| Level-1 | Level-2 |
|---|---|
| Ambiguity | Ambiguous Definition, Ambiguous Procedure |
| Incompleteness | Missing Method Procedure, Missing Model Structure, Missing Data Specification, Missing Configuration Protocol, Missing Evaluation Specification |
| Inconsistency | Conflicting Objective, Conflicting Model Design, Conflicting Formal Definition |

`codification_slot` uses 10 specification slots: `TASK_AND_IO`,
`CORE_ALGORITHM`, `MODEL_ARCHITECTURE`, `OBJECTIVE_AND_SUPERVISION`,
`TRAINING_PROCEDURE`, `DATA_AND_PREPROCESSING`, `INFERENCE_AND_DECISION`,
`EVALUATION_PROTOCOL`, `INTERNAL_CONSISTENCY`, `NONE`.

## Notes

- Every instance is single-defect: `defects` (and the corresponding
  `eval_targets.underspecified.expected_defects`) always has exactly one
  element.
- The `reference` side of `eval_targets` always describes a READY
  specification with no defect: `expected_defects: []` and
  `expected_clarification_actions: []` are expected there, not a bug.
- Construction-time-only bookkeeping (internal file paths, LLM routing
  rationale, per-step QA/audit trails) has been stripped from these files;
  see the paper (Appendix E) for the full construction and verification
  protocol.
