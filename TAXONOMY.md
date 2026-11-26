# Canonical IdeaAMBIG taxonomy

The following three Level-1 types and ten Level-2 labels are the final taxonomy.
`taxonomy.json` is its machine-readable registry and is identical in the website,
ARR paper, and arXiv paper repositories. Table numbers vary between paper formats;
the taxonomy does not.

| Level 1 | Level 2 |
|---|---|
| Ambiguity | Ambiguous Definition |
| Ambiguity | Ambiguous Procedure |
| Incompleteness | Missing Method Procedure |
| Incompleteness | Missing Model Structure |
| Incompleteness | Missing Data Specification |
| Incompleteness | Missing Configuration Protocol |
| Incompleteness | Missing Evaluation Specification |
| Inconsistency | Conflicting Objective |
| Inconsistency | Conflicting Model Design |
| Inconsistency | Conflicting Formal Definition |

## Release and reproducibility policy

- All published `level1` / `level2` fields use the canonical names above.
- Main benchmark: 163 real-world and 497 synthetic single-defect records.
- Multi-defect ablation: 50 records, with two canonical target defects per record.
- Historical model outputs, input specifications, evidence, IDs, and evaluation
  scores are preserved. Labels are not silently rewritten by this audit.
- `codification_slot` and affected-component names are separate from defect types.
- `none` in report routing means that no usable candidate was extracted. It is
  a routing sentinel, not an additional Level-1 or Level-2 category.
- Old objective/loss category strings in `DISALLOWED_LEVEL2` or
  `disallowed_labels` are intentional rejection rules. Keep these filters: replacing
  them with a valid current label could incorrectly reject valid instances.
- `taxonomy.json` matches the canonical registry in both paper repositories and
  the project website. The legacy display aliases are historical names only.

## Verify before publishing

Run `python3 scripts/check_taxonomy.py` from the repository root.
The check reads all released data and parses the Python taxonomy declarations,
without importing pipeline modules, calling a model, or modifying any records.
