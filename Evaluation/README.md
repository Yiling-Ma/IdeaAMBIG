# Evaluation

This directory contains the public evaluation pipeline for the benchmark.

The active entry points are:

- `task1/`: Readiness Assessment
- `task2/`: Defect Localization
- `task3/`: Clarification Action Generation

These task directories are the source of truth for the current evaluation
setup. Older top-level helper modules have been removed from this public
release because they were stale and not part of the maintained pipeline.

All commands below should be run from the repository root:

```bash
cd /path/to/ARR_codes
```

Set API credentials:

```bash
export OPENROUTER_API_KEY="..."
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
```

## Shared structure

Each task directory contains:

- `build_evaluation_pool.py`: construct or regenerate task-specific evaluation files
- `run_baselines_openrouter.py`: generate model predictions
- task-specific judge scripts where needed
- task-specific scoring scripts

All three tasks support:

- `--subset real`
- `--subset synthetic`
- `--subset all`

Models only see the public evaluation pool. Subset filtering is done from
private gold metadata.

## Task 1

Directory:

```text
Evaluation/task1/
```

Build the default evaluation pool:

```bash
python Evaluation/task1/build_evaluation_pool.py
```

Run one model on the real subset:

```bash
python Evaluation/task1/run_baselines_openrouter.py \
  --models openai/gpt-5.6-sol \
  --prompt-version v1 \
  --subset real \
  --workers 4
```

Run one model on the synthetic subset:

```bash
python Evaluation/task1/run_baselines_openrouter.py \
  --models openai/gpt-5.6-sol \
  --prompt-version v1 \
  --subset synthetic \
  --workers 4
```

Score Task 1 classification:

```bash
python Evaluation/task1/score_task1.py \
  --predictions Evaluation/task1/outputs/predictions/v1/full/all \
  --subsets all,real,synthetic
```

Score Task 1 reason grounding:

```bash
python Evaluation/task1/score_reason_grounding.py \
  --predictions Evaluation/task1/outputs_synth/predictions/gpt-5.6-sol.jsonl \
  --gold Evaluation/task1/synthetic120_evaluation_gold_reason_grounding.jsonl \
  --benchmark /path/to/Syn_bench/syn_benchmark_instance.jsonl \
  --judge-model anthropic/claude-opus-4.8 \
  --output-dir Evaluation/task1/outputs_synth/reason_grounding/gpt-5.6-sol
```

## Task 2

Directory:

```text
Evaluation/task2/
```

Build the evaluation pool:

```bash
python Evaluation/task2/build_evaluation_pool.py
```

Run one model on the real subset:

```bash
python Evaluation/task2/run_baselines_openrouter.py \
  --models openai/gpt-5.6-sol \
  --subset real \
  --workers 4 \
  --output Evaluation/task2/outputs_real/predictions/gpt-5.6-sol.jsonl
```

Run semantic matching:

```bash
python Evaluation/task2/run_semantic_judge.py \
  --predictions Evaluation/task2/outputs_real/predictions/gpt-5.6-sol.jsonl \
  --output Evaluation/task2/outputs_real/judgments/gpt-5.6-sol.jsonl \
  --judge-model anthropic/claude-opus-4.8 \
  --subset real \
  --models openai/gpt-5.6-sol \
  --workers 4
```

Score Task 2:

```bash
python Evaluation/task2/score_track1.py \
  --predictions Evaluation/task2/outputs_real/predictions/gpt-5.6-sol.jsonl \
  --judgments Evaluation/task2/outputs_real/judgments/gpt-5.6-sol.jsonl \
  --output-dir Evaluation/task2/outputs_real/scores/gpt-5.6-sol \
  --judge-model anthropic/claude-opus-4.8 \
  --subsets real \
  --models openai/gpt-5.6-sol
```

## Task 3

Directory:

```text
Evaluation/task3/
```

Build the evaluation pool:

```bash
python Evaluation/task3/build_evaluation_pool.py
```

Run one model on the real subset:

```bash
python Evaluation/task3/run_baselines_openrouter.py \
  --models openai/gpt-5.6-sol \
  --subset real \
  --workers 4
```

Run the action judge:

```bash
python Evaluation/task3/run_action_judge.py \
  --predictions Evaluation/task3/outputs/predictions/v1/real \
  --output Evaluation/task3/outputs/action_judgments_gpt56_real.jsonl \
  --judge-model anthropic/claude-opus-4.8 \
  --models openai/gpt-5.6-sol \
  --subset real \
  --workers 4
```

Score Task 3:

```bash
python Evaluation/task3/score_task3.py \
  --predictions Evaluation/task3/outputs/predictions/v1/real \
  --judgments Evaluation/task3/outputs/action_judgments_gpt56_real.jsonl \
  --output-dir Evaluation/task3/outputs/scores_gpt56_real \
  --judge-model anthropic/claude-opus-4.8 \
  --subsets real \
  --models openai/gpt-5.6-sol
```

## Table 9 — End-to-End vs Defect-Guided (information bottleneck, Appendix D.2)

Isolates whether the Task 2→3 gap is caused by defect *discovery* (END-TO-END,
no blocker given) or clarification *formulation* (DEFECT-GUIDED, the standard
Task 3 setting above). Both use GPT-5.6-Sol, identical prompts/decoding.

Run the End-to-End condition (no annotated blocker):

```bash
python Evaluation/task3/run_information_bottleneck_e2e.py \
  --models openai/gpt-5.6-sol \
  --subset real \
  --workers 4
```

Score it the same way as the Defect-Guided run (`score_task3.py`, see Task 3
above), then combine both `metrics.json` outputs into the Table 9 rows:

```bash
python Evaluation/task3/summarize_information_bottleneck.py \
  --end-to-end Evaluation/task3/outputs/information_bottleneck/scores_end_to_end/metrics.json \
  --defect-guided Evaluation/task3/outputs/scores_gpt56_real/metrics.json \
  --model openai/gpt-5.6-sol \
  --subset real
```

## Table 2 — Oracle clarification utility (Section 4.4)

Compares DIRECT GENERATION (spec only) against CLARIFICATION-ASSISTED
GENERATION (spec + gold clarification + oracle resolution) on paired
instances, using an LLM judge for READY rate, completeness, missing-detail
recovery, and unsupported-assumption rate.

Generate both conditions' downstream specifications:

```bash
python Evaluation/task3/downstream_clarification.py \
  --model openai/gpt-5.6-sol \
  --sample-size 50 \
  --workers 4
```

Judge the generated specifications:

```bash
python Evaluation/task3/run_downstream_judge.py \
  --judge-model anthropic/claude-opus-4.8 \
  --workers 4
```

Score and run paired significance tests:

```bash
python Evaluation/task3/score_downstream_judge.py \
  --judge-model anthropic/claude-opus-4.8 \
  --bootstrap-samples 10000
```

Note: `score_downstream_judge.py` runs a paired bootstrap for all four
metrics. The paper's Table 2 additionally reports two-sided exact McNemar
tests for the binary outcomes (READY Rate, Unsupported Assumption Rate), a
paired permutation test for Completeness, and Holm-adjusted p-values — apply
those separately over this script's judgment output if reproducing the exact
Table 2 statistics.

`run_baselines_non_llm.py` provides non-LLM (heuristic) baselines for Task 3
and is not part of the main model-comparison pipeline above.

## Release hygiene

This public directory should not include:

- `__pycache__/`
- `.DS_Store`
- local output artifacts

Use a clean working tree before publishing.
