"""Construct k=2 multi-defect synthetic instances by combining two
already-validated, independently-injected single-defect sibling instances
that share the same codification_ready_reference (same base project).

This does not fabricate new defects: both target defects, their gold
details, and their evidence already passed the existing single-defect
construction and audit pipeline. We only ask the model to write ONE
underspecified_spec that simultaneously obscures both known details,
instead of regenerating content from scratch.

Run from the repository root (paths are relative to it):
    export OPENROUTER_API_KEY=...
    export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
    python Syn_Route/multi_defect_ablation/build_multi_defect_synthetic.py \
        --n_same_slot 25 --n_diff_slot 25 --seed 0
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "reproducibility"))
from common import LLMJsonClient, load_jsonl, save_jsonl  # noqa: E402

SYSTEM_PROMPT = """You are constructing a controlled multi-defect instance for a research-method \
implementation benchmark. You are given a codification-ready reference \
specification and TWO independently validated target defects, each already \
associated with a known gold detail and a known vague/omitted surface form \
that was used to obscure it in a single-defect version of this project.

Your task is to write ONE underspecified specification that simultaneously \
obscures BOTH target details, reusing the given surface-form style for each \
detail where natural, while preserving every other part of the reference \
unchanged. Do not resolve, hint at, or partially reveal either gold detail. \
Do not introduce a third defect. Do not use diagnostic words such as \
missing, unspecified, defect, gold, removed detail, or benchmark. The result \
must read as a single natural, coherent method specification between 120 \
and 260 words.

Return strict JSON only: {"underspecified_spec": "...", "preservation_check": \
{"both_details_obscured": true/false, "no_third_defect_introduced": true/false, \
"non_target_content_preserved": true/false}, "notes": "<=40 words on any \
concern with combining these two defects, or empty string"}
"""

USER_TEMPLATE = """Codification-ready reference specification:
{reference}

Target defect 1 (level1={level1_a}, level2={level2_a}, slot={slot_a}):
- gold detail: {gold_a}
- known obscuring surface form: {surface_a}

Target defect 2 (level1={level1_b}, level2={level2_b}, slot={slot_b}):
- gold detail: {gold_b}
- known obscuring surface form: {surface_b}
"""


def group_by_project(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        m = re.match(r"(.+)_direct_synthetic_defect_\d+$", row["id"])
        base = m.group(1) if m else row["id"]
        groups.setdefault(base, []).append(row)
    return groups


def build_instance(client: LLMJsonClient, raw_dir: Path, sib_a: dict, sib_b: dict) -> dict:
    ref = sib_a["gold"]["codification_ready_reference"]
    da, db = sib_a["defects"][0], sib_b["defects"][0]
    user = USER_TEMPLATE.format(
        reference=ref,
        level1_a=da["level1"], level2_a=da["level2"], slot_a=da["codification_slot"],
        gold_a=da["gold_detail_removed_or_corrupted"],
        surface_a=da["surface_form_in_underspecified_spec"],
        level1_b=db["level1"], level2_b=db["level2"], slot_b=db["codification_slot"],
        gold_b=db["gold_detail_removed_or_corrupted"],
        surface_b=db["surface_form_in_underspecified_spec"],
    )
    combo_id = f"{sib_a['id']}__PLUS__{sib_b['id'].split('_defect_')[-1]}"
    raw_path = raw_dir / f"{combo_id}.json"
    result = client.call_json(SYSTEM_PROMPT, user, raw_path)
    return {
        "id": combo_id,
        "project_id": re.match(r"(.+)_direct_synthetic_defect_\d+$", sib_a["id"]).group(1),
        "num_defects": 2,
        "gold": {"codification_ready_reference": ref},
        "input": {"underspecified_spec": result.get("underspecified_spec")},
        "defects": [da, db],
        "preservation_check": result.get("preservation_check"),
        "construction_notes": result.get("notes", ""),
        "baseline_single_defect_ids": [sib_a["id"], sib_b["id"]],
    }


def pair_options(sibs: list[dict]) -> tuple[list[tuple[dict, dict]], list[tuple[dict, dict]]]:
    """Return (same_slot_pairs, diff_slot_pairs) for all sibling pairs in a project."""
    same, diff = [], []
    for i in range(len(sibs)):
        for j in range(i + 1, len(sibs)):
            a, b = sibs[i], sibs[j]
            if a["defects"][0]["codification_slot"] == b["defects"][0]["codification_slot"]:
                same.append((a, b))
            else:
                diff.append((a, b))
    return same, diff


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_same_slot", type=int, default=25)
    ap.add_argument("--n_diff_slot", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="deepseek/deepseek-v4-pro")
    ap.add_argument("--out", default="Data/syn_multi_defect_k2.jsonl")
    args = ap.parse_args()

    rows = load_jsonl(Path("Data/syn_benchmark_instance.jsonl"))
    groups = group_by_project(rows)
    eligible = [k for k, v in groups.items() if len(v) >= 2]
    rng = random.Random(args.seed)
    rng.shuffle(eligible)

    # Assign each project to at most one stratum so no project is reused
    # across the two strata, prioritizing the scarcer diff-slot pool first.
    used_projects: set[str] = set()
    diff_selected: list[tuple[str, tuple[dict, dict]]] = []
    same_selected: list[tuple[str, tuple[dict, dict]]] = []

    for proj in eligible:
        if len(diff_selected) >= args.n_diff_slot:
            break
        same, diff = pair_options(groups[proj])
        if diff:
            diff_selected.append((proj, rng.choice(diff)))
            used_projects.add(proj)

    for proj in eligible:
        if len(same_selected) >= args.n_same_slot:
            break
        if proj in used_projects:
            continue
        same, diff = pair_options(groups[proj])
        if same:
            same_selected.append((proj, rng.choice(same)))
            used_projects.add(proj)

    if len(diff_selected) < args.n_diff_slot or len(same_selected) < args.n_same_slot:
        raise RuntimeError(
            f"Not enough eligible projects: got {len(diff_selected)} diff-slot "
            f"and {len(same_selected)} same-slot (needed {args.n_diff_slot} and "
            f"{args.n_same_slot})."
        )

    selected = [(proj, pair, "diff_slot") for proj, pair in diff_selected] + [
        (proj, pair, "same_slot") for proj, pair in same_selected
    ]
    rng.shuffle(selected)

    client = LLMJsonClient(args.model, True)
    raw_dir = Path("multi_defect_raw")
    out_rows = []
    for i, (proj, (sib_a, sib_b), stratum) in enumerate(selected):
        inst = build_instance(client, raw_dir, sib_a, sib_b)
        inst["slot_stratum"] = stratum
        out_rows.append(inst)
        print(f"{i + 1}/{len(selected)} built {inst['id']} [{stratum}]")

    save_jsonl(out_rows, Path(args.out))
    print(f"Wrote {len(out_rows)} instances to {args.out} "
          f"({len(diff_selected)} diff_slot, {len(same_selected)} same_slot)")


if __name__ == "__main__":
    main()
