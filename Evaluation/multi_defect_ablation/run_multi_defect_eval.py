"""Run the full multi-defect ablation: GPT-5.6-Sol on Task 1 (readiness) and
Task 2 (localization) for the 50 constructed k=2 multi-defect instances and
their 100 matched single-defect baseline siblings, then score Task 2 with
the Claude Opus 4.8 semantic judge (same judge used elsewhere in the paper).

Run from the repository root (paths are relative to it); requires
Data/syn_multi_defect_k2.jsonl from build_multi_defect_synthetic.py:
    export OPENROUTER_API_KEY=...
    export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
    python Evaluation/multi_defect_ablation/run_multi_defect_eval.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "task1"))
sys.path.insert(0, str(Path(__file__).parent.parent / "task2"))
from task1_config import build_user_prompt as t1_prompt, system_prompt as t1_system, option_mapping  # noqa: E402
from track1_config import build_user_prompt as t2_prompt_single, SYSTEM_PROMPT as t2_system  # noqa: E402
from run_semantic_judge import JUDGE_SYSTEM_PROMPT, JUDGE_SCHEMA, build_judge_prompt  # noqa: E402

from openai import OpenAI

GEN_MODEL = "openai/gpt-5.6-sol"
JUDGE_MODEL = "anthropic/claude-opus-4.8"


def t2_prompt_multi(input_text: str) -> str:
    base = t2_prompt_single(input_text)
    base = base.replace(
        "Each specification in this benchmark is constructed with exactly one annotated\n"
        "target defect. Your goal is not to find all possible weaknesses. Your goal is\n"
        "to recover the single target defect category that best explains why the\n"
        "specification is not implementation-ready.",
        "This specification may contain more than one implementation-critical\n"
        "specification defect. Identify every such defect you can find and classify\n"
        "each one according to the IDEAAMBIG taxonomy. Do not list ordinary\n"
        "engineering choices or speculative concerns as defects.",
    )
    base = base.replace(
        'Return only valid JSON:\n{\n  "description": "one concrete atomic defect diagnosis",\n'
        '  "level1": "Ambiguity|Incompleteness|Inconsistency",\n  "level2": "one allowed Level-2 label"\n}',
        'Return only valid JSON:\n{\n  "defects": [\n    {"description": "one concrete atomic defect diagnosis",\n'
        '     "level1": "Ambiguity|Incompleteness|Inconsistency",\n     "level2": "one allowed Level-2 label"}\n  ]\n}',
    )
    return base


def gen_call(client: OpenAI, system: str, user: str, json_mode: bool = False, retries: int = 3) -> str:
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    last = None
    for _ in range(retries):
        try:
            resp = client.chat.completions.create(
                model=GEN_MODEL, temperature=0.0,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                **kwargs,
            )
            return resp.choices[0].message.content
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2)
    raise last


def judge_call(client: OpenAI, spec: str, target_defect: dict, predicted_description: str, retries: int = 3) -> bool:
    prompt = build_judge_prompt(spec, target_defect, predicted_description)
    last = None
    for _ in range(retries):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL, temperature=0,
                messages=[{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                response_format={"type": "json_schema", "json_schema": JUDGE_SCHEMA},
            )
            val = json.loads(resp.choices[0].message.content)
            return bool(val["match"])
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2)
    raise last


def target_defect_from(defect: dict) -> dict:
    return {
        "description": defect.get("why_this_blocks_or_affects_codification") or defect.get("gold_detail_removed_or_corrupted"),
        "surface_form": defect.get("surface_form_in_underspecified_spec"),
        "missing_or_corrupted_detail": defect.get("gold_detail_removed_or_corrupted"),
        "blocking_missing_specs": [],
    }


def eval_single(client: OpenAI, inst: dict) -> dict:
    spec = inst["input"]["underspecified_spec"]
    defect = inst["defects"][0]

    mapping = option_mapping(inst["id"], seed=0)
    t1_reply = gen_call(client, t1_system("v1"), t1_prompt(spec, mapping, "v1")).strip().upper()
    letter = "A" if "A" in t1_reply[:2] else ("B" if "B" in t1_reply[:2] else None)
    readiness_pred = mapping.get(letter, "PARSE_ERROR")

    t2_reply = gen_call(client, t2_system, t2_prompt_single(spec), json_mode=True)
    pred_desc = json.loads(t2_reply).get("description", "")
    matched = judge_call(client, spec, target_defect_from(defect), pred_desc)

    return {
        "id": inst["id"], "condition": "single",
        "readiness_pred": readiness_pred,
        "loc_match": matched,
    }


def eval_multi(client: OpenAI, inst: dict) -> dict:
    spec = inst["input"]["underspecified_spec"]
    defects = inst["defects"]

    mapping = option_mapping(inst["id"], seed=0)
    t1_reply = gen_call(client, t1_system("v1"), t1_prompt(spec, mapping, "v1")).strip().upper()
    letter = "A" if "A" in t1_reply[:2] else ("B" if "B" in t1_reply[:2] else None)
    readiness_pred = mapping.get(letter, "PARSE_ERROR")

    t2_reply = gen_call(client, t2_system, t2_prompt_multi(spec), json_mode=True)
    pred_defects = json.loads(t2_reply).get("defects", [])
    pred_descs = [d.get("description", "") for d in pred_defects if d.get("description")]

    hits = []
    for gold in defects:
        target = target_defect_from(gold)
        found = any(judge_call(client, spec, target, pd) for pd in pred_descs) if pred_descs else False
        hits.append(found)

    return {
        "id": inst["id"], "condition": "multi", "slot_stratum": inst["slot_stratum"],
        "readiness_pred": readiness_pred,
        "num_predicted_defects": len(pred_descs),
        "hits": hits,
        "any_hit": any(hits),
        "all_hit": all(hits),
    }


def main() -> None:
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )

    multi_rows = [json.loads(l) for l in open("Data/syn_multi_defect_k2.jsonl")]
    syn_by_id = {json.loads(l)["id"]: json.loads(l) for l in open("Data/syn_benchmark_instance.jsonl")}

    results = {"single": [], "multi": []}

    for i, inst in enumerate(multi_rows):
        r = eval_multi(client, inst)
        results["multi"].append(r)
        print(f"[multi {i+1}/{len(multi_rows)}] {inst['id']} readiness={r['readiness_pred']} any={r['any_hit']} all={r['all_hit']} n_pred={r['num_predicted_defects']}")

        for base_id in inst["baseline_single_defect_ids"]:
            base_inst = syn_by_id[base_id]
            rb = eval_single(client, base_inst)
            results["single"].append(rb)
            print(f"    [baseline] {base_id} readiness={rb['readiness_pred']} match={rb['loc_match']}")

    Path("Data/multi_defect_eval_results.json").write_text(json.dumps(results, indent=2))
    print("Wrote Data/multi_defect_eval_results.json")


if __name__ == "__main__":
    main()
