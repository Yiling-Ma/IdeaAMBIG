import json
import os
import argparse
from collections import Counter


# ----------------------------
# SPEC SLOTS (NEW KEY IMPROVEMENT)
# ----------------------------

SPEC_SLOTS = {
    "training": [
        "training", "train", "optimization", "optimizer", "backprop",
        "epoch", "epochs", "batch size", "lr", "learning rate"
    ],
    "architecture": [
        "architecture", "layer", "hidden size", "embedding", "transformer",
        "encoder", "decoder", "backbone"
    ],
    "evaluation": [
        "evaluation", "metric", "metrics", "benchmark", "test set",
        "validation", "accuracy", "f1", "auc"
    ],
    "data": [
        "dataset", "data", "split", "preprocess", "preprocessing",
        "normalization", "tokenization"
    ],
    "objective": [
        "loss", "objective", "function", "reward"
    ],
    "inference": [
        "inference", "decoding", "sampling", "generation"
    ],
    "hyperparameters": [
        "hyperparameter", "config", "configuration", "seed", "checkpoint"
    ]
}


POSITIVE_SPEC_KEYWORDS = [
    "paper", "implementation", "implement",
    "result", "results", "table", "figure",
    "missing", "unclear", "ambiguous",
    "not specified", "does not mention",
    "what is the", "how to", "which"
]

# STRONG noise (expanded)
NOISE_KEYWORDS = [
    "install", "installation", "pip", "conda",
    "cuda", "gpu memory", "oom", "out of memory",
    "docker", "permission denied",
    "download", "dataset access",
    "version conflict", "dependency",
    "requirements", "syntax error",
    "importerror", "modulenotfounderror",
    "environment", "setup issue"
]

# reproduction-only signal 
REPRO_ONLY_KEYWORDS = [
    "cannot reproduce", "unable to reproduce",
    "reproduce results", "replicate results",
    "performance lower", "does not match paper"
]


def count_hits(text, keywords):
    text = (text or "").lower()
    return [k for k in keywords if k in text]


def extract_text(issue):
    parts = [issue.get("title", ""), issue.get("body", "")]
    for c in issue.get("comments", []):
        parts.append(c.get("body", ""))
    return "\n".join(parts)


# ----------------------------
# SLOT DETECTION (NEW CORE)
# ----------------------------
def detect_spec_slots(text):
    text = text.lower()
    slot_hits = {}

    for slot, kws in SPEC_SLOTS.items():
        hits = [k for k in kws if k in text]
        if hits:
            slot_hits[slot] = hits

    return slot_hits


def compute_score(issue):
    text = extract_text(issue)

    pos_hits = count_hits(text, POSITIVE_SPEC_KEYWORDS)
    noise_hits = count_hits(text, NOISE_KEYWORDS)
    repro_hits = count_hits(text, REPRO_ONLY_KEYWORDS)

    slot_hits = detect_spec_slots(text)

    maintainer_signal = issue.get("weak_resolution_signal", {}).get("has_maintainer_comment", False)
    pr_signal = issue.get("weak_resolution_signal", {}).get("has_linked_pr", False)
    commit_signal = issue.get("weak_resolution_signal", {}).get("has_linked_commit", False)

    score = 0

    # ----------------------------
    # 1. spec gap strength
    # ----------------------------
    score += 2 * len(pos_hits)

    # slot-aware bonus (VERY IMPORTANT)
    score += 2 * len(slot_hits)

    # ----------------------------
    # 2. resolution strength
    # ----------------------------
    if maintainer_signal:
        score += 3
    if pr_signal:
        score += 3
    if commit_signal:
        score += 2

    # ----------------------------
    # 3. reproduction penalty (NEW)
    # ----------------------------
    score -= 2 * len(repro_hits)

    # ----------------------------
    # 4. noise penalty (stronger than before)
    # ----------------------------
    score -= 3 * len(noise_hits)

    return score, pos_hits, noise_hits, repro_hits, slot_hits


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=str,
        default="Real_bench/github_issue_mining/outputs_step2/raw_issues_closed_answered.jsonl"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="Real_bench/github_issue_mining/outputs_step3"
    )

    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--max_output", type=int, default=800)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    out_path = os.path.join(args.out_dir, "step3_spec_gap_candidates.jsonl")
    stats_path = os.path.join(args.out_dir, "step3_stats.json")

    kept = []
    stats = Counter()

    with open(args.input_path, "r", encoding="utf-8") as f:
        for line in f:
            issue = json.loads(line)

            score, pos_hits, noise_hits, repro_hits, slot_hits = compute_score(issue)

            stats["total"] += 1
            stats["kept_raw"] += 1 if score >= args.threshold else 0

            issue["step3_score"] = score
            issue["step3_pos_hits"] = pos_hits
            issue["step3_noise_hits"] = noise_hits
            issue["step3_repro_hits"] = repro_hits
            issue["step3_slot_hits"] = slot_hits

            # ----------------------------
            # FINAL FILTER RULE
            # ----------------------------
            if (
                score >= args.threshold
                and len(pos_hits) >= 2
                and len(noise_hits) <= 3
            ):
                kept.append(issue)
                stats["kept_final"] += 1

            if len(kept) >= args.max_output:
                break

    kept = sorted(kept, key=lambda x: x["step3_score"], reverse=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for x in kept:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n[done]")
    print(f"total issues: {stats['total']}")
    print(f"kept: {stats['kept_final']}")
    print(f"output: {out_path}")

    print("\n[top 10]")
    for i, x in enumerate(kept[:10]):
        print("=" * 80)
        print(i + 1)
        print("repo:", x.get("repo"))
        print("score:", x.get("step3_score"))
        print("slots:", x.get("step3_slot_hits"))
        print("pos_hits:", x.get("step3_pos_hits"))
        print("repro_hits:", x.get("step3_repro_hits"))
        print("noise_hits:", x.get("step3_noise_hits"))


if __name__ == "__main__":
    main()