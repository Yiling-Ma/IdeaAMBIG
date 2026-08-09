import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Print level1 / level2 label distribution for a synthetic benchmark index JSONL."
    )
    parser.add_argument(
        "--index_path",
        default="Syn_bench/synthetic_benchmark_postfiltered/final_index.jsonl",
    )
    args = parser.parse_args()

    rows = load_jsonl(Path(args.index_path))
    level1_counter = Counter(str(r.get("level1") or "n/a") for r in rows)
    level2_counter = Counter(str(r.get("level2") or "n/a") for r in rows)
    joint_counter = Counter(
        f"{r.get('level1', 'n/a')} / {r.get('level2', 'n/a')}" for r in rows
    )

    print(f"Index: {args.index_path}")
    print(f"Total instances: {len(rows)}")
    print()

    print("level1 distribution:")
    for k, v in sorted(level1_counter.items(), key=lambda x: (-x[1], x[0])):
        pct = 100.0 * v / len(rows) if rows else 0.0
        print(f"  {k}: {v} ({pct:.1f}%)")

    print()
    print("level2 distribution:")
    for k, v in sorted(level2_counter.items(), key=lambda x: (-x[1], x[0])):
        pct = 100.0 * v / len(rows) if rows else 0.0
        print(f"  {k}: {v} ({pct:.1f}%)")

    print()
    print("level1 / level2 joint distribution:")
    for k, v in sorted(joint_counter.items(), key=lambda x: (-x[1], x[0])):
        pct = 100.0 * v / len(rows) if rows else 0.0
        print(f"  {k}: {v} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
