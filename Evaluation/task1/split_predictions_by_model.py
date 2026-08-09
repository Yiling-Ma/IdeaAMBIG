"""Split a legacy combined Task 1 prediction JSONL into per-model files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from task1_config import DEFAULT_PREDICTIONS_DIR, load_jsonl, model_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--pool",
        type=Path,
        help="Optional pool whose case IDs restrict which successful rows are migrated.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--prompt-version", default="v2")
    parser.add_argument("--pool-tag", default="real100")
    parser.add_argument("--subset", choices=("all", "real", "synthetic"), default="real")
    return parser.parse_args()


def successful_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "ok" or not isinstance(row.get("predicted_ready"), bool):
            continue
        model = str(row.get("model") or "").strip()
        case_id = str(row.get("case_id") or "").strip()
        if model and case_id:
            selected[(model, case_id)] = row
    return selected


def main() -> None:
    args = parse_args()
    rows = successful_rows(load_jsonl(args.input.resolve()))
    if args.pool is not None:
        allowed_case_ids = {
            str(row["case_id"]) for row in load_jsonl(args.pool.resolve())
        }
        rows = {
            key: row for key, row in rows.items() if key[1] in allowed_case_ids
        }
    by_model: dict[str, list[dict[str, Any]]] = {}
    for (model, _), row in rows.items():
        by_model.setdefault(model, []).append(row)

    for model, model_rows in sorted(by_model.items()):
        path = (
            args.output_dir
            / args.prompt_version
            / args.pool_tag
            / args.subset
            / model_filename(model)
        )
        existing = successful_rows(load_jsonl(path)) if path.exists() else {}
        pending = [
            row
            for row in sorted(model_rows, key=lambda value: str(value["case_id"]))
            if (model, str(row["case_id"])) not in existing
        ]
        if pending:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                for row in pending:
                    handle.write(
                        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
        print(f"{model}: added {len(pending)}, total {len(existing) + len(pending)} -> {path}")


if __name__ == "__main__":
    main()
