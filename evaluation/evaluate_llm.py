#!/usr/bin/env python3
"""Evaluate LLM JSONL predictions with Precision, Recall, and F1.

Expected record::

    {"user_id": "subject_x", "conversation_index": 0,
     "submission_id": "abc", "depression": false}

Golden users missing from a prediction file are treated as negative predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.metrics import evaluate_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, nargs="+", help="LLM JSONL files.")
    parser.add_argument("--golden", type=Path, required=True, help="Golden label text file.")
    parser.add_argument(
        "--require-all-users",
        action="store_true",
        help="Fail when a golden user has no prediction records.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def parse_binary(value: Any, location: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and not isinstance(value, bool) and value in {0, 1}:
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "positive", "pos"}:
            return 1
        if normalized in {"0", "false", "no", "negative", "neg", "control"}:
            return 0
    raise ValueError(f"{location}: expected a binary value, got {value!r}")


def load_golden(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    golden: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_number}: expected '<user_id> <label>'")
            user_id, label = fields
            if user_id in golden:
                raise ValueError(f"{path}:{line_number}: duplicate user {user_id!r}")
            golden[user_id] = parse_binary(label, f"{path}:{line_number}")
    if not golden:
        raise ValueError(f"empty golden file: {path}")
    return golden


def load_run(path: Path, golden: dict[str, int], require_all_users: bool):
    if not path.is_file():
        raise FileNotFoundError(path)
    grouped: dict[str, list[tuple[int, str, int, float]]] = {}
    seen: set[tuple[str, int]] = set()

    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")

            user_id = record.get("user_id")
            if not isinstance(user_id, str) or not user_id:
                raise ValueError(f"{path}:{line_number}: missing user_id")
            if user_id not in golden:
                raise ValueError(f"{path}:{line_number}: unknown user {user_id!r}")
            index = record.get("conversation_index")
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise ValueError(f"{path}:{line_number}: invalid conversation_index")
            key = (user_id, index)
            if key in seen:
                raise ValueError(f"{path}:{line_number}: duplicate prediction {key}")
            seen.add(key)

            decision = parse_binary(record.get("depression"), f"{path}:{line_number}")
            round_id = str(record.get("submission_id", index))
            score_value = record.get("score", decision)
            if isinstance(score_value, bool) or not isinstance(score_value, (int, float)):
                raise ValueError(f"{path}:{line_number}: score must be numeric")
            grouped.setdefault(user_id, []).append((index, round_id, decision, float(score_value)))

    missing = sorted(set(golden) - set(grouped))
    if require_all_users and missing:
        raise ValueError(f"missing {len(missing)} golden users; first: {missing[:5]}")

    run = {}
    for user_id in golden:
        records = sorted(grouped.get(user_id, []))
        indices = [index for index, *_ in records]
        if indices and indices != list(range(len(indices))):
            raise ValueError(f"{user_id}: conversation indices must be contiguous from zero")
        run[user_id] = [(round_id, decision, score) for _, round_id, decision, score in records]
    return run, len(missing)


def main() -> None:
    args = parse_args()
    golden = load_golden(args.golden)
    results = []
    for path in args.predictions:
        run, missing = load_run(path, golden, args.require_all_users)
        results.append(
            {
                "predictions": str(path),
                "users": len(golden),
                "missing_users_as_negative": missing,
                "metrics": evaluate_run(run, golden),
            }
        )

    if args.json:
        print(json.dumps(results, indent=2))
        return
    for result in results:
        print(f"\n{result['predictions']}")
        print(f"  users: {result['users']} ({result['missing_users_as_negative']} missing)")
        for metric, value in result["metrics"].items():
            print(f"  {metric:>9}: {value:.4f}")


if __name__ == "__main__":
    main()
