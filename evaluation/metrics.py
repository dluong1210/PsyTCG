"""User-level Precision, Recall, and F1.

A run maps each user ID to chronological ``(round_id, decision, score)`` tuples.
The user-level decision is positive once any round is positive. Golden users
missing from a run are treated as negative predictions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

Prediction = tuple[str, int, float]
Run = Mapping[str, Sequence[Prediction]]
Golden = Mapping[str, int]
REPORT_METRICS = ["precision", "recall", "f1"]


def confusion_counts(run: Run, golden: Golden) -> dict[str, int]:
    """Return user-level TP, FP, FN, and TN counts."""
    tp = fp = fn = tn = 0
    for user_id, truth_value in golden.items():
        predictions = run.get(user_id, ())
        predicted = any(bool(decision) for _, decision, _ in predictions)
        truth = bool(truth_value)
        if predicted and truth:
            tp += 1
        elif predicted and not truth:
            fp += 1
        elif not predicted and truth:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def precision_recall_f1(run: Run, golden: Golden) -> tuple[float, float, float]:
    """Compute user-level Precision, Recall, and F1."""
    counts = confusion_counts(run, golden)
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / ((tp + fp) or 1)
    recall = tp / ((tp + fn) or 1)
    f1 = 2 * tp / ((2 * tp + fp + fn) or 1)
    return precision, recall, f1


def evaluate_run(run: Run, golden: Golden) -> dict[str, float]:
    """Return the three metrics reported by the release code."""
    precision, recall, f1 = precision_recall_f1(run, golden)
    return {"precision": precision, "recall": recall, "f1": f1}


def is_better(score: float | None, best: float | None, metric: str) -> bool:
    """Whether a validation score improves the selected classification metric."""
    if metric not in REPORT_METRICS:
        raise ValueError(f"unsupported selection metric: {metric}")
    if score is None or not math.isfinite(score):
        return False
    return best is None or score > best
