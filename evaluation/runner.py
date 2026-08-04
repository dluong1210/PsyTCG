"""Run model inference and compute classification metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from evaluation.metrics import evaluate_run
from infer.runner import predict_run


def evaluate_model(
    model,
    user_ids: Sequence[str],
    timelines: dict[str, list[dict[str, Any]]],
    golden: dict[str, int],
    threshold: float = 0.5,
    post_batch_size: int = 128,
    device: torch.device | None = None,
    collect_rows: bool = False,
    desc: str = "eval",
):
    """Infer over all labelled users, then return Precision/Recall/F1.

    ``user_ids`` must include pruned users. Their missing timelines become
    negative predictions and therefore remain eligible false negatives.
    """
    result = predict_run(
        model,
        user_ids,
        timelines,
        threshold=threshold,
        post_batch_size=post_batch_size,
        device=device,
        collect_rows=collect_rows,
        desc=desc,
    )
    run, rows = result if collect_rows else (result, None)
    metrics = evaluate_run(run, golden)
    return (metrics, run, rows) if collect_rows else (metrics, run)
