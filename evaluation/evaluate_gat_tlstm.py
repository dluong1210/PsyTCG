#!/usr/bin/env python3
"""Evaluate a trained GAT--TLSTM checkpoint on one collection."""

from __future__ import annotations

import argparse
from pathlib import Path

from config.schema import Config
from data_processing.dataset import load_timelines
from data_processing.labels import load_labels
from evaluation.metrics import REPORT_METRICS
from evaluation.runner import evaluate_model
from infer.loaders import load_gat_tlstm
from utils.common import get_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--timelines", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    device = get_device()
    timelines = load_timelines(args.timelines or config.data.test_timelines)
    labels = load_labels(args.labels or config.data.test_labels)
    threshold = args.threshold if args.threshold is not None else config.train.decision_threshold

    model = load_gat_tlstm(args.checkpoint, device)

    user_ids = sorted(labels)
    metrics, _, rows = evaluate_model(
        model,
        user_ids,
        timelines,
        labels,
        threshold=threshold,
        post_batch_size=config.train.post_batch_size,
        device=device,
        collect_rows=True,
        desc="eval GAT-TLSTM",
    )
    for metric in REPORT_METRICS:
        print(f"  {metric:>10}: {metrics[metric]:.4f}")

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        rows.to_csv(args.output / "predictions.csv", index=False)
        write_json(
            args.output / "metrics.json",
            {"checkpoint": args.checkpoint, "threshold": threshold, "metrics": metrics},
        )
        print(f"\nartifacts -> {args.output}")


if __name__ == "__main__":
    main()
