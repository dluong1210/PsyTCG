#!/usr/bin/env python3
"""Score a trained checkpoint on a collection.

Example::

    python -m evaluation.evaluate_psytcg --config config/psytcg_erisk2026.yaml \
        --checkpoint runs/erisk2026/seed_3407/checkpoint.pt \
        --timelines data/graphs/2026.pkl --labels data/labels/2026.txt \
        --output runs/erisk2026/seed_3407/eval
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config.schema import Config
from data_processing.dataset import load_timelines
from data_processing.labels import load_labels
from evaluation.metrics import REPORT_METRICS
from evaluation.runner import evaluate_model
from infer.loaders import load_psytcg
from utils.common import get_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Config the checkpoint was trained with."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--timelines", type=Path, default=None, help="Defaults to the config's test timelines."
    )
    parser.add_argument(
        "--labels", type=Path, default=None, help="Defaults to the config's test labels."
    )
    parser.add_argument(
        "--threshold", type=float, default=None, help="Defaults to the config's threshold."
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Directory for metrics and predictions."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    device = get_device()

    timelines = load_timelines(args.timelines or config.data.test_timelines)
    labels = load_labels(args.labels or config.data.test_labels)
    threshold = args.threshold if args.threshold is not None else config.train.decision_threshold

    model = load_psytcg(config, args.checkpoint, device)

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
        desc="eval",
    )

    print(f"{len(user_ids)} users (pos={sum(labels.values())}) | threshold {threshold}")
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
