#!/usr/bin/env python3
"""Run GAT--TLSTM inference and write per-round predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

from config.schema import Config
from data_processing.dataset import load_timelines
from infer.loaders import load_gat_tlstm, load_user_ids
from infer.runner import predict_run
from utils.common import get_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--timelines", type=Path, default=None)
    parser.add_argument(
        "--user-ids", type=Path, default=None, help="Optional one-ID-per-line file."
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    timelines = load_timelines(args.timelines or config.data.test_timelines)
    user_ids = load_user_ids(args.user_ids, timelines)
    threshold = args.threshold if args.threshold is not None else config.train.decision_threshold
    device = get_device()
    model = load_gat_tlstm(args.checkpoint, device)
    run, rows = predict_run(
        model,
        user_ids,
        timelines,
        threshold=threshold,
        post_batch_size=config.train.post_batch_size,
        device=device,
        collect_rows=True,
        desc="infer GAT-TLSTM",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output / "predictions.csv", index=False)
    write_json(args.output / "run.json", run)
    print(f"{len(user_ids)} users -> {args.output}")


if __name__ == "__main__":
    main()
