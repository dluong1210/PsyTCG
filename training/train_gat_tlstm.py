#!/usr/bin/env python3
"""Train and evaluate the controlled GAT--TLSTM baseline.

The data split, windowing, validation-only checkpoint selection, metrics, and
artifacts are identical to PsyTCG training. Only the model factory changes.

Example::

    python -m training.train_gat_tlstm --config config/gat_tlstm_erisk2026.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from config.schema import Config, GATTLSTMConfig
from model.gat_tlstm import build_gat_tlstm
from training.experiment import run_experiment
from utils.common import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    with args.config.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if "baseline" not in raw:
        raise SystemExit(f"{args.config} has no 'baseline' section")
    baseline = GATTLSTMConfig(**raw["baseline"])

    if args.seeds:
        config.train.seeds = args.seeds
    if args.epochs:
        config.train.epochs = args.epochs
    if args.output_dir:
        config.train.output_dir = args.output_dir

    device = get_device()
    run_experiment(
        lambda: build_gat_tlstm(baseline, device),
        config,
        device,
        extra_metadata={
            "architecture": "GAT-TLSTM",
            "baseline": {
                "post_encoder_checkpoint": str(baseline.post_encoder_checkpoint),
                "dropout": baseline.dropout,
                "freeze_post_encoder": baseline.freeze_post_encoder,
                "gat_initialization": "random",
            },
        },
    )


if __name__ == "__main__":
    main()
