#!/usr/bin/env python3
"""Train PsyTCG and evaluate the validation-selected checkpoint on test.

Protocol, per seed:

1. split the training collection at user level into train / validation;
2. train on training users and score validation after every epoch;
3. retain the epoch with the best validation ``selection_metric``;
4. evaluate that checkpoint on the test collection exactly once.

Example::

    python -m training.train_psytcg --config config/psytcg_erisk2026.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config.schema import Config
from model.psytcg import build_model
from training.experiment import run_experiment
from utils.common import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Override config seeds.")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    if args.seeds:
        config.train.seeds = args.seeds
    if args.epochs:
        config.train.epochs = args.epochs
    if args.output_dir:
        config.train.output_dir = args.output_dir

    device = get_device()
    run_experiment(
        lambda: build_model(config.model, device),
        config,
        device,
        extra_metadata={"architecture": "PsyTCG"},
    )


if __name__ == "__main__":
    main()
