"""Shared experiment driver.

Every model in this repository is trained under the same protocol, so the split,
the windowing, the seed loop and the artifacts live here rather than in each
training script. A script only supplies a ``model_factory``.

Protocol, per seed:

1. split the training collection at user level into train / validation;
2. train on the training users, scoring **validation only** after each epoch;
3. keep the epoch with the best validation ``selection_metric``;
4. evaluate that checkpoint on the test collection exactly once.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from config.schema import Config
from data_processing.dataset import load_timelines, make_windows
from data_processing.labels import load_labels
from evaluation.metrics import REPORT_METRICS
from evaluation.runner import evaluate_model
from training.trainer import fit
from utils.common import free_cuda, seed_everything, write_json


def split_users(
    user_ids: List[str], labels: Dict[str, int], fraction: float, seed: int
) -> tuple[List[str], List[str]]:
    """Stratified user-level split, fixed across seeds so only training varies."""
    stratify = [labels[user_id] for user_id in user_ids]
    train_ids, val_ids = train_test_split(
        user_ids, test_size=fraction, random_state=seed, stratify=stratify
    )
    train_ids, val_ids = sorted(train_ids), sorted(val_ids)
    assert not set(train_ids) & set(val_ids), "user leaked between train and validation"
    return train_ids, val_ids


def aggregate(frame: pd.DataFrame, prefix: str) -> Dict[str, Dict[str, float]]:
    """Mean and sample standard deviation per classification metric."""
    summary = {}
    for metric in REPORT_METRICS:
        column = f"{prefix}_{metric}"
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        finite = values[np.isfinite(values)]
        summary[metric] = {
            "mean": float(finite.mean()) if len(finite) else float("nan"),
            "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
            "n": int(len(finite)),
        }
    return summary


def run_experiment(
    model_factory: Callable[[], nn.Module],
    config: Config,
    device: torch.device,
    extra_metadata: Dict | None = None,
) -> pd.DataFrame:
    """Train and evaluate ``model_factory()`` once per seed; write all artifacts.

    ``model_factory`` must return a fresh model on the target device whose
    ``forward(user, post_batch_size=..., only_last_conversation=...)`` returns
    ``{conv_id: logits}``.
    """
    output_dir = config.train.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = config.to_dict()
    if extra_metadata:
        resolved_config["release_metadata"] = extra_metadata
    write_json(output_dir / "config.json", resolved_config)
    print(f"device={device} | output={output_dir}")

    train_timelines = load_timelines(config.data.train_timelines)
    test_timelines = load_timelines(config.data.test_timelines)
    train_labels = load_labels(config.data.train_labels)
    test_labels = load_labels(config.data.test_labels)

    labelled = sorted(uid for uid in train_timelines if uid in train_labels)
    if len({train_labels[uid] for uid in labelled}) < 2:
        raise SystemExit("training pool must contain both classes")
    train_ids, val_ids = split_users(
        labelled, train_labels, config.data.val_fraction, config.data.split_seed
    )

    # The test collection is scored over its full golden truth: users whose
    # timeline was pruned stay in and count as missed detections.
    test_ids = sorted(test_labels)
    n_pruned = sum(1 for uid in test_ids if uid not in test_timelines)
    print(
        f"train users {len(train_ids)} (pos={sum(train_labels[u] for u in train_ids)}) | "
        f"val users {len(val_ids)} (pos={sum(train_labels[u] for u in val_ids)})\n"
        f"test users {len(test_ids)} (pos={sum(test_labels.values())}), "
        f"{n_pruned} without a timeline"
    )

    windows = []
    for user_id in tqdm(train_ids, desc="windows"):
        windows.extend(
            make_windows(
                train_timelines[user_id],
                train_labels[user_id],
                user_id,
                max_conversations=config.data.max_conversations,
                stride=config.data.window_stride,
            )
        )
    print(f"{len(windows)} training windows (pos={sum(1 for w in windows if w.label == 1)})")

    val_labels = {uid: train_labels[uid] for uid in val_ids}
    rows, histories = [], []

    for index, seed in enumerate(config.train.seeds, start=1):
        print(f"\n=== seed {seed} ({index}/{len(config.train.seeds)}) ===")
        seed_everything(seed)
        model = model_factory()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  trainable parameters: {trainable:,}")

        result = fit(
            model,
            windows,
            val_ids,
            train_timelines,
            val_labels,
            config.train,
            device,
            seed=seed,
        )
        histories.append(result["history"])

        # Restore the validation-selected epoch, then touch the test set once.
        model.load_state_dict(result["best_state"], strict=True)
        model.to(device).eval()
        val_metrics, _ = evaluate_model(
            model,
            val_ids,
            train_timelines,
            val_labels,
            threshold=config.train.decision_threshold,
            post_batch_size=config.train.post_batch_size,
            device=device,
            desc="val (best)",
        )
        test_metrics, _, test_rows = evaluate_model(
            model,
            test_ids,
            test_timelines,
            test_labels,
            threshold=config.train.decision_threshold,
            post_batch_size=config.train.post_batch_size,
            device=device,
            collect_rows=True,
            desc="test",
        )

        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        torch.save(result["best_state"], seed_dir / "checkpoint.pt")
        result["history"].to_csv(seed_dir / "history.csv", index=False)
        test_rows.to_csv(seed_dir / "test_predictions.csv", index=False)
        write_json(
            seed_dir / "metrics.json",
            {
                "seed": seed,
                "best_epoch": result["best_epoch"],
                "trainable_parameters": trainable,
                "selection": {"split": "validation", "metric": config.train.selection_metric},
                "validation": val_metrics,
                "test": test_metrics,
            },
        )
        print(
            f"  best epoch {result['best_epoch']} | "
            f"val F1 {val_metrics['f1']:.4f} | "
            f"test P {test_metrics['precision']:.4f} | "
            f"test R {test_metrics['recall']:.4f} | "
            f"test F1 {test_metrics['f1']:.4f}"
        )

        rows.append(
            {
                "seed": seed,
                "best_epoch": result["best_epoch"],
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
        )
        del model, result
        free_cuda()

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(output_dir / "results_per_seed.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(
        output_dir / "history_all_seeds.csv", index=False
    )
    summary = {"validation": aggregate(per_seed, "val"), "test": aggregate(per_seed, "test")}
    write_json(
        output_dir / "results_summary.json",
        {
            "name": config.name,
            "seeds": config.train.seeds,
            "selection": {"split": "validation", "metric": config.train.selection_metric},
            "decision_threshold": config.train.decision_threshold,
            **(extra_metadata or {}),
            "summary": summary,
            "per_seed": per_seed.to_dict(orient="records"),
        },
    )

    print(f"\n=== results over {len(config.train.seeds)} seed(s), mean ± std ===")
    for split in ("validation", "test"):
        print(f"  [{split}]")
        for metric in REPORT_METRICS:
            if metric in summary[split]:
                stats = summary[split][metric]
                print(f"    {metric:>10}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"\nartifacts -> {output_dir}")
    return per_seed
