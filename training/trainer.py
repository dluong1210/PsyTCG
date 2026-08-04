"""Training loop.

Model selection is validation-only: after every epoch the model is scored on the
held-out validation users and the best state is kept. The test collection is not
read here; the shared experiment driver evaluates it once after training.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config.schema import TrainConfig
from data_processing.dataset import UserTimeline, WindowDataset, collate_windows
from evaluation.metrics import is_better
from evaluation.runner import evaluate_model
from model.psytcg import pool_conv_logits


def build_optimizer(
    model: nn.Module, config: TrainConfig, steps_per_epoch: int, epochs: int
) -> Tuple[optim.Optimizer, optim.lr_scheduler.LambdaLR]:
    """AdamW with a lower-or-equal rate for the pretrained post encoder.

    The schedule is linear warmup followed by cosine decay over the full budget.
    """
    post_encoder = getattr(model, "post_encoder", None)
    pretrained = (
        [parameter for parameter in post_encoder.parameters() if parameter.requires_grad]
        if post_encoder is not None
        else []
    )
    pretrained_ids = {id(parameter) for parameter in pretrained}
    fresh = [p for p in model.parameters() if p.requires_grad and id(p) not in pretrained_ids]
    groups = []
    if fresh:
        groups.append({"params": fresh, "lr": config.lr})
    if pretrained:
        groups.append({"params": pretrained, "lr": config.lr_post_encoder})
    if not groups:
        raise ValueError("model has no trainable parameters")
    optimizer = optim.AdamW(groups, weight_decay=config.weight_decay)

    total_steps = max(1, epochs * max(1, math.ceil(steps_per_epoch / config.accumulation_steps)))
    warmup_steps = int(config.warmup_ratio * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optimizer, optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LambdaLR,
    criterion: nn.Module,
    config: TrainConfig,
    device: torch.device,
) -> float:
    """One pass over the training windows.

    The loss is applied to the last round of each window: that is the point at
    which the user-level label is known to hold given the observed history.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, n_windows, batch_index = 0.0, 0, -1
    progress = tqdm(loader, desc="train", leave=False)

    for batch_index, batch in enumerate(progress):
        for window in batch:
            window = window.to(device)
            target = torch.tensor([float(window.label)], device=device)
            conv_logits = model(
                window,
                post_batch_size=config.post_batch_size,
                only_last_conversation=True,
            )
            if not conv_logits:
                continue
            pooled = pool_conv_logits(conv_logits, "max")
            rounds = [c.conv_id for c in window.conversations if c.conv_id in pooled]
            if not rounds:
                continue
            loss = criterion(pooled[rounds[-1]].view(1), target)
            (loss / config.accumulation_steps).backward()
            total_loss += loss.item()
            n_windows += 1

        if (batch_index + 1) % config.accumulation_steps == 0:
            _step(model, optimizer, scheduler, config)
        progress.set_postfix(loss=total_loss / max(n_windows, 1))

    if batch_index >= 0 and (batch_index + 1) % config.accumulation_steps != 0:
        _step(model, optimizer, scheduler, config)
    if n_windows == 0:
        raise RuntimeError("no training window produced a logit")
    return total_loss / n_windows


def _step(model, optimizer, scheduler, config: TrainConfig) -> None:
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def fit(
    model: nn.Module,
    windows: Sequence[UserTimeline],
    val_user_ids: Sequence[str],
    val_timelines: Dict[str, List[Dict[str, Any]]],
    val_labels: Dict[str, int],
    config: TrainConfig,
    device: torch.device,
    seed: int,
    log: bool = True,
) -> Dict[str, Any]:
    """Train and return the state that scored best on validation.

    Returns the best state dict, the epoch it came from, its validation metrics,
    and the per-epoch history.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        WindowDataset(windows),
        batch_size=1,
        shuffle=True,
        collate_fn=collate_windows,
        generator=generator,
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer, scheduler = build_optimizer(model, config, len(loader), config.epochs)

    best_score: Optional[float] = None
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch: Optional[int] = None
    best_metrics: Optional[Dict[str, float]] = None
    stale = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, loader, optimizer, scheduler, criterion, config, device)
        val_metrics, _ = evaluate_model(
            model,
            val_user_ids,
            val_timelines,
            val_labels,
            threshold=config.decision_threshold,
            post_batch_size=config.post_batch_size,
            device=device,
            desc=f"val e{epoch}",
        )
        score = val_metrics[config.selection_metric]
        improved = is_better(score, best_score, config.selection_metric)
        if improved:
            best_score = float(score)
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        history.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "is_best": int(improved),
                **{f"val_{name}": value for name, value in val_metrics.items()},
            }
        )
        if log:
            print(
                f"  epoch {epoch:02d} | loss {train_loss:.4f} | "
                f"val P {val_metrics['precision']:.4f} | "
                f"val R {val_metrics['recall']:.4f} | "
                f"val F1 {val_metrics['f1']:.4f}"
                + ("  <- best" if improved else f"  (stale {stale})")
            )
        if config.early_stop_patience and stale >= config.early_stop_patience:
            if log:
                print(f"  early stop at epoch {epoch} (best epoch {best_epoch})")
            break

    if best_state is None:
        raise RuntimeError("no epoch produced a finite validation score")
    return {
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_val_metrics": best_metrics,
        "history": pd.DataFrame(history),
    }
