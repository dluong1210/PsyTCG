"""Round-by-round inference over full timelines."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from tqdm.auto import tqdm

from data_processing.dataset import assemble
from model.psytcg import pool_conv_logits

Prediction = Tuple[str, int, float]
Run = Dict[str, Sequence[Prediction]]

#: A user without a usable timeline emits a negative decision at every round,
#: which is how the official evaluation treats a system that never reports.
NO_DECISION: List[Tuple[str, int, float]] = [("", 0, 0.0)]


@torch.no_grad()
def predict_run(
    model,
    user_ids: Sequence[str],
    timelines: Dict[str, List[Dict[str, Any]]],
    threshold: float = 0.5,
    post_batch_size: int = 128,
    pool_mode: str = "max",
    device: Optional[torch.device] = None,
    collect_rows: bool = False,
    desc: str = "infer",
    free_every: int = 25,
) -> Run | Tuple[Run, pd.DataFrame]:
    """Score every round of every user.

    Timelines are assembled one user at a time and released immediately after the
    forward pass; a whole collection of pre-assembled timelines does not fit in
    GPU memory.
    """
    model.eval()
    device = device or next(model.parameters()).device
    run: Run = {}
    rows: List[Dict[str, Any]] = []

    for position, user_id in enumerate(tqdm(user_ids, desc=desc, leave=False)):
        timeline = timelines.get(user_id)
        if not timeline:
            run[user_id] = list(NO_DECISION)
            continue

        user = assemble(timeline, 0, user_id).to(device)
        conv_logits = model(user, post_batch_size=post_batch_size, only_last_conversation=False)
        if not conv_logits:
            run[user_id] = list(NO_DECISION)
        else:
            pooled = pool_conv_logits(conv_logits, pool_mode)
            # Rounds are 0-indexed by conv_id; rounds with no target post keep the
            # default negative decision so the run stays aligned with the server.
            n_rounds = max(int(conv_id) for conv_id in pooled) + 1
            predictions: List[Tuple[str, int, float]] = [("", 0, 0.0)] * n_rounds
            for conv_id, logit in pooled.items():
                score = torch.sigmoid(logit).item()
                decision = int(score > threshold)
                predictions[int(conv_id)] = (str(conv_id), decision, score)
                if collect_rows:
                    rows.append(
                        {
                            "user_id": user_id,
                            "conv_id": int(conv_id),
                            "score": score,
                            "decision": decision,
                        }
                    )
            run[user_id] = predictions

        del user, conv_logits
        if torch.cuda.is_available() and (position + 1) % free_every == 0:
            torch.cuda.empty_cache()

    return (run, pd.DataFrame(rows)) if collect_rows else run
