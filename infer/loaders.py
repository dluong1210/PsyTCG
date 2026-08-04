"""Checkpoint and user-list loading shared by inference and evaluation CLIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from config.schema import Config
from model.gat_tlstm import GATTLSTM
from model.psytcg import build_model


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(path)
    state: Any = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for wrapper in ("model_state_dict", "state_dict", "model", "weights"):
            if isinstance(state.get(wrapper), dict):
                state = state[wrapper]
                break
    if not isinstance(state, dict) or not state:
        raise TypeError(f"{path} does not contain a model state dict")
    return state


def load_psytcg(config: Config, checkpoint: Path, device: torch.device):
    model = build_model(config.model, device)
    model.load_state_dict(load_checkpoint(checkpoint), strict=True)
    return model.to(device).eval()


def load_gat_tlstm(checkpoint: Path, device: torch.device):
    # The final training checkpoint contains the complete post encoder, the GAT
    # trained from scratch, and the temporal classifier.
    model = GATTLSTM(None, load_post_checkpoint=False).to(device)
    model.load_state_dict(load_checkpoint(checkpoint), strict=True)
    return model.eval()


def load_user_ids(path: Path | None, available_ids) -> list[str]:
    if path is None:
        return sorted(str(user_id) for user_id in available_ids)
    with path.open(encoding="utf-8") as handle:
        identifiers = [line.strip().split()[0] for line in handle if line.strip()]
    return list(dict.fromkeys(identifiers))
