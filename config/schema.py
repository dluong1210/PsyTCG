"""Experiment configuration.

A run is fully described by one YAML file: which timelines to train and test on,
the model architecture, and the optimisation budget. See ``config/`` for the
settings reported in the paper.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    """Inputs and the user-level train/validation split.

    ``train_timelines`` and ``test_timelines`` are pickles produced by
    ``data_processing.build_graph_pkl``. The test collection is touched exactly
    once after training by the shared experiment runner.
    """

    train_timelines: Path
    test_timelines: Path
    train_labels: Path
    test_labels: Path
    val_fraction: float = 0.1
    split_seed: int = 42
    max_conversations: int = 12
    window_stride: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("train_timelines", "test_timelines", "train_labels", "test_labels"):
            setattr(self, name, Path(getattr(self, name)))
        if self.window_stride is None:
            self.window_stride = self.max_conversations
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("val_fraction must be between 0 and 1")
        if self.max_conversations < 1:
            raise ValueError("max_conversations must be positive")
        if self.window_stride < 1:
            raise ValueError("window_stride must be positive")


@dataclass
class ModelConfig:
    """PsyTCG architecture."""

    post_input_dim: int = 1024
    post_hidden_dim: int = 1024
    memory_dim: int = 256
    other_rnn: str = "gru"
    target_rnn: str = "t-lstm"
    dropout: float = 0.2
    gat_heads: int = 8
    gat_time_dim: int = 8
    use_time_attention: bool = False
    use_context_gate: bool = True
    other_memory_scope: str = "persistent"
    child_memory_mode: str = "post"
    post_encoder_checkpoint: Optional[Path] = None
    freeze_post_encoder: bool = False

    def __post_init__(self) -> None:
        if self.post_encoder_checkpoint is not None:
            self.post_encoder_checkpoint = Path(self.post_encoder_checkpoint)
        dimensions = {
            "post_input_dim": self.post_input_dim,
            "post_hidden_dim": self.post_hidden_dim,
            "memory_dim": self.memory_dim,
            "gat_heads": self.gat_heads,
        }
        if any(value < 1 for value in dimensions.values()):
            raise ValueError(f"model dimensions must be positive: {dimensions}")
        if self.memory_dim % self.gat_heads:
            raise ValueError("memory_dim must be divisible by gat_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.other_rnn.lower() not in {"gru", "lstm", "t-lstm", "timelstm"}:
            raise ValueError(f"unsupported other_rnn: {self.other_rnn}")
        if self.target_rnn.lower() not in {"gru", "lstm", "t-lstm", "timelstm"}:
            raise ValueError(f"unsupported target_rnn: {self.target_rnn}")
        if self.other_memory_scope not in {"conversation", "persistent"}:
            raise ValueError(f"unsupported other_memory_scope: {self.other_memory_scope}")
        if self.child_memory_mode not in {"pre", "post"}:
            raise ValueError(f"unsupported child_memory_mode: {self.child_memory_mode}")


@dataclass
class GATTLSTMConfig:
    """Controlled GAT--TLSTM architecture.

    Only the common post encoder is initialized from a checkpoint. The GATv2
    layer and all user-level temporal layers are initialized from scratch.
    """

    post_encoder_checkpoint: Path = Path("checkpoints/post_encoder.pt")
    dropout: float = 0.2
    freeze_post_encoder: bool = False

    def __post_init__(self) -> None:
        self.post_encoder_checkpoint = Path(self.post_encoder_checkpoint)
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass
class TrainConfig:
    """Optimisation budget and model selection.

    Epoch selection uses ``selection_metric`` on the validation split only; the
    test collection never participates in selection.
    """

    output_dir: Path = Path("runs/psytcg")
    epochs: int = 25
    seeds: List[int] = field(default_factory=lambda: [3407])
    lr: float = 3e-5
    lr_post_encoder: float = 3e-5
    weight_decay: float = 1e-3
    warmup_ratio: float = 0.1
    accumulation_steps: int = 2
    grad_clip_norm: float = 1.0
    post_batch_size: int = 128
    early_stop_patience: Optional[int] = 8
    selection_metric: str = "f1"
    decision_threshold: float = 0.5

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.seeds = [int(s) for s in self.seeds]
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if self.lr <= 0 or self.lr_post_encoder <= 0:
            raise ValueError("learning rates must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError("warmup_ratio must be in [0, 1]")
        if self.accumulation_steps < 1 or self.post_batch_size < 1:
            raise ValueError("batch and accumulation sizes must be positive")
        if self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive")
        if self.early_stop_patience is not None and self.early_stop_patience < 1:
            raise ValueError("early_stop_patience must be positive or null")
        if self.selection_metric not in {"precision", "recall", "f1"}:
            raise ValueError("selection_metric must be precision, recall, or f1")
        if not 0.0 <= self.decision_threshold <= 1.0:
            raise ValueError("decision_threshold must be in [0, 1]")


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    name: str = "psytcg"

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Config":
        with Path(path).open(encoding="utf-8") as handle:
            raw: Dict[str, Any] = yaml.safe_load(handle) or {}
        return cls(
            name=raw.get("name", "psytcg"),
            data=DataConfig(**raw["data"]),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        def plain(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {k: plain(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [plain(v) for v in value]
            return value

        return plain(asdict(self))
