"""Static conversation GAT followed by a user-level Time-LSTM.

This is the controlled GAT--TLSTM baseline from the paper. Each retained
conversation is encoded independently with one GATv2 layer over its reply graph.
Target-node representations are then consumed chronologically by a Time-LSTM.
Unlike PsyTCG, the baseline does not keep author-specific interlocutor memories
or feed accumulated neighbour state into graph attention.

The common post encoder may be initialized from its evidence-encoder checkpoint,
but the GATv2 layer is always initialized from scratch and trained for this task.

``torch-geometric`` is imported lazily so installing the core PsyTCG package
does not require the baseline dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from config.schema import GATTLSTMConfig
from data_processing.dataset import Conversation, UserTimeline
from model.components import PostEncoder

NS_PER_DAY = 24 * 3600 * 10**9


def _state_dict(path: Path | str) -> Dict[str, torch.Tensor]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        for wrapper in ("state_dict", "model_state_dict", "model", "weights"):
            if isinstance(raw.get(wrapper), dict):
                raw = raw[wrapper]
                break
    if not isinstance(raw, dict):
        raise TypeError(f"{path} does not contain a state dict")
    state = {}
    for key, value in raw.items():
        if not isinstance(value, torch.Tensor):
            continue
        clean = str(key)
        while clean.startswith("module."):
            clean = clean.removeprefix("module.")
        state[clean] = value
    if not state:
        raise ValueError(f"{path} contains no tensor weights")
    return state


def _state_for_module(
    state: Dict[str, torch.Tensor],
    module: nn.Module,
    prefixes: Sequence[str] = (),
) -> Dict[str, torch.Tensor]:
    expected = module.state_dict()
    expected_keys = set(expected)
    for prefix in ("", *prefixes):
        candidate = {
            (key.removeprefix(prefix) if prefix else key): value
            for key, value in state.items()
            if not prefix or key.startswith(prefix)
        }
        if set(candidate) == expected_keys and all(
            candidate[key].shape == expected[key].shape for key in expected
        ):
            return candidate

    matched = {}
    for expected_key, expected_value in expected.items():
        hits = [
            value
            for key, value in state.items()
            if (key == expected_key or key.endswith(f".{expected_key}"))
            and value.shape == expected_value.shape
        ]
        if len(hits) == 1:
            matched[expected_key] = hits[0]
    if set(matched) != expected_keys:
        missing = sorted(expected_keys - set(matched))
        raise RuntimeError(
            f"checkpoint cannot be matched strictly; missing or ambiguous keys: {missing[:20]}"
        )
    return matched


def _load_module(module: nn.Module, path: Path | str, prefixes: Sequence[str] = ()) -> None:
    module.load_state_dict(_state_for_module(_state_dict(path), module, prefixes), strict=True)


class BaselineTimeLSTM(nn.Module):
    """Time-LSTM used by the static-graph baseline.

    Elapsed time is measured in days. A carried state honours the first gap of
    the resumed sequence, which is required when conversations are streamed one
    at a time.
    """

    def __init__(self, input_size: int = 1024, hidden_size: int = 1024, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)
        for gate in ("i", "f", "o", "c"):
            setattr(self, f"W{gate}", nn.Linear(input_size, hidden_size, bias=False))
            setattr(self, f"U{gate}", nn.Linear(hidden_size, hidden_size, bias=False))
            setattr(self, f"b{gate}", nn.Parameter(torch.zeros(hidden_size)))
        self.W_decomp = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_decomp = nn.Parameter(torch.zeros(hidden_size))
        self.linear_time = nn.Linear(1, 1)

    def forward(self, inputs, timestamps, state=None):
        batch, steps, _ = inputs.shape
        carried = state is not None
        if carried:
            hidden, cell = state
        else:
            hidden = inputs.new_zeros(batch, self.hidden_size)
            cell = inputs.new_zeros(batch, self.hidden_size)

        outputs = []
        for step in range(steps):
            current = inputs[:, step]
            if self.training and self.dropout_p > 0:
                current = self.dropout(current)
            elapsed = (
                inputs.new_zeros(batch, 1)
                if step == 0 and not carried
                else timestamps[:, step].reshape(batch, 1).float()
            )
            short_term = torch.tanh(self.W_decomp(cell) + self.b_decomp)
            decay = torch.sigmoid(self.linear_time(elapsed)).expand(-1, self.hidden_size)
            adjusted = cell - short_term + short_term * decay

            input_gate = torch.sigmoid(self.Wi(current) + self.Ui(hidden) + self.bi)
            forget_gate = torch.sigmoid(self.Wf(current) + self.Uf(hidden) + self.bf)
            output_gate = torch.sigmoid(self.Wo(current) + self.Uo(hidden) + self.bo)
            candidate = torch.tanh(self.Wc(current) + self.Uc(hidden) + self.bc)
            cell = forget_gate * adjusted + input_gate * candidate
            hidden = output_gate * torch.tanh(cell)
            outputs.append(hidden)
        return torch.stack(outputs, dim=1), (hidden, cell)


class GATPostEncoder(nn.Module):
    """A newly initialized GATv2 layer over a single conversation graph."""

    def __init__(
        self,
        post_checkpoint: Optional[Path | str],
        dropout: float = 0.2,
        load_post_checkpoint: bool = True,
    ):
        super().__init__()
        try:
            from torch_geometric.nn import GATv2Conv
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise ImportError(
                "GAT--TLSTM requires torch-geometric; install `psytcg[gat-baseline]`"
            ) from exc

        self.post_level_model = PostEncoder(1024, 1024, dropout=0.1)
        if load_post_checkpoint:
            if post_checkpoint is None:
                raise ValueError("a post-encoder checkpoint is required for baseline training")
            _load_module(
                self.post_level_model,
                post_checkpoint,
                ("post_level_model.", "gat_level_model.post_level_model."),
            )

        self.linear = nn.Linear(1024, 1024)
        self.LN = nn.ModuleList([nn.LayerNorm(1024), nn.LayerNorm(1024)])
        self.gat_layers = nn.ModuleList(
            [GATv2Conv(1024, 128, heads=8, dropout=dropout, concat=True)]
        )

    def encode_nodes(
        self,
        rel_hidden: List[torch.Tensor],
        emo_sev: List[torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        embeddings = []
        for start in range(0, len(rel_hidden), batch_size):
            rel_batch = rel_hidden[start : start + batch_size]
            emo_batch = emo_sev[start : start + batch_size]
            lengths = torch.tensor([item.size(0) for item in rel_batch], device=rel_batch[0].device)
            _, encoded = self.post_level_model(
                (
                    rnn_utils.pad_sequence(rel_batch, batch_first=True),
                    rnn_utils.pad_sequence(emo_batch, batch_first=True),
                ),
                lengths,
            )
            embeddings.append(encoded)
        return torch.cat(embeddings, dim=0)

    def forward(self, conversation: Conversation, post_batch_size: int):
        events = conversation.events
        node_vectors = self.encode_nodes(
            [event.rel_hidden for event in events],
            [event.emo_sev for event in events],
            post_batch_size,
        )
        post_vectors = self.LN[0](self.linear(node_vectors))
        edge_index = _edge_index(conversation, post_vectors.device)
        graph_vectors = post_vectors
        for index, layer in enumerate(self.gat_layers):
            graph_vectors = self.LN[index + 1](layer(graph_vectors, edge_index))
        target = torch.tensor(
            [event.is_target for event in events], dtype=torch.bool, device=post_vectors.device
        )
        return (
            graph_vectors[target],
            post_vectors[target],
            [event.ts_raw for event in events if event.is_target],
        )


def _edge_index(conversation: Conversation, device: torch.device) -> torch.Tensor:
    events = conversation.events
    identifiers = {
        str(event.node_id): index for index, event in enumerate(events) if event.node_id is not None
    }
    edges = set()
    for child, event in enumerate(events):
        if event.parent is None:
            continue
        parent = identifiers.get(str(event.parent))
        if parent is not None and parent != child:
            edges.update(((child, parent), (parent, child)))
    for target, event in enumerate(events):
        for neighbour in event.local_neighbour_indices or []:
            if 0 <= neighbour < len(events) and neighbour != target:
                edges.update(((neighbour, target), (target, neighbour)))

    roots = [index for index, event in enumerate(events) if event.parent is None]
    if roots and len(events) > 1:
        root = roots[0]
        adjacency = {index: set() for index in range(len(events))}
        for source, destination in edges:
            adjacency[source].add(destination)
            adjacency[destination].add(source)
        visited = set()
        components = []
        for start in range(len(events)):
            if start in visited:
                continue
            stack, component = [start], set()
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                stack.extend(adjacency[node] - visited)
            components.append(component)
        for component in components:
            if root not in component:
                anchor = min(component)
                edges.update(((anchor, root), (root, anchor)))

    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.tensor(sorted(edges), dtype=torch.long, device=device).t().contiguous()


class GATTLSTM(nn.Module):
    """Fresh per-conversation GATv2 followed by chronological Time-LSTM."""

    def __init__(
        self,
        post_encoder_checkpoint: Optional[Path | str],
        dropout: float = 0.2,
        freeze_post_encoder: bool = False,
        load_post_checkpoint: bool = True,
    ):
        super().__init__()
        self.gat_level_model = GATPostEncoder(
            post_encoder_checkpoint,
            dropout,
            load_post_checkpoint=load_post_checkpoint,
        )
        self.concat_linear = nn.Linear(2048, 1024)
        self.user_layer_norm = nn.LayerNorm(1024)
        self.user_model = BaselineTimeLSTM(1024, 1024, dropout)
        self.classifier_layer_norm = nn.LayerNorm(1024)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.freeze_post_encoder = freeze_post_encoder
        if freeze_post_encoder:
            self.post_encoder.requires_grad_(False)
            self.post_encoder.eval()

    @property
    def post_encoder(self) -> PostEncoder:
        """Expose the common optimizer grouping interface."""
        return self.gat_level_model.post_level_model

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_post_encoder:
            self.post_encoder.eval()
        return self

    def forward(
        self,
        user: UserTimeline,
        post_batch_size: int = 128,
        only_last_conversation: bool = False,
    ) -> Dict[Any, torch.Tensor]:
        target_conversations = [
            conversation
            for conversation in user.conversations
            if any(event.is_target for event in conversation.events)
        ]
        if not target_conversations:
            return {}
        last_target_conv = target_conversations[-1].conv_id
        state = None
        previous_ts: Optional[int] = None
        logits: Dict[Any, List[torch.Tensor]] = {}

        for conversation in user.conversations:
            if not any(event.is_target for event in conversation.events):
                continue
            graph_vectors, post_vectors, timestamps = self.gat_level_model(
                conversation, post_batch_size
            )
            seeds = self.user_layer_norm(
                self.concat_linear(torch.cat([graph_vectors, post_vectors], dim=-1))
            )
            conversation_logits = []
            for seed, timestamp in zip(seeds, timestamps, strict=True):
                gap = (
                    0.0
                    if previous_ts is None or timestamp is None
                    else max(0.0, (timestamp - previous_ts) / NS_PER_DAY)
                )
                elapsed = seed.new_tensor([[gap]])
                outputs, state = self.user_model(seed.view(1, 1, -1), elapsed, state)
                previous_ts = timestamp if timestamp is not None else previous_ts
                hidden = self.classifier_layer_norm(outputs[:, -1])
                conversation_logits.append(self.classifier(hidden).squeeze())
            if conversation_logits and (
                not only_last_conversation or conversation.conv_id == last_target_conv
            ):
                # Match the reference baseline: the last target node represents
                # a conversation in the user-level sequence.
                logits[conversation.conv_id] = torch.stack([conversation_logits[-1]])
        return logits

    def active_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def build_gat_tlstm(config: GATTLSTMConfig, device: torch.device) -> GATTLSTM:
    return GATTLSTM(
        config.post_encoder_checkpoint,
        dropout=config.dropout,
        freeze_post_encoder=config.freeze_post_encoder,
    ).to(device)
