"""PsyTCG: role-asymmetric temporal conversation graph model.

One forward pass walks a user's timeline once, in event order, and maintains one
memory per author:

* an **interlocutor** memory, updated by a plain recurrent step whenever that
  person posts (role-symmetric information, no attention);
* the **target** memory, updated by a time-aware recurrent step whose input is
  the target post embedding concatenated with the gated context aggregated from
  the interlocutors who replied to it.

The asymmetry is deliberate: only the target's memory is time-decayed and
context-fused, because only the target's psychological state is being tracked.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from data_processing.dataset import UserTimeline
from model.components import MemoryAttention, PostEncoder, TimeLSTM, build_rnn

NS_PER_DAY = 24 * 3600 * 10**9


class PsyTCG(nn.Module):
    def __init__(
        self,
        post_input_dim: int = 1024,
        post_hidden_dim: int = 1024,
        memory_dim: int = 256,
        other_rnn: str = "gru",
        target_rnn: str = "t-lstm",
        dropout: float = 0.2,
        gat_heads: int = 8,
        gat_time_dim: int = 8,
        use_time_attention: bool = False,
        use_context_gate: bool = True,
        other_memory_scope: str = "persistent",
        child_memory_mode: str = "post",
        post_encoder_checkpoint: Optional[str] = None,
        freeze_post_encoder: bool = False,
    ):
        super().__init__()
        if other_memory_scope not in ("conversation", "persistent"):
            raise ValueError(other_memory_scope)
        if child_memory_mode not in ("pre", "post"):
            raise ValueError(child_memory_mode)

        self.post_encoder = PostEncoder(post_input_dim, post_hidden_dim, dropout=dropout)
        if post_encoder_checkpoint:
            checkpoint_path = Path(post_encoder_checkpoint)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"post encoder checkpoint not found: {checkpoint_path}")
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.post_encoder.load_state_dict(state, strict=True)
        if freeze_post_encoder:
            self.post_encoder.requires_grad_(False)
            self.post_encoder.eval()

        self.post_project = nn.Linear(post_hidden_dim, memory_dim)
        self.attention = MemoryAttention(
            embed_dim=memory_dim,
            memory_dim=memory_dim,
            num_heads=gat_heads,
            time_dim=gat_time_dim,
            use_time=use_time_attention,
        )
        self.use_context_gate = use_context_gate
        if use_context_gate:
            # Gate reads the post, the aggregated context, and the current target
            # memory, so context can be suppressed when it adds nothing.
            self.context_gate = nn.Linear(memory_dim * 3, memory_dim)

        self.other_user_rnn = build_rnn(other_rnn, memory_dim, memory_dim, dropout)
        self.target_user_rnn = build_rnn(target_rnn, memory_dim * 2, memory_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(memory_dim, 256),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        self.memory_dim = memory_dim
        self.other_memory_scope = other_memory_scope
        self.child_memory_mode = child_memory_mode
        self.freeze_post_encoder = freeze_post_encoder

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_post_encoder:
            self.post_encoder.eval()
        return self

    # ---- one recurrent step, used for both roles ----
    def _rnn_step(self, rnn: nn.Module, x: torch.Tensor, state=None, delta_t=None):
        x = x.unsqueeze(0).unsqueeze(0)
        if isinstance(rnn, TimeLSTM):
            gap = delta_t if delta_t is not None else torch.zeros(1, device=x.device)
            output, (hidden, cell) = rnn(x, gap.view(1, 1).to(x.device), state)
            return output[:, -1, :].squeeze(0), (hidden, cell)
        if isinstance(rnn, nn.LSTM):
            output, (hidden, cell) = rnn(x, state)
            return output[:, -1, :].squeeze(0), (hidden, cell)
        if isinstance(rnn, nn.GRU):
            output, hidden = rnn(x, state)
            return output[:, -1, :].squeeze(0), hidden
        raise TypeError(type(rnn))

    def _encode_posts(
        self, rel_hidden: List[torch.Tensor], emo_sev: List[torch.Tensor], batch_size: int
    ) -> torch.Tensor:
        embeddings = []
        for start in range(0, len(rel_hidden), batch_size):
            rel_batch = rel_hidden[start : start + batch_size]
            emo_batch = emo_sev[start : start + batch_size]
            lengths = torch.tensor([r.size(0) for r in rel_batch], device=rel_batch[0].device)
            _, embedding = self.post_encoder(
                (
                    rnn_utils.pad_sequence(rel_batch, batch_first=True),
                    rnn_utils.pad_sequence(emo_batch, batch_first=True),
                ),
                lengths,
            )
            embeddings.append(self.post_project(embedding))
        return torch.cat(embeddings, dim=0)

    def _step_interlocutor(
        self,
        index: int,
        post_embeddings: torch.Tensor,
        events: List[Any],
        event_conv_ids: List[Any],
        memories: Dict[str, torch.Tensor],
        states: Dict[str, Any],
        last_conv: Dict[str, Any],
    ) -> None:
        event = events[index]
        author = event.user_id
        if (
            self.other_memory_scope == "conversation"
            and last_conv.get(author) is not None
            and last_conv[author] != event_conv_ids[index]
        ):
            states[author] = None
            memories[author] = torch.zeros(self.memory_dim, device=post_embeddings.device)
        memory, state = self._rnn_step(
            self.other_user_rnn, post_embeddings[index], states[author], event.delta_t
        )
        memories[author] = memory
        states[author] = state
        last_conv[author] = event_conv_ids[index]

    def forward(
        self,
        user: UserTimeline,
        post_batch_size: int = 128,
        only_last_conversation: bool = False,
        return_details: bool = False,
    ):
        """Return ``{conv_id: logits}`` with one logit per target post of the round.

        Logits are left unpooled so the caller decides how a round's posts combine
        (see :func:`pool_conv_logits`). With ``only_last_conversation`` only the
        final round is scored, which is what training needs.
        """
        target_uid = user.target_user_id
        if not user.conversations:
            return ({}, {"gates": [], "attention": []}) if return_details else {}

        events: List[Any] = []
        event_conv_ids: List[Any] = []
        for conversation in user.conversations:
            for event in conversation.events:
                events.append(event)
                event_conv_ids.append(conversation.conv_id)
        if not events:
            return ({}, {"gates": [], "attention": []}) if return_details else {}

        device = events[0].rel_hidden.device
        post_embeddings = self._encode_posts(
            [event.rel_hidden for event in events],
            [event.emo_sev for event in events],
            post_batch_size,
        )

        # Elapsed time is per author: each memory decays by the gap since that
        # person last spoke, not since the previous event in the thread.
        last_seen: Dict[str, int] = {}
        for event in events:
            if event.ts_raw is None:
                event.delta_t = torch.zeros(1, device=device)
                continue
            previous = last_seen.get(event.user_id)
            gap = 0.0 if previous is None else max(0.0, (event.ts_raw - previous) / NS_PER_DAY)
            event.delta_t = torch.tensor([gap], dtype=torch.float32, device=device)
            last_seen[event.user_id] = event.ts_raw

        memories: Dict[str, torch.Tensor] = defaultdict(
            lambda: torch.zeros(self.memory_dim, device=device)
        )
        states: Dict[str, Any] = defaultdict(lambda: None)
        last_conv: Dict[str, Any] = {}
        details = {"gates": [], "attention": []} if return_details else None

        target_indices = [i for i, event in enumerate(events) if event.user_id == target_uid]
        target_conv = {i: event_conv_ids[i] for i in target_indices}

        def neighbours_of(index: int) -> List[int]:
            """Non-target nodes one reply hop from this target post, same round."""
            event = events[index]
            if not event.neighbour_indices:
                return []
            return [
                n
                for n in event.neighbour_indices
                if 0 <= n < len(events)
                and n != index
                and events[n].user_id != target_uid
                and event_conv_ids[n] == target_conv[index]
            ]

        # Segment the event stream: everything before each target post (in
        # chronological order) updates interlocutor memories first.
        segments: List[Tuple[List[int], List[int], Optional[int]]] = []
        previous = -1
        for index in target_indices:
            before = [i for i in range(previous + 1, index) if events[i].user_id != target_uid]
            segments.append((before, neighbours_of(index), index))
            previous = index
        trailing = [i for i in range(previous + 1, len(events)) if events[i].user_id != target_uid]
        if trailing:
            segments.append((trailing, [], None))

        # A context-only round can occur after the last target event in a custom
        # timeline. Training still supervises the terminal target update.
        last_target_conv_id = target_conv[target_indices[-1]] if target_indices else None
        conv_logits: Dict[Any, List[torch.Tensor]] = defaultdict(list)
        processed_interlocutors: set[int] = set()

        for before, children, target_index in segments:
            for index in sorted(set(before)):
                if index in processed_interlocutors:
                    continue
                self._step_interlocutor(
                    index, post_embeddings, events, event_conv_ids, memories, states, last_conv
                )
                processed_interlocutors.add(index)
            if target_index is None:
                continue

            event = events[target_index]
            target_embedding = post_embeddings[target_index]
            conv_id = event_conv_ids[target_index]

            # "post": replies that arrive after the target post still update their
            # author's memory before attention reads it, so the context reflects
            # the full exchange around that post.
            if self.child_memory_mode == "post":
                for index in sorted(set(children)):
                    if index not in processed_interlocutors:
                        self._step_interlocutor(
                            index,
                            post_embeddings,
                            events,
                            event_conv_ids,
                            memories,
                            states,
                            last_conv,
                        )
                        processed_interlocutors.add(index)

            if children:
                neighbour_embeddings = torch.stack([post_embeddings[n] for n in children])
                neighbour_memories = torch.stack([memories[events[n].user_id] for n in children])
                if event.ts_raw is not None:
                    gaps = torch.tensor(
                        [
                            [abs((event.ts_raw - (events[n].ts_raw or event.ts_raw)) / NS_PER_DAY)]
                            for n in children
                        ],
                        dtype=torch.float32,
                        device=device,
                    )
                else:
                    gaps = torch.zeros(len(children), 1, device=device)
                if return_details:
                    context, attention = self.attention(
                        target_embedding,
                        neighbour_embeddings,
                        neighbour_memories,
                        gaps,
                        return_attention=True,
                    )
                    details["attention"].append(
                        (conv_id, None if attention is None else attention.detach().cpu())
                    )
                else:
                    context = self.attention(
                        target_embedding, neighbour_embeddings, neighbour_memories, gaps
                    )
            else:
                context = torch.zeros(self.memory_dim, device=device)

            if self.use_context_gate:
                gate = torch.sigmoid(
                    self.context_gate(
                        torch.cat([target_embedding, context, memories[target_uid]], dim=-1)
                    )
                )
                context = gate * context
                if return_details:
                    details["gates"].append((conv_id, gate.mean().item()))

            memory, state = self._rnn_step(
                self.target_user_rnn,
                torch.cat([target_embedding, context], dim=-1),
                states[target_uid],
                event.delta_t,
            )
            states[target_uid] = state
            memories[target_uid] = memory

            if (not only_last_conversation) or conv_id == last_target_conv_id:
                conv_logits[conv_id].append(self.classifier(memory))

        logits = {
            conv_id: torch.stack(values, dim=0).squeeze(-1)
            for conv_id, values in conv_logits.items()
            if values
        }
        return (logits, details) if return_details else logits

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def pool_conv_logits(
    conv_logits: Dict[Any, torch.Tensor], mode: str = "max"
) -> Dict[Any, torch.Tensor]:
    """Collapse the per-post logits of each round into one round-level logit."""
    if mode == "max":
        return {conv_id: values.max() for conv_id, values in conv_logits.items()}
    if mode == "mean":
        return {conv_id: values.mean() for conv_id, values in conv_logits.items()}
    raise ValueError(mode)


def build_model(model_config, device: torch.device) -> PsyTCG:
    """Instantiate :class:`PsyTCG` from a :class:`config.schema.ModelConfig`."""
    checkpoint = model_config.post_encoder_checkpoint
    return PsyTCG(
        post_input_dim=model_config.post_input_dim,
        post_hidden_dim=model_config.post_hidden_dim,
        memory_dim=model_config.memory_dim,
        other_rnn=model_config.other_rnn,
        target_rnn=model_config.target_rnn,
        dropout=model_config.dropout,
        gat_heads=model_config.gat_heads,
        gat_time_dim=model_config.gat_time_dim,
        use_time_attention=model_config.use_time_attention,
        use_context_gate=model_config.use_context_gate,
        other_memory_scope=model_config.other_memory_scope,
        child_memory_mode=model_config.child_memory_mode,
        post_encoder_checkpoint=None if checkpoint is None else str(checkpoint),
        freeze_post_encoder=model_config.freeze_post_encoder,
    ).to(device)
