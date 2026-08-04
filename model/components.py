"""PsyTCG building blocks: post encoder, Time-LSTM, and memory attention."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

EMO_SEV_DIM = 28  # 7 emotion probabilities + 21 severity scores


class PostEncoder(nn.Module):
    """Sentence sequence -> single post embedding.

    Emotion/severity scores are projected and fused with the contextual sentence
    representation before an LSTM reads the sentences in order; the state at the
    last real sentence is the post embedding. The auxiliary logit is the
    post-level depression score used to pretrain this module.
    """

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.mapping = nn.Linear(EMO_SEV_DIM, 256)
        self.fusion = nn.Linear(input_dim + 256, input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(512, 1),
        )

    def forward(
        self, padded_posts: Tuple[torch.Tensor, torch.Tensor], post_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rel_hidden, emo_sev = padded_posts
        fused = self.fusion(torch.cat([rel_hidden, self.mapping(emo_sev)], dim=-1))
        outputs, _ = self.lstm(fused)
        last = (post_lengths - 1).clamp(min=0)
        rows = torch.arange(outputs.size(0), device=outputs.device)
        post_embeddings = outputs[rows, last]
        return self.mlp(post_embeddings), post_embeddings


class TimeLSTM(nn.Module):
    """LSTM whose cell state decays with the gap since the author's previous post.

    The short-term component of the cell state is discounted by a learned
    monotone function of the elapsed time, so a long silence erases recent
    detail while long-term state survives. Gaps are given in days and passed
    through ``log1p`` to keep multi-month gaps from saturating the decay gate.
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
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

    def forward(
        self,
        inputs: torch.Tensor,
        timestamps: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """``inputs`` [B, S, D]; ``timestamps`` [B, S] elapsed days per step.

        Passing ``state`` continues an existing sequence, which is how inference
        streams one round at a time. The first step of a *fresh* sequence has no
        predecessor and therefore no gap; when state is carried in, the gap of the
        first step is real and must be honoured.
        """
        batch, steps, _ = inputs.size()
        carried = state is not None
        if carried:
            hidden, cell = state
        else:
            hidden = inputs.new_zeros(batch, self.hidden_size)
            cell = inputs.new_zeros(batch, self.hidden_size)

        outputs = []
        for step in range(steps):
            x = inputs[:, step]
            if self.training and self.dropout_p > 0:
                x = self.dropout(x)

            if step == 0 and not carried:
                elapsed = inputs.new_zeros(batch, 1)
            else:
                elapsed = timestamps[:, step].reshape(batch, 1).float()
            elapsed = torch.log1p(torch.clamp(elapsed, min=0.0))

            short_term = torch.tanh(self.W_decomp(cell) + self.b_decomp)
            decay = torch.sigmoid(self.linear_time(elapsed)).expand(-1, self.hidden_size)
            discounted = cell - short_term + short_term * decay

            input_gate = torch.sigmoid(self.Wi(x) + self.Ui(hidden) + self.bi)
            forget_gate = torch.sigmoid(self.Wf(x) + self.Uf(hidden) + self.bf)
            output_gate = torch.sigmoid(self.Wo(x) + self.Uo(hidden) + self.bo)
            candidate = torch.tanh(self.Wc(x) + self.Uc(hidden) + self.bc)

            cell = forget_gate * discounted + input_gate * candidate
            hidden = output_gate * torch.tanh(cell)
            outputs.append(hidden)

        return torch.stack(outputs, dim=1), (hidden, cell)


def build_rnn(rnn_type: str, input_size: int, hidden_size: int, dropout: float = 0.1) -> nn.Module:
    rnn_type = rnn_type.lower()
    if rnn_type == "gru":
        return nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
    if rnn_type == "lstm":
        return nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
    if rnn_type in ("t-lstm", "timelstm"):
        return TimeLSTM(input_size, hidden_size, dropout=dropout)
    raise ValueError(f"unsupported rnn type: {rnn_type}")


class MemoryAttention(nn.Module):
    """Multi-head attention from a target post over its interlocutors.

    Queries and keys are post embeddings, so relevance is judged on what was
    written. Values additionally carry the interlocutor's accumulated memory,
    which is what makes the aggregation role-asymmetric: the target reads *who*
    replied and *what history they bring*, not just the reply text. With
    ``use_time`` the key is extended by a Time2Vec-style encoding of the gap
    between the reply and the target post.
    """

    def __init__(
        self,
        embed_dim: int,
        memory_dim: int,
        num_heads: int = 8,
        time_dim: int = 8,
        use_time: bool = False,
    ):
        super().__init__()
        if embed_dim % num_heads or memory_dim % num_heads:
            raise ValueError("embed_dim and memory_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.use_time = use_time
        self.time_dim = time_dim if use_time else 0
        self.memory_dim = memory_dim
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim + self.time_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim + memory_dim, memory_dim)
        self.out = nn.Linear(memory_dim, memory_dim)
        self.scale = (embed_dim // num_heads) ** 0.5
        if use_time:
            self.time_freq = nn.Linear(1, time_dim)

    def _encode_time(self, gap_days: torch.Tensor) -> torch.Tensor:
        projected = self.time_freq(torch.log1p(torch.clamp(gap_days, min=0.0)))
        return torch.cat([projected[:, :1], torch.sin(projected[:, 1:])], dim=-1)

    def forward(
        self,
        target_embedding: torch.Tensor,
        neighbour_embeddings: torch.Tensor,
        neighbour_memories: torch.Tensor,
        neighbour_gaps: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        if neighbour_embeddings.size(0) == 0:
            context = torch.zeros(self.memory_dim, device=target_embedding.device)
            return (context, None) if return_attention else context

        n_neighbours = neighbour_embeddings.size(0)
        key_input = neighbour_embeddings
        if self.use_time:
            if neighbour_gaps is None:
                neighbour_gaps = torch.zeros(n_neighbours, 1, device=target_embedding.device)
            key_input = torch.cat([neighbour_embeddings, self._encode_time(neighbour_gaps)], dim=-1)

        query = self.W_q(target_embedding)
        keys = self.W_k(key_input)
        values = self.W_v(torch.cat([neighbour_embeddings, neighbour_memories], dim=-1))

        heads = self.num_heads
        query = query.view(heads, -1)
        keys = keys.view(n_neighbours, heads, -1).permute(1, 0, 2)
        values = values.view(n_neighbours, heads, -1).permute(1, 0, 2)
        attention = torch.softmax(torch.einsum("hne,he->hn", keys, query) / self.scale, dim=-1)
        context = self.out(torch.einsum("hn,hnm->hm", attention, values).reshape(-1))
        return (context, attention) if return_attention else context
