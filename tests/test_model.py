"""Forward/backward guards on synthetic timelines.

These check the invariants that are easy to break silently: neighbour indices
must stay aligned across conversations, elapsed time must actually influence the
target memory, and gradients must reach the attention and gate.
"""

from __future__ import annotations

from collections import Counter
from types import MethodType

import torch
import torch.nn as nn

from data_processing.dataset import Conversation, Event, UserTimeline
from model.components import TimeLSTM
from model.psytcg import PsyTCG, pool_conv_logits

NS_PER_DAY = 24 * 3600 * 10**9


def make_event(user_id: str, day: float, neighbours=None, n_sentences: int = 2) -> Event:
    return Event(
        user_id=user_id,
        rel_hidden=torch.randn(n_sentences, 1024),
        emo_sev=torch.randn(n_sentences, 28),
        ts_raw=int(day * NS_PER_DAY),
        neighbour_indices=neighbours,
    )


def make_user() -> UserTimeline:
    """Two rounds; sentence counts and neighbour sets differ between them."""
    first = Conversation(
        "0",
        [
            make_event("other-a", 1.0),
            make_event("target", 1.1, neighbours=[0]),
            make_event("other-b", 1.2, n_sentences=3),
            make_event("target", 2.0, neighbours=[0, 2]),
        ],
    )
    second = Conversation(
        "1", [make_event("other-a", 30.0), make_event("target", 30.5, neighbours=[4])]
    )
    return UserTimeline("target", 1, [first, second])


def build(**overrides) -> PsyTCG:
    kwargs = dict(memory_dim=64, gat_heads=4, use_context_gate=True, use_time_attention=True)
    kwargs.update(overrides)
    return PsyTCG(**kwargs)


def test_forward_scores_every_round_with_a_target_post():
    model = build().train()
    logits = model(make_user(), post_batch_size=8)
    assert set(logits) == {"0", "1"}
    assert logits["0"].shape == (2,)  # two target posts in the first round
    assert logits["1"].shape == (1,)


def test_only_last_conversation_scores_one_round():
    model = build().train()
    logits = model(make_user(), post_batch_size=8, only_last_conversation=True)
    assert set(logits) == {"1"}


def test_gradients_reach_attention_and_gate():
    model = build().train()
    pooled = pool_conv_logits(model(make_user(), post_batch_size=8), "max")
    loss = nn.functional.binary_cross_entropy_with_logits(pooled["1"].view(1), torch.ones(1))
    loss.backward()
    assert model.attention.W_q.weight.grad is not None
    assert model.context_gate.weight.grad is not None
    assert model.target_user_rnn.Wi.weight.grad is not None


def test_frozen_post_encoder_receives_no_gradient():
    model = build(freeze_post_encoder=True).train()
    pooled = pool_conv_logits(model(make_user(), post_batch_size=8), "max")
    nn.functional.binary_cross_entropy_with_logits(pooled["1"].view(1), torch.ones(1)).backward()
    assert all(p.grad is None for p in model.post_encoder.parameters())


def test_details_expose_gate_and_attention_per_round():
    model = build().eval()
    with torch.no_grad():
        _, details = model(make_user(), post_batch_size=8, return_details=True)
    assert len(details["gates"]) == 3  # one per target post
    assert all(0.0 <= value <= 1.0 for _, value in details["gates"])
    for _, attention in details["attention"]:
        assert torch.allclose(attention.sum(dim=-1), torch.ones(attention.size(0)), atol=1e-5)


def test_time_lstm_honours_the_gap_of_a_carried_state():
    """A resumed sequence must react to elapsed time on its very first step."""
    lstm = TimeLSTM(8, 8).eval()
    x = torch.randn(1, 1, 8)
    _, state = lstm(torch.randn(1, 1, 8), torch.zeros(1, 1))
    short, _ = lstm(x, torch.tensor([[0.5]]), state)
    long, _ = lstm(x, torch.tensor([[300.0]]), state)
    assert (short - long).abs().max() > 1e-6


def test_context_gate_can_be_disabled():
    model = build(use_context_gate=False).train()
    assert not hasattr(model, "context_gate")
    assert model(make_user(), post_batch_size=8)


def test_post_mode_updates_each_interlocutor_event_once_before_attention():
    """A reply between two target posts must not be skipped or replayed."""
    first_target = make_event("target", 1.0, neighbours=[1])
    reply = make_event("other-a", 1.1)
    second_target = make_event("target", 1.2, neighbours=[1])
    user = UserTimeline(
        "target",
        1,
        [Conversation("0", [first_target, reply, second_target])],
    )
    model = build(child_memory_mode="post").eval()
    calls = Counter()
    original = model._step_interlocutor

    def counted(self, index, *args, **kwargs):
        calls[index] += 1
        return original(index, *args, **kwargs)

    model._step_interlocutor = MethodType(counted, model)
    with torch.no_grad():
        model(user, post_batch_size=8)
    assert calls == Counter({1: 1})


def test_configured_post_checkpoint_must_exist(tmp_path):
    missing = tmp_path / "missing.pt"
    try:
        build(post_encoder_checkpoint=str(missing))
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("a configured missing checkpoint must not be silently ignored")
