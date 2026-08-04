"""Dependency-light guards for the controlled GAT--TLSTM baseline."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from data_processing.dataset import Conversation, Event, UserTimeline
from model.gat_tlstm import GATTLSTM, BaselineTimeLSTM, _edge_index


def event(node_id, parent=None):
    return Event(
        user_id=str(node_id),
        rel_hidden=torch.randn(1, 1024),
        emo_sev=torch.randn(1, 28),
        node_id=str(node_id),
        parent=parent,
    )


def test_reply_tree_becomes_bidirectional_edges():
    conversation = Conversation("0", [event("root"), event("child", "root")])
    edges = {tuple(edge) for edge in _edge_index(conversation, torch.device("cpu")).t().tolist()}
    assert edges == {(0, 1), (1, 0)}


def test_local_neighbours_are_a_topology_fallback_without_reply_ids():
    root = event("root")
    child = event("child")
    root.node_id = child.node_id = None
    root.local_neighbour_indices = [1]
    conversation = Conversation("0", [root, child])
    edges = {tuple(edge) for edge in _edge_index(conversation, torch.device("cpu")).t().tolist()}
    assert edges == {(0, 1), (1, 0)}


def test_baseline_time_lstm_honours_a_carried_gap():
    model = BaselineTimeLSTM(8, 8).eval()
    _, state = model(torch.randn(1, 1, 8), torch.zeros(1, 1))
    current = torch.randn(1, 1, 8)
    short, _ = model(current, torch.tensor([[0.5]]), state)
    long, _ = model(current, torch.tensor([[300.0]]), state)
    assert (short - long).abs().max() > 1e-6


def test_full_baseline_trains_new_gat_without_a_gat_checkpoint():
    root = event("target")
    root.user_id = "target"
    root.is_target = True
    root.ts_raw = 0
    root.local_neighbour_indices = [1]
    child = event("reply", "target")
    child.ts_raw = 1
    user = UserTimeline("target", 1, [Conversation("0", [root, child])])

    model = GATTLSTM(
        None,
        dropout=0.1,
        freeze_post_encoder=True,
        load_post_checkpoint=False,
    ).train()
    logits = model(user, post_batch_size=2)
    assert logits["0"].shape == (1,)
    F.binary_cross_entropy_with_logits(logits["0"], torch.ones(1)).backward()
    assert model.concat_linear.weight.grad is not None
    assert model.user_model.Wi.weight.grad is not None
    assert model.gat_level_model.gat_layers[0].lin_l.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.post_encoder.parameters())
