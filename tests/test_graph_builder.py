"""Graph construction checks, run against a stub encoder (no checkpoints needed)."""

from __future__ import annotations

import json
from typing import Dict, List, Sequence

import torch

from data_processing.dataset import make_windows
from data_processing.pre_infer_graph import (
    ANON_PREFIX,
    build_conversation_timeline,
    build_post_timeline,
    read_discussion_nodes,
)
from data_processing.pre_infer_sent import split_sentences

TARGET = "subject1"


class StubEncoder:
    """Marks a sentence relevant iff it contains 'depressed'."""

    def encode_nodes(
        self, sentence_lists: Sequence[Sequence[str]]
    ) -> List[Dict[str, torch.Tensor]]:
        nodes = []
        for sentences in sentence_lists:
            n = len(sentences)
            relevance = torch.tensor([0.9 if "depressed" in s.lower() else 0.1 for s in sentences])
            nodes.append(
                {
                    "rel_hidden": torch.randn(n, 1024),
                    "emotion": torch.rand(n, 7),
                    "severity": torch.rand(n, 21),
                    "relevance": relevance,
                }
            )
        return nodes


def thread(conv_index: int, target_text: str, reply_texts: Sequence[str], day: int):
    """One thread: target submission plus replies from named other users."""
    date = f"2024-03-{day:02d} 10:00:00"
    return {
        "submission": {
            "submission_id": f"s{conv_index}",
            "user_id": TARGET,
            "title": "",
            "body": target_text,
            "created_utc": date,
            "target": "True",
        },
        "comments": [
            {
                "comment_id": f"s{conv_index}c{i}",
                "parent_id": f"s{conv_index}",
                "user_id": author,
                "body": text,
                "created_utc": f"2024-03-{day:02d} 11:0{i}:00",
                "target": "False",
            }
            for i, (author, text) in enumerate(reply_texts)
        ],
    }


def write_discussions(tmp_path, discussions):
    path = tmp_path / f"{TARGET}.json"
    path.write_text(json.dumps(discussions), encoding="utf-8")
    return path


def test_split_sentences_never_returns_empty():
    assert split_sentences("[deleted]") == ["X"]
    assert split_sentences("") == ["X"]
    assert len(split_sentences("I feel bad. It got worse.")) == 2


def test_reply_edges_link_target_to_its_children(tmp_path):
    path = write_discussions(
        tmp_path, [thread(0, "I am depressed", [("alice", "hang in there")], day=1)]
    )
    timeline = build_conversation_timeline(path, TARGET, StubEncoder(), threshold=0.5)
    assert len(timeline) == 1
    events = timeline[0]["events"]
    target = next(e for e in events if e["is_target"])
    assert [events[j]["user_id"] for j in target["local_neighbours"]] == ["alice"]
    # Non-target events never carry neighbours: only the target reads context.
    assert all(not e["local_neighbours"] for e in events if not e["is_target"])


def test_user_without_a_relevant_post_is_pruned(tmp_path):
    path = write_discussions(tmp_path, [thread(0, "nice weather", [("alice", "yes")], day=1)])
    assert build_conversation_timeline(path, TARGET, StubEncoder(), threshold=0.5) is None


def test_rounds_after_the_last_relevant_post_are_dropped(tmp_path):
    path = write_discussions(
        tmp_path,
        [
            thread(0, "I am depressed", [("alice", "hi")], day=1),
            thread(1, "just chatting", [("alice", "hi")], day=2),
        ],
    )
    timeline = build_conversation_timeline(path, TARGET, StubEncoder(), threshold=0.5)
    assert [c["conv_id"] for c in timeline] == [0]


def test_only_interlocutors_of_a_seed_enter_the_graph(tmp_path):
    """A round is kept for its seed plus the users who ever replied to a seed."""
    path = write_discussions(
        tmp_path,
        [
            thread(0, "I am depressed", [("alice", "hang in there")], day=1),
            thread(1, "I am depressed again", [("bob", "hello"), ("alice", "still here")], day=5),
        ],
    )
    timeline = build_conversation_timeline(path, TARGET, StubEncoder(), threshold=0.5)
    authors_round_2 = {e["user_id"] for e in timeline[1]["events"] if not e["is_target"]}
    assert authors_round_2 == {"alice", "bob"}  # both replied to a seed post


def test_severity_is_zeroed_for_irrelevant_sentences(tmp_path):
    path = write_discussions(
        tmp_path, [thread(0, "I am depressed. The sky is blue.", [("alice", "ok")], day=1)]
    )
    timeline = build_conversation_timeline(path, TARGET, StubEncoder(), threshold=0.5)
    target = next(e for e in timeline[0]["events"] if e["is_target"])
    emo_sev = target["emo_sev"]
    assert emo_sev.shape == (2, 28)
    assert emo_sev[0, 7:].abs().sum() > 0  # relevant sentence keeps its severity
    assert emo_sev[1, 7:].abs().sum() == 0  # irrelevant sentence is gated to zero


def test_deleted_authors_get_separate_identities(tmp_path):
    discussions = [thread(0, "I am depressed", [("alice", "hi")], day=1)]
    discussions[0]["comments"].append(
        {
            "comment_id": "s0c9",
            "parent_id": "s0",
            "user_id": "[deleted]",
            "body": "gone",
            "created_utc": "2024-03-01 12:00:00",
            "target": "False",
        }
    )
    nodes, _ = read_discussion_nodes(discussions, TARGET)
    assert nodes["s0c9"]["author"].startswith(ANON_PREFIX)
    assert nodes["s0c9"]["author"] != nodes["s0c0"]["author"]


def test_post_mode_builds_one_round_per_relevant_post(tmp_path):
    path = tmp_path / f"{TARGET}.json"
    path.write_text(
        json.dumps(
            [
                {"post_id": "p1", "date": "2017-05-02 11:00:00", "text": "I am depressed"},
                {"post_id": "p2", "date": "2017-05-09 08:00:00", "text": "nice weather"},
                {"post_id": "p3", "date": "2017-05-20 08:00:00", "text": "still depressed"},
            ]
        ),
        encoding="utf-8",
    )
    timeline = build_post_timeline(
        path,
        TARGET,
        StubEncoder(),
        threshold=0.5,
        synthetic_comments={"p1": ["that sounds hard", "I hear you"]},
    )
    assert [c["conv_id"] for c in timeline] == [0, 1]  # p2 dropped as irrelevant
    first = timeline[0]["events"]
    assert first[0]["is_target"] and first[0]["local_neighbours"] == [1, 2]
    assert len({e["user_id"] for e in first[1:]}) == 2  # distinct synthetic interlocutors
    assert all(e["ts_raw"] > first[0]["ts_raw"] for e in first[1:])
    assert not timeline[1]["events"][0]["local_neighbours"]  # p3 has no replies


def test_training_windows_end_on_target_rounds():
    event = {
        "user_id": TARGET,
        "rel_hidden": torch.randn(1, 1024),
        "emo_sev": torch.randn(1, 28),
        "ts_raw": 0,
        "local_neighbours": [],
    }
    timeline = [
        {"conv_id": 0, "events": [{**event, "is_target": True}]},
        {"conv_id": 1, "events": [{**event, "user_id": "alice", "is_target": False}]},
        {"conv_id": 2, "events": [{**event, "is_target": True}]},
        {"conv_id": 3, "events": [{**event, "user_id": "alice", "is_target": False}]},
        {"conv_id": 4, "events": [{**event, "is_target": True}]},
    ]
    windows = make_windows(timeline, 1, TARGET, max_conversations=2, stride=2)
    assert windows
    assert all(window.conversations[-1].events[-1].is_target for window in windows)
