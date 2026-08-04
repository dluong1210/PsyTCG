"""In-memory graph containers, loading, and user-level windowing."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset


@dataclass
class Event:
    """One post or comment: a sequence of sentence features plus its position."""

    user_id: str
    rel_hidden: torch.Tensor  # [n_sentences, 1024]
    emo_sev: torch.Tensor  # [n_sentences, 28]
    ts_raw: Optional[int] = None  # Unix nanoseconds
    is_target: bool = False
    neighbour_indices: Optional[List[int]] = None  # indices into the flattened event list
    local_neighbour_indices: Optional[List[int]] = None  # indices within this conversation
    delta_t: Optional[torch.Tensor] = None  # filled in by the model, per author
    # Reply identifiers. PsyTCG does not need them (it reads ``neighbour_indices``),
    # but a model that builds the full reply tree of a conversation does.
    node_id: Optional[str] = None
    parent: Optional[str] = None


@dataclass
class Conversation:
    conv_id: Any
    events: List[Event] = field(default_factory=list)


@dataclass
class UserTimeline:
    """A window of consecutive rounds for one target user."""

    target_user_id: str
    label: int
    conversations: List[Conversation] = field(default_factory=list)

    def to(self, device: torch.device) -> "UserTimeline":
        for conversation in self.conversations:
            for event in conversation.events:
                event.rel_hidden = event.rel_hidden.to(device)
                event.emo_sev = event.emo_sev.to(device)
        return self


def load_timelines(path: Path | str) -> Dict[str, List[Dict[str, Any]]]:
    """Load a timeline pickle and enforce round order.

    Conversations must be processed in the order the evaluation server reveals
    them, so ``conv_id`` is sorted ascending and required to be unique.
    """
    with Path(path).open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a dict of user_id -> timeline")

    timelines = {}
    for user_id, timeline in raw.items():
        if not timeline:
            continue
        ordered = sorted(timeline, key=lambda conversation: int(conversation["conv_id"]))
        conv_ids = [int(conversation["conv_id"]) for conversation in ordered]
        if len(conv_ids) != len(set(conv_ids)):
            raise ValueError(f"{user_id}: duplicate conv_id in timeline")
        timelines[str(user_id)] = ordered
    return timelines


def assemble(timeline: Sequence[Dict[str, Any]], label: int, target_user_id: str) -> UserTimeline:
    """Convert raw timeline dicts into :class:`UserTimeline`.

    ``local_neighbours`` are stored per conversation; the model works on one
    flattened event list, so they are shifted by the running event offset.
    """
    conversations, offset = [], 0
    for conversation in timeline:
        events = []
        for event in conversation["events"]:
            neighbours = event.get("local_neighbours") or None
            events.append(
                Event(
                    user_id=event["user_id"],
                    rel_hidden=event["rel_hidden"],
                    emo_sev=event["emo_sev"],
                    ts_raw=event.get("ts_raw"),
                    is_target=bool(event.get("is_target", False)),
                    neighbour_indices=[offset + j for j in neighbours] if neighbours else None,
                    local_neighbour_indices=list(neighbours) if neighbours else None,
                    node_id=event.get("node_id"),
                    parent=event.get("parent"),
                )
            )
        conversations.append(Conversation(conv_id=conversation["conv_id"], events=events))
        offset += len(conversation["events"])
    return UserTimeline(
        target_user_id=target_user_id, label=int(label), conversations=conversations
    )


def make_windows(
    timeline: Sequence[Dict[str, Any]],
    label: int,
    target_user_id: str,
    max_conversations: int,
    stride: Optional[int] = None,
) -> List[UserTimeline]:
    """Split a long timeline into training windows of consecutive rounds.

    Every window ends at a round that carries a prediction, and the final window
    always ends at the last round so the full history is represented. Windows of a
    given user never cross a split boundary because splitting happens at user
    level before this is called.
    """
    if not timeline:
        return []
    target_rounds = [
        index
        for index, conversation in enumerate(timeline)
        if any(bool(event.get("is_target", False)) for event in conversation["events"])
    ]
    if not target_rounds:
        return []
    if len(timeline) <= max_conversations:
        return [assemble(timeline[: target_rounds[-1] + 1], label, target_user_id)]

    stride = stride or max(1, max_conversations // 2)

    # Context-only rounds advance interlocutor memory but emit no score. Move
    # each nominal endpoint back to the latest target-bearing round so no
    # training window is silently skipped for lacking a terminal logit.
    nominal_ends = [
        start + max_conversations - 1
        for start in range(0, len(timeline) - max_conversations + 1, stride)
    ]
    nominal_ends.append(len(timeline) - 1)
    ends = {
        max(index for index in target_rounds if index <= nominal_end)
        for nominal_end in nominal_ends
        if any(index <= nominal_end for index in target_rounds)
    }
    windows = []
    for end in sorted(ends):
        start = max(0, end - max_conversations + 1)
        windows.append(assemble(timeline[start : end + 1], label, target_user_id))
    return windows


class WindowDataset(Dataset):
    """Thin wrapper so windows can be shuffled by a ``DataLoader``."""

    def __init__(self, windows: Sequence[UserTimeline]):
        self.windows = list(windows)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> UserTimeline:
        return self.windows[index]


def collate_windows(batch: List[UserTimeline]) -> List[UserTimeline]:
    """Windows have ragged graph structure; keep them as a list."""
    return batch
