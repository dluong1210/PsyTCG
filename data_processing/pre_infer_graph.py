"""Turn a raw user history into the temporal conversation graph PsyTCG consumes.

A *timeline* is a list of conversations ordered by round (``conv_id`` ascending,
i.e. the order in which the evaluation server reveals them). A conversation is a
list of *events*; every event is one post or comment:

```
{"node_id", "parent", "user_id", "rel_hidden", "emo_sev",
 "ts_raw", "is_target", "local_neighbours"}
```

``local_neighbours`` holds the indices, within the same conversation, of the
non-target nodes one reply hop away from a target node (its parent and its direct
children). These are the edges the role-asymmetric attention reads.

Two collection layouts are supported:

* ``conversation`` -- each round is a real Reddit thread with reply structure.
* ``posts``        -- each round is a single target post; conversational context
  comes from synthetic replies supplied in a separate file.

Both produce the same timeline schema, so training and inference are unaware of
which layout a collection came from.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from data_processing.pre_infer_sent import (
    SentenceEncoder,
    gate_features,
    is_relevant,
    split_sentences,
)

NS_PER_MINUTE = 60 * 10**9

#: Author values that mean "no identifiable author".
_DELETED_AUTHORS = {
    "",
    "none",
    "null",
    "[deleted]",
    "[removed]",
    "[deleted by user]",
    "[removed by user]",
}
#: Anonymous authors get a per-node identity so their memories never merge.
ANON_PREFIX = "__anon__"


def _timestamp_ns(value: Any) -> int:
    parsed = pd.to_datetime(str(value).strip(), errors="coerce", utc=True)
    return 0 if parsed is pd.NaT else int(parsed.value)


def _is_deleted_author(author: Any) -> bool:
    return author is None or str(author).strip().lower() in _DELETED_AUTHORS


# ---------------------------------------------------------------------------
# Conversation collections
# ---------------------------------------------------------------------------
def read_discussion_nodes(
    discussions: Sequence[Dict[str, Any]], target_uid: str
) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, List[str]]]:
    """Flatten raw threads into ``nodes`` and ``conversation -> node ids``.

    Expected raw layout (one JSON file per user)::

        [{"submission": {"submission_id", "user_id", "title", "body",
                         "created_utc", "target"},
          "comments": [{"comment_id", "parent_id", "user_id", "body",
                        "created_utc", "target"}, ...]}, ...]

    The author of a target node is forced to ``target_uid`` so the model can
    always identify the subject of the timeline.
    """
    nodes: Dict[str, Dict[str, Any]] = {}
    conversation_nodes: Dict[int, List[str]] = defaultdict(list)

    def author_of(raw_author: Any, is_target: bool, conv_index: int, node_id: str) -> str:
        if is_target:
            return target_uid
        if _is_deleted_author(raw_author):
            return f"{ANON_PREFIX}{conv_index}_{node_id}"
        return str(raw_author)

    for conv_index, conversation in enumerate(discussions):
        submission = conversation.get("submission") or {}
        submission_id = str(submission.get("submission_id") or "")
        if submission_id:
            title = (submission.get("title") or "").strip()
            body = (submission.get("body") or "").strip()
            if body in ("", "[deleted]", "[removed]"):
                body = ""
            is_target = str(submission.get("target")) == "True" or (
                submission.get("user_id") == target_uid
            )
            nodes[submission_id] = {
                "author": author_of(
                    submission.get("user_id"), is_target, conv_index, submission_id
                ),
                "parent": None,
                "text": (title + " " + body).strip() or ".",
                "ts": _timestamp_ns(submission.get("created_utc", "")),
                "conv": conv_index,
                "is_target": is_target,
            }
            conversation_nodes[conv_index].append(submission_id)

        for comment in conversation.get("comments") or []:
            comment_id = str(comment.get("comment_id") or "")
            if not comment_id:
                continue
            body = (comment.get("body") or "").strip()
            if body in ("[deleted]", "[removed]", "[deleted by user]"):
                body = ""
            parent = comment.get("parent_id")
            parent = str(parent) if parent not in (None, "None", "") else submission_id
            is_target = str(comment.get("target")) == "True" or (
                comment.get("user_id") == target_uid
            )
            nodes[comment_id] = {
                "author": author_of(comment.get("user_id"), is_target, conv_index, comment_id),
                "parent": parent,
                "text": body,
                "ts": _timestamp_ns(comment.get("created_utc", submission.get("created_utc", ""))),
                "conv": conv_index,
                "is_target": is_target,
            }
            conversation_nodes[conv_index].append(comment_id)

    return nodes, dict(conversation_nodes)


def interlocutor_users(
    nodes: Dict[str, Dict[str, Any]],
    conversation_nodes: Dict[int, List[str]],
    target_uid: str,
    seed_ids: Set[str],
) -> Set[str]:
    """Users who ever replied to, or were replied to by, a seed post.

    The scan spans the whole history rather than a single thread: interlocutor
    memories persist across conversations, so every node these users wrote has to
    be available to the model.
    """
    users: Set[str] = set()
    for conv_index, node_ids in conversation_nodes.items():
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        for node_id in node_ids:
            parent = nodes[node_id]["parent"]
            if parent is not None and parent in nodes and int(nodes[parent]["conv"]) == conv_index:
                adjacency[node_id].add(parent)
                adjacency[parent].add(node_id)
        for anchor in (n for n in node_ids if n in seed_ids):
            users |= {
                nodes[neighbour]["author"]
                for neighbour in adjacency[anchor]
                if not nodes[neighbour]["is_target"]
            }
    users.discard(target_uid)
    return users


def plan_conversations(
    nodes: Dict[str, Dict[str, Any]],
    conversation_nodes: Dict[int, List[str]],
    seed_ids: Set[str],
    interlocutors: Set[str],
) -> Tuple[List[Tuple[int, List[str]]], Set[str]]:
    """Select which nodes of each round enter the graph.

    A round keeps its seed posts plus every node written by an interlocutor.
    Rounds without a seed are still kept when they contain interlocutor activity:
    they emit no prediction but they do advance interlocutor memory. The timeline
    stops after the last round containing a seed, since later rounds can no longer
    influence any decision.
    """
    ordered = sorted(conversation_nodes)
    with_seed = [
        conv_index
        for conv_index in ordered
        if any(nodes[n]["is_target"] and n in seed_ids for n in conversation_nodes[conv_index])
    ]
    if not with_seed:
        return [], set()
    last_seed_conv = with_seed[-1]

    plan: List[Tuple[int, List[str]]] = []
    context_ids: Set[str] = set()
    for conv_index in ordered:
        if conv_index > last_seed_conv:
            break
        node_ids = conversation_nodes[conv_index]
        seeds = [n for n in node_ids if n in seed_ids]
        context = [
            n
            for n in node_ids
            if not nodes[n]["is_target"]
            and n not in set(seeds)
            and nodes[n]["author"] in interlocutors
        ]
        if seeds:
            plan.append((conv_index, seeds + context))
            context_ids.update(context)
        elif context:
            plan.append((conv_index, context))
            context_ids.update(context)
    return plan, context_ids


def assemble_timeline(
    nodes: Dict[str, Dict[str, Any]],
    plan: Sequence[Tuple[int, List[str]]],
    features: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Materialise the planned rounds into the timeline schema."""
    timeline = []
    for conv_index, selected in plan:
        order = sorted(selected, key=lambda n: nodes[n]["ts"])
        local_index = {node_id: i for i, node_id in enumerate(order)}
        selected_set = set(selected)

        adjacency: Dict[int, Set[int]] = defaultdict(set)
        for node_id in selected:
            parent = nodes[node_id]["parent"]
            if parent in selected_set and parent != node_id:
                adjacency[local_index[node_id]].add(local_index[parent])
                adjacency[local_index[parent]].add(local_index[node_id])

        events = []
        for position, node_id in enumerate(order):
            neighbours = (
                sorted(j for j in adjacency[position] if not nodes[order[j]]["is_target"])
                if nodes[node_id]["is_target"]
                else []
            )
            events.append(
                {
                    "node_id": node_id,
                    "parent": nodes[node_id]["parent"],
                    "user_id": nodes[node_id]["author"],
                    "rel_hidden": features[node_id]["rel_hidden"],
                    "emo_sev": features[node_id]["emo_sev"],
                    "ts_raw": nodes[node_id]["ts"],
                    "is_target": nodes[node_id]["is_target"],
                    "local_neighbours": neighbours,
                }
            )

        timeline.append(
            {
                "conv_id": int(conv_index),
                "events": events,
                "context_only": not any(event["is_target"] for event in events),
                "min_ts": min(nodes[n]["ts"] for n in selected),
            }
        )
    timeline.sort(key=lambda conversation: conversation["conv_id"])
    return timeline


def build_conversation_timeline(
    path: Path,
    target_uid: str,
    encoder: SentenceEncoder,
    threshold: float,
) -> Optional[List[Dict[str, Any]]]:
    """Build one user's timeline from a raw discussion file.

    Returns ``None`` when no target post passes the relevance threshold; such a
    user is pruned and scored as a negative decision at every round.
    """
    with Path(path).open(encoding="utf-8") as handle:
        discussions = json.load(handle)
    nodes, conversation_nodes = read_discussion_nodes(discussions, target_uid)

    target_ids = sorted(n for n, node in nodes.items() if node["is_target"])
    if not target_ids:
        return None

    target_features = _encode(encoder, nodes, target_ids)
    seed_ids = {n for n in target_ids if is_relevant(target_features[n], threshold)}
    if not seed_ids:
        return None

    interlocutors = interlocutor_users(nodes, conversation_nodes, target_uid, seed_ids)
    plan, context_ids = plan_conversations(nodes, conversation_nodes, seed_ids, interlocutors)
    if not plan:
        return None

    context_features = _encode(encoder, nodes, sorted(context_ids))
    features = {
        node_id: gate_features(node, threshold)
        for node_id, node in {**context_features, **target_features}.items()
    }
    return assemble_timeline(nodes, plan, features)


def _encode(
    encoder: SentenceEncoder, nodes: Dict[str, Dict[str, Any]], node_ids: Sequence[str]
) -> Dict[str, Dict[str, Any]]:
    if not node_ids:
        return {}
    encoded = encoder.encode_nodes([split_sentences(nodes[n]["text"]) for n in node_ids])
    return dict(zip(node_ids, encoded, strict=True))


# ---------------------------------------------------------------------------
# Post-only collections
# ---------------------------------------------------------------------------
def build_post_timeline(
    path: Path,
    target_uid: str,
    encoder: SentenceEncoder,
    threshold: float,
    synthetic_comments: Optional[Dict[str, List[str]]] = None,
    comment_offset_minutes: int = 1,
) -> Optional[List[Dict[str, Any]]]:
    """Build a timeline for collections that only provide target-authored posts.

    Expected raw layout (one JSON file per user)::

        [{"post_id": str, "date": str, "text": str}, ...]

    Each relevant post becomes a one-round conversation rooted at that post.
    ``synthetic_comments`` maps ``post_id`` to generated replies; each reply is a
    non-target child written by a distinct interlocutor. Posts without replies
    still train: attention simply sees no context for that round.
    """
    with Path(path).open(encoding="utf-8") as handle:
        posts = json.load(handle)
    posts = sorted(posts, key=lambda post: _timestamp_ns(post.get("date", "")))
    if not posts:
        return None

    synthetic_comments = synthetic_comments or {}
    post_features = encoder.encode_nodes([split_sentences(post.get("text", "")) for post in posts])
    keep = [
        (post, features)
        for post, features in zip(posts, post_features, strict=True)
        if is_relevant(features, threshold)
    ]
    if not keep:
        return None

    comment_texts: List[str] = []
    comment_owner: List[Tuple[int, int]] = []
    for round_index, (post, _) in enumerate(keep):
        for reply_index, text in enumerate(synthetic_comments.get(str(post["post_id"]), [])):
            comment_texts.append(text)
            comment_owner.append((round_index, reply_index))
    comment_features = (
        encoder.encode_nodes([split_sentences(text) for text in comment_texts])
        if comment_texts
        else []
    )
    replies: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for (round_index, _), features in zip(comment_owner, comment_features, strict=True):
        replies[round_index].append(features)

    timeline = []
    for round_index, (post, features) in enumerate(keep):
        post_id = str(post["post_id"])
        timestamp = _timestamp_ns(post.get("date", ""))
        events = [
            {
                "node_id": post_id,
                "parent": None,
                "user_id": target_uid,
                **gate_features(features, threshold),
                "ts_raw": timestamp,
                "is_target": True,
                "local_neighbours": [],
            }
        ]
        for reply_index, reply in enumerate(replies.get(round_index, [])):
            events.append(
                {
                    "node_id": f"{post_id}#c{reply_index}",
                    "parent": post_id,
                    # Synthetic interlocutors are distinct per post: nothing is
                    # known that would let their memories carry across rounds.
                    "user_id": f"{ANON_PREFIX}{post_id}_{reply_index}",
                    **gate_features(reply, threshold),
                    "ts_raw": timestamp
                    + (reply_index + 1) * comment_offset_minutes * NS_PER_MINUTE,
                    "is_target": False,
                    "local_neighbours": [],
                }
            )
        events[0]["local_neighbours"] = list(range(1, len(events)))
        timeline.append(
            {
                "conv_id": round_index,
                "events": events,
                "context_only": False,
                "min_ts": timestamp,
            }
        )
    return timeline


def load_synthetic_comments(path: Optional[Path]) -> Dict[str, Dict[str, List[str]]]:
    """Load ``{user_id: {post_id: [reply, ...]}}`` produced by the reply generator."""
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        str(user_id): {str(post_id): list(replies) for post_id, replies in posts.items()}
        for user_id, posts in payload.items()
    }
