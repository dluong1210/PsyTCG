#!/usr/bin/env python3
"""Encode a collection and write its temporal conversation graphs to a pickle.

This is the only stage that needs the sentence encoders, and it runs once per
collection. The output pickle maps ``user_id -> timeline`` and is consumed by
the training, inference, and evaluation entry points.

Examples
--------
Conversation collection (real Reddit threads, one JSON file per user)::

    python -m data_processing.build_graph_pkl --mode conversation \
        --raw-dir data/raw/2026 --output data/graphs/2026.pkl \
        --rel-sev-checkpoint checkpoints/rel_sev.pt \
        --emotion-model-dir checkpoints/emotion

Post-only collection (target posts plus generated replies)::

    python -m data_processing.build_graph_pkl --mode posts \
        --raw-dir data/raw/2017 --output data/graphs/2017.pkl \
        --synthetic-comments data/raw/2017_replies.json \
        --rel-sev-checkpoint checkpoints/rel_sev.pt \
        --emotion-model-dir checkpoints/emotion
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from tqdm.auto import tqdm

from config.preprocessing import (
    BASE_MODEL,
    BATCH_SIZE,
    EMOTION_MODEL_DIR,
    MAX_LENGTH,
    REL_SEV_CHECKPOINT,
    RELEVANCE_THRESHOLD,
)
from data_processing.pre_infer_graph import (
    build_conversation_timeline,
    build_post_timeline,
    load_synthetic_comments,
)
from data_processing.pre_infer_sent import SentenceEncoder
from utils.common import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=("conversation", "posts"), required=True)
    parser.add_argument("--raw-dir", type=Path, required=True, help="One JSON file per user.")
    parser.add_argument("--output", type=Path, required=True, help="Destination pickle.")
    parser.add_argument("--glob", default="*.json", help="Filename pattern inside --raw-dir.")
    parser.add_argument(
        "--user-prefix",
        default="",
        help="Prefix prepended to each filename stem to form the user id "
        "(use it when the golden truth is keyed as e.g. 2017_<id>).",
    )
    parser.add_argument("--rel-sev-checkpoint", type=Path, default=REL_SEV_CHECKPOINT)
    parser.add_argument("--emotion-model-dir", type=Path, default=EMOTION_MODEL_DIR)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=RELEVANCE_THRESHOLD,
        help="A target post is a seed when any of its sentences exceeds this; "
        "severity of non-passing sentences is zeroed.",
    )
    parser.add_argument(
        "--synthetic-comments",
        type=Path,
        default=None,
        help="posts mode only: JSON of {user_id: {post_id: [reply, ...]}}.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Sentences per encoder batch."
    )
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--max-users", type=int, default=None, help="Limit for a smoke test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    paths = sorted(Path(args.raw_dir).glob(args.glob))
    if not paths:
        raise SystemExit(f"no files matching {args.glob!r} under {args.raw_dir}")
    if args.max_users:
        paths = paths[: args.max_users]

    encoder = SentenceEncoder(
        base_model=args.base_model,
        rel_sev_checkpoint=args.rel_sev_checkpoint,
        emotion_model_dir=args.emotion_model_dir,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    comments = load_synthetic_comments(args.synthetic_comments) if args.mode == "posts" else {}

    timelines = {}
    pruned = 0
    for path in tqdm(paths, desc=f"build {args.mode}"):
        user_id = f"{args.user_prefix}{path.stem}"
        if args.mode == "conversation":
            timeline = build_conversation_timeline(path, user_id, encoder, args.relevance_threshold)
        else:
            timeline = build_post_timeline(
                path,
                user_id,
                encoder,
                args.relevance_threshold,
                synthetic_comments=comments.get(user_id) or comments.get(path.stem),
            )
        if timeline:
            timelines[user_id] = timeline
        else:
            pruned += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(timelines, handle, protocol=pickle.HIGHEST_PROTOCOL)

    rounds = sum(len(timeline) for timeline in timelines.values())
    events = sum(len(c["events"]) for timeline in timelines.values() for c in timeline)
    print(
        f"{len(timelines)}/{len(paths)} users kept ({pruned} pruned: no relevant target post)\n"
        f"{rounds} rounds, {events} events -> {args.output}"
    )


if __name__ == "__main__":
    main()
