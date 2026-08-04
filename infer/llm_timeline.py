#!/usr/bin/env python3
"""Infer depression from the k most recent TARGET posts/comments with vLLM.

At conversation i, the prompt contains the k chronologically latest target-user
nodes observed in conversations 0..i.  Users are stopped immediately after their
first positive decision.  The JSONL output is resumable.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from config.llm import (
    BATCH_SIZE,
    DTYPE,
    GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN,
    MAX_NEW_TOKENS,
    MAX_RETRIES,
    MODEL,
    OUTPUT_DIR,
    OVERFLOW_SIDE,
    RAW_DIR,
    RECENT_K,
    TENSOR_PARALLEL_SIZE,
    TIMELINE_PROMPT,
    TRUST_REMOTE_CODE,
)
from infer.llm_conversation import (
    append_records,
    chat_token_count,
    clean_text,
    conversation_nodes,
    discover_user_files,
    hard_fit_content,
    infer_decisions,
    is_target_node,
    load_json,
    load_resume_state,
    make_llm,
    make_messages,
    node_id,
    node_sort_key,
    submission_id,
    user_id_from_path,
)

RECENT_USER_PREFIX = TIMELINE_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "-k", type=int, default=RECENT_K, help="Number of most recent TARGET posts/comments."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Number of active users/inference prompts per vLLM batch.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=TENSOR_PARALLEL_SIZE)
    parser.add_argument("--gpu-memory-utilization", type=float, default=GPU_MEMORY_UTILIZATION)
    parser.add_argument("--dtype", default=DTYPE)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument(
        "--max-users", type=int, default=None, help="Optional user limit for a small test run."
    )
    parser.add_argument(
        "--overflow-side",
        choices=("latest", "earliest"),
        default=OVERFLOW_SIDE,
        help="If k nodes still exceed the context, hard-truncate while preserving the latest (default) or earliest characters.",
    )
    parser.add_argument(
        "--trust-remote-code", action=argparse.BooleanOptionalAction, default=TRUST_REMOTE_CODE
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate recent-node construction without loading vLLM or writing output.",
    )
    args = parser.parse_args()
    if args.k <= 0:
        parser.error("-k must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.tensor_parallel_size <= 0:
        parser.error("--tensor-parallel-size must be positive")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if not 1 <= args.max_new_tokens < MAX_MODEL_LEN:
        parser.error(f"--max-new-tokens must be in [1, {MAX_MODEL_LEN - 1}]")
    if args.max_retries <= 0:
        parser.error("--max-retries must be positive")
    if args.max_users is not None and args.max_users <= 0:
        parser.error("--max-users must be positive")
    if args.output is None:
        args.output = OUTPUT_DIR / f"timeline_k{args.k}.jsonl"
    return args


def target_nodes(conversation: dict[str, Any], user_id: str) -> list[dict[str, Any]]:
    return [node for node in conversation_nodes(conversation) if is_target_node(node, user_id)]


def update_history(
    history: dict[str, dict[str, Any]], conversation: dict[str, Any], user_id: str
) -> None:
    current_targets = target_nodes(conversation, user_id)
    if not current_targets:
        raise ValueError(f"conversation has no target node for {user_id}")
    for node in current_targets:
        current_id = node_id(node)
        if current_id:
            history[current_id] = node


def recent_history(history: dict[str, dict[str, Any]], k: int) -> list[dict[str, Any]]:
    return sorted(history.values(), key=node_sort_key)[-k:]


def render_recent_target_nodes(nodes: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = []
    for position, node in enumerate(nodes, start=1):
        kind = "POST" if node.get("type") == "submission" else "COMMENT"
        metadata = [f"TARGET {kind}", f"id={node_id(node)}"]
        if node.get("created_utc"):
            metadata.append(f"time={node['created_utc']}")
        if node.get("parent_id") is not None:
            metadata.append(f"reply_to={node['parent_id']}")
        title = clean_text(node.get("title"))
        body = clean_text(node.get("body"))
        text = " — ".join(part for part in (title, body) if part) or "."
        lines.append(f"{position}. [{' | '.join(metadata)}] {text}")
    return "\n".join(lines)


def build_fitted_recent_prompt(
    nodes: Sequence[dict[str, Any]],
    tokenizer: Any,
    max_new_tokens: int,
    overflow_side: str,
) -> tuple[list[dict[str, str]], str, int]:
    content = render_recent_target_nodes(nodes)
    messages = make_messages(content, RECENT_USER_PREFIX)
    max_input_tokens = MAX_MODEL_LEN - max_new_tokens
    count = chat_token_count(tokenizer, messages)
    if count <= max_input_tokens:
        return messages, "recent_k", count
    keep = "suffix" if overflow_side == "latest" else "prefix"
    fitted, count = hard_fit_content(
        content,
        tokenizer,
        max_input_tokens,
        keep=keep,
        user_prefix=RECENT_USER_PREFIX,
    )
    return make_messages(fitted, RECENT_USER_PREFIX), f"recent_k_char_{keep}", count


@dataclass
class RecentUserState:
    user_id: str
    path: Path
    conversations: list[dict[str, Any]]
    next_index: int = 0
    history: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_recent_user_state(path: Path, next_index: int = 0) -> RecentUserState | None:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Raw file must contain a list: {path}")
    if next_index >= len(data):
        return None
    user_id = user_id_from_path(path)
    state = RecentUserState(user_id=user_id, path=path, conversations=data, next_index=next_index)
    for conversation in data[:next_index]:
        update_history(state.history, conversation, user_id)
    return state


def dry_run(files: Sequence[Path], k: int) -> None:
    conversation_count = target_count = max_history = 0
    for path in files:
        user_id = user_id_from_path(path)
        data = load_json(path)
        if not isinstance(data, list):
            raise ValueError(f"Raw file must contain a list: {path}")
        history: dict[str, dict[str, Any]] = {}
        for conversation in data:
            current = target_nodes(conversation, user_id)
            if not current:
                raise ValueError(f"No target node in {path}")
            target_count += len(current)
            update_history(history, conversation, user_id)
            selected = recent_history(history, k)
            if len(selected) > k:
                raise AssertionError("recent history exceeds k")
            max_history = max(max_history, len(selected))
        conversation_count += len(data)
    print(
        f"Dry run OK: users={len(files)}, conversations={conversation_count}, "
        f"target_nodes={target_count}, max_recent_nodes={max_history}, k={k}. "
        "vLLM was not loaded."
    )


def main() -> None:
    args = parse_args()
    files = discover_user_files(args.raw_dir, args.max_users)
    if args.dry_run:
        dry_run(files, args.k)
        return

    resume = load_resume_state(args.output)
    pending_paths = iter(files)
    pool: list[RecentUserState] = []

    def fill_pool() -> None:
        while len(pool) < args.batch_size:
            try:
                path = next(pending_paths)
            except StopIteration:
                break
            user_id = user_id_from_path(path)
            saved = resume.get(user_id, {"next_index": 0, "positive": False})
            if saved["positive"]:
                continue
            loaded = load_recent_user_state(path, int(saved["next_index"]))
            if loaded is not None:
                pool.append(loaded)

    llm, sampling_params = make_llm(args)
    tokenizer = llm.get_tokenizer()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fill_pool()
    inferred = positives = 0
    with args.output.open("a", encoding="utf-8") as output_handle:
        while pool:
            messages: list[list[dict[str, str]]] = []
            prompt_info: list[tuple[str, int, int]] = []
            for state in pool:
                conversation = state.conversations[state.next_index]
                update_history(state.history, conversation, state.user_id)
                selected = recent_history(state.history, args.k)
                fitted, strategy, input_tokens = build_fitted_recent_prompt(
                    selected, tokenizer, args.max_new_tokens, args.overflow_side
                )
                messages.append(fitted)
                prompt_info.append((strategy, input_tokens, len(selected)))

            decisions = infer_decisions(llm, sampling_params, messages, args.max_retries)
            records: list[dict[str, Any]] = []
            survivors: list[RecentUserState] = []
            for state, decision, (strategy, input_tokens, selected_count) in zip(
                pool, decisions, prompt_info, strict=True
            ):
                conversation = state.conversations[state.next_index]
                records.append(
                    {
                        "user_id": state.user_id,
                        "conversation_index": state.next_index,
                        "submission_id": submission_id(conversation),
                        "depression": decision,
                    }
                )
                inferred += 1
                print(
                    f"[{inferred}] {state.user_id} conversation={state.next_index} "
                    f"depression={int(decision)} input={strategy}:{input_tokens}tokens "
                    f"recent_nodes={selected_count}"
                )
                if decision:
                    positives += 1
                else:
                    state.next_index += 1
                    if state.next_index < len(state.conversations):
                        survivors.append(state)
            append_records(output_handle, records)
            pool[:] = survivors
            fill_pool()

    print(
        f"Done: inferred={inferred}, first_positives={positives}, k={args.k}, output={args.output}"
    )


if __name__ == "__main__":
    main()
