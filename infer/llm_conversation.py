#!/usr/bin/env python3
"""Infer depression from one pruned conversation at a time with vLLM.

For each target user, conversations are processed in raw-file order.  A user is
removed from the inference pool immediately after the first positive decision.
The primary pruning strategy keeps every root-to-target path and every direct
child of a target node.  If that prompt is too long, it falls back to target
nodes plus their direct parents/children, then to a character-prefix truncation.

Output is append-only JSONL and can be resumed safely.  Each inferred
conversation produces exactly one boolean decision plus its identifiers.
"""

from __future__ import annotations

import argparse
import atexit
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from config.llm import (
    BATCH_SIZE,
    CONVERSATION_PROMPT,
    DTYPE,
    ENABLE_THINKING,
    GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN,
    MAX_NEW_TOKENS,
    MAX_RETRIES,
    MODEL,
    OUTPUT_DIR,
    RAW_DIR,
    SYSTEM_PROMPT,
    TENSOR_PARALLEL_SIZE,
    TRUST_REMOTE_CODE,
)

# Qwen3 enables reasoning by default.  This task needs one classification token,
# so thinking must be disabled both for token counting and for vLLM rendering.
CHAT_TEMPLATE_KWARGS = {"enable_thinking": ENABLE_THINKING}
USER_PREFIX = CONVERSATION_PROMPT

DELETED_TEXT = {"", "[deleted]", "[removed]", "[deleted by user]"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "conversation.jsonl")
    parser.add_argument("--model", default=MODEL)
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
        "--trust-remote-code", action=argparse.BooleanOptionalAction, default=TRUST_REMOTE_CODE
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate/prune inputs without loading vLLM or writing output.",
    )
    args = parser.parse_args()
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
    return args


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def clean_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in DELETED_TEXT:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def node_id(node: dict[str, Any]) -> str:
    if node.get("type") == "submission":
        return str(node.get("submission_id") or "")
    return str(node.get("comment_id") or "")


def is_target_node(node: dict[str, Any], user_id: str | None = None) -> bool:
    if str(node.get("target")).strip().lower() == "true":
        return True
    return bool(user_id and str(node.get("user_id") or "") == user_id)


def conversation_nodes(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    submission = conversation.get("submission")
    if not isinstance(submission, dict):
        raise ValueError("conversation is missing its submission object")
    comments = conversation.get("comments") or []
    if not isinstance(comments, list):
        raise ValueError("conversation.comments must be a list")
    return [submission, *(item for item in comments if isinstance(item, dict))]


def node_sort_key(node: dict[str, Any]) -> tuple[str, str]:
    return (str(node.get("created_utc") or ""), node_id(node))


def _node_lookups(
    conversation: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[str]]]:
    nodes = conversation_nodes(conversation)
    by_id = {node_id(node): node for node in nodes if node_id(node)}
    children: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        current_id = node_id(node)
        parent_id = node.get("parent_id")
        if current_id and parent_id is not None:
            children[str(parent_id)].append(current_id)
    for child_ids in children.values():
        child_ids.sort(key=lambda item: node_sort_key(by_id[item]))
    return nodes, by_id, children


def select_root_paths_and_target_children(
    conversation: dict[str, Any], user_id: str
) -> list[dict[str, Any]]:
    """Keep target nodes, all available ancestors, and target direct children."""
    nodes, by_id, children = _node_lookups(conversation)
    targets = [node for node in nodes if is_target_node(node, user_id)]
    if not targets:
        raise ValueError(f"conversation has no target node for {user_id}")

    selected: set[str] = set()
    for target in targets:
        current: dict[str, Any] | None = target
        visited: set[str] = set()
        while current is not None:
            current_id = node_id(current)
            if not current_id or current_id in visited:
                break
            visited.add(current_id)
            selected.add(current_id)
            parent_id = current.get("parent_id")
            current = by_id.get(str(parent_id)) if parent_id is not None else None
        selected.update(children.get(node_id(target), []))
    return [node for node in nodes if node_id(node) in selected]


def select_target_one_hop(conversation: dict[str, Any], user_id: str) -> list[dict[str, Any]]:
    """Keep target nodes and their direct parents and children."""
    nodes, by_id, children = _node_lookups(conversation)
    targets = [node for node in nodes if is_target_node(node, user_id)]
    if not targets:
        raise ValueError(f"conversation has no target node for {user_id}")

    selected: set[str] = set()
    for target in targets:
        target_id = node_id(target)
        selected.add(target_id)
        parent_id = target.get("parent_id")
        if parent_id is not None and str(parent_id) in by_id:
            selected.add(str(parent_id))
        selected.update(children.get(target_id, []))
    return [node for node in nodes if node_id(node) in selected]


def _render_node(node: dict[str, Any], user_id: str, depth: int, parent_kept: bool) -> str:
    role = "TARGET" if is_target_node(node, user_id) else "OTHER"
    kind = "POST" if node.get("type") == "submission" else "COMMENT"
    fields = [role, kind, f"id={node_id(node)}"]
    if node.get("created_utc"):
        fields.append(f"time={node['created_utc']}")
    parent_id = node.get("parent_id")
    if parent_id is not None and not parent_kept:
        fields.append(f"reply_to={parent_id}")
    title = clean_text(node.get("title"))
    body = clean_text(node.get("body"))
    text = " — ".join(part for part in (title, body) if part) or "."
    return f"{'  ' * depth}- [{' | '.join(fields)}] {text}"


def render_selected_nodes(
    conversation: dict[str, Any], selected_nodes: Sequence[dict[str, Any]], user_id: str
) -> str:
    """Render selected nodes as a reply tree, followed by any disconnected roots."""
    selected = {node_id(node): node for node in selected_nodes if node_id(node)}
    children: dict[str, list[str]] = defaultdict(list)
    roots: list[str] = []
    for current_id, node in selected.items():
        parent_id = str(node.get("parent_id")) if node.get("parent_id") is not None else None
        if parent_id and parent_id in selected:
            children[parent_id].append(current_id)
        else:
            roots.append(current_id)
    for ids in children.values():
        ids.sort(key=lambda item: node_sort_key(selected[item]))
    roots.sort(
        key=lambda item: (
            0 if selected[item].get("type") == "submission" else 1,
            node_sort_key(selected[item]),
        )
    )

    lines: list[str] = []
    visited: set[str] = set()

    def visit(current_id: str, depth: int) -> None:
        if current_id in visited:
            return
        visited.add(current_id)
        node = selected[current_id]
        parent_id = str(node.get("parent_id")) if node.get("parent_id") is not None else ""
        lines.append(_render_node(node, user_id, depth, parent_id in selected))
        for child_id in children.get(current_id, []):
            visit(child_id, depth + 1)

    for root_id in roots:
        visit(root_id, 0)
    for current_id in sorted(
        set(selected) - visited, key=lambda item: node_sort_key(selected[item])
    ):
        visit(current_id, 0)
    return "\n".join(lines)


def make_messages(content: str, user_prefix: str = USER_PREFIX) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prefix + content},
    ]


def chat_token_count(tokenizer: Any, messages: Sequence[dict[str, str]]) -> int:
    token_ids = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        **CHAT_TEMPLATE_KWARGS,
    )
    if hasattr(token_ids, "numel"):
        return int(token_ids.numel())
    if isinstance(token_ids, dict):
        token_ids = token_ids["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return len(token_ids)


def hard_fit_content(
    content: str,
    tokenizer: Any,
    max_input_tokens: int,
    *,
    keep: str = "prefix",
    user_prefix: str = USER_PREFIX,
) -> tuple[str, int]:
    """Keep the longest fitting character prefix/suffix of content."""
    if keep not in {"prefix", "suffix"}:
        raise ValueError("keep must be 'prefix' or 'suffix'")
    empty_tokens = chat_token_count(tokenizer, make_messages("", user_prefix))
    if empty_tokens > max_input_tokens:
        raise ValueError(
            f"Prompt instructions alone use {empty_tokens} tokens, over limit {max_input_tokens}"
        )

    def candidate(length: int) -> str:
        if length <= 0:
            return ""
        return content[:length] if keep == "prefix" else content[-length:]

    low, high = 0, len(content)
    best_text, best_tokens = "", empty_tokens
    while low <= high:
        middle = (low + high) // 2
        text = candidate(middle)
        count = chat_token_count(tokenizer, make_messages(text, user_prefix))
        if count <= max_input_tokens:
            best_text, best_tokens = text, count
            low = middle + 1
        else:
            high = middle - 1
    while (
        best_text
        and chat_token_count(tokenizer, make_messages(best_text, user_prefix)) > max_input_tokens
    ):
        best_text = best_text[:-1] if keep == "prefix" else best_text[1:]
    best_tokens = chat_token_count(tokenizer, make_messages(best_text, user_prefix))
    return best_text, best_tokens


def build_fitted_pruned_prompt(
    conversation: dict[str, Any],
    user_id: str,
    tokenizer: Any,
    max_new_tokens: int,
) -> tuple[list[dict[str, str]], str, int]:
    max_input_tokens = MAX_MODEL_LEN - max_new_tokens

    primary_nodes = select_root_paths_and_target_children(conversation, user_id)
    primary_text = render_selected_nodes(conversation, primary_nodes, user_id)
    primary_messages = make_messages(primary_text)
    count = chat_token_count(tokenizer, primary_messages)
    if count <= max_input_tokens:
        return primary_messages, "root_paths", count

    fallback_nodes = select_target_one_hop(conversation, user_id)
    fallback_text = render_selected_nodes(conversation, fallback_nodes, user_id)
    fallback_messages = make_messages(fallback_text)
    count = chat_token_count(tokenizer, fallback_messages)
    if count <= max_input_tokens:
        return fallback_messages, "target_1hop", count

    fitted_text, count = hard_fit_content(fallback_text, tokenizer, max_input_tokens, keep="prefix")
    return make_messages(fitted_text), "target_1hop_char_prefix", count


def make_llm(args: argparse.Namespace) -> tuple[Any, Any]:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        # A forced one-token choice cannot be cut off like a JSON object can.
        structured_outputs=StructuredOutputsParams(choice=["0", "1"]),
    )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.batch_size,
        enable_prefix_caching=True,
        trust_remote_code=args.trust_remote_code,
    )
    atexit.register(cleanup_distributed)
    return llm, sampling_params


def parse_depression_response(response: Any) -> bool:
    output = response.outputs[0]
    text = output.text.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    if text == "1":
        return True
    if text == "0":
        return False

    # Backward-compatible parsing for responses from the former JSON schema.
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    decision = parsed.get("depression") if isinstance(parsed, dict) else None
    if isinstance(decision, bool):
        return decision

    finish_reason = getattr(output, "finish_reason", None)
    stop_reason = getattr(output, "stop_reason", None)
    token_ids = getattr(output, "token_ids", None) or []
    raise ValueError(
        "expected structured choice 0/1; "
        f"finish_reason={finish_reason!r}, stop_reason={stop_reason!r}, "
        f"generated_tokens={len(token_ids)}, raw={text!r}"
    )


class BatchInferenceError(RuntimeError):
    """Inference failed after some prompts in the batch may have succeeded."""

    def __init__(self, message: str, decisions: list[bool | None]):
        super().__init__(message)
        self.decisions = decisions


def infer_decisions(
    llm: Any,
    sampling_params: Any,
    message_batches: Sequence[Sequence[dict[str, str]]],
    max_retries: int,
) -> list[bool]:
    decisions: list[bool | None] = [None] * len(message_batches)
    pending = list(range(len(message_batches)))
    last_errors: dict[int, str] = {}
    for attempt in range(1, max_retries + 1):
        if not pending:
            break
        current = pending
        pending = []
        try:
            responses = llm.chat(
                [message_batches[index] for index in current],
                sampling_params=sampling_params,
                chat_template_kwargs=CHAT_TEMPLATE_KWARGS,
                use_tqdm=False,
            )
        except Exception as exc:
            if attempt == max_retries:
                raise BatchInferenceError(
                    f"vLLM batch failed after {max_retries} attempts: {exc}",
                    decisions,
                ) from exc
            print(f"[WARN] vLLM batch attempt {attempt}/{max_retries} failed: {exc}")
            pending = current
            continue
        if len(responses) != len(current):
            raise BatchInferenceError(
                f"vLLM returned {len(responses)} responses for {len(current)} prompts",
                decisions,
            )
        for index, response in zip(current, responses, strict=True):
            try:
                decisions[index] = parse_depression_response(response)
            except Exception as exc:
                last_errors[index] = str(exc)
                pending.append(index)
    if pending:
        details = "; ".join(
            f"prompt {index}: {last_errors.get(index, 'parse failed')}" for index in pending
        )
        raise BatchInferenceError(
            f"Could not parse structured vLLM output after {max_retries} attempts: {details}",
            decisions,
        )
    return [bool(item) for item in decisions]


def user_id_from_path(path: Path) -> str:
    return path.stem


def submission_id(conversation: dict[str, Any]) -> str:
    submission = conversation.get("submission") or {}
    return str(submission.get("submission_id") or "")


def load_resume_state(output_path: Path) -> dict[str, dict[str, Any]]:
    state: dict[str, dict[str, Any]] = {}
    if not output_path.is_file():
        return state

    raw_lines = output_path.read_bytes().splitlines(keepends=True)
    valid_bytes = 0
    record_count = 0
    for line_number, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.strip():
            valid_bytes += len(raw_line)
            continue
        try:
            line = raw_line.decode("utf-8")
            record = json.loads(line)
            user_id = str(record["user_id"])
            index = int(record["conversation_index"])
            decision = record["depression"]
            if not isinstance(decision, bool):
                raise ValueError("depression is not boolean")
        except Exception as exc:
            if line_number == len(raw_lines):
                # A process killed during write may leave one partial tail line.
                # Remove only that unparseable tail so future appends stay valid.
                with output_path.open("r+b") as handle:
                    handle.truncate(valid_bytes)
                print(
                    f"[WARN] Removed incomplete final JSONL line {line_number} "
                    f"from {output_path}: {exc}"
                )
                break
            raise ValueError(f"Invalid JSONL record at {output_path}:{line_number}") from exc
        item = state.setdefault(user_id, {"next_index": 0, "positive": False})
        item["next_index"] = max(int(item["next_index"]), index + 1)
        item["positive"] = bool(item["positive"] or decision)
        record_count += 1
        valid_bytes += len(raw_line)
    print(
        f"[RESUME] Loaded {record_count} completed inferences for "
        f"{len(state)} users from {output_path}"
    )
    return state


@dataclass
class UserState:
    user_id: str
    path: Path
    conversations: list[dict[str, Any]]
    next_index: int = 0


def load_user_state(path: Path, next_index: int = 0) -> UserState | None:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Raw file must contain a list: {path}")
    if next_index >= len(data):
        return None
    return UserState(
        user_id=user_id_from_path(path), path=path, conversations=data, next_index=next_index
    )


def discover_user_files(raw_dir: Path, max_users: int | None = None) -> list[Path]:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
    files = sorted(raw_dir.glob("subject_*.json"))
    if not files:
        raise FileNotFoundError(f"No subject_*.json files found in {raw_dir}")
    return files[:max_users] if max_users is not None else files


def dry_run(files: Sequence[Path]) -> None:
    conversation_count = target_count = 0
    primary_node_count = fallback_node_count = 0
    for path in files:
        user_id = user_id_from_path(path)
        data = load_json(path)
        if not isinstance(data, list):
            raise ValueError(f"Raw file must contain a list: {path}")
        conversation_count += len(data)
        for conversation in data:
            targets = [
                node for node in conversation_nodes(conversation) if is_target_node(node, user_id)
            ]
            if not targets:
                raise ValueError(f"No target node in {path}")
            target_count += len(targets)
            primary_node_count += len(select_root_paths_and_target_children(conversation, user_id))
            fallback_node_count += len(select_target_one_hop(conversation, user_id))
    print(
        f"Dry run OK: users={len(files)}, conversations={conversation_count}, "
        f"target_nodes={target_count}, primary_nodes={primary_node_count}, "
        f"fallback_nodes={fallback_node_count}. vLLM was not loaded."
    )


def append_records(handle: Any, records: Iterable[dict[str, Any]]) -> None:
    for record in records:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()


def cleanup_distributed() -> None:
    """Avoid an NCCL process-group leak warning on normal/error exits."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    files = discover_user_files(args.raw_dir, args.max_users)
    if args.dry_run:
        dry_run(files)
        return

    resume = load_resume_state(args.output)
    pending_paths = iter(files)
    pool: list[UserState] = []

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
            loaded = load_user_state(path, int(saved["next_index"]))
            if loaded is not None:
                pool.append(loaded)

    llm, sampling_params = make_llm(args)
    tokenizer = llm.get_tokenizer()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fill_pool()
    inferred = sum(int(item["next_index"]) for item in resume.values())
    positives = sum(bool(item["positive"]) for item in resume.values())
    with args.output.open("a", encoding="utf-8") as output_handle:
        while pool:
            messages: list[list[dict[str, str]]] = []
            prompt_info: list[tuple[str, int]] = []
            for state in pool:
                conversation = state.conversations[state.next_index]
                fitted, strategy, input_tokens = build_fitted_pruned_prompt(
                    conversation, state.user_id, tokenizer, args.max_new_tokens
                )
                messages.append(fitted)
                prompt_info.append((strategy, input_tokens))

            batch_error: BatchInferenceError | None = None
            try:
                decisions: list[bool | None] = infer_decisions(
                    llm, sampling_params, messages, args.max_retries
                )
            except BatchInferenceError as exc:
                # Persist every valid response from this batch. On restart, only
                # the unresolved user/conversation will be inferred again.
                decisions = exc.decisions
                batch_error = exc
            records: list[dict[str, Any]] = []
            survivors: list[UserState] = []
            failed_items: list[str] = []
            for state, decision, (strategy, input_tokens) in zip(
                pool, decisions, prompt_info, strict=True
            ):
                conversation = state.conversations[state.next_index]
                if decision is None:
                    failed_items.append(f"{state.user_id}:conversation={state.next_index}")
                    continue
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
                    f"depression={int(decision)} input={strategy}:{input_tokens}tokens"
                )
                if decision:
                    positives += 1
                else:
                    state.next_index += 1
                    if state.next_index < len(state.conversations):
                        survivors.append(state)
            append_records(output_handle, records)
            if batch_error is not None:
                print(
                    f"[ERROR] Checkpointed {len(records)} successful responses from "
                    f"the failed batch. Re-run the same command to resume only: "
                    f"{', '.join(failed_items)}"
                )
                raise batch_error
            pool[:] = survivors
            fill_pool()

    print(f"Done: inferred={inferred}, first_positives={positives}, output={args.output}")


if __name__ == "__main__":
    main()
