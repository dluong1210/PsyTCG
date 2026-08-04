"""Load the standalone LLM baseline settings from ``config/llm.yaml``."""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).with_name("llm.yaml")
with CONFIG_PATH.open(encoding="utf-8") as handle:
    SETTINGS = yaml.safe_load(handle) or {}

MODEL = str(SETTINGS["model"])
RAW_DIR = Path(SETTINGS["raw_dir"])
OUTPUT_DIR = Path(SETTINGS["output_dir"])
MAX_MODEL_LEN = int(SETTINGS["max_model_len"])
BATCH_SIZE = int(SETTINGS["batch_size"])
TENSOR_PARALLEL_SIZE = int(SETTINGS["tensor_parallel_size"])
GPU_MEMORY_UTILIZATION = float(SETTINGS["gpu_memory_utilization"])
DTYPE = str(SETTINGS["dtype"])
MAX_NEW_TOKENS = int(SETTINGS["max_new_tokens"])
MAX_RETRIES = int(SETTINGS["max_retries"])
TRUST_REMOTE_CODE = bool(SETTINGS["trust_remote_code"])
ENABLE_THINKING = bool(SETTINGS["enable_thinking"])
RECENT_K = int(SETTINGS["recent_k"])
OVERFLOW_SIDE = str(SETTINGS["overflow_side"])
SYSTEM_PROMPT = str(SETTINGS["system_prompt"])
CONVERSATION_PROMPT = str(SETTINGS["conversation_prompt"])
TIMELINE_PROMPT = str(SETTINGS["timeline_prompt"])
