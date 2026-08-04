"""Load graph-construction defaults from ``config/data_processing.yaml``."""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).with_name("data_processing.yaml")
with CONFIG_PATH.open(encoding="utf-8") as handle:
    SETTINGS = yaml.safe_load(handle) or {}

BASE_MODEL = str(SETTINGS["base_model"])
REL_SEV_CHECKPOINT = Path(SETTINGS["rel_sev_checkpoint"])
EMOTION_MODEL_DIR = Path(SETTINGS["emotion_model_dir"])
RELEVANCE_THRESHOLD = float(SETTINGS["relevance_threshold"])
BATCH_SIZE = int(SETTINGS["batch_size"])
MAX_LENGTH = int(SETTINGS["max_length"])
