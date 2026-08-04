"""User-level ground-truth loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

_TRUE = {"1", "true", "yes", "positive", "pos"}
_FALSE = {"0", "false", "no", "negative", "neg", "control"}


def _binary(value: object) -> int:
    text = str(value).strip().lower()
    if text in _TRUE:
        return 1
    if text in _FALSE:
        return 0
    return int(float(text))


def load_labels(path: Path | str) -> Dict[str, int]:
    """Read ``{user_id: label}`` from a whitespace-separated ``.txt`` or a ``.json``.

    The eRisk collections ship the golden truth as ``<user_id> <label>`` lines;
    ``.json`` files holding the same mapping are accepted as well.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"label file not found: {path}")

    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        labels = {str(uid): _binary(label) for uid, label in payload.items()}
    else:
        labels = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) == 2:
                    labels[str(parts[0])] = _binary(parts[1])

    if not labels:
        raise ValueError(f"no labels parsed from {path}")
    return labels
