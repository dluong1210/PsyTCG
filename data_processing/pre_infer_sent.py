"""Pre-inference sentence encoders used to featurise conversation-graph nodes.

Each sentence is mapped to three things:

* ``rel_hidden``  -- 1024-d contextual representation from the relevance branch;
* ``severity``    -- 21 symptom-severity scores;
* ``emotion``     -- 7 emotion probabilities.

A node (one post or comment) is therefore a variable-length sequence of
sentence vectors; the post encoder in :mod:`model.components` collapses it to a
single post embedding.

The relevance/severity model is a two-branch RoBERTa: the first 16 layers are
shared, the upper layers are branch-specific (relevance and severity). The
severity output is gated by relevance so that sentences the relevance head
considers irrelevant contribute no symptom signal.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn

# ``transformers`` is imported lazily: only graph construction needs it, so
# training, inference and the pure-python graph logic stay importable without it.

SENTENCE_DIM = 1024
N_SEVERITY = 21
N_EMOTION = 7
EMO_SEV_DIM = N_EMOTION + N_SEVERITY

_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[^.!?][.!?])\s+|(?<=[^.!?][.!?][\"'”’)\]])\s+")
_PLACEHOLDER = "X"


def split_sentences(text: str) -> List[str]:
    """Normalise Reddit markup and split into sentences.

    Never returns an empty list: a node whose body was deleted still has to
    occupy a graph position, so it falls back to a single placeholder sentence.
    """
    cleaned = (
        (text or "")
        .replace("[deleted]", "")
        .replace("[removed]", "")
        .replace("[deleted by user]", "")
        .replace("\n", " ")
        .strip()
    )
    cleaned = _MD_LINK_RE.sub(r"\1", cleaned)
    cleaned = re.sub(r"https://\S+", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"#\w+ ", "", cleaned)
    if not cleaned:
        return [_PLACEHOLDER]
    try:
        from nltk.tokenize import sent_tokenize

        sentences = [s.strip() for s in sent_tokenize(cleaned) if s.strip()]
    except Exception:  # nltk data unavailable -> regex fallback
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(cleaned) if s.strip()]
    sentences = [s for s in sentences if len(s) > 1]
    return sentences or [_PLACEHOLDER]


class RelevanceSeverityModel(nn.Module):
    """Two-branch RoBERTa producing sentence relevance and symptom severity."""

    def __init__(self, base_model: str, checkpoint: Path | str | None = None):
        from transformers import RobertaModel

        super().__init__()
        self.roberta_1 = RobertaModel.from_pretrained(base_model)
        self.roberta_2 = RobertaModel.from_pretrained(base_model)
        self.pooler_1 = self.roberta_1.pooler
        self.pooler_2 = self.roberta_2.pooler
        self.classifier_1 = nn.Sequential(nn.Dropout(0.1), nn.Linear(SENTENCE_DIM, N_SEVERITY))
        self.classifier_2 = nn.Sequential(nn.Dropout(0.1), nn.Linear(SENTENCE_DIM, N_SEVERITY))
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            report = self.load_state_dict(self._remap(state), strict=False)
            if report.missing_keys:
                raise RuntimeError(
                    f"relevance/severity checkpoint is incomplete: "
                    f"{len(report.missing_keys)} missing keys, e.g. {report.missing_keys[:5]}"
                )

    @staticmethod
    def _remap(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Expand the shared/split layer layout of the training checkpoint."""
        mapped: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            key = key.removeprefix("module.").removeprefix("model.")
            if key.startswith("embeddings."):
                mapped[f"roberta_1.{key}"] = value
                mapped[f"roberta_2.{key}"] = value
            elif key.startswith("first_half."):
                parts = key.split(".")
                layer, rest = int(parts[1]), ".".join(parts[2:])
                mapped[f"roberta_1.encoder.layer.{layer}.{rest}"] = value
                mapped[f"roberta_2.encoder.layer.{layer}.{rest}"] = value
            elif key.startswith(("second_half_1.", "second_half_2.")):
                branch = key[len("second_half_")]
                parts = key.split(".")
                layer, rest = int(parts[1]) + 16, ".".join(parts[2:])
                mapped[f"roberta_{branch}.encoder.layer.{layer}.{rest}"] = value
            elif key.startswith(("pooler_1.", "pooler_2.")):
                branch = key[len("pooler_")]
                mapped[f"roberta_{branch}.pooler.{key[len('pooler_1.') :]}"] = value
                mapped[key] = value
            elif key.startswith(("classifier_1.", "classifier_2.")):
                mapped[key] = value
        return mapped

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden_1 = self.roberta_1(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        ).last_hidden_state
        pooled_1 = self.pooler_1(hidden_1)
        relevance = torch.sigmoid(
            self.classifier_1(pooled_1 if pooled_1 is not None else hidden_1[:, 0])
        )

        hidden_2 = self.roberta_2(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        ).last_hidden_state
        pooled_2 = self.pooler_2(hidden_2)
        severity = torch.sigmoid(
            self.classifier_2(pooled_2 if pooled_2 is not None else hidden_2[:, 0])
        )
        # [S, 1024], [S, 21], [S]
        return hidden_1[:, 0], severity, relevance.max(dim=-1).values


class SentenceEncoder:
    """Encodes lists of sentences into per-node feature tensors."""

    def __init__(
        self,
        base_model: str,
        rel_sev_checkpoint: Path | str,
        emotion_model_dir: Path | str,
        device: torch.device,
        batch_size: int = 512,
        max_length: int = 128,
    ):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.rel_sev = RelevanceSeverityModel(base_model, rel_sev_checkpoint).to(device).eval()
        self.emotion_tokenizer = AutoTokenizer.from_pretrained(
            str(emotion_model_dir), local_files_only=True
        )
        self.emotion_model = (
            AutoModelForSequenceClassification.from_pretrained(
                str(emotion_model_dir), local_files_only=True
            )
            .to(device)
            .eval()
        )

    @torch.no_grad()
    def encode_nodes(
        self, sentence_lists: Sequence[Sequence[str]]
    ) -> List[Dict[str, torch.Tensor]]:
        """Encode several nodes in one pass.

        Returns one dict per node with keys ``rel_hidden`` [S, 1024],
        ``emotion`` [S, 7], ``severity`` [S, 21] and ``relevance`` [S], all on
        CPU. Relevance gating is deliberately *not* applied here so that the
        same encoding can be reused at any threshold.
        """
        flat: List[str] = []
        owner: List[int] = []
        for index, sentences in enumerate(sentence_lists):
            flat.extend(sentences)
            owner.extend([index] * len(sentences))

        if not flat:
            return [_empty_node() for _ in sentence_lists]

        rel_parts, sev_parts, rel_max_parts, emo_parts = [], [], [], []
        for start in range(0, len(flat), self.batch_size):
            batch = flat[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            rel_hidden, severity, relevance = self.rel_sev(
                encoded["input_ids"].to(self.device),
                encoded["attention_mask"].to(self.device),
            )
            emotion_input = self.emotion_tokenizer(
                batch,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            emotion_logits = self.emotion_model(
                input_ids=emotion_input["input_ids"].to(self.device),
                attention_mask=emotion_input["attention_mask"].to(self.device),
            ).logits
            rel_parts.append(rel_hidden.float().cpu())
            sev_parts.append(severity.float().cpu())
            rel_max_parts.append(relevance.float().cpu())
            emo_parts.append(torch.sigmoid(emotion_logits).float().cpu())

        rel_hidden = torch.cat(rel_parts)
        severity = torch.cat(sev_parts)
        relevance = torch.cat(rel_max_parts)
        emotion = torch.cat(emo_parts)

        offsets: Dict[int, List[int]] = {}
        for position, index in enumerate(owner):
            offsets.setdefault(index, []).append(position)

        nodes = []
        for index in range(len(sentence_lists)):
            rows = offsets.get(index, [])
            nodes.append(
                {
                    "rel_hidden": rel_hidden[rows],
                    "emotion": emotion[rows],
                    "severity": severity[rows],
                    "relevance": relevance[rows],
                }
            )
        return nodes


def _empty_node() -> Dict[str, torch.Tensor]:
    return {
        "rel_hidden": torch.empty(0, SENTENCE_DIM),
        "emotion": torch.empty(0, N_EMOTION),
        "severity": torch.empty(0, N_SEVERITY),
        "relevance": torch.empty(0),
    }


def gate_features(node: Dict[str, torch.Tensor], threshold: float) -> Dict[str, torch.Tensor]:
    """Apply relevance gating and pack the graph-ready node features.

    ``emo_sev`` concatenates emotion probabilities with severity scores that are
    zeroed for sentences whose relevance is at or below ``threshold``.
    """
    mask = (node["relevance"] > threshold).float().unsqueeze(-1)
    return {
        "rel_hidden": node["rel_hidden"],
        "emo_sev": torch.cat([node["emotion"], node["severity"] * mask], dim=-1),
    }


def is_relevant(node: Dict[str, torch.Tensor], threshold: float) -> bool:
    """True when any sentence of the node passes the relevance threshold."""
    return bool((node["relevance"] > threshold).any())
