"""Configuration validation guards for public command-line runs."""

from __future__ import annotations

import pytest

from config.schema import DataConfig, ModelConfig, TrainConfig


def data_config(**overrides):
    values = {
        "train_timelines": "train.pkl",
        "test_timelines": "test.pkl",
        "train_labels": "train.txt",
        "test_labels": "test.txt",
    }
    values.update(overrides)
    return DataConfig(**values)


def test_window_stride_defaults_to_window_size():
    config = data_config(max_conversations=12)
    assert config.window_stride == 12


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.1])
def test_validation_fraction_must_be_internal(fraction):
    with pytest.raises(ValueError):
        data_config(val_fraction=fraction)


def test_attention_dimensions_must_align():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(memory_dim=255, gat_heads=8)


def test_training_requires_a_seed():
    with pytest.raises(ValueError, match="seed"):
        TrainConfig(seeds=[])
