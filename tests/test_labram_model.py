"""Fast sanity tests for the LaBraM-style classifier. No real training —
just shape, gradient-flow, and determinism checks."""

import os

import numpy as np
import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

torch = pytest.importorskip("torch")

from labram.config import LaBraMConfig
from labram.model import LaBraMStyleClassifier


def _make_model(num_classes=2):
    config = LaBraMConfig(
        num_tokens=5, token_length=32, num_channels=8, num_classes=num_classes,
        embed_dim=16, depth=2, num_heads=2,
    )
    return config, LaBraMStyleClassifier(config)


def test_output_shape():
    config, model = _make_model()
    x = torch.randn(4, config.num_tokens, config.token_length, config.num_channels)
    logits = model(x)
    assert logits.shape == (4, config.num_classes)


def test_gradients_flow_to_embedding_and_head():
    config, model = _make_model()
    x = torch.randn(3, config.num_tokens, config.token_length, config.num_channels)
    logits = model(x)
    logits.sum().backward()
    assert model.token_embed.weight.grad is not None
    assert torch.any(model.token_embed.weight.grad != 0)
    assert model.head.weight.grad is not None


def test_config_rejects_incompatible_heads():
    with pytest.raises(ValueError):
        LaBraMConfig(num_tokens=5, token_length=32, num_channels=8, embed_dim=15, num_heads=4)


def test_single_sample_batch():
    config, model = _make_model()
    model.eval()
    x = torch.randn(1, config.num_tokens, config.token_length, config.num_channels)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, config.num_classes)


def test_multiclass_head_size():
    config, model = _make_model(num_classes=4)
    x = torch.randn(2, config.num_tokens, config.token_length, config.num_channels)
    logits = model(x)
    assert logits.shape == (2, 4)


def test_deterministic_in_eval_mode():
    config, model = _make_model()
    model.eval()
    x = torch.randn(2, config.num_tokens, config.token_length, config.num_channels)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    torch.testing.assert_close(out1, out2)
