"""
A lightweight, self-contained LaBraM-style transformer classifier used to
demonstrate that SEDAT's token tensors integrate cleanly with a large-EEG
foundation-model backbone.

This is **not** the official LaBraM codebase or its pretrained weights
(Jiang, Zhao & Lu, 2024, arXiv:2405.18765) — it is a from-scratch
reimplementation of LaBraM's architectural pattern (learned token
embedding, a CLS token, learnable positional embeddings, and a standard
pre-norm Transformer encoder with a linear classification head),
deliberately kept small so it trains from random initialization on CPU in
minutes. It exists to validate that SEDAT's ``(K, L_tgt, C)`` output tensor
is a drop-in replacement for a foundation model's native tokenizer, and to
produce a real (if modest) classification result on real EEG data.
"""

from labram.config import LaBraMConfig
from labram.model import LaBraMStyleClassifier

__all__ = ["LaBraMConfig", "LaBraMStyleClassifier"]
