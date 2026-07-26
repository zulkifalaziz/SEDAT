"""Configuration for the LaBraM-style classifier."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LaBraMConfig:
    """Architecture hyperparameters for :class:`labram.model.LaBraMStyleClassifier`.

    Defaults are deliberately small: with only a few hundred labeled EEG
    epochs available, a large transformer would simply memorize the
    training set. Sizing follows the standard ViT/BEiT pattern (embed_dim
    divisible by num_heads) scaled down for this data regime.
    """

    num_tokens: int
    """K — number of SEDAT tokens per trial (sequence length before CLS)."""

    token_length: int
    """L_tgt — number of resampled temporal samples per SEDAT token."""

    num_channels: int
    """C — number of EEG channels."""

    num_classes: int = 2

    embed_dim: int = 64
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if self.num_tokens < 1 or self.token_length < 1 or self.num_channels < 1:
            raise ValueError("num_tokens, token_length, and num_channels must all be >= 1.")
