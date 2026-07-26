"""
Lightweight LaBraM-style transformer classifier.

Architecture (mirrors the ViT/BEiT pattern LaBraM itself is built on):

    SEDAT tokens (B, K, L_tgt, C)
        -> flatten each token to (L_tgt * C,) and linearly embed -> (B, K, D)
        -> prepend a learnable [CLS] token                        -> (B, K+1, D)
        -> add learnable positional embeddings
        -> pre-norm Transformer encoder, `depth` layers
        -> final LayerNorm, take the [CLS] position
        -> linear classification head -> (B, num_classes)

The token-embedding step is the one deliberate departure from the official
LaBraM: LaBraM patchifies each channel independently and embeds per-channel
temporal patches. SEDAT already performs spatial aggregation and produces
one token spanning all channels per time segment, so each SEDAT token is
embedded as a single flattened vector instead. Everything downstream (CLS
token, positional embeddings, pre-norm Transformer blocks, linear head) is
the standard construction LaBraM uses.
"""

from __future__ import annotations

import torch
from torch import nn

from labram.config import LaBraMConfig


class LaBraMStyleClassifier(nn.Module):
    def __init__(self, config: LaBraMConfig) -> None:
        super().__init__()
        self.config = config
        patch_dim = config.token_length * config.num_channels

        self.token_embed = nn.Linear(patch_dim, config.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, config.num_tokens + 1, config.embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=int(config.embed_dim * config.mlp_ratio),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens:
            SEDAT token tensor of shape ``(B, K, L_tgt, C)``.

        Returns
        -------
        torch.Tensor
            Class logits of shape ``(B, num_classes)``.
        """
        b = tokens.shape[0]
        flat = tokens.reshape(b, self.config.num_tokens, -1)
        embedded = self.token_embed(flat)

        cls = self.cls_token.expand(b, -1, -1)
        sequence = torch.cat([cls, embedded], dim=1) + self.pos_embed

        encoded = self.encoder(sequence)
        cls_out = self.norm(encoded[:, 0])
        return self.head(cls_out)
