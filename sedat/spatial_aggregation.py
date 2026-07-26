"""
SEDAT Step 1 — Squeeze-and-Excitation (SE) spatial aggregation.

Reduces a multi-channel EEG trial to a single 1D "drive signal" by adaptively
weighting channels according to their temporal energy (variance), following
Eqs. (1)-(5) of the SEDAT manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sedat.config import SpatialAggregationConfig


@dataclass(frozen=True)
class SpatialAggregationResult:
    """Output of the SE spatial aggregation stage."""

    drive_signal: np.ndarray
    """1D array of shape (T,) — the aggregated drive signal s(t)."""

    channel_weights: np.ndarray
    """1D array of shape (C,) — the L1-normalized channel weights omega_c."""

    channel_energy: np.ndarray
    """1D array of shape (C,) — raw per-channel energy descriptors E_c."""


class SpatialAggregator:
    """Implements the SE-based spatial aggregation described in Section 2.1.1."""

    def __init__(self, config: SpatialAggregationConfig | None = None) -> None:
        self.config = config or SpatialAggregationConfig()

    def aggregate(self, x: np.ndarray) -> SpatialAggregationResult:
        """Reduce ``x`` of shape (T, C) to a single drive signal of shape (T,).

        For single-channel input, or when the stage is disabled via
        configuration, aggregation is bypassed and the (single) channel is
        returned unchanged, matching the SEDAT experimental protocol for
        single-channel datasets (Section 2.4).
        """
        t, c = x.shape

        if c == 1 or not self.config.enabled:
            weights = np.ones(c, dtype=np.float64) / c
            drive = x @ weights
            energy = np.var(x, axis=0)
            return SpatialAggregationResult(drive, weights, energy)

        # Eq. (1): second central moment (population variance) per channel.
        energy = np.var(x, axis=0)

        # Eq. (2): min-max normalization across channels.
        e_inf, e_sup = energy.min(), energy.max()
        z = (energy - e_inf) / (e_sup - e_inf + self.config.eps)

        # Eq. (3): sigmoid activation with temperature gamma and shift delta.
        phi = 1.0 / (1.0 + np.exp(-self.config.gamma * (z - self.config.delta)))

        # Eq. (4): L1 normalization to a valid probability simplex.
        weights = phi / phi.sum()

        # Eq. (5): weighted aggregation across channels.
        drive = x @ weights

        return SpatialAggregationResult(drive, weights, energy)
