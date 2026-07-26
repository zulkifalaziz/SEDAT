"""
SEDAT Step 4 — Fourier-domain resampling and token formation.

Partitions the raw multi-channel EEG trial according to the boundary set
produced by adaptive segmentation, selects the K longest segments (padding
with zero segments if fewer than K survive pruning), resamples each to a
common target length in the frequency domain, and stacks the result into a
fixed-dimensional tensor Z in R^{K x L_tgt x C}, following Eqs. (19)-(20).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import resample

from sedat.config import ResamplingConfig


@dataclass(frozen=True)
class ResamplingResult:
    """Output of the resampling / token formation stage."""

    tokens: np.ndarray
    """Final tensor Z of shape (K, L_tgt, C)."""

    segment_lengths: np.ndarray
    """Lengths L_v of every raw (pre-selection) segment, for diagnostics."""

    selected_indices: np.ndarray
    """Indices (in chronological order) of the segments retained as tokens.
    A value of -1 marks a zero-padded placeholder token."""


class FourierResampler:
    """Implements resampling and token formation described in Section 2.1.4."""

    def __init__(self, config: ResamplingConfig | None = None) -> None:
        self.config = config or ResamplingConfig()

    def tokenize(
        self,
        x: np.ndarray,
        boundaries: np.ndarray,
        num_tokens: int,
        target_length: int | None = None,
    ) -> ResamplingResult:
        """Build the final token tensor for one trial.

        Parameters
        ----------
        x:
            Raw multi-channel EEG matrix of shape (T, C).
        boundaries:
            Sorted boundary indices from :class:`~sedat.segmentation.AdaptiveSegmenter`.
        num_tokens:
            Target token count K.
        target_length:
            Fixed token length L_tgt. If None, ``self.config.target_length``
            is used; if that is also None, the caller must resolve a value
            upstream (the :class:`~sedat.tokenizer.SEDATTokenizer` does this
            automatically based on observed segment lengths).
        """
        l_tgt = target_length if target_length is not None else self.config.target_length
        if l_tgt is None:
            raise ValueError(
                "target_length must be resolved before calling FourierResampler.tokenize; "
                "SEDATTokenizer resolves this automatically."
            )

        n_channels = x.shape[1]
        segment_bounds = list(zip(boundaries[:-1], boundaries[1:]))
        lengths = np.array([hi - lo for lo, hi in segment_bounds], dtype=np.int64)

        # Rank segments by length (descending) and keep the top `num_tokens`.
        order = np.argsort(lengths)[::-1]
        top = order[:num_tokens]
        # Restore chronological order so the token sequence remains
        # temporally meaningful.
        top_sorted = np.sort(top)

        tokens = np.zeros((num_tokens, l_tgt, n_channels), dtype=np.float64)
        selected_indices = np.full(num_tokens, -1, dtype=np.int64)

        for slot, seg_idx in enumerate(top_sorted):
            lo, hi = segment_bounds[seg_idx]
            segment = x[lo:hi, :]
            tokens[slot] = self._resample_segment(segment, l_tgt)
            selected_indices[slot] = seg_idx

        return ResamplingResult(
            tokens=tokens, segment_lengths=lengths, selected_indices=selected_indices
        )

    @staticmethod
    def _resample_segment(segment: np.ndarray, l_tgt: int) -> np.ndarray:
        """FFT-domain resampling to a fixed length, Eq. (19).

        ``scipy.signal.resample`` implements exactly the operation
        described by the manuscript: the segment's spectrum is either
        symmetrically truncated (downsampling) or zero-padded at the
        Nyquist bin (upsampling) before the inverse transform. Both
        temporal axis and all channels are resampled in a single vectorized
        FFT call.
        """
        if segment.shape[0] == l_tgt:
            return segment.astype(np.float64, copy=True)
        return resample(segment, l_tgt, axis=0)
