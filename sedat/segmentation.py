"""
SEDAT Step 3 — Instantaneous-frequency-guided adaptive segmentation.

For each IMF produced by DAGAF, locates the single time index at which the
instantaneous frequency changes most abruptly (the point of greatest
oscillatory instability) via the analytic signal / Hilbert transform, then
merges these per-mode boundaries with the trial's start/end and prunes any
boundaries that are too close together, following Eqs. (12)-(18) of the
SEDAT manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from scipy.signal import hilbert

from sedat.config import SegmentationConfig


@dataclass(frozen=True)
class SegmentationResult:
    """Output of the adaptive segmentation stage."""

    boundaries: np.ndarray
    """Sorted 1D int array of length V+1 delimiting V final segments; always
    starts at 0 and ends at N (the trial length)."""

    raw_boundaries: np.ndarray
    """The unpruned boundary set B_raw, Eq. (17), kept for diagnostics."""

    per_imf_boundary: np.ndarray
    """1D int array of length K holding each IMF's boundary index p_k."""


def _imf_boundary(imf: np.ndarray, fs: float) -> tuple[int, float]:
    """Locate the single dominant temporal boundary of one IMF, Eqs. (12)-(16).

    Returns
    -------
    (index, dominance)
        ``index`` is the sample position of maximum instantaneous-frequency
        rate of change; ``dominance`` is that maximum rate value, used later
        to break ties during boundary pruning.
    """
    n = imf.shape[0]
    if n < 4:
        # Degenerate (too-short) IMF: fall back to the midpoint with zero
        # dominance so it is always pruned first if it conflicts.
        return n // 2, 0.0

    analytic = hilbert(imf)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = (fs / (2.0 * np.pi)) * np.diff(phase)  # Eq. (14), length N-1
    rate = np.abs(np.diff(inst_freq))  # Eq. (15), length N-2

    if rate.size == 0:
        return n // 2, 0.0

    idx = int(np.argmax(rate))
    return idx, float(rate[idx])


class AdaptiveSegmenter:
    """Implements the adaptive segmentation described in Section 2.1.3."""

    def __init__(self, config: SegmentationConfig | None = None) -> None:
        self.config = config or SegmentationConfig()

    def segment(self, imfs: Sequence[np.ndarray], n_samples: int, fs: float) -> SegmentationResult:
        """Derive the pruned boundary set for a trial of length ``n_samples``.

        Parameters
        ----------
        imfs:
            Ordered list of IMFs (highest to lowest frequency) as returned
            by :class:`~sedat.decomposition.DAGAFDecomposer`.
        n_samples:
            Total number of temporal samples in the trial (N).
        fs:
            Sampling rate in Hz, used to scale the instantaneous frequency.
        """
        per_imf_idx: List[int] = []
        dominance: dict[int, float] = {0: np.inf, n_samples: np.inf}

        for imf in imfs:
            idx, dom = _imf_boundary(imf, fs)
            idx = int(np.clip(idx, 1, n_samples - 1))
            per_imf_idx.append(idx)
            # If two IMFs happen to select the same boundary, keep the
            # stronger (more dominant) rate-of-change value.
            dominance[idx] = max(dominance.get(idx, -np.inf), dom)

        raw_boundaries = np.array(
            sorted({0, n_samples, *per_imf_idx}), dtype=np.int64
        )

        pruned = self._prune(raw_boundaries, dominance)

        return SegmentationResult(
            boundaries=pruned,
            raw_boundaries=raw_boundaries,
            per_imf_boundary=np.array(per_imf_idx, dtype=np.int64),
        )

    def _prune(self, boundaries: np.ndarray, dominance: dict[int, float]) -> np.ndarray:
        """Iteratively remove boundaries violating the minimum segment
        length constraint, Eq. (18), always dropping the less dominant of
        the two boundaries forming an offending (too-short) interval. The
        trial's start (0) and end (N) are never removed since they carry
        infinite dominance by construction.
        """
        b = list(boundaries)
        l_min = self.config.min_segment_length

        while True:
            diffs = np.diff(b)
            violations = np.flatnonzero(diffs < l_min)
            if violations.size == 0:
                break
            i = int(violations[0])
            lo, hi = b[i], b[i + 1]
            drop = lo if dominance.get(lo, 0.0) <= dominance.get(hi, 0.0) else hi
            b.remove(drop)
            if len(b) <= 2:
                # Cannot prune further without losing the anchor endpoints.
                break

        return np.array(b, dtype=np.int64)
