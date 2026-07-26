"""
SEDAT Step 2 — Data-Adaptive Gaussian Average Filtering (DAGAF) decomposition.

Implements the iterative sifting procedure of Lin, Tan & Tian (2022) as
specialized in the SEDAT manuscript (Section 2.1.2, Eqs. 6-11), producing an
ordered sequence of intrinsic mode functions (IMFs) from highest to lowest
frequency.

Unlike EMD/EEMD, DAGAF derives its "instantaneous mean" via a mathematically
rigorous, data-adaptive Gaussian moving-average filter instead of cubic
spline envelope interpolation. This removes end-effects and mode mixing
while remaining ``O(N log N)`` per iteration, since the moving average is
computed as an FFT-based convolution (`scipy.signal.fftconvolve`) rather
than a direct O(N*M) sliding-window sum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.signal import fftconvolve

from sedat.config import DAGAFConfig
from sedat.utils import count_extrema, min_extrema_threshold


@dataclass(frozen=True)
class DAGAFResult:
    """Output of the DAGAF decomposition stage."""

    imfs: List[np.ndarray] = field(default_factory=list)
    """Ordered list of IMFs, highest to lowest frequency, each shape (N,)."""

    residual: np.ndarray = field(default_factory=lambda: np.empty(0))
    """Final residual (trend) signal after all IMFs have been extracted."""

    stop_reason: str = ""
    """Human-readable reason the sifting procedure terminated."""

    @property
    def num_imfs(self) -> int:
        return len(self.imfs)


def _gaussian_window(half_length: int, alpha: float) -> np.ndarray:
    """Discrete truncated Gaussian window w[m] for m = -M..M, Eq. (8)-(9).

    Returns the *normalized* window (sums to unity) so that energy is
    preserved during filtering, laid out as a length ``2M+1`` array indexed
    ``0..2M`` corresponding to ``m = -M..M``.
    """
    m = np.arange(-half_length, half_length + 1, dtype=np.float64)
    w = np.exp(-0.5 * (alpha * m / half_length) ** 2)
    return w / w.sum()


def _extend_reflect(r: np.ndarray, half_length: int) -> np.ndarray:
    """Double-symmetrical reflection extension anchored at the boundary
    values, Eq. (10). Returns an array of length ``N + 2*half_length``.
    """
    n = r.shape[0]
    mean_r = r.mean()
    m = half_length

    # Left extension: indices n = -M..-1  ->  r_tilde[n] = 2*mean - r[-n]
    left = 2.0 * mean_r - r[1 : m + 1][::-1]
    # Right extension: indices n = N..N+M-1 -> r_tilde[n] = 2*mean - r[2N-n-2]
    right = 2.0 * mean_r - r[-2 : -m - 2 : -1] if m > 0 else r[:0]

    return np.concatenate([left, r, right])


def _instantaneous_mean(r: np.ndarray, half_length: int, alpha: float) -> np.ndarray:
    """Compute mu_k[n] = sum_m w_G[m] * r_tilde[n+m] via FFT convolution.

    Because the Gaussian window is symmetric, correlation and convolution
    coincide, so a single ``fftconvolve(..., mode='valid')`` call yields
    the exact moving-average defined in Eq. (10)-(11) in O(N log N) time.
    """
    window = _gaussian_window(half_length, alpha)
    extended = _extend_reflect(r, half_length)
    mu = fftconvolve(extended, window, mode="valid")
    # 'valid' convolution of length (N+2M) with a (2M+1)-tap kernel yields
    # exactly N samples, aligned with n = 0..N-1.
    return mu[: r.shape[0]]


def _half_window_length(n: int, n_extrema: int, chi: float) -> int:
    """Adaptive Gaussian half-window length M^(k), Eq. (7)."""
    if n_extrema <= 0:
        upper = (n - 1) // 2
        return max(1, upper)
    candidate = 2 * int(np.floor(chi * n / n_extrema))
    upper = (n - 1) // 2
    return max(1, min(candidate, upper))


class DAGAFDecomposer:
    """Implements the DAGAF sifting procedure described in Section 2.1.2."""

    def __init__(self, config: DAGAFConfig | None = None) -> None:
        self.config = config or DAGAFConfig()

    def decompose(self, s: np.ndarray) -> DAGAFResult:
        """Decompose a 1D drive signal ``s`` into up to ``max_imfs`` IMFs.

        The sifting procedure terminates when any of three conditions is
        met (Section 2.1.2): the extrema count falls below the
        length-derived minimum threshold, the extracted IMF's energy
        approaches numerical zero, or the maximum decomposition depth is
        reached.
        """
        s = np.asarray(s, dtype=np.float64)
        n = s.shape[0]
        threshold = min_extrema_threshold(n)

        imfs: List[np.ndarray] = []
        residual = s
        stop_reason = f"reached max_imfs={self.config.max_imfs}"

        for k in range(1, self.config.max_imfs + 1):
            n_extrema = count_extrema(residual)
            if n_extrema <= threshold:
                stop_reason = (
                    f"extrema count {n_extrema} <= threshold {threshold:.2f} "
                    f"at iteration {k}"
                )
                break

            half_length = _half_window_length(n, n_extrema, self.config.chi)
            mu = _instantaneous_mean(residual, half_length, self.config.alpha)
            imf = residual - mu

            imf_energy = float(np.sum(imf ** 2))
            imfs.append(imf)
            residual = mu

            if imf_energy < self.config.energy_tol:
                stop_reason = (
                    f"IMF energy {imf_energy:.3e} < tol {self.config.energy_tol:.1e} "
                    f"at iteration {k}"
                )
                break

        return DAGAFResult(imfs=imfs, residual=residual, stop_reason=stop_reason)
