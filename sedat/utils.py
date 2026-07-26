"""Shared numerical helpers used across the SEDAT pipeline stages."""

from __future__ import annotations

import math

import numpy as np

from sedat.exceptions import InvalidInputError


def validate_eeg_array(x: np.ndarray, *, min_samples: int = 8) -> np.ndarray:
    """Validate and canonicalize a raw EEG epoch.

    Parameters
    ----------
    x:
        Array of shape ``(T, C)`` (time samples, channels) or ``(T,)`` for a
        single channel.
    min_samples:
        Minimum number of temporal samples required for the pipeline to be
        numerically well-defined.

    Returns
    -------
    np.ndarray
        A ``float64`` array of shape ``(T, C)``.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise InvalidInputError(
            f"Expected input of shape (T, C) or (T,), got array with ndim={arr.ndim}."
        )
    if arr.shape[0] < min_samples:
        raise InvalidInputError(
            f"Input has only {arr.shape[0]} time samples; at least {min_samples} are required."
        )
    if not np.all(np.isfinite(arr)):
        raise InvalidInputError("Input contains NaN or infinite values.")
    return arr


def zscore_standardize(x: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Per-channel z-score standardization.

    SEDAT is defined over a standardized multi-channel input (Section 2.1).
    This helper is provided as a convenience preprocessing step; callers may
    substitute their own standardization pipeline.
    """
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / (std + eps)


def count_extrema(signal: np.ndarray) -> int:
    """Count local extrema (maxima + minima) of a 1D signal, Eq. (6).

    Implements ``N_e = |{m : sgn(dr[m]) * sgn(dr[m-1]) < 0}|`` where ``dr``
    is the first-order forward difference restricted to its strictly
    non-zero entries (flat regions are ignored so they do not spuriously
    register as extrema).
    """
    d = np.diff(signal)
    nz = d[d != 0.0]
    if nz.size < 2:
        return 0
    signs = np.sign(nz)
    # sgn(a)*sgn(b) < 0  <=>  signs differ, since both are +/-1 here.
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


def min_extrema_threshold(n: int) -> float:
    """Minimum admissible extrema count, Eq. (18): ceil(4N / (N/2 - 1))."""
    denom = n / 2.0 - 1.0
    if denom <= 0:
        return math.inf
    return math.ceil(4.0 * n / denom)
