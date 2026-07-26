"""Small shared helper for loading `.mat` EEG epochs used by the examples."""

from __future__ import annotations

import numpy as np
import scipy.io as sio


def load_mat_epoch(path: str, key: str = "Data") -> np.ndarray:
    """Load a single (T, C) EEG epoch stored under ``key`` in a MATLAB file."""
    mat = sio.loadmat(path)
    return np.asarray(mat[key], dtype=np.float64)
