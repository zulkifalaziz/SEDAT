import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def synthetic_multichannel(rng):
    """A small deterministic multi-channel test signal for fast unit tests."""
    n_samples, n_channels, fs = 512, 8, 200.0
    t = np.arange(n_samples) / fs
    base = np.sin(2 * np.pi * 5 * t)
    base[n_samples // 2 :] = np.sin(2 * np.pi * 25 * t[n_samples // 2 :])
    topo = np.linspace(0.5, 1.5, n_channels)
    x = base[:, None] * topo[None, :] + 0.05 * rng.standard_normal((n_samples, n_channels))
    return x, fs
