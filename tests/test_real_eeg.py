"""Integration tests exercising SEDAT on real EEG epochs.

Uses motor-imagery trials from the MI EEG Dataset IVa (BCI Competition
III), aliased MI-D1 in the SEDAT manuscript (118 channels, 100 Hz, 3.5 s
epochs). The files live in ``data/`` alongside the rest of the repository
so these tests have no external dataset dependency:

- ``Subject_1 Class0 10.mat`` -- the original single-epoch example.
- ``Test EEG 1.mat`` .. ``Test EEG 10.mat`` -- ten epochs spanning all five
  subjects and both motor-imagery classes, used to check SEDAT behaves
  (finite, correctly shaped, boundary-anchored output) across heterogeneous
  real trials, not just one cherry-picked example.
"""

import os

import numpy as np
import pytest

from sedat import SEDATConfig, SEDATTokenizer
from sedat.utils import zscore_standardize

scipy_io = pytest.importorskip("scipy.io")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SINGLE_EPOCH_PATH = os.path.join(DATA_DIR, "Subject_1 Class0 10.mat")
TEST_EEG_PATHS = [os.path.join(DATA_DIR, f"Test EEG {i}.mat") for i in range(1, 11)]


def _load_and_tokenize(path: str, num_tokens: int = 5):
    mat = scipy_io.loadmat(path)
    x_raw = np.asarray(mat["Data"], dtype=np.float64)
    x = zscore_standardize(x_raw)
    config = SEDATConfig(sampling_rate=100.0, num_tokens=num_tokens)
    result = SEDATTokenizer(config).transform(x)
    return x_raw, result


@pytest.mark.skipif(
    not os.path.exists(SINGLE_EPOCH_PATH), reason="Real EEG test file not present."
)
def test_sedat_on_real_mi_eeg_epoch():
    x_raw, result = _load_and_tokenize(SINGLE_EPOCH_PATH)
    assert x_raw.shape == (350, 118)  # T=350 samples (3.5s @ 100Hz), C=118 channels

    assert result.tokens.shape == (5, result.target_length, 118)
    assert np.all(np.isfinite(result.tokens))
    assert result.segmentation.boundaries[0] == 0
    assert result.segmentation.boundaries[-1] == x_raw.shape[0]
    assert result.dagaf.num_imfs >= 1


@pytest.mark.parametrize("path", TEST_EEG_PATHS)
def test_sedat_on_test_eeg_batch(path):
    if not os.path.exists(path):
        pytest.skip(f"{path} not present.")

    x_raw, result = _load_and_tokenize(path)
    assert x_raw.shape == (350, 118)
    assert result.tokens.shape == (5, result.target_length, 118)
    assert np.all(np.isfinite(result.tokens))
    assert result.segmentation.boundaries[0] == 0
    assert result.segmentation.boundaries[-1] == x_raw.shape[0]
    assert result.dagaf.num_imfs >= 1
    # Every intermediate boundary must lie strictly inside the trial.
    assert np.all(result.segmentation.boundaries[1:-1] > 0)
    assert np.all(result.segmentation.boundaries[1:-1] < x_raw.shape[0])
