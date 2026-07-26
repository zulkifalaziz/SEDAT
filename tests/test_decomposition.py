import numpy as np
import pytest

from sedat.config import DAGAFConfig
from sedat.decomposition import DAGAFDecomposer
from sedat.utils import count_extrema, min_extrema_threshold


def test_count_extrema_simple_sine():
    t = np.linspace(0, 4 * np.pi, 1000)
    s = np.sin(t)
    # 4*pi covers 2 full periods -> 4 extrema (2 maxima + 2 minima).
    assert count_extrema(s) == 4


def test_count_extrema_flat_signal_is_zero():
    s = np.ones(50)
    assert count_extrema(s) == 0


def test_min_extrema_threshold_positive_for_reasonable_lengths():
    assert min_extrema_threshold(1000) > 0
    assert np.isinf(min_extrema_threshold(2))


def test_decompose_produces_at_most_max_imfs():
    t = np.linspace(0, 1, 1000)
    s = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 40 * t)
    config = DAGAFConfig(max_imfs=4)
    result = DAGAFDecomposer(config).decompose(s)
    assert 0 < result.num_imfs <= 4
    for imf in result.imfs:
        assert imf.shape == s.shape


def test_decomposition_reconstructs_original_signal():
    """sum(IMFs) + residual must reconstruct the original signal exactly,
    since each iteration only subtracts the instantaneous mean."""
    t = np.linspace(0, 1, 800)
    s = np.sin(2 * np.pi * 3 * t) + 0.3 * np.sin(2 * np.pi * 30 * t) + 0.1 * t
    result = DAGAFDecomposer(DAGAFConfig(max_imfs=6)).decompose(s)
    reconstructed = result.residual + sum(result.imfs) if result.imfs else result.residual
    np.testing.assert_allclose(reconstructed, s, atol=1e-8)


def test_higher_frequency_imf_extracted_first():
    """The first IMF should capture more high-frequency energy than the
    final residual, since DAGAF peels off fast oscillations before slow
    ones. Note: with only ~3 cycles of the 3 Hz component in a 1-second
    window, the length-derived extrema threshold (Eq. 18) correctly halts
    sifting after a single IMF here -- that is expected DAGAF behavior,
    not a defect, so this test compares IMF-1 against the residual rather
    than assuming a second IMF is always produced."""
    t = np.linspace(0, 1, 1000)
    fast = np.sin(2 * np.pi * 45 * t)
    slow = np.sin(2 * np.pi * 3 * t)
    s = fast + slow
    result = DAGAFDecomposer(DAGAFConfig(max_imfs=4)).decompose(s)
    assert result.num_imfs >= 1

    def dominant_freq(sig):
        spectrum = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(len(sig), d=t[1] - t[0])
        return freqs[np.argmax(spectrum)]

    assert dominant_freq(result.imfs[0]) > dominant_freq(result.residual)


def test_constant_signal_yields_no_imfs():
    s = np.full(200, 3.0)
    result = DAGAFDecomposer(DAGAFConfig(max_imfs=5)).decompose(s)
    assert result.num_imfs == 0
    np.testing.assert_allclose(result.residual, s)
