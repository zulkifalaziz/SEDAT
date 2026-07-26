import numpy as np

from sedat.config import SegmentationConfig
from sedat.segmentation import AdaptiveSegmenter


def _make_imfs(n_samples=500, fs=200.0, n_imfs=3, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs
    imfs = []
    for k in range(n_imfs):
        freq = 5.0 * (k + 1)
        imfs.append(np.sin(2 * np.pi * freq * t) + 0.02 * rng.standard_normal(n_samples))
    return imfs, n_samples


def test_boundaries_start_and_end_anchored():
    imfs, n = _make_imfs()
    result = AdaptiveSegmenter().segment(imfs, n_samples=n, fs=200.0)
    assert result.boundaries[0] == 0
    assert result.boundaries[-1] == n


def test_boundaries_are_sorted_and_unique():
    imfs, n = _make_imfs()
    result = AdaptiveSegmenter().segment(imfs, n_samples=n, fs=200.0)
    assert np.all(np.diff(result.boundaries) > 0)


def test_pruning_respects_min_segment_length():
    imfs, n = _make_imfs()
    l_min = 20
    result = AdaptiveSegmenter(SegmentationConfig(min_segment_length=l_min)).segment(
        imfs, n_samples=n, fs=200.0
    )
    assert np.all(np.diff(result.boundaries) >= l_min)


def test_no_imfs_yields_trivial_segmentation():
    result = AdaptiveSegmenter().segment([], n_samples=300, fs=200.0)
    np.testing.assert_array_equal(result.boundaries, [0, 300])


def test_per_imf_boundary_length_matches_input():
    imfs, n = _make_imfs(n_imfs=5)
    result = AdaptiveSegmenter().segment(imfs, n_samples=n, fs=200.0)
    assert result.per_imf_boundary.shape == (5,)
    assert np.all(result.per_imf_boundary >= 0)
    assert np.all(result.per_imf_boundary <= n)
