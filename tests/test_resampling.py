import numpy as np

from sedat.resampling import FourierResampler


def test_output_shape(synthetic_multichannel):
    x, _ = synthetic_multichannel
    boundaries = np.array([0, 100, 250, 400, 512])
    result = FourierResampler().tokenize(x, boundaries, num_tokens=4, target_length=64)
    assert result.tokens.shape == (4, 64, x.shape[1])


def test_pads_with_zero_tokens_when_too_few_segments(synthetic_multichannel):
    x, _ = synthetic_multichannel
    boundaries = np.array([0, 256, 512])  # only 2 segments
    result = FourierResampler().tokenize(x, boundaries, num_tokens=5, target_length=32)
    assert result.tokens.shape[0] == 5
    assert np.count_nonzero(result.selected_indices == -1) == 3
    for slot in np.where(result.selected_indices == -1)[0]:
        np.testing.assert_allclose(result.tokens[slot], 0.0)


def test_selects_longest_segments(synthetic_multichannel):
    x, _ = synthetic_multichannel
    # Segment lengths: 10, 400, 50, 52 -> top 2 longest are indices 1 and 3.
    boundaries = np.array([0, 10, 410, 460, 512])
    result = FourierResampler().tokenize(x, boundaries, num_tokens=2, target_length=16)
    np.testing.assert_array_equal(np.sort(result.selected_indices), [1, 3])


def test_resampled_segment_preserves_mean_amplitude_scale(rng):
    x = np.sin(np.linspace(0, 4 * np.pi, 300))[:, None]
    boundaries = np.array([0, 300])
    result = FourierResampler().tokenize(x, boundaries, num_tokens=1, target_length=128)
    assert np.max(np.abs(result.tokens[0])) < 1.5  # no blow-up from resampling
