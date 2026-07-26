import numpy as np

from sedat.config import SpatialAggregationConfig
from sedat.spatial_aggregation import SpatialAggregator


def test_weights_form_probability_simplex(synthetic_multichannel):
    x, _ = synthetic_multichannel
    result = SpatialAggregator().aggregate(x)
    assert np.isclose(result.channel_weights.sum(), 1.0)
    assert np.all(result.channel_weights >= 0.0)


def test_drive_signal_shape(synthetic_multichannel):
    x, _ = synthetic_multichannel
    result = SpatialAggregator().aggregate(x)
    assert result.drive_signal.shape == (x.shape[0],)


def test_higher_energy_channel_gets_higher_weight(rng):
    n_samples = 300
    t = np.linspace(0, 1, n_samples)
    low_energy = 0.1 * np.sin(2 * np.pi * 5 * t)
    high_energy = 2.0 * np.sin(2 * np.pi * 5 * t)
    x = np.stack([low_energy, high_energy], axis=1)
    result = SpatialAggregator().aggregate(x)
    assert result.channel_weights[1] > result.channel_weights[0]


def test_single_channel_bypasses_aggregation(rng):
    x = rng.standard_normal((100, 1))
    result = SpatialAggregator().aggregate(x)
    np.testing.assert_allclose(result.drive_signal, x[:, 0])
    np.testing.assert_allclose(result.channel_weights, [1.0])


def test_disabled_aggregation_uses_uniform_weights(synthetic_multichannel):
    x, _ = synthetic_multichannel
    result = SpatialAggregator(SpatialAggregationConfig(enabled=False)).aggregate(x)
    expected_weight = 1.0 / x.shape[1]
    np.testing.assert_allclose(result.channel_weights, expected_weight)
