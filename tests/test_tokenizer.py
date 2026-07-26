import numpy as np
import pytest

from sedat import SEDATConfig, SEDATTokenizer
from sedat.config import ResamplingConfig
from sedat.exceptions import InvalidInputError


def test_end_to_end_shape(synthetic_multichannel):
    x, fs = synthetic_multichannel
    config = SEDATConfig(sampling_rate=fs, num_tokens=5)
    result = SEDATTokenizer(config).transform(x)
    assert result.tokens.shape == (5, result.target_length, x.shape[1])
    assert np.all(np.isfinite(result.tokens))


def test_fixed_target_length_is_respected(synthetic_multichannel):
    x, fs = synthetic_multichannel
    config = SEDATConfig(
        sampling_rate=fs, num_tokens=4, resampling=ResamplingConfig(target_length=64)
    )
    result = SEDATTokenizer(config).transform(x)
    assert result.tokens.shape == (4, 64, x.shape[1])


def test_single_channel_input(rng):
    x = np.sin(np.linspace(0, 20 * np.pi, 800)) + 0.05 * rng.standard_normal(800)
    config = SEDATConfig(sampling_rate=200.0, num_tokens=4)
    result = SEDATTokenizer(config).transform(x)
    assert result.tokens.shape[0] == 4
    assert result.tokens.shape[2] == 1


def test_rejects_too_short_input():
    config = SEDATConfig(sampling_rate=200.0, num_tokens=3)
    with pytest.raises(InvalidInputError):
        SEDATTokenizer(config).transform(np.random.randn(3, 4))


def test_rejects_nan_input():
    x = np.random.randn(200, 4)
    x[10, 2] = np.nan
    config = SEDATConfig(sampling_rate=200.0, num_tokens=3)
    with pytest.raises(InvalidInputError):
        SEDATTokenizer(config).transform(x)


def test_batch_matches_individual_transform(synthetic_multichannel):
    x, fs = synthetic_multichannel
    config = SEDATConfig(sampling_rate=fs, num_tokens=4)
    tokenizer = SEDATTokenizer(config)
    individual = tokenizer.transform(x)
    batch = tokenizer.transform_batch([x], n_jobs=1)
    np.testing.assert_allclose(individual.tokens, batch[0].tokens)


def test_timings_recorded(synthetic_multichannel):
    x, fs = synthetic_multichannel
    config = SEDATConfig(sampling_rate=fs, num_tokens=4)
    result = SEDATTokenizer(config).transform(x)
    expected_stages = {"spatial_aggregation", "decomposition", "segmentation", "resampling"}
    assert expected_stages.issubset(result.stage_timings_ms.keys())
    assert result.total_time_ms >= 0


def test_config_rejects_invalid_num_tokens():
    with pytest.raises(InvalidInputError):
        SEDATConfig(num_tokens=0)


def test_dagaf_max_imfs_synced_with_num_tokens():
    config = SEDATConfig(num_tokens=7)
    assert config.dagaf.max_imfs == 7
