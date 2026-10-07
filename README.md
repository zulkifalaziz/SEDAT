# SEDAT: SE-DAGAF Adaptive Tokenizer

A reference Python implementation of **SEDAT**, a hybrid, physiologically
grounded tokenizer for large EEG foundation models.

Existing EEG tokenizers impose arbitrary fixed-length boundaries that are
misaligned with real neural state transitions, ignore inter-channel spatial
information, and use segmentation criteria that fail to generalize across
heterogeneous EEG paradigms. SEDAT addresses this with a single, efficient
pipeline that combines squeeze-and-excitation (SE) spatial aggregation,
data-adaptive Gaussian average filtering (DAGAF) signal decomposition,
instantaneous-frequency-guided adaptive segmentation, and Fourier-domain
resampling — producing physiologically anchored, fixed-dimensional tokens
directly from raw multi-channel EEG.

This repository accompanies the paper below and exists so that other
researchers can use SEDAT directly rather than reimplementing it from the
manuscript's equations:

> M. Z. Aziz, Z. Yue, H. Binwen, Y. Xiaojun. **"SEDAT: A Hybrid Tokenizer
> for Large EEG Models."** Preprint submitted to *IOP Journal of Neural
> Engineering*.

The DAGAF decomposition stage is based on:

> Y.-D. Lin, Y. K. Tan, B. Tian. **"A novel approach for decomposition of
> biomedical signals in different applications based on data-adaptive
> Gaussian average filtering."** *Biomedical Signal Processing and
> Control*, 71 (2022) 103104.

## What's in this repository

The repository contains the tokenizer only: the **`sedat/`** package, a
dependency-light, pure NumPy/SciPy implementation of all four SEDAT stages.
Each stage is an independently usable module, and `SEDATTokenizer` chains
them behind a single entry point. Docstrings throughout cite the
corresponding equations and sections of the manuscript.

```
sedat/
├── __init__.py              Public API: SEDATConfig, SEDATTokenizer, TokenizationResult, ...
├── config.py                Dataclass configuration for every stage
├── tokenizer.py             SEDATTokenizer orchestrator (public entry point)
├── spatial_aggregation.py   Step 1: SE channel weighting
├── decomposition.py         Step 2: DAGAF sifting (FFT-based, O(N log N))
├── segmentation.py          Step 3: instantaneous-frequency boundary detection
├── resampling.py            Step 4: Fourier-domain resampling and token stacking
├── utils.py                 Shared helpers: input validation, z-scoring, extrema counting
├── visualization.py         Optional matplotlib diagnostics (imported lazily)
└── exceptions.py            Custom exception hierarchy
```

## The SEDAT pipeline

SEDAT transforms a raw `(T, C)` multi-channel EEG trial into a
fixed-dimensional token tensor `Z ∈ R^(K × L_tgt × C)` through four
sequential stages:

1. **Spatial Aggregation** (`sedat/spatial_aggregation.py`) — a
   squeeze-and-excitation (SE) mechanism computes each channel's temporal
   energy, converts it to an adaptive importance weight via a sigmoid
   activation, and collapses the trial into a single 1D "drive signal"
   that preserves inter-channel spatial information a single-channel
   projection would discard.
2. **Signal Decomposition** (`sedat/decomposition.py`) — data-adaptive
   Gaussian average filtering (DAGAF) iteratively sifts the drive signal
   into intrinsic mode functions (IMFs). Unlike EMD/EEMD, DAGAF derives its
   instantaneous mean via a mathematically rigorous, data-adaptive Gaussian
   moving average rather than cubic-spline envelope interpolation,
   eliminating end-effects and reducing mode mixing.
3. **Adaptive Segmentation** (`sedat/segmentation.py`) — for each IMF, the
   discrete Hilbert transform's instantaneous frequency identifies the
   single moment of greatest oscillatory instability, yielding exactly one
   physiologically anchored boundary per mode. Boundaries that would
   produce degenerate micro-segments are pruned.
4. **Resampling & Token Formation** (`sedat/resampling.py`) — the raw trial
   is cut at the surviving boundaries, the `K` longest segments are kept (in
   chronological order), FFT-resampled to a common target length `L_tgt`,
   and stacked into the final token tensor. If fewer than `K` usable
   segments survive pruning, the remaining slots are zero-padded so every
   trial produces an identically shaped output.

`SEDATTokenizer` (in `sedat/tokenizer.py`) is the single public entry point
that orchestrates all four stages. It returns a `TokenizationResult`
dataclass containing the token tensor, every intermediate stage's
diagnostics, and per-stage timing.

## Installation

```bash
git clone https://github.com/zulkifalaziz/SEDAT.git
cd SEDAT
pip install numpy scipy
```

Requirements: Python 3.9+, NumPy ≥ 1.24, SciPy ≥ 1.10. The plotting helpers
in `sedat.visualization` additionally need `matplotlib` ≥ 3.7
(`pip install matplotlib`); the core package never imports it.

`sedat/` is a self-contained package. Run Python from the repository root so
that `import sedat` resolves, add the repository root to your `PYTHONPATH`,
or copy the `sedat/` folder into your own project.

## Quick start

```python
import numpy as np
from sedat import SEDATConfig, SEDATTokenizer
from sedat.utils import zscore_standardize

x = np.random.randn(1000, 32)   # (T, C): 1000 samples, 32 channels; use your own epoch here
x = zscore_standardize(x)       # per-channel standardization (see "Input conventions")

config = SEDATConfig(sampling_rate=250.0, num_tokens=8)
tokenizer = SEDATTokenizer(config)

result = tokenizer.transform(x)
print(result.tokens.shape)          # (8, L_tgt, 32)
print(result.stage_timings_ms)      # per-stage timing breakdown
```

### Input conventions

- `x` has shape `(T, C)` (time samples × channels), or `(T,)` for a single
  channel. It must be finite (no NaN or infinity) and contain at least 8
  samples; otherwise `InvalidInputError` is raised.
- SEDAT is defined over a standardized input. `sedat.utils.zscore_standardize`
  provides per-channel z-scoring as a convenience; substitute your own
  preprocessing if you prefer.

### Batch tokenization

Each trial is tokenized independently (boundaries are trial-specific), so
batches are embarrassingly parallel. `n_jobs=1` (the default) runs in the
current process; larger values distribute trials across a process pool.

```python
if __name__ == "__main__":   # required on Windows and macOS when n_jobs > 1
    epochs = [zscore_standardize(np.random.randn(1000, 32)) for _ in range(20)]
    results = tokenizer.transform_batch(epochs, n_jobs=4)
```

### Fixed token length

By default `L_tgt` is inferred per trial as the median segment length
(capped at `max_target_length`, 512 by default), so different trials can
produce different token lengths. To get identically shaped tensors across an
entire dataset, which is required before batching tokens into a downstream
neural network, pin `target_length` explicitly:

```python
from sedat import ResamplingConfig

config = SEDATConfig(
    sampling_rate=100.0,
    num_tokens=5,
    resampling=ResamplingConfig(target_length=64),
)
```

Every trial then yields a tensor of shape `(5, 64, C)`.

## The result object

`SEDATTokenizer.transform` returns a `TokenizationResult`;
`transform_batch` returns a list of them.

| Field | Contents |
| --- | --- |
| `tokens` | Final tensor `Z`, shape `(K, L_tgt, C)` |
| `spatial` | Step 1: `drive_signal` `(T,)`, `channel_weights` `(C,)`, `channel_energy` `(C,)` |
| `dagaf` | Step 2: `imfs` (list of `(T,)` arrays, highest to lowest frequency), `residual`, `stop_reason`, `num_imfs` |
| `segmentation` | Step 3: `boundaries` (pruned; always starts at 0 and ends at `T`), `raw_boundaries`, `per_imf_boundary` |
| `resampling` | Step 4: `tokens`, `segment_lengths`, `selected_indices` (`-1` marks a zero-padded token) |
| `target_length` | The `L_tgt` that was used |
| `stage_timings_ms` | Per-stage time in milliseconds: `spatial_aggregation`, `decomposition`, `segmentation`, `resampling` |
| `total_time_ms` | Sum of the stage timings |

## Configuration reference

Every stage has a frozen dataclass that validates its parameters when it is
constructed, raising `InvalidInputError` for invalid values. Defaults follow
the values reported in the manuscript.

| Config (argument of `SEDATConfig`) | Parameter | Default | Description |
| --- | --- | --- | --- |
| `SEDATConfig` | `sampling_rate` | `250.0` | Sampling frequency of the input, in Hz |
| | `num_tokens` | `8` | Number of tokens `K`; also sets the DAGAF decomposition depth |
| | `max_target_length` | `512` | Upper bound on an automatically inferred `L_tgt` |
| `SpatialAggregationConfig` (`spatial`) | `gamma` | `10.0` | Sigmoid temperature |
| | `delta` | `0.5` | Sigmoid shift center |
| | `eps` | `1e-8` | Numerical guard in the min-max normalization |
| | `enabled` | `True` | `False` bypasses SE aggregation (it is also bypassed automatically for single-channel input) |
| `DAGAFConfig` (`dagaf`) | `chi` | `2.0` | Scales the Gaussian window half-length |
| | `alpha` | `3.0` | Gaussian window shape |
| | `max_imfs` | `num_tokens` | Maximum decomposition depth; `SEDATConfig` always keeps it equal to `num_tokens` |
| | `energy_tol` | `1e-10` | Sifting stops once an extracted IMF's energy falls below this value |
| `SegmentationConfig` (`segmentation`) | `min_segment_length` | `4` | Minimum segment length `L_min`, in samples; closer boundaries are pruned |
| `ResamplingConfig` (`resampling`) | `target_length` | `None` | Fixed `L_tgt`; `None` infers it per trial |

## Visualization (optional)

`sedat.visualization.plot_pipeline_overview(x, result, fs, save_path=...)`
renders a multi-panel figure covering every stage: the raw signal, the SE
drive signal and channel weights, the DAGAF IMFs, the segmentation
boundaries, and the final token power map. It requires `matplotlib` and is
imported lazily, so the core package works without it.

## Design notes

- **Efficiency**: the DAGAF instantaneous-mean filter is computed as an
  FFT-based convolution (`scipy.signal.fftconvolve`) rather than a direct
  sliding-window sum, giving the `O(N log N)` per-iteration cost reported
  in the manuscript regardless of the (data-adaptive) window length.
  Fourier resampling likewise uses a single vectorized FFT call across all
  channels per segment. Overall pipeline complexity is `O(CN + KN log N)`.
- **Single-channel data**: spatial aggregation is automatically bypassed
  when the input has one channel (or when explicitly disabled via
  `SpatialAggregationConfig(enabled=False)`), matching the SEDAT
  experimental protocol for single-channel datasets (e.g. sleep staging).
- **Fixed-dimensional output guarantee**: if pruning during segmentation
  leaves fewer than `K` usable segments, the token tensor is padded with
  zero tokens so every trial produces an identically shaped tensor.
- **Validation**: all configuration dataclasses validate their parameters
  eagerly in `__post_init__`, and `SEDATTokenizer.transform` validates the
  input array (shape, finiteness, minimum length) before any numerical work
  begins.

## Citation

If you use SEDAT in your research, please cite:

```bibtex
@article{aziz2026sedat,
  title   = {SEDAT: A Hybrid Tokenizer for Large EEG Models},
  author  = {Aziz, Muhammad Zulkifal and Yue, Zhuo and Binwen, Huang and Xiaojun, Yu},
  journal = {Preprint submitted to IOP Journal of Neural Engineering},
  year    = {2026}
}

@article{lin2022dagaf,
  title   = {A novel approach for decomposition of biomedical signals in different applications based on data-adaptive Gaussian average filtering},
  author  = {Lin, Yue-Der and Tan, Yong Kok and Tian, Baofeng},
  journal = {Biomedical Signal Processing and Control},
  volume  = {71},
  pages   = {103104},
  year    = {2022},
  publisher = {Elsevier}
}
```

## License

Released under the [MIT License](LICENSE).
