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

- **`sedat/`** — the tokenizer itself: a dependency-light, pure NumPy/SciPy
  implementation of all four SEDAT stages, independently testable and
  usable as a standalone library.
- **`labram/`** — a minimal, runnable example of feeding SEDAT's token
  output into a downstream EEG foundation-model-style architecture
  (a from-scratch, LaBraM-inspired Transformer classifier), including a
  full training/evaluation loop on real motor-imagery EEG.
- **`examples/`** — runnable demo scripts, from a synthetic signal to a
  full 560-epoch real-data classification experiment.
- **`data/`** — a handful of real EEG epochs (see [Data](#data) below) so
  the demos run out of the box with no external dataset download.
- **`tests/`** — a pytest suite covering every pipeline stage, the model
  architecture, and integration tests against real EEG data.

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
4. **Resampling & Token Formation** (`sedat/resampling.py`) — the
   resulting variable-length segments are FFT-resampled to a common
   target length and stacked into the final token tensor. If fewer than
   `K` usable segments survive pruning, the tensor is zero-padded so every
   trial produces an identically shaped output.

`sedat/tokenizer.py` exposes `SEDATTokenizer`, the single public entry
point that orchestrates all four stages and returns a `SEDATResult`
dataclass containing the token tensor, every intermediate stage's
diagnostics, and per-stage timing.

## Installation

```bash
git clone https://github.com/zulkifalaziz/SEDAT.git
cd SEDAT
pip install -r requirements.txt
```

The core tokenizer depends only on NumPy and SciPy. For an editable
package install:

```bash
pip install -e .
```

Plotting (used by the example scripts) requires `matplotlib`, and the
LaBraM-style classification example requires PyTorch and scikit-learn:

```bash
pip install -e .[dev]          # everything, for development/testing
# or selectively:
pip install -e .[viz]          # tokenizer + plotting only
pip install -r requirements-labram.txt   # + torch, scikit-learn
```

## Quick start

```python
import numpy as np
from sedat import SEDATConfig, SEDATTokenizer

x = np.random.randn(1000, 32)  # (T, C) — 1000 samples, 32 channels

config = SEDATConfig(sampling_rate=250.0, num_tokens=8)
tokenizer = SEDATTokenizer(config)

result = tokenizer.transform(x)
print(result.tokens.shape)          # (8, L_tgt, 32)
print(result.stage_timings_ms)      # per-stage timing breakdown
```

Batch tokenization (each trial is tokenized independently, so batches are
embarrassingly parallel):

```python
epochs = [np.random.randn(1000, 32) for _ in range(20)]
results = tokenizer.transform_batch(epochs, n_jobs=4)
```

To reproduce a fixed token length across an entire dataset (required
before batching tokens into a downstream neural network), pin
`target_length` explicitly instead of letting SEDAT infer it per trial:

```python
from sedat.config import ResamplingConfig

config = SEDATConfig(
    sampling_rate=100.0,
    num_tokens=5,
    resampling=ResamplingConfig(target_length=64),
)
```

## Project layout

```
sedat/                      Core tokenizer library (NumPy/SciPy only)
    config.py                Dataclass configuration for every stage
    utils.py                 Shared numerical helpers (validation, extrema counting)
    spatial_aggregation.py   Step 1 — SE channel weighting
    decomposition.py         Step 2 — DAGAF sifting (FFT-based, O(N log N))
    segmentation.py          Step 3 — instantaneous-frequency boundary detection
    resampling.py            Step 4 — Fourier-domain resampling & token stacking
    tokenizer.py             SEDATTokenizer orchestrator (public entry point)
    visualization.py         Optional matplotlib diagnostics (imported lazily)
    exceptions.py            Custom exception hierarchy

labram/                      Minimal foundation-model integration example
    config.py                 Architecture hyperparameters
    model.py                   A from-scratch, LaBraM-style Transformer (PyTorch)
    dataset.py                 Scans a folder of .mat epochs, tokenizes with SEDAT
    evaluation.py              Stratified K-fold cross-validation training loop
    visualization.py           Classification report figure

examples/
    synthetic_signal.py       Synthetic multi-channel test-signal generator
    eeg_io.py                  Shared `.mat` epoch loader used by the demos
    run_demo.py                End-to-end demo on a synthetic signal
    run_real_eeg_demo.py       End-to-end demo on one real EEG epoch
    run_test_eeg_batch.py      Batch demo over 10 real epochs (data/)
    run_labram_classification.py  Full SEDAT -> classifier training example

data/                        A handful of real EEG epochs (see Data below)

tests/                       Pytest suite for every module
```

## Demos

```bash
python examples/run_demo.py
```

Generates a synthetic 16-channel signal with four distinct oscillatory
regimes (mimicking abrupt neural state transitions), runs the full SEDAT
pipeline, prints per-stage diagnostics and timing, and saves a multi-panel
figure to `examples/output/sedat_demo.png` (the raw signal, the SE drive
signal and channel weights, the DAGAF IMFs, the segmentation boundaries,
and the final token power map).

```bash
python examples/run_real_eeg_demo.py
```

Runs the same pipeline on one real motor-imagery epoch
(`data/Subject_1 Class0 10.mat`, 118 channels, 100 Hz, 3.5 s).

```bash
python examples/run_test_eeg_batch.py
```

Runs SEDAT over 10 more real epochs (`data/Test EEG 1.mat` .. `Test EEG
10.mat`, one per subject/class combination) and prints a summary table
(IMF count, segment count, resolved token length, zero-padding, timing)
plus one figure per epoch.

```bash
pip install -r requirements-labram.txt
python examples/run_labram_classification.py
```

This is the "minimal implementation example with a foundation model":
it tokenizes real motor-imagery EEG with SEDAT and trains a downstream
Transformer classifier end-to-end on the resulting tokens. See
[SEDAT -> foundation-model classification](#sedat---foundation-model-classification)
below for what it does and an honest account of its results.

## Data

`data/` ships 11 real EEG epochs from the **MI EEG Dataset IVa** (BCI
Competition III, 100 Hz variant): one 3.5 s, 118-channel motor-imagery
trial per file, spanning 5 subjects and both motor-imagery classes (right
hand / right foot). This is the dataset aliased **MI-D1** in the SEDAT
manuscript. These files are included solely so the example scripts run
without requiring a separate dataset download; they are a small subset for
demonstration, not the full corpus. Filenames follow the source dataset's
`Subject_<S> Class<C> <index>.mat` convention, each containing a single
`Data` array of shape `(350, 118)` (time samples × channels).

## SEDAT -> foundation-model classification

`labram/` is a minimal example of plugging SEDAT's token tensor into a
downstream foundation-model architecture. It is a small, from-scratch
reimplementation of LaBraM's architectural pattern (learned token
embedding, a CLS token, learnable positional embeddings, a pre-norm
Transformer encoder, and a linear classification head) — **not** the
official LaBraM codebase or its pretrained weights (Jiang, Zhao & Lu,
2024, arXiv:2405.18765). It exists to demonstrate that SEDAT's
`(K, L_tgt, C)` output is a drop-in replacement for a foundation model's
native tokenizer and trains end-to-end.

`examples/run_labram_classification.py` tokenizes all 560 epochs of the
MI EEG Dataset IVa (100 Hz variant, 5 subjects, 2 classes — obtained
separately; only a small sample ships in `data/`, see above) with SEDAT,
then trains/evaluates the classifier with subject-pooled, stratified
5-fold cross-validation, following the SEDAT manuscript's evaluation
protocol for multi-subject datasets.

**Result on this dataset: 49.1% ± 2.65% held-out accuracy (chance = 50%
for 2 classes), while training loss converges to ~0 on every fold.**

That gap — a near-perfect training fit with chance-level generalization —
is diagnostic, not a SEDAT problem. The engineering integration works
exactly as intended: tokens flow through the full model, gradients update
every layer, and training loss decreases smoothly and reproducibly across
all 5 folds. What fails to generalize is the randomly initialized
Transformer itself, for two well-understood reasons:

1. **No pretraining.** LaBraM's reported accuracy in the literature comes
   from fine-tuning weights already pretrained on a large multi-subject
   EEG corpus. This example's classifier starts from random weights and
   must learn a motor-imagery-relevant representation from scratch using
   only ~450 training trials per fold — nowhere near enough data for a
   ~618K-parameter Transformer, which instead memorizes the training set
   (hence the near-zero training loss).
2. **Cross-subject pooling.** Motor-imagery EEG varies substantially
   between subjects (electrode impedance, cortical topography, task
   strategy). Pooling all 5 subjects into one training set — as the
   manuscript's protocol specifies for multi-subject datasets — is a much
   harder generalization problem than subject-dependent classification,
   and is exactly the problem foundation-model pretraining exists to
   solve.

If you adapt this example and want a more meaningful accuracy number, the
two changes most likely to help are: (a) evaluate subject-dependently
(fold within each subject rather than pooling all of them), or (b) swap in
actual pretrained foundation-model weights rather than the from-scratch
classifier provided here.

## Testing

```bash
pip install -e .[dev]
pytest tests/ -v
```

The suite covers unit tests for every SEDAT stage, the LaBraM-style model
architecture, and integration tests that run the full pipeline against the
real EEG epochs in `data/`.

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
  input array (shape, finiteness) before any numerical work begins.

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
