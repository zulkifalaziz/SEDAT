"""
SEDAT demonstration on a real EEG epoch.

Loads a single motor-imagery epoch from the MI EEG Dataset IVa (BCI
Competition III), one of the seven datasets used to evaluate SEDAT in the
manuscript (aliased MI-D1: 118 channels, 100 Hz, 3.5 s epochs). Runs the
full SEDAT pipeline on the real signal, prints per-stage diagnostics and
timing, and saves a stage-by-stage figure.

Run with::

    python examples/run_real_eeg_demo.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.eeg_io import load_mat_epoch
from sedat import SEDATConfig, SEDATTokenizer
from sedat.utils import zscore_standardize
from sedat.visualization import plot_pipeline_overview

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "Test EEG 1.mat"
)
FS = 100.0  # Hz, per the MI-D1 dataset acquisition parameters (Table 1).
NUM_TOKENS = 5  # per the manuscript's ablation (Section 3.5): a 3.5 s epoch at
# this sampling rate was found to require K = 5 tokens.


def load_epoch(path: str) -> np.ndarray:
    return load_mat_epoch(path)  # shape (T, C) = (350, 118)


def main() -> None:
    print("=" * 70)
    print("SEDAT Tokenizer — Real EEG Epoch Demonstration (MI-D1, BCI-IVa)")
    print("=" * 70)

    x_raw = load_epoch(DATA_PATH)
    print(f"\nLoaded epoch: {DATA_PATH}")
    print(f"  Shape: {x_raw.shape}  (T={x_raw.shape[0]} samples, C={x_raw.shape[1]} channels)")
    print(f"  Duration: {x_raw.shape[0] / FS:.2f} s at fs={FS} Hz")

    # SEDAT operates on a standardized multi-channel input (Section 2.1).
    x = zscore_standardize(x_raw)

    config = SEDATConfig(sampling_rate=FS, num_tokens=NUM_TOKENS)
    tokenizer = SEDATTokenizer(config)

    t0 = time.perf_counter()
    result = tokenizer.transform(x)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    print("\n--- Stage 1: Spatial Aggregation (SE) ---")
    top5 = np.argsort(result.spatial.channel_weights)[::-1][:5]
    print(f"  Top-5 weighted channels (index: weight): "
          f"{[(int(c), round(float(result.spatial.channel_weights[c]), 4)) for c in top5]}")

    print("\n--- Stage 2: DAGAF Decomposition ---")
    print(f"  Extracted IMFs: {result.dagaf.num_imfs} / max {config.dagaf.max_imfs}")
    print(f"  Stop reason: {result.dagaf.stop_reason}")

    print("\n--- Stage 3: Adaptive Segmentation ---")
    print(f"  Raw boundaries (pre-pruning): {result.segmentation.raw_boundaries}")
    print(f"  Final boundaries (post-pruning): {result.segmentation.boundaries}")
    seg_ms = np.diff(result.segmentation.boundaries) / FS * 1000.0
    print(f"  Segment durations (ms): {np.round(seg_ms, 1)}")

    print("\n--- Stage 4: Resampling & Token Formation ---")
    print(f"  Resolved target length L_tgt: {result.target_length}")
    print(f"  Selected segment indices: {result.resampling.selected_indices}")
    print(f"  Final token tensor shape: {result.tokens.shape}  (K, L_tgt, C)")

    print("\n--- Timing ---")
    for stage, ms in result.stage_timings_ms.items():
        print(f"  {stage:>20s}: {ms:8.3f} ms")
    print(f"  {'total (measured)':>20s}: {result.total_time_ms:8.3f} ms")
    print(f"  {'total (wall clock)':>20s}: {wall_ms:8.3f} ms")

    assert result.tokens.shape == (NUM_TOKENS, result.target_length, x.shape[1])
    assert np.all(np.isfinite(result.tokens))
    assert result.segmentation.boundaries[0] == 0
    assert result.segmentation.boundaries[-1] == x.shape[0]
    print("\nAll sanity checks passed.")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "sedat_real_eeg_demo.png")
    plot_pipeline_overview(x, result, fs=FS, save_path=fig_path, show=False)
    print(f"\nFigure saved to: {fig_path}")


if __name__ == "__main__":
    main()
