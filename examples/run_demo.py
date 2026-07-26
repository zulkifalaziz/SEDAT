"""
End-to-end SEDAT demonstration.

Generates a synthetic multi-channel test signal with four distinct
oscillatory regimes, runs it through the full SEDAT pipeline, prints
per-stage diagnostics and timing, and saves a matplotlib figure showing the
tokenizer's progress at every stage.

Run with::

    python examples/run_demo.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# Allow running this script directly (``python examples/run_demo.py``)
# without having installed the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.synthetic_signal import generate_synthetic_eeg
from sedat import SEDATConfig, SEDATTokenizer
from sedat.visualization import plot_pipeline_overview


def main() -> None:
    fs = 250.0
    n_samples = 1000
    n_channels = 16
    num_tokens = 6

    print("=" * 70)
    print("SEDAT Tokenizer — End-to-End Demonstration")
    print("=" * 70)

    x = generate_synthetic_eeg(n_samples=n_samples, n_channels=n_channels, fs=fs)
    print(f"\nSynthetic test signal: shape={x.shape}, fs={fs} Hz, "
          f"duration={n_samples / fs:.2f} s")

    config = SEDATConfig(sampling_rate=fs, num_tokens=num_tokens)
    tokenizer = SEDATTokenizer(config)

    t0 = time.perf_counter()
    result = tokenizer.transform(x)
    total_wall_ms = (time.perf_counter() - t0) * 1000.0

    print("\n--- Stage 1: Spatial Aggregation (SE) ---")
    print(f"  Channel weights (top 5): "
          f"{np.round(np.sort(result.spatial.channel_weights)[::-1][:5], 4)}")
    print(f"  Drive signal shape: {result.spatial.drive_signal.shape}")

    print("\n--- Stage 2: DAGAF Decomposition ---")
    print(f"  Extracted IMFs: {result.dagaf.num_imfs} / max {config.dagaf.max_imfs}")
    print(f"  Stop reason: {result.dagaf.stop_reason}")

    print("\n--- Stage 3: Adaptive Segmentation ---")
    print(f"  Raw boundaries (pre-pruning): {result.segmentation.raw_boundaries}")
    print(f"  Final boundaries (post-pruning): {result.segmentation.boundaries}")
    print(f"  Number of final segments: {len(result.segmentation.boundaries) - 1}")

    print("\n--- Stage 4: Resampling & Token Formation ---")
    print(f"  Resolved target length L_tgt: {result.target_length}")
    print(f"  Selected segment indices: {result.resampling.selected_indices}")
    print(f"  Final token tensor shape: {result.tokens.shape}  (K, L_tgt, C)")

    print("\n--- Timing ---")
    for stage, ms in result.stage_timings_ms.items():
        print(f"  {stage:>20s}: {ms:8.3f} ms")
    print(f"  {'total (measured)':>20s}: {result.total_time_ms:8.3f} ms")
    print(f"  {'total (wall clock)':>20s}: {total_wall_ms:8.3f} ms")

    # Sanity checks -------------------------------------------------------
    assert result.tokens.shape == (num_tokens, result.target_length, n_channels)
    assert np.all(np.isfinite(result.tokens)), "Token tensor contains non-finite values."
    assert result.segmentation.boundaries[0] == 0
    assert result.segmentation.boundaries[-1] == n_samples
    print("\nAll sanity checks passed.")

    # Figure ---------------------------------------------------------------
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "sedat_demo.png")
    plot_pipeline_overview(x, result, fs=fs, save_path=fig_path, show=False)
    print(f"\nFigure saved to: {fig_path}")

    # Quick batch-processing demonstration ---------------------------------
    print("\n--- Batch processing demo (5 synthetic trials) ---")
    trials = [
        generate_synthetic_eeg(n_samples=n_samples, n_channels=n_channels, fs=fs, seed=s)
        for s in range(5)
    ]
    t0 = time.perf_counter()
    batch_results = tokenizer.transform_batch(trials, n_jobs=1)
    batch_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  Tokenized {len(batch_results)} trials in {batch_ms:.2f} ms "
          f"({batch_ms / len(batch_results):.2f} ms/trial)")


if __name__ == "__main__":
    main()
