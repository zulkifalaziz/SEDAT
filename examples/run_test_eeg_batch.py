"""
SEDAT batch demonstration across the 10 "Test EEG" epochs.

Runs the full SEDAT pipeline independently on each of
``data/Test EEG 1.mat`` .. ``data/Test EEG 10.mat`` (real motor-imagery
trials drawn from five different subjects and both classes of the MI EEG
Dataset IVa, BCI Competition III), printing a per-epoch summary and saving
one stage-by-stage figure per epoch.

Run with::

    python examples/run_test_eeg_batch.py
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

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
FS = 100.0
NUM_TOKENS = 5
N_EPOCHS = 10


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config = SEDATConfig(sampling_rate=FS, num_tokens=NUM_TOKENS)
    tokenizer = SEDATTokenizer(config)

    print("=" * 88)
    print("SEDAT Tokenizer — Batch Demonstration over 10 Real EEG Epochs")
    print("=" * 88)
    header = (
        f"{'Epoch':<12}{'Shape':<14}{'IMFs':<7}{'Segments':<10}"
        f"{'L_tgt':<8}{'Pad':<6}{'Time (ms)':<10}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for i in range(1, N_EPOCHS + 1):
        name = f"Test EEG {i}"
        path = os.path.join(DATA_DIR, f"{name}.mat")
        if not os.path.exists(path):
            print(f"{name:<12} MISSING FILE, skipped")
            continue

        x_raw = load_mat_epoch(path)
        x = zscore_standardize(x_raw)

        t0 = time.perf_counter()
        result = tokenizer.transform(x)
        wall_ms = (time.perf_counter() - t0) * 1000.0

        n_segments = len(result.segmentation.boundaries) - 1
        n_padded = int(np.count_nonzero(result.resampling.selected_indices == -1))

        print(
            f"{name:<12}{str(x_raw.shape):<14}{result.dagaf.num_imfs:<7}"
            f"{n_segments:<10}{result.target_length:<8}{n_padded:<6}{wall_ms:<10.3f}"
        )

        fig_name = f"sedat_{name.lower().replace(' ', '_')}.png"
        fig_path = os.path.join(OUTPUT_DIR, fig_name)
        plot_pipeline_overview(x, result, fs=FS, save_path=fig_path, show=False)

        rows.append(
            dict(
                name=name,
                shape=x_raw.shape,
                num_imfs=result.dagaf.num_imfs,
                num_segments=n_segments,
                target_length=result.target_length,
                num_padded=n_padded,
                wall_ms=wall_ms,
                fig_path=fig_path,
            )
        )

    print("-" * len(header))
    total_ms = sum(r["wall_ms"] for r in rows)
    print(f"Processed {len(rows)}/{N_EPOCHS} epochs in {total_ms:.2f} ms total "
          f"({total_ms / max(len(rows), 1):.2f} ms/epoch).")
    print(f"\nFigures saved under: {OUTPUT_DIR}")
    for r in rows:
        print(f"  - {os.path.basename(r['fig_path'])}")


if __name__ == "__main__":
    main()
