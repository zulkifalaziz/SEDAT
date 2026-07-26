"""
SEDAT -> LaBraM-style classification experiment on the full MI-D1 dataset.

Tokenizes every epoch in the MI EEG Dataset IVa (BCI Competition III, 100 Hz
sampling rate variant — 560 epochs across 5 subjects, 2 motor-imagery
classes) with SEDAT, then trains and evaluates a lightweight LaBraM-style
transformer classifier (see ``labram/``) using subject-pooled, stratified
5-fold cross-validation, following the SEDAT manuscript's evaluation
protocol for multi-subject datasets (Section 2.4).

This uses a from-scratch reimplementation of LaBraM's architectural
pattern, not the official pretrained checkpoint (see ``labram/__init__.py``
for why) — the goal is to demonstrate that SEDAT's token tensor integrates
correctly with a LaBraM-style backbone end-to-end and produces a real
classification result on real EEG data, not to reproduce the manuscript's
exact reported accuracy.

Run with::

    python examples/run_labram_classification.py
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from labram.dataset import build_tokenized_dataset
from labram.evaluation import run_cross_validation
from labram.visualization import plot_classification_report

DATASET_DIR = r"E:\Research\Datasets\1. MI EEG Dataset IVa from BCI Competition III\Fs=100"
SAMPLING_RATE = 100.0
NUM_TOKENS = 5
TARGET_LENGTH = 64  # fixed SEDAT token length so all epochs batch together

N_FOLDS = 5
NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
CLASS_NAMES = ["Right hand", "Right foot"]  # MI-D1 class semantics, Table 1


def main() -> None:
    print("=" * 88)
    print("SEDAT -> LaBraM-style Classifier: Full MI-D1 Evaluation")
    print("=" * 88)

    print(f"\nScanning and tokenizing epochs from:\n  {DATASET_DIR}")
    t0 = time.perf_counter()
    dataset = build_tokenized_dataset(
        dataset_dir=DATASET_DIR,
        sampling_rate=SAMPLING_RATE,
        num_tokens=NUM_TOKENS,
        target_length=TARGET_LENGTH,
    )
    tokenize_s = time.perf_counter() - t0

    n_total = dataset.tokens.shape[0]
    n_class0 = int((dataset.labels == 0).sum())
    n_class1 = int((dataset.labels == 1).sum())
    n_subjects = len(set(dataset.subjects.tolist()))

    print(f"  Tokenized {n_total} epochs in {tokenize_s:.2f} s "
          f"({tokenize_s / n_total * 1000:.2f} ms/epoch)")
    print(f"  Token tensor shape per epoch: "
          f"({dataset.num_tokens}, {dataset.token_length}, {dataset.num_channels})")
    print(f"  Subjects: {n_subjects}   Class 0: {n_class0}   Class 1: {n_class1}")

    print(f"\nRunning {N_FOLDS}-fold stratified CV "
          f"({NUM_EPOCHS} epochs/fold, batch size {BATCH_SIZE}) ...")
    t0 = time.perf_counter()
    result = run_cross_validation(
        tokens=dataset.tokens,
        labels=dataset.labels,
        n_folds=N_FOLDS,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
    )
    cv_s = time.perf_counter() - t0

    print(f"\nDone in {cv_s:.1f} s.\n")
    print(f"{'Fold':<8}{'Test Acc.':<12}{'Train time (s)':<16}")
    for f in result.folds:
        print(f"{f.fold + 1:<8}{f.test_accuracy * 100:<12.2f}{f.train_time_s:<16.2f}")
    print("-" * 36)
    print(f"Mean accuracy: {result.mean_accuracy * 100:.2f}% "
          f"± {result.std_accuracy * 100:.2f}% (chance level: "
          f"{100.0 / len(set(dataset.labels.tolist())):.1f}%)")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "sedat_labram_classification.png")
    plot_classification_report(result, class_names=CLASS_NAMES, save_path=fig_path)
    print(f"\nFigure saved to: {fig_path}")


if __name__ == "__main__":
    main()
