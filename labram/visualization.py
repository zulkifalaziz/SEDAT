"""Diagnostic plotting for the SEDAT -> LaBraM-style classification experiment."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from labram.evaluation import CrossValidationResult


def plot_classification_report(
    result: CrossValidationResult,
    class_names: Optional[Sequence[str]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Render a 3-panel figure: per-fold accuracy, training loss curves, and
    a pooled confusion matrix across all held-out folds.
    """
    import matplotlib.pyplot as plt

    n_classes = int(max(result.pooled_y_true.max(), result.pooled_y_pred.max()) + 1)
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # --- Panel 1: per-fold accuracy -------------------------------------
    ax = axes[0]
    accs = result.accuracies
    bars = ax.bar(np.arange(len(accs)) + 1, accs * 100, color="steelblue")
    ax.axhline(result.mean_accuracy * 100, color="crimson", linestyle="--",
               label=f"mean = {result.mean_accuracy * 100:.1f}% ± {result.std_accuracy * 100:.1f}%")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{acc * 100:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.axhline(100.0 / n_classes, color="gray", linestyle=":", label="chance level")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Held-out accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(len(accs)) + 1)
    ax.set_title("Per-fold classification accuracy")
    ax.legend(fontsize=8, loc="lower right")

    # --- Panel 2: training loss curves -----------------------------------
    ax = axes[1]
    for f in result.folds:
        ax.plot(np.arange(1, len(f.train_losses) + 1), f.train_losses, label=f"fold {f.fold + 1}")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Mean training loss")
    ax.set_title("Training loss per fold")
    ax.legend(fontsize=8)

    # --- Panel 3: pooled confusion matrix ---------------------------------
    ax = axes[2]
    y_true, y_pred = result.pooled_y_true, result.pooled_y_pred
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    im = ax.imshow(cm, cmap="Blues")
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Pooled confusion matrix (all held-out folds)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "SEDAT Tokens -> LaBraM-style Classifier: 5-Fold Stratified CV on MI-D1",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig
