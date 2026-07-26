"""
Diagnostic plotting utilities for the SEDAT pipeline.

These are intentionally kept separate from the numerical stages: importing
``sedat`` never requires matplotlib. Only call into this module when a
figure is actually needed (e.g. from an example script or notebook).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def plot_pipeline_overview(
    x: np.ndarray,
    result,
    fs: float,
    channel_names: Optional[list[str]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Render a multi-panel figure summarizing every SEDAT stage.

    Parameters
    ----------
    x:
        Raw input trial, shape (T, C).
    result:
        A :class:`sedat.tokenizer.TokenizationResult` produced by
        ``SEDATTokenizer.transform``.
    fs:
        Sampling rate in Hz (used only to build a time axis).
    channel_names:
        Optional channel labels for the raw-signal panel legend.
    save_path:
        If given, the figure is saved to this path (any matplotlib-
        supported extension, e.g. ``.png``).
    show:
        If True, calls ``plt.show()`` after rendering.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    t_axis = np.arange(x.shape[0]) / fs
    n_imfs = result.dagaf.num_imfs
    n_channels = x.shape[1]

    fig = plt.figure(figsize=(14, 3 * (3 + max(1, n_imfs) * 0.35)))
    n_rows = 4
    gs = GridSpec(n_rows, 2, figure=fig, height_ratios=[1.3, 1.0, max(1.0, 0.5 * n_imfs), 1.3])

    # --- Panel 1: raw multi-channel signal -----------------------------
    ax_raw = fig.add_subplot(gs[0, :])
    max_channels_to_plot = min(n_channels, 8)
    for c in range(max_channels_to_plot):
        label = channel_names[c] if channel_names else f"ch{c}"
        ax_raw.plot(t_axis, x[:, c], lw=0.6, alpha=0.7, label=label)
    ax_raw.set_title(f"Step 0 — Raw EEG input (showing {max_channels_to_plot}/{n_channels} channels)")
    ax_raw.set_ylabel("Amplitude")
    ax_raw.legend(loc="upper right", ncol=min(max_channels_to_plot, 4), fontsize=7)

    # --- Panel 2: SE drive signal + channel weights ---------------------
    ax_drive = fig.add_subplot(gs[1, 0])
    ax_drive.plot(t_axis, result.spatial.drive_signal, color="black", lw=0.8)
    ax_drive.set_title("Step 1 — SE-aggregated drive signal s(t)")
    ax_drive.set_ylabel("Amplitude")

    ax_weights = fig.add_subplot(gs[1, 1])
    ax_weights.bar(np.arange(n_channels), result.spatial.channel_weights, color="steelblue")
    ax_weights.set_title("Channel weights $\\omega_c$")
    ax_weights.set_xlabel("Channel index")
    ax_weights.set_ylabel("Weight")

    # --- Panel 3: DAGAF IMFs --------------------------------------------
    ax_imfs = fig.add_subplot(gs[2, :])
    offset = 0.0
    spacing = 3.0 * (np.std(result.spatial.drive_signal) + 1e-9)
    for k, imf in enumerate(result.dagaf.imfs):
        ax_imfs.plot(t_axis, imf - offset, lw=0.7, label=f"IMF {k + 1}")
        offset += spacing
    ax_imfs.plot(t_axis, result.dagaf.residual - offset, lw=0.9, color="black", label="residual")
    ax_imfs.set_title(f"Step 2 — DAGAF decomposition ({n_imfs} IMFs, stop: {result.dagaf.stop_reason})")
    ax_imfs.set_yticks([])
    ax_imfs.legend(loc="upper right", ncol=min(n_imfs + 1, 6), fontsize=7)

    # --- Panel 4: segmentation boundaries on drive signal ---------------
    ax_seg = fig.add_subplot(gs[3, 0])
    ax_seg.plot(t_axis, result.spatial.drive_signal, color="gray", lw=0.7)
    for b in result.segmentation.boundaries:
        ax_seg.axvline(b / fs, color="crimson", lw=1.0, linestyle="--", alpha=0.8)
    ax_seg.set_title("Step 3 — Adaptive segmentation boundaries")
    ax_seg.set_xlabel("Time (s)")
    ax_seg.set_ylabel("Amplitude")

    # --- Panel 5: final tokens (mean power per token per channel) -------
    ax_tok = fig.add_subplot(gs[3, 1])
    tokens = result.tokens  # (K, L_tgt, C)
    token_power = np.mean(tokens ** 2, axis=1)  # (K, C)
    im = ax_tok.imshow(token_power.T, aspect="auto", cmap="viridis", origin="lower")
    ax_tok.set_title(f"Step 4 — Token power map ({tokens.shape[0]} tokens x {tokens.shape[2]} ch, L_tgt={tokens.shape[1]})")
    ax_tok.set_xlabel("Token index")
    ax_tok.set_ylabel("Channel index")
    fig.colorbar(im, ax=ax_tok, label="Mean power")

    fig.suptitle("SEDAT Tokenization Pipeline — Stage-by-Stage Overview", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig
