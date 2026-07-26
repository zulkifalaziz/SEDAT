"""
Dataset assembly: scan the MI EEG Dataset IVa folder, parse subject/class
labels from filenames, run every epoch through the SEDAT tokenizer with a
fixed target length, and pack the result into tensors ready for training.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import scipy.io as sio

from sedat import SEDATConfig, SEDATTokenizer
from sedat.config import ResamplingConfig
from sedat.utils import zscore_standardize

logger = logging.getLogger("labram.dataset")

_FILENAME_RE = re.compile(r"Subject_(\d+) Class(\d+) (\d+)\.mat")


@dataclass(frozen=True)
class TokenizedDataset:
    """SEDAT-tokenized version of the full MI-D1 dataset."""

    tokens: np.ndarray
    """Float32 array of shape (N, K, L_tgt, C)."""

    labels: np.ndarray
    """Int64 array of shape (N,) — class labels (0/1)."""

    subjects: np.ndarray
    """Int64 array of shape (N,) — subject id per epoch, for diagnostics."""

    num_tokens: int
    token_length: int
    num_channels: int


def _scan_epoch_files(dataset_dir: str) -> List[tuple]:
    """Return a sorted list of (path, subject, cls) for every epoch file."""
    entries = []
    for path in Path(dataset_dir).glob("*.mat"):
        match = _FILENAME_RE.match(path.name)
        if not match:
            continue
        subject, cls, _idx = (int(g) for g in match.groups())
        entries.append((path, subject, cls))
    entries.sort(key=lambda e: e[0].name)
    return entries


def build_tokenized_dataset(
    dataset_dir: str,
    sampling_rate: float,
    num_tokens: int,
    target_length: int,
    limit: int | None = None,
) -> TokenizedDataset:
    """Load every epoch in ``dataset_dir``, tokenize with SEDAT, and stack.

    A fixed ``target_length`` is required (rather than SEDAT's auto-inferred
    per-trial default) so every epoch produces an identically shaped tensor
    and can be batched by the downstream classifier.

    Parameters
    ----------
    dataset_dir:
        Folder containing ``Subject_<S> Class<C> <idx>.mat`` files.
    sampling_rate:
        Sampling rate in Hz of the epochs.
    num_tokens:
        SEDAT's target token count K.
    target_length:
        Fixed SEDAT token length L_tgt.
    limit:
        If given, only the first ``limit`` files (after sorting) are used.
        Useful for fast smoke tests before running the full dataset.
    """
    entries = _scan_epoch_files(dataset_dir)
    if not entries:
        raise FileNotFoundError(f"No 'Subject_<S> Class<C> <idx>.mat' files found in {dataset_dir}")
    if limit is not None:
        entries = entries[:limit]

    config = SEDATConfig(
        sampling_rate=sampling_rate,
        num_tokens=num_tokens,
        resampling=ResamplingConfig(target_length=target_length),
    )
    tokenizer = SEDATTokenizer(config)

    tokens: List[np.ndarray] = []
    labels: List[int] = []
    subjects: List[int] = []

    t0 = time.perf_counter()
    for i, (path, subject, cls) in enumerate(entries):
        mat = sio.loadmat(path)
        x_raw = np.asarray(mat["Data"], dtype=np.float64)
        x = zscore_standardize(x_raw)
        result = tokenizer.transform(x)
        tokens.append(result.tokens.astype(np.float32))
        labels.append(cls)
        subjects.append(subject)

        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            logger.info("Tokenized %d/%d epochs (%.1f s elapsed)", i + 1, len(entries), elapsed)

    num_channels = tokens[0].shape[-1]
    return TokenizedDataset(
        tokens=np.stack(tokens, axis=0),
        labels=np.array(labels, dtype=np.int64),
        subjects=np.array(subjects, dtype=np.int64),
        num_tokens=num_tokens,
        token_length=target_length,
        num_channels=num_channels,
    )
