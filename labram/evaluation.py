"""
Subject-pooled, stratified K-fold cross-validation of the LaBraM-style
classifier on SEDAT-tokenized EEG epochs.

Follows the SEDAT manuscript's evaluation protocol for multi-subject
datasets (Section 2.4): "epochs across all subjects were pooled prior to
applying 5-fold stratified cross-validation." Each fold trains a freshly
initialized model from scratch (we do not have LaBraM's original
pretrained weights, per the from-scratch reimplementation documented in
``labram/__init__.py``) and evaluates held-out accuracy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from labram.config import LaBraMConfig
from labram.model import LaBraMStyleClassifier


@dataclass
class FoldResult:
    fold: int
    train_losses: List[float] = field(default_factory=list)
    test_accuracy: float = 0.0
    y_true: np.ndarray = field(default_factory=lambda: np.empty(0))
    y_pred: np.ndarray = field(default_factory=lambda: np.empty(0))
    train_time_s: float = 0.0


@dataclass
class CrossValidationResult:
    folds: List[FoldResult]

    @property
    def accuracies(self) -> np.ndarray:
        return np.array([f.test_accuracy for f in self.folds])

    @property
    def mean_accuracy(self) -> float:
        return float(self.accuracies.mean())

    @property
    def std_accuracy(self) -> float:
        return float(self.accuracies.std())

    @property
    def pooled_y_true(self) -> np.ndarray:
        return np.concatenate([f.y_true for f in self.folds])

    @property
    def pooled_y_pred(self) -> np.ndarray:
        return np.concatenate([f.y_pred for f in self.folds])


def _train_one_fold(
    fold_idx: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model_config: LaBraMConfig,
    num_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> FoldResult:
    torch.manual_seed(seed + fold_idx)

    model = LaBraMStyleClassifier(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )

    t0 = time.perf_counter()
    losses: List[float] = []
    model.train()
    for _epoch in range(num_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))
    train_time = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        x_test_t = torch.from_numpy(x_test).to(device)
        logits = model(x_test_t)
        y_pred = logits.argmax(dim=1).cpu().numpy()

    accuracy = float((y_pred == y_test).mean())

    return FoldResult(
        fold=fold_idx,
        train_losses=losses,
        test_accuracy=accuracy,
        y_true=y_test,
        y_pred=y_pred,
        train_time_s=train_time,
    )


def run_cross_validation(
    tokens: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    num_epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    embed_dim: int = 64,
    depth: int = 4,
    num_heads: int = 4,
    seed: int = 0,
    device: str = "cpu",
) -> CrossValidationResult:
    """Run stratified K-fold CV of the LaBraM-style classifier.

    Parameters
    ----------
    tokens:
        SEDAT token tensor of shape (N, K, L_tgt, C).
    labels:
        Integer class labels of shape (N,).
    """
    n_tokens, token_length, num_channels = tokens.shape[1:]
    num_classes = int(labels.max() + 1)

    model_config = LaBraMConfig(
        num_tokens=n_tokens,
        token_length=token_length,
        num_channels=num_channels,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
    )

    torch_device = torch.device(device)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_results: List[FoldResult] = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(tokens, labels)):
        result = _train_one_fold(
            fold_idx=fold_idx,
            x_train=tokens[train_idx],
            y_train=labels[train_idx],
            x_test=tokens[test_idx],
            y_test=labels[test_idx],
            model_config=model_config,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            device=torch_device,
            seed=seed,
        )
        fold_results.append(result)

    return CrossValidationResult(folds=fold_results)
