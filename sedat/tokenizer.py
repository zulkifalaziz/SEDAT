"""
SEDAT orchestrator — chains spatial aggregation, DAGAF decomposition,
adaptive segmentation, and Fourier resampling into a single tokenization
pipeline (Section 2.1 of the SEDAT manuscript).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from sedat.config import SEDATConfig
from sedat.decomposition import DAGAFDecomposer, DAGAFResult
from sedat.resampling import FourierResampler, ResamplingResult
from sedat.segmentation import AdaptiveSegmenter, SegmentationResult
from sedat.spatial_aggregation import SpatialAggregationResult, SpatialAggregator
from sedat.utils import validate_eeg_array

logger = logging.getLogger("sedat")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


@dataclass
class TokenizationResult:
    """Full output of tokenizing a single EEG trial, including every
    intermediate stage's result for diagnostics and visualization."""

    tokens: np.ndarray
    """Final tensor Z of shape (K, L_tgt, C)."""

    spatial: SpatialAggregationResult
    dagaf: DAGAFResult
    segmentation: SegmentationResult
    resampling: ResamplingResult

    target_length: int
    stage_timings_ms: dict = field(default_factory=dict)

    @property
    def total_time_ms(self) -> float:
        return sum(self.stage_timings_ms.values())


class SEDATTokenizer:
    """SE-DAGAF Adaptive Tokenizer (SEDAT).

    Transforms a raw multi-channel EEG trial ``X`` of shape ``(T, C)`` into
    a fixed-dimensional tensor ``Z`` of shape ``(K, L_tgt, C)`` through four
    sequential stages: spatial aggregation, signal decomposition, adaptive
    segmentation, and Fourier-domain resampling.

    Examples
    --------
    >>> import numpy as np
    >>> from sedat import SEDATConfig, SEDATTokenizer
    >>> x = np.random.randn(1000, 16)
    >>> tokenizer = SEDATTokenizer(SEDATConfig(sampling_rate=250.0, num_tokens=6))
    >>> result = tokenizer.transform(x)
    >>> result.tokens.shape
    (6, 512, 16)
    """

    def __init__(self, config: Optional[SEDATConfig] = None) -> None:
        self.config = config or SEDATConfig()
        self._spatial = SpatialAggregator(self.config.spatial)
        self._decomposer = DAGAFDecomposer(self.config.dagaf)
        self._segmenter = AdaptiveSegmenter(self.config.segmentation)
        self._resampler = FourierResampler(self.config.resampling)

    def transform(self, x: np.ndarray) -> TokenizationResult:
        """Tokenize a single EEG trial.

        Parameters
        ----------
        x:
            Array of shape ``(T, C)`` (or ``(T,)`` for single-channel data).
        """
        x = validate_eeg_array(x)
        timings: dict = {}

        t0 = time.perf_counter()
        spatial_result = self._spatial.aggregate(x)
        timings["spatial_aggregation"] = _elapsed_ms(t0)

        t0 = time.perf_counter()
        dagaf_result = self._decomposer.decompose(spatial_result.drive_signal)
        timings["decomposition"] = _elapsed_ms(t0)

        if dagaf_result.num_imfs == 0:
            logger.warning(
                "DAGAF produced zero IMFs (stop reason: %s); falling back to a "
                "single full-length segment.",
                dagaf_result.stop_reason,
            )

        t0 = time.perf_counter()
        seg_result = self._segmenter.segment(
            dagaf_result.imfs, n_samples=x.shape[0], fs=self.config.sampling_rate
        )
        timings["segmentation"] = _elapsed_ms(t0)

        target_length = self._resolve_target_length(seg_result)

        t0 = time.perf_counter()
        resample_result = self._resampler.tokenize(
            x,
            boundaries=seg_result.boundaries,
            num_tokens=self.config.num_tokens,
            target_length=target_length,
        )
        timings["resampling"] = _elapsed_ms(t0)

        return TokenizationResult(
            tokens=resample_result.tokens,
            spatial=spatial_result,
            dagaf=dagaf_result,
            segmentation=seg_result,
            resampling=resample_result,
            target_length=target_length,
            stage_timings_ms=timings,
        )

    def transform_batch(
        self, epochs: List[np.ndarray], n_jobs: int = 1
    ) -> List[TokenizationResult]:
        """Tokenize multiple independent EEG trials.

        Each trial is tokenized independently (boundaries are trial-
        specific), so batches are embarrassingly parallel. Set ``n_jobs >
        1`` to distribute trials across processes for large datasets.
        """
        if n_jobs == 1 or len(epochs) <= 1:
            return [self.transform(epoch) for epoch in epochs]

        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            results = list(pool.map(_transform_with_config, epochs, [self.config] * len(epochs)))
        return results

    def _resolve_target_length(self, seg_result: SegmentationResult) -> int:
        configured = self.config.resampling.target_length
        if configured is not None:
            return configured

        lengths = np.diff(seg_result.boundaries)
        if lengths.size == 0:
            return min(2, self.config.max_target_length)

        median_len = int(np.median(lengths))
        median_len = max(2, median_len)
        return min(median_len, self.config.max_target_length)


def _elapsed_ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _transform_with_config(epoch: np.ndarray, config: SEDATConfig) -> TokenizationResult:
    """Module-level helper so batch tokenization is picklable for
    :class:`~concurrent.futures.ProcessPoolExecutor`."""
    return SEDATTokenizer(config).transform(epoch)
