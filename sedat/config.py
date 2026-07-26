"""
Configuration objects for every stage of the SEDAT pipeline.

All numeric defaults reproduce the values reported in the SEDAT manuscript
(Section 2.1) and the DAGAF paper (Lin et al., 2022). Each dataclass
validates its own parameters in ``__post_init__`` so misconfiguration is
caught immediately at construction time rather than deep inside the
numerical pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from sedat.exceptions import InvalidInputError


@dataclass(frozen=True)
class SpatialAggregationConfig:
    """Squeeze-and-excitation (SE) spatial aggregation (SEDAT Step 1)."""

    gamma: float = 10.0
    """Sigmoid temperature, Eq. (3)."""

    delta: float = 0.5
    """Sigmoid shift center, Eq. (3)."""

    eps: float = 1e-8
    """Numerical guard against zero-division in min-max normalization, Eq. (2)."""

    enabled: bool = True
    """If False (or if the input is single-channel) the raw signal is used
    directly as the drive signal and this stage is skipped, matching the
    SEDAT experimental protocol for single-channel datasets."""

    def __post_init__(self) -> None:
        if self.gamma <= 0:
            raise InvalidInputError("gamma must be positive.")
        if self.eps <= 0:
            raise InvalidInputError("eps must be positive.")


@dataclass(frozen=True)
class DAGAFConfig:
    """Data-adaptive Gaussian average filtering decomposition (SEDAT Step 2)."""

    chi: float = 2.0
    """Scalar controlling the Gaussian window half-length, Eq. (7)."""

    alpha: float = 3.0
    """Gaussian window shape parameter, Eq. (8)."""

    max_imfs: int = 8
    """Maximum decomposition depth K_max = K, i.e. target token count."""

    energy_tol: float = 1e-10
    """Stopping criterion: extracted IMF energy below this value halts sifting."""

    def __post_init__(self) -> None:
        if self.chi <= 0:
            raise InvalidInputError("chi must be positive.")
        if self.alpha <= 0:
            raise InvalidInputError("alpha must be positive.")
        if self.max_imfs < 1:
            raise InvalidInputError("max_imfs must be >= 1.")
        if self.energy_tol <= 0:
            raise InvalidInputError("energy_tol must be positive.")


@dataclass(frozen=True)
class SegmentationConfig:
    """Instantaneous-frequency-guided adaptive segmentation (SEDAT Step 3)."""

    min_segment_length: int = 4
    """Minimum admissible segment length L_min (samples), Eq. (18)."""

    def __post_init__(self) -> None:
        if self.min_segment_length < 1:
            raise InvalidInputError("min_segment_length must be >= 1.")


@dataclass(frozen=True)
class ResamplingConfig:
    """Fourier-domain resampling and token formation (SEDAT Step 4)."""

    target_length: Optional[int] = None
    """Fixed token length L_tgt. If None, it is inferred at run time as the
    median length of the produced segments (capped as documented in
    SEDATConfig.max_target_length)."""

    def __post_init__(self) -> None:
        if self.target_length is not None and self.target_length < 2:
            raise InvalidInputError("target_length must be >= 2 when specified.")


@dataclass(frozen=True)
class SEDATConfig:
    """Top-level configuration bundling every SEDAT stage.

    Parameters
    ----------
    sampling_rate:
        Sampling frequency f_s (Hz) of the input EEG.
    num_tokens:
        Target number of tokens K. Drives the DAGAF maximum decomposition
        depth, the number of segmentation boundaries, and the number of
        segments retained during resampling.
    max_target_length:
        Upper bound applied when ``resampling.target_length`` is inferred
        automatically, preventing pathologically long tensors on
        long-duration epochs.
    """

    sampling_rate: float = 250.0
    num_tokens: int = 8
    max_target_length: int = 512

    spatial: SpatialAggregationConfig = field(default_factory=SpatialAggregationConfig)
    dagaf: DAGAFConfig = field(default_factory=lambda: DAGAFConfig(max_imfs=8))
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    resampling: ResamplingConfig = field(default_factory=ResamplingConfig)

    def __post_init__(self) -> None:
        if self.sampling_rate <= 0:
            raise InvalidInputError("sampling_rate must be positive.")
        if self.num_tokens < 1:
            raise InvalidInputError("num_tokens must be >= 1.")
        if self.max_target_length < 2:
            raise InvalidInputError("max_target_length must be >= 2.")
        # Keep the DAGAF decomposition depth consistent with num_tokens
        # unless the caller explicitly overrode it to something different.
        if self.dagaf.max_imfs != self.num_tokens:
            object.__setattr__(
                self, "dagaf", DAGAFConfig(
                    chi=self.dagaf.chi,
                    alpha=self.dagaf.alpha,
                    max_imfs=self.num_tokens,
                    energy_tol=self.dagaf.energy_tol,
                )
            )
