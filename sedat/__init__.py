"""
SEDAT: SE-DAGAF Adaptive Tokenizer for large EEG foundation models.

Reference
---------
Aziz, M.Z., Yue, Z., Binwen, H., Xiaojun, Y. "SEDAT: A Hybrid Tokenizer for
Large EEG Models." Preprint submitted to IOP Journal of Neural Engineering.

Lin, Y.-D., Tan, Y.K., Tian, B. "A novel approach for decomposition of
biomedical signals in different applications based on data-adaptive
Gaussian average filtering." Biomedical Signal Processing and Control,
71 (2022) 103104.

The public API is intentionally small: configure a :class:`SEDATConfig`
and drive the whole pipeline through :class:`SEDATTokenizer`.
"""

from sedat.config import (
    DAGAFConfig,
    ResamplingConfig,
    SEDATConfig,
    SegmentationConfig,
    SpatialAggregationConfig,
)
from sedat.exceptions import DecompositionError, InvalidInputError, SEDATError
from sedat.tokenizer import SEDATTokenizer, TokenizationResult

__all__ = [
    "SEDATConfig",
    "SpatialAggregationConfig",
    "DAGAFConfig",
    "SegmentationConfig",
    "ResamplingConfig",
    "SEDATTokenizer",
    "TokenizationResult",
    "SEDATError",
    "InvalidInputError",
    "DecompositionError",
]

__version__ = "1.0.0"
