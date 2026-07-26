"""Custom exception hierarchy for the SEDAT package."""


class SEDATError(Exception):
    """Base class for all SEDAT-related errors."""


class InvalidInputError(SEDATError):
    """Raised when the input EEG array fails validation."""


class DecompositionError(SEDATError):
    """Raised when the DAGAF decomposition stage cannot proceed."""


class SegmentationError(SEDATError):
    """Raised when the adaptive segmentation stage fails."""
