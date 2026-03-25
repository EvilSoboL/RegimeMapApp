class ApproximationError(Exception):
    """Base exception for approximation workflow errors."""


class ValidationError(ApproximationError):
    """Raised when user input or file structure is invalid."""


class CsvValidationError(ValidationError):
    """Raised when a CSV file cannot be accepted for processing."""


class ProcessingError(ApproximationError):
    """Raised when approximation or export fails."""


class CancellationError(ApproximationError):
    """Raised when background processing is cancelled."""
