class DiffSurfaceError(Exception):
    """Base exception for differential surface workflow errors."""


class ValidationError(DiffSurfaceError):
    """Raised when user input or file structure is invalid."""


class CsvValidationError(ValidationError):
    """Raised when the CSV file cannot be accepted for processing."""


class ProcessingError(DiffSurfaceError):
    """Raised when the differential surface cannot be calculated."""


class SaveError(DiffSurfaceError):
    """Raised when result export fails."""


class CancellationError(DiffSurfaceError):
    """Raised when background processing is cancelled."""
