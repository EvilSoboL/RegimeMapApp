class RegimeMapError(Exception):
    """Base exception for regime map workflow errors."""


class ValidationError(RegimeMapError):
    """Raised when user input or file structure is invalid."""


class ProcessingError(RegimeMapError):
    """Raised when the regime map cannot be calculated."""


class SaveError(RegimeMapError):
    """Raised when result export fails."""


class CancellationError(RegimeMapError):
    """Raised when background processing is cancelled."""
