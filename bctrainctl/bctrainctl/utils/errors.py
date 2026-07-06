"""Application error types."""

from __future__ import annotations


class BCTrainctlError(Exception):
    """Base exception with optional recovery hint."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class ConfigNotFoundError(BCTrainctlError):
    """Raised when local configuration is missing."""


class ValidationFailedError(BCTrainctlError):
    """Raised when user input or job spec is invalid."""


class ClientOperationError(BCTrainctlError):
    """Raised when a cloud client action fails."""


class NotFoundError(BCTrainctlError):
    """Raised when a local or remote job cannot be found."""

