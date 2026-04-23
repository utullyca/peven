"""Errors and validation records for the implemented peven layers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


__all__ = ["PevenValidationError", "ValidationIssue"]


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One Python-side validation issue surfaced before engine validation."""

    code: str
    object_id: str
    message: str


class PevenValidationError(RuntimeError):
    """Raised when authoring or handoff validation fails on the Python side."""

    def __init__(self, issues: Sequence[object]) -> None:
        self.issues = tuple(issues)
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if not self.issues:
            return "peven validation failed"
        issue = self.issues[0]
        message = getattr(issue, "message", None)
        if isinstance(message, str) and message:
            return message
        return "peven validation failed"
