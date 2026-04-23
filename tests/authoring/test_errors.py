from __future__ import annotations

from peven.shared.errors import PevenValidationError, ValidationIssue


def test_validation_error_formats_empty_and_message_less_issues() -> None:
    assert str(PevenValidationError([])) == "peven validation failed"
    assert str(PevenValidationError([object()])) == "peven validation failed"
    issue = ValidationIssue(code="bad", object_id="x", message="boom")
    assert str(PevenValidationError([issue])) == "boom"
