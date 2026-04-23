"""Shared token and token-bucket normalization helpers for handoff/runtime paths."""

from __future__ import annotations

from ..shared.token import Marking, Token, validate_token_value


def normalize_marking(marking: Marking) -> dict[str, list[Token]]:
    """Normalize one authored marking into validated token buckets."""
    return normalize_token_buckets(
        marking.to_dict(),
        container_name="run initial_marking",
    )


def normalize_token_bucket(
    value: object,
    *,
    message: str,
    expected_run_key: str | None = None,
    run_key_container: str | None = None,
    run_key_error_message: str | None = None,
) -> list[Token]:
    """Normalize one token or token list into one validated token bucket."""
    if type(value) is list:
        return validate_token_list(
            value,
            message=message,
            expected_run_key=expected_run_key,
            run_key_container=run_key_container,
        )
    if not isinstance(value, Token):
        raise TypeError(message)
    validate_token_value(value)
    if expected_run_key is not None and value.run_key != expected_run_key:
        if run_key_error_message is None:
            raise ValueError(f"{run_key_container} must preserve the bundle run_key")
        raise ValueError(run_key_error_message)
    return [value]


def normalize_token_buckets(
    buckets: object,
    *,
    container_name: str,
    expected_run_key: str | None = None,
) -> dict[str, list[Token]]:
    """Validate one mapping of place ids to token buckets."""
    if type(buckets) is not dict:
        raise TypeError(f"{container_name} must be a dict")
    for place, bucket in buckets.items():
        if type(place) is not str or not place:
            raise TypeError(f"{container_name} place ids must be non-empty strings")
        validate_token_list(
            bucket,
            message=f"{container_name} buckets must be lists of Token values",
            expected_run_key=expected_run_key,
            run_key_container=container_name,
        )
    return buckets


def validate_token_list(
    values: object,
    *,
    message: str,
    expected_run_key: str | None = None,
    run_key_container: str | None = None,
) -> list[Token]:
    """Validate one concrete list of Token values."""
    if type(values) is not list:
        raise TypeError(message)
    for token_value in values:
        if not isinstance(token_value, Token):
            raise TypeError(message)
        validate_token_value(token_value)
        if expected_run_key is not None and token_value.run_key != expected_run_key:
            raise ValueError(f"{run_key_container} must preserve the bundle run_key")
    return values
