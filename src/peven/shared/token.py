"""Public token and marking helpers aligned with the implemented engine subset."""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias

import msgspec


StructuredPayload: TypeAlias = (
    None | bool | int | float | str | list["StructuredPayload"] | dict[str, "StructuredPayload"]
)
_STRUCTURED_INT_MIN: Final[int] = -(2**63)
_STRUCTURED_INT_MAX: Final[int] = 2**63 - 1

__all__ = [
    "Marking",
    "StructuredPayload",
    "Token",
    "marking",
    "run_keys",
    "run_marking",
    "token",
    "validate_structured_payload",
]


class Token(msgspec.Struct, frozen=True, kw_only=True):
    run_key: str
    color: str = "default"
    payload: object = None

    def __post_init__(self) -> None:
        validate_token_value(self)


def validate_structured_payload(value: object) -> None:
    if value is None or type(value) is bool or type(value) is str:
        return
    if type(value) is int:
        if not _STRUCTURED_INT_MIN <= value <= _STRUCTURED_INT_MAX:
            raise OverflowError("StructuredPayload ints must fit in signed 64-bit range")
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("StructuredPayload floats must be finite")
        return
    if type(value) is list:
        for item in value:
            validate_structured_payload(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("StructuredPayload maps must use string keys")
            validate_structured_payload(item)
        return
    raise TypeError(f"StructuredPayload unsupported type: {type(value).__name__}")


def token(
    payload: object = None,
    *,
    run_key: str,
    color: str = "default",
) -> Token:
    """Construct one token value after structured-payload validation."""
    validate_structured_payload(payload)
    _validate_token_fields(run_key=run_key, color=color)
    return Token(payload=payload, run_key=run_key, color=color)


@dataclass(frozen=True, slots=True)
class Marking:
    """One authored initial marking using concrete ``peven.Token`` values only."""

    tokens_by_place: dict[str, tuple[Token, ...]]

    def __init__(self, tokens_by_place: Mapping[str, Sequence[Token]] | None = None) -> None:
        normalized: dict[str, tuple[Token, ...]] = {}
        if tokens_by_place is not None:
            for place_name, bucket in tokens_by_place.items():
                if type(place_name) is not str or not place_name:
                    raise ValueError("marking place ids must be non-empty strings")
                if not isinstance(bucket, Sequence):
                    raise TypeError("marking buckets must be sequences of Token values")
                tokens: list[Token] = []
                for item in bucket:
                    if not isinstance(item, Token):
                        raise TypeError("marking buckets must contain Token values")
                    validate_token_value(item)
                    tokens.append(item)
                normalized[place_name] = tuple(tokens)
        object.__setattr__(self, "tokens_by_place", normalized)

    def to_dict(self) -> dict[str, list[Token]]:
        """Return a mutable dict/list view for later handoff normalization."""
        return {place: list(tokens) for place, tokens in self.tokens_by_place.items()}


def marking(*, run_key: str | None = None, **places: Sequence[object]) -> Marking:
    """Build a Marking from `place=[payload, ...]` kwargs, wrapping each payload as a token.

    A run_key is generated if not supplied, so every token in the marking shares it.
    Pass already-constructed tokens via the `Marking({...})` constructor instead.
    """
    rk = run_key if run_key is not None else uuid.uuid4().hex
    normalized: dict[str, list[Token]] = {}
    for place, payloads in places.items():
        if (
            type(payloads) is str
            or isinstance(payloads, Mapping)
            or not isinstance(payloads, Sequence)
        ):
            raise TypeError(
                "marking() buckets must be sequences of payloads; wrap single payloads in a list"
            )
        normalized[place] = [token(payload, run_key=rk) for payload in payloads]
    return Marking(normalized)


def run_keys(marking: Marking) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for tokens in marking.tokens_by_place.values():
        for token_value in tokens:
            if token_value.run_key in seen:
                continue
            seen.add(token_value.run_key)
            ordered.append(token_value.run_key)
    return ordered


def run_marking(marking: Marking, rk: str) -> Marking:
    if type(rk) is not str or not rk:
        raise ValueError("run_key must be a non-empty string")
    return Marking(
        {
            place: [token_value for token_value in tokens if token_value.run_key == rk]
            for place, tokens in marking.tokens_by_place.items()
            if any(token_value.run_key == rk for token_value in tokens)
        }
    )


def validate_token_value(token_value: Token) -> None:
    _validate_token_fields(run_key=token_value.run_key, color=token_value.color)
    validate_structured_payload(token_value.payload)


def _validate_token_fields(*, run_key: object, color: object) -> None:
    if type(run_key) is not str or not run_key:
        raise TypeError("token run_key must be a non-empty string")
    if type(color) is not str or not color:
        raise TypeError("token color must be a non-empty string")
