"""Shared private helpers for authoring DSL expression trees."""

from __future__ import annotations

from typing import NoReturn, TypeAlias, TypeVar, cast

from ..shared.token import validate_structured_payload


ScalarStructured: TypeAlias = None | bool | int | float | str
_LiteralNodeT = TypeVar("_LiteralNodeT")


def require_identifier_segment(name: str, *, error_message: str) -> str:
    if not name.isidentifier():
        raise ValueError(error_message)
    return name


def root_identifier_path(name: str, *, error_message: str) -> tuple[str, ...]:
    return (require_identifier_segment(name, error_message=error_message),)


def extend_identifier_path(
    path: tuple[str, ...], name: str, *, error_message: str
) -> tuple[str, ...]:
    return (*path, require_identifier_segment(name, error_message=error_message))


def require_identifier_path(path: tuple[str, ...], *, error_message: str) -> None:
    if not path:
        raise ValueError(error_message)
    for segment in path:
        if type(segment) is not str or not segment.isidentifier():
            raise ValueError(error_message)


def reject_indexing(*, error_message: str) -> NoReturn:
    raise TypeError(error_message)


def coerce_scalar_literal(
    value: object,
    *,
    base_type: type[object],
    literal_type: type[_LiteralNodeT],
    error_message: str,
) -> _LiteralNodeT:
    if isinstance(value, base_type):
        if not isinstance(value, literal_type):
            raise TypeError(error_message)
        return value
    if value is None or isinstance(value, (bool, int, float, str)):
        scalar_value = cast(ScalarStructured, value)
    else:
        raise TypeError(error_message)
    validate_structured_payload(scalar_value)
    return literal_type(scalar_value)
