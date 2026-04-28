"""Declarative authoring constructors for peven topology."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..shared.errors import PevenValidationError, ValidationIssue
from .guard import GuardNode, validate_guard_tree
from .join import JoinNode, validate_join_tree


__all__ = ["InputDecl", "OutputDecl", "PlaceDecl", "TransitionDecl", "input", "output", "place", "transition"]


@dataclass(frozen=True, slots=True)
class PlaceDecl:
    """A place declaration captured from an env class body."""

    capacity: int | None = None
    schema: object | None = None
    terminal: bool = False


@dataclass(frozen=True, slots=True)
class InputDecl:
    """An input-arc declaration captured from one transition decorator."""

    place: str
    weight: int = 1
    optional: bool = False


@dataclass(frozen=True, slots=True)
class OutputDecl:
    """An output-arc declaration captured from one transition decorator."""

    place: str


@dataclass(frozen=True, slots=True)
class TransitionDecl:
    """Transition metadata captured from the decorator surface."""

    executor: str
    inputs: tuple[InputDecl, ...]
    outputs: tuple[OutputDecl, ...]
    guard: GuardNode | None = None
    retries: int = 0
    join_by: JoinNode | None = None


def place(
    *,
    capacity: int | None = None,
    schema: object | None = None,
    terminal: bool = False,
) -> PlaceDecl:
    """Declare one place on an env class body."""
    if capacity is not None and (type(capacity) is not int or capacity <= 0):
        raise ValueError("place capacity must be a positive int or None")
    if type(terminal) is not bool:
        raise TypeError("place terminal must be a bool")
    return PlaceDecl(capacity=capacity, schema=schema, terminal=terminal)


def input(place: str, *, weight: int = 1, optional: bool = False) -> InputDecl:
    """Construct one input arc declaration."""
    _validate_place_name(place)
    if type(weight) is not int or weight <= 0:
        raise ValueError("input weight must be a positive int")
    if type(optional) is not bool:
        raise TypeError("input optional must be a bool")
    return InputDecl(place=place, weight=weight, optional=optional)


def output(place: str) -> OutputDecl:
    """Construct one output arc declaration."""
    _validate_place_name(place)
    return OutputDecl(place=place)


def transition(
    *,
    inputs: Sequence[str | InputDecl] | str | InputDecl,
    outputs: Sequence[str | OutputDecl] | str | OutputDecl,
    executor: str,
    guard: GuardNode | None = None,
    retries: int = 0,
    join_by: JoinNode | None = None,
) -> TransitionDecl:
    """Construct one transition declaration to assign on an env class body."""
    if type(executor) is not str or not executor:
        raise ValueError("transition executor must be a non-empty string")
    normalized_inputs = _normalize_input_decls(inputs)
    normalized_outputs = _normalize_output_decls(outputs)
    if type(retries) is not int or retries < 0:
        raise ValueError("transition retries must be a non-negative int")
    if guard is not None:
        if not isinstance(guard, GuardNode):
            raise PevenValidationError(
                [
                    ValidationIssue(
                        "invalid_guard",
                        "<transition>",
                        "guard must be a peven.guard expression",
                    )
                ]
            )
        if not guard.produces_bool:
            raise PevenValidationError(
                [
                    ValidationIssue(
                        "invalid_guard",
                        "<transition>",
                        "guard root must produce a boolean",
                    )
                ]
            )
        try:
            validate_guard_tree(guard)
        except (TypeError, ValueError) as exc:
            raise PevenValidationError(
                [ValidationIssue("invalid_guard", "<transition>", str(exc))]
            ) from exc
    if join_by is not None:
        if not isinstance(join_by, JoinNode):
            raise PevenValidationError(
                [
                    ValidationIssue(
                        "invalid_join_by",
                        "<transition>",
                        "join_by must be a peven.join selector",
                    )
                ]
            )
        try:
            validate_join_tree(join_by)
        except (TypeError, ValueError) as exc:
            raise PevenValidationError(
                [ValidationIssue("invalid_join_by", "<transition>", str(exc))]
            ) from exc

    return TransitionDecl(
        executor=executor,
        inputs=normalized_inputs,
        outputs=normalized_outputs,
        guard=guard,
        retries=retries,
        join_by=join_by,
    )


def _normalize_input_decls(values: Sequence[str | InputDecl] | str | InputDecl) -> tuple[InputDecl, ...]:
    return _normalize_decls(
        values,
        declared_type=InputDecl,
        coerce_one=_coerce_input_decl,
        issue_code="invalid_input",
        issue_message="inputs must be str or peven.input(...)",
    )


def _normalize_output_decls(
    values: Sequence[str | OutputDecl] | str | OutputDecl,
) -> tuple[OutputDecl, ...]:
    return _normalize_decls(
        values,
        declared_type=OutputDecl,
        coerce_one=_coerce_output_decl,
        issue_code="invalid_output",
        issue_message="outputs must be str or peven.output(...)",
    )


def _coerce_input_decl(value: str | InputDecl) -> InputDecl:
    return _coerce_decl(
        value,
        declared_type=InputDecl,
        constructor=input,
        issue_code="invalid_input",
        issue_message="inputs must be str or peven.input(...)",
    )


def _coerce_output_decl(value: str | OutputDecl) -> OutputDecl:
    return _coerce_decl(
        value,
        declared_type=OutputDecl,
        constructor=output,
        issue_code="invalid_output",
        issue_message="outputs must be str or peven.output(...)",
    )


def _normalize_decls(
    values: Sequence[object] | object,
    *,
    declared_type: type,
    coerce_one: object,
    issue_code: str,
    issue_message: str,
) -> tuple[object, ...]:
    if isinstance(values, declared_type) or type(values) is str:
        return (coerce_one(values),)
    if not isinstance(values, Sequence):
        raise PevenValidationError(
            [ValidationIssue(issue_code, "<transition>", issue_message)]
        )
    return tuple(coerce_one(value) for value in values)


def _coerce_decl(
    value: object,
    *,
    declared_type: type,
    constructor: object,
    issue_code: str,
    issue_message: str,
) -> object:
    if isinstance(value, declared_type):
        return value
    if type(value) is str:
        return constructor(value)
    raise PevenValidationError(
        [ValidationIssue(issue_code, "<transition>", issue_message)]
    )


def _validate_place_name(place_name: object) -> None:
    if type(place_name) is not str or not place_name:
        raise ValueError("place ids must be non-empty strings")
