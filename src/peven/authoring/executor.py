"""Top-level reusable executor registration for peven authoring."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from types import FunctionType
from typing import ParamSpec, TypeVar, cast

from ..runtime.sinks import Sink as _Sink
from ..shared.errors import PevenValidationError, ValidationIssue
from ..shared.events import BundleRef
from ..shared.token import Token, token as _token
from .ir import ExecutorFn, ExecutorSpec


__all__ = [
    "ExecutorContext",
    "executor",
    "get_executor",
    "get_executor_registry_version",
    "unregister_executor",
]

_EXECUTORS: dict[str, ExecutorSpec] = {}
_EXECUTOR_REGISTRY_VERSION = 0
_P = ParamSpec("_P")
_R = TypeVar("_R")


@dataclass(frozen=True, slots=True)
class ExecutorContext:
    """Host-side execution context delivered to one executor callback."""

    env: object
    bundle: BundleRef
    executor_name: str
    attempt: int
    inputs_by_place: dict[str, list[Token]] | None = None
    sink: _Sink | None = None

    def token(self, payload: object = None, *, color: str = "default") -> Token:
        """Emit one token under the active firing's run_key."""
        return _token(payload, run_key=self.bundle.run_key, color=color)

    def trace(self, record: object) -> None:
        """Write one executor-local trace record to the active run sink, if any."""
        if self.sink is not None:
            self.sink.write(record)


def executor(
    name: str,
) -> Callable[[Callable[_P, Awaitable[_R]]], Callable[_P, Awaitable[_R]]]:
    """Register one top-level reusable async executor by explicit name."""
    if type(name) is not str or not name:
        raise ValueError("executor name must be a non-empty string")

    def decorator(function: Callable[_P, Awaitable[_R]]) -> Callable[_P, Awaitable[_R]]:
        if not inspect.iscoroutinefunction(function) or not _is_top_level_function(function):
            raise PevenValidationError(
                [
                    ValidationIssue(
                        "invalid_executor",
                        name,
                        "executors must be registered from top-level async functions",
                    )
                ]
            )
        executor_function = cast(ExecutorFn, function)
        existing = _EXECUTORS.get(name)
        if existing is not None and not _is_executor_reload(existing.fn, executor_function):
            raise PevenValidationError(
                [ValidationIssue("duplicate_executor", name, f"duplicate executor {name}")]
            )
        _bump_executor_registry_version()
        _EXECUTORS[name] = ExecutorSpec(name=name, fn=executor_function)
        return function

    return decorator


def get_executor(name: str) -> ExecutorSpec | None:
    """Return one registered executor by explicit name."""
    return _EXECUTORS.get(name)


def get_executor_registry_version() -> int:
    """Return one monotonically increasing executor registry version."""
    return _EXECUTOR_REGISTRY_VERSION


def unregister_executor(name: str) -> None:
    """Remove one registered executor by explicit name."""
    removed = _EXECUTORS.pop(name, None)
    if removed is not None:
        _bump_executor_registry_version()


def validate_executor_signature(
    spec: ExecutorSpec,
    *,
    input_count: int,
    object_id: str,
) -> None:
    """Validate that one executor matches the normalized authored input shape."""
    parameters = list(inspect.signature(spec.fn).parameters.values())
    if not parameters:
        raise PevenValidationError(
            [
                ValidationIssue(
                    "invalid_executor_signature",
                    object_id,
                    "executor signatures must start with a positional `ctx` parameter",
                )
            ]
        )
    if parameters[0].name != "ctx" or parameters[0].kind not in {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }:
        raise PevenValidationError(
            [
                ValidationIssue(
                    "invalid_executor_signature",
                    object_id,
                    "executor signatures must start with a positional `ctx` parameter",
                )
            ]
        )
    for parameter in parameters[1:]:
        if parameter.kind not in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            raise PevenValidationError(
                [
                    ValidationIssue(
                        "invalid_executor_signature",
                        object_id,
                        "executor callback parameters must be positional",
                    )
                ]
            )
    if len(parameters) - 1 != input_count:
        raise PevenValidationError(
            [
                ValidationIssue(
                    "invalid_executor_signature",
                    object_id,
                    "executor signature does not match normalized input arcs",
                )
            ]
        )


def _is_top_level_function(function: object) -> bool:
    return isinstance(function, FunctionType) and function.__qualname__ == function.__name__


def _is_executor_reload(
    existing: ExecutorFn,
    replacement: ExecutorFn,
) -> bool:
    return (
        existing.__module__ == replacement.__module__
        and existing.__qualname__ == replacement.__qualname__
    )


def _bump_executor_registry_version() -> None:
    global _EXECUTOR_REGISTRY_VERSION
    _EXECUTOR_REGISTRY_VERSION += 1
