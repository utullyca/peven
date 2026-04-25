"""Immutable authoring IR for Python-authored peven nets."""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Protocol


class ExecutorFn(Protocol):
    """Async top-level function accepted by executor registration."""

    __module__: str
    __name__: str
    __qualname__: str

    def __call__(self, *args: Any, **kwargs: Any) -> Awaitable[object]: ...


@dataclass(frozen=True, slots=True)
class PlaceSpec:
    """One authored place definition."""

    id: str
    capacity: int | None = None
    schema: object | None = None
    terminal: bool = False


@dataclass(frozen=True, slots=True)
class InputArcSpec:
    """One authored input arc definition."""

    place: str
    weight: int = 1


@dataclass(frozen=True, slots=True)
class OutputArcSpec:
    """One authored output arc definition."""

    place: str


@dataclass(frozen=True, slots=True)
class TransitionSpec:
    """One authored transition definition."""

    id: str
    executor: str
    inputs: tuple[InputArcSpec, ...]
    outputs: tuple[OutputArcSpec, ...]
    guard_spec: dict[str, object] | None = None
    retries: int = 0
    join_by_spec: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class ExecutorSpec:
    """One named reusable executor binding."""

    name: str
    fn: ExecutorFn


@dataclass(frozen=True, slots=True)
class EnvSpec:
    """The authored topology for one env class."""

    env_name: str
    places: tuple[PlaceSpec, ...]
    transitions: tuple[TransitionSpec, ...]
