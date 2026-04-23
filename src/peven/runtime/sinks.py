"""Runtime-side observer protocol for run-local trace records."""

from typing import Protocol, runtime_checkable


__all__ = ["Sink"]


@runtime_checkable
class Sink(Protocol):
    def write(self, record: object) -> None: ...
    def close(self, exc: BaseException | None) -> None: ...
