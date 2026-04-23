"""Engine-aligned public event and trace models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias, get_args

from .token import Token


TransitionStatus: TypeAlias = Literal["completed", "failed", "fuse_blocked"]
RunStatus: TypeAlias = Literal["completed", "failed", "incomplete"]
TerminalReason: TypeAlias = Literal[
    "executor_failed",
    "guard_error",
    "selection_error",
    "fuse_exhausted",
    "no_enabled_transition",
] | None

TRANSITION_STATUS_VALUES: frozenset[str] = frozenset(get_args(TransitionStatus))
RUN_STATUS_VALUES: frozenset[str] = frozenset(get_args(RunStatus))
_TERMINAL_REASON_LITERAL = next(
    arg for arg in get_args(TerminalReason) if arg is not type(None)
)
TERMINAL_REASON_VALUES: frozenset[str | None] = frozenset(
    get_args(_TERMINAL_REASON_LITERAL)
) | {None}


@dataclass(frozen=True, slots=True)
class BundleRef:
    transition_id: str
    run_key: str
    selected_key: object | None = None
    ordinal: int = 1

    @property
    def key(self) -> object | None:
        return self.selected_key

    @property
    def idx(self) -> int:
        return self.ordinal


@dataclass(frozen=True, slots=True)
class TransitionStarted:
    bundle: BundleRef
    firing_id: int
    attempt: int
    inputs: list[Token]
    inputs_by_place: dict[str, list[Token]] = field(default_factory=dict)
    kind: str = field(init=False, default="transition_started")


@dataclass(frozen=True, slots=True)
class TransitionCompleted:
    bundle: BundleRef
    firing_id: int
    attempt: int
    outputs: dict[str, list[Token]]
    kind: str = field(init=False, default="transition_completed")


@dataclass(frozen=True, slots=True)
class TransitionFailed:
    bundle: BundleRef
    firing_id: int
    attempt: int
    error: str
    retrying: bool
    kind: str = field(init=False, default="transition_failed")


@dataclass(frozen=True, slots=True)
class GuardErrored:
    bundle: BundleRef
    error: str
    kind: str = field(init=False, default="guard_errored")


@dataclass(frozen=True, slots=True)
class SelectionErrored:
    transition_id: str
    run_key: str
    error: str
    kind: str = field(init=False, default="selection_errored")


@dataclass(frozen=True, slots=True)
class TransitionResult:
    bundle: BundleRef
    firing_id: int
    status: TransitionStatus
    outputs: dict[str, list[Token]]
    error: str | None = None
    attempts: int = 1


@dataclass(frozen=True, slots=True)
class RunResult:
    run_key: str
    status: RunStatus
    error: str | None = None
    terminal_reason: TerminalReason = None
    terminal_bundle: BundleRef | None = None
    terminal_transition: str | None = None
    trace: list[TransitionResult] = field(default_factory=list)
    final_marking: dict[str, list[Token]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunFinished:
    result: RunResult
    kind: str = field(init=False, default="run_finished")


RuntimeEvent: TypeAlias = (
    TransitionStarted
    | TransitionCompleted
    | TransitionFailed
    | GuardErrored
    | SelectionErrored
    | RunFinished
)

__all__ = [
    "RUN_STATUS_VALUES",
    "TERMINAL_REASON_VALUES",
    "TRANSITION_STATUS_VALUES",
    "BundleRef",
    "GuardErrored",
    "RunFinished",
    "RunResult",
    "SelectionErrored",
    "TransitionCompleted",
    "TransitionFailed",
    "TransitionResult",
    "TransitionStarted",
    "completed_firings",
    "failed_firings",
    "firing_result",
    "firing_status",
    "fuse_blocked_firings",
]


def completed_firings(result: RunResult) -> list[TransitionResult]:
    return [firing for firing in result.trace if firing.status == "completed"]


def failed_firings(result: RunResult) -> list[TransitionResult]:
    return [firing for firing in result.trace if firing.status == "failed"]


def fuse_blocked_firings(result: RunResult) -> list[TransitionResult]:
    return [firing for firing in result.trace if firing.status == "fuse_blocked"]


def firing_result(result: RunResult, firing_id: int) -> TransitionResult | None:
    for firing in result.trace:
        if firing.firing_id == firing_id:
            return firing
    return None


def firing_status(result: RunResult, firing_id: int) -> TransitionStatus | None:
    firing = firing_result(result, firing_id)
    if firing is None:
        return None
    return firing.status
