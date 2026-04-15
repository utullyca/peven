"""Petri net result types."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from peven.petri.schema import Token
from rubric import CriterionReport


# -- Colored tokens (executor outputs) ----------------------------------------


class GenerateOutput(Token):
    text: str


class JudgeOutput(Token):
    score: float
    raw_score: float | None = Field(default=None)
    llm_raw_score: float | None = Field(default=None)
    report: list[CriterionReport] | None = Field(default=None)


class TokenSnapshot(Token):
    """Stored representation for custom token outputs."""

    type_name: str
    payload: dict[str, Any] = Field(default_factory=dict)


# -- Results -------------------------------------------------------------------

RunStatus = Literal["completed", "failed", "incomplete"]
TerminalReason = Literal[
    "completed",
    "executor_failed",
    "guard_error",
    "missing_score",
    "no_enabled_transition",
    "fuse_exhausted",
]


class TransitionResult(BaseModel):
    transition_id: str
    status: Literal["completed", "failed"]
    output: Token | None = Field(default=None)
    error: str | None = Field(default=None)
    run_id: str | None = Field(default=None)


class RunResult(BaseModel):
    run_id: str | None = Field(default=None)
    status: RunStatus = Field(default="completed")
    terminal_reason: TerminalReason | None = Field(default=None)
    score: float | None = Field(default=None)
    error: str | None = Field(default=None)
    trace: list[TransitionResult] = Field(default_factory=list)


# -- Stored run types ----------------------------------------------------------


class StoredRun(BaseModel):
    """Full stored run with hydrated results."""

    id: str
    timestamp: str
    file: str | None = Field(default=None)
    status: RunStatus
    score: float | None = Field(default=None)
    error: str | None = Field(default=None)
    result_count: int
    results: list[RunResult]


class RunSummary(BaseModel):
    """Lightweight run summary for listing."""

    id: str
    timestamp: str
    file: str | None = Field(default=None)
    status: RunStatus
    score: float | None = Field(default=None)
    result_count: int
