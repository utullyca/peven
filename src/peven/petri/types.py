"""Petri net result types."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field
from rubric import CriterionReport

from peven.petri.schema import Token

# -- Colored tokens (executor outputs) ----------------------------------------


class GenerateOutput(Token):
    text: str


class JudgeOutput(Token):
    score: float
    raw_score: Optional[float] = Field(default=None)
    llm_raw_score: Optional[float] = Field(default=None)
    report: Optional[list[CriterionReport]] = Field(default=None)


# -- Results -------------------------------------------------------------------


class TransitionResult(BaseModel):
    transition_id: str
    status: Literal["completed", "failed"]
    output: Optional[GenerateOutput | JudgeOutput] = Field(default=None)
    error: Optional[str] = Field(default=None)
    run_id: Optional[str] = Field(default=None)


class RunResult(BaseModel):
    run_id: Optional[str] = Field(default=None)
    status: Literal["completed", "failed"] = Field(default="completed")
    score: Optional[float] = Field(default=None)
    error: Optional[str] = Field(default=None)
    trace: list[TransitionResult] = Field(default_factory=list)


# -- Stored run types ----------------------------------------------------------


class StoredRun(BaseModel):
    """Full stored run with hydrated results."""

    id: str
    timestamp: str
    file: Optional[str] = Field(default=None)
    status: Literal["completed", "failed"]
    score: Optional[float] = Field(default=None)
    error: Optional[str] = Field(default=None)
    result_count: int
    results: list[RunResult]


class RunSummary(BaseModel):
    """Lightweight run summary for listing."""

    id: str
    timestamp: str
    file: Optional[str] = Field(default=None)
    status: Literal["completed", "failed"]
    score: Optional[float] = Field(default=None)
    result_count: int
