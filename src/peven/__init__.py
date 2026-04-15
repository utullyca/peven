"""Peven — Petri net engine for multi-agent evaluations."""

from peven.petri.dsl import NetBuilder, agent, judge
from peven.petri.engine import execute
from peven.petri.executors import Executor, register
from peven.petri.guards import score_at_least
from peven.petri.render import render, render_net
from peven.petri.schema import Token
from peven.petri.types import GenerateOutput, JudgeOutput, RunResult, TokenSnapshot

__version__ = "0.1.1"

__all__ = [
    "Executor",
    "GenerateOutput",
    "JudgeOutput",
    "NetBuilder",
    "RunResult",
    "Token",
    "TokenSnapshot",
    "agent",
    "execute",
    "judge",
    "register",
    "render",
    "render_net",
    "score_at_least",
]
