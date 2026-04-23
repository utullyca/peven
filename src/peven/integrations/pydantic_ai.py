"""Helpers for writing pydantic_ai traces into an active peven sink."""

from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Callable
from typing import Any

from pydantic_ai.messages import AgentStreamEvent

from ..authoring.executor import ExecutorContext


__all__ = ["event_stream_handler"]


def event_stream_handler(
    ctx: ExecutorContext,
    *,
    model: str | None = None,
) -> Callable[[object, AsyncIterable[AgentStreamEvent]], Awaitable[None]] | None:
    """Build one pydantic_ai event-stream handler that writes trace records to ctx.sink."""
    if ctx.sink is None:
        return None

    async def _handle(_: object, events: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in events:
            record: dict[str, Any] = {
                "kind": "agent_trace",
                "transition_id": ctx.bundle.transition_id,
                "run_key": ctx.bundle.run_key,
                "attempt": ctx.attempt,
                "event": event,
            }
            if model is not None:
                record["model"] = model
            ctx.trace(record)

    return _handle
