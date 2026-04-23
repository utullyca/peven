from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import FinalResultEvent

import peven
from peven.authoring.executor import ExecutorContext
from peven.integrations.pydantic_ai import event_stream_handler


async def _event_stream() -> AsyncIterator[FinalResultEvent]:
    yield FinalResultEvent(tool_name=None, tool_call_id=None)


@pytest.mark.asyncio
async def test_pydantic_ai_event_stream_handler_writes_trace_records() -> None:
    records: list[object] = []
    ctx = ExecutorContext(
        env=SimpleNamespace(),
        bundle=peven.BundleRef(
            transition_id="answer",
            run_key="rk-1",
            selected_key="case-17",
            ordinal=2,
        ),
        executor_name="answer",
        attempt=3,
        sink=SimpleNamespace(write=records.append),
    )

    handler = event_stream_handler(ctx, model="qwen3.5:9b")

    assert handler is not None
    await handler(SimpleNamespace(), _event_stream())
    assert records == [
        {
            "kind": "agent_trace",
            "transition_id": "answer",
            "run_key": "rk-1",
            "attempt": 3,
            "event": FinalResultEvent(tool_name=None, tool_call_id=None),
            "model": "qwen3.5:9b",
        }
    ]


def test_pydantic_ai_event_stream_handler_returns_none_without_a_sink() -> None:
    ctx = ExecutorContext(
        env=SimpleNamespace(),
        bundle=peven.BundleRef(transition_id="answer", run_key="rk-1", ordinal=1),
        executor_name="answer",
        attempt=1,
    )

    assert event_stream_handler(ctx, model="qwen3.5:9b") is None
