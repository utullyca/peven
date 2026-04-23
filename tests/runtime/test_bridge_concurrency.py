from __future__ import annotations

import asyncio
import time

import pytest

import peven
import peven.runtime.bridge as bridge_module
from peven.handoff.lowering import compile_env
from peven.handoff.messages import (
    CallbackBundle,
    CallbackRequest,
    RunFinishedMessage,
    RunResultMessage,
)
from peven.runtime.state import SharedRuntime, open_run

from .conftest import make_session, make_transition_callback


CALLBACK_LATENCY = 0.5
FANOUT = 4


async def bridge_concurrency_sleep(ctx):
    await asyncio.sleep(CALLBACK_LATENCY)
    return ctx.token({"ok": True})


@pytest.mark.asyncio
async def test_run_until_terminal_result_processes_callback_requests_concurrently() -> None:
    peven.executor("bridge_concurrency_sleep")(bridge_concurrency_sleep)

    try:

        @peven.env("bridge_concurrency_env")
        class BridgeConcurrencyEnv(peven.Env):
            done = peven.place()

            finish = peven.transition(
                inputs=[], outputs=["done"], executor="bridge_concurrency_sleep"
            )

        frames: list[object] = [
            CallbackRequest(
                req_id=2 * (i + 1),
                env_run_id=42,
                transition_id="finish",
                bundle=CallbackBundle(
                    transition_id="finish", run_key=f"rk-{i}", ordinal=1
                ),
                tokens=[],
                attempt=1,
            )
            for i in range(FANOUT)
        ]
        frames.append(
            RunFinishedMessage(
                env_run_id=42,
                result=RunResultMessage(
                    run_key="rk-final",
                    status="completed",
                    final_marking={},
                ),
            )
        )

        runtime = SharedRuntime(
            session=make_session(frames=frames),
            loop=asyncio.get_running_loop(),
            command=("fake-runtime",),
        )
        open_run(runtime, 42)
        env = BridgeConcurrencyEnv()
        compiled = compile_env(BridgeConcurrencyEnv.spec())

        started = time.perf_counter()
        result = await bridge_module.run_until_terminal_result(
            runtime=runtime,
            compiled_env=compiled,
            env=env,
            env_run_id=42,
            initial_marking={},
            callback=make_transition_callback(compiled, env),
        )
        elapsed = time.perf_counter() - started

        assert result.status == "completed"
        assert elapsed < CALLBACK_LATENCY * FANOUT / 2, (
            f"expected concurrent dispatch (<{CALLBACK_LATENCY * FANOUT / 2:.2f}s), got {elapsed:.2f}s"
        )
        assert len(runtime.session.writer.writes) == FANOUT
    finally:
        peven.unregister_executor("bridge_concurrency_sleep")
