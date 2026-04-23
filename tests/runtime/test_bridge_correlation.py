from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

import peven
import peven.runtime.bridge as bridge_module
from peven.handoff.lowering import compile_env
from peven.handoff.messages import (
    CallbackBundle,
    CallbackRequest,
    RunFinishedMessage,
    RunResultMessage,
    TransitionStartedMessage,
)
from peven.runtime.state import SharedRuntime, open_run

from .conftest import make_session, make_transition_callback


def _register_executor(source: str) -> dict[str, object]:
    namespace = {
        "__name__": "tests.runtime.generated_bridge_correlation_executors",
        "peven": peven,
    }
    exec(source, namespace)
    return namespace


@dataclass
class _RecordingSink:
    events: list[object]

    def write(self, event: object) -> None:
        self.events.append(event)

    def close(self, exc: BaseException | None = None) -> None:
        del exc


@pytest.mark.asyncio
async def test_run_until_terminal_result_buffers_other_run_events_without_cross_delivery() -> None:
    _register_executor(
        """
@peven.executor("bridge_correlation_finish")
async def bridge_correlation_finish(ctx):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("bridge_correlation_interleaved_env")
    class BridgeCorrelationInterleavedEnv(peven.Env):
        done = peven.place()

        finish = peven.transition(
            inputs=[],
            outputs=["done"],
            executor="bridge_correlation_finish",
        )

    runtime = SharedRuntime(
        session=make_session(
            frames=[
                TransitionStartedMessage(
                    env_run_id=2,
                    bundle=CallbackBundle(
                        transition_id="finish",
                        run_key="rk-2",
                        ordinal=1,
                    ),
                    firing_id=21,
                    attempt=1,
                    inputs=[],
                ),
                RunFinishedMessage(
                    env_run_id=2,
                    result=RunResultMessage(
                        run_key="rk-2",
                        status="completed",
                        final_marking={},
                    ),
                ),
                CallbackRequest(
                    req_id=2,
                    env_run_id=1,
                    transition_id="finish",
                    bundle=CallbackBundle(
                        transition_id="finish",
                        run_key="rk-1",
                        ordinal=1,
                    ),
                    tokens=[],
                    attempt=1,
                ),
                RunFinishedMessage(
                    env_run_id=1,
                    result=RunResultMessage(
                        run_key="rk-1",
                        status="completed",
                        final_marking={
                            "done": [peven.token({"ok": True}, run_key="rk-1")]
                        },
                    ),
                ),
            ]
        ),
        loop=asyncio.get_running_loop(),
        command=("fake-runtime",),
    )
    sink_1 = _RecordingSink(events=[])
    sink_2 = _RecordingSink(events=[])
    open_run(runtime, 1, sink=sink_1)
    open_run(runtime, 2, sink=sink_2)
    env = BridgeCorrelationInterleavedEnv()
    compiled = compile_env(BridgeCorrelationInterleavedEnv.spec())

    result = await bridge_module.run_until_terminal_result(
        runtime=runtime,
        compiled_env=compiled,
        env=env,
        env_run_id=1,
        initial_marking={},
        callback=make_transition_callback(compiled, env),
    )

    assert result.run_key == "rk-1"
    assert [type(event).__name__ for event in sink_1.events] == [
        "RunFinished",
    ]
    assert [type(event).__name__ for event in sink_2.events] == [
        "TransitionStarted",
        "RunFinished",
    ]
    assert sink_2.events[0].bundle.run_key == "rk-2"
    assert sink_2.events[1].result.run_key == "rk-2"


@pytest.mark.asyncio
async def test_run_until_terminal_result_rejects_callback_requests_for_the_wrong_run() -> None:
    _register_executor(
        """
@peven.executor("bridge_correlation_wrong_run")
async def bridge_correlation_wrong_run(ctx):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("bridge_correlation_wrong_run_env")
    class BridgeCorrelationWrongRunEnv(peven.Env):
        done = peven.place()

        finish = peven.transition(
            inputs=[],
            outputs=["done"],
            executor="bridge_correlation_wrong_run",
        )

    runtime = SharedRuntime(
        session=make_session(
            frames=[
                CallbackRequest(
                    req_id=2,
                    env_run_id=2,
                    transition_id="finish",
                    bundle=CallbackBundle(
                        transition_id="finish",
                        run_key="rk-2",
                        ordinal=1,
                    ),
                    tokens=[],
                    attempt=1,
                )
            ]
        ),
        loop=asyncio.get_running_loop(),
        command=("fake-runtime",),
    )
    open_run(runtime, 1)
    compiled = compile_env(BridgeCorrelationWrongRunEnv.spec())

    with pytest.raises(
        bridge_module.AdapterProtocolError,
        match="env_run_id did not match the active run",
    ):
        await bridge_module.run_until_terminal_result(
            runtime=runtime,
            compiled_env=compiled,
            env=BridgeCorrelationWrongRunEnv(),
            env_run_id=1,
            initial_marking={},
            callback=make_transition_callback(compiled, BridgeCorrelationWrongRunEnv()),
        )
