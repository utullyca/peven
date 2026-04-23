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
    GuardErroredMessage,
    RunFinishedMessage,
    RunResultMessage,
    SelectionErroredMessage,
    TransitionCompletedMessage,
    TransitionFailedMessage,
    TransitionResultMessage,
    TransitionStartedMessage,
)
from peven.runtime.state import SharedRuntime, open_run

from .conftest import make_session, make_transition_callback


def _register_executor(source: str) -> dict[str, object]:
    namespace = {
        "__name__": "tests.runtime.generated_bridge_event_executors",
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
async def test_run_until_terminal_result_buffers_and_dispatches_mixed_message_stream() -> None:
    _register_executor(
        """
@peven.executor("bridge_event_finish")
async def bridge_event_finish(ctx):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("bridge_event_env")
    class BridgeEventEnv(peven.Env):
        done = peven.place()

        finish = peven.transition(inputs=[], outputs=["done"], executor="bridge_event_finish")

    bundle = CallbackBundle(transition_id="finish", run_key="rk-1", ordinal=1)
    runtime = SharedRuntime(
        session=make_session(
            frames=[
                TransitionStartedMessage(
                    env_run_id=17,
                    bundle=bundle,
                    firing_id=1,
                    attempt=1,
                    inputs=[],
                    inputs_by_place={"ready": []},
                ),
                CallbackRequest(
                    req_id=2,
                    env_run_id=17,
                    transition_id="finish",
                    bundle=bundle,
                    tokens=[],
                    attempt=1,
                ),
                TransitionFailedMessage(
                    env_run_id=17,
                    bundle=bundle,
                    firing_id=1,
                    attempt=1,
                    error="transient",
                    retrying=True,
                ),
                GuardErroredMessage(env_run_id=17, bundle=bundle, error="guard"),
                SelectionErroredMessage(
                    env_run_id=17,
                    transition_id="finish",
                    run_key="rk-1",
                    error="select",
                ),
                TransitionCompletedMessage(
                    env_run_id=17,
                    bundle=bundle,
                    firing_id=1,
                    attempt=1,
                    outputs={"done": [peven.token({"ok": True}, run_key="rk-1")]},
                ),
                RunFinishedMessage(
                    env_run_id=17,
                    result=RunResultMessage(
                        run_key="rk-1",
                        status="completed",
                        trace=[
                            TransitionResultMessage(
                                bundle=bundle,
                                firing_id=1,
                                status="completed",
                                outputs={
                                    "done": [peven.token({"ok": True}, run_key="rk-1")]
                                },
                            )
                        ],
                        final_marking={"done": [peven.token({"ok": True}, run_key="rk-1")]},
                    ),
                ),
            ]
        ),
        loop=asyncio.get_running_loop(),
        command=("fake-runtime",),
    )
    sink = _RecordingSink(events=[])
    open_run(runtime, 17, sink=sink)
    env = BridgeEventEnv()
    compiled = compile_env(BridgeEventEnv.spec())

    result = await bridge_module.run_until_terminal_result(
        runtime=runtime,
        compiled_env=compiled,
        env=env,
        env_run_id=17,
        initial_marking={},
        callback=make_transition_callback(compiled, env),
    )

    assert result.run_key == "rk-1"
    assert [type(event).__name__ for event in sink.events] == [
        "TransitionStarted",
        "TransitionFailed",
        "GuardErrored",
        "SelectionErrored",
        "TransitionCompleted",
        "RunFinished",
    ]
    assert sink.events[0].inputs_by_place == {"ready": []}
    assert result.final_marking == {
        "done": [peven.Token(run_key="rk-1", color="default", payload={"ok": True})]
    }
    assert len(runtime.session.writer.writes) == 1
