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
    TransitionCompletedMessage,
    TransitionResultMessage,
    TransitionStartedMessage,
)
from peven.runtime.state import SharedRuntime, open_run

from .conftest import make_session, make_transition_callback


def _register_executor(source: str) -> dict[str, object]:
    namespace = {
        "__name__": "tests.runtime.generated_bridge_completion_executors",
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
async def test_run_until_terminal_result_returns_a_decoded_run_result() -> None:
    _register_executor(
        """
@peven.executor("bridge_completion_finish")
async def bridge_completion_finish(ctx):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("bridge_completion_env")
    class BridgeCompletionEnv(peven.Env):
        done = peven.place()

        finish = peven.transition(inputs=[], outputs=["done"], executor="bridge_completion_finish")

    runtime = SharedRuntime(
        session=make_session(
            frames=[
                TransitionStartedMessage(
                    env_run_id=9,
                    bundle=CallbackBundle(transition_id="finish", run_key="rk-1", ordinal=1),
                    firing_id=1,
                    attempt=1,
                    inputs=[],
                ),
                CallbackRequest(
                    req_id=2,
                    env_run_id=9,
                    transition_id="finish",
                    bundle=CallbackBundle(transition_id="finish", run_key="rk-1", ordinal=1),
                    tokens=[],
                    attempt=1,
                ),
                TransitionCompletedMessage(
                    env_run_id=9,
                    bundle=CallbackBundle(transition_id="finish", run_key="rk-1", ordinal=1),
                    firing_id=1,
                    attempt=1,
                    outputs={"done": [peven.token({"ok": True}, run_key="rk-1")]},
                ),
                RunFinishedMessage(
                    env_run_id=9,
                    result=RunResultMessage(
                        run_key="rk-1",
                        status="completed",
                        trace=[
                            TransitionResultMessage(
                                bundle=CallbackBundle(
                                    transition_id="finish",
                                    run_key="rk-1",
                                    ordinal=1,
                                ),
                                firing_id=1,
                                status="completed",
                                outputs={
                                    "done": [peven.token({"ok": True}, run_key="rk-1")]
                                },
                            )
                        ],
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
    sink = _RecordingSink(events=[])
    open_run(runtime, 9, sink=sink)
    env = BridgeCompletionEnv()
    compiled = compile_env(BridgeCompletionEnv.spec())

    result = await bridge_module.run_until_terminal_result(
        runtime=runtime,
        compiled_env=compiled,
        env=env,
        env_run_id=9,
        initial_marking={},
        callback=make_transition_callback(compiled, env),
    )

    assert isinstance(result, peven.RunResult)
    assert result.status == "completed"
    assert result.final_marking == {
        "done": [peven.Token(run_key="rk-1", color="default", payload={"ok": True})]
    }
    assert peven.completed_firings(result) == result.trace
    assert sink.events[-1] == peven.RunFinished(result=result)


@pytest.mark.asyncio
async def test_terminal_place_turns_no_enabled_transition_into_completion() -> None:
    @peven.env("bridge_terminal_place_env")
    class BridgeTerminalPlaceEnv(peven.Env):
        done = peven.place(terminal=True)

    runtime = SharedRuntime(
        session=make_session(
            frames=[
                RunFinishedMessage(
                    env_run_id=11,
                    result=RunResultMessage(
                        run_key="rk-1",
                        status="incomplete",
                        terminal_reason="no_enabled_transition",
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
    sink = _RecordingSink(events=[])
    open_run(runtime, 11, sink=sink)
    env = BridgeTerminalPlaceEnv()
    compiled = compile_env(BridgeTerminalPlaceEnv.spec())

    result = await bridge_module.run_until_terminal_result(
        runtime=runtime,
        compiled_env=compiled,
        env=env,
        env_run_id=11,
        initial_marking={},
        callback=make_transition_callback(compiled, env),
    )

    assert result.status == "completed"
    assert result.terminal_reason is None
    assert result.final_marking["done"][0].payload == {"ok": True}
    assert sink.events[-1] == peven.RunFinished(result=result)


@pytest.mark.asyncio
async def test_non_terminal_deadlock_stays_incomplete() -> None:
    @peven.env("bridge_non_terminal_deadlock_env")
    class BridgeNonTerminalDeadlockEnv(peven.Env):
        waiting = peven.place()

    runtime = SharedRuntime(
        session=make_session(
            frames=[
                RunFinishedMessage(
                    env_run_id=12,
                    result=RunResultMessage(
                        run_key="rk-1",
                        status="incomplete",
                        terminal_reason="no_enabled_transition",
                        final_marking={
                            "waiting": [peven.token({"blocked": True}, run_key="rk-1")]
                        },
                    ),
                ),
            ]
        ),
        loop=asyncio.get_running_loop(),
        command=("fake-runtime",),
    )
    sink = _RecordingSink(events=[])
    open_run(runtime, 12, sink=sink)
    env = BridgeNonTerminalDeadlockEnv()
    compiled = compile_env(BridgeNonTerminalDeadlockEnv.spec())

    result = await bridge_module.run_until_terminal_result(
        runtime=runtime,
        compiled_env=compiled,
        env=env,
        env_run_id=12,
        initial_marking={},
        callback=make_transition_callback(compiled, env),
    )

    assert result.status == "incomplete"
    assert result.terminal_reason == "no_enabled_transition"
    assert sink.events[-1] == peven.RunFinished(result=result)


@pytest.mark.asyncio
async def test_run_until_terminal_result_decodes_failed_terminal_results_without_raising() -> None:
    @peven.env("bridge_completion_failed_env")
    class BridgeCompletionFailedEnv(peven.Env):
        done = peven.place()

    runtime = SharedRuntime(
        session=make_session(
            frames=[
                RunFinishedMessage(
                    env_run_id=10,
                    result=RunResultMessage(
                        run_key="rk-1",
                        status="failed",
                        error="executor boom",
                        terminal_reason="executor_failed",
                        final_marking={},
                    ),
                ),
            ]
        ),
        loop=asyncio.get_running_loop(),
        command=("fake-runtime",),
    )
    sink = _RecordingSink(events=[])
    open_run(runtime, 10, sink=sink)
    env = BridgeCompletionFailedEnv()
    compiled = compile_env(BridgeCompletionFailedEnv.spec())

    result = await bridge_module.run_until_terminal_result(
        runtime=runtime,
        compiled_env=compiled,
        env=env,
        env_run_id=10,
        initial_marking={},
        callback=make_transition_callback(compiled, env),
    )

    assert isinstance(result, peven.RunResult)
    assert result.status == "failed"
    assert result.error == "executor boom"
    assert sink.events[-1] == peven.RunFinished(result=result)


@pytest.mark.asyncio
async def test_run_until_terminal_result_ignores_stale_run_finished_events() -> None:
    _register_executor(
        """
@peven.executor("bridge_completion_stale_finish")
async def bridge_completion_stale_finish(ctx):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("bridge_completion_stale_env")
    class BridgeCompletionStaleEnv(peven.Env):
        done = peven.place()

        finish = peven.transition(
            inputs=[], outputs=["done"], executor="bridge_completion_stale_finish"
        )

    runtime = SharedRuntime(
        session=make_session(
            frames=[
                RunFinishedMessage(
                    env_run_id=8,
                    result=RunResultMessage(
                        run_key="rk-old",
                        status="completed",
                        final_marking={"done": [peven.token({"old": True}, run_key="rk-old")]},
                    ),
                ),
                RunFinishedMessage(
                    env_run_id=9,
                    result=RunResultMessage(
                        run_key="rk-new",
                        status="completed",
                        final_marking={"done": [peven.token({"new": True}, run_key="rk-new")]},
                    ),
                ),
            ]
        ),
        loop=asyncio.get_running_loop(),
        command=("fake-runtime",),
    )
    stale_sink = _RecordingSink(events=[])
    current_sink = _RecordingSink(events=[])
    open_run(runtime, 8, sink=stale_sink)
    open_run(runtime, 9, sink=current_sink)
    from peven.runtime.state import finish_run

    finish_run(runtime, 8)
    env = BridgeCompletionStaleEnv()
    compiled = compile_env(BridgeCompletionStaleEnv.spec())

    result = await bridge_module.run_until_terminal_result(
        runtime=runtime,
        compiled_env=compiled,
        env=env,
        env_run_id=9,
        initial_marking={},
        callback=make_transition_callback(compiled, env),
    )

    assert isinstance(result, peven.RunResult)
    assert result.run_key == "rk-new"
    assert current_sink.events == [peven.RunFinished(result=result)]
    assert stale_sink.events == []
