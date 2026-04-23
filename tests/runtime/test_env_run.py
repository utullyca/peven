from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import peven
import peven.runtime.bridge as bridge_module
from peven.handoff.lowering import EnvSpecMessage
from peven.handoff.messages import LoadEnvOk, RunEnvOk, RunFinishedMessage, RunResultMessage
from peven.runtime.bootstrap import BootstrappedRuntime
from peven.runtime.state import _reset_shared_runtime_for_tests

from .conftest import make_session


def _bootstrapped_runtime(*, replies: list[object] | None = None) -> BootstrappedRuntime:
    return make_session(frames=replies)


def test_bridge_module_does_not_keep_the_private_runner_alias() -> None:
    assert not hasattr(bridge_module, "_run_until_terminal_result")


@dataclass
class _RecordingSink:
    events: list[object]
    closed_with: BaseException | None = None

    def write(self, event: object) -> None:
        self.events.append(event)

    def close(self, exc: BaseException | None = None) -> None:
        self.closed_with = exc


def test_env_run_delegates_through_the_bridge_and_reuses_the_sync_runtime_loop() -> None:
    _reset_shared_runtime_for_tests()

    @peven.env("env_run_sync_env")
    class EnvRunSyncEnv(peven.Env):
        done = peven.place()

        def initial_marking(self, seed: int | None = None) -> peven.Marking:
            del seed
            return peven.Marking()

    bootstrap_calls = 0
    seen_loop_ids: list[int] = []
    seen_runtime_ids: list[int] = []

    async def bootstrap_runtime(command: tuple[str, ...]):
        nonlocal bootstrap_calls
        bootstrap_calls += 1
        assert command == ("fake-runtime",)
        return _bootstrapped_runtime(
            replies=[
                LoadEnvOk(req_id=1),
                RunEnvOk(req_id=3, env_run_id=1),
                RunFinishedMessage(
                    env_run_id=1,
                    result=RunResultMessage(
                        run_key="rk-1",
                        status="completed",
                        trace=[],
                        final_marking={},
                    ),
                ),
                LoadEnvOk(req_id=5),
                RunEnvOk(req_id=7, env_run_id=2),
                RunFinishedMessage(
                    env_run_id=2,
                    result=RunResultMessage(
                        run_key="rk-2",
                        status="completed",
                        trace=[],
                        final_marking={},
                    ),
                ),
            ]
        )

    async def runtime_runner(**kwargs: object) -> object:
        runtime = kwargs["runtime"]
        seen_loop_ids.append(id(asyncio.get_running_loop()))
        seen_runtime_ids.append(id(runtime))
        return await bridge_module.run_until_terminal_result(**kwargs)

    env = EnvRunSyncEnv()
    try:
        result_1 = env.run(
            command=("fake-runtime",),
            bootstrap_runtime=bootstrap_runtime,
            runtime_runner=runtime_runner,
        )
        result_2 = env.run(
            command=("fake-runtime",),
            bootstrap_runtime=bootstrap_runtime,
            runtime_runner=runtime_runner,
        )
    finally:
        _reset_shared_runtime_for_tests()

    assert bootstrap_calls == 1
    assert seen_loop_ids[0] == seen_loop_ids[1]
    assert seen_runtime_ids[0] == seen_runtime_ids[1]
    assert isinstance(result_1, peven.RunResult)
    assert isinstance(result_2, peven.RunResult)
    assert result_1.run_key == "rk-1"
    assert result_2.run_key == "rk-2"


@pytest.mark.asyncio
async def test_env_run_does_not_conflict_with_an_existing_event_loop() -> None:
    _reset_shared_runtime_for_tests()

    @peven.env("env_run_async_context_env")
    class EnvRunAsyncContextEnv(peven.Env):
        done = peven.place()

        def initial_marking(self, seed: int | None = None) -> peven.Marking:
            del seed
            return peven.Marking()

    async def bootstrap_runtime(command: tuple[str, ...]):
        assert command == ("fake-runtime",)
        return _bootstrapped_runtime(
            replies=[
                LoadEnvOk(req_id=1),
                RunEnvOk(req_id=3, env_run_id=1),
                RunFinishedMessage(
                    env_run_id=1,
                    result=RunResultMessage(
                        run_key="rk-1",
                        status="completed",
                        trace=[],
                        final_marking={},
                    ),
                ),
            ]
        )

    env = EnvRunAsyncContextEnv()
    try:
        result = env.run(
            command=("fake-runtime",),
            bootstrap_runtime=bootstrap_runtime,
            runtime_runner=bridge_module.run_until_terminal_result,
        )
    finally:
        _reset_shared_runtime_for_tests()

    assert isinstance(result, peven.RunResult)
    assert result.status == "completed"


def test_env_run_writes_runtime_events_to_a_sink_and_closes_it() -> None:
    _reset_shared_runtime_for_tests()

    @peven.env("env_run_sink_env")
    class EnvRunSinkEnv(peven.Env):
        done = peven.place()

        def initial_marking(self, seed: int | None = None) -> peven.Marking:
            del seed
            return peven.Marking()

    async def bootstrap_runtime(command: tuple[str, ...]):
        assert command == ("fake-runtime",)
        return _bootstrapped_runtime(
            replies=[
                LoadEnvOk(req_id=1),
                RunEnvOk(req_id=3, env_run_id=1),
                RunFinishedMessage(
                    env_run_id=1,
                    result=RunResultMessage(
                        run_key="rk-1",
                        status="completed",
                        trace=[],
                        final_marking={},
                    ),
                ),
            ]
        )

    sink = _RecordingSink(events=[])
    try:
        result = EnvRunSinkEnv().run(
            command=("fake-runtime",),
            bootstrap_runtime=bootstrap_runtime,
            runtime_runner=bridge_module.run_until_terminal_result,
            sink=sink,
        )
    finally:
        _reset_shared_runtime_for_tests()

    assert isinstance(result, peven.RunResult)
    assert len(sink.events) == 1
    assert isinstance(sink.events[0], peven.RunFinished)
    assert sink.closed_with is None


def test_env_run_sink_receives_executor_traces_and_terminal_events() -> None:
    _reset_shared_runtime_for_tests()

    namespace: dict[str, object] = {"peven": peven}
    exec(
        """
@peven.executor("env_run_trace_finish")
async def env_run_trace_finish(ctx, ready):
    ctx.trace({"kind": "agent_trace", "payload": ready.payload})
    return peven.token({"done": True}, run_key=ctx.bundle.run_key)
""",
        namespace,
    )

    @peven.env("env_run_trace_sink_env")
    class EnvRunTraceSinkEnv(peven.Env):
        ready = peven.place()
        done = peven.place()

        def initial_marking(self, seed: int | None = None) -> peven.Marking:
            del seed
            return peven.marking(run_key="rk-1", ready=[{"kind": "ready"}])

        finish = peven.transition(
            inputs=["ready"],
            outputs=["done"],
            executor="env_run_trace_finish",
        )

    async def bootstrap_runtime(command: tuple[str, ...]):
        assert command == ("fake-runtime",)
        return _bootstrapped_runtime(
            replies=[
                LoadEnvOk(req_id=1),
                RunEnvOk(req_id=3, env_run_id=1),
            ]
        )

    async def runtime_runner(**kwargs: object) -> object:
        runtime = kwargs["runtime"]
        env_run_id = kwargs["env_run_id"]
        callback = kwargs["callback"]
        bundle = peven.BundleRef(transition_id="finish", run_key="rk-1", ordinal=1)
        outputs = await callback(
            "finish",
            bundle,
            [peven.token({"kind": "ready"}, run_key="rk-1")],
            attempt=1,
        )
        result = peven.RunResult(
            run_key="rk-1",
            status="completed",
            final_marking={"done": outputs["done"]},
        )
        bridge_module._buffer_runtime_event(
            runtime=runtime,
            env_run_id=env_run_id,
            event=peven.RunFinished(result),
        )
        return result

    sink = _RecordingSink(events=[])
    try:
        result = EnvRunTraceSinkEnv().run(
            command=("fake-runtime",),
            bootstrap_runtime=bootstrap_runtime,
            runtime_runner=runtime_runner,
            sink=sink,
        )
    finally:
        _reset_shared_runtime_for_tests()

    assert isinstance(result, peven.RunResult)
    assert result.final_marking["done"][0].payload == {"done": True}
    assert sink.events == [
        {"kind": "agent_trace", "payload": {"kind": "ready"}},
        peven.RunFinished(result),
    ]
    assert sink.closed_with is None


def test_env_run_defaults_to_the_public_runtime_runner() -> None:
    _reset_shared_runtime_for_tests()

    @peven.env("env_run_default_runner_env")
    class EnvRunDefaultRunnerEnv(peven.Env):
        done = peven.place()

        def initial_marking(self, seed: int | None = None) -> peven.Marking:
            del seed
            return peven.Marking()

    async def bootstrap_runtime(command: tuple[str, ...]):
        assert command == ("fake-runtime",)
        return _bootstrapped_runtime(
            replies=[
                LoadEnvOk(req_id=1),
                RunEnvOk(req_id=3, env_run_id=1),
                RunFinishedMessage(
                    env_run_id=1,
                    result=RunResultMessage(
                        run_key="rk-1",
                        status="completed",
                        trace=[],
                        final_marking={},
                    ),
                ),
            ]
        )

    try:
        result = EnvRunDefaultRunnerEnv().run(
            command=("fake-runtime",),
            bootstrap_runtime=bootstrap_runtime,
        )
    finally:
        _reset_shared_runtime_for_tests()

    assert isinstance(result, peven.RunResult)
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_run_env_uses_the_cached_compiled_env_from_the_env_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled_sentinel = SimpleNamespace(
        name="compiled",
        authored_env=EnvSpecMessage(
            schema_version=1,
            env_name="compiled",
            places=[],
            transitions=[],
        ),
    )
    compiled_calls: list[type[peven.Env]] = []

    @peven.env("env_run_cached_compiled_env")
    class EnvRunCachedCompiledEnv(peven.Env):
        done = peven.place()

        def initial_marking(self, seed: int | None = None) -> peven.Marking:
            del seed
            return peven.Marking()

    def fake_compiled(cls: type[peven.Env]) -> object:
        compiled_calls.append(cls)
        return compiled_sentinel

    monkeypatch.setattr(
        EnvRunCachedCompiledEnv,
        "compiled",
        classmethod(fake_compiled),
        raising=False,
    )

    async def bootstrap_runtime(command: tuple[str, ...]) -> BootstrappedRuntime:
        assert command == ("fake-runtime",)
        return _bootstrapped_runtime(
            replies=[LoadEnvOk(req_id=1), RunEnvOk(req_id=3, env_run_id=1)]
        )

    async def runtime_runner(**kwargs: object) -> object:
        assert kwargs["compiled_env"] is compiled_sentinel
        return {"ok": True}

    result = await bridge_module.run_env(
        EnvRunCachedCompiledEnv(),
        command=("fake-runtime",),
        bootstrap_runtime=bootstrap_runtime,
        runtime_runner=runtime_runner,
    )

    assert result == {"ok": True}
    assert compiled_calls == [EnvRunCachedCompiledEnv]
