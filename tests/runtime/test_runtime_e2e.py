from __future__ import annotations

import asyncio

import msgspec
import pytest

import peven
import peven.runtime.bridge as bridge_module
from peven.handoff.framing import DEFAULT_MAX_FRAME_BYTES, FrameDecoder, encode_frame
from peven.handoff.messages import (
    CallbackBundle,
    CallbackReply,
    CallbackRequest,
    LoadEnvError,
    LoadEnvOk,
    RunEnvOk,
    RunFinishedMessage,
    RunResultMessage,
    TransitionResultMessage,
    TransitionStartedMessage,
    decode_callback_reply,
)
from peven.runtime.bridge import run_env

from .conftest import FakeWriter, make_session


class _BrokenPipeWriter(FakeWriter):
    def write(self, data: bytes) -> None:
        del data
        raise BrokenPipeError("broken pipe")


def _bootstrapped_runtime(*, replies: list[object] | None = None):
    return make_session(frames=replies)


def _register_executor(source: str) -> dict[str, object]:
    namespace = {
        "__name__": "tests.runtime.generated_runtime_executors",
        "peven": peven,
    }
    exec(source, namespace)
    return namespace


def _decode_single_frame(data: bytes) -> bytes:
    decoder = FrameDecoder()
    frames = decoder.feed(data)
    assert len(frames) == 1
    return frames[0]


@pytest.mark.asyncio
async def test_run_env_reuses_the_shared_runtime_and_keeps_store_alive_for_the_full_run() -> None:
    from peven.runtime.state import _reset_shared_runtime_for_tests

    _reset_shared_runtime_for_tests()
    _register_executor(
        """
@peven.executor("runtime_e2e_store_writer")
async def runtime_e2e_store_writer(ctx, ready):
    ref = peven.store.put({"seed": ready.payload["seed"]})
    return peven.token({"ref": ref}, run_key=ctx.bundle.run_key)

@peven.executor("runtime_e2e_store_reader")
async def runtime_e2e_store_reader(ctx, stored):
    value = peven.store.get(stored.payload["ref"])
    return peven.token({"kind": "done", "payload": value}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("runtime_run_env")
    class RuntimeRunEnv(peven.Env):
        ready = peven.place()
        stored = peven.place()
        done = peven.place()

        def initial_marking(self, seed: int | None = None) -> peven.Marking:
            return peven.Marking(
                {"ready": [peven.token({"seed": seed}, run_key="rk-1")]}
            )

        write = peven.transition(
            inputs=["ready"],
            outputs=["stored"],
            executor="runtime_e2e_store_writer",
        )
        read = peven.transition(
            inputs=["stored"],
            outputs=["done"],
            executor="runtime_e2e_store_reader",
        )

    bootstrap_calls = 0
    seen_runtime_ids: list[int] = []
    seen_env_run_ids: list[int] = []
    seen_loop_ids: list[int] = []

    async def bootstrap_runtime(command: tuple[str, ...]):
        nonlocal bootstrap_calls
        bootstrap_calls += 1
        assert command == ("fake-runtime",)
        return _bootstrapped_runtime(
            replies=[
                LoadEnvOk(req_id=1),
                RunEnvOk(req_id=3, env_run_id=1),
                CallbackRequest(
                    req_id=2,
                    env_run_id=1,
                    transition_id="write",
                    bundle=CallbackBundle(transition_id="write", run_key="rk-1", ordinal=1),
                    tokens=[peven.token({"seed": 1}, run_key="rk-1")],
                    attempt=1,
                ),
            ]
        )

    async def runtime_runner(
        *,
        runtime,
        compiled_env,
        env,
        env_run_id,
        initial_marking,
        callback,
    ):
        seen_loop_ids.append(id(asyncio.get_running_loop()))
        seen_runtime_ids.append(id(runtime))
        seen_env_run_ids.append(env_run_id)
        writer = runtime.session.writer
        writes_before = len(writer.writes)
        req_base = 4 if env_run_id == 1 else 8

        async def drive_adapter() -> None:
            while len(writer.writes) <= writes_before:
                await asyncio.sleep(0)
            stored_reply = decode_callback_reply(_decode_single_frame(writer.writes[writes_before]))
            assert isinstance(stored_reply, CallbackReply)

            runtime.session.reader.feed_data(
                encode_frame(
                    msgspec.msgpack.encode(
                        CallbackRequest(
                            req_id=req_base,
                            env_run_id=env_run_id,
                            transition_id="read",
                            bundle=CallbackBundle(
                                transition_id="read",
                                run_key="rk-1",
                                ordinal=2,
                            ),
                            tokens=stored_reply.outputs["stored"],
                            attempt=1,
                        )
                    )
                )
            )

            while len(writer.writes) <= writes_before + 1:
                await asyncio.sleep(0)
            done_reply = decode_callback_reply(_decode_single_frame(writer.writes[writes_before + 1]))
            assert isinstance(done_reply, CallbackReply)

            runtime.session.reader.feed_data(
                encode_frame(
                    msgspec.msgpack.encode(
                        RunFinishedMessage(
                            env_run_id=env_run_id,
                            result=RunResultMessage(
                                run_key="rk-1",
                                status="completed",
                                trace=[
                                    TransitionResultMessage(
                                        bundle=CallbackBundle(
                                            transition_id="write",
                                            run_key="rk-1",
                                            ordinal=1,
                                        ),
                                        firing_id=1,
                                        status="completed",
                                        outputs=stored_reply.outputs,
                                    ),
                                    TransitionResultMessage(
                                        bundle=CallbackBundle(
                                            transition_id="read",
                                            run_key="rk-1",
                                            ordinal=2,
                                        ),
                                        firing_id=2,
                                        status="completed",
                                        outputs=done_reply.outputs,
                                    ),
                                ],
                                final_marking=done_reply.outputs,
                            ),
                        )
                    )
                )
            )
            if env_run_id == 1:
                runtime.session.reader.feed_data(
                    encode_frame(msgspec.msgpack.encode(LoadEnvOk(req_id=5)))
                )
                runtime.session.reader.feed_data(
                    encode_frame(msgspec.msgpack.encode(RunEnvOk(req_id=7, env_run_id=2)))
                )
                runtime.session.reader.feed_data(
                    encode_frame(
                        msgspec.msgpack.encode(
                            CallbackRequest(
                                req_id=6,
                                env_run_id=2,
                                transition_id="write",
                                bundle=CallbackBundle(
                                    transition_id="write",
                                    run_key="rk-1",
                                    ordinal=1,
                                ),
                                tokens=[peven.token({"seed": 2}, run_key="rk-1")],
                                attempt=1,
                            )
                        )
                    )
                )

        driver = asyncio.create_task(drive_adapter())
        try:
            return await bridge_module.run_until_terminal_result(
                runtime=runtime,
                compiled_env=compiled_env,
                env=env,
                env_run_id=env_run_id,
                initial_marking=initial_marking,
                callback=callback,
            )
        finally:
            await driver

    env = RuntimeRunEnv()
    try:
        result_1 = await run_env(
            env,
            command=("fake-runtime",),
            bootstrap_runtime=bootstrap_runtime,
            runtime_runner=runtime_runner,
            seed=1,
        )
        result_2 = await run_env(
            env,
            command=("fake-runtime",),
            bootstrap_runtime=bootstrap_runtime,
            runtime_runner=runtime_runner,
            seed=2,
        )
    finally:
        _reset_shared_runtime_for_tests()

    assert bootstrap_calls == 1
    assert seen_runtime_ids[0] == seen_runtime_ids[1]
    assert seen_loop_ids[0] == seen_loop_ids[1]
    assert seen_env_run_ids == [1, 2]
    assert result_1.status == "completed"
    assert result_2.status == "completed"
    assert result_1.final_marking["done"][0].payload == {
        "kind": "done",
        "payload": {"seed": 1},
    }
    assert result_2.final_marking["done"][0].payload == {
        "kind": "done",
        "payload": {"seed": 2},
    }


@pytest.mark.asyncio
async def test_run_env_serializes_full_run_ownership_on_one_shared_runtime() -> None:
    from peven.runtime.state import _reset_shared_runtime_for_tests

    _reset_shared_runtime_for_tests()
    _register_executor(
        """
@peven.executor("runtime_serialized_alpha")
async def runtime_serialized_alpha(ctx):
    return peven.token({"kind": "alpha"}, run_key=ctx.bundle.run_key)

@peven.executor("runtime_serialized_beta")
async def runtime_serialized_beta(ctx):
    return peven.token({"kind": "beta"}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("runtime_serialized_alpha_env")
    class RuntimeSerializedAlphaEnv(peven.Env):
        done = peven.place()

        def initial_marking(self, seed: int | None = None) -> peven.Marking:
            del seed
            return peven.Marking()

        finish = peven.transition(
            inputs=[],
            outputs=["done"],
            executor="runtime_serialized_alpha",
        )

    @peven.env("runtime_serialized_beta_env")
    class RuntimeSerializedBetaEnv(peven.Env):
        done = peven.place()

        def initial_marking(self, seed: int | None = None) -> peven.Marking:
            del seed
            return peven.Marking()

        finish = peven.transition(
            inputs=[],
            outputs=["done"],
            executor="runtime_serialized_beta",
        )

    bootstrap_calls = 0
    first_runner_started = asyncio.Event()
    release_first_runner = asyncio.Event()
    second_runner_started = asyncio.Event()
    seen_env_run_ids: list[int] = []
    boot = _bootstrapped_runtime(
        replies=[
            LoadEnvOk(req_id=1),
            RunEnvOk(req_id=3, env_run_id=1),
            LoadEnvOk(req_id=5),
            RunEnvOk(req_id=7, env_run_id=2),
        ]
    )

    async def bootstrap_runtime(command: tuple[str, ...]):
        nonlocal bootstrap_calls
        bootstrap_calls += 1
        assert command == ("fake-runtime",)
        return boot

    async def runtime_runner(
        *,
        runtime,
        compiled_env,
        env,
        env_run_id,
        initial_marking,
        callback,
    ):
        del runtime, compiled_env, env, initial_marking, callback
        seen_env_run_ids.append(env_run_id)
        if not first_runner_started.is_set():
            first_runner_started.set()
            await release_first_runner.wait()
        else:
            second_runner_started.set()
        return peven.RunResult(run_key=f"rk-{env_run_id}", status="completed", trace=[], final_marking={})

    try:
        task_1 = asyncio.create_task(
            run_env(
                RuntimeSerializedAlphaEnv(),
                command=("fake-runtime",),
                bootstrap_runtime=bootstrap_runtime,
                runtime_runner=runtime_runner,
            )
        )
        task_2 = asyncio.create_task(
            run_env(
                RuntimeSerializedBetaEnv(),
                command=("fake-runtime",),
                bootstrap_runtime=bootstrap_runtime,
                runtime_runner=runtime_runner,
            )
        )

        await first_runner_started.wait()
        await asyncio.sleep(0)

        assert second_runner_started.is_set() is False
        assert len(boot.writer.writes) == 2

        release_first_runner.set()

        result_1, result_2 = await asyncio.gather(task_1, task_2)
    finally:
        _reset_shared_runtime_for_tests()

    assert bootstrap_calls == 1
    assert seen_env_run_ids == [1, 2]
    assert len(boot.writer.writes) == 4
    assert result_1.status == "completed"
    assert result_2.status == "completed"


@pytest.mark.asyncio
async def test_run_env_defaults_to_the_terminal_runtime_runner() -> None:
    from peven.runtime.state import _reset_shared_runtime_for_tests

    _reset_shared_runtime_for_tests()

    _register_executor(
        """
@peven.executor("runtime_validate_only")
async def runtime_validate_only(ctx):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("runtime_runner_required_env")
    class RuntimeRunnerRequiredEnv(peven.Env):
        done = peven.place()

        def initial_marking(self, seed: int | None = None) -> peven.Marking:
            del seed
            return peven.Marking()

        finish = peven.transition(
            inputs=[],
            outputs=["done"],
            executor="runtime_validate_only",
        )

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

    result = await run_env(
        RuntimeRunnerRequiredEnv(),
        command=("fake-runtime",),
        bootstrap_runtime=bootstrap_runtime,
    )

    assert isinstance(result, peven.RunResult)
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_run_env_uses_the_default_runtime_command_when_none_is_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from peven.runtime.state import _reset_shared_runtime_for_tests

    _reset_shared_runtime_for_tests()
    sentinel = object()
    _register_executor(
        """
@peven.executor("runtime_command_required")
async def runtime_command_required(ctx):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("runtime_command_required_env")
    class RuntimeCommandRequiredEnv(peven.Env):
        done = peven.place()

        def initial_marking(self, seed: int | None = None) -> peven.Marking:
            del seed
            return peven.Marking()

        finish = peven.transition(
            inputs=[],
            outputs=["done"],
            executor="runtime_command_required",
        )

    async def runtime_runner(**kwargs: object) -> object:
        del kwargs
        return sentinel

    monkeypatch.setattr(
        bridge_module,
        "default_runtime_command",
        lambda: ("default-runtime",),
    )

    async def bootstrap_runtime(command: tuple[str, ...]):
        assert command == ("default-runtime",)
        return _bootstrapped_runtime(
            replies=[LoadEnvOk(req_id=1), RunEnvOk(req_id=3, env_run_id=1)]
        )

    try:
        result = await run_env(
            RuntimeCommandRequiredEnv(),
            bootstrap_runtime=bootstrap_runtime,
            runtime_runner=runtime_runner,
        )
    finally:
        _reset_shared_runtime_for_tests()

    assert result is sentinel


@pytest.mark.asyncio
async def test_run_env_requires_initial_marking_to_return_a_marking() -> None:
    _register_executor(
        """
@peven.executor("runtime_bad_marking_executor")
async def runtime_bad_marking_executor(ctx):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("runtime_bad_marking_env")
    class RuntimeBadMarkingEnv(peven.Env):
        done = peven.place()

        def initial_marking(self, seed: int | None = None):
            del seed
            return {"done": []}

        finish = peven.transition(
            inputs=[],
            outputs=["done"],
            executor="runtime_bad_marking_executor",
        )

    async def runtime_runner(**kwargs: object) -> object:
        del kwargs
        return object()

    with pytest.raises(TypeError, match=r"must return a peven\.Marking"):
        await run_env(
            RuntimeBadMarkingEnv(),
            command=("fake-runtime",),
            runtime_runner=runtime_runner,
        )


@pytest.mark.asyncio
async def test_run_env_marks_the_run_failed_when_the_runtime_runner_raises() -> None:
    from peven.runtime.state import _reset_shared_runtime_for_tests, get_shared_runtime

    _reset_shared_runtime_for_tests()
    _register_executor(
        """
@peven.executor("runtime_fail_runner_executor")
async def runtime_fail_runner_executor(ctx):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("runtime_fail_runner_env")
    class RuntimeFailRunnerEnv(peven.Env):
        done = peven.place()

        def initial_marking(self, seed: int | None = None) -> peven.Marking:
            del seed
            return peven.Marking()

        finish = peven.transition(
            inputs=[],
            outputs=["done"],
            executor="runtime_fail_runner_executor",
        )

    async def bootstrap_runtime(command: tuple[str, ...]):
        del command
        return _bootstrapped_runtime(
            replies=[LoadEnvOk(req_id=1), RunEnvOk(req_id=3, env_run_id=1)]
        )

    async def runtime_runner(**kwargs: object) -> object:
        del kwargs
        raise RuntimeError("runner boom")

    try:
        with pytest.raises(RuntimeError, match="runner boom"):
            await run_env(
                RuntimeFailRunnerEnv(),
                command=("fake-runtime",),
                bootstrap_runtime=bootstrap_runtime,
                runtime_runner=runtime_runner,
            )

        runtime = await get_shared_runtime(
            command=("fake-runtime",), bootstrap_runtime=bootstrap_runtime
        )
        assert runtime.runs == {}
    finally:
        _reset_shared_runtime_for_tests()


@pytest.mark.asyncio
async def test_run_env_marks_the_runtime_crashed_after_protocol_errors() -> None:
    from peven.runtime.state import _reset_shared_runtime_for_tests

    _reset_shared_runtime_for_tests()
    bootstrap_calls = 0

    @peven.env("runtime_protocol_error_env")
    class RuntimeProtocolErrorEnv(peven.Env):
        done = peven.place()

        def initial_marking(self, seed: int | None = None) -> peven.Marking:
            del seed
            return peven.Marking()

    async def bootstrap_runtime(command: tuple[str, ...]):
        nonlocal bootstrap_calls
        bootstrap_calls += 1
        del command
        if bootstrap_calls == 1:
            return _bootstrapped_runtime(
                replies=[
                    LoadEnvOk(req_id=1),
                    RunEnvOk(req_id=3, env_run_id=1),
                    TransitionStartedMessage(
                        env_run_id=999,
                        bundle=CallbackBundle(
                            transition_id="ghost",
                            run_key="rk-1",
                            ordinal=1,
                        ),
                        firing_id=1,
                        attempt=1,
                        inputs=[],
                    ),
                ]
            )
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
        with pytest.raises(bridge_module.AdapterProtocolError, match="unknown env_run_id"):
            await run_env(
                RuntimeProtocolErrorEnv(),
                command=("fake-runtime",),
                bootstrap_runtime=bootstrap_runtime,
                runtime_runner=bridge_module.run_until_terminal_result,
            )

        result = await run_env(
            RuntimeProtocolErrorEnv(),
            command=("fake-runtime",),
            bootstrap_runtime=bootstrap_runtime,
            runtime_runner=bridge_module.run_until_terminal_result,
        )
    finally:
        _reset_shared_runtime_for_tests()

    assert bootstrap_calls == 2
    assert isinstance(result, peven.RunResult)
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_run_env_reuses_live_runtime_after_load_rejection() -> None:
    from peven.runtime.state import _reset_shared_runtime_for_tests

    _reset_shared_runtime_for_tests()
    bootstrap_calls = 0

    @peven.env("runtime_load_rejection_env")
    class RuntimeLoadRejectionEnv(peven.Env):
        done = peven.place()

        def initial_marking(self, seed: int | None = None) -> peven.Marking:
            del seed
            return peven.Marking()

    async def bootstrap_runtime(command: tuple[str, ...]):
        nonlocal bootstrap_calls
        bootstrap_calls += 1
        del command
        return _bootstrapped_runtime(
            replies=[
                LoadEnvError(req_id=1, error="bad env"),
                LoadEnvOk(req_id=3),
                RunEnvOk(req_id=5, env_run_id=2),
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

    try:
        with pytest.raises(bridge_module.LoadEnvRejectedError, match="bad env"):
            await run_env(
                RuntimeLoadRejectionEnv(),
                command=("fake-runtime",),
                bootstrap_runtime=bootstrap_runtime,
                runtime_runner=bridge_module.run_until_terminal_result,
            )

        result = await run_env(
            RuntimeLoadRejectionEnv(),
            command=("fake-runtime",),
            bootstrap_runtime=bootstrap_runtime,
            runtime_runner=bridge_module.run_until_terminal_result,
        )
    finally:
        _reset_shared_runtime_for_tests()

    assert bootstrap_calls == 1
    assert isinstance(result, peven.RunResult)
    assert result.run_key == "rk-2"


@pytest.mark.asyncio
async def test_run_env_reuses_live_runtime_after_run_rejection() -> None:
    from peven.runtime.state import _reset_shared_runtime_for_tests

    _reset_shared_runtime_for_tests()
    bootstrap_calls = 0

    @peven.env("runtime_run_rejection_env")
    class RuntimeRunRejectionEnv(peven.Env):
        done = peven.place()

        def initial_marking(self, seed: int | None = None) -> peven.Marking:
            del seed
            return peven.Marking()

    async def bootstrap_runtime(command: tuple[str, ...]):
        nonlocal bootstrap_calls
        bootstrap_calls += 1
        del command
        return _bootstrapped_runtime(
            replies=[
                LoadEnvOk(req_id=1),
                bridge_module.RunEnvError(req_id=3, env_run_id=1, error="bad run"),
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

    try:
        with pytest.raises(bridge_module.RunEnvRejectedError, match="bad run"):
            await run_env(
                RuntimeRunRejectionEnv(),
                command=("fake-runtime",),
                bootstrap_runtime=bootstrap_runtime,
                runtime_runner=bridge_module.run_until_terminal_result,
            )

        result = await run_env(
            RuntimeRunRejectionEnv(),
            command=("fake-runtime",),
            bootstrap_runtime=bootstrap_runtime,
            runtime_runner=bridge_module.run_until_terminal_result,
        )
    finally:
        _reset_shared_runtime_for_tests()

    assert bootstrap_calls == 1
    assert isinstance(result, peven.RunResult)
    assert result.run_key == "rk-2"


@pytest.mark.asyncio
async def test_run_env_reboots_after_request_write_failures() -> None:
    from peven.runtime.state import _reset_shared_runtime_for_tests

    _reset_shared_runtime_for_tests()
    bootstrap_calls = 0

    @peven.env("runtime_request_write_failure_env")
    class RuntimeRequestWriteFailureEnv(peven.Env):
        done = peven.place()

        def initial_marking(self, seed: int | None = None) -> peven.Marking:
            del seed
            return peven.Marking()

    async def bootstrap_runtime(command: tuple[str, ...]):
        nonlocal bootstrap_calls
        bootstrap_calls += 1
        del command
        if bootstrap_calls == 1:
            session = _bootstrapped_runtime()
            session.writer = _BrokenPipeWriter()
            return session
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
        with pytest.raises(bridge_module.AdapterTransportError, match="write failed"):
            await run_env(
                RuntimeRequestWriteFailureEnv(),
                command=("fake-runtime",),
                bootstrap_runtime=bootstrap_runtime,
                runtime_runner=bridge_module.run_until_terminal_result,
            )

        result = await run_env(
            RuntimeRequestWriteFailureEnv(),
            command=("fake-runtime",),
            bootstrap_runtime=bootstrap_runtime,
            runtime_runner=bridge_module.run_until_terminal_result,
        )
    finally:
        _reset_shared_runtime_for_tests()

    assert bootstrap_calls == 2
    assert isinstance(result, peven.RunResult)
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_run_env_marks_the_runtime_crashed_after_reader_eof_mid_run() -> None:
    from peven.runtime.state import _reset_shared_runtime_for_tests

    _reset_shared_runtime_for_tests()
    bootstrap_calls = 0

    @peven.env("runtime_reader_eof_env")
    class RuntimeReaderEofEnv(peven.Env):
        done = peven.place()

        def initial_marking(self, seed: int | None = None) -> peven.Marking:
            del seed
            return peven.Marking()

    async def bootstrap_runtime(command: tuple[str, ...]):
        nonlocal bootstrap_calls
        bootstrap_calls += 1
        del command
        if bootstrap_calls == 1:
            return _bootstrapped_runtime(
                replies=[LoadEnvOk(req_id=1), RunEnvOk(req_id=3, env_run_id=1)]
            )
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

    async def runtime_runner(**kwargs: object) -> object:
        runtime = kwargs["runtime"]
        runtime.session.reader.feed_eof()
        return await bridge_module.run_until_terminal_result(**kwargs)

    try:
        with pytest.raises(
            bridge_module.AdapterProtocolError,
            match="closed while waiting for adapter message",
        ):
            await run_env(
                RuntimeReaderEofEnv(),
                command=("fake-runtime",),
                bootstrap_runtime=bootstrap_runtime,
                runtime_runner=runtime_runner,
            )

        result = await run_env(
            RuntimeReaderEofEnv(),
            command=("fake-runtime",),
            bootstrap_runtime=bootstrap_runtime,
            runtime_runner=bridge_module.run_until_terminal_result,
        )
    finally:
        _reset_shared_runtime_for_tests()

    assert bootstrap_calls == 2
    assert isinstance(result, peven.RunResult)
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_run_env_marks_the_runtime_crashed_after_malformed_frame_mid_run() -> None:
    from peven.runtime.state import _reset_shared_runtime_for_tests

    _reset_shared_runtime_for_tests()
    bootstrap_calls = 0

    @peven.env("runtime_malformed_frame_env")
    class RuntimeMalformedFrameEnv(peven.Env):
        done = peven.place()

        def initial_marking(self, seed: int | None = None) -> peven.Marking:
            del seed
            return peven.Marking()

    async def bootstrap_runtime(command: tuple[str, ...]):
        nonlocal bootstrap_calls
        bootstrap_calls += 1
        del command
        if bootstrap_calls == 1:
            return _bootstrapped_runtime(
                replies=[LoadEnvOk(req_id=1), RunEnvOk(req_id=3, env_run_id=1)]
            )
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

    async def runtime_runner(**kwargs: object) -> object:
        runtime = kwargs["runtime"]
        runtime.session.reader.feed_data((DEFAULT_MAX_FRAME_BYTES + 1).to_bytes(4, "big"))
        return await bridge_module.run_until_terminal_result(**kwargs)

    try:
        with pytest.raises(
            bridge_module.AdapterProtocolError,
            match="malformed frame",
        ):
            await run_env(
                RuntimeMalformedFrameEnv(),
                command=("fake-runtime",),
                bootstrap_runtime=bootstrap_runtime,
                runtime_runner=runtime_runner,
            )

        result = await run_env(
            RuntimeMalformedFrameEnv(),
            command=("fake-runtime",),
            bootstrap_runtime=bootstrap_runtime,
            runtime_runner=bridge_module.run_until_terminal_result,
        )
    finally:
        _reset_shared_runtime_for_tests()

    assert bootstrap_calls == 2
    assert isinstance(result, peven.RunResult)
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_run_env_returns_failed_run_results_without_python_reinterpretation() -> None:
    from peven.runtime.state import _reset_shared_runtime_for_tests, get_shared_runtime

    _reset_shared_runtime_for_tests()

    @peven.env("runtime_terminal_failure_env")
    class RuntimeTerminalFailureEnv(peven.Env):
        done = peven.place()

        def initial_marking(self, seed: int | None = None) -> peven.Marking:
            del seed
            return peven.Marking()

    async def bootstrap_runtime(command: tuple[str, ...]):
        del command
        return _bootstrapped_runtime(
            replies=[
                LoadEnvOk(req_id=1),
                RunEnvOk(req_id=3, env_run_id=1),
                RunFinishedMessage(
                    env_run_id=1,
                    result=RunResultMessage(
                        run_key="rk-1",
                        status="failed",
                        error="executor boom",
                        terminal_reason="executor_failed",
                        trace=[],
                        final_marking={},
                    ),
                ),
            ]
        )

    try:
        result = await run_env(
            RuntimeTerminalFailureEnv(),
            command=("fake-runtime",),
            bootstrap_runtime=bootstrap_runtime,
            runtime_runner=bridge_module.run_until_terminal_result,
        )

        assert isinstance(result, peven.RunResult)
        assert result.status == "failed"
        assert result.error == "executor boom"
        assert result.terminal_reason == "executor_failed"

        runtime = await get_shared_runtime(
            command=("fake-runtime",), bootstrap_runtime=bootstrap_runtime
        )
        assert runtime.runs == {}
    finally:
        _reset_shared_runtime_for_tests()


@pytest.mark.asyncio
async def test_run_env_evicts_run_state_when_cancelled() -> None:
    from peven.runtime.state import _reset_shared_runtime_for_tests, get_shared_runtime

    _reset_shared_runtime_for_tests()
    _register_executor(
        """
@peven.executor("runtime_cancelled_runner_executor")
async def runtime_cancelled_runner_executor(ctx):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("runtime_cancelled_runner_env")
    class RuntimeCancelledRunnerEnv(peven.Env):
        done = peven.place()

        def initial_marking(self, seed: int | None = None) -> peven.Marking:
            del seed
            return peven.Marking()

        finish = peven.transition(
            inputs=[],
            outputs=["done"],
            executor="runtime_cancelled_runner_executor",
        )

    runner_started = asyncio.Event()

    async def bootstrap_runtime(command: tuple[str, ...]):
        del command
        return _bootstrapped_runtime(
            replies=[LoadEnvOk(req_id=1), RunEnvOk(req_id=3, env_run_id=1)]
        )

    async def runtime_runner(**kwargs: object) -> object:
        del kwargs
        runner_started.set()
        await asyncio.Future()

    task = asyncio.create_task(
        run_env(
            RuntimeCancelledRunnerEnv(),
            command=("fake-runtime",),
            bootstrap_runtime=bootstrap_runtime,
            runtime_runner=runtime_runner,
        )
    )

    try:
        await runner_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        runtime = await get_shared_runtime(
            command=("fake-runtime",), bootstrap_runtime=bootstrap_runtime
        )
        assert runtime.runs == {}
    finally:
        _reset_shared_runtime_for_tests()
