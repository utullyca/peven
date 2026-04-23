from __future__ import annotations

import asyncio

import msgspec
import pytest

import peven
import peven.runtime.bridge as bridge_module
from peven.handoff.framing import FrameDecoder, encode_frame
from peven.handoff.lowering import compile_env, normalize_initial_marking
from peven.handoff.messages import LoadEnvError, LoadEnvOk, RunEnvError, RunEnvOk
from peven.runtime.state import SharedRuntime

from .conftest import FakeWriter, make_session


def _register_executor(source: str) -> dict[str, object]:
    namespace = {
        "__name__": "tests.runtime.generated_bridge_load_run_executors",
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
async def test_load_and_run_messages_are_sent_in_order_over_the_session() -> None:
    _register_executor(
        """
@peven.executor("bridge_load_run_executor")
async def bridge_load_run_executor(ctx, ready):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("bridge_load_run_env")
    class BridgeLoadRunEnv(peven.Env):
        ready = peven.place()
        done = peven.place()

        finish = peven.transition(
            inputs=["ready"],
            outputs=["done"],
            executor="bridge_load_run_executor",
        )

    writer = FakeWriter()
    session = make_session(
        frames=[LoadEnvOk(req_id=1), RunEnvOk(req_id=3, env_run_id=1)],
        writer=writer,
    )
    runtime = SharedRuntime(
        session=session,
        loop=asyncio.get_running_loop(),
        command=("fake-runtime",),
    )
    compiled = compile_env(BridgeLoadRunEnv.spec())
    marking = normalize_initial_marking(
        peven.Marking({"ready": [peven.token({"seed": 1}, run_key="rk-1")]})
    )

    await bridge_module._load_compiled_env(runtime, compiled)
    await bridge_module._start_run(runtime, env_run_id=1, initial_marking=marking, fuse=7)

    assert len(writer.writes) == 2
    load_payload = msgspec.msgpack.decode(
        _decode_single_frame(writer.writes[0]),
        type=dict[str, object],
    )
    run_payload = msgspec.msgpack.decode(
        _decode_single_frame(writer.writes[1]),
        type=dict[str, object],
    )
    assert load_payload["kind"] == "load_env"
    assert load_payload["req_id"] == 1
    assert run_payload["kind"] == "run_env"
    assert run_payload["req_id"] == 3
    assert run_payload["env_run_id"] == 1
    assert run_payload["fuse"] == 7


@pytest.mark.asyncio
async def test_load_and_run_failures_raise_typed_bridge_errors() -> None:
    reader = asyncio.StreamReader()
    session = make_session(reader=reader)
    runtime = SharedRuntime(
        session=session,
        loop=asyncio.get_running_loop(),
        command=("fake-runtime",),
    )

    @peven.env("bridge_load_run_error_env")
    class BridgeLoadRunErrorEnv(peven.Env):
        done = peven.place()

    compiled = compile_env(BridgeLoadRunErrorEnv.spec())

    reader.feed_data(encode_frame(msgspec.msgpack.encode(LoadEnvError(req_id=1, error="bad env"))))

    with pytest.raises(bridge_module.LoadEnvRejectedError, match="bad env"):
        await bridge_module._load_compiled_env(runtime, compiled)

    reader = asyncio.StreamReader()
    session = make_session(reader=reader)
    runtime = SharedRuntime(
        session=session,
        loop=asyncio.get_running_loop(),
        command=("fake-runtime",),
    )
    reader.feed_data(encode_frame(msgspec.msgpack.encode(RunEnvError(req_id=1, env_run_id=1, error="bad run"))))

    with pytest.raises(bridge_module.RunEnvRejectedError, match="bad run"):
        await bridge_module._start_run(runtime, env_run_id=1, initial_marking={})


@pytest.mark.asyncio
async def test_load_and_run_reject_reply_id_mismatches_inline() -> None:
    runtime = SharedRuntime(
        session=make_session(frames=[LoadEnvOk(req_id=9)]),
        loop=asyncio.get_running_loop(),
        command=("fake-runtime",),
    )

    @peven.env("bridge_load_run_req_id_env")
    class BridgeLoadRunReqIdEnv(peven.Env):
        done = peven.place()

    compiled = compile_env(BridgeLoadRunReqIdEnv.spec())

    with pytest.raises(
        bridge_module.AdapterProtocolError,
        match="load reply did not match the request",
    ):
        await bridge_module._load_compiled_env(runtime, compiled)

    runtime = SharedRuntime(
        session=make_session(frames=[RunEnvOk(req_id=9, env_run_id=1)]),
        loop=asyncio.get_running_loop(),
        command=("fake-runtime",),
    )

    with pytest.raises(
        bridge_module.AdapterProtocolError,
        match="run reply did not match the request",
    ):
        await bridge_module._start_run(runtime, env_run_id=1, initial_marking={})


@pytest.mark.asyncio
async def test_exchange_session_frame_serializes_access_per_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = asyncio.StreamReader()
    writer = FakeWriter()
    runtime = SharedRuntime(
        session=make_session(reader=reader, writer=writer),
        loop=asyncio.get_running_loop(),
        command=("fake-runtime",),
    )
    first_read_started = asyncio.Event()
    release_first_read = asyncio.Event()
    writes: list[bytes] = []
    second_read_seen = False

    async def fake_write_frame(seen_writer: object, payload: bytes) -> None:
        assert seen_writer is writer
        writes.append(payload)

    async def fake_read_frame(seen_reader: asyncio.StreamReader) -> bytes:
        nonlocal second_read_seen
        assert seen_reader is reader
        if not first_read_started.is_set():
            first_read_started.set()
            await release_first_read.wait()
            return b"first"
        second_read_seen = True
        return b"second"

    monkeypatch.setattr(bridge_module, "write_frame", fake_write_frame)
    monkeypatch.setattr(bridge_module, "read_frame", fake_read_frame)

    task_1 = asyncio.create_task(bridge_module._exchange_session_frame(runtime, b"one"))
    task_2 = asyncio.create_task(bridge_module._exchange_session_frame(runtime, b"two"))

    await first_read_started.wait()
    await asyncio.sleep(0)

    assert second_read_seen is False
    assert writes == [b"one"]

    release_first_read.set()

    assert await task_1 == b"first"
    assert await task_2 == b"second"
    assert writes == [b"one", b"two"]
