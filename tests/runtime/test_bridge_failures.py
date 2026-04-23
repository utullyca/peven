from __future__ import annotations

import asyncio

import msgspec
import pytest

import peven
import peven.runtime.bridge as bridge_module
from peven.handoff.framing import DEFAULT_MAX_FRAME_BYTES, encode_frame
from peven.handoff.lowering import compile_env
from peven.runtime.state import SharedRuntime, open_run

from .conftest import FakeWriter, make_session, make_transition_callback


class _BrokenPipeWriter(FakeWriter):
    def write(self, data: bytes) -> None:
        del data
        raise BrokenPipeError("broken pipe")


class _ExplodingIsClosingWriter(FakeWriter):
    def is_closing(self) -> bool:
        raise RuntimeError("boom")


def _register_executor(source: str) -> dict[str, object]:
    namespace = {
        "__name__": "tests.runtime.generated_bridge_failure_executors",
        "peven": peven,
    }
    exec(source, namespace)
    return namespace


def _build_run(
    *,
    name: str,
    raw_frames: list[bytes] | None = None,
    frames: list[object] | None = None,
    writer: object | None = None,
) -> tuple[SharedRuntime, object, object]:
    _register_executor(
        f"""
@peven.executor("{name}_exec")
async def {name}_exec(ctx):
    return peven.token({{"ok": True}}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env(f"{name}_env")
    class Env(peven.Env):
        done = peven.place()

        finish = peven.transition(inputs=[], outputs=["done"], executor=f"{name}_exec")

    runtime = SharedRuntime(
        session=make_session(raw_frames=raw_frames, frames=frames, writer=writer),
        loop=asyncio.get_running_loop(),
        command=("fake-runtime",),
    )
    open_run(runtime, 1)
    return runtime, Env(), compile_env(Env.spec())


async def _run(runtime: SharedRuntime, env: object, compiled: object) -> None:
    await bridge_module.run_until_terminal_result(
        runtime=runtime,
        compiled_env=compiled,
        env=env,
        env_run_id=1,
        initial_marking={},
        callback=make_transition_callback(compiled, env),
    )


@pytest.mark.asyncio
async def test_run_until_terminal_result_treats_reader_eof_as_transport_error() -> None:
    runtime, env, compiled = _build_run(name="failure_eof")
    runtime.session.reader.feed_eof()

    with pytest.raises(
        bridge_module.AdapterTransportError,
        match="closed while waiting for adapter message",
    ):
        await _run(runtime, env, compiled)


@pytest.mark.asyncio
async def test_run_until_terminal_result_rejects_malformed_frames() -> None:
    oversized_frame = (DEFAULT_MAX_FRAME_BYTES + 1).to_bytes(4, "big")
    runtime, env, compiled = _build_run(name="failure_frame", raw_frames=[oversized_frame])

    with pytest.raises(bridge_module.AdapterProtocolError, match="malformed frame"):
        await _run(runtime, env, compiled)


@pytest.mark.asyncio
async def test_run_until_terminal_result_rejects_malformed_callback_requests() -> None:
    malformed_callback = encode_frame(
        msgspec.msgpack.encode(
            {
                "kind": "callback_request",
                "req_id": 1,
                "env_run_id": 1,
                "transition_id": "finish",
                "bundle": {
                    "transition_id": "finish",
                    "run_key": "rk-1",
                    "ordinal": 1,
                },
                "tokens": [],
                "attempt": 1,
            }
        )
    )
    runtime, env, compiled = _build_run(
        name="failure_callback", raw_frames=[malformed_callback]
    )

    with pytest.raises(
        bridge_module.AdapterProtocolError,
        match="malformed callback request",
    ):
        await _run(runtime, env, compiled)


@pytest.mark.asyncio
async def test_run_until_terminal_result_treats_callback_reply_write_failures_as_transport_errors() -> None:
    callback_frame = encode_frame(
        msgspec.msgpack.encode(
            {
                "kind": "callback_request",
                "req_id": 2,
                "env_run_id": 1,
                "transition_id": "finish",
                "bundle": {
                    "transition_id": "finish",
                    "run_key": "rk-1",
                    "ordinal": 1,
                },
                "tokens": [],
                "attempt": 1,
            }
        )
    )
    runtime, env, compiled = _build_run(
        name="failure_write",
        raw_frames=[callback_frame],
        writer=_BrokenPipeWriter(),
    )

    with pytest.raises(bridge_module.AdapterTransportError, match="write failed"):
        await _run(runtime, env, compiled)


@pytest.mark.asyncio
async def test_writer_helpers_cover_closed_and_exceptional_paths() -> None:
    session = make_session(writer=_ExplodingIsClosingWriter())
    reader, writer = bridge_module._require_session_streams(session)

    assert reader is session.reader
    assert writer is session.writer

    closed_session = make_session()
    closed_session.writer.close()
    with pytest.raises(RuntimeError, match="writer is already closed"):
        bridge_module._require_session_streams(closed_session)


def test_describe_callback_error_uses_class_name_for_empty_messages() -> None:
    class SilentError(RuntimeError):
        def __str__(self) -> str:
            return ""

    assert bridge_module._describe_callback_error(SilentError()) == "SilentError"
