from __future__ import annotations

import asyncio

import pytest

import peven.runtime.bridge as bridge_module
from peven.runtime.state import SharedRuntime

from .conftest import FakeWriter, make_session


def _runtime(
    *,
    writer: object,
    reader: asyncio.StreamReader | None = None,
) -> SharedRuntime:
    return SharedRuntime(
        session=make_session(writer=writer, reader=reader),
        loop=asyncio.get_running_loop(),
        command=("fake-runtime",),
    )


@pytest.mark.asyncio
async def test_exchange_session_frame_uses_framed_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = asyncio.StreamReader()
    writer = FakeWriter()
    runtime = _runtime(writer=writer, reader=reader)
    seen: dict[str, object] = {}

    async def fake_write_frame(seen_writer: object, payload: bytes) -> None:
        seen["writer"] = seen_writer
        seen["payload"] = payload

    async def fake_read_frame(seen_reader: asyncio.StreamReader) -> bytes:
        seen["reader"] = seen_reader
        return b"reply"

    monkeypatch.setattr(bridge_module, "write_frame", fake_write_frame)
    monkeypatch.setattr(bridge_module, "read_frame", fake_read_frame)

    reply = await bridge_module._exchange_session_frame(runtime, b"request")

    assert reply == b"reply"
    assert seen == {
        "writer": writer,
        "payload": b"request",
        "reader": reader,
    }


@pytest.mark.asyncio
async def test_exchange_session_frame_rejects_missing_or_closed_writer() -> None:
    missing_writer_runtime = _runtime(writer=FakeWriter())
    missing_writer_runtime.session.writer = None

    with pytest.raises(
        bridge_module.AdapterTransportError, match="missing its reader/writer"
    ):
        await bridge_module._exchange_session_frame(missing_writer_runtime, b"request")

    with pytest.raises(
        bridge_module.AdapterTransportError, match="writer is already closed"
    ):
        await bridge_module._exchange_session_frame(
            _runtime(writer=FakeWriter(closed=True)),
            b"request",
        )


@pytest.mark.asyncio
async def test_exchange_session_frame_rejects_reader_eof_before_first_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(writer=FakeWriter())

    async def fake_read_frame(reader: asyncio.StreamReader) -> bytes:
        del reader
        raise asyncio.IncompleteReadError(partial=b"", expected=4)

    monkeypatch.setattr(bridge_module, "read_frame", fake_read_frame)

    with pytest.raises(
        bridge_module.AdapterTransportError, match="closed before the first reply"
    ):
        await bridge_module._exchange_session_frame(runtime, b"request")
