from __future__ import annotations

import asyncio

import pytest

from peven.handoff.framing import (
    DEFAULT_MAX_FRAME_BYTES,
    FrameDecoder,
    FrameTooLargeError,
    encode_frame,
    read_frame,
    write_frame,
)


def test_encode_frame_prefixes_length_in_big_endian_order() -> None:
    frame = encode_frame(b"hello")

    assert frame[:4] == b"\x00\x00\x00\x05"
    assert frame[4:] == b"hello"


def test_frame_decoder_handles_partial_prefix_and_body_chunks() -> None:
    decoder = FrameDecoder()
    encoded = encode_frame(b"payload")

    assert decoder.feed(encoded[:2]) == []
    assert decoder.feed(encoded[2:6]) == []
    assert decoder.feed(encoded[6:]) == [b"payload"]


def test_frame_decoder_returns_multiple_frames_from_one_chunk() -> None:
    decoder = FrameDecoder()
    chunk = encode_frame(b"one") + encode_frame(b"two")

    assert decoder.feed(chunk) == [b"one", b"two"]


def test_encode_frame_rejects_oversized_payloads() -> None:
    with pytest.raises(FrameTooLargeError, match="exceeds max frame size"):
        encode_frame(b"x" * 5, max_frame_bytes=4)


def test_frame_decoder_rejects_oversized_frames_from_prefix() -> None:
    decoder = FrameDecoder(max_frame_bytes=4)

    with pytest.raises(FrameTooLargeError, match="exceeds max frame size"):
        decoder.feed((5).to_bytes(4, "big"))


@pytest.mark.asyncio
async def test_read_frame_reads_one_length_prefixed_payload() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(encode_frame(b"wire"))
    reader.feed_eof()

    assert await read_frame(reader) == b"wire"


@pytest.mark.asyncio
async def test_read_frame_rejects_oversized_payload_prefix() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data((DEFAULT_MAX_FRAME_BYTES + 1).to_bytes(4, "big"))
    reader.feed_eof()

    with pytest.raises(FrameTooLargeError, match="exceeds max frame size"):
        await read_frame(reader)


class _FakeWriter:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.drained = False

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def drain(self) -> None:
        self.drained = True


@pytest.mark.asyncio
async def test_write_frame_writes_encoded_payload_and_drains() -> None:
    writer = _FakeWriter()

    await write_frame(writer, b"ack")

    assert writer.writes == [encode_frame(b"ack")]
    assert writer.drained is True
