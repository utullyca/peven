"""Length-prefixed wire framing for handoff/runtime transport."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Protocol


FRAME_LENGTH_BYTES = 4
DEFAULT_MAX_FRAME_BYTES = 8 * 1024 * 1024

__all__ = [
    "DEFAULT_MAX_FRAME_BYTES",
    "FRAME_LENGTH_BYTES",
    "FrameDecoder",
    "FrameTooLargeError",
    "encode_frame",
    "read_frame",
    "write_frame",
]


class FrameTooLargeError(ValueError):
    """Raised when one announced or encoded frame exceeds the configured limit."""


class _AsyncFrameWriter(Protocol):
    def write(self, data: bytes) -> object: ...

    async def drain(self) -> None: ...


def encode_frame(payload: bytes, *, max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES) -> bytes:
    """Prefix one payload with its 4-byte big-endian frame length."""
    if len(payload) > max_frame_bytes:
        raise FrameTooLargeError(
            f"frame size {len(payload)} exceeds max frame size {max_frame_bytes}"
        )
    return len(payload).to_bytes(FRAME_LENGTH_BYTES, "big") + payload


@dataclass(slots=True)
class FrameDecoder:
    """Incrementally decode length-prefixed frames from arbitrary byte chunks."""

    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES
    _buffer: bytearray = field(default_factory=bytearray)

    def feed(self, data: bytes) -> list[bytes]:
        self._buffer.extend(data)
        frames: list[bytes] = []

        while True:
            if len(self._buffer) < FRAME_LENGTH_BYTES:
                return frames
            frame_length = int.from_bytes(self._buffer[:FRAME_LENGTH_BYTES], "big")
            _validate_frame_length(frame_length, max_frame_bytes=self.max_frame_bytes)
            frame_end = FRAME_LENGTH_BYTES + frame_length
            if len(self._buffer) < frame_end:
                return frames
            frames.append(bytes(self._buffer[FRAME_LENGTH_BYTES:frame_end]))
            del self._buffer[:frame_end]


async def read_frame(
    reader: asyncio.StreamReader,
    *,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> bytes:
    """Read exactly one framed payload from an asyncio stream."""
    length_prefix = await reader.readexactly(FRAME_LENGTH_BYTES)
    frame_length = int.from_bytes(length_prefix, "big")
    _validate_frame_length(frame_length, max_frame_bytes=max_frame_bytes)
    return await reader.readexactly(frame_length)


async def write_frame(
    writer: _AsyncFrameWriter,
    payload: bytes,
    *,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> None:
    """Encode and write exactly one framed payload to an async writer."""
    writer.write(encode_frame(payload, max_frame_bytes=max_frame_bytes))
    await writer.drain()


def _validate_frame_length(frame_length: int, *, max_frame_bytes: int) -> None:
    if frame_length > max_frame_bytes:
        raise FrameTooLargeError(
            f"frame size {frame_length} exceeds max frame size {max_frame_bytes}"
        )
