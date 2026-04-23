from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path

import msgspec
import pytest

from peven.handoff.callbacks import invoke_transition
from peven.handoff.framing import encode_frame
from peven.handoff.lowering import CompiledEnv
from peven.runtime.bootstrap import HANDSHAKE_TAG, BootstrappedRuntime, Handshake
from peven.runtime.sinks import Sink
from peven.shared.events import BundleRef
from peven.shared.token import Token


class FakeProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class FakeWriter:
    def __init__(self, *, closed: bool = False) -> None:
        self.closed = closed
        self.writes: list[bytes] = []

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def drain(self) -> None:
        return None


@dataclass
class RecordingSink:
    events: list[object]
    closed_with: BaseException | None = None

    def write(self, event: object) -> None:
        self.events.append(event)

    def close(self, exc: BaseException | None = None) -> None:
        self.closed_with = exc


def make_session(
    *,
    frames: list[object] | None = None,
    raw_frames: list[bytes] | None = None,
    writer: object | None = None,
    reader: asyncio.StreamReader | None = None,
) -> BootstrappedRuntime:
    if reader is None:
        reader = asyncio.StreamReader()
    if frames is not None:
        for frame in frames:
            reader.feed_data(encode_frame(msgspec.msgpack.encode(frame)))
    if raw_frames is not None:
        for frame in raw_frames:
            reader.feed_data(frame)
    return BootstrappedRuntime(
        process=FakeProcess(),
        reader=reader,
        writer=FakeWriter() if writer is None else writer,
        socket_dir=Path("/tmp/peven-runtime"),
        socket_path=Path("/tmp/peven-runtime/runtime.sock"),
        handshake=Handshake(
            tag=HANDSHAKE_TAG,
            protocol_version="0.1.0",
            peven_version="0.2.0",
        ),
    )


def make_transition_callback(
    compiled_env: CompiledEnv,
    env: object,
    *,
    sink: Sink | None = None,
):
    async def callback(
        transition_id: str,
        bundle: BundleRef,
        tokens: list[Token] | tuple[Token, ...],
        *,
        attempt: int,
        inputs_by_place: dict[str, list[Token]] | None = None,
    ) -> dict[str, list[Token]]:
        return await invoke_transition(
            compiled_env,
            transition_id,
            env,
            bundle,
            tokens,
            attempt=attempt,
            inputs_by_place=inputs_by_place,
            sink=sink,
        )

    return callback


def require_adapter_command() -> tuple[str, ...]:
    raw = os.environ.get("PEVEN_ADAPTER_COMMAND_JSON")
    if raw is None:
        pytest.skip("set PEVEN_ADAPTER_COMMAND_JSON to run real adapter integration tests")
    try:
        command = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("PEVEN_ADAPTER_COMMAND_JSON must be valid JSON") from exc
    if type(command) is not list or not command or not all(
        type(part) is str and part for part in command
    ):
        raise RuntimeError(
            "PEVEN_ADAPTER_COMMAND_JSON must decode to a non-empty JSON string array"
        )
    return tuple(command)


def require_pevenpy_project() -> Path:
    configured = os.environ.get("PEVENPY_PROJECT_PATH")
    if configured is not None:
        project = Path(configured).expanduser()
    else:
        project = Path.home() / "PevenPy.jl"
    if not project.exists():
        pytest.skip(
            "set PEVENPY_PROJECT_PATH or create ~/PevenPy.jl to run tests that inspect the external Julia adapter repo"
        )
    return project


def require_external_pevenpy_adapter_command(
    *, fail_event_kind: str | None = None
) -> tuple[str, ...]:
    require_adapter_command()
    project = require_pevenpy_project()
    expression = "using PevenPy; "
    if fail_event_kind is None:
        expression += "PevenPy.main(ARGS[1])"
    else:
        expression += f'PevenPy._test_main(ARGS[1]; fail_event_kind="{fail_event_kind}")'
    return (
        "julia",
        f"--project={project}",
        "-e",
        expression,
        "{socket_path}",
    )
