from __future__ import annotations

import asyncio
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from peven.runtime.bootstrap import (
    DEFAULT_ADAPTER_ENTRYPOINT,
    HANDSHAKE_TAG,
    PEVEN_VERSION,
    PROTOCOL_VERSION,
    BootstrapError,
    BootstrappedRuntime,
    Handshake,
    HandshakeTagError,
    InstalledRuntime,
    PevenVersionMismatchError,
    ProtocolVersionMismatchError,
    bootstrap_runtime,
    close_bootstrapped_runtime,
    default_runtime_command,
    ensure_runtime_installed,
    prepare_socket_location,
    validate_handshake,
)


class _FakeProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self.waited = False

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        self.waited = True
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class _HungTerminateProcess(_FakeProcess):
    async def wait(self) -> int:
        self.waited = True
        if self.killed:
            self.returncode = -9
            return self.returncode
        await asyncio.Future()


class _FakeWriter:
    def __init__(self) -> None:
        self.closed = False
        self.wait_closed_called = False

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        self.wait_closed_called = True


class _FailingWaitClosedWriter(_FakeWriter):
    async def wait_closed(self) -> None:
        self.wait_closed_called = True
        raise ConnectionResetError("socket reset")


def test_prepare_socket_location_creates_private_runtime_dir(tmp_path: Path) -> None:
    socket_dir, socket_path = prepare_socket_location(tmp_path)

    assert socket_dir.parent == tmp_path
    assert socket_path == socket_dir / "runtime.sock"
    assert stat.S_IMODE(socket_dir.stat().st_mode) == 0o700


def test_validate_handshake_accepts_expected_runtime_versions() -> None:
    handshake = validate_handshake(
        Handshake(
            tag=HANDSHAKE_TAG,
            protocol_version=PROTOCOL_VERSION,
            peven_version=PEVEN_VERSION,
        )
    )

    assert handshake.tag == HANDSHAKE_TAG


def test_validate_handshake_rejects_bad_tag_and_protocol_version() -> None:
    with pytest.raises(HandshakeTagError, match="unexpected handshake tag"):
        validate_handshake(
            Handshake(
                tag="wrong",
                protocol_version=PROTOCOL_VERSION,
                peven_version=PEVEN_VERSION,
            )
        )

    with pytest.raises(ProtocolVersionMismatchError, match="protocol version mismatch"):
        validate_handshake(
            Handshake(
                tag=HANDSHAKE_TAG,
                protocol_version="9.9.9",
                peven_version=PEVEN_VERSION,
            )
        )

    with pytest.raises(PevenVersionMismatchError, match="peven version mismatch"):
        validate_handshake(
            Handshake(
                tag=HANDSHAKE_TAG,
                protocol_version=PROTOCOL_VERSION,
                peven_version="9.9.9",
            )
        )


def test_ensure_runtime_installed_resolves_the_packaged_julia_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    project = tmp_path / "julia-env"

    fake_juliapkg = SimpleNamespace(
        resolve=lambda: calls.append("resolve"),
        executable=lambda: calls.append("executable") or "/tmp/julia",
        project=lambda: calls.append("project") or str(project),
    )
    monkeypatch.setitem(sys.modules, "juliapkg", fake_juliapkg)

    installed = ensure_runtime_installed()

    assert installed == InstalledRuntime(
        julia_executable="/tmp/julia",
        julia_project=project,
    )
    assert calls == ["resolve", "executable", "project"]


def test_default_runtime_command_uses_the_resolved_pevenpy_adapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    installed = InstalledRuntime(
        julia_executable="/tmp/julia",
        julia_project=tmp_path / "project",
    )
    monkeypatch.setattr(
        "peven.runtime.bootstrap.ensure_runtime_installed",
        lambda: installed,
    )

    command = default_runtime_command()

    assert command == (
        "/tmp/julia",
        f"--project={installed.julia_project}",
        "-e",
        DEFAULT_ADAPTER_ENTRYPOINT,
        "{socket_path}",
    )


@pytest.mark.asyncio
async def test_bootstrap_runtime_spawns_connects_and_returns_session(tmp_path: Path) -> None:
    process = _FakeProcess()
    reader = asyncio.StreamReader()
    writer = _FakeWriter()
    seen: dict[str, object] = {}

    async def spawn_process(command: tuple[str, ...], *, socket_path: Path) -> _FakeProcess:
        seen["command"] = command
        seen["socket_path"] = socket_path
        return process

    async def connect_socket(socket_path: Path) -> tuple[asyncio.StreamReader, _FakeWriter]:
        seen["connected_socket_path"] = socket_path
        return reader, writer

    async def read_handshake(
        connected_reader: asyncio.StreamReader,
        connected_writer: _FakeWriter,
    ) -> Handshake:
        assert connected_reader is reader
        assert connected_writer is writer
        return Handshake(
            tag=HANDSHAKE_TAG,
            protocol_version=PROTOCOL_VERSION,
            peven_version=PEVEN_VERSION,
        )

    session = await bootstrap_runtime(
        command=("julia", "--socket={socket_path}"),
        socket_root=tmp_path,
        spawn_process=spawn_process,
        connect_socket=connect_socket,
        read_handshake=read_handshake,
    )

    assert session.process is process
    assert session.reader is reader
    assert session.writer is writer
    assert session.handshake.protocol_version == PROTOCOL_VERSION
    assert seen["command"] == ("julia", f"--socket={session.socket_path}")
    assert seen["socket_path"] == session.socket_path
    assert seen["connected_socket_path"] == session.socket_path
    assert session.socket_dir.exists()


def test_render_command_only_substitutes_socket_path_placeholder() -> None:
    import peven.runtime.bootstrap as bootstrap_module

    rendered = bootstrap_module._render_command(
        ("julia", "-e", "print({'ok': 1})", "--socket={socket_path}"),
        Path("/tmp/runtime.sock"),
    )

    assert rendered == (
        "julia",
        "-e",
        "print({'ok': 1})",
        "--socket=/tmp/runtime.sock",
    )


@pytest.mark.asyncio
async def test_bootstrap_runtime_cleans_up_on_handshake_failure(tmp_path: Path) -> None:
    process = _FakeProcess()
    reader = asyncio.StreamReader()
    writer = _FakeWriter()
    seen: dict[str, object] = {}

    async def spawn_process(command: tuple[str, ...], *, socket_path: Path) -> _FakeProcess:
        seen["socket_dir"] = socket_path.parent
        return process

    async def connect_socket(socket_path: Path) -> tuple[asyncio.StreamReader, _FakeWriter]:
        return reader, writer

    async def read_handshake(
        connected_reader: asyncio.StreamReader,
        connected_writer: _FakeWriter,
    ) -> Handshake:
        return Handshake(
            tag=HANDSHAKE_TAG,
            protocol_version="9.9.9",
            peven_version=PEVEN_VERSION,
        )

    with pytest.raises(ProtocolVersionMismatchError, match="protocol version mismatch"):
        await bootstrap_runtime(
            command=("julia", "--socket={socket_path}"),
            socket_root=tmp_path,
            spawn_process=spawn_process,
            connect_socket=connect_socket,
            read_handshake=read_handshake,
        )

    assert writer.closed is True
    assert writer.wait_closed_called is True
    assert process.terminated is True
    assert process.waited is True
    assert not Path(seen["socket_dir"]).exists()


@pytest.mark.asyncio
async def test_bootstrap_runtime_kills_if_terminated_child_does_not_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import peven.runtime.bootstrap as bootstrap_module

    process = _HungTerminateProcess()
    reader = asyncio.StreamReader()
    writer = _FakeWriter()

    monkeypatch.setattr(bootstrap_module, "DEFAULT_TERMINATE_TIMEOUT", 0.01)
    monkeypatch.setattr(bootstrap_module, "DEFAULT_KILL_TIMEOUT", 0.01)

    async def spawn_process(command: tuple[str, ...], *, socket_path: Path) -> _HungTerminateProcess:
        del command, socket_path
        return process

    async def connect_socket(socket_path: Path) -> tuple[asyncio.StreamReader, _FakeWriter]:
        del socket_path
        return reader, writer

    async def read_handshake(
        connected_reader: asyncio.StreamReader,
        connected_writer: _FakeWriter,
    ) -> Handshake:
        del connected_reader, connected_writer
        return Handshake(
            tag=HANDSHAKE_TAG,
            protocol_version="9.9.9",
            peven_version=PEVEN_VERSION,
        )

    with pytest.raises(ProtocolVersionMismatchError, match="protocol version mismatch"):
        await bootstrap_runtime(
            command=("julia", "--socket={socket_path}"),
            socket_root=tmp_path,
            spawn_process=spawn_process,
            connect_socket=connect_socket,
            read_handshake=read_handshake,
        )

    assert process.terminated is True
    assert process.killed is True


@pytest.mark.asyncio
async def test_bootstrap_runtime_times_out_waiting_for_the_handshake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import peven.runtime.bootstrap as bootstrap_module

    process = _FakeProcess()
    reader = asyncio.StreamReader()
    writer = _FakeWriter()
    seen: dict[str, object] = {}

    monkeypatch.setattr(bootstrap_module, "DEFAULT_HANDSHAKE_TIMEOUT", 0.01)

    async def spawn_process(command: tuple[str, ...], *, socket_path: Path) -> _FakeProcess:
        seen["socket_dir"] = socket_path.parent
        del command
        return process

    async def connect_socket(socket_path: Path) -> tuple[asyncio.StreamReader, _FakeWriter]:
        del socket_path
        return reader, writer

    async def read_handshake(
        connected_reader: asyncio.StreamReader,
        connected_writer: _FakeWriter,
    ) -> Handshake:
        del connected_reader, connected_writer
        await asyncio.Future()

    with pytest.raises(BootstrapError, match="timed out waiting for runtime handshake"):
        await bootstrap_runtime(
            command=("julia", "--socket={socket_path}"),
            socket_root=tmp_path,
            spawn_process=spawn_process,
            connect_socket=connect_socket,
            read_handshake=read_handshake,
        )

    assert writer.closed is True
    assert writer.wait_closed_called is True
    assert process.terminated is True
    assert process.waited is True
    assert not Path(seen["socket_dir"]).exists()


@pytest.mark.asyncio
async def test_bootstrap_runtime_cleans_up_when_startup_is_cancelled(tmp_path: Path) -> None:
    process = _FakeProcess()
    reader = asyncio.StreamReader()
    writer = _FakeWriter()
    seen: dict[str, object] = {}

    async def spawn_process(command: tuple[str, ...], *, socket_path: Path) -> _FakeProcess:
        seen["socket_dir"] = socket_path.parent
        del command
        return process

    async def connect_socket(socket_path: Path) -> tuple[asyncio.StreamReader, _FakeWriter]:
        del socket_path
        return reader, writer

    async def read_handshake(
        connected_reader: asyncio.StreamReader,
        connected_writer: _FakeWriter,
    ) -> Handshake:
        del connected_reader, connected_writer
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await bootstrap_runtime(
            command=("julia", "--socket={socket_path}"),
            socket_root=tmp_path,
            spawn_process=spawn_process,
            connect_socket=connect_socket,
            read_handshake=read_handshake,
        )

    assert writer.closed is True
    assert writer.wait_closed_called is True
    assert process.terminated is True
    assert process.waited is True
    assert not Path(seen["socket_dir"]).exists()


@pytest.mark.asyncio
async def test_bootstrap_cleanup_continues_when_wait_closed_errors(tmp_path: Path) -> None:
    process = _FakeProcess()
    reader = asyncio.StreamReader()
    writer = _FailingWaitClosedWriter()
    seen: dict[str, object] = {}

    async def spawn_process(command: tuple[str, ...], *, socket_path: Path) -> _FakeProcess:
        seen["socket_dir"] = socket_path.parent
        del command
        return process

    async def connect_socket(socket_path: Path) -> tuple[asyncio.StreamReader, _FakeWriter]:
        del socket_path
        return reader, writer

    async def read_handshake(
        connected_reader: asyncio.StreamReader,
        connected_writer: _FakeWriter,
    ) -> Handshake:
        del connected_reader, connected_writer
        return Handshake(
            tag=HANDSHAKE_TAG,
            protocol_version="9.9.9",
            peven_version=PEVEN_VERSION,
        )

    with pytest.raises(ProtocolVersionMismatchError, match="protocol version mismatch"):
        await bootstrap_runtime(
            command=("julia", "--socket={socket_path}"),
            socket_root=tmp_path,
            spawn_process=spawn_process,
            connect_socket=connect_socket,
            read_handshake=read_handshake,
        )

    assert writer.closed is True
    assert writer.wait_closed_called is True
    assert process.terminated is True
    assert process.waited is True
    assert not Path(seen["socket_dir"]).exists()


@pytest.mark.asyncio
async def test_close_bootstrapped_runtime_continues_when_wait_closed_errors(
    tmp_path: Path,
) -> None:
    socket_dir = tmp_path / "runtime"
    socket_dir.mkdir()
    session = BootstrappedRuntime(
        process=_FakeProcess(),
        reader=asyncio.StreamReader(),
        writer=_FailingWaitClosedWriter(),
        socket_dir=socket_dir,
        socket_path=socket_dir / "runtime.sock",
        handshake=Handshake(
            tag=HANDSHAKE_TAG,
            protocol_version=PROTOCOL_VERSION,
            peven_version=PEVEN_VERSION,
        ),
    )

    await close_bootstrapped_runtime(session)

    assert session.writer.closed is True
    assert session.writer.wait_closed_called is True
    assert session.process.terminated is True
    assert session.process.waited is True
    assert socket_dir.exists() is False


@pytest.mark.asyncio
async def test_spawn_process_discards_child_stdio_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    import peven.runtime.bootstrap as bootstrap_module

    seen: dict[str, object] = {}

    async def fake_create_subprocess_exec(*command: str, **kwargs: object) -> _FakeProcess:
        seen["command"] = command
        seen["kwargs"] = kwargs
        return _FakeProcess()

    monkeypatch.setattr(
        bootstrap_module.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    await bootstrap_module._spawn_process(("julia", "--project"), socket_path=Path("/tmp/runtime.sock"))

    assert seen["command"] == ("julia", "--project")
    kwargs = seen["kwargs"]
    assert kwargs["stdin"] is bootstrap_module.asyncio.subprocess.DEVNULL
    assert kwargs["stdout"] is bootstrap_module.asyncio.subprocess.DEVNULL
    assert kwargs["stderr"] is bootstrap_module.asyncio.subprocess.DEVNULL


@pytest.mark.asyncio
async def test_connect_socket_retries_until_success(monkeypatch: pytest.MonkeyPatch) -> None:
    import peven.runtime.bootstrap as bootstrap_module

    attempts = 0
    reader = asyncio.StreamReader()
    writer = _FakeWriter()

    async def fake_open_unix_connection(path: str) -> tuple[asyncio.StreamReader, _FakeWriter]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("not ready")
        assert path == "/tmp/runtime.sock"
        return reader, writer

    monkeypatch.setattr(
        bootstrap_module.asyncio,
        "open_unix_connection",
        fake_open_unix_connection,
    )

    connected_reader, connected_writer = await bootstrap_module._connect_socket(
        Path("/tmp/runtime.sock")
    )

    assert attempts == 2
    assert connected_reader is reader
    assert connected_writer is writer


@pytest.mark.asyncio
async def test_connect_socket_fails_fast_when_process_exits_before_socket_appears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import peven.runtime.bootstrap as bootstrap_module

    process = _FakeProcess()
    process.returncode = 7

    async def fake_open_unix_connection(path: str) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        del path
        raise OSError("not ready")

    monkeypatch.setattr(
        bootstrap_module.asyncio,
        "open_unix_connection",
        fake_open_unix_connection,
    )

    with pytest.raises(BootstrapError, match="exited before opening socket"):
        await bootstrap_module._connect_socket(Path("/tmp/runtime.sock"), process=process)


@pytest.mark.asyncio
async def test_connect_socket_times_out_when_runtime_never_appears(monkeypatch: pytest.MonkeyPatch) -> None:
    import peven.runtime.bootstrap as bootstrap_module

    async def fake_open_unix_connection(path: str) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        del path
        raise OSError("still not ready")

    monkeypatch.setattr(
        bootstrap_module.asyncio,
        "open_unix_connection",
        fake_open_unix_connection,
    )
    monkeypatch.setattr(bootstrap_module, "DEFAULT_CONNECT_TIMEOUT", 0.0)
    monkeypatch.setattr(bootstrap_module, "DEFAULT_CONNECT_POLL_INTERVAL", 0.0)

    with pytest.raises(BootstrapError, match="timed out connecting to runtime socket"):
        await bootstrap_module._connect_socket(Path("/tmp/runtime.sock"))


@pytest.mark.asyncio
async def test_read_handshake_rejects_malformed_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    import peven.runtime.bootstrap as bootstrap_module

    async def fake_read_frame(reader: asyncio.StreamReader) -> bytes:
        del reader
        return b"\xc1"

    monkeypatch.setattr(bootstrap_module, "read_frame", fake_read_frame)

    with pytest.raises(bootstrap_module.HandshakeError, match="malformed handshake payload"):
        await bootstrap_module._read_handshake(asyncio.StreamReader(), _FakeWriter())
