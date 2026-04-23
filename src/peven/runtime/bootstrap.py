"""Bootstrap helpers for the shared Julia runtime process and socket session."""

from __future__ import annotations

import asyncio
import importlib
import shutil
import tempfile
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import msgspec

from ..handoff.framing import read_frame


PROTOCOL_VERSION = "0.1.0"
PEVEN_VERSION = "0.2.0"
HANDSHAKE_TAG = "peven-runtime-handshake"
DEFAULT_ADAPTER_ENTRYPOINT = "using PevenPy; PevenPy.main(ARGS[1])"
DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_HANDSHAKE_TIMEOUT = 5.0
DEFAULT_CONNECT_POLL_INTERVAL = 0.01
DEFAULT_TERMINATE_TIMEOUT = 1.0
DEFAULT_KILL_TIMEOUT = 1.0

__all__ = [
    "DEFAULT_CONNECT_POLL_INTERVAL",
    "DEFAULT_CONNECT_TIMEOUT",
    "DEFAULT_HANDSHAKE_TIMEOUT",
    "DEFAULT_KILL_TIMEOUT",
    "DEFAULT_TERMINATE_TIMEOUT",
    "HANDSHAKE_TAG",
    "PEVEN_VERSION",
    "PROTOCOL_VERSION",
    "BootstrapError",
    "BootstrappedRuntime",
    "Handshake",
    "HandshakeError",
    "HandshakeTagError",
    "InstalledRuntime",
    "PevenVersionMismatchError",
    "ProtocolVersionMismatchError",
    "bootstrap_runtime",
    "close_bootstrapped_runtime",
    "default_runtime_command",
    "ensure_runtime_installed",
    "prepare_socket_location",
    "validate_handshake",
]


class BootstrapError(RuntimeError):
    """Raised when the runtime cannot be launched or connected cleanly."""


class HandshakeError(BootstrapError):
    """Raised when the runtime handshake is malformed or incompatible."""


class HandshakeTagError(HandshakeError):
    """Raised when the runtime announces an unexpected handshake tag."""


class ProtocolVersionMismatchError(HandshakeError):
    """Raised when the runtime protocol version does not match Python."""


class PevenVersionMismatchError(HandshakeError):
    """Raised when the runtime peven version does not match Python."""


class _RuntimeProcess(Protocol):
    returncode: int | None

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    async def wait(self) -> int: ...


class _RuntimeWriter(Protocol):
    def close(self) -> None: ...

    async def wait_closed(self) -> None: ...


class Handshake(msgspec.Struct, frozen=True):
    tag: str
    protocol_version: str
    peven_version: str


@dataclass(slots=True)
class BootstrappedRuntime:
    process: _RuntimeProcess
    reader: asyncio.StreamReader
    writer: _RuntimeWriter
    socket_dir: Path
    socket_path: Path
    handshake: Handshake


@dataclass(slots=True, frozen=True)
class InstalledRuntime:
    julia_executable: str
    julia_project: Path


class SpawnProcess(Protocol):
    def __call__(
        self, command: tuple[str, ...], *, socket_path: Path
    ) -> Awaitable[_RuntimeProcess]: ...


class ConnectSocket(Protocol):
    def __call__(
        self, socket_path: Path
    ) -> Awaitable[tuple[asyncio.StreamReader, _RuntimeWriter]]: ...


class ReadHandshake(Protocol):
    def __call__(
        self, reader: asyncio.StreamReader, writer: _RuntimeWriter
    ) -> Awaitable[Handshake]: ...


def prepare_socket_location(root: Path | None = None) -> tuple[Path, Path]:
    """Create one private runtime socket directory and return its socket path."""
    socket_dir = Path(tempfile.mkdtemp(prefix="peven-runtime-", dir=root))
    return socket_dir, socket_dir / "runtime.sock"


def ensure_runtime_installed() -> InstalledRuntime:
    """Resolve the packaged Julia runtime environment and return its command inputs."""
    try:
        juliapkg = importlib.import_module("juliapkg")
    except ImportError as exc:  # pragma: no cover - project dependency
        raise BootstrapError("juliapkg is required to provision the default runtime") from exc
    juliapkg.resolve()
    return InstalledRuntime(
        julia_executable=str(juliapkg.executable()),
        julia_project=Path(juliapkg.project()),
    )


def default_runtime_command() -> tuple[str, ...]:
    """Build the default Julia adapter command from the resolved juliapkg environment."""
    installed = ensure_runtime_installed()
    return (
        installed.julia_executable,
        f"--project={installed.julia_project}",
        "-e",
        DEFAULT_ADAPTER_ENTRYPOINT,
        "{socket_path}",
    )


def validate_handshake(handshake: Handshake) -> Handshake:
    """Validate one runtime handshake against the local protocol contract."""
    if handshake.tag != HANDSHAKE_TAG:
        raise HandshakeTagError(f"unexpected handshake tag {handshake.tag!r}")
    if handshake.protocol_version != PROTOCOL_VERSION:
        raise ProtocolVersionMismatchError(
            f"protocol version mismatch: expected {PROTOCOL_VERSION}, got {handshake.protocol_version}"
        )
    if handshake.peven_version != PEVEN_VERSION:
        raise PevenVersionMismatchError(
            f"peven version mismatch: expected {PEVEN_VERSION}, got {handshake.peven_version}"
        )
    return handshake


async def bootstrap_runtime(
    *,
    command: Sequence[str],
    socket_root: Path | None = None,
    spawn_process: SpawnProcess | None = None,
    connect_socket: ConnectSocket | None = None,
    read_handshake: ReadHandshake | None = None,
) -> BootstrappedRuntime:
    """Launch the Julia runtime, connect its socket, and validate the handshake."""
    socket_dir, socket_path = prepare_socket_location(socket_root)
    rendered_command = _render_command(command, socket_path)
    spawn = _spawn_process if spawn_process is None else spawn_process
    connect = _connect_socket if connect_socket is None else connect_socket
    read_runtime_handshake = _read_handshake if read_handshake is None else read_handshake

    process: _RuntimeProcess | None = None
    writer: _RuntimeWriter | None = None
    try:
        process = await spawn(rendered_command, socket_path=socket_path)
        if connect_socket is None:
            reader, writer = await _connect_socket(socket_path, process=process)
        else:
            reader, writer = await connect(socket_path)
        try:
            handshake = validate_handshake(
                await asyncio.wait_for(
                    read_runtime_handshake(reader, writer),
                    timeout=DEFAULT_HANDSHAKE_TIMEOUT,
                )
            )
        except TimeoutError as exc:
            raise BootstrapError("timed out waiting for runtime handshake") from exc
        return BootstrappedRuntime(
            process=process,
            reader=reader,
            writer=writer,
            socket_dir=socket_dir,
            socket_path=socket_path,
            handshake=handshake,
        )
    except BaseException:
        await asyncio.shield(
            _cleanup_failed_bootstrap(process=process, writer=writer, socket_dir=socket_dir)
        )
        raise


def _render_command(command: Sequence[str], socket_path: Path) -> tuple[str, ...]:
    socket_path_text = str(socket_path)
    return tuple(part.replace("{socket_path}", socket_path_text) for part in command)


async def _spawn_process(
    command: tuple[str, ...], *, socket_path: Path
) -> asyncio.subprocess.Process:
    del socket_path
    return await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )


async def _connect_socket(
    socket_path: Path,
    *,
    process: _RuntimeProcess | None = None,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    deadline = asyncio.get_running_loop().time() + DEFAULT_CONNECT_TIMEOUT
    while True:
        if process is not None and process.returncode is not None:
            raise BootstrapError(
                f"runtime process exited before opening socket {socket_path} "
                f"(return code {process.returncode})"
            )
        try:
            return await asyncio.open_unix_connection(str(socket_path))
        except OSError as exc:
            if asyncio.get_running_loop().time() >= deadline:
                raise BootstrapError(f"timed out connecting to runtime socket {socket_path}") from exc
            await asyncio.sleep(DEFAULT_CONNECT_POLL_INTERVAL)


async def _read_handshake(reader: asyncio.StreamReader, writer: _RuntimeWriter) -> Handshake:
    del writer
    payload = await read_frame(reader)
    try:
        return msgspec.msgpack.decode(payload, type=Handshake)
    except msgspec.DecodeError as exc:
        raise HandshakeError("runtime sent a malformed handshake payload") from exc


async def _cleanup_failed_bootstrap(
    *,
    process: _RuntimeProcess | None,
    writer: _RuntimeWriter | None,
    socket_dir: Path,
) -> None:
    if writer is not None:
        await _close_writer_best_effort(writer)
    if process is not None:
        await _wait_for_process_shutdown(process)
    shutil.rmtree(socket_dir, ignore_errors=True)


async def close_bootstrapped_runtime(session: BootstrappedRuntime) -> None:
    writer = session.writer
    if writer is not None:
        await _close_writer_best_effort(writer)
    await _wait_for_process_shutdown(session.process)
    shutil.rmtree(session.socket_dir, ignore_errors=True)


async def _close_writer_best_effort(writer: _RuntimeWriter) -> None:
    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        return


async def _wait_for_process_shutdown(process: _RuntimeProcess) -> None:
    if process.returncode is None:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=DEFAULT_TERMINATE_TIMEOUT)
            return
        except TimeoutError:
            process.kill()
            try:
                await asyncio.wait_for(process.wait(), timeout=DEFAULT_KILL_TIMEOUT)
            except TimeoutError:
                return
    else:
        try:
            await asyncio.wait_for(process.wait(), timeout=DEFAULT_TERMINATE_TIMEOUT)
        except TimeoutError:
            pass
