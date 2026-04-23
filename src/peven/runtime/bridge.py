"""Internal runtime scaffolding that consumes handoff artifacts and shared state."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Sequence
from typing import Protocol, TypeVar

import msgspec

from ..handoff.callbacks import invoke_transition
from ..handoff.framing import read_frame, write_frame
from ..handoff.lowering import CompiledEnv, normalize_initial_marking
from ..handoff.messages import (
    CallbackError,
    CallbackReply,
    CallbackRequest,
    LoadEnvError,
    RunEnvError,
    bundle_ref_from_callback_bundle,
    decode_adapter_message,
    decode_load_env_reply,
    decode_run_env_reply,
    make_callback_error,
    make_callback_reply,
    make_load_env,
    make_run_env,
    normalize_runtime_event,
    validate_callback_request,
)
from ..shared.events import BundleRef, RunFinished, RunResult, RuntimeEvent
from ..shared.token import Marking, Token
from .bootstrap import BootstrappedRuntime, default_runtime_command
from .sinks import Sink
from .state import (
    BootstrapRuntimeFactory,
    SharedRuntime,
    allocate_env_run_id,
    allocate_req_id,
    fail_run,
    finish_run,
    get_shared_runtime,
    mark_runtime_crashed,
    open_run,
    push_run_event,
)
from .store import activate_store, clear_store, open_store, reset_store


__all__ = [
    "LoadEnvRejectedError",
    "RunEnvRejectedError",
    "run_env",
    "run_until_terminal_result",
]


class AdapterProtocolError(RuntimeError):
    """Raised when the adapter sends a malformed or unexpected protocol payload."""


class AdapterTransportError(AdapterProtocolError):
    """Raised when the shared session transport or socket state is no longer usable."""


class LoadEnvRejectedError(RuntimeError):
    """Raised when the adapter rejects an authored env load request."""


class RunEnvRejectedError(RuntimeError):
    """Raised when the adapter rejects a run-start request."""


class TransitionCallback(Protocol):
    def __call__(
        self,
        transition_id: str,
        bundle: BundleRef,
        tokens: Sequence[Token],
        *,
        attempt: int,
        inputs_by_place: dict[str, list[Token]] | None = None,
    ) -> Awaitable[dict[str, list[Token]]]: ...


class RuntimeRunner(Protocol):
    def __call__(
        self,
        *,
        runtime: SharedRuntime,
        compiled_env: CompiledEnv,
        env: object,
        env_run_id: int,
        initial_marking: dict[str, list[Token]],
        callback: TransitionCallback,
    ) -> Awaitable[object]: ...


_ReplyT = TypeVar("_ReplyT")


async def _exchange_session_frame(runtime: SharedRuntime, payload: bytes) -> bytes:
    """Write one framed payload to a bootstrapped session and read one framed reply."""
    session = runtime.session
    try:
        reader, writer = _require_session_streams(session)
    except RuntimeError as exc:
        raise AdapterTransportError(str(exc)) from exc
    async with runtime.session_exchange_lock:
        try:
            await write_frame(writer, payload)
        except (OSError, RuntimeError) as exc:
            raise AdapterTransportError("runtime session write failed") from exc
        try:
            return await read_frame(reader)
        except asyncio.IncompleteReadError as exc:
            raise AdapterTransportError("runtime session closed before the first reply") from exc
        except OSError as exc:
            raise AdapterTransportError("runtime session read failed") from exc


async def _load_compiled_env(runtime: SharedRuntime, compiled_env: CompiledEnv) -> None:
    """Send one authored env load request and validate the inline adapter reply."""
    req_id = allocate_req_id(runtime)
    request = make_load_env(req_id=req_id, env=compiled_env.authored_env)
    reply = await _exchange_control_reply(
        runtime,
        request=request,
        decode_reply=decode_load_env_reply,
        malformed_reply_message="adapter sent a malformed load reply",
    )
    if reply.req_id != req_id:
        raise AdapterProtocolError("adapter load reply did not match the request")
    if isinstance(reply, LoadEnvError):
        raise LoadEnvRejectedError(reply.error)


async def _start_run(
    runtime: SharedRuntime,
    *,
    env_run_id: int,
    initial_marking: dict[str, list[Token]],
    fuse: int | None = None,
) -> None:
    """Send one run-start request for an already-loaded authored env."""
    req_id = allocate_req_id(runtime)
    request = make_run_env(
        req_id=req_id,
        env_run_id=env_run_id,
        initial_marking=initial_marking,
        fuse=fuse,
    )
    reply = await _exchange_control_reply(
        runtime,
        request=request,
        decode_reply=decode_run_env_reply,
        malformed_reply_message="adapter sent a malformed run reply",
    )
    if reply.req_id != req_id or reply.env_run_id != env_run_id:
        raise AdapterProtocolError("adapter run reply did not match the request")
    if isinstance(reply, RunEnvError):
        raise RunEnvRejectedError(reply.error)


async def _exchange_control_reply(
    runtime: SharedRuntime,
    *,
    request: object,
    decode_reply: callable,
    malformed_reply_message: str,
) -> _ReplyT:
    payload = msgspec.msgpack.encode(request)
    try:
        reply_payload = await _exchange_session_frame(runtime, payload)
    except ValueError as exc:
        raise AdapterProtocolError("adapter sent a malformed frame") from exc
    try:
        return decode_reply(reply_payload)
    except (TypeError, ValueError) as exc:
        raise AdapterProtocolError(malformed_reply_message) from exc


async def _reply_to_callback_request(
    *, callback: TransitionCallback, active_env_run_id: int, request: CallbackRequest
) -> tuple[bytes, CallbackReply | CallbackError]:
    if request.env_run_id != active_env_run_id:
        raise AdapterProtocolError("callback request env_run_id did not match the active run")
    try:
        outputs = await callback(
            transition_id=request.transition_id,
            bundle=bundle_ref_from_callback_bundle(request.bundle),
            tokens=request.tokens,
            attempt=request.attempt,
            inputs_by_place=request.inputs_by_place,
        )
    except Exception as exc:
        error = make_callback_error(
            req_id=request.req_id,
            env_run_id=request.env_run_id,
            error=_describe_callback_error(exc),
        )
        return msgspec.msgpack.encode(error), error
    reply = make_callback_reply(
        req_id=request.req_id,
        env_run_id=request.env_run_id,
        outputs=outputs,
    )
    return msgspec.msgpack.encode(reply), reply


def _buffer_runtime_event(
    *, runtime: SharedRuntime, env_run_id: int, event: RuntimeEvent
) -> bool:
    try:
        return push_run_event(runtime, env_run_id, event)
    except ValueError as exc:
        raise AdapterProtocolError("adapter event referenced unknown env_run_id") from exc


async def run_until_terminal_result(
    *,
    runtime: SharedRuntime,
    compiled_env: CompiledEnv,
    env: object,
    env_run_id: int,
    initial_marking: dict[str, list[Token]],
    callback: TransitionCallback,
) -> RunResult:
    del compiled_env, env, initial_marking
    try:
        reader, writer = _require_session_streams(runtime.session)
    except RuntimeError as exc:
        raise AdapterTransportError(str(exc)) from exc

    async def _process_callback(request: CallbackRequest) -> None:
        reply_payload, _ = await _reply_to_callback_request(
            callback=callback,
            active_env_run_id=env_run_id,
            request=request,
        )
        async with runtime.session_exchange_lock:
            try:
                await write_frame(writer, reply_payload)
            except (OSError, RuntimeError) as exc:
                raise AdapterTransportError("runtime session write failed") from exc

    pending_callbacks: set[asyncio.Task[None]] = set()
    read_task: asyncio.Task[bytes] | None = asyncio.create_task(read_frame(reader))
    terminal_result: RunResult | None = None

    try:
        while True:
            wait_set: set[asyncio.Task[object]] = set(pending_callbacks)
            if read_task is not None:
                wait_set.add(read_task)
            done, _ = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)

            completed_read_task = read_task
            if completed_read_task is not None and completed_read_task in done:
                payload = _read_adapter_payload(completed_read_task)
                read_task = None
            else:
                payload = None

            completed_callbacks = done.difference(
                {completed_read_task} if completed_read_task is not None else set()
            )
            for task in completed_callbacks:
                pending_callbacks.discard(task)
                task.result()

            if payload is None:
                continue
            try:
                message = decode_adapter_message(payload)
            except (TypeError, ValueError) as exc:
                raise AdapterProtocolError("adapter sent a malformed protocol payload") from exc
            if isinstance(message, CallbackRequest):
                try:
                    validate_callback_request(message)
                except (TypeError, ValueError) as exc:
                    raise AdapterProtocolError(
                        "adapter sent a malformed callback request"
                    ) from exc
                if message.env_run_id != env_run_id:
                    raise AdapterProtocolError(
                        "callback request env_run_id did not match the active run"
                    )
                pending_callbacks.add(asyncio.create_task(_process_callback(message)))
                read_task = asyncio.create_task(read_frame(reader))
                continue
            try:
                message_env_run_id, event = normalize_runtime_event(message)
            except (TypeError, ValueError) as exc:
                raise AdapterProtocolError("adapter sent a malformed runtime event") from exc
            accepted = _buffer_runtime_event(
                runtime=runtime, env_run_id=message_env_run_id, event=event
            )
            if (
                isinstance(event, RunFinished)
                and accepted
                and message_env_run_id == env_run_id
            ):
                terminal_result = event.result
                break
            read_task = asyncio.create_task(read_frame(reader))
        if read_task is not None:
            read_task.cancel()
            try:
                await read_task
            except asyncio.CancelledError:
                pass
        if pending_callbacks:
            done, _ = await asyncio.wait(pending_callbacks)
            for task in done:
                task.result()
        assert terminal_result is not None
        return terminal_result
    finally:
        if read_task is not None:
            read_task.cancel()
        for task in pending_callbacks:
            task.cancel()
        if read_task is not None:
            try:
                await read_task
            except BaseException:
                pass
        if pending_callbacks:
            await asyncio.gather(*pending_callbacks, return_exceptions=True)


async def run_env(
    env: object,
    *,
    command: tuple[str, ...] | None = None,
    bootstrap_runtime: BootstrapRuntimeFactory | None = None,
    runtime_runner: RuntimeRunner | None = None,
    marking: Marking | None = None,
    fuse: int | None = None,
    sink: Sink | None = None,
    **initial_marking_kwargs: object,
) -> object:
    """Compile one env, normalize its marking, and hand it to an injected runner."""
    compiled_env = type(env).compiled()
    if runtime_runner is None:
        runtime_runner = run_until_terminal_result
    if command is None:
        command = default_runtime_command()
    if marking is None:
        marking = env.initial_marking(**initial_marking_kwargs)
    if not isinstance(marking, Marking):
        raise TypeError("initial_marking() must return a peven.Marking")
    normalized_marking = normalize_initial_marking(marking)

    runtime = await get_shared_runtime(
        command=command,
        bootstrap_runtime=bootstrap_runtime,
    )
    async with runtime.active_run_lock:
        env_run_id = allocate_env_run_id(runtime)
        open_run(runtime, env_run_id, sink=sink)
        store = open_store(env_run_id)
        store_token = activate_store(store)
        caught: BaseException | None = None

        async def callback(
            transition_id: str,
            bundle: BundleRef,
            tokens: Sequence[Token],
            *,
            attempt: int,
            inputs_by_place: dict[str, list[Token]] | None = None,
        ) -> dict[str, list[Token]]:
            return await invoke_transition(
                compiled_env,
                transition_id=transition_id,
                env=env,
                bundle=bundle,
                tokens=tokens,
                attempt=attempt,
                inputs_by_place=inputs_by_place,
                sink=sink,
            )

        try:
            await _load_compiled_env(runtime, compiled_env)
            await _start_run(
                runtime,
                env_run_id=env_run_id,
                initial_marking=normalized_marking,
                fuse=fuse,
            )
            result = await runtime_runner(
                runtime=runtime,
                compiled_env=compiled_env,
                env=env,
                env_run_id=env_run_id,
                initial_marking=normalized_marking,
                callback=callback,
            )
        except BaseException as exc:
            caught = exc
            if isinstance(exc, AdapterProtocolError):
                mark_runtime_crashed(runtime, exc)
            else:
                fail_run(runtime, env_run_id)
            raise
        else:
            finish_run(runtime, env_run_id)
            return result
        finally:
            if sink is not None:
                sink.close(caught)
            clear_store(store)
            reset_store(store_token)


def _require_session_streams(
    session: BootstrappedRuntime,
) -> tuple[asyncio.StreamReader, object]:
    reader = session.reader
    writer = session.writer
    if reader is None or writer is None:
        raise RuntimeError("runtime session is missing its reader/writer")
    if _writer_is_closed(writer):
        raise RuntimeError("runtime session writer is already closed")
    return reader, writer


def _writer_is_closed(writer: object) -> bool:
    is_closing = getattr(writer, "is_closing", None)
    if callable(is_closing):
        try:
            return bool(is_closing())
        except Exception:
            return False
    return bool(getattr(writer, "closed", False))


def _read_adapter_payload(task: asyncio.Task[bytes]) -> bytes:
    try:
        return task.result()
    except asyncio.IncompleteReadError as exc:
        raise AdapterTransportError(
            "runtime session closed while waiting for adapter message"
        ) from exc
    except OSError as exc:
        raise AdapterTransportError("runtime session read failed") from exc
    except ValueError as exc:
        raise AdapterProtocolError("adapter sent a malformed frame") from exc


def _describe_callback_error(error: BaseException) -> str:
    message = str(error)
    if message:
        return message
    return error.__class__.__name__
