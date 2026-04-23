"""Shared runtime handle and per-run state ownership."""

from __future__ import annotations

import asyncio
import inspect
import shutil
import threading
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeAlias, TypeVar

from .bootstrap import (
    BootstrappedRuntime,
    bootstrap_runtime as _bootstrap_runtime,
    close_bootstrapped_runtime,
)
from .sinks import Sink


__all__ = [
    "SharedRuntime",
    "allocate_env_run_id",
    "allocate_req_id",
    "fail_run",
    "finish_run",
    "get_shared_runtime",
    "mark_runtime_crashed",
    "open_run",
    "push_run_event",
    "run_sync",
]

_RECENT_TERMINAL_RUNS_LIMIT = 1024
_T = TypeVar("_T")
BootstrapRuntimeFactory: TypeAlias = Callable[
    [tuple[str, ...]], Awaitable[BootstrappedRuntime]
]


@dataclass(slots=True)
class RunState:
    env_run_id: int
    finished: bool = False
    sink: Sink | None = None


@dataclass(slots=True)
class SharedRuntime:
    session: BootstrappedRuntime
    loop: asyncio.AbstractEventLoop
    command: tuple[str, ...] = field(default_factory=tuple)
    session_exchange_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active_run_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    next_req_id: int = 1
    next_env_run_id: int = 1
    runs: dict[int, RunState] = field(default_factory=dict)
    recent_terminal_run_ids: set[int] = field(default_factory=set)
    recent_terminal_run_order: deque[int] = field(default_factory=deque)
    crashed: bool = False
    crash_error: BaseException | None = None


_shared_runtime: SharedRuntime | None = None
_bootstrap_state_lock = threading.Lock()
_bootstrap_owner_loop: asyncio.AbstractEventLoop | None = None
_bootstrap_owner_ready: threading.Event | None = None
_runtime_loop: asyncio.AbstractEventLoop | None = None
_runtime_loop_thread: threading.Thread | None = None


async def get_shared_runtime(
    *,
    command: tuple[str, ...],
    bootstrap_runtime: BootstrapRuntimeFactory | None = None,
) -> SharedRuntime:
    """Return the cached shared runtime, bootstrapping one if needed."""
    global _shared_runtime
    current_loop = asyncio.get_running_loop()

    reused = _reuse_shared_runtime_if_live(current_loop, command)
    if reused is not None:
        return reused

    while True:
        should_bootstrap, ready = _claim_bootstrap_turn(current_loop)
        if not should_bootstrap:
            if ready is None:
                reused = _reuse_shared_runtime_if_live(current_loop, command)
                if reused is not None:
                    return reused
                continue
            await asyncio.to_thread(ready.wait)
            reused = _reuse_shared_runtime_if_live(current_loop, command)
            if reused is not None:
                return reused
            continue

        stale_runtime = _shared_runtime
        _shared_runtime = None
        try:
            if stale_runtime is not None:
                await _cleanup_stale_runtime(stale_runtime)
            bootstrap = (
                _default_bootstrap_runtime
                if bootstrap_runtime is None
                else bootstrap_runtime
            )
            session = await bootstrap(command)
            _shared_runtime = SharedRuntime(
                session=session,
                loop=current_loop,
                command=command,
            )
            return _shared_runtime
        finally:
            _release_bootstrap_turn()


def allocate_req_id(runtime: SharedRuntime) -> int:
    """Allocate the next odd Python-owned request id."""
    req_id = runtime.next_req_id
    runtime.next_req_id += 2
    return req_id


def allocate_env_run_id(runtime: SharedRuntime) -> int:
    """Allocate the next env-run id for one loaded run."""
    env_run_id = runtime.next_env_run_id
    runtime.next_env_run_id += 1
    return env_run_id


def open_run(
    runtime: SharedRuntime, env_run_id: int, *, sink: Sink | None = None
) -> RunState:
    """Open one new per-run state bucket."""
    _raise_if_runtime_crashed(runtime)
    if env_run_id in runtime.runs:
        raise ValueError(f"env_run_id {env_run_id} is already active")
    run_state = RunState(env_run_id=env_run_id, sink=sink)
    runtime.runs[env_run_id] = run_state
    return run_state


def push_run_event(runtime: SharedRuntime, env_run_id: int, event: object) -> bool:
    """Route one event to the active run sink unless the run already terminated."""
    run_state = runtime.runs.get(env_run_id)
    if run_state is None and env_run_id in runtime.recent_terminal_run_ids:
        return False
    if run_state is None:
        raise ValueError(f"unknown env_run_id {env_run_id}")
    if run_state.finished:
        return False
    if run_state.sink is not None:
        run_state.sink.write(event)
    return True


def finish_run(runtime: SharedRuntime, env_run_id: int) -> None:
    """Mark one run terminal and evict its active run-state entry."""
    run_state = runtime.runs.pop(env_run_id, None)
    if run_state is None:
        raise ValueError(f"unknown env_run_id {env_run_id}")
    run_state.finished = True
    _remember_terminal_run(runtime, env_run_id)


def fail_run(runtime: SharedRuntime, env_run_id: int) -> None:
    """Mark one run terminal with an error and evict its run-state entry."""
    run_state = runtime.runs.pop(env_run_id, None)
    if run_state is None:
        raise ValueError(f"unknown env_run_id {env_run_id}")
    run_state.finished = True
    _remember_terminal_run(runtime, env_run_id)


def mark_runtime_crashed(runtime: SharedRuntime, error: BaseException) -> None:
    """Mark the shared runtime crashed and evict every open run."""
    if runtime.crashed:
        return
    runtime.crashed = True
    runtime.crash_error = error
    runtime.runs.clear()


def _reset_shared_runtime_for_tests() -> None:
    """Clear module-global runtime state for isolated tests."""
    global _bootstrap_owner_loop, _bootstrap_owner_ready, _runtime_loop, _runtime_loop_thread, _shared_runtime
    runtime = _shared_runtime
    _shared_runtime = None
    with _bootstrap_state_lock:
        _bootstrap_owner_loop = None
        _bootstrap_owner_ready = None
    runtime_loop = _runtime_loop
    runtime_thread = _runtime_loop_thread
    _runtime_loop = None
    _runtime_loop_thread = None
    if runtime is not None:
        _close_shared_runtime_for_reset(runtime)
    if runtime_loop is not None and runtime_loop.is_running():
        runtime_loop.call_soon_threadsafe(runtime_loop.stop)
    if runtime_thread is not None:
        runtime_thread.join(timeout=1.0)


def _claim_bootstrap_turn(
    current_loop: asyncio.AbstractEventLoop,
) -> tuple[bool, threading.Event | None]:
    global _bootstrap_owner_loop, _bootstrap_owner_ready
    with _bootstrap_state_lock:
        if _shared_runtime is not None and not _shared_runtime.crashed:
            return False, None
        if _bootstrap_owner_loop is None:
            ready = threading.Event()
            _bootstrap_owner_loop = current_loop
            _bootstrap_owner_ready = ready
            return True, ready
        if _bootstrap_owner_loop is current_loop:
            return False, _bootstrap_owner_ready
        raise RuntimeError("shared runtime belongs to a different event loop")


def _release_bootstrap_turn() -> None:
    global _bootstrap_owner_loop, _bootstrap_owner_ready
    with _bootstrap_state_lock:
        ready = _bootstrap_owner_ready
        _bootstrap_owner_loop = None
        _bootstrap_owner_ready = None
    if ready is not None:
        ready.set()


def _reuse_shared_runtime_if_live(
    current_loop: asyncio.AbstractEventLoop,
    command: tuple[str, ...],
) -> SharedRuntime | None:
    runtime = _shared_runtime
    if runtime is None:
        return None
    _mark_dead_runtime_if_needed(runtime)
    if runtime.crashed:
        return None
    if runtime.loop is not current_loop:
        raise RuntimeError("shared runtime belongs to a different event loop")
    if runtime.command != command:
        raise RuntimeError("shared runtime belongs to a different runtime command")
    return runtime


def _mark_dead_runtime_if_needed(runtime: SharedRuntime) -> None:
    if runtime.session.process.returncode is None:
        return
    mark_runtime_crashed(
        runtime,
        RuntimeError(
            f"runtime process exited with return code {runtime.session.process.returncode}"
        ),
    )


def _raise_if_runtime_crashed(runtime: SharedRuntime) -> None:
    if not runtime.crashed:
        return
    message = "runtime crashed"
    if runtime.crash_error is not None and str(runtime.crash_error):
        message = str(runtime.crash_error)
    raise RuntimeError(message)


async def _cleanup_stale_runtime(runtime: SharedRuntime) -> None:
    try:
        await close_bootstrapped_runtime(runtime.session)
    except Exception:
        return


def run_sync(awaitable: Awaitable[_T]) -> _T:
    """Run one coroutine on the dedicated synchronous runtime loop and wait for its result."""
    loop = _ensure_runtime_loop()
    if _runtime_loop_thread is threading.current_thread():
        if inspect.iscoroutine(awaitable):
            awaitable.close()
        raise RuntimeError("run_sync cannot be called from the peven runtime loop thread")
    future = asyncio.run_coroutine_threadsafe(awaitable, loop)
    return future.result()


def _close_shared_runtime_for_reset(runtime: SharedRuntime) -> None:
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if runtime.loop.is_running():
        if running_loop is runtime.loop:
            _close_bootstrapped_runtime_sync_best_effort(runtime)
            return
        close_coro = close_bootstrapped_runtime(runtime.session)
        try:
            future = asyncio.run_coroutine_threadsafe(close_coro, runtime.loop)
        except Exception:
            close_coro.close()
            _close_bootstrapped_runtime_sync_best_effort(runtime)
            return
        try:
            future.result(timeout=2.0)
            return
        except Exception:
            close_coro.close()
            _close_bootstrapped_runtime_sync_best_effort(runtime)
            return

    close_coro = close_bootstrapped_runtime(runtime.session)
    try:
        asyncio.run(close_coro)
    except Exception:
        close_coro.close()
        _close_bootstrapped_runtime_sync_best_effort(runtime)


def _close_bootstrapped_runtime_sync_best_effort(runtime: SharedRuntime) -> None:
    try:
        runtime.session.writer.close()
    except Exception:
        pass
    try:
        if runtime.session.process.returncode is None:
            runtime.session.process.terminate()
    except Exception:
        pass
    shutil.rmtree(runtime.session.socket_dir, ignore_errors=True)


def _remember_terminal_run(runtime: SharedRuntime, env_run_id: int) -> None:
    if env_run_id in runtime.recent_terminal_run_ids:
        return
    runtime.recent_terminal_run_ids.add(env_run_id)
    runtime.recent_terminal_run_order.append(env_run_id)
    while len(runtime.recent_terminal_run_order) > _RECENT_TERMINAL_RUNS_LIMIT:
        expired = runtime.recent_terminal_run_order.popleft()
        runtime.recent_terminal_run_ids.discard(expired)


def _ensure_runtime_loop() -> asyncio.AbstractEventLoop:
    global _runtime_loop, _runtime_loop_thread
    if _runtime_loop is not None and _runtime_loop_thread is not None and _runtime_loop_thread.is_alive():
        return _runtime_loop

    loop_ready = threading.Event()

    def runner() -> None:
        global _runtime_loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _runtime_loop = loop
        loop_ready.set()
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

    _runtime_loop_thread = threading.Thread(
        target=runner,
        name="peven-runtime-loop",
        daemon=True,
    )
    _runtime_loop_thread.start()
    loop_ready.wait()
    assert _runtime_loop is not None
    return _runtime_loop


async def _default_bootstrap_runtime(command: tuple[str, ...]) -> BootstrappedRuntime:
    return await _bootstrap_runtime(command=command)
