from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from peven.runtime.bootstrap import HANDSHAKE_TAG, BootstrappedRuntime, Handshake
from peven.runtime.state import (
    SharedRuntime,
    _reset_shared_runtime_for_tests,
    allocate_env_run_id,
    allocate_req_id,
    finish_run,
    get_shared_runtime,
    mark_runtime_crashed,
    open_run,
    push_run_event,
    run_sync,
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


class _FakeWriter:
    def __init__(self) -> None:
        self.closed = False
        self.wait_closed_called = False

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        self.wait_closed_called = True


@pytest.fixture(autouse=True)
def _reset_shared_runtime() -> None:
    _reset_shared_runtime_for_tests()


def _bootstrapped_runtime(
    *,
    process: _FakeProcess | None = None,
    writer: _FakeWriter | None = None,
    socket_dir: Path | None = None,
) -> BootstrappedRuntime:
    if socket_dir is None:
        socket_dir = Path("/tmp/peven-runtime")
    return BootstrappedRuntime(
        process=_FakeProcess() if process is None else process,
        reader=asyncio.StreamReader(),
        writer=_FakeWriter() if writer is None else writer,
        socket_dir=socket_dir,
        socket_path=socket_dir / "runtime.sock",
        handshake=Handshake(tag=HANDSHAKE_TAG, protocol_version="0.1.0", peven_version="0.2.0"),
    )


@pytest.mark.asyncio
async def test_get_shared_runtime_bootstraps_once_and_reuses_the_handle() -> None:
    calls = 0

    async def bootstrap(command: tuple[str, ...]) -> BootstrappedRuntime:
        nonlocal calls
        calls += 1
        assert command == ("fake-runtime",)
        return _bootstrapped_runtime()

    runtime_1 = await get_shared_runtime(
        command=("fake-runtime",), bootstrap_runtime=bootstrap
    )
    runtime_2 = await get_shared_runtime(
        command=("fake-runtime",), bootstrap_runtime=bootstrap
    )

    assert isinstance(runtime_1, SharedRuntime)
    assert runtime_1 is runtime_2
    assert calls == 1


@pytest.mark.asyncio
async def test_get_shared_runtime_replaces_a_crashed_handle() -> None:
    calls = 0

    async def bootstrap(command: tuple[str, ...]) -> BootstrappedRuntime:
        nonlocal calls
        calls += 1
        return _bootstrapped_runtime()

    runtime_1 = await get_shared_runtime(
        command=("fake-runtime",), bootstrap_runtime=bootstrap
    )
    mark_runtime_crashed(runtime_1, RuntimeError("boom"))

    runtime_2 = await get_shared_runtime(
        command=("fake-runtime",), bootstrap_runtime=bootstrap
    )

    assert runtime_2 is not runtime_1
    assert calls == 2


@pytest.mark.asyncio
async def test_get_shared_runtime_coalesces_same_loop_bootstrap_waiters() -> None:
    calls = 0
    started = asyncio.Event()

    async def bootstrap(command: tuple[str, ...]) -> BootstrappedRuntime:
        nonlocal calls
        assert command == ("fake-runtime",)
        calls += 1
        started.set()
        await asyncio.sleep(0.05)
        return _bootstrapped_runtime()

    first = asyncio.create_task(
        get_shared_runtime(command=("fake-runtime",), bootstrap_runtime=bootstrap)
    )
    await started.wait()
    second = asyncio.create_task(
        get_shared_runtime(command=("fake-runtime",), bootstrap_runtime=bootstrap)
    )

    runtime_1, runtime_2 = await asyncio.gather(first, second)

    assert runtime_1 is runtime_2
    assert calls == 1


@pytest.mark.asyncio
async def test_get_shared_runtime_reaps_the_previous_crashed_session_before_replacing_it(
    tmp_path: Path,
) -> None:
    calls = 0
    old_process = _FakeProcess()
    old_writer = _FakeWriter()
    old_socket_dir = tmp_path / "old-runtime"
    old_socket_dir.mkdir()

    async def bootstrap(command: tuple[str, ...]) -> BootstrappedRuntime:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _bootstrapped_runtime(
                process=old_process,
                writer=old_writer,
                socket_dir=old_socket_dir,
            )
        return _bootstrapped_runtime(socket_dir=tmp_path / "new-runtime")

    runtime_1 = await get_shared_runtime(
        command=("fake-runtime",), bootstrap_runtime=bootstrap
    )
    mark_runtime_crashed(runtime_1, RuntimeError("boom"))

    runtime_2 = await get_shared_runtime(
        command=("fake-runtime",), bootstrap_runtime=bootstrap
    )

    assert runtime_2 is not runtime_1
    assert old_writer.closed is True
    assert old_writer.wait_closed_called is True
    assert old_process.terminated is True
    assert old_process.waited is True
    assert old_socket_dir.exists() is False


@pytest.mark.asyncio
async def test_get_shared_runtime_rejects_a_different_command_for_a_live_handle() -> None:
    calls = 0

    async def bootstrap(command: tuple[str, ...]) -> BootstrappedRuntime:
        nonlocal calls
        calls += 1
        return _bootstrapped_runtime()

    runtime = await get_shared_runtime(
        command=("fake-runtime-a",), bootstrap_runtime=bootstrap
    )

    with pytest.raises(RuntimeError, match="different runtime command"):
        await get_shared_runtime(
            command=("fake-runtime-b",), bootstrap_runtime=bootstrap
        )

    assert runtime.command == ("fake-runtime-a",)
    assert calls == 1


@pytest.mark.asyncio
async def test_get_shared_runtime_rebootstraps_when_the_cached_process_has_exited() -> None:
    calls = 0

    async def bootstrap(command: tuple[str, ...]) -> BootstrappedRuntime:
        nonlocal calls
        calls += 1
        return _bootstrapped_runtime()

    runtime_1 = await get_shared_runtime(
        command=("fake-runtime",), bootstrap_runtime=bootstrap
    )
    runtime_1.session.process.returncode = 1

    runtime_2 = await get_shared_runtime(
        command=("fake-runtime",), bootstrap_runtime=bootstrap
    )

    assert runtime_2 is not runtime_1
    assert runtime_1.crashed is True
    assert calls == 2


@pytest.mark.asyncio
async def test_runtime_allocates_odd_req_ids_and_monotonic_env_run_ids() -> None:
    async def bootstrap(command: tuple[str, ...]) -> BootstrappedRuntime:
        return _bootstrapped_runtime()

    runtime = await get_shared_runtime(
        command=("fake-runtime",), bootstrap_runtime=bootstrap
    )

    assert allocate_req_id(runtime) == 1
    assert allocate_req_id(runtime) == 3
    assert allocate_req_id(runtime) == 5
    assert allocate_env_run_id(runtime) == 1
    assert allocate_env_run_id(runtime) == 2


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_run_state_accepts_events_until_finish_and_drops_late_events() -> None:
    async def bootstrap(command: tuple[str, ...]) -> BootstrappedRuntime:
        return _bootstrapped_runtime()

    runtime = await get_shared_runtime(
        command=("fake-runtime",), bootstrap_runtime=bootstrap
    )
    env_run_id = allocate_env_run_id(runtime)
    run_state = open_run(runtime, env_run_id)

    assert push_run_event(runtime, env_run_id, "started") is True
    assert push_run_event(runtime, env_run_id, "completed") is True
    finish_run(runtime, env_run_id)

    assert env_run_id not in runtime.runs
    assert push_run_event(runtime, env_run_id, "late") is False
    assert not hasattr(run_state, "events")
    assert not hasattr(run_state, "completion")


@pytest.mark.asyncio
async def test_mark_runtime_crashed_evicts_open_runs() -> None:
    async def bootstrap(command: tuple[str, ...]) -> BootstrappedRuntime:
        return _bootstrapped_runtime()

    runtime = await get_shared_runtime(
        command=("fake-runtime",), bootstrap_runtime=bootstrap
    )
    env_run_id = allocate_env_run_id(runtime)
    open_run(runtime, env_run_id)

    error = RuntimeError("runtime crashed")
    mark_runtime_crashed(runtime, error)

    assert runtime.crashed is True
    assert runtime.runs == {}


@pytest.mark.asyncio
async def test_mark_runtime_crashed_rejects_new_waiters_after_fanout() -> None:
    async def bootstrap(command: tuple[str, ...]) -> BootstrappedRuntime:
        return _bootstrapped_runtime()

    runtime = await get_shared_runtime(
        command=("fake-runtime",), bootstrap_runtime=bootstrap
    )
    mark_runtime_crashed(runtime, RuntimeError("runtime crashed"))

    with pytest.raises(RuntimeError, match="runtime crashed"):
        open_run(runtime, 1)


@pytest.mark.asyncio
async def test_mark_runtime_crashed_is_idempotent() -> None:
    async def bootstrap(command: tuple[str, ...]) -> BootstrappedRuntime:
        del command
        return _bootstrapped_runtime()

    runtime = await get_shared_runtime(
        command=("fake-runtime",), bootstrap_runtime=bootstrap
    )

    first = RuntimeError("first")
    second = RuntimeError("second")
    mark_runtime_crashed(runtime, first)
    mark_runtime_crashed(runtime, second)

    assert runtime.crashed is True
    assert runtime.crash_error is first


@pytest.mark.asyncio
async def test_get_shared_runtime_rejects_reuse_from_a_different_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import peven.runtime.state as runtime_module

    async def bootstrap(command: tuple[str, ...]) -> BootstrappedRuntime:
        del command
        return _bootstrapped_runtime()

    other_loop = asyncio.new_event_loop()
    try:
        monkeypatch.setattr(
            runtime_module,
            "_shared_runtime",
            SharedRuntime(session=_bootstrapped_runtime(), loop=other_loop),
        )
        with pytest.raises(RuntimeError, match="different event loop"):
            await get_shared_runtime(
                command=("fake-runtime",), bootstrap_runtime=bootstrap
            )
    finally:
        other_loop.close()


@pytest.mark.asyncio
async def test_run_state_operations_reject_unknown_or_duplicate_run_ids() -> None:
    from peven.runtime.state import fail_run

    async def bootstrap(command: tuple[str, ...]) -> BootstrappedRuntime:
        return _bootstrapped_runtime()

    runtime = await get_shared_runtime(
        command=("fake-runtime",), bootstrap_runtime=bootstrap
    )
    env_run_id = allocate_env_run_id(runtime)
    open_run(runtime, env_run_id)

    with pytest.raises(ValueError, match="already active"):
        open_run(runtime, env_run_id)

    with pytest.raises(ValueError, match="unknown env_run_id 999"):
        push_run_event(runtime, 999, "missing")

    with pytest.raises(ValueError, match="unknown env_run_id 999"):
        finish_run(runtime, 999)

    runtime.runs.pop(env_run_id)
    with pytest.raises(ValueError, match="unknown env_run_id 999"):
        fail_run(runtime, 999)


def test_run_sync_uses_and_reuses_the_background_runtime_loop() -> None:
    _reset_shared_runtime_for_tests()
    try:
        loop_id_1 = run_sync(_loop_id())
        loop_id_2 = run_sync(_loop_id())
        assert loop_id_1 == loop_id_2
    finally:
        _reset_shared_runtime_for_tests()


def test_run_sync_rejects_calls_from_the_runtime_loop_thread() -> None:
    _reset_shared_runtime_for_tests()
    try:
        async def nested() -> None:
            with pytest.raises(RuntimeError, match="runtime loop thread"):
                run_sync(asyncio.sleep(0))

        run_sync(nested())
    finally:
        _reset_shared_runtime_for_tests()


async def _loop_id() -> int:
    return id(asyncio.get_running_loop())


async def _running_loop() -> asyncio.AbstractEventLoop:
    return asyncio.get_running_loop()


def test_first_use_from_different_loops_does_not_bootstrap_two_runtimes() -> None:
    results: dict[str, SharedRuntime] = {}
    errors: dict[str, BaseException] = {}
    calls = 0
    barrier = threading.Barrier(2)
    lock = threading.Lock()

    async def bootstrap(command: tuple[str, ...]) -> BootstrappedRuntime:
        nonlocal calls
        del command
        with lock:
            calls += 1
        await asyncio.sleep(0.05)
        return _bootstrapped_runtime()

    def run_in_thread(name: str) -> None:
        async def runner() -> None:
            barrier.wait()
            try:
                results[name] = await get_shared_runtime(
                    command=("fake-runtime",),
                    bootstrap_runtime=bootstrap,
                )
            except BaseException as exc:
                errors[name] = exc

        asyncio.run(runner())

    thread_1 = threading.Thread(target=run_in_thread, args=("one",))
    thread_2 = threading.Thread(target=run_in_thread, args=("two",))
    thread_1.start()
    thread_2.start()
    thread_1.join()
    thread_2.join()

    assert calls == 1
    assert len(results) == 1
    assert len(errors) == 1
    error = next(iter(errors.values()))
    assert isinstance(error, RuntimeError)
    assert "different event loop" in str(error)


@pytest.mark.asyncio
async def test_cleanup_stale_runtime_ignores_close_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import peven.runtime.state as runtime_module

    calls = 0

    async def bootstrap(command: tuple[str, ...]) -> BootstrappedRuntime:
        nonlocal calls
        del command
        calls += 1
        return _bootstrapped_runtime()

    async def broken_close(runtime: BootstrappedRuntime) -> None:
        del runtime
        raise RuntimeError("close failed")

    runtime_1 = await get_shared_runtime(
        command=("fake-runtime",), bootstrap_runtime=bootstrap
    )
    mark_runtime_crashed(runtime_1, RuntimeError("boom"))
    monkeypatch.setattr(runtime_module, "close_bootstrapped_runtime", broken_close)

    runtime_2 = await get_shared_runtime(
        command=("fake-runtime",), bootstrap_runtime=bootstrap
    )

    assert runtime_2 is not runtime_1
    assert calls == 2


@pytest.mark.asyncio
async def test_recent_terminal_run_ids_evict_old_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import peven.runtime.state as runtime_module

    async def bootstrap(command: tuple[str, ...]) -> BootstrappedRuntime:
        del command
        return _bootstrapped_runtime()

    monkeypatch.setattr(runtime_module, "_RECENT_TERMINAL_RUNS_LIMIT", 2)
    runtime = await get_shared_runtime(
        command=("fake-runtime",), bootstrap_runtime=bootstrap
    )

    run_ids: list[int] = []
    for _name in ("one", "two", "three"):
        env_run_id = allocate_env_run_id(runtime)
        open_run(runtime, env_run_id)
        finish_run(runtime, env_run_id)
        run_ids.append(env_run_id)

    assert run_ids[0] not in runtime.recent_terminal_run_ids
    assert run_ids[1] in runtime.recent_terminal_run_ids
    assert run_ids[2] in runtime.recent_terminal_run_ids


def test_reset_shared_runtime_closes_the_bootstrapped_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import peven.runtime.state as runtime_module

    runtime_loop = run_sync(_running_loop())
    process = _FakeProcess()
    writer = _FakeWriter()
    socket_dir = tmp_path / "runtime"
    socket_dir.mkdir()
    setup_loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(setup_loop)
        runtime = SharedRuntime(
            session=_bootstrapped_runtime(
                process=process,
                writer=writer,
                socket_dir=socket_dir,
            ),
            loop=runtime_loop,
        )
    finally:
        asyncio.set_event_loop(None)
        setup_loop.close()
    monkeypatch.setattr(runtime_module, "_shared_runtime", runtime)

    _reset_shared_runtime_for_tests()

    assert writer.closed is True
    assert writer.wait_closed_called is True
    assert process.terminated is True
    assert process.waited is True
    assert socket_dir.exists() is False
