from __future__ import annotations

from types import SimpleNamespace

import pytest

import peven
from peven.handoff.callbacks import invoke_transition
from peven.handoff.lowering import compile_env
from peven.runtime.store import activate_store, clear_store, open_store, reset_store
from peven.shared.token import Token


def _register_executor(source: str) -> dict[str, object]:
    namespace = {
        "__name__": "tests.runtime.generated_dispatch_executors",
        "peven": peven,
    }
    exec(source, namespace)
    return namespace


@pytest.mark.asyncio
async def test_invoke_transition_uses_the_existing_run_store() -> None:
    _register_executor(
        """
@peven.executor("runtime_store_writer")
async def runtime_store_writer(ctx, ready):
    ref = peven.store.put({"kind": "stored", "payload": ready.payload})
    stored = peven.store.get(ref)
    peven.store.release(ref)
    return peven.token(stored, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("runtime_store_env")
    class RuntimeStoreEnv(peven.Env):
        ready = peven.place()
        done = peven.place()

        finish = peven.transition(
            inputs=["ready"],
            outputs=["done"],
            executor="runtime_store_writer",
        )

    env = RuntimeStoreEnv()
    compiled = compile_env(RuntimeStoreEnv.spec())
    bundle = peven.BundleRef(transition_id="finish", run_key="rk-1", ordinal=1)
    store = open_store(7)
    store_token = activate_store(store)

    try:
        result = await invoke_transition(
            compiled,
            transition_id="finish",
            env=env,
            bundle=bundle,
            tokens=[peven.token({"kind": "ready"}, run_key="rk-1")],
            attempt=1,
        )

        still_active_ref = peven.store.put({"kind": "outside"})
        assert peven.store.get(still_active_ref) == {"kind": "outside"}
    finally:
        clear_store(store)
        reset_store(store_token)

    assert result == {
        "done": [
            Token(
                run_key="rk-1",
                color="default",
                payload={"kind": "stored", "payload": {"kind": "ready"}},
            )
        ]
    }

    with pytest.raises(RuntimeError, match=r"active Env\.run"):
        peven.store.put({"kind": "outside"})


@pytest.mark.asyncio
async def test_invoke_transition_exposes_the_active_sink_to_executors() -> None:
    traces: list[object] = []

    _register_executor(
        """
@peven.executor("runtime_trace_writer")
async def runtime_trace_writer(ctx, ready):
    ctx.trace(
        {
            "kind": "agent_trace",
            "payload": ready.payload,
            "input_places": sorted(ctx.inputs_by_place or {}),
        }
    )
    return peven.token({"done": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("runtime_trace_env")
    class RuntimeTraceEnv(peven.Env):
        ready = peven.place()
        done = peven.place()

        finish = peven.transition(
            inputs=["ready"],
            outputs=["done"],
            executor="runtime_trace_writer",
        )

    env = RuntimeTraceEnv()
    compiled = compile_env(RuntimeTraceEnv.spec())
    bundle = peven.BundleRef(transition_id="finish", run_key="rk-2", ordinal=1)

    result = await invoke_transition(
        compiled,
        transition_id="finish",
        env=env,
        bundle=bundle,
        tokens=[peven.token({"kind": "ready"}, run_key="rk-2")],
        attempt=1,
        inputs_by_place={"ready": [peven.token({"kind": "ready"}, run_key="rk-2")]},
        sink=SimpleNamespace(write=traces.append),
    )

    assert traces == [
        {
            "kind": "agent_trace",
            "payload": {"kind": "ready"},
            "input_places": ["ready"],
        }
    ]
    assert result == {
        "done": [Token(run_key="rk-2", color="default", payload={"done": True})]
    }


@pytest.mark.asyncio
async def test_invoke_transition_allows_executor_traces_without_a_sink() -> None:
    _register_executor(
        """
@peven.executor("runtime_trace_no_sink")
async def runtime_trace_no_sink(ctx, ready):
    ctx.trace({"kind": "agent_trace", "payload": ready.payload})
    return peven.token({"done": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("runtime_trace_no_sink_env")
    class RuntimeTraceNoSinkEnv(peven.Env):
        ready = peven.place()
        done = peven.place()

        finish = peven.transition(
            inputs=["ready"],
            outputs=["done"],
            executor="runtime_trace_no_sink",
        )

    env = RuntimeTraceNoSinkEnv()
    compiled = compile_env(RuntimeTraceNoSinkEnv.spec())
    bundle = peven.BundleRef(transition_id="finish", run_key="rk-3", ordinal=1)

    result = await invoke_transition(
        compiled,
        transition_id="finish",
        env=env,
        bundle=bundle,
        tokens=[peven.token({"kind": "ready"}, run_key="rk-3")],
        attempt=1,
    )

    assert result == {
        "done": [Token(run_key="rk-3", color="default", payload={"done": True})]
    }
