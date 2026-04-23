from __future__ import annotations

from types import SimpleNamespace

import pytest

import peven
from peven.authoring.executor import unregister_executor
from peven.handoff.callbacks import invoke_transition
from peven.handoff.lowering import compile_env
from peven.shared.errors import PevenValidationError
from peven.shared.token import Token


def _register_executor(source: str) -> dict[str, object]:
    namespace = {
        "__name__": "tests.handoff.generated_dispatch_executors",
        "peven": peven,
        "calls": [],
    }
    exec(source, namespace)
    return namespace


def test_compile_env_builds_stable_transition_bindings() -> None:
    _register_executor(
        """
@peven.executor("compile_writer")
async def compile_writer(ctx, prompt):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)

@peven.executor("compile_router")
async def compile_router(ctx, ready):
    return {
        "accepted": peven.token({"ok": True}, run_key=ctx.bundle.run_key),
        "rejected": [],
    }
"""
    )

    @peven.env("compiled_handoff")
    class CompiledHandoff(peven.Env):
        prompt = peven.place()
        ready = peven.place()
        accepted = peven.place()
        rejected = peven.place()

        write = peven.transition(
            inputs=["prompt"],
            outputs=["ready"],
            executor="compile_writer",
        )
        route = peven.transition(
            inputs=[peven.input("ready", weight=2)],
            outputs=["accepted", "rejected"],
            executor="compile_router",
        )

    compiled = compile_env(CompiledHandoff.spec())

    assert [place.id for place in compiled.authored_env.places] == [
        "prompt",
        "ready",
        "accepted",
        "rejected",
    ]
    assert compiled.transition_bindings["write"].input_weights == (1,)
    assert compiled.transition_bindings["write"].output_places == ("ready",)
    assert compiled.transition_bindings["route"].input_weights == (2,)
    assert compiled.transition_bindings["route"].output_places == (
        "accepted",
        "rejected",
    )


def test_compile_env_canonicalizes_keyed_join_inputs_by_place_id() -> None:
    _register_executor(
        """
@peven.executor("compile_keyed_join")
async def compile_keyed_join(ctx, left, right):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("compiled_keyed_join")
    class CompiledKeyedJoin(peven.Env):
        left = peven.place()
        right = peven.place()
        done = peven.place()

        merge = peven.transition(
            inputs=["right", "left"],
            outputs=["done"],
            executor="compile_keyed_join",
            join_by=peven.join_key(peven.payload.case_id),
        )

    compiled = compile_env(CompiledKeyedJoin.spec())

    assert compiled.transition_bindings["merge"].input_weights == (1, 1)


@pytest.mark.asyncio
async def test_invoke_transition_adapts_weighted_inputs_and_builds_context() -> None:
    namespace = _register_executor(
        """
@peven.executor("dispatch_merge")
async def dispatch_merge(ctx, left, right_pair):
    calls.append((ctx, left, right_pair))
    return [peven.token({"merged": True}, run_key=ctx.bundle.run_key)]
"""
    )

    @peven.env("dispatch_merge_env")
    class DispatchMergeEnv(peven.Env):
        left = peven.place()
        right = peven.place()
        done = peven.place()

        merge = peven.transition(
            inputs=[peven.input("left"), peven.input("right", weight=2)],
            outputs=["done"],
            executor="dispatch_merge",
        )

    compiled = compile_env(DispatchMergeEnv.spec())
    bundle = peven.BundleRef(transition_id="merge", run_key="rk-1", ordinal=1)
    left = peven.token({"side": "left"}, run_key="rk-1")
    right_1 = peven.token({"side": "right-1"}, run_key="rk-1")
    right_2 = peven.token({"side": "right-2"}, run_key="rk-1")

    result = await invoke_transition(
        compiled,
        "merge",
        SimpleNamespace(name="env"),
        bundle,
        [left, right_1, right_2],
        attempt=2,
    )

    assert result == {
        "done": [Token(run_key="rk-1", color="default", payload={"merged": True})]
    }
    ctx, seen_left, seen_right = namespace["calls"][0]
    assert ctx.executor_name == "dispatch_merge"
    assert ctx.attempt == 2
    assert ctx.bundle == bundle
    assert seen_left == left
    assert seen_right == [right_1, right_2]


@pytest.mark.asyncio
async def test_invoke_transition_uses_canonical_place_order_for_keyed_joins() -> None:
    namespace = _register_executor(
        """
@peven.executor("dispatch_keyed_join")
async def dispatch_keyed_join(ctx, left, right):
    calls.append((left, right))
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("dispatch_keyed_join_env")
    class DispatchKeyedJoinEnv(peven.Env):
        left = peven.place()
        right = peven.place()
        done = peven.place()

        merge = peven.transition(
            inputs=["right", "left"],
            outputs=["done"],
            executor="dispatch_keyed_join",
            join_by=peven.join_key(peven.payload.case_id),
        )

    compiled = compile_env(DispatchKeyedJoinEnv.spec())
    bundle = peven.BundleRef(transition_id="merge", run_key="rk-1", ordinal=1)
    left = peven.token({"side": "left"}, run_key="rk-1")
    right = peven.token({"side": "right"}, run_key="rk-1")

    result = await invoke_transition(
        compiled,
        "merge",
        SimpleNamespace(name="env"),
        bundle,
        [left, right],
        attempt=1,
    )

    assert result == {
        "done": [Token(run_key="rk-1", color="default", payload={"ok": True})]
    }
    seen_left, seen_right = namespace["calls"][0]
    assert seen_left == left
    assert seen_right == right


@pytest.mark.asyncio
async def test_invoke_transition_supports_zero_input_and_multi_output() -> None:
    namespace = _register_executor(
        """
@peven.executor("dispatch_zero_route")
async def dispatch_zero_route(ctx):
    calls.append(ctx.bundle.transition_id)
    return {
        "accepted": peven.token({"kind": "ok"}, run_key=ctx.bundle.run_key),
        "rejected": [peven.token({"kind": "no"}, run_key=ctx.bundle.run_key)],
    }
"""
    )

    @peven.env("dispatch_zero_route_env")
    class DispatchZeroRouteEnv(peven.Env):
        accepted = peven.place()
        rejected = peven.place()

        route = peven.transition(
            inputs=[],
            outputs=["accepted", "rejected"],
            executor="dispatch_zero_route",
        )

    compiled = compile_env(DispatchZeroRouteEnv.spec())
    bundle = peven.BundleRef(transition_id="route", run_key="rk-2", ordinal=1)

    result = await invoke_transition(
        compiled,
        "route",
        SimpleNamespace(name="env"),
        bundle,
        [],
        attempt=1,
    )

    assert namespace["calls"] == ["route"]
    assert result == {
        "accepted": [Token(run_key="rk-2", color="default", payload={"kind": "ok"})],
        "rejected": [Token(run_key="rk-2", color="default", payload={"kind": "no"})],
    }


@pytest.mark.asyncio
async def test_invoke_transition_rejects_bad_output_contracts_and_run_keys() -> None:
    _register_executor(
        """
@peven.executor("dispatch_bad_outputs")
async def dispatch_bad_outputs(ctx, ready):
    return {"accepted": peven.token({"kind": "ok"}, run_key=ctx.bundle.run_key)}

@peven.executor("dispatch_bad_run_key")
async def dispatch_bad_run_key(ctx, ready):
    return peven.token({"kind": "ok"}, run_key="other")
"""
    )

    @peven.env("dispatch_bad_output_env")
    class DispatchBadOutputEnv(peven.Env):
        ready = peven.place()
        accepted = peven.place()
        rejected = peven.place()

        route = peven.transition(
            inputs=["ready"],
            outputs=["accepted", "rejected"],
            executor="dispatch_bad_outputs",
        )

    @peven.env("dispatch_bad_run_key_env")
    class DispatchBadRunKeyEnv(peven.Env):
        ready = peven.place()
        done = peven.place()

        finish = peven.transition(
            inputs=["ready"],
            outputs=["done"],
            executor="dispatch_bad_run_key",
        )

    bundle = peven.BundleRef(transition_id="route", run_key="rk-3", ordinal=1)
    ready = peven.token({"kind": "ready"}, run_key="rk-3")

    with pytest.raises(ValueError, match="every declared output place exactly once"):
        await invoke_transition(
            compile_env(DispatchBadOutputEnv.spec()),
            "route",
            SimpleNamespace(name="env"),
            bundle,
            [ready],
            attempt=1,
        )

    with pytest.raises(ValueError, match="preserve the current firing run_key"):
        await invoke_transition(
            compile_env(DispatchBadRunKeyEnv.spec()),
            "finish",
            SimpleNamespace(name="env"),
            peven.BundleRef(transition_id="finish", run_key="rk-3", ordinal=1),
            [ready],
            attempt=1,
        )


@pytest.mark.asyncio
async def test_invoke_transition_uses_compiled_binding_not_live_registry() -> None:
    _register_executor(
        """
@peven.executor("dispatch_registry_independent")
async def dispatch_registry_independent(ctx, ready):
    return peven.token({"kind": "ok"}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("dispatch_registry_env")
    class DispatchRegistryEnv(peven.Env):
        ready = peven.place()
        done = peven.place()

        finish = peven.transition(
            inputs=["ready"],
            outputs=["done"],
            executor="dispatch_registry_independent",
        )

    compiled = compile_env(DispatchRegistryEnv.spec())
    unregister_executor("dispatch_registry_independent")

    result = await invoke_transition(
        compiled,
        "finish",
        SimpleNamespace(name="env"),
        peven.BundleRef(transition_id="finish", run_key="rk-4", ordinal=1),
        [peven.token({"kind": "ready"}, run_key="rk-4")],
        attempt=1,
    )

    assert result == {
        "done": [Token(run_key="rk-4", color="default", payload={"kind": "ok"})]
    }


def test_compile_env_rejects_missing_executor_after_authoring() -> None:
    _register_executor(
        """
@peven.executor("compile_missing_later")
async def compile_missing_later(ctx, ready):
    return peven.token({"kind": "ok"}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("compile_missing_later_env")
    class CompileMissingLaterEnv(peven.Env):
        ready = peven.place()
        done = peven.place()

        finish = peven.transition(
            inputs=["ready"],
            outputs=["done"],
            executor="compile_missing_later",
        )

    unregister_executor("compile_missing_later")

    with pytest.raises(PevenValidationError, match="unknown executor compile_missing_later"):
        compile_env(CompileMissingLaterEnv.spec())


def test_compile_env_rejects_executor_signature_drift_after_authoring() -> None:
    _register_executor(
        """
@peven.executor("compile_signature_drift")
async def compile_signature_drift(ctx, ready):
    return peven.token({"kind": "ok"}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("compile_signature_drift_env")
    class CompileSignatureDriftEnv(peven.Env):
        ready = peven.place()
        done = peven.place()

        finish = peven.transition(
            inputs=["ready"],
            outputs=["done"],
            executor="compile_signature_drift",
        )

    unregister_executor("compile_signature_drift")
    _register_executor(
        """
@peven.executor("compile_signature_drift")
async def compile_signature_drift(ctx):
    return peven.token({"kind": "ok"}, run_key=ctx.bundle.run_key)
"""
    )

    with pytest.raises(PevenValidationError, match="executor signature does not match"):
        compile_env(CompileSignatureDriftEnv.spec())


@pytest.mark.asyncio
async def test_invoke_transition_rejects_unknown_transition_id() -> None:
    _register_executor(
        """
@peven.executor("dispatch_unknown_transition")
async def dispatch_unknown_transition(ctx):
    return peven.token({"kind": "ok"}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("dispatch_unknown_transition_env")
    class DispatchUnknownTransitionEnv(peven.Env):
        done = peven.place()

        finish = peven.transition(
            inputs=[],
            outputs=["done"],
            executor="dispatch_unknown_transition",
        )

    compiled = compile_env(DispatchUnknownTransitionEnv.spec())

    with pytest.raises(ValueError, match="unknown compiled transition missing"):
        await invoke_transition(
            compiled,
            "missing",
            SimpleNamespace(name="env"),
            peven.BundleRef(transition_id="missing", run_key="rk-5", ordinal=1),
            [],
            attempt=1,
        )
