from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace

import pytest

import peven
from peven.shared.errors import PevenValidationError


env_module = importlib.import_module("peven.authoring.env")


def test_env_spec_requires_decorated_class() -> None:
    class PlainEnv(peven.Env):
        pass

    with pytest.raises(TypeError, match="has not been decorated"):
        PlainEnv.spec()


def test_env_spec_does_not_inherit_a_base_env_ir() -> None:
    namespace = {
        "__name__": "tests.authoring.generated_inherited_spec_executors",
        "peven": peven,
    }
    exec(
        """
@peven.executor("inherited_spec_executor")
async def inherited_spec_executor(ctx):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
""",
        namespace,
    )

    @peven.env("base_env")
    class BaseEnv(peven.Env):
        done = peven.place()

        finish = peven.transition(
            inputs=[], outputs=["done"], executor="inherited_spec_executor"
        )

    class ChildEnv(BaseEnv):
        helper = "value"

    with pytest.raises(TypeError, match="has not been decorated"):
        ChildEnv.spec()


def test_env_base_stub_methods_raise_until_later_layers_exist() -> None:
    env = peven.Env()

    with pytest.raises(NotImplementedError, match="initial_marking"):
        env.initial_marking()
    with pytest.raises(TypeError, match="has not been decorated"):
        env.run(command=("fake-runtime",))


def test_env_base_class_no_longer_exposes_python_side_validate() -> None:
    namespace = {
        "__name__": "tests.authoring.generated_validate_executors",
        "peven": peven,
    }
    exec(
        """
@peven.executor("validate_ctx_only")
async def validate_ctx_only(ctx):
    return peven.token({"ok": True}, run_key="rk")
""",
        namespace,
    )

    @peven.env("validated_env")
    class ValidatedEnv(peven.Env):
        done = peven.place()

        finish = peven.transition(inputs=[], outputs=["done"], executor="validate_ctx_only")

    assert not hasattr(ValidatedEnv(), "validate")


def test_decorated_env_run_delegates_without_requiring_an_explicit_adapter_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = {
        "__name__": "tests.authoring.generated_run_executors",
        "peven": peven,
    }
    exec(
        """
@peven.executor("run_ctx_only")
async def run_ctx_only(ctx):
    return peven.token({"ok": True}, run_key="rk")
""",
        namespace,
    )

    @peven.env("run_unavailable_env")
    class RunUnavailableEnv(peven.Env):
        done = peven.place()

        finish = peven.transition(inputs=[], outputs=["done"], executor="run_ctx_only")

    import peven.runtime.bridge as bridge_module
    import peven.runtime.state as state_module

    seen: dict[str, object] = {}

    async def fake_run_env(env: object, **kwargs: object) -> object:
        seen["env"] = env
        seen["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr(bridge_module, "run_env", fake_run_env)
    monkeypatch.setattr(
        state_module,
        "run_sync",
        lambda awaitable: asyncio.run(awaitable),
    )

    assert RunUnavailableEnv().run() == "ok"
    assert isinstance(seen["env"], RunUnavailableEnv)
    assert seen["kwargs"] == {}


def test_decorated_env_caches_one_compiled_handoff_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled_sentinel = SimpleNamespace(name="compiled")
    compile_calls: list[object] = []
    registry_version = 0

    def fake_compile_env(spec: object) -> object:
        compile_calls.append(spec)
        return compiled_sentinel

    monkeypatch.setattr(env_module, "compile_env", fake_compile_env)
    monkeypatch.setattr(
        env_module,
        "get_executor_registry_version",
        lambda: registry_version,
    )

    namespace = {
        "__name__": "tests.authoring.generated_compiled_env_executors",
        "peven": peven,
    }
    exec(
        """
@peven.executor("compiled_env_ctx_only")
async def compiled_env_ctx_only(ctx):
    return peven.token({"ok": True}, run_key="rk")
""",
        namespace,
    )

    @peven.env("compiled_env")
    class CompiledEnv(peven.Env):
        done = peven.place()

        finish = peven.transition(
            inputs=[],
            outputs=["done"],
            executor="compiled_env_ctx_only",
        )

    assert CompiledEnv.compiled() is compiled_sentinel
    assert CompiledEnv.compiled() is compiled_sentinel
    assert compile_calls == [CompiledEnv.spec()]


def test_decorated_env_invalidates_cached_compiled_env_when_executor_registry_changes() -> None:
    namespace = {
        "__name__": "tests.authoring.generated_registry_drift_executors",
        "peven": peven,
    }
    exec(
        """
@peven.executor("compiled_env_registry_drift")
async def compiled_env_registry_drift(ctx, ready):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
""",
        namespace,
    )

    @peven.env("compiled_env_registry_drift_env")
    class CompiledEnvRegistryDrift(peven.Env):
        ready = peven.place()
        done = peven.place()

        finish = peven.transition(
            inputs=["ready"],
            outputs=["done"],
            executor="compiled_env_registry_drift",
        )

    assert CompiledEnvRegistryDrift.compiled() is not None

    peven.unregister_executor("compiled_env_registry_drift")
    with pytest.raises(PevenValidationError, match="unknown executor compiled_env_registry_drift"):
        CompiledEnvRegistryDrift.compiled()

    exec(
        """
@peven.executor("compiled_env_registry_drift")
async def compiled_env_registry_drift(ctx):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
""",
        namespace,
    )
    with pytest.raises(PevenValidationError, match="executor signature does not match"):
        CompiledEnvRegistryDrift.compiled()

def test_env_decorator_rejects_invalid_name_and_non_env_subclass() -> None:
    with pytest.raises(ValueError, match="env name must be a non-empty string"):
        peven.env("")

    with pytest.raises(TypeError, match="requires an Env subclass"):
        peven.env("not_env")(type("NotEnv", (), {}))
