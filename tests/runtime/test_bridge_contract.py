from __future__ import annotations

import importlib
from types import SimpleNamespace

import msgspec
import pytest

import peven
import peven.runtime.bridge as bridge_module
from peven.handoff.framing import FrameDecoder
from peven.handoff.lowering import EnvSpecMessage
from peven.handoff.messages import LoadEnvOk, RunEnvOk
from peven.runtime.bootstrap import BootstrappedRuntime
from peven.runtime.state import _reset_shared_runtime_for_tests

from .conftest import make_session


env_module = importlib.import_module("peven.authoring.env")


@pytest.fixture(autouse=True)
def _reset_shared_runtime() -> None:
    _reset_shared_runtime_for_tests()


def _bootstrapped_runtime(*, replies: list[object] | None = None) -> BootstrappedRuntime:
    return make_session(frames=replies)


def _register_executor(source: str) -> dict[str, object]:
    namespace = {
        "__name__": "tests.runtime.generated_bridge_contract_executors",
        "peven": peven,
        "calls": [],
    }
    exec(source, namespace)
    return namespace


def _decode_single_frame(data: bytes) -> bytes:
    decoder = FrameDecoder()
    frames = decoder.feed(data)
    assert len(frames) == 1
    return frames[0]


@pytest.mark.asyncio
async def test_run_env_reuses_the_env_class_compiled_cache_and_normalizes_before_handing_to_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled_sentinel = SimpleNamespace(
        name="compiled",
        authored_env=EnvSpecMessage(
            schema_version=1,
            env_name="compiled",
            places=[],
            transitions=[],
        ),
    )
    normalized_marking = {"ready": [peven.token({"seed": 3}, run_key="rk-1")]}
    compile_calls: list[object] = []
    normalize_calls: list[object] = []

    def fake_compile_env(spec: object) -> object:
        compile_calls.append(spec)
        return compiled_sentinel

    monkeypatch.setattr(env_module, "compile_env", fake_compile_env)

    @peven.env("bridge_contract_compile_env")
    class BridgeContractCompileEnv(peven.Env):
        ready = peven.place()

        def initial_marking(self, seed: int | None = None) -> peven.Marking:
            return peven.Marking(
                {"ready": [peven.token({"seed": seed}, run_key="rk-1")]}
            )

    def fake_normalize_initial_marking(marking: object) -> object:
        normalize_calls.append(marking)
        assert isinstance(marking, peven.Marking)
        return normalized_marking

    monkeypatch.setattr(
        bridge_module,
        "normalize_initial_marking",
        fake_normalize_initial_marking,
    )

    async def bootstrap_runtime(command: tuple[str, ...]) -> BootstrappedRuntime:
        assert command == ("fake-runtime",)
        return _bootstrapped_runtime(
            replies=[
                LoadEnvOk(req_id=1),
                RunEnvOk(req_id=3, env_run_id=1),
                LoadEnvOk(req_id=5),
                RunEnvOk(req_id=7, env_run_id=2),
            ]
        )

    async def runtime_runner(
        *,
        runtime: object,
        compiled_env: object,
        env: object,
        env_run_id: int,
        initial_marking: object,
        callback: object,
    ) -> object:
        del env, env_run_id, callback
        assert compiled_env is compiled_sentinel
        assert initial_marking is normalized_marking
        session_writer = runtime.session.writer
        assert len(session_writer.writes) % 2 == 0
        write_base = len(session_writer.writes) - 2
        request_ordinal = (write_base // 2) + 1
        load_request = msgspec.msgpack.decode(
            _decode_single_frame(session_writer.writes[write_base]),
            type=dict[str, object],
        )
        run_request = msgspec.msgpack.decode(
            _decode_single_frame(session_writer.writes[write_base + 1]),
            type=dict[str, object],
        )
        assert load_request["kind"] == "load_env"
        assert load_request["req_id"] == (request_ordinal * 4) - 3
        assert run_request["kind"] == "run_env"
        assert run_request["req_id"] == (request_ordinal * 4) - 1
        assert run_request["env_run_id"] == request_ordinal
        assert run_request["fuse"] == 7
        ref = peven.store.put({"kind": "active"})
        assert peven.store.get(ref) == {"kind": "active"}
        return {"ok": True}

    result = await bridge_module.run_env(
        BridgeContractCompileEnv(),
        command=("fake-runtime",),
        bootstrap_runtime=bootstrap_runtime,
        runtime_runner=runtime_runner,
        fuse=7,
        seed=3,
    )
    result_2 = await bridge_module.run_env(
        BridgeContractCompileEnv(),
        command=("fake-runtime",),
        bootstrap_runtime=bootstrap_runtime,
        runtime_runner=runtime_runner,
        fuse=7,
        seed=4,
    )

    assert result == {"ok": True}
    assert result_2 == {"ok": True}
    assert len(compile_calls) == 1
    assert len(normalize_calls) == 2


@pytest.mark.asyncio
async def test_run_env_does_not_invoke_transition_callbacks_on_its_own() -> None:
    namespace = _register_executor(
        """
@peven.executor("bridge_contract_no_callback")
async def bridge_contract_no_callback(ctx):
    calls.append(ctx.bundle.transition_id)
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("bridge_contract_no_callback_env")
    class BridgeContractNoCallbackEnv(peven.Env):
        done = peven.place()

        def initial_marking(self, seed: int | None = None) -> peven.Marking:
            del seed
            return peven.Marking()

        finish = peven.transition(
            inputs=[],
            outputs=["done"],
            executor="bridge_contract_no_callback",
        )

    async def bootstrap_runtime(command: tuple[str, ...]) -> BootstrappedRuntime:
        assert command == ("fake-runtime",)
        return _bootstrapped_runtime(
            replies=[LoadEnvOk(req_id=1), RunEnvOk(req_id=3, env_run_id=1)]
        )

    async def runtime_runner(**kwargs: object) -> object:
        del kwargs
        return {"status": "idle"}

    result = await bridge_module.run_env(
        BridgeContractNoCallbackEnv(),
        command=("fake-runtime",),
        bootstrap_runtime=bootstrap_runtime,
        runtime_runner=runtime_runner,
    )

    assert result == {"status": "idle"}
    assert namespace["calls"] == []


@pytest.mark.asyncio
async def test_run_env_does_not_semantically_preflight_initial_markings_before_adapter_io() -> None:
    bootstrap_calls = 0
    runner_calls = 0

    @peven.env("bridge_contract_invalid_marking_env")
    class BridgeContractInvalidMarkingEnv(peven.Env):
        ready = peven.place(capacity=1)

        def initial_marking(self, seed: int | None = None) -> peven.Marking:
            del seed
            return peven.Marking(
                {
                    "ready": [
                        peven.token({"seed": 1}, run_key="rk-1"),
                        peven.token({"seed": 2}, run_key="rk-1"),
                    ]
                }
            )

    async def bootstrap_runtime(command: tuple[str, ...]) -> BootstrappedRuntime:
        nonlocal bootstrap_calls
        bootstrap_calls += 1
        assert command == ("fake-runtime",)
        return _bootstrapped_runtime(
            replies=[LoadEnvOk(req_id=1), RunEnvOk(req_id=3, env_run_id=1)]
        )

    async def runtime_runner(**kwargs: object) -> object:
        nonlocal runner_calls
        runner_calls += 1
        assert kwargs["initial_marking"]["ready"][0].payload == {"seed": 1}
        assert kwargs["initial_marking"]["ready"][1].payload == {"seed": 2}
        return {"status": "adapter_boundary"}

    result = await bridge_module.run_env(
        BridgeContractInvalidMarkingEnv(),
        command=("fake-runtime",),
        bootstrap_runtime=bootstrap_runtime,
        runtime_runner=runtime_runner,
    )

    assert result == {"status": "adapter_boundary"}
    assert bootstrap_calls == 1
    assert runner_calls == 1


@pytest.mark.asyncio
async def test_run_env_callback_dispatch_does_not_mutate_compiled_env() -> None:
    _register_executor(
        """
@peven.executor("bridge_contract_finish")
async def bridge_contract_finish(ctx):
    return peven.token({"ok": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("bridge_contract_mutation_env")
    class BridgeContractMutationEnv(peven.Env):
        done = peven.place()

        def initial_marking(self, seed: int | None = None) -> peven.Marking:
            del seed
            return peven.Marking()

        finish = peven.transition(
            inputs=[],
            outputs=["done"],
            executor="bridge_contract_finish",
        )

    async def bootstrap_runtime(command: tuple[str, ...]) -> BootstrappedRuntime:
        assert command == ("fake-runtime",)
        return _bootstrapped_runtime(
            replies=[LoadEnvOk(req_id=1), RunEnvOk(req_id=3, env_run_id=1)]
        )

    async def runtime_runner(
        *,
        runtime: object,
        compiled_env,
        env: object,
        env_run_id: int,
        initial_marking: object,
        callback,
    ) -> object:
        del runtime, env, env_run_id, initial_marking
        spec_before = compiled_env.env_spec
        authored_env_before = compiled_env.authored_env
        bindings_before = compiled_env.transition_bindings.copy()

        result = await callback(
            "finish",
            peven.BundleRef(transition_id="finish", run_key="rk-1", ordinal=1),
            [],
            attempt=1,
        )

        assert compiled_env.env_spec == spec_before
        assert compiled_env.authored_env == authored_env_before
        assert compiled_env.transition_bindings == bindings_before
        return result

    result = await bridge_module.run_env(
        BridgeContractMutationEnv(),
        command=("fake-runtime",),
        bootstrap_runtime=bootstrap_runtime,
        runtime_runner=runtime_runner,
    )

    assert result == {
        "done": [peven.Token(run_key="rk-1", color="default", payload={"ok": True})]
    }


@pytest.mark.asyncio
async def test_run_env_keeps_one_store_alive_for_the_full_run() -> None:
    _register_executor(
        """
@peven.executor("bridge_contract_store_writer")
async def bridge_contract_store_writer(ctx, ready):
    ref = peven.store.put({"seed": ready.payload["seed"]})
    return peven.token({"ref": ref}, run_key=ctx.bundle.run_key)

@peven.executor("bridge_contract_store_reader")
async def bridge_contract_store_reader(ctx, stored):
    value = peven.store.get(stored.payload["ref"])
    return peven.token({"payload": value}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("bridge_contract_store_env")
    class BridgeContractStoreEnv(peven.Env):
        ready = peven.place()
        stored = peven.place()
        done = peven.place()

        def initial_marking(self, seed: int | None = None) -> peven.Marking:
            return peven.Marking(
                {"ready": [peven.token({"seed": seed}, run_key="rk-1")]}
            )

        write = peven.transition(
            inputs=["ready"],
            outputs=["stored"],
            executor="bridge_contract_store_writer",
        )
        read = peven.transition(
            inputs=["stored"],
            outputs=["done"],
            executor="bridge_contract_store_reader",
        )

    async def bootstrap_runtime(command: tuple[str, ...]) -> BootstrappedRuntime:
        assert command == ("fake-runtime",)
        return _bootstrapped_runtime(
            replies=[LoadEnvOk(req_id=1), RunEnvOk(req_id=3, env_run_id=1)]
        )

    async def runtime_runner(
        *,
        runtime: object,
        compiled_env: object,
        env: object,
        env_run_id: int,
        initial_marking: dict[str, list[peven.Token]],
        callback,
    ) -> object:
        del runtime, compiled_env, env, env_run_id
        stored = await callback(
            "write",
            peven.BundleRef(transition_id="write", run_key="rk-1", ordinal=1),
            initial_marking["ready"],
            attempt=1,
        )
        return await callback(
            "read",
            peven.BundleRef(transition_id="read", run_key="rk-1", ordinal=2),
            stored["stored"],
            attempt=1,
        )

    result = await bridge_module.run_env(
        BridgeContractStoreEnv(),
        command=("fake-runtime",),
        bootstrap_runtime=bootstrap_runtime,
        runtime_runner=runtime_runner,
        seed=9,
    )

    assert result == {
        "done": [
            peven.Token(
                run_key="rk-1",
                color="default",
                payload={"payload": {"seed": 9}},
            )
        ]
    }
