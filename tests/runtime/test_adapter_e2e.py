from __future__ import annotations

import asyncio

import pytest

import peven
import peven.runtime.bridge as bridge_module
from peven.authoring.ir import EnvSpec, InputArcSpec, OutputArcSpec, PlaceSpec, TransitionSpec
from peven.handoff.lowering import compile_env, normalize_initial_marking
from peven.runtime.bootstrap import bootstrap_runtime, close_bootstrapped_runtime
from peven.runtime.state import SharedRuntime, _reset_shared_runtime_for_tests, get_shared_runtime
from peven.shared.errors import PevenValidationError

from .conftest import require_adapter_command, require_external_pevenpy_adapter_command


def _register_executor(source: str) -> dict[str, object]:
    namespace = {
        "__name__": "tests.runtime.generated_adapter_e2e_executors",
        "peven": peven,
    }
    try:
        exec(source, namespace)
    except PevenValidationError as exc:
        if not any(issue.code == "duplicate_executor" for issue in exc.issues):
            raise
    return namespace


def _env_spec(
    *,
    env_name: str,
    places: list[str | tuple[str, int | None]],
    transitions: list[dict[str, object]],
) -> EnvSpec:
    authored_places = tuple(
        PlaceSpec(
            id=place if isinstance(place, str) else place[0],
            capacity=None if isinstance(place, str) else place[1],
        )
        for place in places
    )
    authored_transitions = tuple(
        TransitionSpec(
            id=str(transition["id"]),
            executor=str(transition.get("executor", "adapter_invalid_load_finish")),
            inputs=tuple(
                InputArcSpec(
                    place=arc[0],
                    weight=arc[1],
                )
                for arc in transition["inputs"]
            ),
            outputs=tuple(
                OutputArcSpec(place=place) for place in transition["outputs"]
            ),
            join_by_spec=transition.get("join_by_spec"),
        )
        for transition in transitions
    )
    return EnvSpec(
        env_name=env_name,
        places=authored_places,
        transitions=authored_transitions,
    )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_python_can_bootstrap_a_local_real_adapter() -> None:
    command = require_adapter_command()
    session = await bootstrap_runtime(command=command)
    try:
        assert session.handshake.tag == "peven-runtime-handshake"
        assert session.handshake.protocol_version == "0.1.0"
        assert session.handshake.peven_version == "0.2.0"
    finally:
        await close_bootstrapped_runtime(session)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_python_can_load_authored_env_against_a_local_real_adapter() -> None:
    command = require_adapter_command()
    _register_executor(
        """
@peven.executor("adapter_load_env_finish")
async def adapter_load_env_finish(ctx, ready):
    return peven.token({"done": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("adapter_load_env")
    class AdapterLoadEnv(peven.Env):
        ready = peven.place()
        done = peven.place()

        finish = peven.transition(
            inputs=["ready"],
            outputs=["done"],
            executor="adapter_load_env_finish",
        )

    session = await bootstrap_runtime(command=command)
    try:
        runtime = SharedRuntime(
            session=session,
            loop=asyncio.get_running_loop(),
            command=command,
        )
        compiled = compile_env(AdapterLoadEnv.spec())
        await bridge_module._load_compiled_env(runtime, compiled)
    finally:
        await close_bootstrapped_runtime(session)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_python_can_start_a_valid_run_against_a_local_real_adapter() -> None:
    command = require_adapter_command()
    _register_executor(
        """
@peven.executor("adapter_run_env_finish")
async def adapter_run_env_finish(ctx, ready):
    return peven.token({"done": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("adapter_run_env")
    class AdapterRunEnv(peven.Env):
        ready = peven.place()
        done = peven.place()

        finish = peven.transition(
            inputs=["ready"],
            outputs=["done"],
            executor="adapter_run_env_finish",
        )

    session = await bootstrap_runtime(command=command)
    try:
        runtime = SharedRuntime(
            session=session,
            loop=asyncio.get_running_loop(),
            command=command,
        )
        compiled = compile_env(AdapterRunEnv.spec())
        await bridge_module._load_compiled_env(runtime, compiled)
        await bridge_module._start_run(
            runtime,
            env_run_id=1,
            initial_marking=normalize_initial_marking(
                peven.Marking(
                    {"ready": [peven.token({"seed": 7}, run_key="rk-1")]}
                )
            ),
        )
    finally:
        await close_bootstrapped_runtime(session)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_real_adapter_event_stream_failures_mark_runtime_crashed_and_replace_it() -> None:
    command = require_external_pevenpy_adapter_command(
        fail_event_kind="transition_started"
    )
    _register_executor(
        """
@peven.executor("adapter_event_failure_finish")
async def adapter_event_failure_finish(ctx, ready):
    return peven.token({"done": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("adapter_event_failure_env")
    class AdapterEventFailureEnv(peven.Env):
        ready = peven.place()
        done = peven.place()

        finish = peven.transition(
            inputs=["ready"],
            outputs=["done"],
            executor="adapter_event_failure_finish",
        )

    _reset_shared_runtime_for_tests()
    runtime_2 = None
    try:
        runtime_1 = await get_shared_runtime(command=command)
        env = AdapterEventFailureEnv()
        marking = peven.Marking(
            {"ready": [peven.token({"seed": 7}, run_key="rk-1")]}
        )
        with pytest.raises(
            bridge_module.AdapterTransportError,
            match="closed while waiting for adapter message",
        ):
            await bridge_module.run_env(
                env,
                command=command,
                runtime_runner=bridge_module.run_until_terminal_result,
                marking=marking,
            )
        assert runtime_1.crashed is True

        runtime_2 = await get_shared_runtime(command=command)
        assert runtime_2 is not runtime_1
        assert runtime_2.crashed is False
    finally:
        _reset_shared_runtime_for_tests()


@pytest.mark.integration
@pytest.mark.slow
def test_real_adapter_surfaces_callback_failures_end_to_end() -> None:
    command = require_adapter_command()
    _register_executor(
        """
@peven.executor("adapter_callback_failure_finish")
async def adapter_callback_failure_finish(ctx, ready):
    raise RuntimeError("callback boom")
"""
    )

    @peven.env("adapter_callback_failure_env")
    class AdapterCallbackFailureEnv(peven.Env):
        ready = peven.place()
        done = peven.place()

        def initial_marking(self, seed: int | None = None) -> peven.Marking:
            return peven.Marking(
                {"ready": [peven.token({"seed": seed}, run_key="rk-1")]}
            )

        finish = peven.transition(
            inputs=["ready"],
            outputs=["done"],
            executor="adapter_callback_failure_finish",
        )

    result = AdapterCallbackFailureEnv().run(
        command=command,
        runtime_runner=bridge_module.run_until_terminal_result,
        seed=13,
    )

    assert result.status == "failed"
    assert result.error == "callback boom"
    failed = peven.failed_firings(result)
    assert len(failed) == 1
    assert failed[0].error == "callback boom"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_public_python_authoring_allows_zero_output_envs_to_load_in_julia() -> None:
    command = require_adapter_command()
    _register_executor(
        """
@peven.executor("adapter_sink_finish")
async def adapter_sink_finish(ctx, ready):
    return None
"""
    )

    @peven.env("adapter_sink_env")
    class AdapterSinkEnv(peven.Env):
        ready = peven.place()

        finish = peven.transition(
            inputs=["ready"],
            outputs=[],
            executor="adapter_sink_finish",
        )

    session = await bootstrap_runtime(command=command)
    try:
        runtime = SharedRuntime(
            session=session,
            loop=asyncio.get_running_loop(),
            command=command,
        )
        compiled = compile_env(AdapterSinkEnv.spec())
        await bridge_module._load_compiled_env(runtime, compiled)
    finally:
        await close_bootstrapped_runtime(session)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_public_python_authoring_defers_duplicate_output_validation_to_julia() -> None:
    command = require_adapter_command()
    _register_executor(
        """
@peven.executor("adapter_duplicate_output_finish")
async def adapter_duplicate_output_finish(ctx, ready):
    return peven.token({"done": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("adapter_duplicate_output_env")
    class AdapterDuplicateOutputEnv(peven.Env):
        ready = peven.place()
        done = peven.place()

        finish = peven.transition(
            inputs=["ready"],
            outputs=["done", "done"],
            executor="adapter_duplicate_output_finish",
        )

    session = await bootstrap_runtime(command=command)
    try:
        runtime = SharedRuntime(
            session=session,
            loop=asyncio.get_running_loop(),
            command=command,
        )
        compiled = compile_env(AdapterDuplicateOutputEnv.spec())
        with pytest.raises(
            bridge_module.LoadEnvRejectedError,
            match="duplicate_output_arc:finish",
        ):
            await bridge_module._load_compiled_env(runtime, compiled)
    finally:
        await close_bootstrapped_runtime(session)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_public_python_authoring_defers_guard_transition_compatibility_to_julia() -> None:
    command = require_adapter_command()
    _register_executor(
        """
@peven.executor("adapter_weighted_guard_finish")
async def adapter_weighted_guard_finish(ctx, ready):
    return peven.token({"done": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("adapter_weighted_guard_env")
    class AdapterWeightedGuardEnv(peven.Env):
        ready = peven.place()
        done = peven.place()

        finish = peven.transition(
            inputs=[peven.input("ready", weight=2)],
            outputs=["done"],
            executor="adapter_weighted_guard_finish",
            guard=~peven.isempty(peven.f.items),
        )

    session = await bootstrap_runtime(command=command)
    try:
        runtime = SharedRuntime(
            session=session,
            loop=asyncio.get_running_loop(),
            command=command,
        )
        compiled = compile_env(AdapterWeightedGuardEnv.spec())
        with pytest.raises(
            bridge_module.LoadEnvRejectedError,
            match="single-input, weight-1 transitions",
        ):
            await bridge_module._load_compiled_env(runtime, compiled)
    finally:
        await close_bootstrapped_runtime(session)


@pytest.mark.parametrize(
    ("spec", "marking", "expected_code"),
    [
        pytest.param(
            _env_spec(
                env_name="adapter_invalid_marking_unknown_place",
                places=["ready", "done"],
                transitions=[
                    {
                        "id": "finish",
                        "inputs": [("ready", 1)],
                        "outputs": ["done"],
                    }
                ],
            ),
            peven.Marking(
                {"missing": [peven.token({"seed": 1}, run_key="rk-1")]}
            ),
            "unknown_place:missing",
            id="unknown_place",
        ),
        pytest.param(
            _env_spec(
                env_name="adapter_invalid_marking_capacity",
                places=[("ready", 1), "done"],
                transitions=[
                    {
                        "id": "finish",
                        "inputs": [("ready", 1)],
                        "outputs": ["done"],
                    }
                ],
            ),
            peven.Marking(
                {
                    "ready": [
                        peven.token({"seed": 1}, run_key="rk-1"),
                        peven.token({"seed": 2}, run_key="rk-2"),
                    ]
                }
            ),
            "capacity_exceeded:ready",
            id="capacity_exceeded",
        ),
        pytest.param(
            _env_spec(
                env_name="adapter_invalid_marking_unreachable",
                places=["ready", "done", "other", "extra"],
                transitions=[
                    {
                        "id": "finish",
                        "inputs": [("ready", 1)],
                        "outputs": ["done"],
                    },
                    {
                        "id": "score",
                        "inputs": [("other", 1)],
                        "outputs": ["extra"],
                    },
                ],
            ),
            peven.Marking(
                {"ready": [peven.token({"seed": 1}, run_key="rk-1")]}
            ),
            "unreachable_transition:score",
            id="unreachable_transition",
        ),
    ],
)
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_python_run_env_rejects_engine_invalid_markings_against_a_local_real_adapter(
    spec: EnvSpec,
    marking: peven.Marking,
    expected_code: str,
) -> None:
    command = require_adapter_command()
    _register_executor(
        """
@peven.executor("adapter_invalid_load_finish")
async def adapter_invalid_load_finish(ctx, ready):
    return peven.token({"done": True}, run_key=ctx.bundle.run_key)
"""
    )

    session = await bootstrap_runtime(command=command)
    try:
        runtime = SharedRuntime(
            session=session,
            loop=asyncio.get_running_loop(),
            command=command,
        )
        compiled = compile_env(spec)
        await bridge_module._load_compiled_env(runtime, compiled)
        with pytest.raises(bridge_module.RunEnvRejectedError, match=expected_code):
            await bridge_module._start_run(
                runtime,
                env_run_id=1,
                initial_marking=normalize_initial_marking(marking),
            )
    finally:
        await close_bootstrapped_runtime(session)


@pytest.mark.parametrize(
    ("spec", "expected_code"),
    [
        pytest.param(
            _env_spec(
                env_name="adapter_invalid_unknown_input",
                places=["done"],
                transitions=[
                    {
                        "id": "finish",
                        "inputs": [("missing", 1)],
                        "outputs": ["done"],
                    }
                ],
            ),
            "unknown_place:missing",
            id="unknown_input_place",
        ),
        pytest.param(
            _env_spec(
                env_name="adapter_invalid_unknown_output",
                places=["ready"],
                transitions=[
                    {
                        "id": "finish",
                        "inputs": [("ready", 1)],
                        "outputs": ["missing"],
                    }
                ],
            ),
            "unknown_place:missing",
            id="unknown_output_place",
        ),
        pytest.param(
            _env_spec(
                env_name="adapter_invalid_duplicate_input",
                places=["left", "done"],
                transitions=[
                    {
                        "id": "join",
                        "executor": "adapter_invalid_load_join",
                        "inputs": [("left", 1), ("left", 2)],
                        "outputs": ["done"],
                        "join_by_spec": {"kind": "payload_ref", "path": ["case_id"]},
                    }
                ],
            ),
            "duplicate_input_arc:join",
            id="duplicate_input_arc",
        ),
        pytest.param(
            _env_spec(
                env_name="adapter_invalid_duplicate_output",
                places=["ready", "done"],
                transitions=[
                    {
                        "id": "finish",
                        "inputs": [("ready", 1)],
                        "outputs": ["done", "done"],
                    }
                ],
            ),
            "duplicate_output_arc:finish",
            id="duplicate_output_arc",
        ),
        pytest.param(
            _env_spec(
                env_name="adapter_invalid_keyed_join",
                places=["left", "done"],
                transitions=[
                    {
                        "id": "join",
                        "inputs": [("left", 1)],
                        "outputs": ["done"],
                        "join_by_spec": {"kind": "payload_ref", "path": ["case_id"]},
                    }
                ],
            ),
            "invalid_keyed_join:join",
            id="invalid_keyed_join",
        ),
        pytest.param(
            _env_spec(
                env_name="adapter_invalid_capacity",
                places=[("ready", 1), "done"],
                transitions=[
                    {
                        "id": "finish",
                        "inputs": [("ready", 2)],
                        "outputs": ["done"],
                    }
                ],
            ),
            "weight_exceeds_capacity:finish",
            id="weight_exceeds_capacity",
        ),
        pytest.param(
            _env_spec(
                env_name="adapter_invalid_orphan",
                places=["ready", "done", "lonely"],
                transitions=[
                    {
                        "id": "finish",
                        "inputs": [("ready", 1)],
                        "outputs": ["done"],
                    }
                ],
            ),
            "orphan_place:lonely",
            id="orphan_place",
        ),
    ],
)
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_python_load_env_rejects_engine_invalid_env_against_a_local_real_adapter(
    spec: EnvSpec,
    expected_code: str,
) -> None:
    command = require_adapter_command()
    _register_executor(
        """
@peven.executor("adapter_invalid_load_finish")
async def adapter_invalid_load_finish(ctx, ready):
    return peven.token({"done": True}, run_key=ctx.bundle.run_key)
"""
    )

    _register_executor(
        """
@peven.executor("adapter_invalid_load_join")
async def adapter_invalid_load_join(ctx, left, right):
    return peven.token({"done": True}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("adapter_invalid_load_env")
    class AdapterInvalidLoadEnv(peven.Env):
        done = peven.place()

        finish = peven.transition(
            inputs=["missing"],
            outputs=["done"],
            executor="adapter_invalid_load_finish",
        )

    session = await bootstrap_runtime(command=command)
    try:
        runtime = SharedRuntime(
            session=session,
            loop=asyncio.get_running_loop(),
            command=command,
        )
        compiled = compile_env(spec)
        with pytest.raises(bridge_module.LoadEnvRejectedError, match=expected_code):
            await bridge_module._load_compiled_env(runtime, compiled)
    finally:
        await close_bootstrapped_runtime(session)


@pytest.mark.integration
@pytest.mark.slow
def test_env_run_round_trips_against_a_real_adapter() -> None:
    command = require_adapter_command()
    _register_executor(
        """
@peven.executor("adapter_e2e_finish")
async def adapter_e2e_finish(ctx, ready):
    return peven.token({"done": ready.payload["seed"]}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("adapter_e2e_env")
    class AdapterE2EEnv(peven.Env):
        ready = peven.place()
        done = peven.place()

        def initial_marking(self, seed: int | None = None) -> peven.Marking:
            return peven.Marking(
                {"ready": [peven.token({"seed": seed}, run_key="rk-1")]}
            )

        finish = peven.transition(
            inputs=["ready"],
            outputs=["done"],
            executor="adapter_e2e_finish",
        )

    result = AdapterE2EEnv().run(
        command=command,
        runtime_runner=bridge_module.run_until_terminal_result,
        seed=7,
    )

    assert isinstance(result, peven.RunResult)
    assert result.status == "completed"
    assert peven.completed_firings(result)
    assert result.final_marking == {
        "done": [peven.Token(run_key="rk-1", color="default", payload={"done": 7})]
    }
