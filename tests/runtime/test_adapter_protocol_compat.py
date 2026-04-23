from __future__ import annotations

import json
import os

import pytest

import peven
import peven.runtime.bridge as bridge_module


def _register_executor(source: str) -> dict[str, object]:
    namespace = {
        "__name__": "tests.runtime.generated_adapter_protocol_executors",
        "peven": peven,
    }
    exec(source, namespace)
    return namespace


def require_adapter_command() -> tuple[str, ...]:
    raw = os.environ.get("PEVEN_ADAPTER_COMMAND_JSON")
    if raw is None:
        pytest.skip(
            "set PEVEN_ADAPTER_COMMAND_JSON to run real adapter integration tests"
        )
    try:
        command = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("PEVEN_ADAPTER_COMMAND_JSON must be valid JSON") from exc
    if type(command) is not list or not command or not all(
        type(part) is str and part for part in command
    ):
        raise RuntimeError(
            "PEVEN_ADAPTER_COMMAND_JSON must decode to a non-empty JSON string array"
        )
    return tuple(command)


@pytest.mark.integration
@pytest.mark.slow
def test_real_adapter_decodes_trace_and_bundle_shapes_into_public_models() -> None:
    command = require_adapter_command()
    _register_executor(
        """
@peven.executor("adapter_protocol_finish")
async def adapter_protocol_finish(ctx, ready):
    return peven.token({"seen": ready.payload["seed"]}, run_key=ctx.bundle.run_key)
"""
    )

    @peven.env("adapter_protocol_env")
    class AdapterProtocolEnv(peven.Env):
        ready = peven.place()
        done = peven.place()

        def initial_marking(self, seed: int | None = None) -> peven.Marking:
            return peven.Marking(
                {"ready": [peven.token({"seed": seed}, run_key="rk-1")]}
            )

        finish = peven.transition(
            inputs=["ready"],
            outputs=["done"],
            executor="adapter_protocol_finish",
        )

    result = AdapterProtocolEnv().run(
        command=command,
        runtime_runner=bridge_module.run_until_terminal_result,
        seed=11,
    )

    completed = peven.completed_firings(result)
    assert len(completed) == 1
    firing = completed[0]
    assert isinstance(firing, peven.TransitionResult)
    assert firing.bundle == peven.BundleRef(
        transition_id="finish",
        run_key="rk-1",
        selected_key=None,
        ordinal=1,
    )
    assert firing.outputs == {
        "done": [peven.Token(run_key="rk-1", color="default", payload={"seen": 11})]
    }
    assert peven.firing_status(result, firing.firing_id) == "completed"
    assert peven.firing_result(result, firing.firing_id) == firing
