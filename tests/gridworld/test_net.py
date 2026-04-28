from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest


pytest.importorskip("minigrid")

import examples.minigrid.net as minigrid_net
from examples.minigrid.net import DoorKeyEnv, run_minigrid
from examples.minigrid.render import render_episode

import peven
from peven.handoff.callbacks import invoke_transition
from peven.handoff.lowering import compile_env

from ..runtime.conftest import require_external_pevenpy_adapter_command


def test_spec_exposes_mover_planner_and_environment_step() -> None:
    spec = DoorKeyEnv.spec()
    transition_ids = {transition.id for transition in spec.transitions}
    assert transition_ids == {"mover", "planner", "env_step"}


def test_spec_places_are_core_environment_state() -> None:
    spec = DoorKeyEnv.spec()
    place_ids = {place.id for place in spec.places}
    assert place_ids == {
        "obs",
        "memory",
        "plan",
        "move_request",
        "action",
        "plan_request",
        "done",
    }


def test_doorkey_initial_marking_has_no_plan_token() -> None:
    marking = DoorKeyEnv().initial_marking(seed=0)

    assert "plan" not in marking.tokens_by_place


def test_mover_declares_plan_as_optional_input() -> None:
    mover = next(
        transition for transition in DoorKeyEnv.spec().transitions if transition.id == "mover"
    )

    assert [(arc.place, arc.optional) for arc in mover.inputs] == [
        ("move_request", False),
        ("obs", False),
        ("memory", False),
        ("plan", True),
    ]


@pytest.mark.asyncio
async def test_mover_normal_action_emits_no_plan_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def choose_move(ctx, *, obs, memory, plan):
        del ctx, obs, memory
        assert plan is None
        return {"kind": "forward", "env_action": "forward"}

    monkeypatch.setattr(minigrid_net, "choose_move", choose_move)
    outputs = await _invoke_mover(plan=None)

    assert outputs["action"][0].payload == {
        "kind": "forward",
        "env_action": "forward",
        "turn": 0,
    }
    assert outputs["plan"] == []
    assert outputs["plan_request"] == []


@pytest.mark.asyncio
async def test_mover_normal_action_clears_present_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def choose_move(ctx, *, obs, memory, plan):
        del ctx, obs, memory
        assert plan == {"advice": "go", "version": 3}
        return {"kind": "left", "env_action": "left"}

    monkeypatch.setattr(minigrid_net, "choose_move", choose_move)
    plan = peven.token({"advice": "go", "version": 3}, run_key="rk-1")
    outputs = await _invoke_mover(plan=plan)

    assert outputs["action"][0].payload["kind"] == "left"
    assert outputs["plan"] == []
    assert outputs["plan_request"] == []


@pytest.mark.asyncio
async def test_mover_planner_request_clears_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def choose_move(ctx, *, obs, memory, plan):
        del ctx, obs, memory, plan
        return {"kind": "ask_planner"}

    monkeypatch.setattr(minigrid_net, "choose_move", choose_move)
    plan = peven.token({"advice": "go", "version": 3}, run_key="rk-1")
    outputs = await _invoke_mover(plan=plan)

    assert outputs["action"] == []
    assert outputs["plan"] == []
    assert outputs["plan_request"][0].payload == {
        "turn": 0,
        "version": 1,
        "reason": "ask_planner",
    }


async def _invoke_mover(*, plan: peven.Token | None) -> dict[str, list[peven.Token]]:
    compiled = compile_env(DoorKeyEnv.spec())
    env = SimpleNamespace(world=object())
    bundle = peven.BundleRef(transition_id="mover", run_key="rk-1", ordinal=1)
    move_request = peven.token({"turn": 0}, run_key="rk-1")
    obs = peven.token({"agent_pos": [1, 1]}, run_key="rk-1")
    memory = peven.token(
        {
            "planner_calls": 0,
            "planner_limit": 2,
            "steps_since_planner": 0,
            "last_action": "reset",
        },
        run_key="rk-1",
    )
    tokens = [move_request, obs, memory]
    inputs_by_place = {
        "move_request": [move_request],
        "obs": [obs],
        "memory": [memory],
        "plan": [],
    }
    if plan is not None:
        tokens.append(plan)
        inputs_by_place["plan"] = [plan]

    return await invoke_transition(
        compiled,
        "mover",
        env,
        bundle,
        tokens,
        attempt=1,
        inputs_by_place=inputs_by_place,
    )


@pytest.mark.integration
@pytest.mark.ollama
@pytest.mark.slow
def test_doorkey_runs_through_peven_net_with_llm_tools() -> None:
    if os.environ.get("PEVEN_RUN_MINIGRID_OLLAMA") != "1":
        pytest.skip("set PEVEN_RUN_MINIGRID_OLLAMA=1 to run the Minigrid LLM test")

    command = require_external_pevenpy_adapter_command()
    log_dir = Path("logs") / "doorkey_8x8" / "net_seed0_test"
    log_dir.mkdir(parents=True, exist_ok=True)
    episode_log = log_dir / "episode.jsonl"
    episode_log.unlink(missing_ok=True)
    sink = peven.JSONLSink(episode_log)

    result = run_minigrid(command=command, seed=0, fuse=160, sink=sink)

    assert result.status == "completed", result
    assert result.terminal_reason is None
    actions: list[str] = []
    cells: set[tuple[int, int]] = set()
    for index, entry in enumerate(result.trace[:-1]):
        if entry.bundle.transition_id != "mover":
            continue
        if entry.outputs["plan_request"]:
            assert entry.outputs["action"] == []
            assert result.trace[index + 1].bundle.transition_id == "planner"
        else:
            assert entry.outputs["action"]
            action_payload = cast(dict[str, Any], entry.outputs["action"][0].payload)
            actions.append(str(action_payload["kind"]))
            assert result.trace[index + 1].bundle.transition_id == "env_step"
    for entry in result.trace:
        for token in entry.outputs.get("obs", []):
            payload = cast(dict[str, Any], token.payload)
            pos = cast(list[int], payload["agent_pos"])
            cells.add((int(pos[0]), int(pos[1])))

    memory_tokens = result.final_marking.get("memory", [])
    if memory_tokens:
        memory_payload = cast(dict[str, Any], memory_tokens[0].payload)
        assert memory_payload["planner_calls"] <= memory_payload["planner_limit"]

    done_tokens = result.final_marking.get("done", [])
    assert done_tokens
    done_payload = cast(dict[str, Any], done_tokens[0].payload)
    assert done_payload["terminated"] is True
    assert done_payload["reward"] > 0.0
    assert done_payload["score"] <= done_payload["reward"]
    assert done_payload["planner_calls"] <= done_payload["planner_limit"]
    assert {"forward", "pickup", "toggle"}.issubset(actions)
    assert len(cells) > 3

    rendered = render_episode(episode_log)
    assert "^" in rendered or ">" in rendered or "v" in rendered or "<" in rendered
    assert "done" in rendered
    assert "run completed" in rendered
