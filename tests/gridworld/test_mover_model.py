from __future__ import annotations

import os
from typing import Any

import pytest


pytest.importorskip("minigrid")

from examples.minigrid.agents import _mover_prompt, _tile_ahead, mover
from examples.minigrid.gridworld import DoorKeyWorld
from examples.minigrid.tools import ACTION_OUTPUTS, action_kind

from tests.gridworld.baseline import scripted_action


def _door_ahead_state() -> dict[str, Any]:
    world = DoorKeyWorld(seed=0)
    snapshot = world.reset()
    for _ in range(snapshot["max_steps"]):
        if snapshot["has_key"] and _tile_ahead(snapshot) == "D":
            return snapshot
        snapshot = world.step(scripted_action(world))
    raise AssertionError("failed to find door-ahead state")


def _door_visible_side_state() -> dict[str, Any]:
    snapshot = _door_ahead_state()
    view = [row[:] for row in snapshot["view"]]
    center = len(view[-1]) // 2
    view[-1][center - 1] = "D"
    view[-2][center] = "."
    return {**snapshot, "view": view}


def _open_door_ahead_state() -> dict[str, Any]:
    world = DoorKeyWorld(seed=0)
    snapshot = world.reset()
    for _ in range(snapshot["max_steps"]):
        if snapshot["has_key"] and _tile_ahead(snapshot) == "D":
            snapshot = world.step("toggle")
            assert _tile_ahead(snapshot) == "o"
            return snapshot
        snapshot = world.step(scripted_action(world))
    raise AssertionError("failed to find open-door state")


async def _choose(snapshot: dict[str, Any]) -> str:
    result = await mover.run(
        _mover_prompt(obs=snapshot, plan={"advice": "none"}, remaining=2),
        output_type=ACTION_OUTPUTS,
    )
    return action_kind(result.output)


@pytest.mark.integration
@pytest.mark.ollama
@pytest.mark.asyncio
async def test_mover_toggles_closed_door_directly_ahead() -> None:
    if os.environ.get("PEVEN_RUN_MINIGRID_OLLAMA") != "1":
        pytest.skip("set PEVEN_RUN_MINIGRID_OLLAMA=1 to run the mover model test")

    snapshot = _door_ahead_state()

    assert await _choose(snapshot) == "toggle"


@pytest.mark.integration
@pytest.mark.ollama
@pytest.mark.asyncio
async def test_mover_turns_for_side_visible_door_when_holding_key() -> None:
    if os.environ.get("PEVEN_RUN_MINIGRID_OLLAMA") != "1":
        pytest.skip("set PEVEN_RUN_MINIGRID_OLLAMA=1 to run the mover model test")

    snapshot = _door_visible_side_state()

    assert await _choose(snapshot) in {"left", "right"}


@pytest.mark.integration
@pytest.mark.ollama
@pytest.mark.asyncio
async def test_mover_moves_forward_through_open_door() -> None:
    if os.environ.get("PEVEN_RUN_MINIGRID_OLLAMA") != "1":
        pytest.skip("set PEVEN_RUN_MINIGRID_OLLAMA=1 to run the mover model test")

    snapshot = _open_door_ahead_state()

    assert await _choose(snapshot) == "forward"
