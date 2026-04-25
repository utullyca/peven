from __future__ import annotations

import pytest


pytest.importorskip("minigrid")

from examples.minigrid.gridworld import DoorKeyWorld

from tests.gridworld.baseline import scripted_action


def test_baseline_solves_doorkey_before_minigrid_timeout() -> None:
    for seed in range(5):
        world = DoorKeyWorld(seed=seed)
        snapshot = world.reset()
        for _ in range(snapshot["max_steps"]):
            snapshot = world.step(scripted_action(world))
            if snapshot["terminated"]:
                assert snapshot["reward"] > 0.0
                assert snapshot["step_count"] < snapshot["max_steps"]
                break
        else:
            pytest.fail(f"baseline failed before MiniGrid timeout on seed {seed}")


def test_reward_is_minigrid_efficiency_discount() -> None:
    world = DoorKeyWorld(seed=0)
    snapshot = world.reset()
    for _ in range(snapshot["max_steps"]):
        snapshot = world.step(scripted_action(world))
        if snapshot["terminated"]:
            break

    expected_reward = 1 - 0.9 * snapshot["step_count"] / snapshot["max_steps"]
    assert snapshot["reward"] == pytest.approx(expected_reward)


def test_world_snapshot_uses_minigrid_observation_view() -> None:
    world = DoorKeyWorld(seed=0)
    snapshot = world.reset()
    assert snapshot["mission"] == "use the key to open the door and then get to the goal"
    assert snapshot["view"] == world.ego_view()
    assert len(snapshot["view"]) == world.view_size
    assert all(len(row) == world.view_size for row in snapshot["view"])
    assert snapshot["step_count"] == 0
    assert snapshot["max_steps"] == 640


def test_world_rejects_unknown_action() -> None:
    world = DoorKeyWorld(seed=0)
    world.reset()
    with pytest.raises(ValueError, match="unknown MiniGrid action"):
        world.step("teleport")


def test_world_does_not_render_carried_key_as_world_tile() -> None:
    world = DoorKeyWorld(seed=0)
    world.reset()
    snapshot = world.step("forward")
    snapshot = world.step("left")
    snapshot = world.step("pickup")

    view = snapshot["view"]
    assert snapshot["has_key"] is True
    assert view[-1][len(view[-1]) // 2] == "."
