from __future__ import annotations

from typing import Any

from examples.minigrid.agents import (
    _mover_prompt,
    _plan_advice,
    _planner_advice,
    _planner_prompt,
)


def _obs() -> dict[str, Any]:
    return {
        "agent_pos": [3, 4],
        "agent_dir": 1,
        "has_key": False,
        "max_steps": 640,
        "mission": "use the key to open the door and then get to the goal",
        "reward": 0.42,
        "score": 0.39,
        "step_count": 17,
        "view": [
            [".", ".", "."],
            ["k", ".", "."],
            [".", ".", "."],
        ],
    }


def test_mover_prompt_uses_observation_not_global_pose() -> None:
    prompt = _mover_prompt(
        obs=_obs(),
        plan={
            "advice": (
                "subgoal: get_key\n"
                "tool: pick_up_key\n"
                "reason: key is visible"
            ),
        },
        remaining=1,
    )

    assert "Position:" not in prompt
    assert "direction=" not in prompt
    assert "agent_pos" not in prompt
    assert "agent_dir" not in prompt
    assert "[3, 4]" not in prompt
    assert "max_steps" not in prompt
    assert "step_count" not in prompt
    assert "reward" not in prompt
    assert "score" not in prompt
    assert "640" not in prompt
    assert "0.42" not in prompt
    assert "0.39" not in prompt
    assert "inventory_key=" in prompt
    assert "tile_ahead=" in prompt
    assert "pick_up_key works only with inventory_key=no and tile_ahead=key" in prompt
    assert "Tool preconditions:" in prompt
    assert "subgoal: get_key" in prompt
    assert "tool: pick_up_key" in prompt
    assert "reason: key is visible" in prompt
    assert "Current agent view:" in prompt
    assert "Legend:" in prompt
    assert "Decision table" not in prompt
    assert "ahead=wall -> left" not in prompt
    assert "Current hint" not in prompt
    assert "visible_target" not in prompt
    assert "Progress required" not in prompt
    assert "Last action" not in prompt
    assert "Front tile" not in prompt
    assert "Visible objects" not in prompt
    assert "priority" not in prompt
    assert "Allowed actions" not in prompt
    assert "Observed memory map" not in prompt
    assert "Memory map:" not in prompt


def test_mover_prompt_omits_planner_advice_when_optional_plan_is_absent() -> None:
    prompt = _mover_prompt(obs=_obs(), plan=None, remaining=1)

    assert "Planner advice:" not in prompt
    assert "Planner advice:\nnone" not in prompt
    assert "live state.\n" in prompt
    assert "live state or planner advice" not in prompt


def test_planner_prompt_does_not_receive_global_pose() -> None:
    prompt = _planner_prompt(
        obs=_obs(),
        memory={"fog": [["?"] * 8 for _ in range(8)]},
    )

    assert "Position:" not in prompt
    assert "direction=" not in prompt
    assert "agent_pos" not in prompt
    assert "agent_dir" not in prompt
    assert "[3, 4]" not in prompt
    assert "max_steps" not in prompt
    assert "step_count" not in prompt
    assert "reward" not in prompt
    assert "score" not in prompt
    assert "640" not in prompt
    assert "0.42" not in prompt
    assert "0.39" not in prompt
    assert "Use only the grids below" in prompt
    assert "hidden-map assumptions" not in prompt
    assert "Current agent view:" in prompt
    assert "Memory map:" in prompt
    assert "Peven turn" not in prompt
    assert "Front tile" not in prompt
    assert "Visible objects" not in prompt


def test_planner_advice_formats_structured_fields() -> None:
    assert _planner_advice(
        {
            "advice": (
                "subgoal: open_door\n"
                "tool: open_door\n"
                "reason: closed door is ahead"
            ),
        }
    ) == (
        "Planner advice:\n"
        "subgoal: open_door\n"
        "tool: open_door\n"
        "reason: closed door is ahead\n"
    )


def test_plan_advice_only_normalizes_whitespace() -> None:
    assert _plan_advice("subgoal: get_key") == "subgoal: get_key"
    assert (
        _plan_advice(" subgoal: get_key\n\ttool: pick_up_key\n reason: key visible ")
        == "subgoal: get_key\ntool: pick_up_key\nreason: key visible"
    )
