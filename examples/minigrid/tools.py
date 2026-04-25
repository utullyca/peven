"""Pydantic AI output tools for DoorKey actions."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai.output import ToolOutput


class MoveForward(BaseModel):
    """Move one tile ahead."""


class TurnLeft(BaseModel):
    """Rotate left in place."""


class TurnRight(BaseModel):
    """Rotate right in place."""


class PickUpKey(BaseModel):
    """Pick up a key directly ahead."""


class OpenDoor(BaseModel):
    """Open a closed door directly ahead."""


class AskPlanner(BaseModel):
    """Spend one planner call instead of moving."""


ACTION_OUTPUTS = [
    ToolOutput(
        TurnLeft,
        name="turn_left",
        description=(
            "Rotate left in place. Use when tile_left is the current key, "
            "closed_door, or goal objective."
        ),
        max_retries=2,
        strict=False,
    ),
    ToolOutput(
        TurnRight,
        name="turn_right",
        description=(
            "Rotate right in place. Use when tile_right is the current key, "
            "closed_door, or goal objective."
        ),
        max_retries=2,
        strict=False,
    ),
    ToolOutput(
        MoveForward,
        name="move_forward",
        description=(
            "Move one tile ahead. Use only when tile_ahead is empty, open_door, "
            "or goal and the current objective is not tile_left or tile_right. "
            "Never use when tile_ahead is wall, unseen, key, or closed_door."
        ),
        max_retries=2,
        strict=False,
    ),
    ToolOutput(
        PickUpKey,
        name="pick_up_key",
        description="Pick up the key. Use only when inventory_key=no and tile_ahead=key.",
        max_retries=2,
        strict=False,
    ),
    ToolOutput(
        OpenDoor,
        name="open_door",
        description=(
            "Open the closed door. Use only when inventory_key=yes and "
            "tile_ahead=closed_door."
        ),
        max_retries=2,
        strict=False,
    ),
    ToolOutput(
        AskPlanner,
        name="ask_planner",
        description=(
            "Ask the planner for advice. Use when the next DoorKey object is not "
            "visible, no planner advice is present, and planner_calls_remaining "
            "is positive. Do not use when planner_calls_remaining is 0."
        ),
        max_retries=2,
        strict=False,
    ),
]

ACTION_KIND_BY_OUTPUT: dict[type[BaseModel], str] = {
    MoveForward: "forward",
    TurnLeft: "left",
    TurnRight: "right",
    PickUpKey: "pickup",
    OpenDoor: "toggle",
    AskPlanner: "ask_planner",
}


def action_kind(output: BaseModel) -> str:
    return ACTION_KIND_BY_OUTPUT[type(output)]
